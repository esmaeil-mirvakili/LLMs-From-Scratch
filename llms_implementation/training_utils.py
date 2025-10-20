from abc import ABC
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
import torch
from torch import nn, Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.amp import autocast, GradScaler
from loguru import logger
from tqdm import tqdm


Batch = Union[Dict, List, Tensor, Tuple]
Metric = Union[nn.Module, Callable[[Tensor, Tensor], float]]
Loss = Union[nn.Module, Callable[[Tensor, Tensor], Tensor]]
PrepareBatchFn = Callable[[Batch, torch.device], Batch]


class Callback(ABC):
    def on_train_start(self, trainer: "Trainer", **kwargs):
        pass

    def on_train_end(self, trainer: "Trainer", **kwargs):
        pass

    def on_validation_start(self, trainer: "Trainer", **kwargs):
        pass

    def on_validation_end(self, trainer: "Trainer", logs: Dict[str, Any], **kwargs):
        pass

    def on_epoch_start(self, epoch, trainer: "Trainer", **kwargs):
        pass

    def on_epoch_end(self, epoch, trainer: "Trainer", logs: Dict[str, Any], **kwargs):
        pass

    def on_batch_start(self, step, trainer: "Trainer", logs: Dict[str, Any], **kwargs):
        pass

    def on_batch_end(self, step, trainer: "Trainer", logs: Dict[str, Any], **kwargs):
        pass


class Trainer:

    def __init__(
        self,
        model: nn.Module,
        device: Optional[Union[str, torch.device]],
        optimizer: Optimizer,
        loss_fn: Loss,
        gradient_accumulation_steps: int = 1,
        gradient_clipping: Optional[float] = None,
        lr_scheduler: Optional[_LRScheduler] = None,
        scheduler_step_policy: Literal["step", "epoch"] = "epoch",
        prepare_batch_fn: Optional[PrepareBatchFn] = None,
        callbacks: Optional[List[Callback]] = None,
        metric_fns: Optional[Dict[str, Metric]] = None,
        evaluate_every: int = 1,
        evaluate_max_steps: Optional[int] = None,
        log_every: int = 10,
        use_amp: bool = False,
        amp_dtype: Optional[torch.dtype] = None,
    ):
        self.device = (
            torch.device(device)
            if device is not None
            else (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        )
        self.model = model
        model_devices = {
            p.device for p in list(self.model.parameters()) + list(self.model.buffers())
        }
        assert model_devices == {
            self.device
        }, f"Model tensors on {model_devices}, trainer on {self.device}"
        self.optimizer = optimizer
        self.scheduler = lr_scheduler
        self.scheduler_step_policy = scheduler_step_policy
        self.prepare_batch_fn = prepare_batch_fn
        self.loss_fn = loss_fn
        if isinstance(self.loss_fn, nn.Module):
            self.loss_fn.to(self.device)
        self.gradient_accumulation_steps = gradient_accumulation_steps
        assert (
            gradient_accumulation_steps >= 1
        ), "gradient_accumulation_steps must be >= 1"
        self.gradient_clipping = gradient_clipping
        if gradient_clipping is not None:
            assert gradient_clipping > 0, "gradient_clipping must be > 0"
        self.callbacks = callbacks if callbacks is not None else []
        self.metric_fns = metric_fns if metric_fns is not None else {}
        for metric in self.metric_fns.values():
            if isinstance(metric, nn.Module):
                metric.to(self.device)
        self.evaluate_every = evaluate_every
        self.evaluate_max_steps = evaluate_max_steps
        self.log_every = log_every
        self.global_step = 0
        self.micro_step = 0
        self.amp_dtype = amp_dtype if amp_dtype is not None else torch.float16
        self.cuda_available = torch.cuda.is_available() and self.device.type == "cuda"
        self.use_amp = use_amp and self.cuda_available
        use_scaler = self.use_amp and self.amp_dtype == torch.float16
        self.scaler = GradScaler(device=self.device.type, enabled=use_scaler)

    def train(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: Optional[torch.utils.data.DataLoader] = None,
        epochs: int = 1,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        for callback in self.callbacks:
            callback.on_train_start(self, **kwargs)
        history = []
        for epoch in range(1, epochs + 1):
            for callback in self.callbacks:
                callback.on_epoch_start(epoch, self, **kwargs)
            train_logs = self._train_epoch(epoch, train_dataloader, **kwargs)
            epoch_logs = {"epoch": epoch, **train_logs}
            if (
                val_dataloader is not None
                and self.evaluate_every is not None
                and (epoch % self.evaluate_every == 0)
            ):
                for callback in self.callbacks:
                    callback.on_validation_start(self, **kwargs)
                val_logs = (
                    self.evaluate(
                        val_dataloader,
                        max_steps=self.evaluate_max_steps,
                        **kwargs,
                    )
                    or {}
                )
                for callback in self.callbacks:
                    callback.on_validation_end(self, val_logs, **kwargs)
                for k, v in val_logs.items():
                    epoch_logs[f"val_{k}"] = v
            if self.scheduler is not None and self.scheduler_step_policy == "epoch":
                self.scheduler.step()
            history.append(epoch_logs)
            for callback in self.callbacks:
                callback.on_epoch_end(epoch, self, epoch_logs)
        for callback in self.callbacks:
            callback.on_train_end(self, **kwargs)
        return history

    @torch.inference_mode()
    def evaluate(
        self, dataloader, max_steps: Optional[int] = None, **kwargs
    ) -> Dict[str, Any]:
        self.model.eval()
        loss_sum, count = 0.0, 0
        metric_sums = {name: 0.0 for name in self.metric_fns}
        metric_counts = {name: 0 for name in self.metric_fns}

        # reset stateful metrics
        for m in self.metric_fns.values():
            if all(hasattr(m, a) for a in ("reset", "update", "compute")):
                try:
                    m.reset()
                except Exception:
                    pass

        steps = 0
        for batch in tqdm(
            dataloader, total=len(dataloader), desc=f"Evaluation"
        ):
            out = self._process_batch(self.global_step, batch, **kwargs)
            loss = out.get("loss")
            if loss is None:
                continue
            # average loss by batch (or weight by samples if you prefer)
            loss_sum += float(loss.detach().item())
            count += 1

            preds = out.get("preds")
            targets = out.get("targets")
            if isinstance(preds, torch.Tensor):
                bsz = (
                    targets.shape[0]
                    if isinstance(targets, torch.Tensor) and targets.ndim > 0
                    else 1
                )
                for name, metric in self.metric_fns.items():
                    try:
                        if all(
                            hasattr(metric, a) for a in ("update", "compute", "reset")
                        ):
                            (
                                metric.update(preds, targets)
                                if isinstance(targets, torch.Tensor)
                                else metric.update(preds)
                            )
                        else:
                            val = (
                                metric(preds, targets)
                                if isinstance(targets, torch.Tensor)
                                else metric(preds)
                            )
                            val = float(
                                val.detach().item()
                                if isinstance(val, torch.Tensor) and val.numel() == 1
                                else (
                                    val.mean().item()
                                    if isinstance(val, torch.Tensor)
                                    else float(val)
                                )
                            )
                            metric_sums[name] += val * bsz
                            metric_counts[name] += bsz
                    except Exception:
                        pass

            steps += 1
            if max_steps is not None and steps >= max_steps:
                break

        logs = {"loss": (loss_sum / max(1, count))}
        # add stateless metric averages
        for name in self.metric_fns:
            if metric_counts.get(name, 0) > 0:
                logs[name] = metric_sums[name] / metric_counts[name]
        # add stateful metric computes
        for name, m in self.metric_fns.items():
            if all(hasattr(m, a) for a in ("update", "compute", "reset")):
                try:
                    val = m.compute()
                    if isinstance(val, torch.Tensor):
                        val = (
                            val.item()
                            if val.numel() == 1
                            else val.float().mean().item()
                        )
                    logs[name] = float(val)
                except Exception:
                    pass
        logs_str = " | ".join([f"{name}: {val}" for name, val in logs.items()])
        logger.info(
            f"Evaluation - {logs_str}"
        )
        return logs

    @torch.inference_mode()
    def predict(
        self,
        dataloader: torch.utils.data.DataLoader,
        *,
        return_targets: bool = False,
        to_cpu: bool = False,
        **kwargs,
    ):
        self.model.eval()
        device_type = self.device.type
        # Use autocast for inference if AMP was requested; no grads due to inference_mode
        with autocast(device_type, dtype=self.amp_dtype, enabled=self.use_amp):
            for batch in dataloader:
                out = self._process_batch(self.global_step, batch, **kwargs)
                preds = out.get("preds")
                targets = out.get("targets", None)
                if not isinstance(preds, torch.Tensor):
                    # Ensure we yield a tensor even in edge cases
                    preds = torch.as_tensor(preds, device=self.device)

                if to_cpu:
                    preds = preds.detach().cpu()
                    if isinstance(targets, torch.Tensor):
                        targets = targets.detach().cpu()
                else:
                    preds = preds.detach()
                    if isinstance(targets, torch.Tensor):
                        targets = targets.detach()

                if return_targets and isinstance(targets, torch.Tensor):
                    yield preds, targets
                else:
                    yield preds

    def _train_epoch(
        self,
        epoch: int,
        train_dataloader,
        **kwargs,
    ) -> Dict[str, Any]:
        self.model.train()
        running_loss_sum = 0.0
        running_loss_count = 0
        metric_sums = {name: 0.0 for name in self.metric_fns.keys()}
        metric_counts = {name: 0 for name in self.metric_fns.keys()}
        # Reset stateful metrics (torchmetrics-style) at epoch start
        for m in self.metric_fns.values():
            if hasattr(m, "reset") and hasattr(m, "update") and hasattr(m, "compute"):
                try:
                    m.reset()
                except Exception:
                    pass
        self.optimizer.zero_grad(set_to_none=True)
        try:
            total_batches = len(train_dataloader)
        except:
            total_batches = None
        for batch_idx, batch in tqdm(
            enumerate(train_dataloader, start=1),
            total=len(train_dataloader),
            desc=f"Training epoch {epoch}",
        ):
            self.micro_step += 1
            for callback in self.callbacks:
                callback.on_batch_start(self.micro_step, self, logs={})
            out = self._process_batch(self.micro_step, batch, **kwargs)
            loss = out.get("loss")
            assert loss is not None, "Loss not found in the output of _process_batch"
            assert torch.is_tensor(loss), "Loss must be a torch.Tensor"
            loss = loss / self.gradient_accumulation_steps
            self.scaler.scale(loss).backward()
            is_last_batch = total_batches is not None and batch_idx == total_batches
            need_update = is_last_batch or (
                self.micro_step % self.gradient_accumulation_steps == 0
            )
            if need_update:
                if self.gradient_clipping is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clipping
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                if self.scheduler is not None and self.scheduler_step_policy == "step":
                    self.scheduler.step()
                self.global_step += 1
            running_loss_sum += float(loss.detach().item())
            running_loss_count += 1
            # Metrics on detached tensors (callable + stateful)
            preds = out.get("preds")
            targets = out.get("targets")
            if isinstance(preds, torch.Tensor):
                bsz = 1
                if isinstance(targets, torch.Tensor) and targets.ndim > 0:
                    bsz = targets.shape[0]
                with torch.no_grad():
                    for name, metric in self.metric_fns.items():
                        # Stateful metric: expects update() / compute() / reset()
                        if (
                            hasattr(metric, "update")
                            and hasattr(metric, "compute")
                            and hasattr(metric, "reset")
                        ):
                            try:
                                if isinstance(targets, torch.Tensor):
                                    metric.update(preds.detach(), targets.detach())
                                else:
                                    # If target is unavailable, try permissive update
                                    metric.update(preds.detach())
                            except Exception:
                                pass
                        else:
                            # Callable metric
                            try:
                                if isinstance(targets, torch.Tensor):
                                    val = metric(preds.detach(), targets.detach())
                                else:
                                    # Allow metrics that only take preds
                                    val = metric(preds.detach())
                                if isinstance(val, torch.Tensor):
                                    val = val.detach()
                                    val = (
                                        val.item()
                                        if val.numel() == 1
                                        else float(val.float().mean().item())
                                    )
                                else:
                                    val = float(val)
                                metric_sums[name] += val * bsz
                                metric_counts[name] += bsz
                            except Exception:
                                pass
            for callback in self.callbacks:
                callback.on_batch_end(
                    self.micro_step,
                    self,
                    logs={
                        "loss": loss,
                        "global_step": self.global_step,
                        "micro_step": self.micro_step,
                    },
                )
        metric_avgs = {}
        for name in self.metric_fns.keys():
            if metric_counts.get(name, 0) > 0:
                metric_avgs[name] = metric_sums[name] / metric_counts[name]
        # Compute stateful metrics at epoch end and merge into logs
        for name, metric in self.metric_fns.items():
            if (
                hasattr(metric, "compute")
                and hasattr(metric, "update")
                and hasattr(metric, "reset")
            ):
                try:
                    val = metric.compute()
                    if isinstance(val, torch.Tensor):
                        val = val.detach()
                        val = (
                            val.item()
                            if val.numel() == 1
                            else float(val.float().mean().item())
                        )
                    else:
                        val = float(val)
                    metric_avgs[name] = val
                except Exception:
                    pass
        # Ensure all metric keys appear in logs (even if not computed)
        for name in self.metric_fns.keys():
            if name not in metric_avgs:
                metric_avgs[name] = 0.0
        epoch_logs = {
            "loss": (
                running_loss_sum / running_loss_count if running_loss_count > 0 else 0.0
            ),
            "micro_steps": self.micro_step,
            "global_steps": self.global_step,
            **metric_avgs,
        }
        if self.log_every and (self.micro_step % self.log_every == 0):
            logs_str = " | ".join([f"{name}: {val}" for name, val in epoch_logs.items()])
            logger.info(
                f"Epoch {epoch} - Micro Step {self.micro_step} - {logs_str}"
            )
        return epoch_logs

    def _process_batch(
        self,
        step: int,
        batch: Batch,
        **kwargs,
    ) -> Dict[str, Union[float, torch.Tensor]]:
        # Move batch to device using your existing hook
        prepared = self._prepare_batch(batch, self.device)

        # Split into inputs / targets with simple, common rules
        inputs: Any
        targets: Optional[torch.Tensor] = None
        if isinstance(prepared, dict):
            tgt_key = next(
                (
                    k
                    for k in ("labels", "label", "y", "target", "targets")
                    if k in prepared
                ),
                None,
            )
            if tgt_key is not None:
                targets = prepared[tgt_key]
                inputs = {k: v for k, v in prepared.items() if k != tgt_key}
            else:
                inputs = prepared
        elif isinstance(prepared, (list, tuple)):
            if len(prepared) >= 2:
                inputs, targets = prepared[0], prepared[1]
            elif len(prepared) == 1:
                inputs = prepared[0]
            else:
                inputs = prepared
        else:
            inputs = prepared

        # Forward pass (AMP if enabled)
        with autocast(self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
            if isinstance(inputs, dict):
                try:
                    outputs = self.model(**inputs)
                except TypeError:
                    outputs = self.model(inputs)
            elif isinstance(inputs, (list, tuple)):
                outputs = self.model(*inputs)
            else:
                outputs = self.model(inputs)

            # Normalize preds tensor
            if isinstance(outputs, torch.Tensor):
                preds = outputs
            elif isinstance(outputs, dict) and isinstance(
                outputs.get("logits"), torch.Tensor
            ):
                preds = outputs["logits"]
            elif (
                isinstance(outputs, (list, tuple))
                and len(outputs)
                and isinstance(outputs[0], torch.Tensor)
            ):
                preds = outputs[0]
            else:
                # Fallback tensor to avoid crashes in edge cases
                preds = torch.as_tensor(
                    0.0, device=self.device, dtype=torch.float32
                ).requires_grad_()

            # Compute loss, supporting (preds, targets) -> (preds, prepared) -> (preds)
            if self.loss_fn is None:
                loss = torch.zeros((), device=self.device, dtype=preds.dtype)
            else:
                loss = None
                if isinstance(targets, torch.Tensor):
                    try:
                        loss = self.loss_fn(preds, targets)
                    except TypeError:
                        loss = None
                if loss is None:
                    try:
                        loss = self.loss_fn(preds, prepared)
                    except TypeError:
                        loss = None
                if loss is None:
                    loss = self.loss_fn(preds)
                if not torch.is_tensor(loss):
                    loss = torch.as_tensor(loss, device=self.device, dtype=preds.dtype)

        out: Dict[str, Union[float, torch.Tensor]] = {"loss": loss, "preds": preds}
        if isinstance(targets, torch.Tensor):
            out["targets"] = targets
        return out

    def _prepare_batch(self, batch: Batch, device: torch.device) -> Batch:
        if self.prepare_batch_fn is not None:
            return self.prepare_batch_fn(batch, device)
        if isinstance(batch, dict):
            return {
                k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()
            }
        elif isinstance(batch, list):
            return [
                item.to(device) if torch.is_tensor(item) else item for item in batch
            ]
        elif isinstance(batch, tuple):
            return tuple(
                item.to(device) if torch.is_tensor(item) else item for item in batch
            )
        elif isinstance(batch, torch.Tensor):
            return batch.to(device)
        raise ValueError(f"Unsupported batch type {type(batch)}")
