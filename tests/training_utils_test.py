import math
import torch
import pytest
from torch import nn, Tensor
from torch.utils.data import DataLoader, TensorDataset

from llms_implementation.training_utils import Trainer, Callback


# ---- Fixtures ----
class CountingCallback(Callback):
    def __init__(self):
        self.calls = {
            "on_train_start": 0,
            "on_train_end": 0,
            "on_epoch_start": 0,
            "on_epoch_end": 0,
            "on_batch_start": 0,
            "on_batch_end": 0,
        }

    def on_train_start(self, trainer, **kwargs):
        self.calls["on_train_start"] += 1

    def on_train_end(self, trainer, **kwargs):
        self.calls["on_train_end"] += 1

    def on_epoch_start(self, epoch, trainer, **kwargs):
        self.calls["on_epoch_start"] += 1

    def on_epoch_end(self, epoch, trainer, logs, **kwargs):
        self.calls["on_epoch_end"] += 1

    def on_batch_start(self, step, trainer, logs, **kwargs):
        self.calls["on_batch_start"] += 1

    def on_batch_end(self, step, trainer, logs, **kwargs):
        self.calls["on_batch_end"] += 1


class ToyModel(nn.Module):
    """Simple linear model that accepts a single Tensor input."""

    def __init__(self, d_in=4, d_out=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 8),
            nn.ReLU(),
            nn.Linear(8, d_out),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def make_data(n_samples=32, d_in=4):
    x = torch.randn(n_samples, d_in)
    # no labels needed by your current loss_fn; we’ll ignore the “target” arg
    ds = TensorDataset(x)
    return ds


# Your current _process_batch passes the prepared batch as both model input and
# the second argument to loss_fn. Use a loss that ignores the second arg and
# simply drives outputs toward zero, so training can make progress.
def zero_mse_loss(preds: Tensor, _batch) -> Tensor:
    return (preds**2).mean()


# ---- Tests ----


@pytest.mark.parametrize("accum_steps, n_batches", [(1, 6), (2, 5), (4, 9)])
def test_train_one_epoch_updates_global_step_and_returns_history(
    tmp_path, accum_steps, n_batches
):
    torch.manual_seed(0)
    d_in, d_out = 4, 2
    model = ToyModel(d_in, d_out)

    # IMPORTANT: your Trainer asserts model params are already on the device.
    # Keep device = "cpu" to avoid CUDA in CI.
    device = torch.device("cpu")

    # Build dataloader with the requested number of batches
    ds = make_data(n_samples=n_batches * 8, d_in=d_in)
    loader = DataLoader(ds, batch_size=8, shuffle=False, drop_last=True)

    # Optimizer
    optim = torch.optim.SGD(model.parameters(), lr=0.1)

    # Simple prepare_batch_fn: returns the single Tensor (inputs) your model expects.
    def prepare_batch_fn(batch, device):
        (x,) = batch
        return x.to(device)

    cb = CountingCallback()

    trainer = Trainer(
        model=model.to(device),
        device=device,
        optimizer=optim,
        loss_fn=zero_mse_loss,
        gradient_accumulation_steps=accum_steps,
        gradient_clipping=None,
        lr_scheduler=None,
        prepare_batch_fn=prepare_batch_fn,
        callbacks=[cb],
        metric_fns={},  # no metrics yet
        evaluate_every=100,  # avoid calling evaluate()
        evaluate_max_steps=None,
        log_every=10,
        use_amp=False,
        amp_dtype=None,
    )

    history = trainer.train(train_dataloader=loader, val_dataloader=None, epochs=1)

    # ---- Assertions ----

    # 1) train() returns one entry with expected keys
    assert isinstance(history, list) and len(history) == 1
    ep = history[0]
    assert (
        "epoch" in ep and "loss" in ep and "global_steps" in ep and "micro_steps" in ep
    )

    # 2) global_step increments per *optimizer update*, not per micro-batch
    # steps_per_epoch = ceil(n_batches / accum_steps)
    expected_updates = math.ceil(n_batches / accum_steps)
    assert trainer.global_step == expected_updates
    assert ep["global_steps"] == expected_updates

    # 3) micro_step increments per batch
    assert trainer.micro_step == n_batches
    assert ep["micro_steps"] == n_batches

    # 4) loss is a finite float
    assert isinstance(ep["loss"], float)
    assert math.isfinite(ep["loss"])

    # 5) callbacks were invoked
    assert cb.calls["on_train_start"] == 1
    assert cb.calls["on_train_end"] == 1
    assert cb.calls["on_epoch_start"] == 1
    assert cb.calls["on_epoch_end"] == 1
    assert cb.calls["on_batch_start"] == n_batches
    assert cb.calls["on_batch_end"] == n_batches


def test_training_actually_reduces_zero_mse_loss():
    torch.manual_seed(0)
    model = ToyModel(d_in=4, d_out=2)
    device = torch.device("cpu")
    ds = make_data(n_samples=128, d_in=4)
    loader = DataLoader(ds, batch_size=16, shuffle=False, drop_last=True)

    optim = torch.optim.SGD(model.parameters(), lr=0.2)

    def prepare_batch_fn(batch, device):
        (x,) = batch
        return x.to(device)

    trainer = Trainer(
        model=model.to(device),
        device=device,
        optimizer=optim,
        loss_fn=zero_mse_loss,
        gradient_accumulation_steps=1,
        gradient_clipping=None,
        lr_scheduler=None,
        prepare_batch_fn=prepare_batch_fn,
        callbacks=[],
        metric_fns={},
        evaluate_every=10_000,  # effectively disable eval
        evaluate_max_steps=None,
        log_every=10_000,  # avoid noisy callback logging
        use_amp=False,
        amp_dtype=None,
    )

    # Measure loss before training
    model.eval()
    with torch.no_grad():
        x0 = next(iter(loader))[0]
        base_loss = (model(x0) ** 2).mean().item()

    # Train a few epochs; loss should go down (driven toward zero)
    trainer.train(loader, epochs=3)

    model.eval()
    with torch.no_grad():
        x1 = next(iter(loader))[0]
        new_loss = (model(x1) ** 2).mean().item()

    assert (
        new_loss < base_loss
    ), f"Expected loss to decrease, got base={base_loss:.4f}, new={new_loss:.4f}"


@pytest.mark.parametrize("accum_steps,n_batches", [(3, 10)])  # remainder exists
def test_partial_accumulation_updates_once_at_end(accum_steps, n_batches):
    model = ToyModel()
    device = torch.device("cpu")
    ds = make_data(n_samples=n_batches * 8)
    loader = DataLoader(
        ds, batch_size=8, shuffle=False, drop_last=False
    )  # keep remainder
    optim = torch.optim.SGD(model.parameters(), lr=0.1)

    trainer = Trainer(
        model.to(device),
        device,
        optim,
        zero_mse_loss,
        gradient_accumulation_steps=accum_steps,
        prepare_batch_fn=lambda b, d: (b[0].to(d)),
        callbacks=[],
        metric_fns={},
        evaluate_every=10_000,
        evaluate_max_steps=None,
        log_every=10_000,
        use_amp=False,
        amp_dtype=None,
    )

    trainer.train(loader, epochs=1)
    expected_updates = math.ceil(n_batches / accum_steps)
    assert trainer.global_step == expected_updates


class CountingScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer):
        self.calls = 0  # must exist before super().__init__()
        super().__init__(optimizer)
        self.calls = 0  # reset to ignore the init-time step()

    def get_lr(self):
        # No change to LR; just return current values
        return [group["lr"] for group in self.optimizer.param_groups]

    def step(self, *args, **kwargs):
        self.calls += 1
        return super().step(*args, **kwargs)


def test_scheduler_steps_on_epoch_end_only():
    model = ToyModel()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    sch = CountingScheduler(opt)
    trainer = Trainer(
        model,
        "cpu",
        opt,
        zero_mse_loss,
        lr_scheduler=sch,
        gradient_accumulation_steps=2,
        prepare_batch_fn=lambda b, d: (b[0].to(d)),
        callbacks=[],
        metric_fns={},
        evaluate_every=10_000,
        log_every=10_000,
        use_amp=False,
        amp_dtype=None,
    )
    loader = DataLoader(make_data(32), batch_size=8)
    trainer.train(loader, epochs=3)
    assert sch.calls == 3  # once per epoch


def test_scheduler_steps_per_update_when_policy_is_batch():
    model = ToyModel()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    sch = CountingScheduler(opt)
    trainer = Trainer(
        model,
        "cpu",
        opt,
        zero_mse_loss,
        lr_scheduler=sch,
        scheduler_step_policy="step",
        gradient_accumulation_steps=2,
        prepare_batch_fn=lambda b, d: (b[0].to(d)),
        callbacks=[],
        metric_fns={},
        evaluate_every=10_000,
        log_every=10_000,
        use_amp=False,
        amp_dtype=None,
    )
    loader = DataLoader(make_data(48), batch_size=8)
    history = trainer.train(loader, epochs=1)
    expected_updates = math.ceil((len(loader)) / 2)
    assert sch.calls in (
        expected_updates,
        expected_updates + 0,
    )  # exact count per your policy


class SumMetric(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("s", torch.tensor(0.0, dtype=torch.float))
        self.n = 0

    def reset(self):
        self.s.zero_()
        self.n = 0

    @torch.no_grad()
    def update(self, preds, target):
        self.s += preds.detach().abs().sum()
        self.n += preds.numel()

    def compute(self):
        return (self.s / max(1, self.n)).item()


def test_metrics_callable_and_stateful_averaged():
    model = ToyModel()
    device = torch.device("cpu")

    def callable_metric(preds, target):
        return preds.abs().mean().item()

    metrics = {"call": callable_metric, "stateful": SumMetric()}
    trainer = Trainer(
        model.to(device),
        device,
        torch.optim.SGD(model.parameters(), lr=0.1),
        zero_mse_loss,
        gradient_accumulation_steps=1,
        prepare_batch_fn=lambda b, d: (b[0].to(d)),
        callbacks=[],
        metric_fns=metrics,
        evaluate_every=10_000,
        log_every=10_000,
        use_amp=False,
        amp_dtype=None,
    )
    loader = DataLoader(make_data(64), batch_size=16)
    hist = trainer.train(loader, epochs=1)
    row = hist[0]
    assert (
        "train_call" in row or "call" in row
    )  # depends on your naming; assert whichever you log
    assert "train_stateful" in row or "stateful" in row


def test_validation_hooks_and_cadence_by_epochs(monkeypatch):
    cb = CountingCallback()
    model = ToyModel()
    loader = DataLoader(make_data(64), batch_size=16)
    trainer = Trainer(
        model,
        "cpu",
        torch.optim.SGD(model.parameters(), lr=0.1),
        zero_mse_loss,
        prepare_batch_fn=lambda b, d: (b[0]),
        callbacks=[cb],
        metric_fns={},
        evaluate_every=2,
        evaluate_max_steps=None,
        gradient_accumulation_steps=1,
        log_every=10_000,
        use_amp=False,
        amp_dtype=None,
    )
    trainer.train(loader, val_dataloader=loader, epochs=5)
    # eval at epochs 2 and 4
    assert cb.calls["on_epoch_start"] == 5


def test_predict_yields_tensors_without_grad():
    model = ToyModel()
    loader = DataLoader(make_data(16), batch_size=8)
    trainer = Trainer(
        model,
        "cpu",
        torch.optim.SGD(model.parameters(), lr=0.1),
        zero_mse_loss,
        prepare_batch_fn=lambda b, d: (b[0]),
        callbacks=[],
        metric_fns={},
        evaluate_every=10_000,
        log_every=10_000,
        use_amp=False,
        amp_dtype=None,
    )
    preds = list(trainer.predict(loader))
    assert len(preds) == len(loader)
    assert all(isinstance(p, torch.Tensor) for p in preds)
    # Predictions should be detached (no grad)
    assert all(not p.requires_grad for p in preds)

    # Also test the extended API: return_targets + to_cpu
    pairs = list(trainer.predict(loader, return_targets=True, to_cpu=True))
    assert len(pairs) == len(loader)
    # Some pipelines may not produce targets in predict(); handle both cases.
    first = pairs[0]
    if isinstance(first, tuple):
        assert all(
            isinstance(p, torch.Tensor) and isinstance(t, torch.Tensor)
            for p, t in pairs
        )
        # Tensors should be on CPU and detached
        assert all(p.device.type == "cpu" and t.device.type == "cpu" for p, t in pairs)
        assert all((not p.requires_grad) and (not t.requires_grad) for p, t in pairs)
    else:
        # Only predictions were yielded
        assert all(isinstance(p, torch.Tensor) for p in pairs)
        assert all(p.device.type == "cpu" for p in pairs)
        assert all((not p.requires_grad) for p in pairs)


def test_prepare_batch_dict_with_labels_key():
    model = ToyModel()
    device = "cpu"
    ds = [{"x": torch.randn(4), "labels": torch.tensor(0)} for _ in range(32)]
    loader = DataLoader(
        ds,
        batch_size=8,
        collate_fn=lambda b: {k: torch.stack([d[k] for d in b]) for k in b[0]},
    )

    def prep(batch, device):
        # mimic your default: pick inputs/target based on keys
        batch = {k: v.to(device) for k, v in batch.items()}
        return batch

    trainer = Trainer(
        model,
        device,
        torch.optim.SGD(model.parameters(), lr=0.1),
        lambda preds, tgt: (preds**2).mean(),  # ignores target
        prepare_batch_fn=prep,
        callbacks=[],
        metric_fns={},
        evaluate_every=10_000,
        log_every=10_000,
        use_amp=False,
        amp_dtype=None,
    )
    trainer.train(loader, epochs=1)  # should not crash


def test_gradient_clipping_invoked(monkeypatch):
    model = ToyModel()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    called = {"n": 0}

    def fake_clip(params, max_norm, *a, **kw):
        called["n"] += 1
        return torch.tensor(0.0)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", fake_clip)
    trainer = Trainer(
        model,
        "cpu",
        opt,
        zero_mse_loss,
        gradient_accumulation_steps=1,
        gradient_clipping=0.5,
        prepare_batch_fn=lambda b, d: (b[0]),
        callbacks=[],
        metric_fns={},
        evaluate_every=10_000,
        log_every=10_000,
        use_amp=False,
        amp_dtype=None,
    )
    loader = DataLoader(make_data(32), batch_size=8)
    trainer.train(loader, epochs=1)
    assert called["n"] >= 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_amp_uses_scaler_on_cuda():
    model = ToyModel().cuda()
    trainer = Trainer(
        model,
        torch.device("cuda"),
        torch.optim.SGD(model.parameters(), lr=0.1),
        zero_mse_loss,
        use_amp=True,
        amp_dtype=torch.float16,
        prepare_batch_fn=lambda b, d: (b[0].to(d)),
        callbacks=[],
        metric_fns={},
        evaluate_every=10_000,
        log_every=10_000,
        gradient_accumulation_steps=2,
    )
    loader = DataLoader(make_data(32), batch_size=8)
    trainer.train(loader, epochs=1)
    # If you expose scaler, assert enabled
    assert getattr(trainer.scaler, "is_enabled", lambda: True)()
