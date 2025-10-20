# train_gpt2_wikitext.py
import math
from dataclasses import dataclass
from typing import List, Tuple, Dict

import torch
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset

from transformers import AutoTokenizer

# import your model and trainer (ensure repo root on PYTHONPATH or run as module)
from llms_implementation.gpt2.model import GPT2
from llms_implementation.training_utils import Trainer


class LMSequenceDataset(Dataset):

    def __init__(self, data: Tensor, seq_len: int):
        assert data.ndim == 1
        self.data = data
        self.seq_len = seq_len
        # number of full chunks that can be made
        self.n = (len(self.data) - 1) // seq_len

    def __len__(self) -> int:
        return max(0, self.n)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        i = idx * self.seq_len
        x = self.data[i : i + self.seq_len]
        y = self.data[i + 1 : i + 1 + self.seq_len]
        return x, y


def collate_fn(batch: List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Tensor]:
    xs, ys = zip(*batch)
    return torch.stack(xs, dim=0), torch.stack(ys, dim=0)


def prepare_batch_fn(batch, device):
    x, y = batch
    return x.to(device), y.to(device)


# ------------------------------ Loss & metrics ------------------------------


def sequence_ce_loss(logits: Tensor, targets: Tensor) -> Tensor:
    # logits: (B,T,V), targets: (B,T)
    B, T, V = logits.shape
    return nn.functional.cross_entropy(logits.view(B * T, V), targets.view(B * T))


def token_accuracy(logits: Tensor, targets: Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return (pred == targets).float().mean().item()


def perplexity_from_loss(logits: Tensor, targets: Tensor) -> float:
    with torch.no_grad():
        loss = sequence_ce_loss(logits, targets)
        return float(math.exp(min(20.0, loss.item())))  # clamp to avoid inf


# ------------------------------ Config ------------------------------


@dataclass
class TrainConfig:
    seq_len: int = 128  # keep small for CPU/Mac
    embed_dim: int = 128  # MUST equal seq_len for your current model.py
    num_head: int = 4
    num_layers: int = 2
    vocab_limit: int = 0  # 0 = use all chars
    batch_size: int = 32
    epochs: int = 10
    lr: float = 3e-4
    weight_decay: float = 0.01
    attn_pdrop: float = 0.0  # keep 0 for determinism
    use_fused: bool = True  # if PyTorch SDPA is available
    device: str = "cpu"  # use "cuda" if available and you want it


# ------------------------------ Main ------------------------------


def main():
    cfg = TrainConfig()

    ds_train = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:5%]")
    ds_valid = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation[:1_000]")

    train_texts = [t for t in ds_train["text"] if len(t) > 0]
    valid_texts = [t for t in ds_valid["text"] if len(t) > 0]
    tok = AutoTokenizer.from_pretrained("openai-community/gpt2", use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def encode_corpus(texts):
        ids = []
        for line in texts:
            if not line: 
                continue
            ids.extend(tok.encode(line + "\n", add_special_tokens=False))
        return torch.tensor(ids, dtype=torch.long)

    train_ids = encode_corpus(train_texts)
    valid_ids = encode_corpus(valid_texts)

    train_dataset = LMSequenceDataset(train_ids, seq_len=cfg.seq_len)
    valid_dataset = LMSequenceDataset(valid_ids, seq_len=cfg.seq_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=True,
        collate_fn=collate_fn,
    )

    model = GPT2(
        {
            "embed_dim": cfg.embed_dim,  # equals seq_len
            "seq_len": cfg.seq_len,  # equals embed_dim
            "num_head": cfg.num_head,
            "num_layers": cfg.num_layers,
            "vocab_size": tok.vocab_size,
            "attn_config": {"attn_pdrop": cfg.attn_pdrop, "use_fused": cfg.use_fused},
        }
    )

    device = torch.device(
        cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu"
    )
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = None

    trainer = Trainer(
        model=model,
        device=device,  # IMPORTANT: must match model tensors’ device
        optimizer=optimizer,
        loss_fn=sequence_ce_loss,
        gradient_accumulation_steps=1,
        gradient_clipping=1.0,
        lr_scheduler=scheduler,
        scheduler_step_policy="epoch",
        prepare_batch_fn=prepare_batch_fn,
        callbacks=[],
        metric_fns={
            "acc": token_accuracy,
            "ppl": perplexity_from_loss,  # derived from CE loss
        },
        evaluate_every=5,
        evaluate_max_steps=100,  # keep eval cheap; adjust as needed
        log_every=1,
        use_amp=False,  # your Trainer only AMP’s on CUDA float16
        amp_dtype=None,
    )

    history = trainer.train(
        train_loader, val_dataloader=valid_loader, epochs=cfg.epochs
    )
    print("Final epoch logs:", history[-1])

    with torch.inference_mode():
        batch = next(iter(valid_loader))
        x, y = prepare_batch_fn(batch, device)
        logits = model(x)
        print("Logits shape:", tuple(logits.shape))


if __name__ == "__main__":
    main()
