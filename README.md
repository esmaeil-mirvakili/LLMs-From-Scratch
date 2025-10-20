LLMs From Scratch
=================

This repository contains reference implementations of modern large language
model (LLM) components and end-to-end architectures built “from scratch” with
plain PyTorch. The goal is to provide readable, test-backed examples that show
how popular transformer variants can be assembled, trained, and evaluated
without relying on high-level frameworks.

Implemented architectures
-------------------------

- **GPT‑2** — Decoder-only transformer with multi-head attention, rotary position encodings, and a minimal training harness (`llms_implementation/gpt2`).
- **LLaMA‑3 style** — Grouped-query attention (GQA), SwiGLU feed-forward blocks, RMSNorm, weight tying, and custom training utilities (`llms_implementation/llama3`).
- **DeepSeek‑V3 (experimental)** — Multi-head latent attention (MLA), mixture-of-experts (MoE) layers, and a multi-token prediction head (`llms_implementation/deepseek3`).

Each subpackage exposes a minimal, dependency-light implementation together with unit tests under `tests/` that validate tensor shapes, masking behaviour, RoPE math, MoE routing, and multi-token prediction pipelines.

Requirements
------------

* Python **3.11 – 3.13**
* [PyTorch](https://pytorch.org/) 2.2 – 2.6
* `transformers` 4.41 – 4.49 (for tokenisers)
* `datasets` 3.x (for sample training data)
* Additional utilities: `tiktoken`, `tqdm`, `pytest`, `loguru`

All runtime dependencies are declared in `pyproject.toml`.

Installation
------------

```bash
git clone https://github.com/esmaeil-mirvakili/LLMs-From-Scratch.git
cd LLMs-From-Scratch

python -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`

pip install --upgrade pip
pip install -e .
```

Running the test suite
----------------------

The repository ships with extensive tests for attention layers, RoPE helpers,
MoE routing, multitoken prediction, and end-to-end model behaviour.

```bash
pytest
# or target individual suites:
pytest tests/rope_test.py
pytest tests/llama3
pytest tests/deepseek3
```

Sample training scripts
-----------------------

Each model family comes with a lightweight training harness that exercises the
architecture on a trimmed slice of the WikiText-2 dataset. These scripts are
designed for experimentation—not for reproducing the original models’ quality.

* GPT‑2 example: `python llms_implementation/gpt2/gpt2_training_example.py`
* LLaMA‑3 style example: `python llms_implementation/llama3/llama3_training_example.py`
* DeepSeek‑V3 smoke test: `python llms_implementation/deepseek3/deepseek3_training_example.py`

The training examples share common utilities defined in
`llms_implementation/training_utils.py` which provides a simple Trainer class,
cross-entropy losses, metric hooks, and dataset helpers for sequential text
chunks.

Repository structure
--------------------

```
llms_implementation/
├── gpt2/
│   ├── attention.py            # GPT‑2 attention layers
│   ├── model.py                # GPT‑2 model definition
│   └── gpt2_training_example.py
├── llama3/
│   ├── attention.py            # Grouped Query Attention
│   ├── model.py                # LLaMA‑style stack
│   └── llama3_training_example.py
├── deepseek3/
│   ├── attention.py            # Multi-head latent attention
│   ├── moe.py                  # Mixture-of-experts block
│   ├── mtp.py                  # Multi-token prediction head
│   ├── model.py                # DeepSeek-V3 assembly
│   └── deepseek3_training_example.py
├── rope.py                     # Rotary positional embedding helpers
└── training_utils.py           # Generic trainer & utilities
tests/
├── rope_test.py                # RoPE correctness suite
├── llama3/                     # LLaMA component tests
├── deepseek3/                  # MLA, MoE, MTP, model tests
├── gpt2/                       # GPT‑2 blocks and attention tests
└── training_utils_test.py      # Trainer behaviour and metrics
```
