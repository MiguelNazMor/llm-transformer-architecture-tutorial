# Transformers, Adapters, LoRA & QLoRA

Hands-on study of **Transformer architectures** and **Parameter-Efficient Fine-Tuning (PEFT)** methods for Large Language Models.

## Topics

Each topic has its own subfolder with an in-depth theory guide and (eventually) code examples.

| # | Topic | Guide |
|---|-------|-------|
| 1 | **The Transformer Architecture** | [`01-transformer-architecture/README.md`](01-transformer-architecture/README.md) |
| 2 | **Pre-trained Models & Fine-Tuning** | [`02-pretraining-finetuning/README.md`](02-pretraining-finetuning/README.md) |
| 3 | **Adapters** | [`03-adapters/README.md`](03-adapters/README.md) |
| 4 | **LoRA (Low-Rank Adaptation)** | [`04-lora/README.md`](04-lora/README.md) |
| 5 | **QLoRA (Quantized LoRA)** | [`05-qlora/README.md`](05-qlora/README.md) |
| 6 | **The PEFT Ecosystem** | [`06-peft-ecosystem/README.md`](06-peft-ecosystem/README.md) |

## Project Structure

Each topic is a self-contained subfolder with its own theory guide, source code, and tests.

```
01-transformers-adapters-lora-qlora/
├── README.md                          # ← you are here
├── pyproject.toml
├── uv.lock
├── .python-version
├── 01-transformer-architecture/
│   ├── README.md
│   ├── src/
│   └── tests/
├── 02-pretraining-finetuning/
│   ├── README.md
│   ├── src/
│   └── tests/
├── 03-adapters/
│   ├── README.md
│   ├── src/
│   └── tests/
├── 04-lora/
│   ├── README.md
│   ├── src/
│   └── tests/
├── 05-qlora/
│   ├── README.md
│   ├── src/
│   └── tests/
└── 06-peft-ecosystem/
    ├── README.md
    ├── src/
    └── tests/
```

## Setup

```bash
# Clone and enter the repo
cd 01-transformers-adapters-lora-qlora

# uv will automatically install Python 3.14 and create a venv
uv sync

# Run code from any subfolder, e.g.:
uv run python 01-transformer-architecture/src/main.py
```

## Key References

- ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) — Vaswani et al., 2017
- ["Parameter-Efficient Transfer Learning for NLP"](https://arxiv.org/abs/1902.00751) (Adapters) — Houlsby et al., 2019
- ["LoRA: Low-Rank Adaptation of Large Language Models"](https://arxiv.org/abs/2106.09685) — Hu et al., 2021
- ["QLoRA: Efficient Finetuning of Quantized LLMs"](https://arxiv.org/abs/2305.14314) — Dettmers et al., 2023
- [Hugging Face PEFT Documentation](https://huggingface.co/docs/peft)
