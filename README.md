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

# Run chapter 1 training to generate base_model.npz / base_tokenizer.json
# (or use the pre-trained artifacts already committed in the repo):
PYTHONPATH=data:01-transformer-architecture/src uv run python 01-transformer-architecture/src/train.py

# Run the PEFT comparison on the committed base model:
PYTHONPATH=data:06-peft-ecosystem/src:05-qlora/src:04-lora/src:03-adapters/src:02-pretraining-finetuning/src:01-transformer-architecture/src \
    uv run python 06-peft-ecosystem/src/compare.py
```

## Pre-trained artifacts

A pre-trained base model is committed in this repo so the PEFT comparisons in chapters 2-6 can be run without retraining:

- `01-transformer-architecture/base_model.npz` (5.2M parameters)
- `01-transformer-architecture/base_tokenizer.json`

The model was trained on the public-domain text `data/general_corpus.txt` (Alice's Adventures in Wonderland) and the tokenizer was fit to the combined `data/general_corpus.txt` + `data/hp_corpus.txt`.  To regenerate the artifacts, run `01-transformer-architecture/src/train.py`.

## Key References

- ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) — Vaswani et al., 2017
- ["Parameter-Efficient Transfer Learning for NLP"](https://arxiv.org/abs/1902.00751) (Adapters) — Houlsby et al., 2019
- ["LoRA: Low-Rank Adaptation of Large Language Models"](https://arxiv.org/abs/2106.09685) — Hu et al., 2021
- ["QLoRA: Efficient Finetuning of Quantized LLMs"](https://arxiv.org/abs/2305.14314) — Dettmers et al., 2023
- [Hugging Face PEFT Documentation](https://huggingface.co/docs/peft)
