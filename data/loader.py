"""Corpus loader for the PEFT comparison project.

Provides utilities to load the general pre-training corpus (extracted from
Project Gutenberg's "Alice's Adventures in Wonderland") and the domain-specific
fine-tuning corpus (Harry Potter-themed sentences).

Both files live in this ``data/`` directory as plain text, one sentence per line.
"""

from __future__ import annotations

import random
from pathlib import Path

_DATA_DIR = Path(__file__).resolve().parent


def load_general_corpus() -> list[str]:
    """Loads the general pre-training corpus (Alice in Wonderland sentences).

    Returns:
        List of sentences, one per line from general_corpus.txt.
    """
    path = _DATA_DIR / "general_corpus.txt"
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_hp_corpus() -> list[str]:
    """Loads the Harry Potter domain corpus for fine-tuning.

    Returns:
        List of HP-themed sentences, one per line from hp_corpus.txt.
    """
    path = _DATA_DIR / "hp_corpus.txt"
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_combined_corpus() -> list[str]:
    """Loads both corpora combined (for tokenizer training).

    The tokenizer is trained on the combined corpus so that HP-specific
    vocabulary (Hogwarts, Gryffindor, Quidditch, etc.) gets proper subword
    merges rather than being split into individual characters.

    Returns:
        List of all sentences from both corpora.
    """
    return load_general_corpus() + load_hp_corpus()


def split_corpus(
    corpus: list[str], eval_fraction: float = 0.2, seed: int = 42
) -> tuple[list[str], list[str]]:
    """Splits a corpus into deterministic training and held-out evaluation sets."""
    if not 0 < eval_fraction < 1:
        raise ValueError("eval_fraction must be between 0 and 1.")
    if len(corpus) < 2:
        raise ValueError("corpus must contain at least two sentences.")

    shuffled = corpus.copy()
    random.Random(seed).shuffle(shuffled)
    eval_size = max(1, round(len(shuffled) * eval_fraction))
    return shuffled[eval_size:], shuffled[:eval_size]
