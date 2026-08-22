"""Pytest configuration for 02-pretraining-finetuning tests.

Adds the 01-transformer-architecture/src directory to sys.path so tests
can import the GPT model, tokenizer, and trainer.
"""

import sys
from pathlib import Path

_transformer_src = (
    Path(__file__).resolve().parents[2] / "01-transformer-architecture" / "src"
)
sys.path.insert(0, str(_transformer_src))
