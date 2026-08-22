"""Pytest configuration for 04-lora tests."""
import sys
from pathlib import Path

_transformer_src = Path(__file__).resolve().parents[2] / "01-transformer-architecture" / "src"
_lora_src = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_transformer_src))
sys.path.insert(0, str(_lora_src))
