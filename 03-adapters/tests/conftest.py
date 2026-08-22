"""Pytest configuration for 03-adapters tests."""
import sys
from pathlib import Path

_transformer_src = Path(__file__).resolve().parents[2] / "01-transformer-architecture" / "src"
_adapter_src = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(_transformer_src))
sys.path.insert(0, str(_adapter_src))
