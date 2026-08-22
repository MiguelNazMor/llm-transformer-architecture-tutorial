"""Pytest configuration for 05-qlora tests."""
import sys
from pathlib import Path

for sub in ["01-transformer-architecture", "04-lora", "05-qlora"]:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / sub / "src"))
