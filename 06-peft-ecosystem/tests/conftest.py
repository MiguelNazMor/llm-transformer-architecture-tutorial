"""Pytest configuration for 06-peft-ecosystem tests."""
import sys
from pathlib import Path

for sub in ["01-transformer-architecture", "06-peft-ecosystem"]:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / sub / "src"))
