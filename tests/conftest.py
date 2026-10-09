"""tests/conftest.py – shared pytest fixtures."""
import sys
from pathlib import Path

# Ensure project root is importable during tests
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
