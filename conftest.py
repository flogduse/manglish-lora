"""Repo-root conftest: guarantees `app` (or repo-local packages) are importable
from tests under any pytest invocation: plain `pytest`, `python -m pytest`, CI
runners. Without this, plain `pytest` on Linux does not put the root on sys.path
and collection fails with ModuleNotFoundError: No module named 'app'.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
