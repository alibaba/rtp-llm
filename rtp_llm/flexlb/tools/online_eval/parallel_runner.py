#!/usr/bin/env python3
"""Compatibility entrypoint; implementation lives in scripts/parallel_runner.py."""
import importlib
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
if __name__ == "__main__":
    raise SystemExit(importlib.import_module("scripts.parallel_runner").main())
sys.modules[__name__] = importlib.import_module("scripts.parallel_runner")
