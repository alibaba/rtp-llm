#!/usr/bin/env python3
"""Developer command; implementation is under src/stress."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
import runpy

if __name__ == "__main__":
    runpy.run_path(str(ROOT / "src/stress/analysis/aggregate.py"), run_name="__main__")
