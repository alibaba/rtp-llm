#!/usr/bin/env python3
"""Stable-topology cache GC diagnostic experiment command."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from experiments.cache_gc_background.report import main

if __name__ == "__main__":
    raise SystemExit(main())
