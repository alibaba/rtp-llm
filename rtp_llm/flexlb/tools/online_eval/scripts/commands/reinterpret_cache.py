#!/usr/bin/env python3
"""Offline gate analysis; implementation is under src."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from cases.cache_scale_in.replay import cache_main

if __name__ == "__main__":
    raise SystemExit(cache_main())
