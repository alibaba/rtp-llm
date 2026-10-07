#!/usr/bin/env python3
"""Run a monitored Java Master + Mock Engine stress experiment."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from runtime.stress import main

if __name__ == "__main__":
    raise SystemExit(main())
