#!/usr/bin/env python3
"""Developer command; implementation is under src."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from analysis.compare_twin import main

if __name__ == "__main__":
    raise SystemExit(main())
