#!/usr/bin/env python3
"""Developer command; implementation is under src/flexlb_eval."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from flexlb_eval.analysis.compare_twin import main

if __name__ == "__main__":
    raise SystemExit(main())
