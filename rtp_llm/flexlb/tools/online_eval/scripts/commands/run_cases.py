#!/usr/bin/env python3
"""Run instances selected by config/suites.yaml or an explicit kind filter.

Use --suite for explicit groups, --kind for all variants of a kind,
or --all for the complete catalog. Listing and execution share defaults.
Port leasing and lane planning remain in the existing parallel runner.
"""
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from scripts.pipeline import execute_cases

if __name__ == "__main__":
    raise SystemExit(execute_cases.main())
