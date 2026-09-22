#!/usr/bin/env python3
"""Select a test class; five core contracts are the default developer loop.

Use --suite functional for the extended contract matrix, --suite workload for
sustained scenarios, or --suite all for full regression.
Port leasing and lane planning remain in the existing parallel runner.
"""
import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from scripts.pipeline import execute_cases

if __name__ == "__main__":
    if not any(arg == "--suite" or arg.startswith("--suite=") for arg in sys.argv[1:]):
        sys.argv.extend(["--suite", "core"])
    raise SystemExit(execute_cases.main())
