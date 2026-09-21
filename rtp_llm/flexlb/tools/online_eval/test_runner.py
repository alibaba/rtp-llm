#!/usr/bin/env python3
"""Compatibility entrypoint; implementation lives in scripts/test_runner.py."""
import runpy
from pathlib import Path
if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent / "scripts/test_runner.py"), run_name="__main__")
