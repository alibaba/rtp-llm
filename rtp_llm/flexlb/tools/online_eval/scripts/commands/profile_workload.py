#!/usr/bin/env python3
"""Offline prefix and LRU diagnostics for a token plan."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from analysis.workload_profile import __doc__ as PROFILE_DESCRIPTION
from diagnostics.workload_profile import run


def main():
    p = argparse.ArgumentParser(description=PROFILE_DESCRIPTION)
    p.add_argument("trace", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--capacities", type=int, nargs="+", default=[128, 256, 512, 1024])
    a = p.parse_args()
    if any(n < 1 for n in a.capacities):
        p.error("capacities must be positive")
    a.output.write_text(json.dumps(run(a.trace, a.capacities), indent=2) + "\n")


if __name__ == "__main__":
    main()
