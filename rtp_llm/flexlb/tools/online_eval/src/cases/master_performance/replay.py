"""Explicit offline re-adjudication of this case; presentation never triggers it."""

import argparse
import json
from pathlib import Path

from cases.master_performance.analysis import analyze as analyze_performance
from cases.master_performance.publication import publish_performance


def performance_main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--json-only", action="store_true", help="write evidence and verdict without HTML")
    args = parser.parse_args()
    e = json.loads(args.evidence.read_text())
    r = analyze_performance(e)
    if args.json_only:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "performance-gate-evidence.json").write_text(json.dumps(e, allow_nan=False))
        (args.output / "analysis.json").write_text(json.dumps(r, indent=2, allow_nan=False))
    else:
        publish_performance(args.output, e, r, args.evidence.parent)
    print(json.dumps(r, allow_nan=False))
    return {"PASS": 0, "FAIL": 1, "INVALID": 2}[r["verdict"]]
