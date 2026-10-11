"""Explicit offline re-adjudication of this case; presentation never triggers it."""

import argparse
import json
from pathlib import Path

from cases.master_performance import analysis as performance_analysis
from cases.master_performance.analysis import analyze as analyze_performance
from cases.master_performance.publication import publish_performance


def performance_main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reinterpret", action="store_true",
                        help="explicitly re-adjudicate evidence into a NEW output directory")
    parser.add_argument("--json-only", action="store_true", help="write evidence and verdict without HTML")
    args = parser.parse_args()
    from workload.reinterpretation import load_reinterpretation, import_metrics
    from artifacts.json_io import write_json
    e = load_reinterpretation(parser, args, performance_analysis.__file__)
    r = analyze_performance(e)
    if args.json_only:
        args.output.mkdir(parents=True, exist_ok=True)
        from workload.gate_result import freeze_gate
        freeze_gate(args.output, "performance", e, r)
        write_json(args.output / "analysis.json", r)
    else:
        import_metrics(args.output, args.evidence.parent)
        publish_performance(args.output, e, r)
    print(json.dumps(r, allow_nan=False))
    return {"PASS": 0, "FAIL": 1, "INVALID": 2}[r["verdict"]]
