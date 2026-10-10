"""Explicit offline re-adjudication of this case; presentation never triggers it."""

import argparse
import hashlib
import json
from pathlib import Path

from cases.cache_scale_in import analysis as cache_analysis
from cases.cache_scale_in.analysis import MEASUREMENT_POLICY, align_send_counters, analyze, attribute_client
from cases.cache_scale_in.report import prepare_report
from cases.cache_scale_in.publication import publish_cache


def cache_main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reinterpret", action="store_true",
                        help="explicitly re-adjudicate historical evidence into a NEW output directory")
    parser.add_argument("--client-snapshot", type=Path,
                        help="complete java_flow evidence_snapshot JSON for historical attribution")
    args = parser.parse_args()
    from workload.reinterpretation import load_reinterpretation, import_metrics
    evidence = load_reinterpretation(parser, args, cache_analysis.__file__)
    original_errors = list(evidence.get("errors", []))
    if args.client_snapshot:
        snapshot = json.loads(args.client_snapshot.read_text())
        if len(snapshot["issued"]) == snapshot.get("status", {}).get("submitted"):
            align_send_counters(evidence, snapshot["issued"])
        else:
            evidence.setdefault("errors", []).append("issued send accounting incomplete")
        attribute_client(evidence, snapshot)
    evidence["reinterpretation"].update(
        client_snapshot=(dict(path=str(args.client_snapshot.resolve()),
                              sha256=hashlib.sha256(args.client_snapshot.read_bytes()).hexdigest())
                         if args.client_snapshot else None),
        original_errors=original_errors,
        window_policy="preserve archived baseline/post windows and all thresholds",
        limitations=[] if evidence.get("measurement_policy") == MEASUREMENT_POLICY else
            ["historical admission timing cannot be reconstructed; detach-based windows retained"],
    )
    args.output.mkdir(parents=True, exist_ok=True)
    result = analyze(evidence)
    import_metrics(args.output, args.evidence.parent, "cache_scale_in.yaml")
    publish_cache(args.output, evidence, result,
                 prepared=prepare_report(args.output, evidence))
    print(result["verdict"])
    return {"PASS": 0, "FAIL": 1, "INVALID": 2}[result["verdict"]]
