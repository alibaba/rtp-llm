#!/usr/bin/env python3
"""Build a Java load-client plan from one of the two registered traffic sources."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from traffic.traffic_source import materialize
from traffic.datasets import read_manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--spec", type=Path, help="JSON specification of a registered traffic source")
    source.add_argument("--lineage-model", type=Path, help="pinned prefix DAG model")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--output-tokens", type=int, default=420)
    parser.add_argument("--priority", type=int, default=50)
    parser.add_argument("--max-requests", type=int, help="sender projection; keeps the pinned source unchanged")
    args = parser.parse_args(argv)
    if args.spec:
        spec = json.loads(args.spec.read_text())
        base_dir = args.spec.resolve().parent
    else:
        model = args.lineage_model.resolve()
        manifest = read_manifest(model)
        spec = dict(kind="trace", model="prefix_lineage", version=str(manifest["codec"]["version"]), parameters=dict(
            path=model.name, sha256=manifest["sha256"], count=manifest["count"],
            output_tokens=args.output_tokens, priority=args.priority,
        ))
        base_dir = model.parent
    materialize(args.out, spec, args.namespace, base_dir, max_requests=args.max_requests)
    print(args.out.with_suffix(".manifest.json"))


if __name__ == "__main__":
    main()
