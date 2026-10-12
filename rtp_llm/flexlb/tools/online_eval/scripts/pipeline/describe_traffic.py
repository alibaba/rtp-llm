#!/usr/bin/env python3
"""Recompute a real capture's sidecar statistics, preserving manual source metadata."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from traffic.datasets import build_manifest
from traffic.manifest_contract import normalize, layered


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--source-info", type=Path,
                        help="JSON source object with confirmed Spectrum/model information")
    parser.add_argument("--capture-info", type=Path, help="采集执行事实 JSON；不从样本时间推断执行时间")
    args = parser.parse_args(argv)
    sidecar = args.model.with_suffix(".manifest.json")
    existing = json.loads(sidecar.read_text()) if sidecar.exists() else {}
    source = (json.loads(args.source_info.read_text()) if args.source_info else
              normalize(existing).get("source") if existing else None)
    document = layered(build_manifest(args.model, source), previous=existing,
                       capture_execution=json.loads(args.capture_info.read_text()) if args.capture_info else None)
    sidecar.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n")
    print(sidecar)


if __name__ == "__main__":
    main()
