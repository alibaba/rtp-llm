#!/usr/bin/env python3
"""Recompute a real capture's sidecar statistics, preserving manual source metadata."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from traffic.datasets import build_manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--source-info", type=Path,
                        help="JSON source object with confirmed Spectrum/model information")
    args = parser.parse_args(argv)
    sidecar = args.model.with_suffix(".manifest.json")
    existing = json.loads(sidecar.read_text()) if sidecar.exists() else {}
    source = json.loads(args.source_info.read_text()) if args.source_info else existing.get("source")
    document = build_manifest(args.model, source)
    sidecar.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n")
    print(sidecar)


if __name__ == "__main__":
    main()
