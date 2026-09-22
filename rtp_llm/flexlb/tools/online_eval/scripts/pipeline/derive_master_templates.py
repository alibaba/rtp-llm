#!/usr/bin/env python3
"""Rebuild the small Java regression view of the pinned anonymous prefix DAG."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from traffic.prefix_lineage import BLOCK, decode, expand
from traffic.traffic_source import sha256_file
from traffic.catalog import model_entry


def derive(model, count=128, max_tokens=32768, output_tokens=420):
    model = Path(model)
    pinned = json.loads(model.with_suffix(".manifest.json").read_text())
    if model.stat().st_size != pinned["bytes"] or sha256_file(model) != pinned["sha256"]:
        raise ValueError("lineage model differs from its pinned manifest")
    metadata, events = decode(model.read_bytes())
    if metadata["count"] != pinned["count"] or len(events) < count:
        raise ValueError("lineage event count differs from its pinned manifest")
    if max_tokens < BLOCK or max_tokens % BLOCK or output_tokens < 1:
        raise ValueError("template limit must be positive and block aligned")
    templates = []
    for _, labels in expand(events[:count]):
        blocks = labels[:max_tokens // BLOCK]
        templates.append(dict(il=len(blocks) * BLOCK, ol=output_tokens, labels=blocks))
    return dict(schema_version=1, source_sha256=pinned["sha256"],
                block_size=BLOCK, templates=templates)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    default_paths = model_entry()[1]
    parser.add_argument("--model", type=Path, default=default_paths["model"])
    parser.add_argument("--out", type=Path, default=default_paths["java_fixture"])
    args = parser.parse_args(argv)
    args.out.write_text(json.dumps(derive(args.model), separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
