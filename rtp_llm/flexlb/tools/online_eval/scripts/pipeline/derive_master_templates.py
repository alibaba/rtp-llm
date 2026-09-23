#!/usr/bin/env python3
"""Rebuild the small Java regression view of the pinned anonymous prefix DAG."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from traffic.capture_contract import BLOCK_SIZE as BLOCK
from traffic.prefix_lineage import expand
from traffic.codecs import decode
from traffic.datasets import read_manifest
from traffic.traffic_source import sha256_file
from traffic.datasets import model_path


def derive(model, count=128, max_tokens=32768, output_tokens=420):
    model = Path(model)
    pinned = read_manifest(model)
    if model.stat().st_size != pinned["bytes"] or sha256_file(model) != pinned["sha256"]:
        raise ValueError("lineage model differs from its pinned manifest")
    metadata, events = decode(model.read_bytes(), pinned)
    if metadata["count"] != pinned["count"] or len(events) < count:
        raise ValueError("lineage event count differs from its pinned manifest")
    if max_tokens < BLOCK or max_tokens % BLOCK or output_tokens < 1:
        raise ValueError("template limit must be positive and block aligned")
    templates = []
    if metadata["version"] == 2:
        for _, labels in expand(events[:count]):
            blocks = labels[:max_tokens // BLOCK]
            templates.append(dict(il=len(blocks) * BLOCK, ol=output_tokens, labels=blocks))
    else:
        # v3 的残缺尾块是私有标签，长度仍保留实测值。
        paths, next_label = [], 1
        for _, length, parent, shared in events[:count]:
            labels = paths[parent][:shared] if parent >= 0 else []
            fresh = (length + BLOCK - 1) // BLOCK - shared
            labels.extend(range(next_label, next_label + fresh))
            next_label += fresh
            paths.append(labels)
            length = min(length, max_tokens)
            templates.append(dict(il=length, ol=output_tokens,
                                  labels=labels[:(length + BLOCK - 1) // BLOCK]))
    return dict(schema_version=1, source_sha256=pinned["sha256"],
                block_size=BLOCK, templates=templates)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    default_model = model_path()
    parser.add_argument("--model", type=Path, default=default_model)
    parser.add_argument("--out", type=Path, default=default_model.with_suffix(".templates.json"))
    args = parser.parse_args(argv)
    args.out.write_text(json.dumps(derive(args.model), separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
