#!/usr/bin/env python3
"""Rebuild the small Java regression view of the pinned anonymous prefix DAG."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from traffic.lineage_model import LineageExpander
from traffic.codecs import decode_model
from traffic.datasets import read_manifest
from file_digest import sha256_file
from traffic.catalog import model_path


def derive(model, count=128, max_tokens=32768, output_tokens=420):
    model = Path(model)
    pinned = read_manifest(model)
    if model.stat().st_size != pinned["bytes"] or sha256_file(model) != pinned["sha256"]:
        raise ValueError("lineage model differs from its pinned manifest")
    lineage = decode_model(model.read_bytes(), pinned)
    metadata, events = lineage.to_wire()
    if metadata["count"] != pinned["count"] or len(events) < count:
        raise ValueError("lineage event count differs from its pinned manifest")
    block_size = metadata["block_size"]
    if max_tokens < block_size or max_tokens % block_size or output_tokens < 1:
        raise ValueError("template limit must be positive and block aligned")
    templates = []
    clipped_count = removed_tokens = 0
    for request in LineageExpander(lineage):
        if len(templates) == count:
            break
        length = min(request.input_tokens, max_tokens)
        clipped_count += request.input_tokens > max_tokens
        removed_tokens += request.input_tokens - length
        templates.append(dict(il=length, ol=output_tokens,
                              labels=list(request.block_labels[:(length + block_size - 1) // block_size])))
    return dict(master_template_schema_version=1, source_sha256=pinned["sha256"],
                block_size=block_size, templates=templates,
                transformations=dict(source_sha256=pinned["sha256"],
                    source_requests=len(events), selected_requests=count, applied=[
                    dict(kind="fixture_shape_clip", max_input_tokens=max_tokens,
                         selected=count, clipped_requests=clipped_count,
                         removed_tokens=removed_tokens),
                    dict(kind="request_limit", limit=count, selected=count,
                         excluded_after_limit=len(events)-count),
                    dict(kind="fixed_output_length", output_tokens=output_tokens,
                         affected_requests=count)]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    default_model = model_path()
    parser.add_argument("--model", type=Path, default=default_model)
    parser.add_argument("--out", type=Path, required=True,
                        help="write the derived fixture to an explicit run directory")
    args = parser.parse_args(argv)
    args.out.write_text(json.dumps(derive(args.model), separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
