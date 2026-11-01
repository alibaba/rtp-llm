#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Hold one small deterministic shard for check_shard_mapping.py.

Private handles are written to publication/handles.json, never stdout. Create
publication/STOP only after consumers have exited. This is synthetic mapping
validation, not a model checkpoint or an inference acceptance test.
"""
import argparse
import base64
import json
import os
from pathlib import Path
import sys
import threading

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publication", type=Path, required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, default=2)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--numa", type=int, default=0)
    parser.add_argument("--rows-per-shard", type=int, default=262144)
    parser.add_argument("--padding-rows", type=int, default=13)
    args = parser.parse_args()
    if not args.publication.is_absolute() or not 0 <= args.shard < args.shards <= 16:
        parser.error("absolute publication and valid shard required")
    if not 0 <= args.padding_rows < args.rows_per_shard:
        parser.error("padding must leave each shard nonempty")
    from rtp_llm.models_py.modules.dsv41.engram_shard import plan_shards
    import torch  # Initialize ATen/Python bindings before importing the extension.
    import engram_native_dev_492ecf45 as native

    plan = plan_shards(args.rows_per_shard * args.shards - args.padding_rows, 256, args.shards)
    begin, end = plan.row_range(args.shard)
    if plan.shard_bytes > 256 * 1024**2:
        parser.error("synthetic probe is limited to 256 MiB per shard")
    args.publication.mkdir(mode=0o700, parents=True, exist_ok=False)
    owners = []
    try:
        handles = {}
        for role, allocation_bytes, row_bytes, value in (
            ("weight", plan.weight_bytes, 256, args.shard + 1),
            ("scale", plan.scale_bytes, 8, args.shard + 101),
        ):
            owner = native.Owner(allocation_bytes, args.numa, args.device)
            owners.append(owner)
            size = (end - begin) * row_bytes
            chunk = bytes([value]) * min(8 * 1024**2, size)
            for offset in range(0, size, len(chunk)):
                owner.load(chunk[:min(len(chunk), size-offset)], offset)
            handles[role] = base64.b64encode(owner.export_handle()).decode()
        document = {"schema": 1, "plan": plan.as_dict(), "shards": {str(args.shard): handles}}
        fd = os.open(args.publication / "handles.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as stream:
            json.dump(document, stream)
        print(json.dumps({"status": "READY", "shard": args.shard, "bytes": plan.shard_bytes}), flush=True)
        wait = threading.Event()
        while not (args.publication / "STOP").exists():
            wait.wait(1)
    finally:
        for owner in reversed(owners):
            owner.close()
    print(json.dumps({"status": "CLOSED", "counters": native.resource_counters()}), flush=True)


if __name__ == "__main__":
    main()
