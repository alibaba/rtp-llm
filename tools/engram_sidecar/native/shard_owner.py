#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Publish one Engram shard's weight/scale fabric allocations and wait.

Run this on the node that owns the shard.  It writes a private JSON capability
file (mode 0600) containing the two fabric handles for that shard, prints
``READY``, and keeps both owners alive until the coordinator creates the
``--stop-file``.  A consumer on another node imports the handles and maps them
into its two VA spans.

    python3 shard_owner.py --shard 1 --shards 2 \
        --out /workspace/shard-check/shard1.json \
        --stop-file /workspace/shard-check/STOP

Never log or copy the handle file outside the authenticated transfer path.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path
from uuid import uuid4

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from load_native import load_native  # noqa: E402

from rtp_llm.models_py.modules.dsv41.engram_shard import plan_shards  # noqa: E402

DIM = 256
BLOCK = 32
ROWS_PER_SHARD = 262144
LOAD_CHUNK = 64 * 1024 * 1024


def fill(owner, value: int, length: int) -> None:
    if length <= 0:
        return
    pattern = bytes([value]) * min(LOAD_CHUNK, length)
    offset = 0
    while offset < length:
        chunk = min(len(pattern), length - offset)
        owner.load(pattern[:chunk], offset)
        offset += chunk


def atomic_write(path: Path, payload: bytes, mode: int) -> None:
    temporary = path.with_name(path.name + ".tmp-" + uuid4().hex)
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(fd, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, default=2)
    parser.add_argument("--rows-per-shard", type=int, default=ROWS_PER_SHARD)
    parser.add_argument("--padding-rows", type=int, default=0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--numa", type=int, default=0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--stop-file", required=True)
    parser.add_argument("--timeout", type=int, default=1800)
    args = parser.parse_args()

    native = load_native(verbose=False)
    total_rows = args.rows_per_shard * args.shards - args.padding_rows
    plan = plan_shards(total_rows, DIM, args.shards, block=BLOCK)
    start, end = plan.row_range(args.shard)
    real_rows = end - start

    weight = native.Owner(plan.weight_bytes, args.numa, args.device)
    scale = native.Owner(plan.scale_bytes, args.numa, args.device)
    fill(weight, args.shard + 1, real_rows * DIM)
    fill(scale, args.shard + 101, real_rows * (DIM // BLOCK))

    document = {
        "shards": {
            str(args.shard): {
                "weight": base64.b64encode(weight.export_handle()).decode("ascii"),
                "scale": base64.b64encode(scale.export_handle()).decode("ascii"),
                "rows": [start, end],
            }
        },
        "plan": plan.as_dict(),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(out, json.dumps(document, sort_keys=True).encode("utf-8"), 0o600)
    print(f"READY shard={args.shard} rows=[{start},{end}) out={out}", flush=True)

    deadline = time.monotonic() + args.timeout
    while not Path(args.stop_file).exists():
        if time.monotonic() > deadline:
            print("TIMEOUT waiting for STOP", flush=True)
            break
        time.sleep(0.5)

    weight.close()
    scale.close()
    print("STOPPED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
