#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""GPU check for the sharded fabric mapping (run on a GB200 dev node).

Builds the development native bridge, publishes the local shards, maps every
shard through :class:`ShardedLayerMapping`, and verifies that global row
indexing lands in the right shard.  Shards listed in ``--remote-handles`` are
owned by another node and are only imported here; that is the cross-node VA
stitching path.

    python3 check_shard_mapping.py            # 1 shard
    python3 check_shard_mapping.py --shards 2 # 2 shards, all local
    python3 check_shard_mapping.py --shards 2 --remote-handles shard1.json
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from load_native import load_native  # noqa: E402

from rtp_llm.models_py.modules.dsv41.engram_shard import plan_shards  # noqa: E402
from rtp_llm.models_py.modules.dsv41.engram_shard_storage import (  # noqa: E402
    ShardedLayerMapping,
)

DIM = 256
BLOCK = 32
ROWS_PER_SHARD = 262144  # one alignment unit: 64 MiB of weights
LOAD_CHUNK = 64 * 1024 * 1024


def fill(owner, value: int, length: int) -> None:
    """Write ``length`` bytes of ``value`` without exceeding the load ceiling."""
    if length <= 0:
        return
    pattern = bytes([value]) * min(LOAD_CHUNK, length)
    offset = 0
    while offset < length:
        chunk = min(len(pattern), length - offset)
        owner.load(pattern[:chunk], offset)
        offset += chunk


def load_remote(path: str) -> dict[int, tuple[bytes, bytes]]:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    remote = {}
    for shard, entry in document["shards"].items():
        remote[int(shard)] = (
            base64.b64decode(entry["weight"], validate=True),
            base64.b64decode(entry["scale"], validate=True),
        )
    return remote


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--numa", type=int, default=0)
    parser.add_argument("--remote-handles", default="")
    parser.add_argument("--rows-per-shard", type=int, default=ROWS_PER_SHARD)
    parser.add_argument("--padding-rows", type=int, default=0)
    args = parser.parse_args()

    native = load_native(verbose=False)
    total_rows = args.rows_per_shard * args.shards - args.padding_rows
    plan = plan_shards(total_rows, DIM, args.shards, block=BLOCK)
    print(f"plan: {plan.as_dict()}")

    remote = load_remote(args.remote_handles) if args.remote_handles else {}
    for shard in remote:
        if not 0 <= shard < plan.shard_count:
            raise SystemExit(f"remote shard {shard} outside plan")

    owners = []
    weight_handles: list[bytes | None] = [None] * plan.shard_count
    scale_handles: list[bytes | None] = [None] * plan.shard_count
    for shard in range(plan.shard_count):
        if shard in remote:
            weight_handles[shard], scale_handles[shard] = remote[shard]
            print(f"shard {shard}: remote")
            continue
        start, end = plan.row_range(shard)
        real_rows = end - start
        # Weight and scale are separate allocations: a fabric handle can only
        # be mapped at offset zero, so the two regions cannot share one owner.
        weight = native.Owner(plan.weight_bytes, args.numa, args.device)
        scale = native.Owner(plan.scale_bytes, args.numa, args.device)
        fill(weight, shard + 1, real_rows * DIM)
        fill(scale, shard + 101, real_rows * (DIM // BLOCK))
        owners.extend((weight, scale))
        weight_handles[shard] = weight.export_handle()
        scale_handles[shard] = scale.export_handle()
        print(f"shard {shard}: local rows=[{start},{end})")

    mapping = ShardedLayerMapping(plan, weight_handles, scale_handles, args.device).attach()
    weight_view = mapping.weight_view()
    scale_view = mapping.scale_view()
    print(f"weight view: {tuple(weight_view.shape)} ptr=0x{weight_view.data_ptr():x}")
    print(f"scale  view: {tuple(scale_view.shape)} ptr=0x{scale_view.data_ptr():x}")

    ok = True
    for shard in range(plan.shard_count):
        start, end = plan.row_range(shard)
        probes = [start, (start + end) // 2, end - 1]
        for row in probes:
            got_weight = int(weight_view[row, 0].item())
            got_scale = int(scale_view[row, 0].item())
            want_weight = shard + 1
            want_scale = shard + 101
            if got_weight != want_weight or got_scale != want_scale:
                ok = False
                print(
                    f"MISMATCH shard={shard} row={row} "
                    f"weight={got_weight} want={want_weight} "
                    f"scale={got_scale} want={want_scale}"
                )
        # Check the last column too, so a wrong row stride cannot pass.
        got = int(weight_view[end - 1, DIM - 1].item())
        if got != shard + 1:
            ok = False
            print(f"MISMATCH shard={shard} row={end - 1} col={DIM - 1} weight={got} want={shard + 1}")

    # Padding rows are never addressed by a lookup, but they must stay
    # readable and zero because owners only load real rows.
    for row in range(total_rows, plan.padded_rows):
        if int(weight_view[row, 0].item()) != 0 or int(scale_view[row, 0].item()) != 0:
            ok = False
            print(f"MISMATCH padding row={row} not zero")
            break
    print(f"padding rows: {plan.padded_rows - total_rows}")

    del weight_view, scale_view
    mapping.close()
    for owner in owners:
        owner.close()
    print("RESULT", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
