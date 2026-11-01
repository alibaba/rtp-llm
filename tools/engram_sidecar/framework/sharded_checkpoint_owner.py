#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Publish one N-way Engram shard from the real checkpoint and hold it.

The owner loads only its row range of each layer's weight/scale tensor into
separate fabric allocations (a fabric handle can only be mapped at offset 0),
writes a private shard descriptor, and keeps the allocations alive until the
coordinator creates ``STOP``. The coordinator merges the descriptors from all
owners into one manifest/capability bundle.

    python3 sharded_checkpoint_owner.py --checkpoint /path/ckpt \
        --model-revision rev --host-id <physical-SN> --generation <uuid> --shard 0 --shards 2 \
        --publication /run/shard0 --device 0 --numa 0

Never log or copy the private descriptor outside the authenticated transfer.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import signal
import sys
import time
from pathlib import Path
from uuid import uuid4

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

LOAD_CHUNK = 64 << 20


def emit(status, **fields):
    try:
        print(json.dumps(dict(status=status, **fields), sort_keys=True), flush=True)
    except (OSError, ValueError):
        pass


def private_write(path, payload):
    temporary = path.with_name(path.name + ".tmp-" + uuid4().hex)
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_rows(owner, root, tensor, snapshot, start, end, columns,
              *, discard_file_cache=False, chunk_bytes=8 << 20):
    """Read exact FP8/scale bytes without creating full-file tensor mappings."""
    from rtp_llm.models_py.modules.dsv41.engram_checkpoint import (
        _cache_page_size, _discard_consumed_pages, _open_tensor_source, _read_exact,
    )

    if not 0 <= start < end <= tensor.shape[0] or columns != tensor.shape[1]:
        raise ValueError("invalid shard row range")
    if not columns <= chunk_bytes <= LOAD_CHUNK:
        raise ValueError("chunk must hold a row and fit the native load ceiling")
    page_size = _cache_page_size(discard_file_cache)
    rows_per_chunk = chunk_bytes // columns
    digest = hashlib.sha256()
    offset = 0
    with _open_tensor_source(root, snapshot) as (source, data_start):
        consumed = data_start + tensor.data_range[0] + start * columns
        expected_end = data_start + tensor.data_range[0] + end * columns
        if expected_end > data_start + tensor.data_range[1]:
            raise ValueError("shard exceeds source tensor extent")
        source.seek(consumed)
        discarded = consumed
        for row in range(start, end, rows_per_chunk):
            length = min(rows_per_chunk, end - row) * columns
            payload = _read_exact(source, length)
            digest.update(payload)
            owner.load(payload, offset)
            offset += length
            consumed += length
            del payload
            if page_size is not None and consumed - discarded >= LOAD_CHUNK:
                discarded = _discard_consumed_pages(source, discarded, consumed, page_size)
        if page_size is not None:
            _discard_consumed_pages(source, discarded, consumed, page_size)
    expected = (end - start) * columns
    if offset != expected:
        raise RuntimeError("shard row range has an unexpected byte count")
    return "sha256:" + digest.hexdigest()


def tensor_shard_layout(tensor, weight_shape, shard, shards, granularity):
    """Use the weight row partition for both tensors; only byte widths differ."""
    from rtp_llm.models_py.modules.dsv41.engram_shard import plan_shards

    rows, dim = weight_shape
    plan = plan_shards(rows, dim, shards, granularity=granularity)
    role = tensor.name.rsplit(".", 1)[-1]
    if role not in ("weight", "scale"):
        raise ValueError("unsupported Engram tensor role")
    width = dim if role == "weight" else dim // plan.block
    if tuple(tensor.shape) != (rows, width):
        raise ValueError("weight and scale shapes must share the same row layout")
    start, end = plan.row_range(shard)
    allocation_bytes = plan.weight_bytes if role == "weight" else plan.scale_bytes
    return start, end, allocation_bytes


def layer_numa_placement(raw, layers, default, label="NUMA"):
    """Complete explicit local placement; no automatic fallback after allocation failure."""
    if raw is None:
        return {layer: default for layer in layers}
    def unique(pairs):
        if len(pairs) != len({key for key, _ in pairs}):
            raise ValueError(f"duplicate layer {label} key")
        return dict(pairs)
    values = json.loads(raw, object_pairs_hook=unique)
    if not isinstance(values, dict) or set(values) != {str(layer) for layer in layers}:
        raise ValueError(f"layer {label} map must cover exactly the model's Engram layers")
    if any(type(node) is not int or node < 0 for node in values.values()):
        raise ValueError(f"layer {label} IDs must be nonnegative local integers")
    return {layer: values[str(layer)] for layer in layers}


def validate_local_budget(plan, shard, shards, granularity, max_allocation, budget):
    from rtp_llm.models_py.modules.dsv41.engram_capacity import validate_capacity
    validate_capacity(max_allocation, budget)
    total = 0
    for layer in plan.layers:
        shape = next(t.shape for t in layer.tensors if t.name.endswith(".weight"))
        for tensor in layer.tensors:
            _, _, size = tensor_shard_layout(tensor, shape, shard, shards, granularity)
            if size > max_allocation:
                raise ValueError("local shard exceeds allocation limit")
            total += size
    if total > budget:
        raise ValueError("local shards exceed host budget")
    return total


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--host-id", required=True, help="Trusted physical host identity (SN), shared with topology")
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--publication", type=Path, required=True)
    parser.add_argument("--generation", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--numa", type=int, default=0)
    parser.add_argument("--layer-numa-map",
                        help='Explicit local placement for every Engram layer, e.g. {"1":0,"14":1}; '
                             'weight and scale stay together. Default uses --numa for all layers.')
    parser.add_argument("--layer-device-map",
                        help='Explicit CUDA-visible device ordinal for every layer; '
                             'pair each selected HOST_NUMA with its local GPU. Default uses --device.')
    parser.add_argument("--max-allocation-bytes", type=int, default=128 << 30)
    parser.add_argument("--budget-bytes", type=int, default=256 << 30)
    parser.add_argument("--report-timeout", type=float, default=3600)
    parser.add_argument("--release-input-page-cache", action="store_true",
                        help="advise eviction of input file cache after each tensor; use private checkpoint copies")
    args = parser.parse_args(argv)

    if (not args.checkpoint.is_absolute() or not args.publication.is_absolute()
            or args.device < 0 or args.numa < 0 or not 2 <= args.shards <= 64
            or not args.host_id.strip() or len(args.host_id) > 256
            or not 0 < args.report_timeout < float("inf")
            or not 0 <= args.shard < args.shards):
        parser.error("absolute paths, valid shard index and nonnegative ordinals required")
    args.publication.mkdir(mode=0o700, parents=True, exist_ok=False)

    import torch
    from rtp_llm.models_py.modules.dsv41.engram import EngramLayout, NgramHashState
    from rtp_llm.models_py.modules.dsv41.engram_bootstrap import _load_native, hash_state_digest
    from rtp_llm.models_py.modules.dsv41.engram_checkpoint import (
        _cache_page_size, _discard_consumed_pages, _open_tensor_source,
        _same_snapshot, inspect_checkpoint,
    )
    from rtp_llm.models_py.modules.dsv41.engram_manifest import ExpectedLayout
    checkpoint = args.checkpoint
    config = json.loads((checkpoint / "config.json").read_text())
    config = {**config, **config.get("text_config", {})}
    layout = EngramLayout(config)
    numa_by_layer = layer_numa_placement(args.layer_numa_map, layout.layer_ids, args.numa)
    device_by_layer = layer_numa_placement(args.layer_device_map, layout.layer_ids, args.device, "device")
    actual_hash_abi = hash_state_digest(NgramHashState(layout, str(checkpoint)))
    expected = ExpectedLayout(
        layers=tuple(zip(layout.layer_ids, layout.num_embeddings)),
        head_dim=layout.head_dim, model_revision=args.model_revision,
        hash_abi=actual_hash_abi)

    torch.cuda.set_device(args.device)
    native = _load_native()
    granularities = {native.host_numa_granularity(numa_by_layer[layer], device_by_layer[layer])
                     for layer in layout.layer_ids}
    if len(granularities) != 1:
        raise ValueError("selected NUMA nodes must share the same allocation granularity")
    granularity = granularities.pop()
    from rtp_llm.models_py.modules.dsv41.engram_capacity import (
        validate_capacity, HARD_MAX_ALLOCATION_BYTES, HARD_MAX_TOTAL_ALLOCATION_BYTES)
    validate_capacity(args.max_allocation_bytes, args.budget_bytes)
    # Inspect the complete checkpoint metadata, but enforce the caller's memory
    # budget against only this host's padded weight/scale allocations.
    plan = inspect_checkpoint(
        checkpoint, expected, allocation_granularity=granularity,
        budget_bytes=HARD_MAX_TOTAL_ALLOCATION_BYTES, max_allocation_bytes=HARD_MAX_ALLOCATION_BYTES)
    validate_local_budget(plan, args.shard, args.shards, granularity,
                          args.max_allocation_bytes, args.budget_bytes)
    snapshots = {snapshot.relative_path: snapshot for snapshot in plan.files}
    page_size = _cache_page_size(args.release_input_page_cache)
    if page_size is not None:
        for snapshot in plan.files:
            with _open_tensor_source(checkpoint, snapshot) as (source, _):
                _discard_consumed_pages(source, 0, snapshot.stat_token[2], page_size)

    owners = []
    tensors = {}
    published = False

    def interrupt(signum, frame):
        if published:
            emit("SIGNAL_WHILE_PUBLISHED", signal=signum, action="retained_until_STOP")
        else:
            raise SystemExit(128 + signum)

    previous = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        for layer_plan in plan.layers:
            numa = numa_by_layer[layer_plan.layer_id]
            device = device_by_layer[layer_plan.layer_id]
            weight_shape = next(t.shape for t in layer_plan.tensors if t.name.endswith(".weight"))
            for tensor in layer_plan.tensors:
                rows, columns = tensor.shape
                start, end, local_bytes = tensor_shard_layout(
                    tensor, weight_shape, args.shard, args.shards, granularity)
                emit("ALLOCATING", tensor=tensor.name, shard=args.shard,
                     allocation_bytes=local_bytes, device=device, numa=numa)
                owner = native.Owner(local_bytes, numa, device)
                owners.append(owner)
                digest = load_rows(owner, checkpoint, tensor, snapshots[tensor.relative_path],
                                   start, end, columns,
                                   discard_file_cache=args.release_input_page_cache)
                emit("LOADED", tensor=tensor.name, shard=args.shard,
                     logical_bytes=(end - start) * columns)
                tensors[tensor.name] = dict(
                    shape=list(tensor.shape), allocation_id=str(uuid4()),
                    handle=base64.b64encode(owner.export_handle()).decode("ascii"),
                    row_begin=start, row_end=end,
                    logical_bytes=(end - start) * columns,
                    allocation_bytes=local_bytes,
                    allocation_granularity=granularity,
                    sha256=digest)
        _same_snapshot(checkpoint, plan.index, index=True)
        for snapshot in plan.files:
            _same_snapshot(checkpoint, snapshot)
        document = dict(schema_version=2, shard=args.shard, shards=args.shards,
                        generation=args.generation, model_revision=args.model_revision,
                        hash_abi=actual_hash_abi, host_id=args.host_id, owner_id=args.host_id,
                        owner_incarnation=str(uuid4()), tensors=tensors)
        # Protect backing before the first externally visible capability. Once
        # published, only the coordinator's all-consumers-exited STOP may retire.
        published = True
        private_write(args.publication / "shard.json",
                      json.dumps(document, sort_keys=True).encode("utf-8"))
        emit("READY", shard=args.shard, shards=args.shards, generation=args.generation,
             tensors=len(tensors), device_by_layer=device_by_layer, numa_by_layer=numa_by_layer,
             counters=native.resource_counters())
        deadline = time.monotonic() + args.report_timeout
        while not (args.publication / "STOP").is_file():
            if time.monotonic() > deadline:
                emit("WAITING_FOR_STOP", action="backing_retained")
                deadline = float("inf")
            time.sleep(1)
    finally:
        if published:
            while not (args.publication / "STOP").is_file():
                time.sleep(1)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        for owner in reversed(owners):
            try:
                owner.close()
            except BaseException:
                pass
        emit("CLOSED", counters=native.resource_counters())
        (args.publication / "shard.json").unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
