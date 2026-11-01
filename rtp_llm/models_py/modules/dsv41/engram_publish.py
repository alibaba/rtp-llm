# SPDX-License-Identifier: Apache-2.0
"""Allocate a member's Engram shard in fabric memory and describe it.

A cohort member owns exactly the shards the group assigns to it. The serving
process allocates its row range of every layer's weight/scale tensor, fills it
from the local checkpoint and exports one capability descriptor per shard. The
cohort exchange then hands those descriptors to the peers, so every instance
reads the others' memory over the fabric. No descriptor is written to disk and
no separate publisher process is involved.
"""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from uuid import uuid4

DEFAULT_CHUNK_BYTES = 64 << 20
# Allocations must outlive every consumer; the process keeps them here.
_KEEPERS = {}


def shard_layout(tensor, weight_shape, shard, shards, granularity):
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


def load_rows(owner, root, tensor, snapshot, start, end, columns,
              *, discard_file_cache=False, chunk_bytes=DEFAULT_CHUNK_BYTES):
    """Read exact FP8/scale bytes without creating full-file tensor mappings."""
    from rtp_llm.models_py.modules.dsv41.engram_checkpoint import (
        _cache_page_size, _discard_consumed_pages, _open_tensor_source, _read_exact,
    )

    if not 0 <= start < end <= tensor.shape[0] or columns != tensor.shape[1]:
        raise ValueError("invalid shard row range")
    if not columns <= chunk_bytes <= DEFAULT_CHUNK_BYTES:
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
            if page_size is not None and consumed - discarded >= DEFAULT_CHUNK_BYTES:
                discarded = _discard_consumed_pages(source, discarded, consumed, page_size)
        if page_size is not None:
            _discard_consumed_pages(source, discarded, consumed, page_size)
    if offset != (end - start) * columns:
        raise RuntimeError("shard row range has an unexpected byte count")
    return "sha256:" + digest.hexdigest()


def _expected_layout(layout, model_revision, hash_abi):
    from rtp_llm.models_py.modules.dsv41.engram_manifest import ExpectedLayout

    return ExpectedLayout(layers=tuple(zip(layout.layer_ids, layout.num_embeddings)),
                          head_dim=layout.head_dim, model_revision=model_revision,
                          hash_abi=hash_abi)


def publish_shard(checkpoint, model_revision, host_id, generation, shard, shards,
                  *, device=0, numa=0, layer_devices=None, layer_numas=None,
                  max_allocation_bytes=None, budget_bytes=None,
                  release_input_page_cache=False):
    """Allocate one shard here, fill it from ``checkpoint`` and describe it."""
    import torch

    from rtp_llm.models_py.modules.dsv41.engram import EngramLayout, NgramHashState
    from rtp_llm.models_py.modules.dsv41.engram_bootstrap import _load_native, hash_state_digest
    from rtp_llm.models_py.modules.dsv41.engram_capacity import (
        HARD_MAX_ALLOCATION_BYTES, HARD_MAX_TOTAL_ALLOCATION_BYTES, validate_capacity,
    )
    from rtp_llm.models_py.modules.dsv41.engram_checkpoint import (
        _cache_page_size, _discard_consumed_pages, _open_tensor_source, inspect_checkpoint,
    )

    if type(shard) is not int or type(shards) is not int or not 0 <= shard < shards:
        raise ValueError("invalid shard index")
    checkpoint = Path(checkpoint).resolve(strict=True)
    config = json.loads((checkpoint / "config.json").read_text())
    config = {**config, **config.get("text_config", {})}
    layout = EngramLayout(config)
    devices = layer_devices or {layer: device for layer in layout.layer_ids}
    numas = layer_numas or {layer: numa for layer in layout.layer_ids}
    hash_abi = hash_state_digest(NgramHashState(layout, str(checkpoint)))
    torch.cuda.set_device(device)
    native = _load_native()
    granularities = {native.host_numa_granularity(numas[layer], devices[layer])
                     for layer in layout.layer_ids}
    if len(granularities) != 1:
        raise ValueError("selected NUMA nodes must share the same allocation granularity")
    granularity = granularities.pop()
    if max_allocation_bytes is None:
        max_allocation_bytes = HARD_MAX_ALLOCATION_BYTES
    if budget_bytes is None:
        budget_bytes = HARD_MAX_TOTAL_ALLOCATION_BYTES
    validate_capacity(max_allocation_bytes, budget_bytes)
    plan = inspect_checkpoint(
        checkpoint, _expected_layout(layout, model_revision, hash_abi),
        allocation_granularity=granularity, budget_bytes=HARD_MAX_TOTAL_ALLOCATION_BYTES,
        max_allocation_bytes=HARD_MAX_ALLOCATION_BYTES)
    snapshots = {snapshot.relative_path: snapshot for snapshot in plan.files}
    page_size = _cache_page_size(release_input_page_cache)
    if page_size is not None:
        for snapshot in plan.files:
            with _open_tensor_source(checkpoint, snapshot) as (source, _):
                _discard_consumed_pages(source, 0, snapshot.stat_token[2], page_size)
    owners, tensors, local_bytes = [], {}, 0
    for layer_plan in plan.layers:
        layer_devices_value = devices[layer_plan.layer_id]
        layer_numa_value = numas[layer_plan.layer_id]
        weight_shape = next(t.shape for t in layer_plan.tensors if t.name.endswith(".weight"))
        for tensor in layer_plan.tensors:
            rows, columns = tensor.shape
            start, end, allocation_bytes = shard_layout(
                tensor, weight_shape, shard, shards, granularity)
            if allocation_bytes > max_allocation_bytes:
                raise ValueError("local shard exceeds allocation limit")
            local_bytes += allocation_bytes
            if local_bytes > budget_bytes:
                raise ValueError("local shards exceed host budget")
            owner = native.Owner(allocation_bytes, layer_numa_value, layer_devices_value)
            owners.append(owner)
            digest = load_rows(owner, checkpoint, tensor, snapshots[tensor.relative_path],
                               start, end, columns,
                               discard_file_cache=release_input_page_cache)
            tensors[tensor.name] = dict(
                shape=list(tensor.shape), allocation_id=str(uuid4()),
                handle=base64.b64encode(owner.export_handle()).decode("ascii"),
                row_begin=start, row_end=end, logical_bytes=(end - start) * columns,
                allocation_bytes=allocation_bytes, allocation_granularity=granularity,
                sha256=digest)
    document = dict(schema_version=2, shard=shard, shards=shards, generation=generation,
                    model_revision=model_revision, hash_abi=hash_abi, host_id=host_id,
                    owner_id=host_id, owner_incarnation=str(uuid4()), tensors=tensors)
    _KEEPERS.setdefault(generation, []).extend(owners)
    return document


def release(generation):
    """Drop the allocations of one generation (process shutdown or restart)."""
    owners = _KEEPERS.pop(generation, [])
    failures = []
    for owner in reversed(owners):
        try:
            owner.close()
        except Exception as error:  # native close is best effort at exit
            failures.append(error)
    return failures