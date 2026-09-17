"""Actual CP8 collectives versus complete-page attention on the same CUDA inputs.

Run on one authorized CUDA13 Blackwell eight-GPU host. Direct invocation starts
eight workers; torchrun may also launch this file with eight ranks explicitly.
This is a component integration test, not model or deployment acceptance.
"""

import json
import os
import subprocess
import sys
from dataclasses import replace
from datetime import timedelta
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from rtp_llm.models_py.distributed import collective_torch
from rtp_llm.models_py.distributed.collective_torch import Group
from rtp_llm.models_py.modules.dsv4.cp import build_cp_context_for_forward
from rtp_llm.models_py.modules.dsv41 import cp as cp_attention
from rtp_llm.models_py.modules.dsv41.attention import V41Attention, V41AttentionCache
from rtp_llm.models_py.modules.dsv41.cache_layout import (
    PAIR_OWNERS,
    CacheLayout,
    CacheRegion,
    RegionSlot,
    layer_sources,
)
from rtp_llm.models_py.modules.dsv41.ced import ReplayConfig, ReplayMode
from rtp_llm.models_py.modules.dsv41.compressor import OwnerCompressor, PairCarry
from rtp_llm.models_py.modules.dsv41.cp import V41CPAttentionContext, begin_cp_request
from torch import nn

_LAYERS = (0, 2, 3, 8, 9, 14, 15, 20, 21, 24, 25, 28, 29, 32, 33, 36, 39)
_FLAGS = (
    "DSV41_CP_COMPACT_MODEL_ROWS",
    "DSV41_CP_COMPACT_QUERY_ROWS",
    "DSV41_CP_SINGLE_OWNER_TRANSPORT",
)


def _metadata(lengths, starts, rank, device):
    chunks = [2 * ((length + 15) // 16) for length in lengths]
    padded = [8 * length for length in chunks]
    mask = torch.zeros(sum(padded), dtype=torch.int32)
    restore = torch.empty(sum(padded), dtype=torch.int32)
    local_first = global_first = 0
    for length, chunk, size in zip(lengths, chunks, padded):
        mask[global_first : global_first + length] = 1
        half = chunk // 2
        for peer in range(8):
            positions = list(range(peer * half, (peer + 1) * half))
            positions += list(range(size - (peer + 1) * half, size - peer * half))
            for local, position in enumerate(positions):
                restore[global_first + position] = (
                    peer * sum(chunks) + local_first + local
                )
        global_first += size
        local_first += chunk
    info = SimpleNamespace(
        prefill_actual_input_lengths_cpu=torch.tensor(lengths, dtype=torch.int32),
        prefill_cp_chunk_lengths=torch.tensor(chunks, dtype=torch.int32),
        prefill_qkv_padding_mask=mask.to(device),
        prefill_qkv_restore_indice=restore.to(device),
    )
    prefixes = torch.tensor(starts, dtype=torch.int32)
    return build_cp_context_for_forward(
        info,
        8,
        rank,
        sum(chunks),
        device,
        prefix_lengths=prefixes.to(device),
        prefix_lengths_host=prefixes,
        chunk_lengths_device=torch.tensor(chunks, dtype=torch.int32, device=device),
        kv_cache_sharded=True,
    )


def _framework_pages(layout, rank, device):
    pools, tables, pair_pools, pair_tables = {}, {}, {}, {}
    for page in layout.pages:
        pools[page.slot] = torch.zeros(
            (40, page.prefill_shard_bytes), dtype=torch.uint8, device=device
        )
        tables[page.slot] = torch.tensor(
            [[rank + 2, rank + 19]], dtype=torch.int32, device=device
        )
    for owner in PAIR_OWNERS:
        snapshots = next(
            state.snapshots
            for state in layout.pair_states
            if state.owner_layer == owner
        )
        shard_bytes = ((snapshots * 4112 + 511) // 512) * 512 // layout.cp_size
        pair_pools[owner] = torch.zeros(
            (40, shard_bytes), dtype=torch.uint8, device=device
        )
        pair_tables[owner] = torch.tensor(
            [[rank + 3, rank + 20]], dtype=torch.int32, device=device
        )
    return dict(
        pools=pools, tables=tables, pair_pools=pair_pools, pair_tables=pair_tables
    )


def _models(layout, device):
    def linear(inputs, outputs):
        module = nn.Linear(
            inputs, outputs, bias=False, dtype=torch.bfloat16, device=device
        )
        module.weight.data.zero_()
        return module

    wq_a, wq_b = linear(5120, 1280), linear(1280, 64 * 512)
    wkv, wo_b = linear(5120, 512), linear(8192, 5120)
    index_wq_b = linear(1280, 32 * 128)
    channels = torch.arange(512, device=device)
    wkv.weight.data[channels, channels] = 1
    wq_a.weight.data[
        torch.arange(1280, device=device), torch.arange(1280, device=device)
    ] = 1
    wo_b.weight.data[
        torch.arange(5120, device=device),
        (torch.arange(5120, device=device) % 8) * 1024,
    ] = 1
    wo_a = torch.zeros((8, 1024, 4096), dtype=torch.bfloat16, device=device)
    wo_a[:, 0, 0] = 1
    index_wk = torch.zeros((128, 512), dtype=torch.bfloat16, device=device)
    index_wk[torch.arange(128, device=device), torch.arange(128, device=device)] = 1
    models = {}
    for layer in _LAYERS:
        source = layer_sources(layer)
        compressor = None
        if source.writes_global:
            compressor = OwnerCompressor(
                layer,
                wkv.weight.detach().to(
                    torch.float32 if source.ratio == 2 else torch.bfloat16
                ),
                torch.ones(512, dtype=torch.bfloat16, device=device),
                (
                    torch.zeros((512, 5120), dtype=torch.float32, device=device)
                    if source.ratio == 2
                    else None
                ),
                layout=layout,
            )
        models[layer] = V41Attention(
            layer,
            wq_a=wq_a,
            wq_b=wq_b,
            wkv=wkv,
            wo_b=wo_b,
            wo_a=wo_a,
            q_norm=torch.ones(1280, dtype=torch.bfloat16, device=device),
            kv_norm=torch.ones(512, dtype=torch.bfloat16, device=device),
            sinks=torch.zeros(64, dtype=torch.float32, device=device),
            compressor=compressor,
            index_wq_b=index_wq_b if source.scores_queries else None,
            index_weights=(
                torch.zeros((32, 5120), dtype=torch.bfloat16, device=device)
                if source.scores_queries
                else None
            ),
            index_wk=index_wk if source.writes_index_k else None,
            index_norm=(
                torch.ones(128, dtype=torch.bfloat16, device=device)
                if source.writes_index_k
                else None
            ),
        )
    return models


def _hidden(first, last, request, device):
    token = torch.arange(first, last, device=device, dtype=torch.float32)[:, None]
    channel = torch.arange(5120, device=device, dtype=torch.float32)[None, :]
    return (0.75 + torch.sin(token * 0.17 + channel * 0.07 + request)).bfloat16()


def _equal(actual, expected, description):
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, msg=description)


@lru_cache(maxsize=8192)
def _encoded_bytes(position, salt):
    row = bytearray(
        (position * 73 + column * 29 + salt * 11) % 256 for column in range(528)
    )
    row[:8] = position.to_bytes(8, "little")
    row[8:16] = salt.to_bytes(8, "little")
    return bytes(row)


def _expected_swa_pages(positions, spec, replay_floor, salt):
    storage = bytearray((len(positions) + 1) * spec.page_stride_bytes)
    for query, position in enumerate(positions):
        for token in range(max(0, replay_floor, position - 127), position + 1):
            first = (query + 1) * spec.page_stride_bytes + (token % spec.entries) * 528
            storage[first : first + 528] = _encoded_bytes(token, salt)
    return torch.frombuffer(storage, dtype=torch.uint8).view(-1, spec.page_stride_bytes)


def _byte_context(cp, layout, pools, tables, request, replay_floor=0):
    identity = ReplayConfig(ReplayMode.BOUNDED).cache_identity("swa-bytes", layout)
    cache = V41AttentionCache("swa-bytes", identity, layout, 4096, {}, {})
    context = V41CPAttentionContext(
        cache, cp, request, pools, tables, {}, {}, 0, replay_floor
    )
    cache.swa_ends[0] = context.start
    return context


@torch.inference_mode()
def _swa_read_batches(rank, device):
    observations = []
    cases = (
        ("origin", (3, 1), (0, 0), 0, 0),
        ("batched_empty", (3, 1), (0, 0), 1, 0),
        ("ring_wrap", (257, 33), (125, 261), 0, 0),
        ("batched_restore", (257, 33), (125, 261), 1, 0),
        ("wide_batch", (541,), (1024,), 0, 0),
        ("replay_floor", (145,), (1024,), 0, 1010),
        ("no_history", (17,), (1024,), 0, 1024),
        ("single_owner_slice", (4096,), (0,), 0, 0),
    )
    for speculative in (0, 5):
        layout = CacheLayout(
            cp_size=8, speculative_tokens=speculative, draft_enabled=bool(speculative)
        )
        slot = RegionSlot(CacheRegion.SWA, 0)
        spec = next(page for page in layout.pages if page.slot == slot)
        for case_id, (label, lengths, starts, request, floor) in enumerate(cases):
            salt = case_id + 100 * speculative
            cp = _metadata(lengths, starts, rank, device)
            pools = {
                slot: torch.full(
                    (40, spec.prefill_shard_bytes),
                    211,
                    dtype=torch.uint8,
                    device=device,
                )
            }
            tables = {
                slot: torch.tensor(
                    [[rank + 2, rank + 10, rank + 18, rank + 26]],
                    dtype=torch.int32,
                    device=device,
                )
            }
            context = _byte_context(cp, layout, pools, tables, request, floor)
            half = (lengths[request] + 15) // 16
            offsets = list(range(rank * half, (rank + 1) * half))
            offsets += list(range((15 - rank) * half, (16 - rank) * half))
            positions = [
                starts[request] + offset if offset < lengths[request] else -1
                for offset in offsets
            ]
            if label == "single_owner_slice":
                # The byte fixture supplies only the completed source boundary.
                context.cache.owners = {
                    owner: SimpleNamespace(materialized_end=1024, pair=None)
                    for owner in PAIR_OWNERS
                }
                context = context.encoder_slice(1024, 1153, epoch=1)
                context.cache.swa_ends[0] = context.start
                positions = list(range(1024, 1153)) if rank == 4 else [-1] * 129
            _equal(
                context._rank_positions[rank].cpu(),
                torch.tensor(positions),
                f"{label} independent query ownership",
            )
            real_rows = [
                index for index, position in enumerate(positions) if position >= 0
            ]
            row_ids = torch.tensor(real_rows, dtype=torch.int64, device=device)
            poisoned = torch.arange(
                context.query_rows * 12, dtype=torch.float32, device=device
            ).view(context.query_rows, 3, 4)
            poisoned[torch.tensor(positions, device=device) < 0] = torch.nan
            packed = context.pack_model_rows(poisoned)
            _equal(
                packed, poisoned.index_select(0, row_ids), f"{label} real model rows"
            )
            restored = context.unpack_model_rows(packed)
            _equal(restored[row_ids], packed, f"{label} canonical model output order")
            assert context.model_query_rows == len(real_rows)
            assert context.attention_query_rows == len(real_rows)
            assert bool(torch.isfinite(restored).all())
            assert bool((restored[~context.valid] == 0).all())
            restored_bytes = bytearray(spec.page_stride_bytes)
            if context.start and floor < context.start:
                restored_bytes[:] = bytes([219]) * spec.page_stride_bytes
                for token in range(
                    max(floor, context.start - spec.entries), context.start
                ):
                    first = (token % spec.entries) * 528
                    restored_bytes[first : first + 528] = _encoded_bytes(token, salt)
                previous = (context.start - 1) // layout.reuse_unit
                page = int(tables[slot][0, previous])
                first, last = spec.swa_byte_slice(rank)
                pools[slot][page].copy_(
                    torch.frombuffer(restored_bytes, dtype=torch.uint8)[first:last].to(
                        device
                    )
                )
                host = pools[slot][page].cpu().clone()
                pools[slot][39 - rank].copy_(host)
                pools[slot][page].fill_(197)
                tables[slot][0, previous] = 39 - rank
            initial = context.restore_swa(0)
            _equal(
                initial.pages.data[1].cpu(),
                torch.frombuffer(restored_bytes, dtype=torch.uint8),
                f"{label} physical-page restore including padding",
            )
            _equal(
                initial.pages.data[0].cpu(),
                torch.zeros(spec.page_stride_bytes, dtype=torch.uint8),
                "restore null page",
            )
            encoded = (
                torch.frombuffer(
                    bytearray().join(
                        (
                            _encoded_bytes(position, salt)
                            if position >= 0
                            else bytes([247]) * 528
                        )
                        for position in positions
                    ),
                    dtype=torch.uint8,
                )
                .view(-1, 528)
                .to(device)
            )
            for batch in (1, 4, 32):
                max_peak_bytes = 0
                before = context.gather_count
                for first in range(0, len(positions), batch):
                    last = min(first + batch, len(positions))
                    torch.cuda.synchronize()
                    allocated = torch.cuda.memory_allocated(device)
                    torch.cuda.reset_peak_memory_stats(device)
                    binding = context.swa_queries(0, first, last, initial, encoded)
                    torch.cuda.synchronize()
                    peak_bytes = torch.cuda.max_memory_allocated(device) - allocated
                    max_peak_bytes = max(max_peak_bytes, peak_bytes)
                    assert peak_bytes + initial.pages.data.numel() <= 64 * 1024 * 1024
                    binding.validate(device)
                    queries = positions[first:last]
                    _equal(
                        binding.pages.data.cpu(),
                        _expected_swa_pages(queries, spec, floor, salt),
                        f"{label} gamma{speculative} batch{batch} exact ring bytes",
                    )
                    _equal(
                        binding.page_ids.cpu(),
                        torch.arange(1, len(queries) + 1, dtype=torch.int32),
                        "query page IDs",
                    )
                    _equal(
                        binding.valid_starts.cpu(),
                        torch.tensor(
                            [
                                max(floor, pos - 127) if pos >= 0 else 0
                                for pos in queries
                            ],
                            dtype=torch.int32,
                        ),
                        "query valid starts",
                    )
                    _equal(
                        binding.valid_ends.cpu(),
                        torch.tensor(
                            [max(0, pos + 1) for pos in queries], dtype=torch.int32
                        ),
                        "query valid ends",
                    )
                    del binding
                    if batch == 32:
                        compact = context.swa_queries(
                            0,
                            first,
                            last,
                            initial,
                            encoded,
                            query_rows=context.query_row_indices(first, last),
                            query_owner=context.single_query_owner,
                        )
                        real_queries = [
                            position for position in queries if position >= 0
                        ]
                        if not real_queries:
                            assert compact is None
                        else:
                            assert compact.validate(device) == len(real_queries)
                            _equal(
                                compact.pages.data.cpu(),
                                _expected_swa_pages(real_queries, spec, floor, salt),
                                f"{label} compact queries preserve exact ring bytes",
                            )
                            _equal(
                                compact.valid_ends.cpu(),
                                torch.tensor(
                                    [position + 1 for position in real_queries],
                                    dtype=torch.int32,
                                ),
                                "compact query order",
                            )
                        del compact
                assert context.gather_count - before == (
                    (len(positions) + batch - 1) // batch
                ) * (2 if batch == 32 else 1)
                assert context.max_gather_live_bytes <= 64 * 1024 * 1024
                observations.append(
                    dict(
                        case=label,
                        speculative_tokens=speculative,
                        read_queries=batch,
                        local_rows=len(positions),
                        valid_rows=sum(pos >= 0 for pos in positions),
                        max_receive_bytes=context.max_receive_bytes,
                        max_gather_live_bytes=context.max_gather_live_bytes,
                        max_measured_read_bytes=max_peak_bytes,
                    )
                )
            _equal(
                initial.pages.data[1].cpu(),
                torch.frombuffer(restored_bytes, dtype=torch.uint8),
                "query reads preserve restored bytes",
            )
            if label == "single_owner_slice":
                for batch in (64, 128):
                    for first in range(0, context.query_rows, batch):
                        last = min(first + batch, context.query_rows)
                        compact = context.swa_queries(
                            0, first, last, initial, encoded,
                            query_rows=context.query_row_indices(first, last),
                            query_owner=context.single_query_owner,
                        )
                        if rank == 4:
                            _equal(
                                compact.pages.data.cpu(),
                                _expected_swa_pages(positions[first:last], spec, floor, salt),
                                "large compact SWA batch preserves canonical bytes",
                            )
                        else:
                            assert compact is None
                        del compact
                # The live-byte guard budget is 1 GiB (512-row production
                # reads account ~830 MiB); no single in-fixture query reaches
                # it, so trip the guard directly instead of through a query.
                try:
                    context._record_gather(0, cp_attention.MAX_GATHER_BYTES + 1)
                except ValueError as error:
                    assert "1 GiB" in str(error), str(error)
                else:
                    raise AssertionError(f"{label} accepted 1 GiB")
            context.publish_swa(0, initial, encoded)
            for token in range(
                max(context.start, context.end - spec.entries), context.end
            ):
                first = (token % spec.entries) * 528
                restored_bytes[first : first + 528] = _encoded_bytes(token, salt)
            cp = _metadata((1,), (context.end,), rank, device)
            resumed = _byte_context(cp, layout, pools, tables, 0, floor)
            actual = resumed.restore_swa(0)
            _equal(
                actual.pages.data[1].cpu(),
                torch.frombuffer(restored_bytes, dtype=torch.uint8),
                "published ring survives next-request restore",
            )
            del actual, resumed, initial, encoded, context
    _encoded_bytes.cache_clear()
    return observations


@torch.inference_mode()
def _swa_index_cache_reads(rank, device):
    """DSV41_SWA_INDEX_CACHE reuse across layers must be bitwise identical."""
    layout = CacheLayout(cp_size=8, speculative_tokens=0, draft_enabled=False)
    slot = RegionSlot(CacheRegion.SWA, 0)
    spec = next(page for page in layout.pages if page.slot == slot)
    lengths, starts, floor, salt = (1000,), (1024,), 1010, 5
    cp = _metadata(lengths, starts, rank, device)
    pools = {
        slot: torch.full(
            (40, spec.prefill_shard_bytes), 211, dtype=torch.uint8, device=device
        )
    }
    tables = {
        slot: torch.tensor(
            [[rank + 2, rank + 10, rank + 18, rank + 26]],
            dtype=torch.int32,
            device=device,
        )
    }
    context = _byte_context(cp, layout, pools, tables, 0, floor)
    half = (lengths[0] + 15) // 16
    offsets = list(range(rank * half, (rank + 1) * half))
    offsets += list(range((15 - rank) * half, (16 - rank) * half))
    positions = [
        starts[0] + offset if offset < lengths[0] else -1 for offset in offsets
    ]
    restored_bytes = bytearray(spec.page_stride_bytes)
    for token in range(max(floor, context.start - spec.entries), context.start):
        first = (token % spec.entries) * 528
        restored_bytes[first : first + 528] = _encoded_bytes(token, salt)
    previous = (context.start - 1) // layout.reuse_unit
    page = int(tables[slot][0, previous])
    shard_first, shard_last = spec.swa_byte_slice(rank)
    pools[slot][page].copy_(
        torch.frombuffer(restored_bytes, dtype=torch.uint8)[shard_first:shard_last].to(
            device
        )
    )
    initial = context.restore_swa(0)
    encoded = (
        torch.frombuffer(
            bytearray().join(
                (
                    _encoded_bytes(position, salt)
                    if position >= 0
                    else bytes([247]) * 528
                )
                for position in positions
            ),
            dtype=torch.uint8,
        )
        .view(-1, 528)
        .to(device)
    )
    empty = torch.tensor([], dtype=torch.int64, device=device)
    # The cache is the default: with the env removed, the first read builds and
    # caches the plan and its output matches the ground truth bitwise.
    with patch.dict(os.environ):
        os.environ.pop("DSV41_SWA_INDEX_CACHE", None)
        default_result = context.swa_queries(0, 0, 32, initial, encoded)
        assert len(context._swa_index_plans) == 1
    _equal(
        default_result.pages.data.cpu(),
        _expected_swa_pages(positions[:32], spec, floor, salt),
        "default-enabled SWA index cache ground-truth ring bytes",
    )
    for layer in (0, 1):
        for first in range(0, len(positions), 32):
            last = min(first + 32, len(positions))
            queries = positions[first:last]
            real_queries = [position for position in queries if position >= 0]
            query_rows = context.query_row_indices(first, last)
            results = {}
            for enabled in ("0", "1"):
                with patch.dict(os.environ, {"DSV41_SWA_INDEX_CACHE": enabled}):
                    results[enabled] = (
                        context.swa_queries(layer, first, last, initial, encoded),
                        context.swa_queries(
                            layer,
                            first,
                            last,
                            initial,
                            encoded,
                            query_rows=query_rows,
                        ),
                        context.swa_queries(
                            layer,
                            first,
                            last,
                            initial,
                            encoded,
                            query_rows=empty,
                        ),
                    )
            plain_ref, compact_ref, drained_ref = results["0"]
            plain_cached, compact_cached, drained_cached = results["1"]
            assert drained_ref is None and drained_cached is None
            for actual, expected, label in (
                (plain_cached, plain_ref, "plain"),
                (compact_cached, compact_ref, "compact"),
            ):
                _equal(
                    actual.pages.data,
                    expected.pages.data,
                    f"layer {layer} tile {first} cached {label} ring bytes",
                )
                _equal(actual.page_ids, expected.page_ids, f"cached {label} page IDs")
                _equal(
                    actual.valid_starts,
                    expected.valid_starts,
                    f"cached {label} valid starts",
                )
                _equal(
                    actual.valid_ends,
                    expected.valid_ends,
                    f"cached {label} valid ends",
                )
            _equal(
                plain_ref.pages.data.cpu(),
                _expected_swa_pages(queries, spec, floor, salt),
                f"layer {layer} tile {first} ground-truth ring bytes",
            )
            _equal(
                plain_ref.page_ids.cpu(),
                torch.arange(1, len(queries) + 1, dtype=torch.int32),
                "plain page IDs",
            )
            _equal(
                plain_ref.valid_starts.cpu(),
                torch.tensor(
                    [max(floor, pos - 127) if pos >= 0 else 0 for pos in queries],
                    dtype=torch.int32,
                ),
                "plain valid starts",
            )
            _equal(
                plain_ref.valid_ends.cpu(),
                torch.tensor([max(0, pos + 1) for pos in queries], dtype=torch.int32),
                "plain valid ends",
            )
            if query_rows is None:
                _equal(
                    compact_cached.pages.data,
                    plain_cached.pages.data,
                    "all-real tile compact read matches plain",
                )
            else:
                _equal(
                    compact_ref.pages.data.cpu(),
                    _expected_swa_pages(real_queries, spec, floor, salt),
                    "compact ground-truth ring bytes",
                )
            del results, plain_ref, compact_ref, plain_cached, compact_cached
    tiles = (len(positions) + 31) // 32
    assert len(context._swa_index_plans) == tiles
    assert all(plan[1] for plan in context._swa_index_plans.values())
    del initial, encoded, context
    _encoded_bytes.cache_clear()
    return [
        dict(
            tiles=tiles,
            plans=tiles,
            valid_rows=sum(position >= 0 for position in positions),
            layers_compared=(0, 1),
        )
    ]


def _selected_transport_context(rank, device, slot, count=33, start=1046000):
    layout = CacheLayout(cp_size=8, speculative_tokens=0, draft_enabled=False)
    spec = next(page for page in layout.pages if page.slot == slot)
    capacity = (start + count + spec.entries * spec.ratio * 8 - 1) // (
        spec.entries * spec.ratio * 8
    )
    table = (
        (torch.arange(capacity, dtype=torch.int32, device=device) + 17 * rank)
        % capacity
        + 1
    )[None, :].contiguous()
    pool = torch.zeros(
        (capacity + 1, spec.page_stride_bytes), dtype=torch.uint8, device=device
    )
    row = torch.arange(spec.entries, dtype=torch.int32, device=device)[:, None]
    column = torch.arange(spec.encoding.entry_bytes, dtype=torch.int32, device=device)
    for virtual in range(capacity):
        physical = (virtual + 17 * rank) % capacity + 1
        positions = (virtual * 8 + rank) * spec.entries + row
        pool[physical, : spec.entries * spec.encoding.entry_bytes].copy_(
            ((positions * 73 + column * 29 + 11) % 256).to(torch.uint8).flatten()
        )
    cp = _metadata((16 * count,), (start - 4 * count,), rank, device)
    identity = ReplayConfig(ReplayMode.BOUNDED).cache_identity("selected-bytes", layout)
    cache = V41AttentionCache(
        "selected-bytes", identity, layout, start + 16 * count, {}, {}
    )
    context = V41CPAttentionContext(
        cache, cp, 0, {slot: pool}, {slot: table}, {}, {}, 0, start
    )
    context.cache.owners = {
        owner: SimpleNamespace(materialized_end=start, pair=None)
        for owner in PAIR_OWNERS
    }
    context = context.encoder_slice(start, start + count, epoch=1)
    assert context.single_query_owner == 4
    return context, spec


@torch.inference_mode()
def _selected_query_transport(rank, device):
    observations = []
    for slot in (
        RegionSlot(CacheRegion.GLOBAL, 2),
        RegionSlot(CacheRegion.GLOBAL, 20),
        RegionSlot(CacheRegion.INDEX_K, 20),
    ):
        context, spec = _selected_transport_context(rank, device, slot, count=129)
        width = 4096 if slot.region == CacheRegion.INDEX_K else 512
        counts = (1, 4, 32) if slot.region == CacheRegion.INDEX_K else (1, 4, 32, 64, 128)
        for count in counts:
            row = torch.arange(count, dtype=torch.int64)[:, None]
            column = torch.arange(width, dtype=torch.int64)[None, :]
            positions = (row * 9973 + column * 1877) % (context.start // spec.ratio)
            positions[:, ::17] = -1
            if rank != 4:
                positions.fill_(-1)
            selected = positions.to(device=device, dtype=torch.int32)
            columns = torch.arange(spec.encoding.entry_bytes, dtype=torch.int64)
            expected = ((positions[:, :, None] * 73 + columns * 29 + 11) % 256).to(
                torch.uint8
            )
            expected.masked_fill_((positions < 0)[:, :, None], 0)
            variants = (None, 4) if count <= 4 else (4,)
            for owner in variants:
                torch.cuda.synchronize()
                allocated = torch.cuda.memory_allocated(device)
                torch.cuda.reset_peak_memory_stats(device)
                actual, lease = context.gather_selected(
                    slot, selected, slot.owner_layer, query_owner=owner
                )
                torch.cuda.synchronize()
                peak = torch.cuda.max_memory_allocated(device) - allocated
                _equal(actual.cpu(), expected, f"{slot} compact selected bytes")
                assert peak <= 64 * 1024 * 1024
                pages, page_table = cp_attention._packed_pages(actual, spec)
                host_pages, host_table = pages.data.cpu(), page_table.cpu()
                expected_pages = torch.zeros_like(host_pages)
                for query in range(count):
                    for column_first in range(0, width, spec.entries):
                        rows = min(spec.entries, width - column_first)
                        physical = int(host_table[query, column_first // spec.entries])
                        expected_pages[physical, : rows * spec.encoding.entry_bytes].copy_(
                            expected[query, column_first : column_first + rows].flatten()
                        )
                _equal(host_pages, expected_pages, "selected page packing and null/tail bytes")
                context.release(lease, slot.owner_layer)
                observations.append(
                    dict(
                        region=slot.region.value,
                        source_owner=slot.owner_layer,
                        query_owner=owner,
                        query_rows=count,
                        selected_width=width,
                        context_tokens=context.start,
                        max_measured_bytes=peak,
                    )
                )
                del actual, lease, pages, page_table, host_pages, host_table, expected_pages
        assert context.max_gather_live_bytes <= 64 * 1024 * 1024
    return observations


def _selected_query_transport_deferred(rank, device):
    # One-shot negative masking after transport must produce bitwise-identical
    # bytes to the default per-call masked path.
    observations = []
    for slot in (
        RegionSlot(CacheRegion.GLOBAL, 2),
        RegionSlot(CacheRegion.GLOBAL, 20),
        RegionSlot(CacheRegion.INDEX_K, 20),
    ):
        context, spec = _selected_transport_context(rank, device, slot, count=129)
        width = 4096 if slot.region == CacheRegion.INDEX_K else 512
        for count in ((1, 32) if slot.region == CacheRegion.INDEX_K else (1, 64)):
            row = torch.arange(count, dtype=torch.int64)[:, None]
            column = torch.arange(width, dtype=torch.int64)[None, :]
            positions = (row * 9973 + column * 1877) % (context.start // spec.ratio)
            positions[:, ::17] = -1
            if rank != 4:
                positions.fill_(-1)
            selected = positions.to(device=device, dtype=torch.int32)
            columns = torch.arange(spec.encoding.entry_bytes, dtype=torch.int64)
            expected = ((positions[:, :, None] * 73 + columns * 29 + 11) % 256).to(
                torch.uint8
            )
            expected.masked_fill_((positions < 0)[:, :, None], 0)
            for owner in (None, 4):
                reference, lease = context.gather_selected(
                    slot, selected, slot.owner_layer, query_owner=owner
                )
                context.release(lease, slot.owner_layer)
                deferred, lease = context.gather_selected(
                    slot,
                    selected,
                    slot.owner_layer,
                    query_owner=owner,
                    mask_negative=False,
                )
                context.release(lease, slot.owner_layer)
                deferred.masked_fill_((selected < 0)[:, :, None], 0)
                torch.cuda.synchronize()
                _equal(deferred.cpu(), expected, f"{slot} deferred selected bytes")
                _equal(
                    deferred.cpu(),
                    reference.cpu(),
                    f"{slot} deferred matches default checked transport",
                )
                observations.append(
                    dict(
                        region=slot.region.value,
                        source_owner=slot.owner_layer,
                        query_owner=owner,
                        query_rows=count,
                    )
                )
                del reference, deferred
    return observations


def _compare_pages(layout, context, reference):
    for layer in _LAYERS:
        slot = RegionSlot(CacheRegion.SWA, layer)
        spec = context._page_specs[slot]
        page = int(context.tables[slot][0, context.current].item())
        source = reference.swa[layer]
        begin, end = spec.swa_byte_slice(context.cp.cp_rank)
        _equal(
            context.pools[slot][page],
            source.pages.data[source.page_ids[0], begin:end],
            f"SWA layer {layer}",
        )
        _equal(
            context.cache.swa[layer].valid_starts,
            source.valid_starts,
            f"SWA start {layer}",
        )
        _equal(
            context.cache.swa[layer].valid_ends, source.valid_ends, f"SWA end {layer}"
        )
    for owner in (2, 8, 14, 20):
        for region in (CacheRegion.GLOBAL, CacheRegion.INDEX_K):
            slot = RegionSlot(region, owner)
            table, pool = context.tables[slot], context.pools[slot]
            ref = reference.owners[owner]
            pages, ref_table = (
                (ref.global_kv.pages, ref.global_kv.page_table)
                if region == CacheRegion.GLOBAL
                else (ref.index_pages, ref.index_table)
            )
            blocks = (
                context.end + layout.token_block_size - 1
            ) // layout.token_block_size
            for logical in range(context.cp.cp_rank, blocks, 8):
                actual_id = table[0, logical // 8]
                expected_id = ref_table[0, logical]
                _equal(
                    pool[actual_id],
                    pages.data[expected_id],
                    f"{region} owner {owner} block {logical}",
                )
        if owner in PAIR_OWNERS:
            actual, expected = (
                context.cache.owners[owner].pair,
                reference.owners[owner].pair,
            )
            assert actual.next_position == expected.next_position == context.end
            if context.end % 2:
                _equal(actual.partial_kv, expected.partial_kv, f"pair KV {owner}")
                _equal(
                    actual.partial_score, expected.partial_score, f"pair score {owner}"
                )


@torch.inference_mode()
def _pair_checkpoint_restore(rank, device):
    observations = []
    for speculative in (0, 5):
        layout = CacheLayout(
            cp_size=8, speculative_tokens=speculative, draft_enabled=bool(speculative)
        )
        identity = ReplayConfig(ReplayMode.FULL).cache_identity("pair-memory", layout)
        framework = _framework_pages(layout, rank, device)
        for name in ("tables", "pair_tables"):
            for slot in framework[name]:
                framework[name][slot] = torch.arange(
                    1, 18, dtype=torch.int32, device=device
                )[None, :].contiguous()

        def begin(start, end, ready=False):
            cp = _metadata((end - start,), (start,), rank, device)
            return begin_cp_request(
                cp,
                0,
                request_id="pair-memory",
                identity=identity,
                layout=layout,
                max_tokens=16384,
                restored_state_ready=ready,
                **framework,
            )

        published = begin(0, 15360)
        for owner in PAIR_OWNERS:
            published.publish_pair(
                owner, PairCarry.empty(owner, "pair-memory", identity, 15360)
            )
        # Native aligned D2H canonicalizes the entire pair region to zero.
        # Restore into a different physical page on each rank, as H2D does.
        for owner, pool in framework["pair_pools"].items():
            host = pool[14].cpu().clone().zero_()
            pool[30 - rank].copy_(host)
            framework["pair_tables"][owner][0, 14] = 30 - rank
        restored = begin(15360, 15361, True)
        for owner in PAIR_OWNERS:
            pair = restored.cache.owners[owner].pair
            assert pair.next_position == 15360
            assert pair.partial_kv is pair.partial_score is None
            page = int(framework["pair_tables"][owner][0, 14])
            assert not bool(framework["pair_pools"][owner][page].any())
            values = torch.arange(512, dtype=torch.float32, device=device) + owner
            restored.publish_pair(
                owner, PairCarry(owner, "pair-memory", identity, 15361, values, -values)
            )
        odd = begin(15361, 15362, True)
        for owner in PAIR_OWNERS:
            pair = odd.cache.owners[owner].pair
            expected = torch.arange(512, dtype=torch.float32, device=device) + owner
            assert pair.next_position == 15361
            _equal(pair.partial_kv, expected, "restored odd KV")
            _equal(pair.partial_score, -expected, "restored odd scores")
        rejected = []
        for label, start, ready, dirty_byte in (
            ("not_ready", 15360, False, None),
            ("unaligned_even", 15362, True, None),
            ("odd_missing_payload", 15361, True, None),
            ("other_snapshot_nonzero", 15360, True, 0),
            ("trailing_padding_nonzero", 15360, True, -1),
            ("stale_position", 15360, True, (speculative + 1) * 4112 + 4096),
            ("invalid_flag", 15360, True, (speculative + 1) * 4112 + 4104),
        ):
            for pool in framework["pair_pools"].values():
                pool.zero_()
            if dirty_byte is not None:
                pool = framework["pair_pools"][2]
                full_bytes = pool.shape[1] * 8
                byte = dirty_byte % full_bytes
                peer, offset = divmod(byte, pool.shape[1])
                if peer == rank:
                    page = int(framework["pair_tables"][2][0, (start - 1) // 1024])
                    pool[page, offset] = 1
            try:
                begin(start, start + 1, ready)
            except ValueError as error:
                assert "restored execution boundary" in str(error)
                rejected.append(label)
            else:
                raise AssertionError("malformed pair state accepted: " + label)
            torch.distributed.barrier()
        observations.append(
            {
                "speculative_tokens": speculative,
                "restored_start": 15360,
                "odd_continuation": 15361,
                "rejected": rejected,
            }
        )
    return observations


@torch.inference_mode()
def _score_status_check(rank, device):
    # A nonzero scorer status in any tile must fail the one per-layer status
    # check before the selection is published.
    layout = CacheLayout(cp_size=8, speculative_tokens=0, draft_enabled=False)
    identity = ReplayConfig(ReplayMode.FULL).cache_identity("score-status", layout)
    framework = _framework_pages(layout, rank, device)
    models = _models(layout, device)
    cp = _metadata((1000, 1028), (0, 0), rank, device)
    context = begin_cp_request(
        cp,
        0,
        request_id="score-status",
        identity=identity,
        layout=layout,
        max_tokens=2048,
        **framework,
    )
    local = _hidden(context.start, context.end, 0, device).index_select(
        0, (context.positions - context.start).long()
    ).contiguous()
    local.masked_fill_(~context.valid[:, None], torch.nan)
    scorer = cp_attention.score_index_source

    def nonzero_status(*args, **kwargs):
        scores = scorer(*args, **kwargs)
        return replace(scores, status=torch.ones_like(scores.status))

    try:
        with patch.object(cp_attention, "score_index_source", nonzero_status):
            models[2](local, context)
    except RuntimeError as error:
        assert "rejected metadata: status=[1]" in str(error)
    else:
        raise AssertionError("nonzero index status accepted")
    assert context.cache.poisoned and 2 not in context.completed_layers
    assert 2 not in context.selections
    return {"rejected_status": [1]}


@torch.inference_mode()
def _publish_status_check(rank, device):
    # A nonzero writer status in any source tile must fail the one per-layer
    # batched status check before the owner is published.
    layout = CacheLayout(cp_size=8, speculative_tokens=0, draft_enabled=False)
    identity = ReplayConfig(ReplayMode.FULL).cache_identity("publish-status", layout)
    framework = _framework_pages(layout, rank, device)
    models = _models(layout, device)
    cp = _metadata((1000, 1028), (0, 0), rank, device)
    context = begin_cp_request(
        cp,
        0,
        request_id="publish-status",
        identity=identity,
        layout=layout,
        max_tokens=2048,
        **framework,
    )
    local = _hidden(context.start, context.end, 0, device).index_select(
        0, (context.positions - context.start).long()
    ).contiguous()
    local.masked_fill_(~context.valid[:, None], torch.nan)
    rejected = []
    for value in ("0", "513", "-1", "not-an-integer"):
        with patch.dict(os.environ, {"DSV41_CP_SOURCE_ROWS": value}):
            try:
                models[2](local, context)
            except ValueError:
                rejected.append(value)
            else:
                raise AssertionError("invalid CP source tile accepted: " + value)
        assert not context.completed_layers and not context.cache.poisoned
    slot_mapping = cp_attention.cp_kv_slot_mapping

    def reserved_slots(*args, **kwargs):
        slots = slot_mapping(*args, **kwargs)
        # Every row targets the reserved page zero, which the writer rejects
        # with status=2 on every rank uniformly (ownership-independent).
        return torch.zeros_like(slots)

    try:
        with patch.object(cp_attention, "cp_kv_slot_mapping", reserved_slots):
            models[2](local, context)
    except RuntimeError as error:
        assert "compact writer rejected rows: status=[2]" in str(error)
    else:
        raise AssertionError("reserved-page owner KV accepted")
    assert context.cache.poisoned and 2 not in context.completed_layers
    assert 2 not in context.published_sources
    return {"rejected_status": [2], "rejected_source_rows": rejected}


@torch.inference_mode()
def _run_rank():
    rank, device = int(os.environ["RANK"]), torch.device(
        "cuda", int(os.environ["LOCAL_RANK"])
    )
    if os.getuid() == 0 or not str(torch.version.cuda).startswith("13."):
        raise RuntimeError("CP integration must run as a non-root user with CUDA13")
    torch.cuda.set_device(device)
    if torch.cuda.get_device_capability(device)[0] != 10:
        raise RuntimeError("CP integration requires Blackwell")
    torch.distributed.init_process_group("nccl", timeout=timedelta(minutes=15))
    if torch.distributed.get_world_size() != 8:
        raise RuntimeError("CP integration requires eight distinct CUDA ranks")
    # Use the real NCCL world as the framework's CP/TP group in this component fixture.
    collective_torch._group_map[Group.TP] = torch.distributed.group.WORLD
    collective_torch._group_map[Group.DP_AND_TP] = torch.distributed.group.WORLD
    collective_torch._parallelism_config = SimpleNamespace(
        tp_size=8, dp_size=1, world_size=8
    )
    collective_torch._initialized = True
    torch.backends.cuda.matmul.allow_tf32 = False
    swa_reads = _swa_read_batches(rank, device)
    swa_index_cache = _swa_index_cache_reads(rank, device)
    selected_reads = _selected_query_transport(rank, device)
    deferred_reads = _selected_query_transport_deferred(rank, device)
    pair_checkpoints = _pair_checkpoint_restore(rank, device)
    score_status = _score_status_check(rank, device)
    publish_status = _publish_status_check(rank, device)
    layout = CacheLayout(cp_size=8, speculative_tokens=0, draft_enabled=False)
    identity = ReplayConfig(ReplayMode.FULL).cache_identity(
        "cp8-attention-integration", layout
    )
    models = _models(layout, device)
    frameworks = [_framework_pages(layout, rank, device) for _ in range(2)]
    references = [
        V41AttentionCache.allocate_local(
            str(index), identity, layout, 2048, device=device
        )
        for index in range(2)
    ]
    starts, records, rejected_batches = [0, 0], [], []
    for epoch, lengths in enumerate(((3, 1), (130, 18), (1000, 1028))):
        read_queries = (cp_attention._READ_QUERIES, 1, 32)[epoch]
        if epoch == 0:
            os.environ.pop("DSV41_CP_READ_QUERIES", None)
        else:
            os.environ["DSV41_CP_READ_QUERIES"] = str(read_queries)
        cp = _metadata(lengths, starts, rank, device)
        contexts = [
            begin_cp_request(
                cp,
                index,
                request_id=str(index),
                identity=identity,
                layout=layout,
                max_tokens=2048,
                epoch=epoch,
                **frameworks[index],
            )
            for index in range(2)
        ]
        local_contexts = [
            references[index].begin_forward(
                epoch=epoch, start=starts[index], end=starts[index] + lengths[index]
            )
            for index in range(2)
        ]
        for layer in _LAYERS:
            for index, context in enumerate(contexts):
                canonical = _hidden(context.start, context.end, index, device)
                local = canonical.index_select(
                    0, (context.positions - context.start).long()
                ).contiguous()
                local.masked_fill_(~context.valid[:, None], torch.nan)
                if epoch == layer == index == 0:
                    for value in ("0", "513", "-1", "not-an-integer"):
                        with patch.dict(os.environ, {"DSV41_CP_READ_QUERIES": value}):
                            try:
                                models[layer](local, context)
                            except ValueError:
                                rejected_batches.append(value)
                            else:
                                raise AssertionError(
                                    "invalid CP read batch accepted: " + value
                                )
                        assert (
                            not context.completed_layers and not context.cache.poisoned
                        )
                expected = models[layer](canonical, local_contexts[index])
                gathers_before = context.gather_count
                projection_rows, hooks = [], []
                for name in ("wq_a", "wq_b", "wkv", "wo_b", "index_wq_b"):
                    projection = getattr(models[layer], name)
                    if projection is not None:
                        hooks.append(
                            projection.register_forward_pre_hook(
                                lambda _, args, name=name: projection_rows.append(
                                    (name, args[0].shape[0])
                                )
                            )
                        )
                reader_module, reader_name = cp_attention, "compact_attention"
                if os.environ.get("DSV41_ATTENTION_BACKEND") == "flashmla":
                    from rtp_llm.models_py.modules.dsv41 import flashmla

                    reader_module, reader_name = flashmla, "flashmla_compact_attention"
                try:
                    with (
                        patch.object(
                            reader_module,
                            reader_name,
                            wraps=getattr(reader_module, reader_name),
                        ) as reader,
                        patch.object(
                            cp_attention,
                            "score_index_source",
                            wraps=cp_attention.score_index_source,
                        ) as source_scorer,
                        patch.object(
                            cp_attention,
                            "score_candidate_tile",
                            wraps=cp_attention.score_candidate_tile,
                        ) as candidate_scorer,
                        patch.object(
                            cp_attention,
                            "encode_compact",
                            wraps=cp_attention.encode_compact,
                        ) as encoder,
                    ):
                        actual = models[layer](local, context)
                finally:
                    for hook in hooks:
                        hook.remove()
                real_rows = int(context.valid.sum())
                reader_rows = [call.args[0].shape[0] for call in reader.call_args_list]
                assert sum(reader_rows) == real_rows
                assert all(rows > 0 for rows in reader_rows)
                assert encoder.call_count == 1
                assert encoder.call_args.args[0].shape[0] == real_rows
                scoring = source_scorer.call_args_list + candidate_scorer.call_args_list
                if layer_sources(layer).scores_queries:
                    source_tiles = (
                        (max(1, context.end // layer_sources(layer).ratio) + 16383)
                        // 16384
                        if layer <= 20
                        else 1
                    )
                    assert (
                        sum(call.args[0].shape[0] for call in scoring)
                        == real_rows * source_tiles
                    )
                    assert all(call.args[0].shape[0] > 0 for call in scoring)
                    assert context.selections[layer].scorer_calls == len(scoring)
                else:
                    assert not scoring
                assert len(projection_rows) == 4 + layer_sources(layer).scores_queries
                assert all(
                    rows == int(context.valid.sum()) for _, rows in projection_rows
                )
                if layer == 0:
                    swa_calls = (context.query_rows + read_queries - 1) // read_queries
                    assert (
                        context.gather_count - gathers_before
                        == swa_calls + 1 + bool(context.start)
                    )
                selected = expected.index_select(
                    0, (context.positions - context.start).long()
                )
                selected.masked_fill_(~context.valid[:, None], 0)
                _equal(
                    actual,
                    selected,
                    f"rank {rank}, request {index}, epoch {epoch}, layer {layer}",
                )
                if layer_sources(layer).scores_queries:
                    ref_selection = local_contexts[index].selections[layer]
                    selected_ids = ref_selection.topk.index_select(
                        0, (context.positions - context.start).long()
                    )
                    selected_ids.masked_fill_(~context.valid[:, None], -1)
                    _equal(
                        context.selections[layer].topk, selected_ids, f"top-k {layer}"
                    )
                    if layer == 20:
                        selected_blocks = ref_selection.candidate_blocks.index_select(
                            0, (context.positions - context.start).long()
                        )
                        selected_blocks.masked_fill_(~context.valid[:, None], -1)
                        _equal(
                            context.selections[layer].candidate_blocks,
                            selected_blocks,
                            "L20 candidate blocks",
                        )
                    assert bool(
                        (context.selections[layer].status[~context.valid] == 0).all()
                    )
        for index, context in enumerate(contexts):
            _compare_pages(layout, context, references[index])
            assert context.completed_layers == set(_LAYERS)
            assert context.max_gather_live_bytes <= cp_attention.MAX_GATHER_BYTES
            assert context.gather_count > 0
            if epoch == 0 and rank >= 3:
                assert not bool(contexts[0].valid.any().item())
            records.append(
                dict(
                    epoch=epoch,
                    request=index,
                    start=context.start,
                    end=context.end,
                    local_rows=context.query_rows,
                    model_rows=context.model_query_rows,
                    read_queries=read_queries,
                    valid_rows=int(context.valid.sum().item()),
                    gathers=context.gather_count,
                    max_receive_bytes=context.max_receive_bytes,
                    max_gather_live_bytes=context.max_gather_live_bytes,
                )
            )
            starts[index] = context.end
        torch.distributed.barrier()
    collective_counts = [None] * 8
    torch.distributed.all_gather_object(
        collective_counts, [record["gathers"] for record in records]
    )
    assert all(counts == collective_counts[0] for counts in collective_counts)
    torch.cuda.synchronize()
    result = dict(
        scope="real CP8 attention component comparison; no model acceptance",
        rank=rank,
        gpu=str(torch.cuda.get_device_properties(device).uuid),
        cases=records,
        pair_checkpoints=pair_checkpoints,
        swa_reads=swa_reads,
        swa_index_cache=swa_index_cache,
        selected_reads=selected_reads,
        deferred_reads=deferred_reads,
        score_status=score_status,
        publish_status=publish_status,
        rejected_read_queries=rejected_batches,
    )
    destination = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
    if destination:
        Path(destination, f"cp_attention_rank{rank}.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
    print(json.dumps(result), flush=True)
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


def main():
    for flag in _FLAGS:
        os.environ[flag] = "1"
    os.environ["DSV41_ATTENTION_BACKEND"] = "native"
    if "LOCAL_RANK" not in os.environ:
        if torch.cuda.device_count() != 8:
            raise RuntimeError("launch CP comparison with exactly eight visible GPUs")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node=8",
                str(Path(__file__).resolve()),
            ],
            check=True,
        )
    else:
        _run_rank()


if __name__ == "__main__":
    main()
