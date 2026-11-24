"""Creation-owned CP SWA batches and actual eight-rank cache staging.

CPU contracts keep SharedForwardState births and SharedNativeCache staging.
Only device/stream/upload and native kernel boundaries have explicit doubles.
The separately gated WORLD8 case retains native transport, tail staging and
both old/new decoders; it is correctness evidence, not model latency evidence.
"""

import gc
import os
import tempfile
import unittest
import weakref
from contextlib import ExitStack
from datetime import timedelta
from itertools import accumulate
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv41 import cp as cp_module
from rtp_llm.models_py.modules.dsv41.fp8 import _v41_swa_metadata as metadata
from rtp_llm.models_py.modules.dsv41.ppu import _quant, cp_prefill_swa_batch as batch
from rtp_llm.models_py.modules.dsv41.ppu import kv_codec, prefill, prefill_swa
from rtp_llm.models_py.modules.dsv41.ppu.native_cache_adapter import (
    SharedNativeCache,
    _StagedSwaSegments,
)
from rtp_llm.models_py.modules.dsv41.ppu.prefill_runtime import SharedForwardState
from rtp_llm.models_py.modules.dsv41.ppu.replay_policy import (
    PrefillPolicy,
    SharedPrefillPolicy,
)
from rtp_llm.models_py.modules.dsv41.ppu.test.test_cp8_cross_layer_mhc import (
    CP_ENV,
    _plan,
)
from rtp_llm.models_py.modules.dsv41.ppu.test.test_cp8_l20_query_projection import (
    _context,
)
from rtp_llm.models_py.modules.dsv41.ppu.test.test_local_prefill_metadata import (
    local_metadata_cpu_stream,
)


FLAGS = dict(
    CP_ENV,
    RTP_V41_PPU_CP8_REUSE_LOCAL_ROWS="1",
    RTP_V41_PPU_CP8_BATCH_SWA_DECODE="1",
    RTP_V41_PPU_CP8_YOCO_SHARED_PREFIX_READBACK="0",
    RTP_V41_PPU_CP8_YOCO_Q8_ATTENTION="0",
    RTP_V41_PPU_CP8_FORWARD_MEGA_CAPACITY="0",
    RTP_V41_PPU_CP8_BATCHED_SWA="1",
    RTP_V41_PPU_CP8_BOUNDED_SWA_RANGE="1",
    RTP_V41_PPU_CP8_DIRECT_SWA_WORKSPACE="1",
    RTP_V41_PPU_CP8_ASYNC_PREFILL_METADATA="1",
    RTP_V41_PPU_OPTIMIZED_DIAGNOSTICS="0",
    RTP_V41_PPU_OPTIMIZED_LATE_FOCUS="0",
    RTP_V41_PPU_PACKED_ATTENTION="0",
    DSV4_FORWARD_TENSOR_DEBUG="0",
    DSV4_DEBUG_HIDDEN_COMPARE="0",
)
SENTINEL = -37.5


def _bytes_equal(left, right):
    return (
        left.dtype == right.dtype
        and left.shape == right.shape
        and torch.equal(
            left.contiguous().view(torch.uint8),
            right.contiguous().view(torch.uint8),
        )
    )


def _packed(positions, requests, layer):
    """Valid FP8/UE8M0 bytes with request/layer/absolute-position identities."""
    columns = torch.arange(512, device=positions.device)
    result = torch.empty(
        (positions.numel(), 528), dtype=torch.uint8, device=positions.device
    )
    result[:, :512] = (
        positions[:, None] * 7
        + requests[:, None] * 19
        + columns[None]
        + layer * 23
    ).remainder(127).to(torch.uint8)
    # Cover signed zero without manufacturing FP8 NaNs. The original decoder
    # remains the device oracle for every actual BF16 output bit.
    if positions.numel():
        result[:, 0] = 0
        result[:, 1] = 128
    result[:, 512:] = (
        positions[:, None].remainder(5) + torch.arange(16, device=positions.device)
    ).remainder(5).add(125).to(torch.uint8)
    return result


class _SwaPools:
    """Native 128-row rings split into eight 8448-byte rank slices.

    Request/page tables use distinct allocated physical checkpoints, including
    block zero as a canary. Small prefixes avoid allocating a full 128K cache.
    """

    group_tags = ["swa_kv"]

    def __init__(self, totals, device, *, alias_pages=False):
        columns = max(1, (max(totals) + 511) // 512)
        if alias_pages:
            # Formal host-geometry-only cases never stage these pages. Ring
            # aliases avoid allocating 40 layers of a hypothetical 128K pool.
            blocks = len(totals) + 1
            host = torch.arange(1, blocks, dtype=torch.int32)[:, None].expand(
                -1, columns
            ).clone()
        else:
            blocks = len(totals) * columns + 1
            host = torch.arange(1, blocks, dtype=torch.int32).reshape(
                len(totals), columns
            )
        target = host.to(device=device, copy=True)
        self.inputs = {
            "swa_kv": NS(
                kv_cache_block_id=host,
                kv_cache_block_id_device=target,
                kv_cache_kernel_block_id=host,
                kv_cache_kernel_block_id_device=target,
            )
        }
        self.handles = {}
        for layer in range(40):
            raw = torch.full(
                (blocks * 8448 + 32,), 173, dtype=torch.uint8, device=device
            )
            self.handles[layer] = NS(
                kv_cache_base=raw[16:-16].view(blocks, 8448),
                layer_id=layer,
                canaries=raw,
            )

    def get_layer_cache(self, layer, tag):
        if tag != "swa_kv":
            raise AssertionError("SWA fixture must not invent source-pool reads")
        return self.handles[layer]

    def get_entries_per_block(self, tag):
        return 128

    def get_seq_size_per_block(self, tag):
        return 512

    get_kernel_seq_size_per_block = get_seq_size_per_block


def _birth(
    rank=0, lengths=(256, 272), prefixes=(129, 17), device="cpu", *, alias_pages=False
):
    lengths, prefixes = tuple(lengths), tuple(prefixes)
    cp, ids = _context(rank, lengths, prefixes, device)
    totals = tuple(p + n for p, n in zip(prefixes, lengths))
    geometry = NS(
        num_layers=40,
        kv_source_layer_ids=(),
        index_source_layer_ids=(),
        head_dim=512,
        window=128,
        ratio=lambda layer: 0,
        kv_source_for=lambda layer: None,
    )
    native = _SwaPools(totals, device, alias_pages=alias_pages)
    config = PrefillPolicy().freeze(
        {
            "_ppu_cp_cache": dict(
                enabled=True, size=8, rank=rank, sliced_prefill=True
            )
        }
    )
    cache = SharedNativeCache(
        config,
        geometry,
        native,
        native.inputs,
        totals,
        device=torch.device(device),
        prefill=True,
        defer_producer_slots=True,
    )
    model = NS(
        _is_decode_role=False,
        _is_speculative=False,
        _prefill_cp_size=8,
        parallelism_config=NS(tp_size=8),
        v4=NS(
            layers=[],
            _propagate_cp_ctx=lambda context: None,
            capture_aux_hidden_layer_ids=(),
        ),
    )
    runtime = NS(model=model, policy=SharedPrefillPolicy(mode="compact"), current=None)
    inputs = NS(input_ids=ids, combo_position_ids=cp.global_positions)
    attn = NS(
        cu_seqlens=torch.tensor(
            (0, *accumulate(cp.chunk_lengths_per_req)),
            dtype=torch.int32,
            device=device,
        ),
        input_lengths=torch.tensor(cp.chunk_lengths_per_req, dtype=torch.int32),
        prefix_lengths=torch.tensor(prefixes, dtype=torch.int32),
        is_cuda_graph=False,
    )
    state = SharedForwardState(
        runtime, inputs, attn, cache,
        prefill=True, idle=False, prefixes=prefixes, lengths=lengths,
    )
    runtime.current = state
    state.bind_context(cp)
    if state._cp_full_rows_birth is None or state._cp_metadata_stream is None:
        raise AssertionError("actual full-row bind must create both row and stream births")
    return state


def _prime(state, layer):
    cache = state.cache
    for request, prefix in enumerate(state.query_prefixes):
        old = torch.arange(
            max(0, prefix - 127), prefix, device=cache.device, dtype=torch.int64
        )
        ids = torch.full_like(old, request)
        cache._copy_rows((layer, "swa_kv"), request, old, _packed(old, ids, layer))


def _stage(state, layer, *, transport=False, bounded=False):
    """Run real stage_swa; CPU doubles only its native transport/write boundary."""
    cache = state.cache
    cp = state.context if bounded else state.original_context
    if not bounded:
        _prime(state, layer)
    positions, requests, real = state.full_rows(cp, selected_only=bounded)
    if transport:
        local = _packed(cp.global_positions, cp.req_id_per_token, layer)
        if bounded:
            packed = cp_module.cp_all_gather_full_varlen(
                local, cp, replay_only=True
            )[real]
        else:
            packed = cp_module.cp_all_gather_full(
                local, cp, receive=cache.gather_receive
            )[real]
    else:
        packed = _packed(positions[real], requests[real], layer)
    before = packed.clone()
    starts = (
        tuple(p + max(n - 140, 0) for p, n in zip(state.query_prefixes, state.lengths))
        if bounded else None
    )
    cache.stage_swa(
        layer, positions[real], requests[real], packed, state.query_prefixes,
        retain_rows=True, retain_prefix_tail=not bounded, bounded_starts=starts,
    )
    if not _bytes_equal(packed, before):
        raise AssertionError("native staging modified its packed producer source")
    if not all(
        type(cache._fresh_swa[layer, request]) is _StagedSwaSegments
        for request in range(len(cache.lengths))
    ):
        raise AssertionError("test requires real direct staged tail/fresh snapshots")


def _compact(state, *, group=None, transport=False):
    plan = _plan(state, group=group)
    state.plan, state.plan_reason = plan, "compact140"
    ids = state.inputs.input_ids
    hidden = torch.zeros((ids.numel(), 1), dtype=torch.bfloat16, device=ids.device)
    if transport:
        compact_ids = plan.exchange.compact(ids)
        hidden = plan.exchange.compact(hidden)
    else:
        # Explicit CPU transport boundary: metadata/birth are real; this case
        # does not claim values were delivered by a process group.
        compact_ids = ids.new_zeros(plan.context.chunk_length)
        hidden = hidden.new_zeros((plan.context.chunk_length, 1))
    state.compact_l20_queries(hidden, compact_ids)
    if state._cp_metadata_compact_birth is None:
        raise AssertionError("actual compact activation must publish its row birth")


def _workspace(state, *, segmented=False, bounded=False, admitted=True):
    cp, cache = state.context, state.cache
    positions, requests, cu, valid = state.rows(cp)
    context = prefill.build_step_context(positions, requests, cu, valid_mask=valid)
    starts = (
        {rid: p + max(n - 140, 0)
         for rid, (p, n) in enumerate(zip(state.query_prefixes, state.lengths))}
        if bounded else {}
    )
    plan = (
        prefill.prepare_cp8_segmented_swa(
            state, cp, context, window=128, starts={}
        ) if segmented else None
    )
    if segmented and plan is None:
        raise AssertionError("full all-live fixture must admit its segmented SWA plan")
    return prefill._CPPrefillWorkspace(
        NS(decode_kv=kv_codec.decode_kv), cache, cache.geometry, context,
        cp, None, 0, 128, starts,
        swa_plan=plan, state=state if admitted else None,
    )


def _gapped(ranges):
    result, offset = [], 17
    for request, _, first, count, _ in ranges:
        result.append((request, offset, first, count, None))
        offset += count + 3
    return result, offset + 5


def _old_ranges(cache, layer, ranges, output, *, native):
    for request, begin, first, count, _ in ranges:
        if not count:
            continue
        out = output[begin:begin + count]
        if native:
            if not cache.try_decode_swa_range(layer, request, first, out):
                raise AssertionError("old native segment decoder unexpectedly declined")
        else:
            saved = cache._fresh_swa[layer, request]
            start = first - saved.prefix + saved.tail.shape[0]
            packed = torch.cat((saved.tail, saved.fresh))[start:start + count]
            _quant.dequantize_swa_kv_rows(packed, out=out)


class _CPUDeviceBoundary:
    """Declared transport/write/decode doubles; cache control flow is unchanged."""

    def __init__(self):
        self.cache = None
        self.call_count = 0

    def tails(self, pool, placement, prefixes, window, **kwargs):
        return tuple(
            _packed(
                positions := torch.arange(max(0, prefix - window + 1), prefix),
                torch.full_like(positions, request), pool.handle.layer_id,
            )
            for request, prefix in enumerate(prefixes)
        )

    def write(self, pool, placement, table, request, start, end, window, packed):
        positions = torch.arange(start, end, dtype=torch.int64)
        page_end = ((positions // pool.span + 1) * pool.span).clamp_max(end)
        keep = positions >= page_end - pool.entries
        placement.write_ring(
            (pool.handle.layer_id, "swa_kv"), pool, request,
            positions[keep], packed[keep],
        )
        return positions[keep].tolist()

    @staticmethod
    def read(domain, packed, positions):
        return packed.index_select(0, positions - domain[0])

    def decode(self, descriptors, output, max_rows, *, reference):
        self.call_count += 1
        owners = {
            value.data_ptr(): value
            for saved in self.cache._fresh_swa.values()
            if type(saved) is _StagedSwaSegments
            for value in (saved.tail, saved.fresh)
        }
        for tail_ptr, fresh_ptr, begin, start, tail_rows, count in descriptors.tolist():
            tail, fresh = owners[tail_ptr], owners[fresh_ptr]
            if tail.shape[0] != tail_rows or count > max_rows:
                raise AssertionError("malformed batch pointer descriptor")
            packed = torch.cat((tail, fresh))[start:start + count]
            _quant.dequantize_swa_kv_rows(packed, out=output[begin:begin + count])
        return True


class CP8SWABatchContracts(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(torch.inference_mode(False))
        self.stack.enter_context(torch.no_grad())
        self.stack.enter_context(patch.dict(os.environ, FLAGS))
        self.stack.enter_context(patch.object(_quant, "is_ppu", return_value=True))
        self.stream = self.stack.enter_context(local_metadata_cpu_stream())
        self.stack.enter_context(patch.object(batch, "_device_supported", return_value=True))
        self.stack.enter_context(patch.object(batch, "_provider_available", return_value=True))
        self.stack.enter_context(patch.object(_quant, "_ppu_fast_path", return_value=False))
        self.upload = self.stack.enter_context(patch.object(
            metadata, "_upload",
            side_effect=lambda values, device, dtype: torch.tensor(values, device=device, dtype=dtype),
        ))
        self.boundary = _CPUDeviceBoundary()
        # A Mock records tensor arguments forever. These explicit boundary
        # functions deliberately retain none, so lifetime assertions measure
        # the receipt/cache owners rather than an instrumentation reference.
        self.stack.enter_context(patch.object(prefill_swa, "gather_tails", new=self.boundary.tails))
        self.stack.enter_context(patch.object(prefill_swa, "write_prefill_range", new=self.boundary.write))
        self.stack.enter_context(patch.object(prefill_swa, "read_staged_rows", new=self.boundary.read))
        self.stack.enter_context(patch.object(
            _quant, "try_dequantize_swa_batch", new=self.boundary.decode,
        ))
        self.decoder = self.boundary

    def fixture(self, **kwargs):
        state = _birth(**kwargs)
        self.boundary.cache = state.cache
        return state

    def prepare(self, state, ranges=None, output=None):
        if ranges is None:
            ranges = [(0, 17, state.query_prefixes[0] - 2, 11, None)]
        if output is None:
            output = torch.full((64, 512), SENTINEL, dtype=torch.bfloat16)
        return batch.prepare(state, state.context, state.cache, output, ranges)

    def test_formal_all_rank_geometry_preserves_two_spans_and_disjoint_outputs(self):
        for rank in range(8):
            spans, legacy = prefill._cp8_swa_spans(
                (4096,) * 8, (126976,) * 8, rank, (512,) * 8, 128
            )
            rows = sum(stop - first for request in spans for first, stop in request)
            self.assertEqual(rows, 5112 if rank == 7 else 6128)
            self.assertEqual(sum(map(len, spans)), 8 if rank == 7 else 16)
            self.assertEqual(legacy, 33784 - rank * 4096)

    def test_formal_full_compact_and_bounded_rows_match_the_receipt_inventory(self):
        expected_rows = (
            145976, 145104, 143592, 142080, 140568, 139056, 137544, 115712
        )
        for rank in range(8):
            with self.subTest(rank=rank):
                state = self.fixture(
                    rank=rank, lengths=(4096,) * 8, prefixes=(126976,) * 8,
                    alias_pages=True,
                )
                full = _workspace(state, segmented=True)
                _compact(state)
                l20 = _workspace(state)
                bounded = _workspace(state, bounded=True)
                rows = (
                    20 * sum(item[3] for item in full.swa_ranges)
                    + sum(item[3] for item in l20.swa_ranges)
                    + 19 * sum(item[3] for item in bounded.swa_ranges)
                )
                segments = (
                    20 * len(full.swa_ranges) + len(l20.swa_ranges)
                    + 19 * len(bounded.swa_ranges)
                )
                self.assertEqual(rows, expected_rows[rank])
                self.assertEqual(segments, 320 if rank == 7 else 480)
                # Global source/workspace switches are not synthesized here;
                # formal plans7/launches40/layers0..39 belong to model receipts.
                self.assertEqual(self.decoder.call_count, 0)

    def test_full_and_compact_workspace_matches_old_ranges_and_raw_sources(self):
        state = self.fixture(rank=1)
        full = _workspace(state, segmented=True)
        self.assertIsNotNone(full._swa_batch)
        for layer in (0, 3, 19):
            _stage(state, layer)
            frozen = {
                key: (value.tail.clone(), value.fresh.clone())
                for key, value in state.cache._fresh_swa.items()
            }
            expected = torch.full_like(full.kv, SENTINEL)
            _old_ranges(state.cache, layer, full.swa_ranges, expected, native=False)
            full.kv.fill_(SENTINEL)
            full.refresh_swa(NS(decode_kv=kv_codec.decode_kv), state.cache, layer)
            self.assertTrue(_bytes_equal(full.kv, expected))
            for key, (tail, fresh) in frozen.items():
                saved = state.cache._fresh_swa[key]
                self.assertTrue(_bytes_equal(saved.tail, tail))
                self.assertTrue(_bytes_equal(saved.fresh, fresh))
        self.assertEqual(state.cache._cp_segmented_swa_stats["dequant_calls"], 3)
        self.assertEqual(state.cache._cp_segmented_swa_stats["fused_decode_hits"], 3)
        _stage(state, 20)
        _compact(state)
        for layer, bounded in ((20, False), (21, True), (39, True)):
            if bounded:
                _stage(state, layer, bounded=True)
            workspace = _workspace(state, bounded=bounded)
            self.assertIsNotNone(workspace._swa_batch)
            expected = torch.full_like(workspace.kv, SENTINEL)
            _old_ranges(state.cache, layer, workspace.swa_ranges, expected, native=False)
            workspace.kv.fill_(SENTINEL)
            workspace.refresh_swa(NS(decode_kv=kv_codec.decode_kv), state.cache, layer)
            self.assertTrue(_bytes_equal(workspace.kv, expected))
        self.assertEqual(self.decoder.call_count, 6)
        self.assertEqual(state.cache._cp_swa_batch_stats["layers"], [0, 3, 19, 20, 21, 39])

    def test_gapped_output_keeps_global_prefix_holes_suffix_and_ignores_positions_field(self):
        state = self.fixture()
        _stage(state, 0)
        workspace = _workspace(state, segmented=True, admitted=False)
        ranges, rows = _gapped(workspace.swa_ranges)
        actual = torch.full((rows, 512), SENTINEL, dtype=torch.bfloat16)
        expected = actual.clone()
        _old_ranges(state.cache, 0, ranges, expected, native=False)
        before = self.upload.call_count
        receipt = self.prepare(state, ranges, actual)
        self.assertIsNotNone(receipt)
        self.assertEqual(self.upload.call_count, before, "prepare must not upload")
        ranges[0] = (*ranges[0][:4], torch.arange(ranges[0][3]))
        self.assertTrue(batch.try_refresh(receipt, state.cache, 0))
        self.assertTrue(_bytes_equal(actual, expected))
        self.assertTrue(torch.equal(actual[:17], torch.full_like(actual[:17], SENTINEL)))
        self.assertEqual(self.upload.call_count, before + 1)

    def test_unknown_hooked_invalid_overlap_and_no_work_decline_before_upload(self):
        state = self.fixture()
        out = torch.full((64, 512), SENTINEL, dtype=torch.bfloat16)
        before = self.upload.call_count
        self.assertIsNone(batch.prepare(NS(), state.context, state.cache, out, []))
        for ranges in (
            [], [(0, 0, 129, 0, None)], [(0, 0, 129, 33, None), (1, 32, 17, 3, None)],
            [(0, 63, 129, 2, None)], [(2, 0, 129, 1, None)], [(0, 0, -1, 1, None)],
            [(False, 0, 129, 1, None)], [(0, 0, 129, 1)],
        ):
            with self.subTest(ranges=ranges):
                self.assertIsNone(self.prepare(state, ranges, out))
        with patch.object(state.cache, "read_swa", wraps=state.cache.read_swa):
            self.assertIsNone(self.prepare(state))
        with patch.object(state.runtime.model, "_forward_hooks", {1: object()}, create=True):
            self.assertIsNone(self.prepare(state))
        for name, value in (
            ("input_lengths_global_host", [256, 272]),
            ("swa_replay_starts_host", ([],)),
        ):
            other = self.fixture()
            setattr(other.context, name, value)
            self.assertIsNone(self.prepare(other), name)
        self.assertEqual(self.upload.call_count, before)
        self.assertEqual(self.decoder.call_count, 0)

    def test_disabled_reuse_and_no_work_keep_the_original_birth_uninvalidated(self):
        state = self.fixture()
        birth = state._cp_full_rows_birth
        before = self.upload.call_count
        invalidations = state._cp_metadata_evidence["invalidations"]
        for flag in ("RTP_V41_PPU_CP8_REUSE_LOCAL_ROWS", "RTP_V41_PPU_CP8_BATCH_SWA_DECODE"):
            with self.subTest(flag=flag), patch.dict(os.environ, {flag: "0"}):
                self.assertIsNone(self.prepare(state))
        batch_flag = "RTP_V41_PPU_CP8_BATCH_SWA_DECODE"
        previous = os.environ.pop(batch_flag)
        try:
            self.assertIsNone(self.prepare(state))
        finally:
            os.environ[batch_flag] = previous
        self.assertIsNone(self.prepare(state, [(0, 17, 129, 0, None)]))
        self.assertIs(state._cp_full_rows_birth, birth)
        self.assertFalse(state._cp_full_rows_invalid)
        self.assertFalse(state._cp_metadata_invalid)
        self.assertEqual(state._cp_metadata_evidence["invalidations"], invalidations)
        self.assertEqual(self.upload.call_count, before)
        self.assertEqual(self.decoder.call_count, 0)

    def test_missing_uncovered_and_aliased_sources_fallback_before_first_upload(self):
        state = self.fixture()
        out = torch.full((64, 512), SENTINEL, dtype=torch.bfloat16)
        receipt = self.prepare(state, output=out)
        before = self.upload.call_count
        self.assertFalse(batch.try_refresh(receipt, state.cache, 0))
        _stage(state, 0)
        uncovered = self.prepare(state, [(0, 0, 0, 1, None)], out)
        self.assertFalse(batch.try_refresh(uncovered, state.cache, 0))
        saved = state.cache._fresh_swa[0, 0]
        aliased = saved.fresh.view(torch.bfloat16).reshape(-1)[:1024].view(2, 512)
        snapshot = aliased.clone()
        alias_receipt = self.prepare(state, [(0, 0, saved.prefix, 2, None)], aliased)
        self.assertIsNotNone(alias_receipt)
        self.assertFalse(batch.try_refresh(alias_receipt, state.cache, 0))
        self.assertTrue(_bytes_equal(aliased, snapshot))
        self.assertEqual(self.upload.call_count, before)
        self.assertEqual(self.decoder.call_count, 0)
        self.assertEqual(state.cache._cp_swa_batch_stats["fallbacks"], 3)

    def test_stale_rows_output_ranges_hooks_and_streams_fail_before_any_write(self):
        for change in (
            "native_version", "native_owner", "cp_rank", "cp_info", "runtime",
            "range", "output", "hook", "stream",
        ):
            with self.subTest(change=change):
                state = self.fixture()
                _stage(state, 0)
                ranges = [(0, 17, 127, 11, None)]
                output = torch.full((64, 512), SENTINEL, dtype=torch.bfloat16)
                receipt = self.prepare(state, ranges, output)
                self.assertIsNotNone(receipt)
                original = output.clone()
                before = self.upload.call_count
                if change == "native_version":
                    state.context.req_id_per_token.add_(0)
                elif change == "native_owner":
                    state.context.global_positions = state.context.global_positions.clone()
                elif change == "cp_rank":
                    state.context.cp_rank = (state.context.cp_rank + 1) % 8
                elif change == "cp_info":
                    state.context.cp_info = NS()
                elif change == "runtime":
                    state.runtime.current = None
                elif change == "range":
                    ranges[0] = (0, 16, 127, 11, None)
                elif change == "output":
                    output.transpose_(0, 1)
                elif change == "hook":
                    state.runtime.model._forward_pre_hooks = {1: object()}
                else:
                    self.stream.handle += 1
                with self.assertRaisesRegex(RuntimeError, "CP SWA batch"):
                    batch.try_refresh(receipt, state.cache, 0)
                if change == "output":
                    output.transpose_(0, 1)
                self.assertTrue(_bytes_equal(output, original))
                self.assertEqual(self.upload.call_count, before)
                self.stream.handle = 19

    def test_compact_replay_layout_replacement_is_not_recertified_by_refresh(self):
        state = self.fixture()
        _compact(state)
        _stage(state, 21, bounded=True)
        workspace = _workspace(state, bounded=True)
        self.assertIsNotNone(workspace._swa_batch)
        before = self.upload.call_count
        state.context.swa_replay_starts_host = tuple(
            value for value in state.context.swa_replay_starts_host
        )
        with self.assertRaisesRegex(RuntimeError, "CP SWA batch"):
            batch.try_refresh(workspace._swa_batch, state.cache, 21)
        self.assertEqual(self.upload.call_count, before)

    def test_same_layer_source_write_rebind_and_release_revoke_an_used_receipt(self):
        for change in ("version", "rebind", "release"):
            with self.subTest(change=change):
                state = self.fixture()
                _stage(state, 0)
                receipt = self.prepare(state)
                self.assertTrue(batch.try_refresh(receipt, state.cache, 0))
                before = self.upload.call_count
                frozen = receipt.output.clone()
                saved = state.cache._fresh_swa[0, 0]
                if change == "version":
                    saved.fresh.add_(0)
                elif change == "rebind":
                    state.cache._fresh_swa[0, 0] = saved._replace(fresh=saved.fresh.clone())
                else:
                    state.cache.release_swa(0)
                with self.assertRaisesRegex(RuntimeError, "CP SWA batch"):
                    batch.try_refresh(receipt, state.cache, 0)
                self.assertEqual(self.upload.call_count, before)
                self.assertTrue(_bytes_equal(receipt.output, frozen))

    def test_previous_sources_live_until_next_enqueue_and_retained_workspace_stays_frozen(self):
        state = self.fixture()
        _stage(state, 0)
        receipt = self.prepare(state)
        self.assertTrue(batch.try_refresh(receipt, state.cache, 0))
        old_output = receipt.output.clone()
        old = tuple(
            weakref.ref(value)
            for value in state.cache._fresh_swa[0, 0][1:]
        )
        state.cache.release_swa(0)
        gc.collect()
        self.assertTrue(
            all(owner() is not None for owner in old),
            "queued work must retain both packed source owners",
        )
        _stage(state, 1)
        self.assertTrue(batch.try_refresh(receipt, state.cache, 1))
        gc.collect()
        self.assertTrue(
            all(owner() is None for owner in old),
            "receipt must not accumulate forty layers of source refs",
        )
        self.assertEqual(receipt.live[0], 1)
        expected = old_output.clone()
        _old_ranges(state.cache, 1, receipt.range_owner, expected, native=False)
        self.assertTrue(_bytes_equal(receipt.output, expected))
        retained = receipt.output
        retained_bytes = retained.clone()
        next_output = torch.full_like(retained, SENTINEL)
        replacement = self.prepare(state, output=next_output)
        self.assertTrue(batch.try_refresh(replacement, state.cache, 1))
        self.assertTrue(_bytes_equal(retained, retained_bytes))
        _compact(state)
        with self.assertRaisesRegex(RuntimeError, "CP SWA batch"):
            batch.try_refresh(replacement, state.cache, 1)


def _world8_worker(rank, store):
    import torch.distributed as dist

    # Do not patch.dict/clear/update libc environ while c10d's heartbeat is
    # alive. Every flag is set before PG init and restored only after destroy.
    previous = {name: os.environ.get(name) for name in FLAGS}
    for name, value in FLAGS.items():
        os.environ[name] = value
    torch.cuda.set_device(rank)
    try:
        dist.init_process_group(
            "nccl", init_method="file://" + store, rank=rank, world_size=8,
            timeout=timedelta(seconds=300),
        )
        with torch.no_grad(), ExitStack() as stack:
            stack.enter_context(patch.object(
                cp_module.collective_torch, "_get_group", return_value=dist.group.WORLD
            ))
            stack.enter_context(patch.object(
                cp_module.collective_torch, "_get_symm_mem",
                return_value=NS(get_symm_mem_communicator=lambda: None),
            ))
            transport = []
            original_gather = dist.all_gather_into_tensor
            original_exchange = dist.all_to_all_single

            def gather(output, value, *args, **kwargs):
                transport.append(("gather", tuple(value.shape), value.dtype))
                return original_gather(output, value, *args, **kwargs)

            def exchange(output, value, *args, **kwargs):
                transport.append(("exchange", tuple(value.shape), value.dtype))
                return original_exchange(output, value, *args, **kwargs)

            stack.enter_context(patch.object(dist, "all_gather_into_tensor", new=gather))
            stack.enter_context(patch.object(dist, "all_to_all_single", new=exchange))
            # Full birth is all-live but requests are ragged. Compact CP140
            # naturally includes padding and a short request, retaining its
            # original mask and host transport metadata.
            state = _birth(
                rank, (256, 272, 128, 512), (129, 17, 0, 257), f"cuda:{rank}"
            )
            workspace = _workspace(state, segmented=True)
            if workspace._swa_batch is None:
                raise AssertionError((rank, "native full workspace declined batch"))

            def compare(layer, current):
                expected = torch.full_like(current.kv, SENTINEL)
                old_sources = {
                    key: (saved.tail.clone(), saved.fresh.clone())
                    for key, saved in state.cache._fresh_swa.items()
                }
                before = len(transport)
                _old_ranges(state.cache, layer, current.swa_ranges, expected, native=True)
                current.kv.fill_(SENTINEL)
                current.refresh_swa(NS(decode_kv=kv_codec.decode_kv), state.cache, layer)
                if len(transport) != before or not _bytes_equal(current.kv, expected):
                    raise AssertionError((rank, layer, "batch changed bytes or collectives"))
                for key, (tail, fresh) in old_sources.items():
                    saved = state.cache._fresh_swa[key]
                    if not _bytes_equal(saved.tail, tail) or not _bytes_equal(saved.fresh, fresh):
                        raise AssertionError((rank, layer, "batch modified a retained raw source"))

            for layer in range(20):
                before = len(transport)
                _stage(state, layer, transport=True)
                if len(transport) != before + 2:
                    raise AssertionError((rank, layer, "full fresh/tail gather count changed"))
                compare(layer, workspace)
                if layer:
                    state.cache.release_swa(layer - 1)
            full_output, frozen_full = workspace.kv, workspace.kv.clone()
            _stage(state, 20, transport=True)
            before = len(transport)
            _compact(state, group=dist.group.WORLD, transport=True)
            if len(transport) != before + 2 or any(
                item[0] != "exchange" for item in transport[before:]
            ):
                raise AssertionError((rank, "compact setup must retain its two real exchanges"))
            l20 = _workspace(state)
            if l20._swa_batch is None:
                raise AssertionError((rank, "compact L20 workspace declined batch"))
            compare(20, l20)
            bounded = _workspace(state, bounded=True)
            if bounded._swa_batch is None:
                raise AssertionError((rank, "compact bounded workspace declined batch"))
            for layer in range(21, 40):
                before = len(transport)
                _stage(state, layer, transport=True, bounded=True)
                if len(transport) != before + 1:
                    raise AssertionError((rank, layer, "bounded fresh gather count changed"))
                compare(layer, bounded)
                state.cache.release_swa(layer - 1)
            if not _bytes_equal(full_output, frozen_full):
                raise AssertionError((rank, "replacement layout overwrote retained full workspace"))

            # A separate disjoint layout preserves Global prefix and gaps even
            # though the source cache fixture itself has no Global pool.
            ranges, rows = _gapped(bounded.swa_ranges)
            actual = torch.full((rows, 512), SENTINEL, dtype=torch.bfloat16,
                                device=state.cache.device)
            expected = actual.clone()
            _old_ranges(state.cache, 39, ranges, expected, native=True)
            receipt = batch.prepare(state, state.context, state.cache, actual, ranges)
            before = len(transport)
            if receipt is None or not batch.try_refresh(receipt, state.cache, 39):
                raise AssertionError((rank, "gapped native batch declined"))
            if len(transport) != before or not _bytes_equal(actual, expected):
                raise AssertionError((rank, "prefix/holes/order or collective count changed"))
            stats = state.cache._cp_swa_batch_stats
            if stats["launches"] != 41 or stats["fallbacks"] != 0:
                raise AssertionError((rank, "unexpected native batch receipt", stats))
            if stats["layers"] != [*range(40), 39]:
                raise AssertionError((rank, "native layer receipt omitted or repeated a layer"))
            if len([item for item in transport if item[0] == "gather"]) != 61:
                raise AssertionError((rank, "overall fresh/tail gather inventory changed"))
            for handle in state.cache.native.handles.values():
                for guard in (handle.canaries[:16], handle.canaries[-16:]):
                    if not torch.equal(guard, torch.full_like(guard, 173)):
                        raise AssertionError((rank, "native pool allocation guard overwritten"))
            torch.cuda.synchronize(rank)
            dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@unittest.skipUnless(
    os.environ.get("RTP_V41_TEST_CP8_WORLD8") == "1",
    "explicit eight-PPU lease required",
)
class CP8SWABatchWorld8(unittest.TestCase):
    def test_all_full_and_compact_layers_match_old_native_ranges_without_new_collectives(self):
        self.assertTrue(_quant.is_ppu())
        with tempfile.TemporaryDirectory(prefix="cp8-swa-batch-") as directory:
            torch.multiprocessing.spawn(
                _world8_worker,
                args=(str(Path(directory) / "store"),),
                nprocs=8,
                join=True,
            )


if __name__ == "__main__":
    unittest.main()
