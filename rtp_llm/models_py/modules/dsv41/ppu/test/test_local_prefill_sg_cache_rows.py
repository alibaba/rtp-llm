"""SG-shaped split bank readback and creation-proved local B1 request slices.

CPU launch doubles prove addressing/protocol only. Explicit PPU tests execute
the existing restore kernel; no fake CP group or modified scorer is used.
"""

import os
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv41 import cp
from rtp_llm.models_py.modules.dsv41.ppu import _quant, prefill_readback
from rtp_llm.models_py.modules.dsv41.ppu.native_cache_adapter import (
    LOCAL_PREFILL_SG_ALIGN_ENV as ENV, SharedNativeCache,
)
from rtp_llm.models_py.modules.dsv41.ppu.prefill_runtime import SharedForwardState
from rtp_llm.models_py.modules.dsv41.ppu.replay_policy import SharedPrefillPolicy
from rtp_llm.models_py.modules.dsv41.ppu.test.test_local_prefill_fused import assert_bytes
from rtp_llm.models_py.modules.dsv41.ppu.test.test_local_prefill_operator_flags import _trace_events
from rtp_llm.models_py.modules.dsv41.ppu.test.test_local_prefill_source_native_read import fixture, original
from rtp_llm.models_py.modules.dsv41.ppu.test.test_local_prefill_swa_segments import FLAGS, compact_plan


class _RestoreCPU:
    def __getitem__(self, grid):
        def launch(packet, plan, out, scale, stripe_count, owner_rows, page_rows,
                   cp_size, block_bytes, page_bytes, entries, width, planar, split, tile):
            assert cp_size == 1 and stripe_count == 1 and owner_rows == page_rows
            count, start = int(plan[0, 2]), int(plan[0, 3])
            row, column = torch.arange(count), torch.arange(width)[None]
            entry = row % page_rows
            address = (torch.where(column < 64, entry[:, None] * 64 + column,
                                   entries * 64 + entry[:, None] * 4 + column - 64)
                       if planar else entry[:, None] * width + column)
            result = packet[row[:, None] // page_rows, address]
            if split:
                assert planar and width == 68
                out[start:start + count].copy_(result[:, :64])
                scale[start:start + count].copy_(result[:, 64:])
            else:
                out[start:start + count].copy_(result)
        return launch


def _state(device, prefix=126976, count=4096, *, model_changes=None, **kwargs):
    cache = fixture(device, (prefix + count,), ratios={20: 1})
    model = NS(v4=NS(layers=[]), _is_decode_role=False, _is_speculative=False,
               _prefill_cp_size=1, parallelism_config=NS(tp_size=1))
    for key, value in (model_changes or {}).items():
        setattr(model, key, value)
    runtime = NS(model=model, policy=SharedPrefillPolicy(mode="compact"))
    inputs = NS(input_ids=torch.arange(count, device=device, dtype=torch.int64))
    attn = NS(cu_seqlens=torch.tensor((0, count), dtype=torch.int32), is_cuda_graph=False)
    for key, value in kwargs.pop("attn_changes", {}).items():
        setattr(attn, key, value)
    return SharedForwardState(runtime, inputs, attn, cache, prefill=kwargs.pop("prefill", True),
                              idle=kwargs.pop("idle", False), prefixes=(prefix,), lengths=(count,))


class _Cases:
    def setUp(self):
        env = patch.dict(os.environ, {**FLAGS, ENV: "1", "TP_SIZE": "1", "PREFILL_CP_SIZE": "1",
                                      "RTP_V41_PPU_LOCAL_PREFILL_SOURCE_CACHE": "1",
                                      "RTP_V41_PPU_LOCAL_PREFILL_SOURCE_NATIVE_READ": "1"})
        env.start()
        self.addCleanup(env.stop)

    @torch.no_grad()
    def test_split_bytes_reuse_packed_compatibility_and_true_planar_geometry(self):
        for ratio in (1, 2):
            for count in (0, 1, 127, 128, 129, 267, 4096, 4097):
                cache = fixture(self.device, (max(count * ratio, 1),), ratios={2: ratio})
                expected = original(cache, 2, count=count)
                payload, scales = cache.read_indexer_fp4(2, 0, global_entries=count)
                assert_bytes(self, payload.view(torch.uint8), expected[1][:, :64].contiguous())
                assert_bytes(self, scales.view(torch.uint8).reshape(count, 4), expected[1][:, 64:].contiguous())
                self.assertEqual((payload.dtype, scales.dtype), (torch.int8, torch.int32))
                self.assertTrue(payload.is_contiguous() and scales.is_contiguous())
                for _ in range(4):
                    p2, s2 = cache.read_indexer_fp4(2, 0, global_entries=count)
                    assert_bytes(self, p2, payload)
                    assert_bytes(self, s2, scales)
                    if count:
                        self.assertEqual((p2.data_ptr(), s2.data_ptr()), (payload.data_ptr(), scales.data_ptr()))
                packed = cache.read_source(2, 0, global_entries=count)
                for got, old in zip(packed, expected):
                    assert_bytes(self, got, old)
                if count:
                    self.assertEqual(cache._local_sg_source_stats["split_reads"], 1)
                    self.assertEqual(cache._local_sg_source_stats["derived_splits"], 0)
                    self.assertEqual(cache._local_source_native_read_stats["reads"], 2)
                    self.assertEqual(payload.shape, (count, 64))
                    self.assertEqual(cache._local_sg_source_stats["packed_rebuilds"], 1)

    @torch.no_grad()
    def test_raw_write_bank_mutation_and_partial_growth_invalidate_before_reuse(self):
        cache = fixture(self.device, (1024,), ratios={2: 2})
        p0, s0 = cache.read_indexer_fp4(2, 0, global_entries=32)
        cache.read_indexer_fp4(2, 0, global_entries=512)
        self.assertEqual(cache._local_sg_source_stats["split_reads"], 2)
        payload, scales = cache.read_indexer_fp4(2, 0, global_entries=512)
        payload.add_(0)  # Ordinary output mutation must not bless a stale hit.
        cache.read_indexer_fp4(2, 0, global_entries=512)
        self.assertEqual(cache._local_sg_source_stats["split_reads"], 3)
        rows = torch.full((1, 68), 29, device=self.device, dtype=torch.uint8)
        cache._copy_rows((2, "indexer_kv"), 0, torch.tensor([0], device=self.device), rows)
        self.assertFalse(cache._source_reads or cache._local_source_reads)
        p1, s1 = cache.read_indexer_fp4(2, 0, global_entries=512)
        assert_bytes(self, p1[:1].view(torch.uint8), rows[:, :64])
        assert_bytes(self, s1[:1].view(torch.uint8).reshape(1, 4), rows[:, 64:].contiguous())
        self.assertEqual(cache._local_sg_source_stats["split_reads"], 4)
        self.assertFalse(torch._C._is_alias_of(p1, cache.pools[2, "indexer_kv"].tensor))

    @torch.no_grad()
    def test_first_read_main_error_priority_inactive_holes_and_alias_source_writes(self):
        for bad in ("main", "indexer", "both"):
            def configure(tag, table, blocks):
                if tag == "indexer_kv" and bad in ("indexer", "both") or tag == "csa_kv" and bad in ("main", "both"):
                    table[0, 0] = -1
                table[1].fill_(-1)
            cache = fixture(self.device, (1024, 512), ratios={2: 2}, configure=configure)
            with self.assertRaises(ValueError) as old:
                original(cache, 2)
            with self.assertRaises(ValueError) as new:
                cache.read_indexer_fp4(2, 0, global_entries=512)
            self.assertEqual(str(old.exception), str(new.exception))
            self.assertFalse(cache._source_reads)
        cache = fixture(self.device, (1024, 1024), ratios={2: 1, 20: 1}, alias=True,
                        configure=lambda tag, table, blocks: table[1].copy_(table[0]))
        for source in (2, 20):
            for rid in (0, 1):
                cache.read_indexer_fp4(source, rid, global_entries=1024)
        changed = torch.full((1, 68), 91, device=self.device, dtype=torch.uint8)
        cache._copy_rows((20, "indexer_kv"), 1, torch.tensor([0], device=self.device), changed)
        self.assertFalse(cache._source_reads)
        payload, scales = cache.read_indexer_fp4(2, 0, global_entries=1024)
        assert_bytes(self, payload[:1].view(torch.uint8), changed[:, :64])

    @torch.no_grad()
    def test_real_full_and_compact_rows_slices_keep_order_and_padded_rows(self):
        for count in (4096, 4097, 140, 139, 5, 0):
            state = _state(self.device, 256, count)
            self.assertTrue(state.local_prefill_sg_align)
            full = state.rows(None)
            self.assertEqual(state.local_indexer_request_slices(None, full, count), (slice(0, count),))
            self.assertIs(state.rows(None), full)
            self.assertEqual(state.local_indexer_request_slices(None, tuple(list(full)), count), (slice(0, count),))
            self.assertIsNone(state.local_indexer_request_slices(None, (full[0].clone(), *full[1:]), count))
            if count:
                context = compact_plan(state)
                compact = state.rows(context)
                rows = context.chunk_length
                self.assertEqual(state.local_indexer_request_slices(context, compact, rows), (slice(0, rows),))
                ids = torch.where(compact[1] == 0)[0]
                assert_bytes(self, compact[0][ids], compact[0][:rows])
                self.assertIsNone(state.local_indexer_request_slices(context, compact, rows + 1))


class LocalSgCacheRowsCPU(_Cases, unittest.TestCase):
    device = "cpu"

    def setUp(self):
        super().setUp()
        for item in (
            patch.object(SharedNativeCache, "_local_source_native_read_device", new=lambda self, pool: True),
            patch.object(prefill_readback, "_restore_rows", _RestoreCPU()),
            patch("rtp_llm.models_py.modules.dsv41.fp8._v41_swa_metadata._upload",
                  new=lambda rows, device, dtype: torch.tensor(rows, dtype=dtype, device=device)),
        ):
            item.start()
            self.addCleanup(item.stop)

    @torch.no_grad()
    def test_constructor_latch_phase_missing_birth_and_no_cache_hit_recertification(self):
        with patch.dict(os.environ, {ENV: "0"}):
            state = _state("cpu", 256, 140)
        rows = state.rows()
        self.assertFalse(state.local_prefill_sg_align)
        self.assertIsNone(state.local_indexer_request_slices(None, rows, 140))
        for kwargs in (dict(prefill=False), dict(idle=True),
                       dict(model_changes={"_is_decode_role": True}),
                       dict(model_changes={"_is_speculative": True}),
                       dict(model_changes={"parallelism_config": NS(tp_size=8)}),
                       dict(attn_changes={"is_cuda_graph": True}), dict(attn_changes={"is_target_verify": True}),
                       dict(attn_changes={"proposal": True})):
            self.assertFalse(_state("cpu", 256, 140, **kwargs).local_prefill_sg_align)
        state = _state("cpu", 256, 140)
        rows = state.rows()
        state._sg_rows_stamps.clear()
        self.assertIs(state.rows(), rows)
        self.assertIsNone(state.local_indexer_request_slices(None, rows, 140))

    @torch.no_grad()
    def test_rows_source_derived_context_epoch_and_plan_versions_fail_sticky(self):
        for change in (lambda s, rows: rows[0].add_(1).sub_(1),
                       lambda s, rows: rows[1].add_(0), lambda s, rows: rows[2].add_(0),
                       lambda s, rows: rows[3].logical_not_().logical_not_(),
                       lambda s, rows: s.attn.cu_seqlens.add_(0),
                       lambda s, rows: setattr(s, "_sg_rows_epoch", object())):
            state = _state("cpu", 256, 140)
            rows = state.rows()
            change(state, rows)
            self.assertIsNone(state.local_indexer_request_slices(None, rows, 140))
            self.assertTrue(state._sg_rows_invalid)
        state = _state("cpu", 256, 4096)
        state.rows()
        context = compact_plan(state)
        rows = state.rows(context)
        rows[1].add_(0)  # Derived .long() is separate from context.req_id.
        self.assertIsNone(state.local_indexer_request_slices(context, rows, 140))
        state = _state("cpu", 256, 4096)
        state.rows()
        context = compact_plan(state)
        context.global_positions.add_(0)  # Mutation before first derived rows.
        rows = state.rows(context)
        self.assertIsNone(state.local_indexer_request_slices(context, rows, 140))

    @torch.no_grad()
    def test_no_gpu_value_download_in_slice_guard_and_split_fallback_bytes(self):
        state = _state("cpu", 256, 4096)
        rows = state.rows()
        with patch.object(torch.Tensor, "item", side_effect=AssertionError("metadata only")), \
                patch.object(torch.Tensor, "tolist", side_effect=AssertionError("metadata only")), \
                patch.object(torch.Tensor, "cpu", side_effect=AssertionError("metadata only")):
            self.assertEqual(state.local_indexer_request_slices(None, rows, 4096), (slice(0, 4096),))
        # Existing native read can decline a geometry; old packed read is still
        # validated first, then split once and retained under the same bank key.
        cache = fixture("cpu", (1024,), ratios={2: 2})
        with patch.object(cache, "_try_read_local_source_pool", return_value=None):
            expected = original(cache, 2)
            payload, scales = cache.read_indexer_fp4(2, 0, global_entries=512)
            assert_bytes(self, payload.view(torch.uint8), expected[1][:, :64].contiguous())
            cache.read_indexer_fp4(2, 0, global_entries=512)
        self.assertEqual(cache._local_sg_source_stats["derived_splits"], 1)


@unittest.skipUnless(os.environ.get("RTP_V41_TEST_LOCAL_WORKSPACE_DEVICE") == "1", "explicit real PPU test")
class LocalSgCacheRowsDevice(_Cases, unittest.TestCase):
    device = "cuda:0"

    def setUp(self):
        super().setUp()
        self.assertTrue(_quant.is_ppu())
        self.assertEqual(torch.cuda.get_device_capability(), (8, 9))

    @torch.no_grad()
    def test_actual_four_native_split_banks_then_late_reuse_no_repacking_profile(self):
        cache = fixture(self.device, (131072,), ratios={2: 2, 8: 2, 14: 2, 20: 1})
        # Compile the split specialization without filling this cache's banks.
        warm = fixture(self.device, (1024,), ratios={2: 2})
        warm.read_indexer_fp4(2, 0, global_entries=512)
        with patch.object(cp.collective_torch, "_get_group", side_effect=AssertionError("no CP group")), \
                torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                   torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as profile:
            for source in (2, 8, 14, 20):
                cache.read_indexer_fp4(source, 0, global_entries=131072 // cache.geometry.ratio(source))
            for _ in (24, 28, 32, 36):
                cache.read_indexer_fp4(20, 0, global_entries=131072)
        events = _trace_events(profile, "sg-four-split-indexer-banks")
        kernels = [e for e in events if e.get("cat") == "kernel"]
        self.assertEqual(sum("_restore_rows" in e.get("name", "") for e in kernels), 8)
        self.assertEqual(cache._local_sg_source_stats["split_reads"], 4)
        self.assertEqual(cache._local_sg_source_stats["hits"], 4)
        self.assertEqual(cache._local_sg_source_stats["derived_splits"], 0)
        self.assertEqual(cache._local_sg_source_stats["packed_rebuilds"], 0)
        for event in events:
            shapes = event.get("args", {}).get("Input Dims", [])
            if event.get("cat") == "cpu_op" and shapes and shapes[0] in (
                [65536, 64], [65536, 4], [131072, 64], [131072, 4],
            ):
                self.assertNotIn(event.get("name"), ("aten::contiguous", "aten::clone"))
        for source in (2, 20):
            payload, scales = cache.read_indexer_fp4(source, 0, global_entries=131072 // cache.geometry.ratio(source))
            expected = original(cache, source)[1]
            assert_bytes(self, payload.view(torch.uint8), expected[:, :64].contiguous())
            assert_bytes(self, scales.view(torch.uint8).reshape(-1, 4), expected[:, 64:].contiguous())


if __name__ == "__main__":
    unittest.main()
