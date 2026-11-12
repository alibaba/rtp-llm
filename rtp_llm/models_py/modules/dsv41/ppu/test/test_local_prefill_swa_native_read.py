"""Local native SWA tails, original errors and per-forward metadata lifetime.

CPU substitutes only physical PPU admission, pinned upload and the existing
restore/write kernel launchers. Explicit PPU tests run the real kernels.
"""

import gc
import os
import unittest
import weakref
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv41.fp8 import _v41_swa_metadata as metadata
from rtp_llm.models_py.modules.dsv41.ppu import _quant, prefill_swa
from rtp_llm.models_py.modules.dsv41.ppu.native_cache_adapter import (
    LOCAL_PREFILL_SWA_NATIVE_READ_ENV, LOCAL_PREFILL_SWA_NATIVE_WRITE_ENV,
    SharedNativeCache, _LocalSwaWriteGeometry,
)
from rtp_llm.models_py.modules.dsv41.ppu.test.test_local_prefill_fused import assert_bytes
from rtp_llm.models_py.modules.dsv41.ppu.test.test_local_prefill_operator_flags import _trace_events
from rtp_llm.models_py.modules.dsv41.ppu.test.test_local_prefill_swa_native_write import (
    _CPUKernel, fixture,
)
from rtp_llm.models_py.modules.dsv41.ppu.test.test_local_prefill_swa_segments import (
    FLAGS, arguments, compact_plan, stage,
)


ENV = LOCAL_PREFILL_SWA_NATIVE_READ_ENV
WRITE = LOCAL_PREFILL_SWA_NATIVE_WRITE_ENV


class _RestoreCPU:
    def __getitem__(self, grid):
        def launch(packet, plan, out, packet_blocks, span, entries, stride, width, tile):
            start, count, first, output = map(int, plan[0])
            assert packet.stride(0) == stride
            for row in range(count):
                pos = start + row
                block = first + pos // span - start // span
                offset = pos % entries * width
                out[output + row].copy_(packet[block, offset:offset + width])
        return launch


class _ReadCases:
    def setUp(self):
        env = patch.dict(os.environ, {**FLAGS, ENV: "1", WRITE: "1"})
        env.start()
        self.addCleanup(env.stop)

    @torch.no_grad()
    def test_direct_packet_zero_boundary_padding_complete_bytes_decode_and_receipts(self):
        cases = (
            ((0,), (140,), 128, 256), ((1,), (1,), 128, 256),
            ((127,), (0,), 128, 256), ((128,), (4096,), 128, 256),
            ((129,), (4097,), 128, 256), ((257,), (1,), 128, 256),
            ((258,), (140,), 129, 257), ((1024,), (0,), 256, 1024),
        )
        for prefixes, lengths, entries, span in cases:
            arms = []
            for enabled in (False, True):
                with patch.dict(os.environ, {ENV: str(int(enabled))}):
                    state = fixture(self.device, prefixes, lengths, entries=entries, span=span)
                    args = arguments(state)
                    stage(state, args)
                    saved = state.cache._fresh_swa[0, 0]
                    decoded = _quant.dequantize_swa_kv_rows(saved.tail)
                    arms.append((state.cache.native.raw[0].clone(), saved, decoded,
                                 list(state.cache.writes), state.cache))
            assert_bytes(self, arms[0][0], arms[1][0])
            assert_bytes(self, arms[0][1].tail, arms[1][1].tail)
            assert_bytes(self, arms[0][1].fresh, arms[1][1].fresh)
            assert_bytes(self, arms[0][2], arms[1][2])
            self.assertEqual(arms[0][3], arms[1][3])
            stats = arms[1][4]._local_swa_native_read_stats
            self.assertEqual(stats["launches"], int(bool(prefixes[0])))
            self.assertEqual(stats["requests"], int(bool(prefixes[0])))
            self.assertEqual(stats["fallbacks"], 0)
            self.assertEqual(arms[1][4]._local_swa_native_write_stats["mirrors"], 1)
            self.assertEqual(arms[0][4]._local_swa_native_read_stats["launches"], 0)

    @torch.no_grad()
    def test_real_physical_zero_last_nonlogical_and_duplicate_pages(self):
        for kind in ("zero", "last", "duplicate"):
            def configure(table):
                if kind == "last":
                    table[0, 0] = table.numel() - 1
                elif kind == "duplicate":
                    table[0, 1] = table[0, 0]
            arms = []
            for enabled in (False, True):
                with patch.dict(os.environ, {ENV: str(int(enabled))}):
                    state = fixture(self.device, (257,) if kind == "duplicate" else (128,),
                                    (1,), configure=configure)
                    stage(state, arguments(state))
                    arms.append(state)
            assert_bytes(self, arms[0].cache._fresh_swa[0, 0].tail, arms[1].cache._fresh_swa[0, 0].tail)
            self.assertEqual(arms[0].cache.writes, arms[1].cache.writes)
            stats = arms[1].cache._local_swa_native_read_stats
            self.assertEqual(stats["packet"], int(kind == "duplicate"))
            self.assertEqual(stats["direct"], int(kind != "duplicate"))

    @torch.no_grad()
    def test_old_tail_errors_and_prior_request_writes_keep_original_priority(self):
        for kind in ("hole", "negative", "outside", "columns", "second-rid"):
            arms = []
            for enabled in (False, True):
                def configure(table):
                    if kind == "columns":
                        table.resize_(1, 1).zero_()
                    else:
                        rid = int(kind == "second-rid")
                        table[rid, 0] = -1 if kind in ("hole", "second-rid") else -2 if kind == "negative" else table.numel()
                with patch.dict(os.environ, {ENV: str(int(enabled))}):
                    prefixes = (128, 128) if kind == "second-rid" else (512,) if kind == "columns" else (128,)
                    state = fixture(self.device, prefixes, (256,) * len(prefixes), configure=configure, seed=False)
                    with self.assertRaises(ValueError) as error:
                        stage(state, arguments(state))
                    arms.append((str(error.exception), state.cache.native.raw[0].clone(), state.cache.writes))
            self.assertEqual(arms[0][0], arms[1][0], kind)
            self.assertIn("SWA layer 0, request", arms[1][0])
            assert_bytes(self, arms[0][1], arms[1][1])
            self.assertEqual(arms[0][2], arms[1][2])

    @torch.no_grad()
    def test_per_rid_and_cross_layer_alias_reads_previous_native_writes(self):
        def alias(table):
            table[1].copy_(table[0])
        arms = []
        for enabled in (False, True):
            with patch.dict(os.environ, {ENV: str(int(enabled))}):
                state = fixture(self.device, (128, 256), (128, 1), span=1024,
                                configure=alias, cross_layer_alias=True)
                for layer in (0, 1):
                    args = arguments(state, layer=layer)
                    stage(state, args, layer=layer)
                    assert_bytes(self, state.cache._fresh_swa[layer, 1].tail, args[2][1:128])
                arms.append(state)
        assert_bytes(self, arms[0].cache.native.raw[0], arms[1].cache.native.raw[0])
        self.assertEqual(arms[0].cache.writes, arms[1].cache.writes)

    @torch.no_grad()
    def test_twentyone_real_stages_release_metadata_reuse_and_no_extra_mirror(self):
        state = fixture(self.device, (128,), (140,), layers=21)
        cache, tails = state.cache, []
        for layer in range(21):
            stage(state, arguments(state, layer=layer), layer=layer)
            tails.append(cache._fresh_swa[layer, 0].tail.clone())
            cache.release_swa(layer)
            self.assertEqual(len(cache._local_swa_read_plans), 1)
        self.assertFalse(torch.equal(tails[0], tails[-1]))
        self.assertEqual(cache._local_swa_native_read_stats, dict(
            enabled=True, launches=21, requests=21, direct=21, packet=0,
            plan_uploads=1, cache_hits=20, fallbacks=0, closed=False,
        ))
        self.assertEqual(cache._local_swa_native_write_stats["mirrors"], 1)
        self.assertIsNone(cache.prefill_plans)

    @torch.no_grad()
    def test_rows_tables_mirror_mutation_rebind_and_changed_back_never_remint(self):
        for kind in ("positions", "requests", "host", "device", "device-rebind", "meta-rebind",
                     "pool-table", "mirror", "mirror-rebind"):
            state = fixture(self.device, (128,), (140,))
            args = arguments(state)
            stage(state, args)
            cache = state.cache
            with patch.dict(os.environ, {ENV: "0"}):
                meta = cache.inputs["swa_kv"]
                if kind == "positions":
                    args[0].add_(1).sub_(1)
                elif kind == "requests":
                    args[1].add_(1).sub_(1)
                elif kind == "host":
                    meta.kv_cache_block_id.add_(1).sub_(1)
                elif kind == "device":
                    meta.kv_cache_block_id_device.add_(1).sub_(1)
                elif kind == "device-rebind":
                    meta.kv_cache_block_id_device = meta.kv_cache_block_id_device.clone()
                elif kind == "meta-rebind":
                    cache.inputs["swa_kv"] = NS(**vars(meta))
                elif kind == "pool-table":
                    cache.pools[1, "swa_kv"].table = cache.pools[1, "swa_kv"].table.clone()
                elif kind == "mirror-rebind":
                    cache._local_swa_write_mirrors["swa_kv"] = cache._local_swa_write_mirrors["swa_kv"].clone()
                else:
                    cache._local_swa_write_mirrors["swa_kv"].add_(1).sub_(1)
            stage(state, arguments(state, layer=1), layer=1)
            self.assertTrue(cache._local_swa_native_read_stats["closed"], kind)
            self.assertFalse(cache._local_swa_read_plans)
            self.assertEqual(cache._local_swa_native_read_stats["plan_uploads"], 1)

    @torch.no_grad()
    def test_constructor_flags_new_forward_nonce_and_closed_scope(self):
        for env in ({ENV: "0"}, {WRITE: "0"}):
            with patch.dict(os.environ, env):
                state = fixture(self.device, (128,), (140,))
            stage(state, arguments(state))
            self.assertEqual(state.cache._local_swa_native_read_stats["launches"], 0)
        first = fixture(self.device, (128,), (140,))
        second = fixture(self.device, (128,), (140,))
        self.assertIsNot(first.cache._local_swa_read_nonce, second.cache._local_swa_read_nonce)
        stage(first, arguments(first))
        stats = first.cache._local_swa_native_read_stats
        first.cache.close_local_swa_native_reads()
        first.cache.close_local_swa_native_reads()
        self.assertIs(stats, first.cache._local_swa_native_read_stats)
        self.assertTrue(stats["closed"])
        stage(first, arguments(first, layer=1), layer=1)
        self.assertEqual(stats["launches"], 1)
        stage(second, arguments(second))
        self.assertEqual(second.cache._local_swa_native_read_stats["plan_uploads"], 1)

    @torch.no_grad()
    def test_actual_compact_hook_closes_reads_without_releasing_staged_snapshot(self):
        state = fixture(self.device, (128,), (4096,), layers=22, decoder=True)
        stage(state, arguments(state, layer=20), layer=20)
        saved = state.cache._fresh_swa[20, 0]
        before = saved.tail.clone()
        compact_plan(state)
        self.assertTrue(state.cache._local_swa_native_read_stats["closed"])
        self.assertFalse(state.cache._local_swa_read_plans)
        self.assertIs(state.cache._fresh_swa[20, 0], saved)
        assert_bytes(self, saved.tail, before)
        stage(state, arguments(state, compact=True, layer=21), layer=21)
        self.assertEqual(state.cache._local_swa_native_read_stats["requests"], 1)


class LocalSwaNativeReadCPU(_ReadCases, unittest.TestCase):
    def setUp(self):
        self.device = "cpu"
        super().setUp()
        patches = (
            patch.object(prefill_swa, "_write_prefill_ring_range", _CPUKernel()),
            patch.object(prefill_swa, "_restore_tails", _RestoreCPU()),
            patch.object(metadata, "_upload", new=lambda values, device, dtype: torch.tensor(values, device=device, dtype=dtype)),
            patch.object(SharedNativeCache, "_local_swa_native_read_stream", return_value=True),
            patch.object(SharedNativeCache, "_local_swa_native_geometry", new=lambda cache, pool, packed:
                         _LocalSwaWriteGeometry(0, pool.entries, pool.tensor.shape[1])
                         if cache._local_swa_write_payload_eligible(pool, packed) else None),
        )
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    @torch.no_grad()
    def test_close_releases_first_metadata_view_but_never_retains_payload_or_guard(self):
        state = fixture("cpu", (128,), (140,))
        args = arguments(state)
        row_ref, packed_ref, guard_ref = weakref.ref(args[0]), weakref.ref(args[2]), weakref.ref(args[-1].guard)
        stage(state, args)
        saved, stats = state.cache._fresh_swa[0, 0], state.cache._local_swa_native_read_stats
        del args
        gc.collect()
        self.assertIsNotNone(row_ref())
        self.assertIsNone(packed_ref())
        self.assertIsNone(guard_ref())
        before = saved.tail.clone()
        state.cache.close_local_swa_native_reads()
        gc.collect()
        self.assertIsNone(row_ref())
        self.assertFalse(state.cache._local_swa_read_mirrors)
        self.assertTrue(stats["closed"])
        assert_bytes(self, saved.tail, before)

    @torch.no_grad()
    def test_nondefault_stream_and_raw_view_gate_decline_before_any_upload(self):
        state = fixture("cpu", (128,), (140,))
        with patch.object(SharedNativeCache, "_local_swa_native_read_stream", return_value=False), \
             patch.object(metadata, "_upload", side_effect=AssertionError("stream fallback must not upload")):
            stage(state, arguments(state))
        self.assertEqual(state.cache._local_swa_native_read_stats["launches"], 0)
        self.assertEqual(state.cache._local_swa_native_read_stats["fallbacks"], 1)
        state = fixture("cpu", (128,), (140,))
        args = arguments(state)
        cache, pool = state.cache, state.cache.pools[0, "swa_kv"]
        mirror = cache._verified_local_swa_write_table(pool)
        geometry = _LocalSwaWriteGeometry(0, pool.entries, pool.tensor.shape[1])
        original = pool.tensor
        pool.tensor = torch._neg_view(original)
        with patch.object(metadata, "_upload", side_effect=AssertionError("raw view upload")):
            self.assertIsNone(cache._try_local_swa_native_tail(0, 0, 128, args[-1], (geometry, mirror)))
        pool.tensor = original

    def test_close_partial_legacy_fixture_is_pure_host_and_idempotent(self):
        partial = SharedNativeCache.__new__(SharedNativeCache)
        marker = object()
        partial._fresh_swa = {(0, 0): marker}
        partial.close_local_swa_native_reads()
        partial.close_local_swa_native_reads()
        self.assertIs(partial._fresh_swa[0, 0], marker)

    @torch.inference_mode()
    def test_readonly_inference_metadata_reuses_but_new_forward_owns_fresh_scope(self):
        state = fixture("cpu", (128,), (140,))
        stage(state, arguments(state))
        stage(state, arguments(state, layer=1), layer=1)
        self.assertEqual(state.cache._local_swa_native_read_stats["cache_hits"], 1)
        newer = fixture("cpu", (129,), (140,))
        stage(newer, arguments(newer))
        self.assertEqual(newer.cache._local_swa_native_read_stats["plan_uploads"], 1)


@unittest.skipUnless(os.environ.get("RTP_V41_TEST_LOCAL_WORKSPACE_DEVICE") == "1", "explicit PPU device test")
class LocalSwaNativeReadDevice(_ReadCases, unittest.TestCase):
    def setUp(self):
        self.assertTrue(_quant.is_ppu())
        self.assertEqual(torch.cuda.get_device_capability(), (8, 9))
        self.device = "cuda:0"
        super().setUp()

    @torch.no_grad()
    def test_actual_nondefault_stream_rolls_back_without_plan_upload(self):
        state = fixture(self.device, (128,), (140,))
        args = arguments(state)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side), patch.object(metadata, "_upload", side_effect=AssertionError("side stream upload")):
            stage(state, args)
        torch.cuda.current_stream().wait_stream(side)
        self.assertEqual(state.cache._local_swa_native_read_stats["launches"], 0)
        self.assertEqual(state.cache._local_swa_native_read_stats["fallbacks"], 1)

    @torch.no_grad()
    def test_real_pinned_upload_gc_pressure_and_every_read_sees_postwrite_pool(self):
        state = fixture(self.device, (128,), (140,))
        cache = state.cache
        # Finish the one table mirror before queuing work: it must not drain
        # the stream between the backlog and this test's first plan upload.
        self.assertIsNotNone(cache._verified_local_swa_write_table(cache.pools[0, "swa_kv"]))
        backlog = torch.empty(4 * 1024 * 1024, dtype=torch.float32, device=self.device)
        for step in range(5):
            expected = cache._copy_rows((0, "swa_kv"), 0, torch.arange(1, 128, device=self.device))
            args = arguments(state)
            for value in range(16):
                backlog.fill_(step + value)
            stage(state, args)
            # No CPU tensor observation/synchronize before GC and allocator
            # pressure. The first stage has just uploaded the pinned plan.
            gc.collect()
            for _ in range(32):
                torch.empty(4, dtype=torch.int64, pin_memory=True).fill_(step + 9000)
                torch.empty(16, dtype=torch.int64, device=self.device).fill_(step + 7000)
            assert_bytes(self, cache._fresh_swa[0, 0].tail, expected)
            cache.release_swa(0)
        self.assertEqual(cache._local_swa_native_read_stats["plan_uploads"], 1)
        self.assertEqual(cache._local_swa_native_read_stats["cache_hits"], 4)
        self.assertEqual(cache._local_swa_native_write_stats["mirrors"], 1)

    @torch.no_grad()
    def test_actual_twentyone_restore_kernels_no_tail_scalar_fences_and_exact_pools(self):
        arms = []
        for enabled in (False, True):
            with patch.dict(os.environ, {ENV: str(int(enabled))}):
                state = fixture(self.device, (128,), (140,), layers=21)
                args = [arguments(state, layer=layer) for layer in range(21)]
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                        torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as profile:
                    for layer in range(21):
                        stage(state, args[layer], layer=layer)
                    torch.cuda.synchronize()
                events = _trace_events(profile, f"local-swa-native-read-{int(enabled)}")
                kernels = [e["name"] for e in events if e.get("cat") == "kernel"]
                self.assertEqual(sum("_restore_tails" in name for name in kernels), 21 * int(enabled))
                self.assertFalse(any("pccl" in name.lower() or "allgather" in name.lower() for name in kernels))
                checks = [e for e in events if e.get("cat") == "cpu_op" and e.get("name") in ("aten::min", "aten::max")
                          and e.get("args", {}).get("Input Dims") == [[127]]]
                self.assertEqual(len(checks), 0 if enabled else 105)
                arms.append((tuple(raw.clone() for raw in state.cache.native.raw.values()),
                             tuple(saved.tail.clone() for saved in state.cache._fresh_swa.values()),
                             state.cache.writes, state.cache))
        for old, got in zip(arms[0][0], arms[1][0]):
            assert_bytes(self, old, got)
        for old, got in zip(arms[0][1], arms[1][1]):
            assert_bytes(self, old, got)
        self.assertEqual(arms[0][2], arms[1][2])
        self.assertEqual(arms[1][3]._local_swa_native_read_stats["plan_uploads"], 1)
        self.assertEqual(arms[1][3]._local_swa_native_read_stats["cache_hits"], 20)
        self.assertEqual(arms[1][3]._local_swa_native_write_stats["mirrors"], 1)


if __name__ == "__main__":
    unittest.main()
