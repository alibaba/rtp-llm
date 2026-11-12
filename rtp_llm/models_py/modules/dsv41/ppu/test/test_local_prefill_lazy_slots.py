"""Lazy local producer-slot lifetime, against unchanged _build_producer_slots.

CPU substitutes only physical PPU admission and existing native primitive
launches. Explicit PPU cases profile the actual constructor, native writer and
reader. Original explicit defer and CP8 behavior remain independent.
"""

import json
import os
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv41.ppu import _quant, graph_cache_kernels, prefill, prefill_readback
from rtp_llm.models_py.modules.dsv41.ppu.kv_codec import encode_kv
from rtp_llm.models_py.modules.dsv41.ppu.native_cache_adapter import (
    LOCAL_PREFILL_LAZY_PRODUCER_SLOTS_ENV as ENV,
    LOCAL_PREFILL_SOURCE_NATIVE_READ_ENV as READ,
    LOCAL_PREFILL_SOURCE_NATIVE_WRITE_ENV as WRITE,
    LOCAL_PREFILL_SWA_NATIVE_WRITE_ENV as SWA,
    SharedNativeCache,
)
from rtp_llm.models_py.modules.dsv41.ppu.ops import PPUOps
from rtp_llm.models_py.modules.dsv41.ppu.test.test_local_prefill_fused import assert_bytes
from rtp_llm.models_py.modules.dsv41.ppu.test.test_local_prefill_operator_flags import _trace_events
from rtp_llm.models_py.modules.dsv41.ppu.test.test_local_prefill_rotary_pack import _fixture as producer_fixture, _produce
from rtp_llm.models_py.modules.dsv41.ppu.test.test_local_prefill_source_native_read import (
    _RestoreCPU, fixture as full_fixture, original, read,
)
from rtp_llm.models_py.modules.dsv41.ppu.test.test_local_prefill_source_native_write import _StoreCPU
from rtp_llm.models_py.modules.dsv41.ppu.test.test_local_prefill_swa_segments import FLAGS, arguments


def _rebuild(cache, *, defer=False, prefill=True):
    return SharedNativeCache(cache.config, cache.geometry, cache.native, cache.inputs, cache.lengths,
                              device=cache.device, prefill=prefill, defer_producer_slots=defer)


def _slots_equal(test, got, expected):
    test.assertEqual(tuple(got), tuple(expected))
    for rid in expected:
        test.assertIs(got[rid].global_slots, got[rid].indexer_slots)
        for attr in ("global_slots", "indexer_slots"):
            test.assertEqual(tuple(getattr(got[rid], attr)), tuple(getattr(expected[rid], attr)))
            for source, value in getattr(expected[rid], attr).items():
                actual = getattr(got[rid], attr)[source]
                assert_bytes(test, actual, value)
                test.assertEqual(actual.dtype, torch.int64)
                test.assertEqual(actual.stride(), value.stride())


def _producer(device, enabled, *, mutate=False):
    with patch.dict(os.environ, {ENV: str(int(enabled))}):
        f = producer_fixture(device, 2, (128,), (140,))
        cache = _rebuild(f.source.cache)
        f.source.cache = f.state.cache = cache
        f.args = arguments(f.state)
        f.positions, f.requests = f.args[:2]
        f.proof = prefill._make_local_producer_rotary_proof(
            f.args[-1], cache=cache, layer_id=0, prefixes=f.source.prefixes, lengths=f.source.fresh,
        )
        if mutate:
            cache.inputs[cache.pools[0, "global_kv_2"].framework_tag].kv_cache_kernel_block_id_device.add_(0)
        return f


class _Cases:
    def setUp(self):
        env = patch.dict(os.environ, {**FLAGS, ENV: "1", READ: "1", WRITE: "1", SWA: "1",
                                      "TP_SIZE": "1", "PREFILL_CP_SIZE": "1",
                                      prefill.LOCAL_PREFILL_ROTARY_PACK_ENV: "1",
                                      "RTP_V41_PPU_LOCAL_PREFILL_CSA_NATIVE_WRITE": "0"})
        env.start()
        self.addCleanup(env.stop)

    @torch.no_grad()
    def test_empty_ragged_slots_exact_and_materialized_once(self):
        for totals in ((0,), (31, 0, 256), (0, 0)):
            base = full_fixture(self.device, totals)
            with patch.dict(os.environ, {ENV: "0"}):
                old = _rebuild(base)
            new = _rebuild(base)
            self.assertEqual(old._local_producer_slot_stats["materializations"], 1)
            deferred = any(totals)
            self.assertEqual(new._local_producer_slot_stats["deferred"], deferred)
            self.assertEqual(new._slots is None, deferred)
            expected = old.slots
            actual = new.slots
            _slots_equal(self, actual, expected)
            self.assertIs(new.slots, actual)
            self.assertEqual(new._local_producer_slot_stats["materializations"], 1)
            json.dumps(new._local_producer_slot_stats)

    @torch.no_grad()
    def test_late_flag_explicit_defer_and_close_preserve_original_property(self):
        base = full_fixture(self.device)
        with patch.dict(os.environ, {ENV: "0"}):
            eager = _rebuild(base)
            explicit = _rebuild(base, defer=True)
        self.assertIsNotNone(eager._slots)
        self.assertIsNone(explicit._slots)
        self.assertEqual(explicit._local_producer_slot_stats,
                         dict(enabled=False, deferred=True, materializations=0, closed=False))
        eager_result = eager.slots
        self.assertIs(eager.slots, eager_result, "late flag1 does not undo eager construction")
        lazy = _rebuild(base)
        self.assertIsNone(lazy._slots)
        stats = lazy._local_producer_slot_stats
        lazy.close_local_source_native_reads()
        lazy.close_local_source_native_reads()
        self.assertTrue(stats["closed"])
        with patch.dict(os.environ, {ENV: "0"}):
            result = lazy.slots
        _slots_equal(self, result, eager_result)
        self.assertIs(lazy.slots, result)
        self.assertEqual(stats["materializations"], 1)
        _slots_equal(self, explicit.slots, eager_result)
        self.assertEqual(explicit._local_producer_slot_stats["materializations"], 1)
        SharedNativeCache.__new__(SharedNativeCache).close_local_source_native_reads()

    @torch.no_grad()
    def test_normal_table_mutation_keeps_exact_old_fallback_and_native_byte_contract(self):
        ops = PPUOps(num_routed_experts=8, num_local_experts=8)
        for mutation in (False, True):
            results = []
            for enabled in (False, True):
                f = _producer(self.device, enabled, mutate=mutation)
                self.assertIsNotNone(f.proof)
                self.assertEqual(f.source.cache._slots is None, enabled)
                _produce(f, ops)
                results.append((f, [pool.tensor.clone() for pool in f.source.cache.pools.values()]))
            self.assertEqual(results[0][0].source.cache.writes, results[1][0].source.cache.writes)
            for old, new in zip(results[0][1], results[1][1]):
                assert_bytes(self, old, new)
            candidate = results[1][0].source.cache
            self.assertEqual(candidate._local_producer_slot_stats["materializations"], int(mutation))
            self.assertEqual(candidate._local_source_native_write_stats["writes"], 1 if mutation else 2)
            for got, expected in zip(read(candidate, 0), original(results[0][0].source.cache, 0)):
                assert_bytes(self, got, expected)


class LocalLazyProducerSlotsCPU(_Cases, unittest.TestCase):
    device = "cpu"

    def setUp(self):
        super().setUp()
        for item in (
            patch.object(SharedNativeCache, "_local_source_native_read_device", new=lambda self, pool: True),
            patch.object(graph_cache_kernels, "_kernels", new=lambda: NS(
                store_cp_prefill_rows=_StoreCPU(), triton=NS(next_power_of_2=lambda n: 1 << (n - 1).bit_length()))),
            patch.object(prefill_readback, "_restore_rows", _RestoreCPU()),
            patch("rtp_llm.models_py.modules.dsv41.fp8._v41_swa_metadata._upload",
                  new=lambda rows, device, dtype: torch.tensor(rows, dtype=dtype, device=device)),
            patch.object(_quant, "try_pack_rotary_fp4_rows", new=lambda kv, freqs, positions, global_pool:
                         encode_kv(prefill._rope_tail(kv, freqs[positions.clamp_max(freqs.shape[0] - 1)]),
                                   pool_tag="global_kv_1" if global_pool else "indexer_kv")),
        ):
            item.start()
            self.addCleanup(item.stop)

    @torch.no_grad()
    def test_phase_flags_device_and_missing_birth_fall_back_eager(self):
        base = full_fixture("cpu")
        for env in ({ENV: "0"}, {READ: "0"}, {WRITE: "0"}, {SWA: "0"},
                    {"TP_SIZE": "8"}, {"PREFILL_CP_SIZE": "8"}):
            with patch.dict(os.environ, env):
                cache = _rebuild(base)
                self.assertIsNotNone(cache._slots)
                self.assertEqual(cache._local_producer_slot_stats["materializations"], 1)
        self.assertFalse(_rebuild(base, prefill=False)._local_producer_slot_stats["deferred"])
        with torch.enable_grad():
            self.assertFalse(_rebuild(base)._local_producer_slot_stats["deferred"])
        with patch.object(SharedNativeCache, "_local_source_native_read_device", new=lambda self, pool: False):
            self.assertFalse(_rebuild(base)._local_producer_slot_stats["deferred"])
        tag = base.pools[2, "global_kv_2"].framework_tag
        meta = base.inputs[tag]
        original_host = meta.kv_cache_kernel_block_id
        meta.kv_cache_kernel_block_id = torch._neg_view(original_host)
        declined = _rebuild(base)
        self.assertFalse(declined._local_producer_slot_stats["deferred"])
        meta.kv_cache_kernel_block_id = original_host
        meta.is_cuda_graph = True
        self.assertFalse(_rebuild(base)._local_producer_slot_stats["deferred"])

    @torch.no_grad()
    def test_negative_lengths_keep_original_constructor_error(self):
        base = full_fixture("cpu", (-1,))
        errors = []
        for flag in ("0", "1"):
            with patch.dict(os.environ, {ENV: flag}), self.assertRaises(RuntimeError) as error:
                _rebuild(base)
            errors.append(str(error.exception))
        self.assertEqual(errors[0], errors[1])

    @torch.no_grad()
    def test_cp8_old_lazy_decision_is_unchanged_and_builder_is_exact(self):
        base = full_fixture("cpu")
        # Only observe the pre-existing branch decision: constructor is still
        # local and creates no fake placement or distributed group.
        with patch.object(SharedNativeCache, "cp8_row_prep_enabled", return_value=True), \
                patch.dict(os.environ, {ENV: "0", "RTP_V41_PPU_CP8_LAZY_PRODUCER_SLOTS": "1"}):
            cache = _rebuild(base)
            self.assertIsNone(cache._slots)
            expected = cache._build_producer_slots()
            _slots_equal(self, cache.slots, expected)
            self.assertEqual(cache._local_producer_slot_stats["materializations"], 1)


@unittest.skipUnless(os.environ.get("RTP_V41_TEST_LOCAL_WORKSPACE_DEVICE") == "1", "explicit real PPU test")
class LocalLazyProducerSlotsDevice(_Cases, unittest.TestCase):
    device = "cuda:0"

    def setUp(self):
        super().setUp()
        self.assertTrue(_quant.is_ppu(), "requires an actual PPU")
        self.assertEqual(torch.cuda.get_device_capability(), (8, 9))

    @torch.no_grad()
    def test_actual_constructor_four_full_source_metadata_chains_absent_until_property(self):
        base = full_fixture(self.device, (131072,), ratios={2: 2, 8: 2, 14: 2, 20: 1})
        profiles = []
        caches = []
        for enabled in (False, True):
            with patch.dict(os.environ, {ENV: str(int(enabled))}), torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], record_shapes=True,
            ) as profile:
                cache = _rebuild(base)
            profiles.append(_trace_events(profile, "lazy-producer-slots-" + str(int(enabled))))
            caches.append(cache)
        for name in ("aten::arange", "aten::mul", "aten::stack"):
            old = [e for e in profiles[0] if e.get("cat") == "cpu_op" and e.get("name") == name]
            new = [e for e in profiles[1] if e.get("cat") == "cpu_op" and e.get("name") == name]
            self.assertGreaterEqual(len(old), 4, name)
            self.assertEqual(new, [], name)
        self.assertIsNone(caches[1]._slots)
        self.assertEqual(caches[1]._local_producer_slot_stats["materializations"], 0)
        _slots_equal(self, caches[1].slots, caches[0].slots)
        self.assertEqual(caches[1]._local_producer_slot_stats["materializations"], 1)

    @torch.no_grad()
    def test_actual_nondefault_stream_capture_and_fresh_forward(self):
        base = full_fixture(self.device)
        torch.cuda.synchronize()
        with torch.cuda.stream(torch.cuda.Stream()):
            self.assertFalse(_rebuild(base)._local_producer_slot_stats["deferred"])
        torch.cuda.synchronize()
        with patch.object(torch.cuda, "is_current_stream_capturing", return_value=True):
            self.assertFalse(_rebuild(base)._local_producer_slot_stats["deferred"])
        fresh = _rebuild(base)
        self.assertTrue(fresh._local_producer_slot_stats["deferred"])
        fresh.close_local_source_native_reads()
        self.assertTrue(fresh._local_producer_slot_stats["closed"])


if __name__ == "__main__":
    unittest.main()
