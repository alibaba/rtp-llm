"""Exact local native-pool SWA range writes; no model weights are loaded.

CPU emulates only the existing kernel launcher for protocol/byte contracts.
Explicit PPU tests run that real kernel through SharedNativeCache/Bridge.
Inference table metadata is read-only within one forward, as in production.
"""

import math
import os
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv41.ppu import prefill_swa
from rtp_llm.models_py.modules.dsv41.ppu._quant import is_ppu
from rtp_llm.models_py.modules.dsv41.ppu.kv_codec import encode_kv
from rtp_llm.models_py.modules.dsv41.ppu.native_cache_adapter import (
    LOCAL_PREFILL_SWA_NATIVE_WRITE_ENV,
    LOCAL_PREFILL_SOURCE_CACHE_ENV,
    SharedNativeCache,
    _LocalSwaWriteGeometry,
    _local_swa_raw_tensor,
)
from rtp_llm.models_py.modules.dsv41.ppu.replay_policy import PrefillPolicy, SharedPrefillPolicy
from rtp_llm.models_py.modules.dsv41.ppu.test.test_local_prefill_fused import _NativePools, assert_bytes
from rtp_llm.models_py.modules.dsv41.ppu.test.test_local_prefill_operator_flags import _trace_events
from rtp_llm.models_py.modules.dsv41.ppu.test.test_local_prefill_rows import state_for
from rtp_llm.models_py.modules.dsv41.ppu.test.test_local_prefill_swa_segments import (
    FLAGS, arguments, compact_plan, stage, values,
)


ENV = LOCAL_PREFILL_SWA_NATIVE_WRITE_ENV


def fixture(device, prefixes=(129,), lengths=(4096,), *, entries=128, span=256,
            layers=2, configure=None, seed=True, decoder=False, cross_layer_alias=False):
    totals = tuple(p + n for p, n in zip(prefixes, lengths))
    geometry = NS(num_layers=layers, kv_source_layer_ids=(20,) if decoder else (),
                  index_source_layer_ids=(), head_dim=512, window=128,
                  ratio=lambda layer: 1, kv_source_for=lambda layer: None)
    native = _NativePools(geometry, totals, device)
    columns = math.ceil(max(totals, default=0) / span) + 1
    host = torch.arange(len(totals) * columns, dtype=torch.int32).reshape(len(totals), columns)
    if configure is not None:
        configure(host)
    tags = ("swa_kv", "decoder_swa_kv") if decoder else ("swa_kv",)
    native.raw = {}
    for tag in tags:
        native.entries[tag], native.spans[tag] = entries, span
        table = host.to(device=device, copy=True)
        native.inputs[tag] = NS(kv_cache_block_id=host.clone(), kv_cache_block_id_device=table,
                                kv_cache_kernel_block_id=host.clone(), kv_cache_kernel_block_id_device=table)
    native.group_tags = sorted(native.inputs)
    for layer in range(layers):
        tag = "decoder_swa_kv" if decoder and layer >= 21 else "swa_kv"
        raw = torch.full((host.numel() * (entries * 528 + 16) + 32,), 173,
                         dtype=torch.uint8, device=device)
        native.raw[layer] = raw
        tensor = raw[16:-16].view(host.numel(), entries * 528 + 16)
        native.handles[layer, tag] = NS(kv_cache_base=tensor, layer_id=layer, tag=tag)
    if cross_layer_alias:
        native.raw[1] = native.raw[0]
        native.handles[1, "swa_kv"].kv_cache_base = native.handles[0, "swa_kv"].kv_cache_base
    policy = PrefillPolicy(ced=decoder, bounded_replay=decoder)
    cache = SharedNativeCache(policy.freeze({}), geometry, native, native.inputs, totals,
                              device=device, prefill=True, defer_producer_slots=True)
    state = state_for(device, prefixes, lengths)
    state.cache, state.runtime.policy = cache, SharedPrefillPolicy(mode="compact")
    state.runtime.model._is_speculative = False
    if seed:
        for layer in range(layers):
            for rid, prefix in enumerate(prefixes):
                old = torch.arange(max(0, prefix - 127), prefix, device=device)
                cache._copy_rows((layer, "swa_kv"), rid, old,
                                 encode_kv(values(old, layer + rid), pool_tag="swa_kv"))
    return state


def expected_positions(table, request, start, end, entries, span):
    """Independent per-token form of the unchanged old GPU keep/missing path."""
    if any(int(table[request, p // span]) == -1 and p >= end - 128 for p in range(start, end)):
        raise ValueError("missing a live ring block")
    return [p for p in range(start, end)
            if p >= min(end, (p // span + 1) * span) - entries and int(table[request, p // span]) != -1]


class _CPUKernel:
    def __getitem__(self, grid):
        def launch(pool, table, packed, start, end, blocks, table_stride, span, entries,
                   rank, row_stride, width, tile):
            assert rank == 0 and row_stride == pool.stride(0)
            for column in range(start // span, (end - 1) // span + 1):
                block, last = int(table[column]), min(end, (column + 1) * span) - 1
                if not 0 <= block < blocks:
                    continue
                for residue in range(entries):
                    pos = last - (last % entries + entries - residue) % entries
                    if pos >= max(start, column * span):
                        pool[block, residue * width:(residue + 1) * width].copy_(packed[pos - start])
        return launch


class _WriteCases:
    @torch.no_grad()
    def test_full_compact_zero_ragged_padding_canaries_snapshots_and_receipts(self):
        for prefixes, lengths, compact in (
            ((129,), (4096,), False), ((17,), (4097,), False),
            ((17, 99, 64), (4096, 5, 0), True), ((17, 9), (0, 0), False),
        ):
            results = []
            for enabled in (False, True):
                with patch.dict(os.environ, {ENV: str(int(enabled))}):
                    state = fixture(self.device, prefixes, lengths)
                    if compact:
                        compact_plan(state)
                    args = arguments(state, compact=compact)
                    self.assertIsNotNone(args[-1])
                    stage(state, args)
                    pools = tuple(raw.clone() for raw in state.cache.native.raw.values())
                    snapshots = tuple((item.prefix, item.tail.clone(), item.fresh.clone())
                                      for item in state.cache._fresh_swa.values())
                    results.append((pools, snapshots, state.cache.writes, state))
            self.assertEqual(results[0][2], results[1][2])
            for left, right in zip(results[0][0], results[1][0]):
                assert_bytes(self, left, right)
            for left, right in zip(results[0][1], results[1][1]):
                self.assertEqual(left[0], right[0])
                assert_bytes(self, left[1], right[1])
                assert_bytes(self, left[2], right[2])
            candidate = results[1][3]
            self.assertEqual(results[0][3].cache._local_swa_native_write_stats["launches"], 0)
            self.assertEqual(candidate.cache._local_swa_native_write_stats["requests"], len(lengths))
            self.assertEqual(candidate.cache._local_swa_native_write_stats["mirrors"], 1)
            for pool in candidate.cache.pools.values():
                self.assertTrue(bool((pool.tensor[:, pool.entries * 528:] == 173).all()))
            for raw in candidate.cache.native.raw.values():
                self.assertTrue(bool((raw[:16] == 173).all() and (raw[-16:] == 173).all()))

    @torch.no_grad()
    def test_every_allocated_page_checkpoint_and_physical_zero(self):
        with patch.dict(os.environ, {ENV: "1"}):
            state = fixture(self.device, (0,), (2048,), entries=128, span=1024)
            stage(state, arguments(state))
        self.assertEqual(state.cache.writes[0]["absolute_positions"],
                         list(range(896, 1024)) + list(range(1920, 2048)))
        self.assertEqual(state.cache.writes[0]["owned_rows"], 256)
        self.assertTrue(bool((state.cache.pools[0, "swa_kv"].tensor[0, :128 * 528] != 173).any()))
        # At end384, column0 ends exactly at live_start256; its -1 is
        # historical, while every fresh row remains available in the snapshot.
        for count in (384, 768):
            arms = []
            for enabled in (False, True):
                with patch.dict(os.environ, {ENV: str(int(enabled))}):
                    state = fixture(self.device, (0,), (count,),
                                    configure=lambda host: host.__setitem__((0, 0), -1))
                    stage(state, arguments(state))
                    self.assertEqual(state.cache._fresh_swa[0, 0].fresh.shape[0], count)
                    arms.append((state.cache.native.raw[0].clone(), state.cache.writes))
            assert_bytes(self, arms[0][0], arms[1][0])
            self.assertEqual(arms[0][1], arms[1][1])

    @torch.no_grad()
    def test_alias_order_and_old_error_priority_are_unchanged(self):
        def alias(host):
            host[1].copy_(host[0])

        arms = []
        for enabled in (False, True):
            with patch.dict(os.environ, {ENV: str(int(enabled))}):
                state = fixture(self.device, (128, 256), (128, 1), entries=128, span=1024,
                                configure=alias, cross_layer_alias=True)
                stage(state, arguments(state))
                first_tail = state.cache._fresh_swa[0, 1].tail.clone()
                stage(state, arguments(state, layer=1), layer=1)
                arms.append((state.cache.native.raw[0].clone(), first_tail, state.cache.writes))
        assert_bytes(self, arms[0][0], arms[1][0])
        assert_bytes(self, arms[0][1], arms[1][1])
        self.assertEqual(arms[0][2], arms[1][2])
        for kind in ("live", "duplicate-live", "negative-live", "negative", "outside", "columns", "tail-zero"):
            errors = []
            for enabled in (False, True):
                def configure(host):
                    if kind in ("duplicate-live", "negative-live"):
                        host[0, 1] = host[0, 0] if kind == "duplicate-live" else -2
                    if kind in ("live", "duplicate-live", "negative-live", "tail-zero"):
                        host[0, -2 if kind != "tail-zero" else 0] = -1
                    elif kind == "negative":
                        host[0, 0] = -2
                    elif kind == "columns":
                        host.resize_(1, 1).zero_()
                    else:
                        host[0, 0] = host.numel()
                with patch.dict(os.environ, {ENV: str(int(enabled))}):
                    state = fixture(self.device, (129,) if kind == "tail-zero" else (0,),
                                    (0,) if kind == "tail-zero" else (768,),
                                    configure=configure, seed=False)
                    try:
                        stage(state, arguments(state))
                    except ValueError as error:
                        errors.append(str(error))
            self.assertEqual(len(errors), 2, kind)
            self.assertEqual(errors[0], errors[1], kind)

    @torch.no_grad()
    def test_duplicate_physical_blocks_fall_back_before_any_native_launch(self):
        def duplicate(host):
            host[0, 1] = host[0, 0]
        with patch.dict(os.environ, {ENV: "1"}):
            state = fixture(self.device, (0,), (512,), configure=duplicate)
            stage(state, arguments(state))
        self.assertEqual(state.cache._local_swa_native_write_stats["launches"], 0)
        self.assertEqual(state.cache._local_swa_native_write_stats["fallbacks"], 1)
        self.assertEqual(state.cache.writes[0]["absolute_positions"],
                         list(range(128, 256)) + list(range(384, 512)))

    @torch.no_grad()
    def test_table_value_mirror_once_versions_rebind_new_forward_and_midflag(self):
        with patch.dict(os.environ, {ENV: "1"}):
            state = fixture(self.device, (17,), (141,))
            pool = state.cache.pools[0, "swa_kv"]
            first = state.cache._verified_local_swa_write_table(pool)
            self.assertIsNotNone(first)
            self.assertIs(state.cache._verified_local_swa_write_table(state.cache.pools[1, "swa_kv"]), first)
            self.assertEqual(state.cache._local_swa_native_write_stats["mirrors"], 1)
            state.cache.inputs[pool.framework_tag].kv_cache_block_id.add_(1).sub_(1)
            self.assertIsNone(state.cache._verified_local_swa_write_table(pool))
            self.assertEqual(state.cache._local_swa_native_write_stats["mirrors"], 1)
            changed = fixture(self.device, (17,), (141,))
            changed_pool = changed.cache.pools[0, "swa_kv"]
            self.assertIsNotNone(changed.cache._verified_local_swa_write_table(changed_pool))
            changed_pool.table.add_(1).sub_(1)
            self.assertIsNone(changed.cache._verified_local_swa_write_table(changed_pool))
            self.assertEqual(changed.cache._local_swa_native_write_stats["mirrors"], 1)
            replacement = fixture(self.device, (17,), (141,))
            replacement.cache.pools[0, "swa_kv"].table = replacement.cache.pools[0, "swa_kv"].table.clone()
            self.assertIsNone(replacement.cache._verified_local_swa_write_table(replacement.cache.pools[0, "swa_kv"]))
            mismatch = fixture(self.device, (17,), (141,))
            # A distinct host table with matching shape but different values,
            # present at construction, must fail the actual equality check.
            meta = mismatch.cache.inputs["swa_kv"]
            meta.kv_cache_block_id.add_(1)
            native = mismatch.cache.native
            rebuilt = SharedNativeCache(PrefillPolicy().freeze({}), mismatch.cache.geometry, native,
                                        native.inputs, mismatch.cache.lengths, device=self.device, prefill=True)
            self.assertIsNone(rebuilt._verified_local_swa_write_table(rebuilt.pools[0, "swa_kv"]))
            self.assertEqual(rebuilt._local_swa_native_write_stats["mirrors"], 1)
            newer = fixture(self.device, (17,), (141,))
            self.assertIsNotNone(newer.cache._verified_local_swa_write_table(newer.cache.pools[0, "swa_kv"]))
        with patch.dict(os.environ, {ENV: "0"}):
            late = fixture(self.device)
        with patch.dict(os.environ, {ENV: "1"}):
            self.assertIsNone(late.cache._verified_local_swa_write_table(late.cache.pools[0, "swa_kv"]))

    @torch.no_grad()
    def test_source_banks_survive_swa_and_full_payload_overlap_uses_old_writer(self):
        with patch.dict(os.environ, {ENV: "1", LOCAL_PREFILL_SOURCE_CACHE_ENV: "1",
                                     "TP_SIZE": "1", "PREFILL_CP_SIZE": "1"}):
            state = fixture(self.device, (0,), (140,), layers=22, decoder=True)
            cache = state.cache
            cache.read_source(20, 0, global_entries=2)
            saved = cache._source_reads[20, 0]
            stamp = cache._local_source_reads[20, 0]
            stage(state, arguments(state))
            cache.release_swa(0)
            self.assertIs(cache._source_reads[20, 0], saved)
            self.assertIs(cache._local_source_reads[20, 0], stamp)
        for overlap in (False, True):
            arms = []
            for enabled in (False, True):
                with patch.dict(os.environ, {ENV: str(int(enabled))}):
                    state = fixture(self.device, (0,), (140,))
                    cache = state.cache
                    pool = cache.pools[0, "swa_kv"]
                    # Both variants share an allocation. Only the first has a
                    # real byte overlap; the canary view is a disjoint prefix.
                    shared = pool.tensor[:1, :528] if overlap else cache.native.raw[0][:16].view(1, 16)
                    cache.pools[20, "global_kv_1"] = NS(tensor=shared, table=pool.table)
                    source, stamp = object(), object()
                    cache._source_reads[(20, 0)] = source
                    cache._local_source_reads[(20, 0)] = stamp
                    stage(state, arguments(state))
                self.assertIs(cache._source_reads[(20, 0)], source)
                self.assertIs(cache._local_source_reads[(20, 0)], stamp)
                self.assertEqual(cache._local_swa_native_write_stats["launches"],
                                 int(enabled and not overlap))
                arms.append((cache.native.raw[0].clone(), cache.writes))
            assert_bytes(self, arms[0][0], arms[1][0])
            self.assertEqual(arms[0][1], arms[1][1])

    @torch.no_grad()
    def test_unresolved_views_and_non_strided_tensors_cannot_reach_raw_writer(self):
        state = fixture(self.device, (0,), (140,))
        pool = state.cache.pools[0, "swa_kv"]
        packed = arguments(state)[2]
        self.assertTrue(state.cache._local_swa_write_payload_eligible(pool, packed))
        neg = torch._neg_view(packed)
        self.assertTrue(neg.is_neg())
        self.assertFalse(state.cache._local_swa_write_payload_eligible(pool, neg))
        original = pool.tensor
        pool.tensor = torch._neg_view(original)
        self.assertFalse(state.cache._local_swa_write_payload_eligible(pool, packed))
        pool.tensor = original
        self.assertFalse(_local_swa_raw_tensor(torch.tensor([1j]).conj()))
        self.assertFalse(_local_swa_raw_tensor(torch.empty(1, 1).to_sparse()))
        for field in ("kv_cache_block_id", "kv_cache_block_id_device"):
            state = fixture(self.device, (0,), (140,))
            meta = state.cache.inputs["swa_kv"]
            # Logical values remain equal to the resolved other table. Raw
            # pointer consumers still cannot interpret the negative view bit.
            setattr(meta, field, torch._neg_view(-getattr(meta, field)))
            rebuilt = SharedNativeCache(PrefillPolicy().freeze({}), state.cache.geometry,
                                        state.cache.native, state.cache.inputs, state.cache.lengths,
                                        device=self.device, prefill=True)
            self.assertIsNone(rebuilt._verified_local_swa_write_table(rebuilt.pools[0, "swa_kv"]))


class LocalSwaNativeWriteCPU(_WriteCases, unittest.TestCase):
    def setUp(self):
        self.device = "cpu"
        environment = patch.dict(os.environ, {**FLAGS, ENV: "1"})
        environment.start()
        self.addCleanup(environment.stop)
        kernel = patch.object(prefill_swa, "_write_prefill_ring_range", _CPUKernel())
        kernel.start()
        self.addCleanup(kernel.stop)
        geometry = patch.object(SharedNativeCache, "_local_swa_native_geometry",
                                lambda self, pool, packed: (
                                    _LocalSwaWriteGeometry(0, pool.entries, pool.tensor.shape[1])
                                    if self._local_swa_write_payload_eligible(pool, packed) else None))
        geometry.start()
        self.addCleanup(geometry.stop)

    def test_independent_per_token_oracle_matches_real_host_planner(self):
        for entries, span, start, count in (
            (128, 1024, 0, 2048), (128, 256, 129, 4096), (129, 257, 1, 4097),
            (256, 128, 127, 140), (128, 256, 256, 0), (128, 256, 255, 1),
        ):
            end = start + count
            columns = (end + span - 1) // span + 1
            table = torch.arange(columns, dtype=torch.int32)[None]
            pool = NS(entries=entries, span=span, framework_tag="swa_kv", table=table,
                      tensor=torch.full((columns, entries * 528 + 16), 173, dtype=torch.uint8))
            packed = torch.arange(count * 528).remainder(251).to(torch.uint8).view(count, 528)
            expected = expected_positions(table, 0, start, end, entries, span)
            actual = prefill_swa.write_prefill_range(
                pool, _LocalSwaWriteGeometry(0, entries, pool.tensor.shape[1]), table, 0, start, end, 128, packed,
            )
            self.assertEqual(actual, expected)
            for pos in expected:
                assert_bytes(self, pool.tensor[int(table[0, pos // span]),
                                               (pos % entries) * 528:(pos % entries + 1) * 528], packed[pos - start])

    def test_rebind_during_mirror_is_not_certified(self):
        state = fixture(self.device, (17,), (141,))
        pool = state.cache.pools[0, "swa_kv"]
        meta = state.cache.inputs["swa_kv"]
        original_to, address = torch.Tensor.to, pool.table.data_ptr()

        def mutate(value, *args, **kwargs):
            result = original_to(value, *args, **kwargs)
            if value.data_ptr() == address and kwargs.get("copy"):
                meta.kv_cache_block_id = meta.kv_cache_block_id.clone()
            return result

        with patch.object(torch.Tensor, "to", new=mutate):
            self.assertIsNone(state.cache._verified_local_swa_write_table(pool))


@unittest.skipUnless(os.environ.get("RTP_V41_TEST_LOCAL_WORKSPACE_DEVICE") == "1", "explicit PPU device test")
class LocalSwaNativeWriteDevice(_WriteCases, unittest.TestCase):
    def setUp(self):
        self.assertTrue(is_ppu(), "native SWA qualification requires PPU")
        self.assertEqual(torch.cuda.get_device_capability(), (8, 9))
        self.device = "cuda:0"
        environment = patch.dict(os.environ, {**FLAGS, ENV: "1"})
        environment.start()
        self.addCleanup(environment.stop)

    @torch.no_grad()
    def test_l20_full_and_l21_compact_real_native_kernel_and_mirrors(self):
        arms = []
        for enabled in (False, True):
            with patch.dict(os.environ, {ENV: str(int(enabled))}):
                state = fixture(self.device, (126976,), (4096,), layers=22, decoder=True)
                full = arguments(state, layer=20)
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                        torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as profile:
                    stage(state, full, layer=20)
                    compact_plan(state)
                    stage(state, arguments(state, compact=True, layer=21), layer=21)
                    torch.cuda.synchronize()
                events = _trace_events(profile, f"local-swa-native-write-{int(enabled)}")
                kernels = [e["name"] for e in events if e.get("cat") == "kernel"]
                self.assertEqual(sum("_write_prefill_ring_range" in name for name in kernels), 2 * int(enabled))
                l20, l21 = state.cache._fresh_swa[20, 0], state.cache._fresh_swa[21, 0]
                self.assertEqual((l20.prefix, l20.tail.shape[0], l20.fresh.shape[0]), (126976, 127, 4096))
                self.assertEqual((l21.prefix, l21.tail.shape[0], l21.fresh.shape[0]), (130932, 0, 140))
                if enabled:
                    self.assertEqual(state.cache._local_swa_native_write_stats["mirrors"], 2)
                samples = state.cache.read_swa(20, 0, torch.arange(130805, 131072, device=self.device))
                arms.append((samples, state.cache.writes,
                             tuple(raw.clone() for raw in state.cache.native.raw.values())))
        assert_bytes(self, arms[0][0], arms[1][0])
        self.assertEqual(arms[0][1], arms[1][1])
        for left, right in zip(arms[0][2], arms[1][2]):
            assert_bytes(self, left, right)


if __name__ == "__main__":
    unittest.main()
