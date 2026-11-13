"""Lazy local source-bank reuse against the unchanged validated native reads.

Register this module and test_local_prefill_fused.py (native padded-pool
fixture and byte comparator) in the existing contracts/device targets.
Inference metadata and returned banks are read-only within one forward;
normal tensor mutations and every adapter full-tag write invalidate reuse.
"""

import gc
import json
import os
from pathlib import Path
import tempfile
import unittest
import weakref
from dataclasses import replace
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv41.ppu import _quant, prefill
from rtp_llm.models_py.modules.dsv41.ppu.kv_codec import decode_kv, encode_kv
from rtp_llm.models_py.modules.dsv41.ppu.native_cache import _FULL
from rtp_llm.models_py.modules.dsv41.ppu.native_cache_adapter import LOCAL_PREFILL_SOURCE_CACHE_ENV, SharedNativeCache
from rtp_llm.models_py.modules.dsv41.ppu.replay_policy import PrefillPolicy
from rtp_llm.models_py.modules.dsv41.ppu.test.test_local_prefill_fused import _NativePools, assert_bytes


ENV = LOCAL_PREFILL_SOURCE_CACHE_ENV


class _SourceFixture:
    def __init__(self, device, ratio=2, prefixes=(129, 257, 31), fresh=(9, 0, 7), sources=(0, 20)):
        self.device, self.ratio, self.prefixes, self.fresh = device, ratio, tuple(prefixes), tuple(fresh)
        self.totals = tuple(p + n for p, n in zip(prefixes, fresh))
        self.geometry = NS(num_layers=1, kv_source_layer_ids=sources, head_dim=512, window=128,
                           ratio=lambda layer: ratio, kv_source_for=lambda layer: sources[0])
        self.native = _NativePools(self.geometry, self.totals, device)
        self.cache = SharedNativeCache(
            PrefillPolicy().freeze({}), self.geometry, self.native, self.native.inputs,
            self.totals, device=device, prefill=True, defer_producer_slots=True,
        )
        for source in sources:
            for request, total in enumerate(self.totals):
                positions = torch.arange(total // ratio, device=device) * ratio
                values = self.values(positions, source + request)
                self.cache._copy_rows((source, self.tag), request, positions, encode_kv(values, pool_tag=self.tag))
                payload, scales = _quant.quantize_indexer_k_rows(values[:, :128].contiguous())
                packed = _quant.indexer_k_packed_rows(payload, scales)
                self.cache._copy_rows((source, "indexer_kv"), request, positions, packed)
        self.stage_swa()

    @property
    def tag(self):
        return "global_kv_2" if self.ratio == 2 else "global_kv_1"

    def values(self, positions, shift=0):
        return ((positions[:, None] % 29 - 14).float() / 8
                + (torch.arange(512, device=self.device)[None] % 17).float() / 64
                + shift / 32).bfloat16()

    def stage_swa(self, shift=0):
        positions, requests, fresh = [], [], []
        for rid, (prefix, count) in enumerate(zip(self.prefixes, self.fresh)):
            old = torch.arange(max(0, prefix - 127), prefix, device=self.device)
            self.cache._copy_rows((0, "swa_kv"), rid, old, encode_kv(self.values(old, rid + shift), pool_tag="swa_kv"))
            current = torch.arange(prefix, prefix + count, device=self.device)
            positions.append(current)
            requests.append(torch.full_like(current, rid))
            fresh.append(encode_kv(self.values(current, rid + shift), pool_tag="swa_kv"))
        self.cache.stage_swa(0, torch.cat(positions), torch.cat(requests), torch.cat(fresh), self.prefixes)

    def packed(self, source, request=0, count=None):
        count = self.totals[request] // self.ratio if count is None else count
        return self.cache.read_source(source, request, global_entries=count)


def _bank_reads(cache):
    original = cache._try_read_local_source_pool
    reads = []

    def copy(key, request, count, **kwargs):
        if key[1] in _FULL:
            reads.append((key, request, count))
        return original(key, request, count, **kwargs)

    return reads, patch.object(cache, "_try_read_local_source_pool", new=copy)


class _SourceCacheCases:
    def setUp(self):
        environment = patch.dict(os.environ, {"TP_SIZE": "1", "PREFILL_CP_SIZE": "1"})
        environment.start()
        self.addCleanup(environment.stop)

    def test_lazy_first_read_release_reuse_exact_formats_and_native_padding(self):
        for ratio in (1, 2):
            with self.subTest(ratio=ratio):
                fixture = _SourceFixture(self.device, ratio, prefixes=(129, 0, 31))
                cache = fixture.cache
                with patch.dict(os.environ, {ENV: "0"}):
                    expected = {rid: fixture.packed(0, rid) for rid in (0, 1, 2)}
                reads, probe = _bank_reads(cache)
                with patch.dict(os.environ, {ENV: "1"}), probe:
                    cache.prepare_source(0)
                    self.assertEqual(reads, [], "prepare must not pre-read inactive requests")
                    for rid in (0, 1, 2):
                        actual = fixture.packed(0, rid)
                        for left, right in zip(actual, expected[rid]):
                            assert_bytes(self, left, right)
                        count = fixture.totals[rid] // ratio
                        for layer in (20, 21, 24, 28, 32, 36):
                            cache.release_swa(layer)
                            assert_bytes(self, cache.read_global(0, rid, global_entries=count), expected[rid][0])
                            payload, scales = cache.read_indexer_fp4(0, rid, global_entries=count)
                            assert_bytes(self, _quant.indexer_k_packed_rows(payload, scales), expected[rid][1])
                            self.assertFalse(any(t.data_ptr() == pool.tensor.data_ptr()
                                                 for t in actual for pool in cache.pools.values() if t.numel()))
                    self.assertEqual(len(reads), 6, "one validated main+indexer read per explicitly requested bank")
                    self.assertEqual(set(cache._source_reads), {(0, 0), (0, 1), (0, 2)})
                for handle in fixture.native.handles.values():
                    assert_bytes(self, handle.kv_cache_base[:, -16:], torch.full_like(handle.kv_cache_base[:, -16:], 173))

    def test_inactive_nonzero_prefix_holes_are_not_read_or_cached(self):
        fixture = _SourceFixture(self.device)
        cache = fixture.cache
        for pool in cache.pools.values():
            pool.table[1].fill_(-1)
        reads, probe = _bank_reads(cache)
        with patch.dict(os.environ, {ENV: "1"}), probe:
            cache.prepare_source(0)
            self.assertEqual(reads, [])
            fixture.packed(0, 0)
            fixture.packed(0, 2)
            cache.prepare_source(20)
            cache.release_swa(21)
            self.assertEqual({rid for _, rid, _ in reads}, {0, 2})
            self.assertNotIn((0, 1), cache._source_reads)
            with self.assertRaisesRegex(ValueError, "unallocated physical block"):
                fixture.packed(0, 1, 1)
            self.assertNotIn((0, 1), cache._source_reads, "a failed read must not publish a partial bank")

    def test_any_main_or_indexer_write_clears_all_aliased_source_requests(self):
        for tag in ("global_kv_1", "indexer_kv"):
            with self.subTest(tag=tag), patch.dict(os.environ, {ENV: "1"}):
                fixture = _SourceFixture(self.device, ratio=1, prefixes=(17, 17), fresh=(4, 4))
                cache = fixture.cache
                # Source20/main aliases source0/main; request1 aliases request0.
                shared = cache.pools[0, fixture.tag]
                cache.pools[20, fixture.tag] = replace(cache.pools[20, fixture.tag], tensor=shared.tensor, table=shared.table)
                for pool in cache.pools.values():
                    pool.table[1].copy_(pool.table[0])
                for source in (0, 20):
                    for rid in (0, 1):
                        fixture.packed(source, rid)
                self.assertEqual(len(cache._local_source_reads), 4)
                positions = torch.tensor([0], device=self.device)
                width = 68 if tag == "indexer_kv" else 288
                changed = torch.full((1, width), 29, device=self.device, dtype=torch.uint8)
                cache._copy_rows((20, tag), 1, positions, changed)
                self.assertFalse(cache._source_reads or cache._local_source_reads)
                expected = []
                with patch.dict(os.environ, {ENV: "0"}):
                    expected = fixture.packed(0, 0)
                for left, right in zip(fixture.packed(0, 0), expected):
                    assert_bytes(self, left, right)
                # Entry invalidation precedes the direct native-store dispatcher,
                # even when its unchanged CP8 eligibility declines local inputs.
                self.assertFalse(cache._try_store_prefill_range((20, tag), 0, 0, changed))
                self.assertFalse(cache._source_reads or cache._local_source_reads)

    def test_pool_table_pointer_layout_and_normal_versions_invalidate(self):
        with torch.inference_mode(False):
            fixture = _SourceFixture(self.device)
            cache = fixture.cache
            reads, probe = _bank_reads(cache)
            with patch.dict(os.environ, {ENV: "1"}), probe:
                baseline = tuple(t.clone() for t in fixture.packed(0))
                operations = (
                    lambda: cache.pools[0, fixture.tag].table.copy_(cache.pools[0, fixture.tag].table.clone()),
                    lambda: cache.inputs[cache.pools[0, fixture.tag].framework_tag].kv_cache_kernel_block_id.copy_(
                        cache.inputs[cache.pools[0, fixture.tag].framework_tag].kv_cache_kernel_block_id.clone()),
                    lambda: cache.pools.__setitem__((0, fixture.tag), replace(cache.pools[0, fixture.tag],
                        tensor=cache.pools[0, fixture.tag].tensor.clone())),
                    lambda: cache.pools.__setitem__((0, "indexer_kv"), replace(cache.pools[0, "indexer_kv"],
                        table=cache.pools[0, "indexer_kv"].table.clone())),
                )
                for mutate in operations:
                    before = len(reads)
                    mutate()
                    actual = fixture.packed(0)
                    self.assertEqual(len(reads), before + 2)
                    for left, right in zip(actual, baseline):
                        assert_bytes(self, left, right)
                cache._source_reads[0, 0][0].zero_()
                for left, right in zip(fixture.packed(0), baseline):
                    assert_bytes(self, left, right)
                before = len(reads)
                pool = cache.pools[0, fixture.tag]
                padded = pool.tensor.new_full((pool.tensor.shape[0], pool.tensor.shape[1] + 16), 173)
                padded[:, :pool.tensor.shape[1]].copy_(pool.tensor)
                cache.pools[0, fixture.tag] = replace(pool, tensor=padded)
                for left, right in zip(fixture.packed(0), baseline):
                    assert_bytes(self, left, right)
                self.assertEqual(len(reads), before + 2)
                cache.pools[0, fixture.tag].table[0, 0] = -1
                with self.assertRaisesRegex(ValueError, "unallocated physical block"):
                    fixture.packed(0)
                self.assertFalse(cache._source_reads, "changed invalid metadata must run the original native checks")

    def test_requested_count_growth_and_new_inference_forward_do_not_reuse_old_banks(self):
        fixture = _SourceFixture(self.device)
        cache = fixture.cache
        reads, probe = _bank_reads(cache)
        with patch.dict(os.environ, {ENV: "1"}), probe:
            first = fixture.packed(0, count=3)
            small = fixture.packed(0, count=2)
            self.assertEqual(len(reads), 2)
            assert_bytes(self, small[0], first[0][:2])
            larger = fixture.packed(0, count=17)
            self.assertEqual(len(reads), 4)
            self.assertEqual(larger[0].shape, (17, 288))
            with self.assertRaises(RuntimeError):
                fixture.packed(0, count=-1)
            self.assertFalse(cache._source_reads)
        with torch.inference_mode(), patch.dict(os.environ, {ENV: "1"}):
            first = _SourceFixture(self.device, prefixes=(17,), fresh=(9,))
            old = first.packed(0)
            # Same native allocations can be reused by a subsequent forward.
            newer = SharedNativeCache(PrefillPolicy().freeze({}), first.geometry, first.native,
                                     first.native.inputs, first.totals, device=self.device, prefill=True,
                                     defer_producer_slots=True)
            self.assertFalse(newer._source_reads)
            positions = torch.tensor([0], device=self.device)
            newer._copy_rows((0, first.tag), 0, positions, torch.full((1, 288), 47, device=self.device, dtype=torch.uint8))
            actual = newer.read_source(0, 0, global_entries=first.totals[0] // first.ratio)
            self.assertFalse(torch.equal(old[0], actual[0]))
            self.assertIsNot(first.cache._local_source_reads, newer._local_source_reads)

    def test_workspace_and_indexer_bytes_preserve_source_and_swa_refresh(self):
        for ratio in (1, 2):
            with self.subTest(ratio=ratio):
                fixture = _SourceFixture(self.device, ratio)
                cache = fixture.cache
                positions = torch.cat([torch.arange(p, p + n, device=self.device)
                                       for p, n in zip(fixture.prefixes, fixture.fresh)])
                requests = torch.tensor([rid for rid, n in enumerate(fixture.fresh) for _ in range(n)], device=self.device)
                chunks = fixture.fresh
                cumulative = [0]
                for n in chunks:
                    cumulative.append(cumulative[-1] + n)
                context = prefill.build_step_context(positions, requests, torch.tensor(cumulative, device=self.device))
                args = dict(ops=NS(decode_kv=decode_kv), cache=cache, layer_id=0, geometry=fixture.geometry,
                            context=context, selected_blocks=torch.zeros(positions.numel(), 5, device=self.device, dtype=torch.int32),
                            window=128, swa_starts=None)
                layout = NS(chunk_lengths_per_req=chunks, prefix_lengths_host=fixture.prefixes)

                def workspace():
                    with patch.dict(os.environ, {"RTP_V41_PPU_LOCAL_PREFILL_WORKSPACE": "1"}):
                        return prefill.build_cp_prefill_workspace(
                            **args, cp=layout, shared={"local_prefill_workspace_layout": layout}, q8=False,
                        )

                for source in (0, 20):
                    fixture.geometry.kv_source_for = lambda layer, source=source: source
                    with patch.dict(os.environ, {ENV: "0"}):
                        expected = prefill._build_reference_workspace(**args)
                    with patch.dict(os.environ, {ENV: "1"}):
                        current = workspace()
                    for left, right in zip(current, expected):
                        assert_bytes(self, left, right)
                with patch.dict(os.environ, {ENV: "1"}):
                    fixture.stage_swa(shift=3)
                    after = workspace()
                self.assertFalse(torch.equal(after[0], current[0]))
                with patch.dict(os.environ, {ENV: "0"}):
                    for left, right in zip(after, prefill._build_reference_workspace(**args)):
                        assert_bytes(self, left, right)


class LocalPrefillSourceCacheCPU(_SourceCacheCases, unittest.TestCase):
    device = "cpu"

    def test_replaced_native_metadata_is_alive_until_bank_invalidation(self):
        with torch.inference_mode(False), patch.dict(os.environ, {ENV: "1"}):
            fixture = _SourceFixture("cpu", sources=(0,))
            cache = fixture.cache
            expected = tuple(t.clone() for t in fixture.packed(0, count=3))

            def replace_native_metadata():
                references = {}
                for tag in (fixture.tag, "indexer_kv"):
                    pool = cache.pools[0, tag]
                    meta = cache.inputs[pool.framework_tag]
                    captured = (pool.tensor, pool.table, pool.handle.kv_cache_base,
                                meta.kv_cache_kernel_block_id, meta.kv_cache_kernel_block_id_device)
                    for name, tensor in zip(("pool", "table", "base", "host", "device"), captured):
                        references[tag, name] = weakref.ref(tensor)
                    replacement = pool.handle.kv_cache_base.clone()
                    pool.handle.kv_cache_base = replacement
                    cache.pools[0, tag] = replace(pool, tensor=replacement.view(pool.tensor.shape),
                                                table=pool.table.clone())
                    # CPU mirrors can share Tensor objects. Replace every
                    # aliasing metadata field, not only the signature's fields.
                    for field in ("kv_cache_block_id", "kv_cache_block_id_device",
                                  "kv_cache_kernel_block_id", "kv_cache_kernel_block_id_device"):
                        setattr(meta, field, getattr(meta, field).clone())
                return references

            references = replace_native_metadata()
            gc.collect()
            self.assertTrue(all(reference() is not None for reference in references.values()),
                            "the old native metadata must stay alive while its bank signature exists")
            reads, probe = _bank_reads(cache)
            with probe:
                actual = fixture.packed(0, count=3)
            self.assertEqual(len(reads), 2, "replacement must invalidate and re-run both native reads")
            for left, right in zip(actual, expected):
                assert_bytes(self, left, right)
            gc.collect()
            self.assertTrue(all(reference() is None for reference in references.values()),
                            "invalidating the old bank must release all replaced metadata owners")

    def test_rollback_decode_graph_cp8_empty_and_existing_tuple_branch(self):
        for reason in ("disabled", "decode", "graph", "cp8", "tp8", "empty"):
            with self.subTest(reason=reason), patch.dict(os.environ, {ENV: "1"}):
                fixture = _SourceFixture("cpu")
                cache = fixture.cache
                if reason == "disabled":
                    environment = patch.dict(os.environ, {ENV: "0"})
                elif reason == "tp8":
                    environment = patch.dict(os.environ, {"TP_SIZE": "8", "PREFILL_CP_SIZE": "8"})
                else:
                    environment = patch.dict(os.environ, {})
                if reason == "decode": cache._is_prefill = False
                if reason == "graph": next(iter(cache.inputs.values())).is_cuda_graph = True
                if reason == "cp8": cache.cp["size"] = 8
                if reason == "empty": cache.lengths = (0, 0, 0)
                with environment:
                    fixture.packed(0, count=3)
                    self.assertFalse(cache._local_source_reads)
                    cache.release_swa(0)
                    self.assertFalse(cache._source_reads)
        for reason in ("disabled", "graph"):
            with self.subTest(populated_then=reason), patch.dict(os.environ, {ENV: "1"}):
                fixture = _SourceFixture("cpu")
                fixture.packed(0, count=3)
                self.assertTrue(fixture.cache._local_source_reads)
                if reason == "graph":
                    next(iter(fixture.cache.inputs.values())).is_cuda_graph = True
                with patch.dict(os.environ, {ENV: "0" if reason == "disabled" else "1"}):
                    fixture.cache.release_swa(0)
                    self.assertFalse(fixture.cache._source_reads or fixture.cache._local_source_reads)
        fixture = _SourceFixture("cpu")
        cache = fixture.cache
        with patch.dict(os.environ, {ENV: "0"}):
            global_rows, packed = fixture.packed(0)
            payload, scales = cache.read_indexer_fp4(0, 0, global_entries=global_rows.shape[0])
        cache.sharded, cache.cp = True, {"size": 8}
        for flag in ("0", "1"):
            cache._source_reads[0, 0] = global_rows, (payload, scales)
            with patch.dict(os.environ, {ENV: flag}):
                actual = fixture.packed(0)
                assert_bytes(self, actual[0], global_rows)
                assert_bytes(self, actual[1], packed)
                got = cache.read_indexer_fp4(0, 0, global_entries=global_rows.shape[0])
                self.assertEqual(got[0].data_ptr(), payload.data_ptr())
                self.assertEqual(got[1].data_ptr(), scales.data_ptr())
                assert_bytes(self, got[0], payload)
                assert_bytes(self, got[1], scales)
                cache.release_swa(0)
                self.assertFalse(cache._source_reads, "CP8 release behavior stays unchanged")


@unittest.skipUnless(os.environ.get("RTP_V41_TEST_LOCAL_WORKSPACE_DEVICE") == "1", "explicit PPU device test")
class LocalPrefillSourceCacheDevice(_SourceCacheCases, unittest.TestCase):
    device = "cuda"

    def setUp(self):
        super().setUp()
        self.assertTrue(_quant.is_ppu(), "source cache qualification requires an actual ZW PPU")

    @torch.inference_mode()
    def test_actual_source20_l21_and_late_consumer_profiles_gather_once(self):
        for ratio in (1, 2):
            fixture = _SourceFixture("cuda", ratio, prefixes=(126976,), fresh=(4096,), sources=(20,))
            count = fixture.totals[0] // ratio
            pool_shapes = {tuple(pool.tensor.shape) for key, pool in fixture.cache.pools.items() if key[1] in _FULL}
            outputs = []
            for flag, expected_reads in ((False, 14), (True, 2)):
                cache = fixture.cache
                cache._source_reads.clear()
                cache._local_source_reads.clear()
                reads, probe = _bank_reads(cache)
                with patch.dict(os.environ, {ENV: str(int(flag))}), probe:
                    torch.cuda.synchronize()
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                           torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as profile:
                        cache.prepare_source(20)
                        payload, scales = cache.read_indexer_fp4(20, 0, global_entries=count)
                        bank = cache.read_global(20, 0, global_entries=count)
                        cache.release_swa(20)
                        cache.read_global(20, 0, global_entries=count)  # L21's new workspace layout.
                        cache.release_swa(21)
                        for layer in (24, 28, 32, 36):
                            cache.read_indexer_fp4(20, 0, global_entries=count)
                            cache.release_swa(layer)
                        torch.cuda.synchronize()
                self.assertEqual(len(reads), expected_reads)
                outputs.append((bank, payload, scales))
                root = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
                with tempfile.TemporaryDirectory(prefix="source-cache-") as temporary:
                    path = Path(root or temporary) / f"local-prefill-source-cache-ratio{ratio}-flag{int(flag)}.trace.json"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    profile.export_chrome_trace(str(path))
                    events = json.loads(path.read_text())["traceEvents"]
                pool_indices = [event for event in events if event.get("cat") == "cpu_op"
                                and event["name"] == "aten::index"
                                and tuple(event.get("args", {}).get("Input Dims", [()])[0]) in pool_shapes]
                self.assertEqual(len(pool_indices), 0)
                restores = [event for event in events if event.get("cat") == "kernel"
                            and "_restore_rows" in event["name"]]
                self.assertEqual(len(restores), expected_reads,
                                 "each cache miss must execute the existing native restore")
            for left, right in zip(outputs[1], outputs[0]):
                assert_bytes(self, left, right)


if __name__ == "__main__":
    unittest.main()
