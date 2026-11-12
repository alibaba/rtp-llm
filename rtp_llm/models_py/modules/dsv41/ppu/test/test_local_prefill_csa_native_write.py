"""Independent CSA physical-table birth and the unchanged native FP32 store.

CPU substitutes only PPU admission and the store launcher. Explicit device
cases prove the actual store_prefill_csa(SEPARATE=True) path. No process
group, fake placement, new codec or tolerance is used.
"""

import gc
import json
import os
import unittest
import weakref
from dataclasses import replace
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv41 import cp
from rtp_llm.models_py.modules.dsv41.ppu import _quant, graph_cache_kernels, prefill
from rtp_llm.models_py.modules.dsv41.ppu.kv_codec import encode_kv
from rtp_llm.models_py.modules.dsv41.ppu.native_cache_adapter import (
    LOCAL_PREFILL_CSA_NATIVE_WRITE_ENV as ENV, SharedNativeCache, _latest_csa_indices,
)
from rtp_llm.models_py.modules.dsv41.ppu.ops import PPUOps
from rtp_llm.models_py.modules.dsv41.ppu.test.test_local_prefill_fused import assert_bytes
from rtp_llm.models_py.modules.dsv41.ppu.test.test_local_prefill_operator_flags import _trace_events
from rtp_llm.models_py.modules.dsv41.ppu.test.test_local_prefill_rotary_pack import (
    _fixture as producer_fixture, _produce,
)
from rtp_llm.models_py.modules.dsv41.ppu.test.test_local_prefill_source_native_write import _fixture as full_fixture
from rtp_llm.models_py.modules.dsv41.ppu.test.test_local_prefill_swa_segments import FLAGS, compact_plan


ROTARY = prefill.LOCAL_PREFILL_ROTARY_PACK_ENV


def _fixture(device, prefixes=(256,), lengths=(140,), *, entries=2, span=256,
             ratios=None, configure=None, blocks=None, **kwargs):
    """Real CSA physical-table metadata exists before the new constructor.

    Full fixture's retained row ticket supplies the existing producer proof;
    CSA birth always comes from this new cache's own physical table/pool.
    """
    f = full_fixture(device, prefixes, lengths, ratios=ratios or {2: 2},
                     spans={"csa_state": span}, configure=configure, blocks=blocks, **kwargs)
    if entries == 2:
        return f
    # Reconstruct with a real alternative ring before birth, never rebaseline
    # the already-created writer's proof or an existing pool binding.
    native = f.cache.native
    native.entries["csa_state"] = entries
    for key, handle in native.handles.items():
        if key[1] == "csa_state":
            old = handle.kv_cache_base
            raw = torch.full((old.shape[0] * (entries * 1024 + 16) + 32,), 173,
                             dtype=torch.float32, device=device)
            native.raw[key] = raw
            handle.kv_cache_base = raw[16:-16].view(old.shape[0], entries * 1024 + 16)
    cache = SharedNativeCache(f.cache.config, f.cache.geometry, native, native.inputs, f.cache.lengths,
                              device=device, prefill=True, defer_producer_slots=True)
    # Produce a fresh real row-order ticket for this fresh cache.
    f.state.cache = cache
    from rtp_llm.models_py.modules.dsv41.ppu.test.test_local_prefill_swa_segments import arguments
    args = arguments(f.state)
    f.cache, f.args = cache, args
    f.proofs = {source: prefill._make_local_producer_rotary_proof(
        args[-1], cache=cache, layer_id=source, prefixes=prefixes, lengths=lengths,
    ) for source in cache.geometry.kv_source_layer_ids}
    return f


def _fields(f, rid=0, *, start=None):
    start = f.prefixes[rid] if start is None else start
    count = f.prefixes[rid] + f.lengths[rid] - start
    index = torch.arange(count * 512, device=f.cache.device, dtype=torch.int64).reshape(count, 512)
    # Arbitrary FP32 bit patterns, including both NaN signs/payloads, signed
    # zero, subnormals and infinities. Native store must be a bit copy.
    patterns = torch.tensor([0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x00800000,
                             0x7f7fffff, 0xff7fffff, 0x7f800000, 0xff800000,
                             0x7fc00001, 0xffc12345, 0x7f800001, 0xff800001,
                             0x3f800000, 0xbf800000], device=f.cache.device, dtype=torch.int64)
    values = patterns[(index + index // 512 * 3) % len(patterns)].to(torch.int32).view(torch.float32)
    scores = patterns[(index * 7 + index // 512 * 5 + 3) % len(patterns)].to(torch.int32).view(torch.float32)
    return values, scores


def _old(f, source, values, scores, rid=0, *, start=None):
    start = f.prefixes[rid] if start is None else start
    positions = torch.arange(values.shape[0], device=f.cache.device) + start
    slots = f.cache.slots_at_positions(source, "csa_state", rid, positions)
    f.cache.insert_rows(source, "csa_state", slots, torch.cat((values, scores), -1).float())


def _try(f, source, values, scores, rid=0, *, start=None):
    return f.cache.try_insert_local_csa_fields(
        source, rid, f.prefixes[rid] if start is None else start, values, scores,
        proof=f.proofs[source], prefill_layout=(f.prefixes, f.lengths),
    )


def _store(f, source, values, scores, rid=0, *, start=None):
    hit = _try(f, source, values, scores, rid, start=start)
    if not hit:
        _old(f, source, values, scores, rid, start=start)
    return hit


def _oracle(f, source, values, scores, rid=0, *, start=None):
    """Independent token loop: old keep -> address -> latest/first tie."""
    start = f.prefixes[rid] if start is None else start
    pool = f.cache.pools[source, "csa_state"]
    output, table = pool.tensor.cpu().clone(), pool.table[rid].cpu()
    fields = torch.cat((values, scores), -1).float().cpu()
    selected = {}
    live = f.cache.lengths[rid] - f.cache.lengths[rid] % 2
    for row in range(values.shape[0]):
        raw = start + row
        column = raw // pool.span
        if not 0 <= column < table.numel():
            raise ValueError("outside table")
        block = int(table[column])
        if block == -1 and raw < live:
            continue
        if block == -1:
            raise ValueError("missing live")
        if not 0 <= block < output.shape[0]:
            raise ValueError("bad block")
        offset = raw % pool.entries
        previous = selected.get((block, offset))
        if previous is None or raw > previous[0]:
            selected[block, offset] = raw, row
    for (block, offset), (_, row) in selected.items():
        output[block, offset * 1024:(offset + 1) * 1024].copy_(fields[row])
    return output, len(selected)


class _StoreCPU:
    def __getitem__(self, grid):
        def launch(pool, plan, values, scores, width, separate):
            assert separate is True and width == 1024 and grid == (plan.shape[0],)
            flat = pool.view(-1)
            for target, source in plan.tolist():
                flat[target:target + 512].copy_(values[source])
                flat[target + 512:target + 1024].copy_(scores[source])
        return launch


class _Cases:
    def setUp(self):
        env = patch.dict(os.environ, {**FLAGS, ENV: "1", ROTARY: "1", "TP_SIZE": "1",
                                      "PREFILL_CP_SIZE": "1",
                                      "RTP_V41_PPU_LOCAL_PREFILL_SOURCE_NATIVE_READ": "0",
                                      "RTP_V41_PPU_LOCAL_PREFILL_SOURCE_NATIVE_WRITE": "0"})
        env.start()
        self.addCleanup(env.stop)

    @torch.no_grad()
    def test_ieee_exact_pool_oracle_padding_physical_zero_and_real_ring_geometry(self):
        def reorder(tag, table, blocks):
            table.copy_(table.flip(1))
            table[0, 0] = 0
        for entries, span, prefix, count in ((2, 256, 0, 1), (2, 256, 17, 140),
                                            (2, 256, 128, 4096), (2, 512, 129, 4097),
                                            (3, 257, 17, 267), (4, 128, 127, 140)):
            with self.subTest(entries=entries, span=span, count=count):
                old = _fixture(self.device, (prefix,), (count,), entries=entries, span=span, configure=reorder)
                new = _fixture(self.device, (prefix,), (count,), entries=entries, span=span, configure=reorder)
                values, scores = _fields(new)
                expected, selected = _oracle(new, 2, values, scores)
                _old(old, 2, values, scores)
                self.assertTrue(_try(new, 2, values, scores))
                assert_bytes(self, new.cache.pools[2, "csa_state"].tensor, old.cache.pools[2, "csa_state"].tensor)
                assert_bytes(self, new.cache.pools[2, "csa_state"].tensor.cpu(), expected)
                self.assertEqual(new.cache.writes, old.cache.writes)
                stats = new.cache._local_csa_native_write_stats
                self.assertEqual((stats["writes"], stats["rows"], stats["bytes"]), (1, selected, selected * 4096))
                raw = new.cache.native.raw[2, "csa_state"]
                assert_bytes(self, raw[:16], torch.full_like(raw[:16], 173))
                assert_bytes(self, raw[-16:], torch.full_like(raw[-16:], 173))
                pool = new.cache.pools[2, "csa_state"]
                assert_bytes(self, pool.tensor[:, entries * 1024:], torch.full_like(pool.tensor[:, entries * 1024:], 173))

    @torch.no_grad()
    def test_missing_historical_live_negative_oob_short_duplicate_and_error_order(self):
        for bad in ("historical", "all_missing", "all_history", "live", "negative", "oob", "short", "duplicate"):
            def configure(tag, table, blocks):
                if tag != "csa_state":
                    return
                if bad == "short":
                    table.resize_(1, 1).fill_(1)
                elif bad == "duplicate":
                    table.fill_(0)
                elif bad in ("all_missing", "all_history"):
                    table.fill_(-1)
                elif bad == "live":
                    table[0, -1] = -1
                else:
                    table[0, 0] = {"historical": -1, "negative": -2, "oob": blocks}[bad]
            results = []
            for enabled in (False, True):
                with patch.dict(os.environ, {ENV: str(int(enabled))}):
                    f = _fixture(self.device, (0,), (512 if bad == "all_history" else 513,), configure=configure)
                    values, scores = _fields(f)
                    # Duplicate physical destinations intentionally take the
                    # unchanged old latest-raw path, with varied source rows.
                    error = None
                    try:
                        _store(f, 2, values, scores)
                    except (ValueError, IndexError) as exc:
                        error = type(exc), str(exc)
                    results.append((f.cache.pools[2, "csa_state"].tensor.clone(), f.cache.writes,
                                    error, f.cache._local_csa_native_write_stats))
            assert_bytes(self, results[0][0], results[1][0])
            self.assertEqual(results[0][1:3], results[1][1:3])
            self.assertEqual(results[1][3]["launches"], int(bad == "historical"))
        self.assertEqual(_latest_csa_indices([0, 0, 0, 1], [1, 1, 1, 0], [7, 9, 9, 3]), [1, 3])

    @torch.no_grad()
    def test_ragged_empty_append_order_and_readonly_source_alias(self):
        for enabled in (False, True):
            f = _fixture(self.device, (129, 256, 0), (9, 0, 140), configure=lambda tag, table, blocks:
                         table[2].copy_(table[0]) if tag == "csa_state" else None)
            with patch.dict(os.environ, {ENV: str(int(enabled))}):
                for rid in (0, 1, 2):
                    start = f.prefixes[rid] + min(f.lengths[rid], 3)
                    values, scores = _fields(f, rid, start=start)
                    # Input-input alias is read-only and must remain legal.
                    hit = _store(f, 2, values, values, rid, start=start)
                    self.assertEqual(hit, enabled and values.shape[0] > 0)
            result = f.cache.pools[2, "csa_state"].tensor.clone(), list(f.cache.writes)
            if enabled:
                assert_bytes(self, result[0], old[0])
                self.assertEqual(result[1], old[1])
            else:
                old = result

    @torch.no_grad()
    def test_birth_mutation_rebind_nonce_close_compaction_and_fresh_forward(self):
        changes = (
            lambda f: f.cache.inputs["csa_state"].kv_cache_block_id.add_(0),
            lambda f: f.cache.inputs["csa_state"].kv_cache_block_id_device.add_(0),
            lambda f: setattr(f.cache.inputs["csa_state"], "kv_cache_block_id_device",
                              f.cache.inputs["csa_state"].kv_cache_block_id_device.clone()),
            lambda f: f.cache.pools.__setitem__((2, "csa_state"), replace(
                f.cache.pools[2, "csa_state"], tensor=f.cache.pools[2, "csa_state"].tensor.clone())),
            lambda f: setattr(f.cache, "_local_csa_native_write_nonce", object()),
            lambda f: f.cache.close_local_source_native_reads(),
        )
        for change in changes:
            f = _fixture(self.device)
            fields = _fields(f)
            change(f)
            before = f.cache.pools[2, "csa_state"].tensor.clone()
            self.assertFalse(_try(f, 2, *fields))
            self.assertFalse(_try(f, 2, *fields), "an invalid birth cannot be reminted")
            assert_bytes(self, before, f.cache.pools[2, "csa_state"].tensor)
        f = _fixture(self.device, (256,), (4096,))
        self.assertTrue(_try(f, 2, *_fields(f)))
        stats = f.cache._local_csa_native_write_stats
        compact_plan(f.state)
        self.assertFalse(stats["closed"], "query compaction is not the forward end")
        f.cache.close_local_source_native_reads()
        f.cache.close_local_source_native_reads()
        self.assertTrue(stats["closed"])
        fresh = _fixture(self.device)
        self.assertIsNot(stats, fresh.cache._local_csa_native_write_stats)
        self.assertTrue(_try(fresh, 2, *_fields(fresh)))
        SharedNativeCache.__new__(SharedNativeCache).close_local_source_native_reads()

    @torch.no_grad()
    def test_real_producer_main_indexer_csa_order_bytes_odd_carry_and_missing_priority(self):
        ops = PPUOps(num_routed_experts=8, num_local_experts=8)
        for prefixes, lengths, append in (((128,), (140,), None), ((129,), (9,), None),
                                          ((128,), (4096,), {0: 132}), ((129, 256), (9, 0), None)):
            results = []
            for enabled in (False, True):
                with patch.dict(os.environ, {ENV: str(int(enabled))}):
                    f = producer_fixture(self.device, 2, prefixes, lengths)
                    _produce(f, ops, append_starts=append)
                    results.append(([p.tensor.clone() for p in f.source.cache.pools.values()],
                                    list(f.source.cache.writes), f))
            self.assertEqual(results[0][1], results[1][1])
            for old, new in zip(results[0][0], results[1][0]):
                assert_bytes(self, old, new)
            self.assertEqual(results[1][2].source.cache._local_csa_native_write_stats["writes"],
                             sum(n > 0 for n in lengths))
            for rid in range(len(prefixes)):
                tags = [write["tag"] for write in results[1][1] if write["request"] == rid]
                if lengths[rid]:
                    self.assertEqual(tags[-3:], ["global_kv_2", "indexer_kv", "csa_state"])
        errors = []
        for enabled in (False, True):
            with patch.dict(os.environ, {ENV: str(int(enabled))}):
                f = producer_fixture(self.device, 2, (129,), (9,), carry=False)
                with self.assertRaises(RuntimeError) as err:
                    _produce(f, ops)
                errors.append(str(err.exception))
                self.assertEqual(f.source.cache._local_csa_native_write_stats["writes"], 0)
        self.assertEqual(errors[0], errors[1])


class LocalCsaNativeWriteCPU(_Cases, unittest.TestCase):
    device = "cpu"

    def setUp(self):
        super().setUp()
        for item in (
            patch.object(SharedNativeCache, "_local_source_native_read_device", new=lambda self, pool: True),
            patch.object(graph_cache_kernels, "_kernels", new=lambda: NS(store_prefill_csa=_StoreCPU())),
            patch.object(_quant, "try_pack_rotary_fp4_rows", new=lambda kv, freqs, positions, global_pool:
                         encode_kv(prefill._rope_tail(kv, freqs[positions.clamp_max(freqs.shape[0] - 1)]),
                                   pool_tag="global_kv_1" if global_pool else "indexer_kv")),
        ):
            item.start()
            self.addCleanup(item.stop)

    @torch.no_grad()
    def test_default_latch_phase_proof_and_read0_independence(self):
        with patch.dict(os.environ, {ENV: "0"}):
            f = _fixture("cpu")
        self.assertFalse(_try(f, 2, *_fields(f)))
        for env in ({ENV: "0"}, {"TP_SIZE": "8"}, {"PREFILL_CP_SIZE": "8"}):
            with patch.dict(os.environ, env):
                f = _fixture("cpu")
                self.assertFalse(_try(f, 2, *_fields(f)))
        for change in (lambda f: setattr(f.cache, "_is_prefill", False),
                       lambda f: setattr(f.cache, "sharded", True),
                       lambda f: f.proofs.__setitem__(2, None),
                       lambda f: f.args[0].add_(1).sub_(1),
                       lambda f: setattr(f.state, "idle", True),
                       lambda f: setattr(f.state.attn, "is_cuda_graph", True)):
            f = _fixture("cpu")
            change(f)
            self.assertFalse(_try(f, 2, *_fields(f)))
            self.assertEqual(f.cache._local_csa_native_write_stats["launches"], 0)
        f = _fixture("cpu")
        self.assertTrue(_try(f, 2, *_fields(f)), "CSA birth is independent of FULL READ0/WRITE0")
        with torch.enable_grad():
            self.assertFalse(_try(f, 2, *_fields(f)))

    @torch.no_grad()
    def test_view_dtype_target_alias_and_constructor_mismatch_decline_before_mutation(self):
        f = _fixture("cpu")
        values, scores = _fields(f)
        target = f.cache.pools[2, "csa_state"].tensor
        # Use a short actual source range which can alias the existing pool.
        short = _fixture("cpu", (256,), (2,))
        alias = short.cache.pools[2, "csa_state"].tensor.view(-1)[:1024].view(2, 512)
        self.assertFalse(_try(short, 2, alias, alias))
        for left, right in ((values.bfloat16(), scores), (values, scores.bfloat16()),
                            (torch._neg_view(values), scores), (values, torch._neg_view(scores)),
                            (values.to_sparse(), scores), (values[:, ::2], scores),
                            (values[:-1], scores[:-1]), (values[:0], scores[:0])):
            before = target.clone()
            self.assertFalse(_try(f, 2, left, right))
            assert_bytes(self, before, target)
        f = _fixture("cpu")
        f.cache.inputs["csa_state"].kv_cache_block_id[0, 0] += 1
        rebuilt = SharedNativeCache(f.cache.config, f.cache.geometry, f.cache.native, f.cache.inputs,
                                   f.cache.lengths, device="cpu", prefill=True, defer_producer_slots=True)
        self.assertIsNone(rebuilt._verified_local_csa_native_table((2, "csa_state"), rebuilt.pools[2, "csa_state"]))

    @torch.no_grad()
    def test_launch_failure_propagates_without_old_retry(self):
        f = _fixture("cpu")
        calls = []
        class Broken:
            def __getitem__(self, grid):
                def launch(*args):
                    calls.append(grid)
                    raise RuntimeError("CSA launch failed")
                return launch
        with patch.object(graph_cache_kernels, "_kernels", new=lambda: NS(store_prefill_csa=Broken())):
            with self.assertRaisesRegex(RuntimeError, "CSA launch failed"):
                _store(f, 2, *_fields(f))
        self.assertEqual(len(calls), 1)
        self.assertFalse(f.cache.writes)
        self.assertEqual(f.cache._local_csa_native_write_stats["launches"], 1)
        self.assertEqual(f.cache._local_csa_native_write_stats["writes"], 0)

    @torch.no_grad()
    def test_constructor_strong_owners_are_released_only_at_close(self):
        f = _fixture("cpu")
        def rebind():
            pool = f.cache.pools[2, "csa_state"]
            meta = f.cache.inputs["csa_state"]
            owners = pool.tensor, pool.handle.kv_cache_base, meta.kv_cache_block_id, meta.kv_cache_block_id_device
            refs = [weakref.ref(value) for value in owners]
            new = pool.tensor.clone()
            pool.handle.kv_cache_base = new
            f.cache.pools[2, "csa_state"] = replace(pool, tensor=new, table=pool.table.clone())
            f.cache.native.raw[2, "csa_state"] = f.cache.native.raw[2, "csa_state"].clone()
            for field in ("kv_cache_block_id", "kv_cache_block_id_device",
                          "kv_cache_kernel_block_id", "kv_cache_kernel_block_id_device"):
                setattr(meta, field, getattr(meta, field).clone())
            return refs
        refs = rebind()
        gc.collect()
        self.assertTrue(all(ref() is not None for ref in refs))
        self.assertFalse(_try(f, 2, *_fields(f)))
        f.cache.close_local_source_native_reads()
        gc.collect()
        self.assertTrue(all(ref() is None for ref in refs))


@unittest.skipUnless(os.environ.get("RTP_V41_TEST_LOCAL_WORKSPACE_DEVICE") == "1", "explicit real PPU test")
class LocalCsaNativeWriteDevice(_Cases, unittest.TestCase):
    device = "cuda:0"

    def setUp(self):
        super().setUp()
        self.assertTrue(_quant.is_ppu(), "requires an actual PPU")
        self.assertEqual(torch.cuda.get_device_capability(), (8, 9))

    @torch.no_grad()
    def test_actual_three_4096_field_writes_native_profile_and_no_collectives(self):
        ratios = {2: 2, 8: 2, 14: 2}
        old = _fixture(self.device, (126976,), (4096,), ratios=ratios, blocks=2880)
        new = _fixture(self.device, (126976,), (4096,), ratios=ratios, blocks=2880)
        fields = {source: _fields(new) for source in ratios}
        for source in ratios:
            self.assertTrue(_try(new, source, *fields[source]))
            _old(old, source, *fields[source])
        stats = new.cache._local_csa_native_write_stats
        before = {key: stats[key] for key in ("writes", "rows", "launches", "bytes", "metadata_uploads")}
        with patch.object(cp.collective_torch, "_get_group", side_effect=AssertionError("no CP group")), \
                patch("torch.distributed.all_reduce", side_effect=AssertionError("no collective")), \
                torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                   torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as profile:
            for source in ratios:
                self.assertTrue(_try(new, source, *fields[source]))
        torch.cuda.synchronize()
        events = _trace_events(profile, "csa-native-three-writes")
        kernels = [str(event.get("name", "")) for event in events if event.get("cat") == "kernel"]
        self.assertEqual(sum("store_prefill_csa" in name for name in kernels), 3)
        cpu_names = [event.get("name") for event in events if event.get("cat") == "cpu_op"]
        for removed in ("aten::cat", "aten::_unique2", "aten::nonzero", "aten::_local_scalar_dense"):
            self.assertNotIn(removed, cpu_names)
        self.assertEqual(stats["writes"] - before["writes"], 3)
        self.assertEqual(stats["launches"] - before["launches"], 3)
        self.assertEqual(stats["metadata_uploads"] - before["metadata_uploads"], 3)
        selected = sum(g["selected_rows"] for g in stats["geometries"][-3:])
        self.assertEqual(stats["rows"] - before["rows"], selected)
        self.assertEqual(stats["bytes"] - before["bytes"], selected * 4096)
        self.assertEqual(stats["table_mirrors"], 1)
        self.assertEqual(stats["fallbacks"], 0)
        for source in ratios:
            assert_bytes(self, new.cache.pools[source, "csa_state"].tensor, old.cache.pools[source, "csa_state"].tensor)
        json.dumps(stats)

    @torch.no_grad()
    def test_default_stream_capture_readonly_inference_and_metadata_guard(self):
        with torch.inference_mode():
            f = _fixture(self.device)
            fields = _fields(f)
            torch.cuda.synchronize()
            with torch.cuda.stream(torch.cuda.Stream()):
                self.assertFalse(_try(f, 2, *fields))
            torch.cuda.synchronize()
            self.assertTrue(_try(f, 2, *fields))
            before = f.cache.pools[2, "csa_state"].tensor.clone()
            with patch.object(torch.cuda, "is_current_stream_capturing", return_value=True):
                self.assertFalse(_try(f, 2, *fields))
            assert_bytes(self, before, f.cache.pools[2, "csa_state"].tensor)
            f.cache.close_local_source_native_reads()


if __name__ == "__main__":
    unittest.main()
