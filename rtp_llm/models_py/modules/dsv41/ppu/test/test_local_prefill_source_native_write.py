"""Local FULL stores: unchanged native kernel, producer proof and byte ABI.

CPU cases substitute only physical device admission / native launch. Device
cases require real PPU89; no distributed group or collective is constructed.
The source-native-read, rotary-pack and SWA-segment fixture modules are data
dependencies of both registrations. The original tests are left unchanged.
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
from rtp_llm.models_py.modules.dsv41.ppu import _quant, graph_cache_kernels, prefill, prefill_readback
from rtp_llm.models_py.modules.dsv41.ppu.kv_codec import decode_kv, encode_kv
from rtp_llm.models_py.modules.dsv41.ppu.native_cache import _WIDTH
from rtp_llm.models_py.modules.dsv41.ppu.native_cache_adapter import (
    LOCAL_PREFILL_SOURCE_CACHE_ENV as SCC,
    LOCAL_PREFILL_SOURCE_NATIVE_READ_ENV as READ,
    LOCAL_PREFILL_SOURCE_NATIVE_WRITE_ENV as ENV,
    SharedNativeCache,
)
from rtp_llm.models_py.modules.dsv41.ppu.ops import PPUOps
from rtp_llm.models_py.modules.dsv41.ppu.replay_policy import SharedPrefillPolicy
from rtp_llm.models_py.modules.dsv41.ppu.test.test_local_prefill_fused import assert_bytes
from rtp_llm.models_py.modules.dsv41.ppu.test.test_local_prefill_operator_flags import _trace_events
from rtp_llm.models_py.modules.dsv41.ppu.test.test_local_prefill_rotary_pack import (
    _fixture as producer_fixture, _produce,
)
from rtp_llm.models_py.modules.dsv41.ppu.test.test_local_prefill_rows import state_for
from rtp_llm.models_py.modules.dsv41.ppu.test.test_local_prefill_source_native_read import (
    _RestoreCPU, cpu_address_oracle, fixture as full_fixture, original, read,
)
from rtp_llm.models_py.modules.dsv41.ppu.test.test_local_prefill_swa_segments import (
    FLAGS, arguments, compact_plan,
)


ROTARY = prefill.LOCAL_PREFILL_ROTARY_PACK_ENV


class _StoreCPU:
    """Address-ABI double, never evidence that a PPU kernel was executed."""

    def __getitem__(self, grid):
        def launch(pool, table, rows, start, first, stride, table_stride, width,
                   entries, page, owner, size, rank, planar, block):
            assert size == 1 and rank == 0 and owner == page and first == start
            assert stride == pool.stride(0) and table_stride == table.stride(0)
            for index in range(grid[0]):
                position = first + index
                physical, offset = int(table[position // page]), position % page
                if planar:
                    pool[physical, offset * 64:offset * 64 + 64].copy_(rows[index, :64])
                    begin = entries * 64 + offset * 4
                    pool[physical, begin:begin + 4].copy_(rows[index, 64:])
                else:
                    pool[physical, offset * width:(offset + 1) * width].copy_(rows[index])
        return launch


def _fixture(device, prefixes=(128,), lengths=(140,), *, ratios=None, **kwargs):
    """Real original FULL-table birth and real full-row creation capability.

    The SWA layer0 ticket is a row-order stamp, independent of FULL payloads.
    The existing proof constructor explicitly binds that stamp to each source;
    no CP context, placement or native table is synthesized after construction.
    """
    cache = full_fixture(device, tuple(p + n for p, n in zip(prefixes, lengths)),
                         ratios=ratios or {2: 2}, **kwargs)
    state = state_for(device, prefixes, lengths)
    state.cache, state.runtime.policy = cache, SharedPrefillPolicy(mode="compact")
    state.runtime.model._is_speculative = False
    args = arguments(state)
    proofs = {source: prefill._make_local_producer_rotary_proof(
        args[-1], cache=cache, layer_id=source, prefixes=prefixes, lengths=lengths,
    ) for source in cache.geometry.kv_source_layer_ids}
    return NS(cache=cache, state=state, args=args, proofs=proofs,
              prefixes=tuple(prefixes), lengths=tuple(lengths))


def _rows(f, source, tag, rid=0, *, start=None):
    ratio = f.cache.geometry.ratio(source)
    start = f.prefixes[rid] if start is None else start
    count = (f.prefixes[rid] + f.lengths[rid]) // ratio - start // ratio
    width = _WIDTH[tag]
    # All byte values, including poison FP4 payload/scales and special FP8
    # encodings: stores must not interpret or canonicalize the existing codec.
    value = (torch.arange(count * width, device=f.cache.device).reshape(count, width) * 17 + 29) % 256
    return value.to(torch.uint8)


def _old(f, source, tag, rows, rid=0, *, start=None):
    start = f.prefixes[rid] if start is None else start
    positions = torch.arange(rows.shape[0], device=f.cache.device) * f.cache.geometry.ratio(source) + start
    slots = f.cache.slots_at_positions(source, tag, rid, positions)
    f.cache.insert_rows(source, tag, slots, rows)


def _try(f, source, tag, rows, rid=0, *, start=None, proof=None):
    return f.cache.try_insert_local_source_rows(
        source, tag, rid, f.prefixes[rid] if start is None else start, rows,
        proof=f.proofs[source] if proof is None else proof,
        prefill_layout=(f.prefixes, f.lengths),
    )


def _store(f, source, tag, rows, rid=0, *, start=None):
    hit = _try(f, source, tag, rows, rid, start=start)
    if not hit:
        _old(f, source, tag, rows, rid, start=start)
    return hit


def _pool_oracle(before, table, *, start, ratio, span, entries, tag, rows):
    """Independent CPU raw-position -> physical byte address scatter."""
    expected, table, rows = before.detach().cpu().clone(), table.detach().cpu(), rows.detach().cpu()
    width, column = rows.shape[1], torch.arange(rows.shape[1])[None]
    for begin in range(0, rows.shape[0], 512):
        end = min(begin + 512, rows.shape[0])
        raw = start + torch.arange(begin, end) * ratio
        offset = (raw % span) // ratio
        address = (torch.where(column < 64, offset[:, None] * 64 + column,
                               entries * 64 + offset[:, None] * 4 + column - 64)
                   if tag == "indexer_kv" else offset[:, None] * width + column)
        expected[table[raw // span].long()[:, None], address] = rows[begin:end]
    return expected


class _Cases:
    def setUp(self):
        env = patch.dict(os.environ, {**FLAGS, ENV: "1", READ: "1", SCC: "0", ROTARY: "1",
                                      "TP_SIZE": "1", "PREFILL_CP_SIZE": "1",
                                      "RTP_V41_PPU_LOCAL_PREFILL_SWA_NATIVE_READ": "0"})
        env.start()
        self.addCleanup(env.stop)

    @torch.no_grad()
    def test_true_geometry_raw_byte_oracle_padding_zero_and_reordered_pages(self):
        def reorder(tag, table, blocks):
            table.copy_(table.flip(1))
            table[0, 0] = 0
        for ratio, prefix, length in ((1, 0, 1), (1, 17, 140), (1, 256, 4096),
                                      (1, 17, 4097), (2, 0, 2), (2, 128, 140),
                                      (2, 256, 4096), (2, 128, 514)):
            with self.subTest(ratio=ratio, prefix=prefix, length=length):
                old = _fixture(self.device, (prefix,), (length,), ratios={2: ratio}, configure=reorder)
                f = _fixture(self.device, (prefix,), (length,), ratios={2: ratio}, configure=reorder)
                self.assertIsNotNone(f.proofs[2])
                tag = "global_kv_2" if ratio == 2 else "global_kv_1"
                for kind in (tag, "indexer_kv"):
                    rows = _rows(f, 2, kind)
                    pool = f.cache.pools[2, kind]
                    oracle = _pool_oracle(pool.tensor, pool.table[0], start=prefix, ratio=ratio,
                                          span=pool.span, entries=pool.entries, tag=kind, rows=rows)
                    _old(old, 2, kind, rows)
                    self.assertTrue(_try(f, 2, kind, rows))
                    assert_bytes(self, pool.tensor, old.cache.pools[2, kind].tensor)
                    assert_bytes(self, pool.tensor.cpu(), oracle)
                    assert_bytes(self, pool.tensor[:, pool.entries * _WIDTH[kind]:],
                                 torch.full_like(pool.tensor[:, pool.entries * _WIDTH[kind]:], 173))
                    raw = f.cache.native.raw[2, pool.framework_tag]
                    assert_bytes(self, raw[:16], torch.full_like(raw[:16], 173))
                    assert_bytes(self, raw[-16:], torch.full_like(raw[-16:], 173))
                self.assertEqual(old.cache.writes, f.cache.writes)
                for kind, got, expected in zip((tag, "indexer_kv"), read(f.cache, 2), original(old.cache, 2)):
                    assert_bytes(self, got, expected)
                    assert_bytes(self, got, cpu_address_oracle(f.cache, (2, kind), 0, got.shape[0]))
                self.assertEqual(f.cache._local_source_native_write_stats["writes"], 2)
                self.assertIsNone(f.cache._slots, "new stores do not materialize generic slot metadata")

    @torch.no_grad()
    def test_ragged_empty_suffix_and_per_request_order(self):
        for ratio in (1, 2):
            for enabled in (False, True):
                f = _fixture(self.device, (256, 512, 0), (140, 0, 8), ratios={2: ratio})
                tag = "global_kv_2" if ratio == 2 else "global_kv_1"
                with patch.dict(os.environ, {ENV: str(int(enabled))}):
                    for rid in (0, 1, 2):
                        for kind in (tag, "indexer_kv"):
                            start = f.prefixes[rid] + min(f.lengths[rid], 4)
                            rows = _rows(f, 2, kind, rid, start=start)
                            hit = _store(f, 2, kind, rows, rid, start=start)
                            self.assertEqual(hit, enabled and rows.shape[0] > 0)
                saved = tuple(pool.tensor.clone() for pool in f.cache.pools.values())
                if enabled:
                    for got, want in zip(saved, old):
                        assert_bytes(self, got, want)
                    self.assertEqual(f.cache.writes, receipts)
                else:
                    old, receipts = saved, list(f.cache.writes)

    @torch.no_grad()
    def test_payload_cache_independence_aliases_invalidate_all_banks_before_native_write(self):
        for scc in ("0", "1"):
            with patch.dict(os.environ, {SCC: scc}):
                f = _fixture(self.device, (256, 256), (140, 140), ratios={2: 1, 20: 1}, alias=True,
                             configure=lambda tag, table, blocks: table[1].copy_(table[0]))
                for source in (2, 20):
                    for rid in (0, 1):
                        read(f.cache, source, rid)
                self.assertEqual(len(f.cache._source_reads), 4 if scc == "1" else 0)
                for kind in ("global_kv_1", "indexer_kv"):
                    rows = _rows(f, 20, kind, 1)
                    self.assertTrue(_try(f, 20, kind, rows, 1))
                    self.assertFalse(f.cache._source_reads or f.cache._local_source_reads)
                    for source in (2, 20):
                        for rid in (0, 1):
                            for got, expected in zip(read(f.cache, source, rid), original(f.cache, source, rid)):
                                assert_bytes(self, got, expected)
                json.dumps(f.cache._local_source_native_write_stats)

    @torch.no_grad()
    def test_unsupported_duplicate_negative_oob_short_and_original_error_priority(self):
        for bad in ("duplicate", "negative", "oob", "short", "indexer_bad", "both_bad"):
            def configure(tag, table, blocks):
                if bad == "short":
                    table.resize_(1, 1).fill_(1)
                elif bad == "duplicate":
                    table.fill_(0)
                elif bad in ("negative", "oob", "both_bad") or (bad == "indexer_bad" and tag == "indexer_kv"):
                    table[0, 1] = blocks if bad == "oob" else -1
            observations = []
            for enabled in (False, True):
                f = _fixture(self.device, (256,), (512,), ratios={2: 2}, configure=configure)
                error = None
                with patch.dict(os.environ, {ENV: str(int(enabled))}):
                    try:
                        for kind in ("global_kv_2", "indexer_kv"):
                            rows = torch.full_like(_rows(f, 2, kind), 37)
                            _store(f, 2, kind, rows)
                    except (ValueError, IndexError) as exc:
                        error = (type(exc), str(exc))
                observations.append((error, [pool.tensor.clone() for pool in f.cache.pools.values()], f))
            self.assertEqual(observations[0][0], observations[1][0], bad)
            for old, new in zip(observations[0][1], observations[1][1]):
                assert_bytes(self, old, new)
            self.assertEqual(observations[1][2].cache._local_source_native_write_stats["launches"], int(bad == "indexer_bad"))

    @torch.no_grad()
    def test_table_birth_mutation_rebind_nonce_close_and_new_forward(self):
        changes = (
            lambda f: f.cache.inputs["csa_kv"].kv_cache_kernel_block_id.add_(0),
            lambda f: f.cache.inputs["csa_kv"].kv_cache_kernel_block_id_device.add_(0),
            lambda f: setattr(f.cache.inputs["csa_kv"], "kv_cache_kernel_block_id_device",
                              f.cache.inputs["csa_kv"].kv_cache_kernel_block_id_device.clone()),
            lambda f: f.cache.pools.__setitem__((2, "global_kv_2"), replace(
                f.cache.pools[2, "global_kv_2"], tensor=f.cache.pools[2, "global_kv_2"].tensor.clone())),
            lambda f: setattr(f.cache, "_local_source_native_read_nonce", object()),
            lambda f: f.cache.close_local_source_native_reads(),
        )
        for change in changes:
            f = _fixture(self.device)
            rows = _rows(f, 2, "global_kv_2")
            change(f)
            before = f.cache.pools[2, "global_kv_2"].tensor.clone()
            self.assertFalse(_try(f, 2, "global_kv_2", rows))
            assert_bytes(self, before, f.cache.pools[2, "global_kv_2"].tensor)
            self.assertFalse(_try(f, 2, "global_kv_2", rows), "invalid birth is never reminted")
        f = _fixture(self.device)
        stats = f.cache._local_source_native_write_stats
        f.cache.close_local_source_native_reads()
        f.cache.close_local_source_native_reads()
        self.assertTrue(stats["closed"])
        fresh = _fixture(self.device)
        self.assertIsNot(stats, fresh.cache._local_source_native_write_stats)
        self.assertTrue(_try(fresh, 2, "global_kv_2", _rows(fresh, 2, "global_kv_2")))
        SharedNativeCache.__new__(SharedNativeCache).close_local_source_native_reads()

    @torch.no_grad()
    def test_query_compaction_preserves_full_birth_and_l20_store_then_read(self):
        f = _fixture(self.device, (256,), (4096,), ratios={20: 1})
        for kind in ("global_kv_1", "indexer_kv"):
            self.assertTrue(_try(f, 20, kind, _rows(f, 20, kind)))
        compact_plan(f.state)
        for got, expected in zip(read(f.cache, 20), original(f.cache, 20)):
            assert_bytes(self, got, expected)
        self.assertFalse(f.cache._local_source_native_read_stats["closed"])
        self.assertFalse(f.cache._local_source_native_write_stats["closed"])
        self.assertEqual(f.cache._local_source_native_read_stats["reads"], 2)

    @torch.no_grad()
    def test_producer_old_bytes_csa_receipts_order_lazy_slots_and_odd_carry_fallback(self):
        ops = PPUOps(num_routed_experts=8, num_local_experts=8)
        for ratio, prefixes, lengths, append in (
            (1, (17,), (140,), None), (1, (17,), (4097,), {0: 21}),
            (2, (128,), (140,), None), (2, (128, 256), (140, 0), {0: 132}),
            (2, (129,), (9,), None), (2, (128,), (141,), None),
            (2, (0,), (1,), None), (1, (17, 9), (0, 0), None),
        ):
            results = []
            for enabled in (False, True):
                with patch.dict(os.environ, {ENV: str(int(enabled))}):
                    f = producer_fixture(self.device, ratio, prefixes, lengths)
                    self.assertIsNotNone(f.proof)
                    _produce(f, ops, append_starts=append)
                results.append((f, [pool.tensor.clone() for pool in f.source.cache.pools.values()]))
            self.assertEqual(results[0][0].source.cache.writes, results[1][0].source.cache.writes)
            for old, new in zip(results[0][1], results[1][1]):
                assert_bytes(self, old, new)
            if ratio == 1 or (all(p % 2 == 0 for p in prefixes) and all(n % 2 == 0 for n in lengths)):
                self.assertIsNone(results[1][0].source.cache._slots)
        errors = []
        for enabled in (False, True):
            with patch.dict(os.environ, {ENV: str(int(enabled))}):
                f = producer_fixture(self.device, 2, (129,), (9,), carry=False)
                with self.assertRaises(RuntimeError) as err:
                    _produce(f, ops)
                errors.append(str(err.exception))
                self.assertEqual(f.source.cache._local_source_native_write_stats["launches"], 0)
        self.assertEqual(errors[0], errors[1])

    @torch.no_grad()
    def test_packed_codec_decode_and_downstream_bf16_bytes_are_unchanged(self):
        for ratio in (1, 2):
            old, new = (_fixture(self.device, (256,), (140,), ratios={2: ratio}) for _ in range(2))
            tag = "global_kv_2" if ratio == 2 else "global_kv_1"
            for kind, width in ((tag, 512), ("indexer_kv", 128)):
                count = 140 // ratio
                values = (torch.arange(count * width, device=self.device).reshape(count, width) % 61 - 30).bfloat16()
                values.view(-1)[:4] = torch.tensor([float("nan"), float("inf"), -0.0, -float("inf")],
                                                  device=self.device, dtype=torch.bfloat16)
                rows = encode_kv(values.contiguous(), pool_tag=kind)
                _old(old, 2, kind, rows)
                self.assertTrue(_try(new, 2, kind, rows))
            for kind, got, expected, dim in zip((tag, "indexer_kv"), read(new.cache, 2),
                                                original(old.cache, 2), (512, 128)):
                assert_bytes(self, got, expected)
                left, right = (torch.empty((got.shape[0], dim), device=self.device, dtype=torch.bfloat16)
                               for _ in range(2))
                decode_kv(got, pool_tag=kind, out=left)
                decode_kv(expected, pool_tag=kind, out=right)
                assert_bytes(self, left, right)
                # Existing downstream BF16 landing, with identical operands.
                assert_bytes(self, left.float().sum(-1).bfloat16(), right.float().sum(-1).bfloat16())


class LocalSourceNativeWriteCPU(_Cases, unittest.TestCase):
    device = "cpu"

    def setUp(self):
        super().setUp()
        patches = (
            patch.object(SharedNativeCache, "_local_source_native_read_device", new=lambda self, pool: True),
            patch.object(graph_cache_kernels, "_kernels", new=lambda: NS(
                store_cp_prefill_rows=_StoreCPU(), triton=NS(next_power_of_2=lambda n: 1 << (n - 1).bit_length()))),
            patch.object(prefill_readback, "_restore_rows", _RestoreCPU()),
            patch("rtp_llm.models_py.modules.dsv41.fp8._v41_swa_metadata._upload",
                  new=lambda rows, device, dtype: torch.tensor(rows, dtype=dtype, device=device)),
            patch.object(_quant, "try_pack_rotary_fp4_rows", new=lambda kv, freqs, positions, global_pool:
                         encode_kv(prefill._rope_tail(kv, freqs[positions.clamp_max(freqs.shape[0] - 1)]),
                                   pool_tag="global_kv_1" if global_pool else "indexer_kv")),
        )
        for item in patches:
            item.start()
            self.addCleanup(item.stop)

    @torch.no_grad()
    def test_default_latch_read0_phase_and_proof_fail_before_store(self):
        for env in ({ENV: "0"}, {READ: "0"}, {"TP_SIZE": "8"}, {"PREFILL_CP_SIZE": "8"}):
            with patch.dict(os.environ, env):
                f = _fixture("cpu")
                before = f.cache.pools[2, "global_kv_2"].tensor.clone()
                self.assertFalse(_try(f, 2, "global_kv_2", _rows(f, 2, "global_kv_2")))
                assert_bytes(self, before, f.cache.pools[2, "global_kv_2"].tensor)
        with patch.dict(os.environ, {ENV: "0"}):
            f = _fixture("cpu")
        self.assertFalse(_try(f, 2, "global_kv_2", _rows(f, 2, "global_kv_2")))
        for mutate in (lambda f: setattr(f.cache, "_is_prefill", False),
                       lambda f: setattr(f.cache, "sharded", True),
                       lambda f: f.proofs.__setitem__(2, None),
                       lambda f: f.args[0].add_(1).sub_(1),
                       lambda f: setattr(f.state, "lengths", (139,))):
            f = _fixture("cpu")
            mutate(f)
            self.assertFalse(_try(f, 2, "global_kv_2", _rows(f, 2, "global_kv_2")))
            self.assertEqual(f.cache._local_source_native_write_stats["launches"], 0)
        f = _fixture("cpu")
        with torch.enable_grad():
            self.assertFalse(_try(f, 2, "global_kv_2", _rows(f, 2, "global_kv_2")))

    @torch.no_grad()
    def test_raw_views_alias_width_empty_odd_and_non_suffix_decline_without_mutation(self):
        f = _fixture("cpu")
        pool, base = f.cache.pools[2, "global_kv_2"], _rows(f, 2, "global_kv_2")
        alias = pool.tensor.flatten()[:base.numel()].view_as(base)
        for rows, start in ((torch._neg_view(base), 128), (base.to_sparse(), 128), (base[:, ::2], 128),
                            (base.long(), 128), (base[:0], 128), (alias, 128),
                            (base[:-1], 128), (base, 129)):
            before = pool.tensor.clone()
            self.assertFalse(_try(f, 2, "global_kv_2", rows, start=start))
            assert_bytes(self, before, pool.tensor)
        for prefix, length in ((129, 9), (128, 141)):
            odd = _fixture("cpu", (prefix,), (length,))
            self.assertFalse(_try(odd, 2, "global_kv_2", _rows(odd, 2, "global_kv_2")))
        # Source views with offsets and read-only aliases remain legal.
        raw = torch.full((base.numel() + 16,), 91, dtype=torch.uint8)
        rows = raw[8:-8].view_as(base)
        snapshot = raw.clone()
        self.assertTrue(_try(f, 2, "global_kv_2", rows))
        assert_bytes(self, raw, snapshot)

    @torch.no_grad()
    def test_launch_error_propagates_and_payload_cache_clears_before_failure(self):
        with patch.dict(os.environ, {SCC: "1"}):
            f = _fixture("cpu")
            read(f.cache, 2)
            calls = []
            class Broken:
                def __getitem__(self, grid):
                    def launch(*args):
                        calls.append(grid)
                        self_test.assertFalse(f.cache._source_reads or f.cache._local_source_reads)
                        raise RuntimeError("native launch failed")
                    return launch
            self_test = self
            with patch.object(graph_cache_kernels, "_kernels", new=lambda: NS(
                store_cp_prefill_rows=Broken(), triton=NS(next_power_of_2=lambda n: 512))):
                with self.assertRaisesRegex(RuntimeError, "native launch failed"):
                    _store(f, 2, "global_kv_2", _rows(f, 2, "global_kv_2"))
            self.assertEqual(len(calls), 1)
            self.assertFalse(f.cache.writes)
            self.assertEqual(f.cache._local_source_native_write_stats["launches"], 1)
            self.assertEqual(f.cache._local_source_native_write_stats["writes"], 0)

    @torch.no_grad()
    def test_constructor_strong_owners_survive_replacement_until_final_close(self):
        f = _fixture("cpu")
        key = (2, "global_kv_2")
        tensors = (f.cache.pools[key].tensor, f.cache.pools[key].handle.kv_cache_base,
                   f.cache.inputs["csa_kv"].kv_cache_kernel_block_id,
                   f.cache.inputs["csa_kv"].kv_cache_kernel_block_id_device)
        refs = [weakref.ref(value) for value in tensors]
        pool = f.cache.pools[key]
        pool.handle.kv_cache_base = pool.handle.kv_cache_base.clone()
        f.cache.pools[key] = replace(pool, tensor=pool.tensor.clone(), table=pool.table.clone())
        f.cache.native.raw[2, "csa_kv"] = f.cache.native.raw[2, "csa_kv"].clone()
        meta = f.cache.inputs["csa_kv"]
        meta.kv_cache_kernel_block_id = meta.kv_cache_kernel_block_id.clone()
        meta.kv_cache_kernel_block_id_device = meta.kv_cache_kernel_block_id_device.clone()
        meta.kv_cache_block_id = meta.kv_cache_kernel_block_id
        meta.kv_cache_block_id_device = meta.kv_cache_kernel_block_id_device
        del tensors, pool
        gc.collect()
        self.assertTrue(all(ref() is not None for ref in refs))
        self.assertFalse(_try(f, 2, "global_kv_2", _rows(f, 2, "global_kv_2")))
        f.cache.close_local_source_native_reads()
        gc.collect()
        self.assertTrue(all(ref() is None for ref in refs))


@unittest.skipUnless(os.environ.get("RTP_V41_TEST_LOCAL_WORKSPACE_DEVICE") == "1", "explicit real PPU test")
class LocalSourceNativeWriteDevice(_Cases, unittest.TestCase):
    device = "cuda:0"

    def setUp(self):
        super().setUp()
        self.assertTrue(_quant.is_ppu(), "requires an actual PPU")
        self.assertEqual(torch.cuda.get_device_capability(), (8, 9))

    @torch.no_grad()
    def test_actual_four_sources_512_pages_2880_blocks_eight_store_profile(self):
        ratios = {2: 2, 8: 2, 14: 2, 20: 1}
        def reorder(tag, table, blocks):
            table.copy_(table.flip(1))
            table[0, -1] = 0
        old = _fixture(self.device, (126976,), (4096,), ratios=ratios, blocks=2880,
                       padding=0, configure=reorder)
        new = _fixture(self.device, (126976,), (4096,), ratios=ratios, blocks=2880,
                       padding=0, configure=reorder)
        payloads = {(source, tag): _rows(new, source, tag) for source, ratio in ratios.items()
                    for tag in ("global_kv_2" if ratio == 2 else "global_kv_1", "indexer_kv")}
        # Compile exactly the existing four width/ratio specializations before
        # capture. The profiler still executes all eight real target stores.
        for (source, tag), rows in payloads.items():
            self.assertTrue(_try(new, source, tag, rows))
            _old(old, source, tag, rows)
        stats = new.cache._local_source_native_write_stats
        before = {k: stats[k] for k in ("writes", "rows", "launches", "bytes")}
        for name in ("all_reduce", "all_gather", "reduce_scatter_tensor", "barrier"):
            guard = patch("torch.distributed." + name, side_effect=AssertionError("local FULL store has no collective"))
            guard.start()
            self.addCleanup(guard.stop)
        with patch.object(cp.collective_torch, "_get_group", side_effect=AssertionError("no CP group")), torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], record_shapes=True,
        ) as profile:
            for (source, tag), rows in payloads.items():
                self.assertTrue(_try(new, source, tag, rows))
        torch.cuda.synchronize()
        events = _trace_events(profile, "source-native-write-eight-full")
        kernels = [str(event.get("name", "")) for event in events if event.get("cat") == "kernel"]
        self.assertEqual(sum("store_cp_prefill_rows" in name for name in kernels), 8)
        cpu = [event for event in events if event.get("cat") == "cpu_op"]
        self.assertFalse(any(event.get("name") in ("aten::_unique2", "aten::nonzero") for event in cpu))
        self.assertFalse(any("index_put" in name or "IndexKernel" in name for name in kernels))
        self.assertEqual({k: stats[k] - before[k] for k in before},
                         dict(writes=8, rows=20480, launches=8, bytes=3645440))
        self.assertEqual(new.cache._local_source_native_read_stats["mirrors"], 3)
        self.assertEqual(stats["fallbacks"], 0)
        for key, pool in new.cache.pools.items():
            assert_bytes(self, pool.tensor, old.cache.pools[key].tensor)
        # Fresh reader kernels consume the changed pages at the original call
        # point; no payload cache or FULL metadata close at query compaction.
        compact_plan(new.state)
        for source in ratios:
            for got, expected in zip(read(new.cache, source), original(old.cache, source)):
                assert_bytes(self, got, expected)
        self.assertEqual(new.cache._local_source_native_read_stats["reads"], 8)
        self.assertEqual(new.cache._local_source_native_read_stats["ids_uploads"], 3)
        self.assertEqual(new.cache._local_source_native_read_stats["plan_uploads"], 2)
        new.cache.close_local_source_native_reads()
        self.assertTrue(stats["closed"])
        json.dumps(stats)

    @torch.no_grad()
    def test_default_stream_capture_guard_readonly_inference_and_no_cp(self):
        with torch.inference_mode():
            f = _fixture(self.device)
            rows = _rows(f, 2, "global_kv_2")
            torch.cuda.synchronize()
            with torch.cuda.stream(torch.cuda.Stream()):
                self.assertFalse(_try(f, 2, "global_kv_2", rows))
            torch.cuda.synchronize()
            self.assertTrue(_try(f, 2, "global_kv_2", rows))
            snapshot = f.cache.pools[2, "global_kv_2"].tensor.clone()
            with patch.object(torch.cuda, "is_current_stream_capturing", return_value=True):
                self.assertFalse(_try(f, 2, "global_kv_2", rows))
            assert_bytes(self, snapshot, f.cache.pools[2, "global_kv_2"].tensor)
            f.cache.close_local_source_native_reads()
        fresh = _fixture(self.device)
        self.assertTrue(_try(fresh, 2, "global_kv_2", _rows(fresh, 2, "global_kv_2")))


if __name__ == "__main__":
    unittest.main()
