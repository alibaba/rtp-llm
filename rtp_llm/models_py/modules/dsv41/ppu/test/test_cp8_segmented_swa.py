"""Remove only unreferenced zigzag SWA holes from the BF16 workspace.

CPU fixtures exercise the real native-row birth and workspace control flow;
their codec is an explicit byte-tag double. Explicit device cases compare
the existing SWA dequantization and actual 6/10-field index assemblers.
Neither case changes collectives or claims model latency evidence.
"""

from contextlib import ExitStack
from dataclasses import replace
import os
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv41.ppu import _quant, kv_codec, prefill
from rtp_llm.models_py.modules.dsv41.ppu.cp_prefill_shared_global import SharedGlobalBank
from rtp_llm.models_py.modules.dsv41.ppu.native_cache_adapter import _StagedSwaSegments
from rtp_llm.models_py.modules.dsv41.ppu.test.test_cp8_cross_layer_mhc import CP_ENV, _state
from rtp_llm.models_py.modules.dsv41.ppu.test.test_cp8_l20_query_projection import _context


def _tagged(positions, request, layer, width):
    result = torch.zeros((positions.numel(), width), dtype=torch.uint8, device=positions.device)
    result[:, 0] = (positions % 251).to(torch.uint8)
    result[:, 1] = (positions // 251 % 251).to(torch.uint8)
    result[:, 2] = (request + layer * 13) % 251
    return result


def _decode_tagged(packed, *, pool_tag, descriptor, out):
    out.zero_()
    out[:, :3].copy_(packed[:, :3])
    return out


def _fixture(rank=0, *, ratio=1, shared_global=True, lengths=(1024, 1024),
             prefixes=(256, 256), device="cpu"):
    cp, ids = _context(rank, lengths, prefixes, device)
    state = _state(cp, ids)
    geometry = NS(head_dim=512, window=128, ratio=lambda layer: ratio,
                  kv_source_for=lambda layer: 2 if ratio else None)
    cache = state.cache
    cache.device, cache.cp, cache.geometry = torch.device(device), dict(size=8, rank=rank), geometry
    cache.cp8_row_prep_enabled = lambda: True
    cache.prefill_plans = None
    cache.descriptors = lambda layer: dict(swa_kv=None, global_kv_1=None, global_kv_2=None)
    prepared = []
    cache.prepare_source = prepared.append
    cache.try_decode_swa_range = lambda *args: False
    cache.read_swa = lambda layer, request, positions: _tagged(positions, request, layer, 528)
    cache._cp_shared_global_stats = dict(dequant_launches=0, dequant_rows=0, workspace_hits=0)
    bank = None
    globals_by_request = []
    if ratio:
        counts = tuple(total // ratio for total in cache.lengths)
        common = min(64, min(counts))
        prefix = _tagged(torch.arange(common, device=device), 0, 30, 288)
        tails = tuple(_tagged(torch.arange(common, count, device=device), rid + 1, 30, 288)
                      for rid, count in enumerate(counts))
        bank = SharedGlobalBank.create(torch.cat((prefix, *tails)), common, counts)
        globals_by_request = [bank.materialize(rid, count) for rid, count in enumerate(counts)]
    cache.read_shared_global = lambda source: bank if shared_global else None
    cache.read_global = lambda source, request, global_entries: globals_by_request[request][:global_entries]
    cu = torch.tensor((0, cp.chunk_lengths_per_req[0], cp.chunk_length), device=device, dtype=torch.int32)
    context = prefill.build_step_context(cp.global_positions, cp.req_id_per_token.long(), cu,
                                         valid_mask=cp.local_is_real)
    selected = None
    if ratio:
        selected = torch.tensor([[0, 63, 64, -1, 64, 100000]] * cp.chunk_length,
                                device=device, dtype=torch.int32)
    options = dict(ops=NS(decode_kv=_decode_tagged), cache=cache, layer_id=2,
                   geometry=geometry, context=context, selected_blocks=selected,
                   window=128, swa_starts={}, cp=cp)
    return state, cp, context, bank, prepared, options


def _references(result, *, columns=3, rows=slice(None), width=None):
    kv, indices, lengths = result
    flat = indices[rows, 0, :width].long()
    values = kv[:, 0, :columns][flat.clamp_min(0)]
    return values * (flat >= 0).unsqueeze(-1), lengths[rows]


class CP8SegmentedSwaContracts(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(torch.no_grad())
        self.stack.enter_context(patch.dict(os.environ, CP_ENV))
        self.stack.enter_context(patch("rtp_llm.models_py.modules.dsv41.ppu.layers.is_ppu", return_value=True))
        self.stack.enter_context(patch.object(_quant, "is_ppu", return_value=True))
        self.stack.enter_context(patch("torch.cuda.is_current_stream_capturing", return_value=False))
        self.stack.enter_context(patch(
            "rtp_llm.models_py.modules.dsv41.ppu.attention_layer.optimized_diagnostics_enabled", return_value=False))

    def test_host_geometry_accounts_for_all_eight_ranks_without_decoding_holes(self):
        old, new = [], []
        for rank in range(8):
            spans, legacy = prefill._cp8_swa_spans((4096,) * 8, (126976,) * 8, rank, (512,) * 8, 128)
            decoded = sum(stop - first for request in spans for first, stop in request)
            old.append(legacy)
            new.append(decoded)
            self.assertEqual(len(spans[0]), 1 if rank == 7 else 2)
            for request in spans:
                self.assertTrue(all(stop > first for first, stop in request))
                if len(request) == 2:
                    self.assertLess(request[0][1], request[1][0])
        self.assertEqual(old, [33784, 29688, 25592, 21496, 17400, 13304, 9208, 5112])
        self.assertEqual(new, [6128] * 7 + [5112])
        self.assertEqual(sum(old) - sum(new), 107576)
        for lengths, prefixes, rank, chunks in (((17, 32), (0, 0), 0, (4, 4)),
                                                ((32,), (-1,), 0, (4,)),
                                                ((32,), (0,), 8, (4,)),
                                                ((32,), (0,), 0, (6,))):
            self.assertIsNone(prefill._cp8_swa_spans(lengths, prefixes, rank, chunks, 128))

    def test_cpu_workspace_matches_every_referenced_bf16_row_and_stable_order(self):
        for rank in range(8):
            for ratio, shared_global in ((0, False), (1, False), (2, True)):
                with self.subTest(rank=rank, ratio=ratio, shared_global=shared_global):
                    state, cp, context, _, prepared, options = _fixture(rank, ratio=ratio,
                                                                         shared_global=shared_global)
                    plan = prefill.prepare_cp8_segmented_swa(state, cp, context, window=128, starts={})
                    self.assertIsNotNone(plan)
                    self.assertIs(prefill.prepare_cp8_segmented_swa(state, cp, context,
                                                                   window=128, starts={}), plan)
                    old_state, new_state = {}, {}
                    old = prefill.build_cp_prefill_workspace(**options, shared=old_state)
                    actual = prefill.build_cp_prefill_workspace(**options, shared=new_state, swa_plan=plan)
                    old_values, old_lengths = _references(old)
                    values, lengths = _references(actual)
                    self.assertTrue(torch.equal(lengths, old_lengths))
                    self.assertTrue(torch.equal(values, old_values))
                    self.assertLessEqual(actual[0].shape[0], old[0].shape[0])
                    workspace = new_state["prefill_kv_workspace"][2]
                    self.assertEqual(workspace.row_meta.shape, (cp.chunk_length, 10))
                    self.assertEqual(old_state["prefill_kv_workspace"][2].row_meta.shape[1],
                                     6 if shared_global and ratio else 4)
                    self.assertEqual(prepared, [2, 2] if ratio else [])
                    stats = state.cache._cp_segmented_swa_stats
                    self.assertEqual(stats["plans"], 1)
                    self.assertEqual(stats["legacy_rows"], plan.legacy_rows)
                    self.assertEqual(stats["decoded_rows"], plan.decoded_rows)
                    self.assertEqual(stats["dequant_calls"], sum(map(len, plan.spans)))
                    self.assertEqual(stats["codec_fallbacks"], stats["dequant_calls"])

    def test_workspace_reuses_the_plan_and_global_owner_but_refreshes_each_layers_swa(self):
        state, cp, context, bank, prepared, options = _fixture()
        plan = prefill.prepare_cp8_segmented_swa(state, cp, context, window=128, starts={})
        shared = {}
        first = prefill.build_cp_prefill_workspace(**options, shared=shared, swa_plan=plan)
        first_refs = _references(first)[0].clone()
        workspace = shared["prefill_kv_workspace"][2]
        context2 = prefill.build_step_context(cp.global_positions, cp.req_id_per_token.long(),
                                              context.request_offsets, valid_mask=cp.local_is_real)
        plan2 = prefill.prepare_cp8_segmented_swa(state, cp, context2, window=128, starts={})
        self.assertIs(plan2, plan)
        options.update(layer_id=3, context=context2)
        second = prefill.build_cp_prefill_workspace(**options, shared=shared, swa_plan=plan2)
        self.assertIs(shared["prefill_kv_workspace"][2], workspace)
        self.assertIs(workspace._shared_global, bank)
        self.assertIs(first[1], second[1])
        self.assertFalse(torch.equal(first_refs, _references(second)[0]))
        self.assertEqual(prepared, [2])
        self.assertEqual(state.cache._cp_shared_global_stats["dequant_launches"], 1)
        self.assertEqual(state.cache._cp_segmented_swa_stats["workspace_hits"], 1)
        self.assertEqual(state.cache._cp_segmented_swa_stats["refreshes"], 2)
        self.assertEqual(state.cache._cp_segmented_swa_stats["decoded_rows"], 2 * plan.decoded_rows)

    def test_prefix_clipping_ragged_lengths_and_overlapping_pieces_preserve_swa_order(self):
        for rank, lengths, prefixes in ((0, (1024, 1040), (0, 17)), (7, (1024, 1040), (0, 17)),
                                        (0, (16, 32), (0, 1))):
            with self.subTest(rank=rank, lengths=lengths, prefixes=prefixes):
                state, cp, context, _, _, options = _fixture(rank, ratio=0, lengths=lengths,
                                                             prefixes=prefixes)
                plan = prefill.prepare_cp8_segmented_swa(state, cp, context, window=128, starts={})
                self.assertIsNotNone(plan)
                actual = prefill.build_cp_prefill_workspace(**options, shared={}, swa_plan=plan)
                old = prefill.build_cp_prefill_workspace(**options, shared={})
                self.assertTrue(torch.equal(_references(actual)[0], _references(old)[0]))
                self.assertTrue(torch.equal(actual[2], old[2]))
                self.assertTrue(all(first >= 0 for request in plan.spans for first, _ in request))

    def test_source_changes_rebuild_only_the_workspace_and_forward_exit_revokes_the_plan(self):
        state, cp, context, bank, prepared, options = _fixture()
        plan = prefill.prepare_cp8_segmented_swa(state, cp, context, window=128, starts={})
        shared = {}
        first = prefill.build_cp_prefill_workspace(**options, shared=shared, swa_plan=plan)
        old_workspace = shared["prefill_kv_workspace"][2]
        old_values = first[0].clone()
        next_bank = SharedGlobalBank.create(bank.packed + 1, bank.prefix_rows, bank.counts)
        state.cache.read_shared_global = lambda source: next_bank
        state.cache._cp_shared_global_epoch = 999  # Expected KV writes do not change query-row birth.
        options["geometry"].kv_source_for = lambda layer: 8
        prefill.build_cp_prefill_workspace(**options, shared=shared, swa_plan=plan)
        self.assertIsNot(shared["prefill_kv_workspace"][2], old_workspace)
        self.assertTrue(plan.current(context))
        self.assertEqual(prepared, [2, 8])
        self.assertEqual(state.cache._cp_segmented_swa_stats["plans"], 1)
        self.assertEqual(state.cache._cp_shared_global_stats["dequant_launches"], 2)
        state.runtime.current = None
        self.assertFalse(plan.current(context))
        self.assertTrue(torch.equal(first[0], old_values))

    def test_plan_uses_native_request_owners_and_declines_shape_only_or_stale_births(self):
        for change in ("position_values", "position_owner", "request_values", "mask_values", "clone_plan"):
            with self.subTest(change=change):
                state, cp, context, _, _, options = _fixture()
                plan = prefill.prepare_cp8_segmented_swa(state, cp, context, window=128, starts={})
                if change == "position_values":
                    cp.global_positions.add_(0)
                elif change == "position_owner":
                    cp.global_positions = cp.global_positions.clone()
                elif change == "request_values":
                    cp.req_id_per_token.add_(0)
                elif change == "mask_values":
                    cp.local_is_real.logical_not_()
                else:
                    plan = replace(plan)
                self.assertFalse(plan.current(context))
                with self.assertRaisesRegex(RuntimeError, "segmented SWA"):
                    prefill.build_cp_prefill_workspace(**options, shared={}, swa_plan=plan)
        state, cp, context, _, _, options = _fixture()
        unknown = replace(context, positions=context.positions.clone())
        self.assertIsNone(prefill.prepare_cp8_segmented_swa(state, cp, unknown, window=128, starts={}))
        plan = prefill.prepare_cp8_segmented_swa(state, cp, context, window=128, starts={})
        expected = prefill.build_cp_prefill_workspace(**options, shared={})
        # The old int64 cast remains a separate consumer ABI. Segmented row
        # metadata reads the native birth owner instead of trusting this copy.
        context.request_ids.fill_(1)
        actual = prefill.build_cp_prefill_workspace(**options, shared={}, swa_plan=plan)
        self.assertTrue(plan.current(context))
        self.assertTrue(torch.equal(actual[2], torch.full_like(actual[2], 132)))
        self.assertTrue(torch.equal(_references(actual)[0], _references(expected)[0]))

    def test_q8_padding_compact_local_and_wrong_window_keep_the_old_path(self):
        state, cp, context, _, _, options = _fixture()
        for extra in (dict(q8=True), dict(window=64), dict(starts={0: 256})):
            kwargs = dict(window=128, starts={})
            kwargs.update(extra)
            self.assertIsNone(prefill.prepare_cp8_segmented_swa(state, cp, context, **kwargs))
        state.compacted = True
        self.assertIsNone(prefill.prepare_cp8_segmented_swa(state, cp, context, window=128, starts={}))
        state, cp, context, _, _, _ = _fixture(lengths=(1025, 1024))
        self.assertIsNone(prefill.prepare_cp8_segmented_swa(state, cp, context, window=128, starts={}))
        state, cp, context, _, _, _ = _fixture()
        state.cache.sharded = False
        self.assertIsNone(prefill.prepare_cp8_segmented_swa(state, cp, context, window=128, starts={}))
        state, cp, context, _, _, _ = _fixture()
        state.is_graph = True
        self.assertIsNone(prefill.prepare_cp8_segmented_swa(state, cp, context, window=128, starts={}))
        state, cp, context, _, _, options = _fixture()
        plan = prefill.prepare_cp8_segmented_swa(state, cp, context, window=128, starts={})
        first = prefill.build_cp_prefill_workspace(**options, shared={})
        second = prefill.build_cp_prefill_workspace(**options, shared={})
        self.assertTrue(torch.equal(first[0], second[0]))
        self.assertTrue(torch.equal(first[1], second[1]))
        self.assertTrue(torch.equal(first[2], second[2]))
        self.assertIsNotNone(plan)


@unittest.skipUnless(os.environ.get("RTP_V41_TEST_LOCAL_WORKSPACE_DEVICE") == "1", "explicit PPU lease required")
class CP8SegmentedSwaDevice(unittest.TestCase):
    @torch.no_grad()
    def test_actual_dequant_and_indices_preserve_all_referenced_bf16_values(self):
        self.assertTrue(_quant.is_ppu())
        with patch.dict(os.environ, CP_ENV):
            for rank in (0, 3, 7):
                for ratio, shared_global in ((0, False), (1, False), (1, True), (2, True)):
                    with self.subTest(rank=rank, ratio=ratio, shared_global=shared_global):
                        state, cp, context, _, _, options = _fixture(rank, ratio=ratio,
                                                                     shared_global=shared_global, device="cuda:0")
                        cache = state.cache
                        del cache.try_decode_swa_range
                        cache._fresh_swa = {}
                        for request, prefix in enumerate(state.query_prefixes):
                            positions = torch.arange(prefix - 127, prefix + state.lengths[request], device="cuda:0")
                            column = torch.arange(512, device="cuda:0")[None]
                            values = (((positions[:, None] >> (column % 16)) & 1) * 2 - 1).to(torch.bfloat16)
                            values[:, 32] = 1 if request else -1
                            packed = kv_codec.encode_kv(values, pool_tag="swa_kv")
                            cache._fresh_swa[2, request] = _StagedSwaSegments(prefix, packed[:127], packed[127:])
                        options["ops"] = NS(decode_kv=kv_codec.decode_kv)
                        if ratio:
                            counts = tuple(total // ratio for total in cache.lengths)
                            common = 64
                            rows = sum(counts) - (len(counts) - 1) * common
                            index = torch.arange(rows, device="cuda:0")[:, None]
                            column = torch.arange(512, device="cuda:0")[None]
                            values = (((index >> (column % 16)) & 1) * 2 - 1).to(torch.bfloat16)
                            packed = kv_codec.encode_kv(values, pool_tag="global_kv_1")
                            bank = SharedGlobalBank.create(packed, common, counts)
                            cache.read_shared_global = lambda source: bank if shared_global else None
                            cache.read_global = lambda source, request, global_entries: bank.materialize(request, global_entries)
                        plan = prefill.prepare_cp8_segmented_swa(state, cp, context, window=128, starts={})
                        self.assertIsNotNone(plan)
                        old = prefill.build_cp_prefill_workspace(**options, shared={})
                        actual = prefill.build_cp_prefill_workspace(**options, shared={}, swa_plan=plan)
                        # Bit patterns distinguish every logical row. Compare
                        # all512 dimensions for every query in bounded slices.
                        self.assertTrue(torch.equal(actual[2], old[2]))
                        width = 128 + (options["selected_blocks"].shape[1] if ratio else 0)
                        for first in range(0, cp.chunk_length, 16):
                            rows = slice(first, first + 16)
                            a = _references(actual, columns=512, rows=rows, width=width)[0]
                            b = _references(old, columns=512, rows=rows, width=width)[0]
                            self.assertTrue(torch.equal(a.view(torch.int16), b.view(torch.int16)))
                        self.assertTrue(bool((actual[1][:, :, width:] == -1).all()))
                        stats = cache._cp_segmented_swa_stats
                        self.assertEqual(stats["fused_decode_hits"], stats["dequant_calls"])
                        self.assertEqual(stats["codec_fallbacks"], 0)
            torch.cuda.synchronize()


if __name__ == "__main__":
    unittest.main()
