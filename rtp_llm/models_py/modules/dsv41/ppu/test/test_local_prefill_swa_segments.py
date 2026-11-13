"""Creation-stamped local SWA snapshots, old native writes and exact decoding."""

from contextlib import ExitStack
from copy import copy
import os
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch
from torch import nn

from rtp_llm.models_py.modules.dsv41.ppu import attention_layer, prefill
from rtp_llm.models_py.modules.dsv41.ppu._quant import is_ppu
from rtp_llm.models_py.modules.dsv41.ppu.ced import _bounded_query_layout
from rtp_llm.models_py.modules.dsv41.ppu.kv_codec import decode_kv, encode_kv
from rtp_llm.models_py.modules.dsv41.ppu.native_cache_adapter import (
    LOCAL_PREFILL_SWA_SEGMENTS_ENV,
    SharedNativeCache,
    _StagedSwaSegments,
)
from rtp_llm.models_py.modules.dsv41.ppu.prefill_runtime import (
    LOCAL_PREFILL_COMPACT_ROWS_ENV, LOCAL_PREFILL_FULL_ROWS_ENV, LOCAL_PREFILL_ROWS_ENV,
)
from rtp_llm.models_py.modules.dsv41.ppu.replay_policy import PrefillPolicy, SharedPrefillPolicy
from rtp_llm.models_py.modules.dsv41.ppu.test.test_local_prefill_fused import _NativePools, assert_bytes
from rtp_llm.models_py.modules.dsv41.ppu.test.test_local_prefill_operator_flags import _trace_events
from rtp_llm.models_py.modules.dsv41.ppu.test.test_local_prefill_rows import state_for
from rtp_llm.models_py.modules.dsv41.prefill.ced import CEDPlan, _bounded_replay_selected


ENV = LOCAL_PREFILL_SWA_SEGMENTS_ENV
FLAGS = {ENV: "1", LOCAL_PREFILL_ROWS_ENV: "1", LOCAL_PREFILL_FULL_ROWS_ENV: "1",
         LOCAL_PREFILL_COMPACT_ROWS_ENV: "1"}


def values(positions, shift=0):
    columns = torch.arange(512, device=positions.device)[None]
    return ((positions[:, None] % 29 - 14) * 0.125 + (columns % 17) * 0.015625
            + shift * 0.03125).to(torch.bfloat16)


def fixture(device="cpu", prefixes=(129,), lengths=(4096,), *, layers=2):
    geometry = NS(num_layers=layers, kv_source_layer_ids=(), index_source_layer_ids=(),
                  head_dim=512, window=128, ratio=lambda layer: 0, kv_source_for=lambda layer: None)
    totals = tuple(p + n for p, n in zip(prefixes, lengths))
    native = _NativePools(geometry, totals, device)
    cache = SharedNativeCache(PrefillPolicy().freeze({}), geometry, native, native.inputs,
                              totals, device=device, prefill=True)
    state = state_for(device, prefixes, lengths)
    state.cache = cache
    state.runtime.policy = SharedPrefillPolicy(mode="compact")
    state.runtime.model._is_speculative = False
    for layer in range(layers):
        for rid, prefix in enumerate(prefixes):
            old = torch.arange(max(0, prefix - 127), prefix, device=device)
            cache._copy_rows((layer, "swa_kv"), rid, old,
                             encode_kv(values(old, layer + rid), pool_tag="swa_kv"))
    return state


def compact_plan(state, *, activate=True):
    original = state.identity_context()  # Genuine full-row creation stamp.
    starts = tuple(max(n - 140, 0) for n in state.lengths)
    selected = _bounded_replay_selected(state.lengths, starts)
    query, exchange, groups = _bounded_query_layout(original, selected)
    # Public layout consumes host selection and places its outputs on the
    # original context device. Hidden state and CED cu must use that device too.
    device = original.global_positions.device
    assert all(tensor.device == device for tensor in (
        query.gather_restore_positions, exchange.send_indices, exchange.receive_positions,
    )), "bounded fixture selection/send/receive must share the actual input device"
    if len(starts) == 1:
        query.swa_replay_start = starts[0]
    else:
        query.swa_replay_starts_host = starts
    cu = [0]
    for size in query.chunk_lengths_per_req:
        cu.append(cu[-1] + size)
    state.original_context, state.plan_reason = original, "compact140"
    state.plan = CEDPlan(original, query, torch.tensor(cu, device=device, dtype=torch.int32),
                         exchange, groups, ())
    state.v4._propagate_cp_ctx = lambda context: None
    if activate:
        state._activate_compact(torch.empty(query.chunk_length, 1, device=device))
    return query


def arguments(state, *, compact=False, layer=0):
    cp = state.context if compact else None
    pos, req, real = state.full_rows(cp, selected_only=compact)
    positions, requests = pos[real], req[real]
    packed = encode_kv(values(positions, layer + 3), pool_tag="swa_kv")
    bounded = tuple(p + max(n - 140, 0) for p, n in zip(state.query_prefixes, state.lengths)) if compact else None
    options = dict(retain_rows=True, retain_prefix_tail=not compact, bounded_starts=bounded)
    ticket = attention_layer._local_swa_stage_ranges(
        state, cp, layer, pos, req, real, positions, requests, packed,
        packed_plan=None, **options,
    )
    return positions, requests, packed, options, ticket


def stage(state, args, *, enabled=True, layer=0):
    pos, req, packed, options, ticket = args
    with patch.dict(os.environ, {ENV: str(int(enabled))}):
        state.cache.stage_swa(layer, pos, req, packed, state.query_prefixes,
                              **options, **({"local_ranges": ticket} if ticket is not None else {}))


class _SwaCases:
    @torch.no_grad()
    def test_full_compact_ragged_zero_and_odd_padding_match_old_bytes(self):
        for prefixes, lengths, compact in (
            ((129,), (4096,), False), ((129,), (4097,), False),
            ((17, 99, 64), (4096, 5, 0), True), ((17, 9), (0, 0), False),
        ):
            arms = []
            for enabled in (False, True):
                state = fixture(self.device, prefixes, lengths)
                if compact:
                    compact_plan(state)
                args = arguments(state, compact=compact)
                self.assertIsNotNone(args[-1])
                if compact:
                    self.assertEqual(args[-1].ranges, ((0, 3973, 140), (140, 99, 5), (145, 64, 0)))
                stage(state, args, enabled=enabled)
                if enabled:
                    self.assertTrue(all(isinstance(item, _StagedSwaSegments) for item in state.cache._fresh_swa.values()))
                samples = []
                for rid, (prefix, length) in enumerate(zip(prefixes, lengths)):
                    first = prefix + max(length - 140, 0) if compact else max(0, prefix - 127)
                    positions = torch.arange(first, prefix + length, device=self.device)
                    samples.append(state.cache.read_swa(0, rid, positions))
                pools = tuple(pool.tensor.clone() for pool in state.cache.pools.values())
                arms.append((state.cache.writes, pools, samples, state, args))
            self.assertEqual(arms[0][0], arms[1][0])
            for baseline, candidate in zip(arms[0][1], arms[1][1]):
                assert_bytes(self, candidate, baseline)
            for baseline, candidate in zip(arms[0][2], arms[1][2]):
                assert_bytes(self, candidate, baseline)
            # Input mutation and native-ring overwrite cannot mutate owned snapshots.
            state, args = arms[1][3:]
            before = [item.fresh.clone() for item in state.cache._fresh_swa.values()]
            args[2].zero_()
            state.cache.pools[0, "swa_kv"].tensor.zero_()
            for item, frozen in zip(state.cache._fresh_swa.values(), before):
                assert_bytes(self, item.fresh, frozen)
                self.assertFalse(torch._C._is_alias_of(item.fresh, args[2]))
            state.cache.release_swa(0)
            self.assertFalse(state.cache._fresh_swa)

    @torch.no_grad()
    def test_creation_versions_never_reauthenticate_bad_ancestors_or_cached_outputs(self):
        for compact in (False, True):
            for target in ("p", "r", "ancestor", "cu", "stamp", "proof", "late_flag"):
                state = fixture(self.device, (17,), (4096,))
                if target == "late_flag":
                    with patch.dict(os.environ, {ENV: "0"}):
                        state = fixture(self.device, (17,), (4096,))
                        state.identity_context()
                if compact:
                    compact_plan(state)
                pos, req, *_ = state.full_rows(state.context if compact else None, selected_only=compact)
                if target in ("p", "r"):
                    tensor = pos if target == "p" else req
                    tensor.add_(1).sub_(1)
                elif target == "ancestor":
                    state._local_swa_full_stamp[1][0].add_(1).sub_(1)
                elif target == "cu":
                    state.attn.cu_seqlens.add_(1).sub_(1)
                elif target == "stamp":
                    state._local_swa_full_stamp = None
                elif target == "proof":
                    if compact:
                        state._local_compact_row_proof = tuple(list(state._local_compact_row_proof))
                    else:
                        state._local_swa_full_stamp = None
                self.assertIsNone(arguments(state, compact=compact)[-1], (compact, target))
        # A bad full ancestor before selected-only construction cannot mint a new stamp.
        state = fixture(self.device, (17,), (4096,))
        compact_plan(state)
        state._local_swa_full_stamp[1][1].add_(1).sub_(1)
        state.full_rows(state.context, selected_only=True)
        self.assertIsNone(state._local_swa_selected_stamp)
        # A selected cache hit cannot mint a missing/late segment stamp either.
        for missing in (True, False):
            state = fixture(self.device, (17,), (4096,))
            compact_plan(state)
            with patch.dict(os.environ, {ENV: "1" if missing else "0"}):
                state.full_rows(state.context, selected_only=True)
            state._local_swa_selected_stamp = None
            self.assertIsNone(arguments(state, compact=True)[-1])
            self.assertIsNone(state._local_swa_selected_stamp)
        # Bind original host metadata and cu at __init__, before the first rows.
        for target in ("prefixes", "lengths", "cu"):
            state = fixture(self.device, (17,), (4096,))
            if target == "cu":
                state.attn.cu_seqlens.add_(0)
            elif target == "prefixes":
                state.query_prefixes = (18,)
            else:
                state.lengths = (4095,)
            self.assertIsNone(arguments(state)[-1], target)
            self.assertIsNone(state._local_swa_full_stamp)
        for target in ("plan", "context", "layout"):
            state = fixture(self.device, (17,), (4096,))
            compact_plan(state)
            self.assertIsNotNone(arguments(state, compact=True)[-1])
            if target == "plan":
                state.plan = copy(state.plan)
            elif target == "context":
                state.context = copy(state.context)
            else:
                state.context.global_positions.add_(0)
            self.assertIsNone(arguments(state, compact=True)[-1], target)

    @torch.no_grad()
    def test_physical_request_aliases_keep_per_rid_tail_read_then_write_order(self):
        arms = []
        for enabled in (False, True):
            state = fixture(self.device, (128, 256), (128, 128))
            pool = state.cache.pools[0, "swa_kv"]
            # Same physical block across both requests; native span/entries remain real.
            pool.table[1].copy_(pool.table[0])
            pool.tensor.fill_(31)
            args = arguments(state)
            stage(state, args, enabled=enabled)
            saved = [state.cache.read_swa(0, rid, torch.arange(p - 127, p + n, device=self.device))
                     for rid, (p, n) in enumerate(zip(state.query_prefixes, state.lengths))]
            arms.append((saved, pool.tensor.clone(), state.cache.writes))
        for baseline, candidate in zip(arms[0][0], arms[1][0]):
            assert_bytes(self, candidate, baseline)
        assert_bytes(self, arms[0][1], arms[1][1])
        self.assertEqual(arms[0][2], arms[1][2])

    @torch.no_grad()
    def test_missing_errors_metadata_and_per_call_fallbacks_are_unchanged(self):
        for case in ("tail", "live", "outside", "zero_request"):
            errors = []
            for enabled in (False, True):
                state = fixture(self.device, (129,), (0,) if case == "zero_request" else (4096,))
                args = arguments(state)
                pool = state.cache.pools[0, "swa_kv"]
                if case in ("tail", "zero_request"):
                    pool.table[0, 0] = -1
                elif case == "live":
                    pool.table[0, -2] = -1
                else:
                    pool.table[0, 0] = pool.tensor.shape[0]
                # Rebuild ticket after the table mutation to retain native errors.
                args = arguments(state)
                try:
                    stage(state, args, enabled=enabled)
                except ValueError as error:
                    errors.append(str(error))
            self.assertEqual(len(errors), 2, case)
            self.assertEqual(errors[0], errors[1])
        state = fixture(self.device, (17,), (4096,))
        args = arguments(state)
        pos, req, packed, options, ticket = args
        self.assertIsNotNone(ticket)
        for altered in (dict(options, retain_rows=False),):
            self.assertIsNone(state.cache._checked_local_swa_ranges(
                0, pos, req, packed, state.query_prefixes, **altered, ticket=ticket,
            ))
        alias_state = fixture(self.device, (17,), (141,))
        apos, areq, apacked, aoptions, _ = arguments(alias_state)
        pool = alias_state.cache.pools[0, "swa_kv"]
        alias = pool.tensor.flatten()[:apacked.numel()].view_as(apacked)
        alias_ticket = attention_layer._local_swa_stage_ranges(
            alias_state, None, 0, alias_state.full_rows(None)[0], alias_state.full_rows(None)[1], slice(None),
            apos, areq, alias, packed_plan=None, **aoptions,
        )
        self.assertIsNotNone(alias_ticket)
        self.assertIsNone(alias_state.cache._checked_local_swa_ranges(
            0, apos, areq, alias, alias_state.query_prefixes, **aoptions, ticket=alias_ticket,
        ))
        # Mutation after ticket creation cannot pass the adapter's second check.
        pos.add_(1).sub_(1)
        stage(state, args)
        self.assertNotIsInstance(state.cache._fresh_swa[0, 0], _StagedSwaSegments)

    @torch.no_grad()
    def test_disabled_packed_plan_and_other_phases_decline_before_launch(self):
        for change in ("disabled", "packed", "retain", "decode", "idle", "graph", "verify", "speculative", "cp8"):
            state = fixture(self.device, (17,), (141,))
            pos, req, real = state.full_rows(None)
            packed = encode_kv(values(pos), pool_tag="swa_kv")
            cp, packed_plan, retain = None, None, True
            if change == "packed":
                packed_plan = object()
            elif change == "retain":
                retain = False
            elif change == "decode":
                state.prefill = False
            elif change == "idle":
                state.idle = True
            elif change == "graph":
                state.is_graph = True
            elif change == "verify":
                state.verify = True
            elif change == "speculative":
                state.runtime.model._is_speculative = True
            elif change == "cp8":
                cp = NS(cp_size=8, kv_cache_sharded=True)
            with patch.dict(os.environ, {ENV: "0" if change == "disabled" else "1"}):
                ticket = attention_layer._local_swa_stage_ranges(
                    state, cp, 0, pos, req, real, pos[:], req[:], packed,
                    packed_plan=packed_plan, retain_rows=retain, retain_prefix_tail=True, bounded_starts=None,
                )
            self.assertIsNone(ticket, change)
        # A CED plan does not mean packed attention; L20's full producer is eligible.
        state = fixture(self.device, (17,), (141,))
        state.plan = object()
        self.assertIsNotNone(arguments(state)[-1])

    @torch.no_grad()
    def test_physical_block_zero_and_read_misses_keep_exact_bytes(self):
        arms = []
        for enabled in (False, True):
            state = fixture(self.device, (0,), (128,))
            state.cache.pools[0, "swa_kv"].table[0, 0] = 0
            args = arguments(state)
            stage(state, args, enabled=enabled)
            samples = state.cache.read_swa(0, 0, torch.tensor([-1, 0, 127, 128], device=self.device))
            self.assertTrue(bool((samples[[0, 3]] == 0).all()))
            arms.append((samples, state.cache.pools[0, "swa_kv"].tensor.clone(), state.cache.writes))
        assert_bytes(self, arms[0][0], arms[1][0])
        assert_bytes(self, arms[0][1], arms[1][1])
        self.assertEqual(arms[0][2], arms[1][2])

    @torch.no_grad()
    def test_layer_forward_ticket_bindings_and_metadata_versions_do_not_leak(self):
        state = fixture(self.device, (17,), (141,))
        args = arguments(state)
        pos, req, packed, options, ticket = args
        state.cache.pools[0, "swa_kv"].table.add_(0)
        self.assertIsNone(state.cache._checked_local_swa_ranges(
            0, pos, req, packed, state.query_prefixes, **options, ticket=ticket,
        ))
        args = arguments(state)
        stage(state, args)
        old = state.cache._fresh_swa[0, 0]
        frozen = old.fresh.clone()
        next_args = arguments(state, layer=1)
        stage(state, next_args, layer=1)
        self.assertIsInstance(state.cache._fresh_swa[1, 0], _StagedSwaSegments)
        assert_bytes(self, old.fresh, frozen)
        # Neither a different layer nor a different forward may consume this ticket.
        pos, req, packed, options, ticket = next_args
        self.assertIsNone(state.cache._checked_local_swa_ranges(
            0, pos, req, packed, state.query_prefixes, **options, ticket=ticket,
        ))
        other = fixture(self.device, (17,), (141,))
        self.assertIsNone(other.cache._checked_local_swa_ranges(
            1, pos, req, packed, other.query_prefixes, **options, ticket=ticket,
        ))
        state.cache.release_swa(1)
        self.assertNotIn((1, 0), state.cache._fresh_swa)
        self.assertIn((0, 0), state.cache._fresh_swa)


class LocalSwaSegmentsCPU(_SwaCases, unittest.TestCase):
    def setUp(self):
        self.device = "cpu"
        self.environment = patch.dict(os.environ, FLAGS)
        self.environment.start()
        self.addCleanup(self.environment.stop)


@unittest.skipUnless(os.environ.get("RTP_V41_TEST_LOCAL_WORKSPACE_DEVICE") == "1", "explicit PPU device test")
class LocalSwaSegmentsDevice(_SwaCases, unittest.TestCase):
    def setUp(self):
        self.assertTrue(is_ppu(), "segments qualification requires a ZW PPU")
        self.device = "cuda:0"
        self.environment = patch.dict(os.environ, {**FLAGS,
            "RTP_V41_PPU_LOCAL_PREFILL_WORKSPACE": "1",
            "RTP_V41_PPU_LOCAL_PREFILL_GROUPED_QO_ROPE": "0",
            "RTP_V41_PPU_FUSED_BF16_ROPE": "1", attention_layer.FUSION_ENV: "0", attention_layer.STREAMS_ENV: "0",
        })
        self.environment.start()
        self.addCleanup(self.environment.stop)

    @torch.no_grad()
    def test_actual_segment_decoder_ranges_padding_edges_and_fallbacks_are_exact(self):
        state = fixture(self.device, (129,), (4096,))
        edge = state.cache._copy_rows((0, "swa_kv"), 0, torch.tensor([128], device=self.device))
        edge[0, :8] = torch.tensor([0, 128, 56, 184, 126, 254, 127, 255], device=self.device, dtype=torch.uint8)
        edge[0, 512:] = torch.tensor([0, 1, 126, 127, 128, 254, 255, 127] * 2,
                                     device=self.device, dtype=torch.uint8)
        state.cache._copy_rows((0, "swa_kv"), 0, torch.tensor([128], device=self.device), edge)
        args = arguments(state)
        pos, req, packed, options, _ = args
        packed[0].copy_(edge[0])
        full_pos, full_req, real = state.full_rows(None)
        ticket = attention_layer._local_swa_stage_ranges(
            state, None, 0, full_pos, full_req, real, pos, req, packed, packed_plan=None, **options,
        )
        args = pos, req, packed, options, ticket
        stage(state, args)
        saved = state.cache._fresh_swa[0, 0]
        for first, length in ((2, 127), (65, 140), (128, 2), (129, 4096), (3958, 267), (4224, 1), (4225, 0)):
            out_base = torch.full((length, 519), 73, device=self.device, dtype=torch.bfloat16)
            out = out_base[:, :512]
            self.assertTrue(state.cache.try_decode_swa_range(0, 0, first, out))
            packed = state.cache.read_swa(0, 0, torch.arange(first, first + length, device=self.device))
            expected = torch.empty_like(out)
            decode_kv(packed, pool_tag="swa_kv", out=expected)
            assert_bytes(self, out, expected)
            self.assertTrue(bool((out_base[:, 512:] == 73).all()))
        out = torch.full((1, 512), 19, device=self.device, dtype=torch.bfloat16)
        for first in (1, 4225):
            self.assertFalse(state.cache.try_decode_swa_range(0, 0, first, out))
            self.assertTrue(bool((out == 19).all()))
        for unsupported in (out.float(), out[:, :-1], torch.empty(1, 1024, device=self.device, dtype=torch.bfloat16)[:, ::2]):
            unsupported.fill_(19)
            self.assertFalse(state.cache.try_decode_swa_range(0, 0, 129, unsupported))
            self.assertTrue(bool((unsupported == 19).all()))
        self.assertNotEqual(saved.tail.untyped_storage().data_ptr(), state.cache.pools[0, "swa_kv"].tensor.untyped_storage().data_ptr())

    @torch.inference_mode()
    def test_actual_l20_body_full_stage_then_267_row_decoder_profile(self):
        outputs, banks, receipts = [], [], []
        for enabled in (False, True):
            state = fixture(self.device, (129,), (4096,), layers=21)
            cp = compact_plan(state, activate=False)
            rows = 4096
            attn = attention_layer.SharedPPUAttention.__new__(attention_layer.SharedPPUAttention)
            nn.Module.__init__(attn)
            attn.geometry = NS(hidden_size=8, n_heads=1, head_dim=512, window=128,
                               o_groups=1, o_lora_rank=8, ratio=lambda layer: 0,
                               kv_source_for=lambda layer: None, sm_scale=lambda: 512 ** -0.5)
            attn.layer_id, attn.compress_ratio = 20, 0
            attn.is_kv_source = attn.is_index_source = False
            attn._cp_ctx = None
            attn.ops = NS(encode_kv=encode_kv, decode_kv=decode_kv)
            q = values(torch.arange(rows, device=self.device), 4).unsqueeze(1)
            kv = values(torch.arange(rows, device=self.device), 5)
            attn.project_query_kv = lambda x, **kwargs: (x, q.clone(), kv.clone(), None)
            attn._project_indexer = lambda x, *args, **kwargs: (x[:, None], x[:, :1])
            attn.wo_a = nn.Linear(512, 8, bias=False, device=self.device, dtype=torch.bfloat16)
            attn.wo_a.weight.data.fill_(1 / 512)
            attn.wo_b = nn.Identity()
            attn.attn_sink = torch.zeros(1, device=self.device)
            freqs = torch.ones(129 + rows, 32, device=self.device, dtype=torch.complex64)
            attn.runtime = NS(current=state, policy=state.runtime.policy,
                              freqs_for=lambda layer: freqs, qkv_streams=None)
            state.v4._propagate_cp_ctx = lambda context: setattr(attn, "_cp_ctx", context)

            def callback():
                self.assertEqual(state.cache.writes[-1]["layer"], 20)
                self.assertFalse(state.compacted)
                state.compact_l20_queries(state.plan.exchange.compact(torch.empty(rows, 1, device=self.device)),
                                           state.plan.exchange.compact(state.inputs.input_ids))

            with ExitStack() as stack:
                stack.enter_context(patch.dict(os.environ, {ENV: str(int(enabled))}))
                stack.enter_context(patch.object(attention_layer, "prepare_packed_attention", return_value=(None, "disabled")))
                stack.enter_context(patch.object(attention_layer, "optimized_diagnostics_enabled", return_value=False))
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                      torch.profiler.ProfilerActivity.CUDA]) as profile:
                    output = attn._body(torch.zeros(rows, 8, device=self.device, dtype=torch.bfloat16),
                                        compact_after_producer=callback)
                    torch.cuda.synchronize()
            events = _trace_events(profile, f"local-swa-segments-l20-{int(enabled)}")
            kernels = [event["name"] for event in events if event.get("cat") == "kernel"]
            self.assertEqual(sum(name == "_dequantize_v41_swa_segments_kernel" for name in kernels), int(enabled), kernels)
            self.assertEqual(any("searchsorted" in name for name in kernels), not enabled, kernels)
            self.assertTrue(any("flash_sparse_prefill" in name for name in kernels), kernels)
            workspace = state.shared["prefill_kv_workspace"][2]
            self.assertEqual(workspace.swa_ranges[0][2:4], (129 + 4096 - 140 - 127, 267))
            self.assertTrue(bool((workspace.index_plan[1] == 128).all()))
            self.assertEqual(output.shape, (140, 8))
            self.assertTrue(bool(state.valid_output_rows[-1]))
            outputs.append(output)
            banks.append(tuple(pool.tensor.clone() for pool in state.cache.pools.values()))
            receipts.append(state.cache.writes)
        assert_bytes(self, outputs[1], outputs[0])
        for control, candidate in zip(banks[0], banks[1]):
            assert_bytes(self, candidate, control)
        self.assertEqual(receipts[0], receipts[1])


if __name__ == "__main__":
    unittest.main()
