"""CP140 raw-QR transport, full producers and per-forward proof contracts."""

from contextlib import ExitStack
from datetime import timedelta
from itertools import accumulate
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F

from rtp_llm.models_py.modules.dsv41.cp import CPContext
from rtp_llm.models_py.modules.dsv41.ppu import ced as ppu_ced
from rtp_llm.models_py.modules.dsv41.ppu import local_prefill_qkv as qkv
from rtp_llm.models_py.modules.dsv41.ppu._quant import is_ppu
from rtp_llm.models_py.modules.dsv41.ppu.cp import build_zigzag_plan
from rtp_llm.models_py.modules.dsv41.ppu.linear import PPUPTPCLinear
from rtp_llm.models_py.modules.dsv41.ppu.optimization_config import DENSE_PROJECTIONS, MEGA_ATTENTION
from rtp_llm.models_py.modules.dsv41.ppu.prefill_runtime import SharedForwardState
from rtp_llm.models_py.modules.dsv41.ppu.replay_policy import SharedPrefillPolicy
from rtp_llm.models_py.modules.dsv41.ppu.test.test_local_prefill_split_l20 import _component
from rtp_llm.models_py.modules.dsv41.prefill.ced import (
    CEDPlan, _bounded_replay_selected, _host_query_layout,
)


class _Norm:
    def __init__(self, width, device):
        self.weight = torch.ones(width, dtype=torch.bfloat16, device=device)
        self.variance_epsilon = 1e-6

    def __call__(self, value, **kwargs):
        x = value.float()
        return (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.variance_epsilon)
                * self.weight.float()).to(value.dtype).contiguous()


class _Projector:
    """Real PTPC projection with a small output-channel fixture on CPU."""

    def __init__(self, attention):
        self.owner = attention
        self.key = qkv._linear_key(attention.wq_b)

    def _current(self):
        return qkv._linear_key(self.owner.wq_b) == self.key

    def project_q(self, latent, quantized_input=None):
        return (self.owner.wq_b(latent) if quantized_input is None else
                self.owner.wq_b.forward_quantized(*quantized_input))


def _context(rank=0, lengths=(256, 272), prefixes=(17, 49), device="cpu"):
    cu = torch.tensor((0, *accumulate(lengths)), dtype=torch.int64)
    ids = torch.arange(sum(lengths), dtype=torch.int64)
    peers = [build_zigzag_plan(ids, cu, cp_size=8, cp_rank=i, prefix_lengths=prefixes) for i in range(8)]
    own = peers[rank]
    restore = torch.empty(own.padded_seq_len, dtype=torch.int64)
    for peer in peers:
        restore[peer.relative_positions] = peer.cp_rank * own.chunk_length + torch.arange(own.chunk_length)
    mask = torch.cat([torch.arange(split.padded) < split.length for split in own.requests])
    info = NS(prefill_qkv_restore_indice_cpu=restore, prefill_qkv_padding_mask_cpu=mask)
    context = CPContext(
        cp_size=8, cp_rank=rank, chunk_length=own.chunk_length,
        padded_seq_len=own.padded_seq_len, seq_len_full=sum(lengths),
        relative_positions=own.relative_positions.to(device), prefix_length=prefixes[0],
        global_positions=own.local_positions.to(device), local_is_real=own.local_is_real.to(device),
        unpad_restore=own.unpad_restore.to(device), seq_len_total=max(p + n for p, n in zip(prefixes, lengths)),
        cp_info=info, req_id_per_token=own.local_request_ids.to(device=device, dtype=torch.int32),
        prefix_lengths=torch.tensor(prefixes, dtype=torch.int64, device=device),
        input_lengths_global=torch.tensor(lengths, dtype=torch.int64, device=device),
        cu_seqlens_global=cu.to(device), chunk_lengths_per_req=tuple(s.chunk for s in own.requests),
        kv_cache_sharded=True, input_lengths_global_host=lengths, prefix_lengths_host=prefixes,
    )
    return context, own.local_input_ids.to(device)


def _state(rank=0, lengths=(256, 272), prefixes=(17, 49), device="cpu", group=None):
    original, ids = _context(rank, lengths, prefixes, device)
    selected = _bounded_replay_selected(lengths, tuple(max(n - 140, 0) for n in lengths))
    compact, exchange, groups = _host_query_layout(original, selected, group, keep_candidate_rows=True)
    compact.swa_replay_starts_host = tuple(max(n - 140, 0) for n in lengths)
    plan = CEDPlan(original, compact,
                   torch.tensor((0, *accumulate(compact.chunk_lengths_per_req)), dtype=torch.int32, device=device),
                   exchange, groups, ())
    plan._ppu_cp8_query_birth = qkv.CP8QueryPlanBirth(plan)
    model = NS(_is_decode_role=False, _is_speculative=False, _prefill_cp_size=8,
               parallelism_config=NS(tp_size=8), v4=NS(layers=[]))
    runtime = NS(model=model, policy=SharedPrefillPolicy(mode="compact"))
    inputs = NS(input_ids=ids)
    attn = NS(cu_seqlens=torch.tensor((0, *accumulate(original.chunk_lengths_per_req)),
                                    dtype=torch.int32, device=device), is_cuda_graph=False)
    state = SharedForwardState(runtime, inputs, attn, NS(sharded=True),
                               prefill=True, idle=False, prefixes=prefixes, lengths=lengths)
    runtime.current = state
    state.original_context = state.context = original
    state.plan, state.plan_reason = plan, "compact140"
    state._cp_l20_projection_proof = qkv.CP8L20ProjectionProof(state, plan._ppu_cp8_query_birth)
    return state


def _linear(rows, columns, device, generator):
    weight = (torch.randn(rows, columns, generator=generator) * 0.03125).to(device=device, dtype=torch.float8_e4m3fn)
    scale = torch.full((rows, 1), 0.125, dtype=torch.float32, device=device)
    return PPUPTPCLinear(weight, scale)


def _attention(state, *, hidden_width=16, query_width=16, index_heads=2, index_width=8):
    device = state.inputs.input_ids.device
    generator = torch.Generator().manual_seed(20261117)
    attention = NS(layer_id=20, is_kv_source=True, is_index_source=True)
    attention.wq_a = _linear(1280, hidden_width, device, generator)
    attention.wkv = _linear(512, hidden_width, device, generator)
    attention.wq_b = _linear(query_width, 1280, device, generator)
    attention.index_wq = _linear(index_heads * index_width, 1280, device, generator)
    attention.q_norm, attention.kv_norm = _Norm(1280, device), _Norm(512, device)
    attention.index_weights = (torch.randn(index_heads, hidden_width, generator=generator) * 0.03125).to(
        device=device, dtype=torch.bfloat16,
    )
    attention.geometry = NS(index_n_heads=index_heads, index_head_dim=index_width)
    frequencies = torch.ones(max(p + n for p, n in zip(state.query_prefixes, state.lengths)),
                             index_width // 2, dtype=torch.complex64, device=device)
    attention.runtime = NS(freqs_for=lambda _: frequencies)
    attention.project_indexer_query = lambda latent, quantized_input: attention.index_wq.forward_quantized(*quantized_input)
    with patch.object(qkv, "_physical_pair_supported", return_value=True):
        attention._local_prefill_qkv = qkv.PreparedQKV.try_prepare(attention.wq_a, attention.wkv)
    return attention, _Projector(attention)


def _identity_rope(value, frequencies):
    return value


class CP8L20ProjectionContracts(unittest.TestCase):
    def setUp(self):
        guard = torch.no_grad()
        guard.__enter__()
        self.addCleanup(guard.__exit__, None, None, None)
        env = patch.dict(os.environ, {DENSE_PROJECTIONS: "1", MEGA_ATTENTION: "1",
                                      "RTP_V41_PPU_PTPC_FUSION": "1"})
        env.start()
        self.addCleanup(env.stop)

    def test_actual_host_plan_supports_ragged_requests_and_owner_redistribution(self):
        for rank in range(8):
            state = _state(rank, lengths=(96, 528, 257), prefixes=(0, 128, 8192))
            proof = state._cp_l20_projection_proof
            self.assertTrue(proof.current(state))
            self.assertEqual(state.plan.context.chunk_lengths_per_req, (12, 18, 18))
            self.assertEqual(state.plan.exchange.original_rows, state.original_context.chunk_length)
            self.assertEqual(state.plan.context.chunk_length, 48)
        state = _state(0, lengths=(4096,) * 8, prefixes=(126976,) * 8)
        self.assertEqual(len(state.plan.exchange.candidate_rows_host), 1120)
        self.assertEqual(state.plan.context.chunk_length, 144)
        for rank in range(1, 8):
            self.assertEqual(len(_state(rank, lengths=(4096,) * 8, prefixes=(126976,) * 8)
                                 .plan.exchange.candidate_rows_host), 0)

    def test_cp140_factory_requests_one_versioned_plan_without_temporary128(self):
        state = _state()
        plan = state.plan
        with patch.object(ppu_ced.CEDPlan, "create", return_value=plan) as create:
            actual, reason = ppu_ced.bounded_plan(NS(), state.original_context, NS(), NS(), state.runtime.policy)
        self.assertIs(actual, plan)
        self.assertEqual(reason, "compact140")
        self.assertEqual(create.call_count, 1)
        self.assertEqual(create.call_args.kwargs["bounded_replay_keep_tokens"], 140)
        self.assertIs(create.call_args.kwargs["versioned_metadata"], True)

    def test_transport_mutation_rebinding_and_unknown_inference_fail_closed(self):
        for change in ("send_mutate", "receive_clone", "positions", "plan", "host", "source"):
            with self.subTest(change=change):
                state = _state()
                proof = state._cp_l20_projection_proof
                if change == "send_mutate":
                    state.plan.exchange.send_indices.add_(0)
                elif change == "receive_clone":
                    state.plan.exchange.receive_positions = state.plan.exchange.receive_positions.clone()
                elif change == "positions":
                    state.plan.context.global_positions.add_(0)
                elif change == "plan":
                    state.plan = CEDPlan(**{name: getattr(state.plan, name) for name in (
                        "original_context", "context", "cu_seqlens", "exchange", "indexer_groups", "_capture_ids")})
                elif change == "host":
                    state.plan.exchange.projection_groups_host = ()
                else:
                    state.original_context.local_is_real.fill_(True)
                self.assertFalse(proof.current(state))
        state = _state()
        with torch.inference_mode():
            state.plan.context.global_positions = state.plan.context.global_positions.clone()
        with self.assertRaisesRegex(ValueError, "versioned transport"):
            qkv.CP8QueryPlanBirth(state.plan)

    def test_original_inference_sources_are_readonly_owners_not_replacement_proof(self):
        state = _state()
        with torch.inference_mode():
            state.original_context.global_positions = state.original_context.global_positions.clone()
        birth = qkv.CP8QueryPlanBirth(state.plan)
        state.plan._ppu_cp8_query_birth = birth
        proof = qkv.CP8L20ProjectionProof(state, birth)
        self.assertTrue(proof.current(state))
        state.original_context.global_positions = state.original_context.global_positions.clone()
        self.assertFalse(proof.current(state))

    def test_unsupported_shared_host_domains_do_not_start_joined_projection(self):
        for change in ("dp", "decode", "spec", "graph", "compacted", "off"):
            state = _state()
            attention, mega = _attention(state)
            attention._local_prefill_qkv.project = Mock(side_effect=AssertionError("must not project"))
            if change == "dp":
                state.runtime.model._prefill_cp_size = 1
            elif change == "decode":
                state.prefill = False
            elif change == "spec":
                state.runtime.model._is_speculative = True
            elif change == "graph":
                state.is_graph = True
            elif change == "compacted":
                state.compacted = True
            with patch.dict(os.environ, {DENSE_PROJECTIONS: "0" if change == "off" else "1"}):
                self.assertIsNone(qkv.DeferredCP8QueryProjection.prepare(
                    attention, state, torch.zeros(state.original_context.chunk_length, 16, dtype=torch.bfloat16), mega,
                ))

    def test_deferred_prepare_preserves_full_producer_and_preflight_never_changes_collective(self):
        state = _state()
        attention, mega = _attention(state)
        hidden = torch.randn(state.original_context.chunk_length, 16, dtype=torch.bfloat16)
        original = attention._local_prefill_qkv.project(hidden)
        pending = qkv.DeferredCP8QueryProjection.prepare(attention, state, hidden, mega, defer_kv_norm=True)
        self.assertTrue(torch.equal(pending.kv, original[1]))
        compact = Mock(side_effect=AssertionError("no collective before full producers"))
        state.plan.exchange.compact = compact
        with self.assertRaisesRegex(RuntimeError, "producer publication"):
            pending.finish(rope_indexer=_identity_rope, producer_complete=False)
        state.plan.exchange.send_indices.add_(0)
        with self.assertRaisesRegex(RuntimeError, "epoch changed"):
            pending.finish(rope_indexer=_identity_rope, producer_complete=True)
        compact.assert_not_called()

    def test_prepared_binding_mutation_is_error_not_local_fallback(self):
        state = _state()
        attention, mega = _attention(state)
        hidden = torch.zeros(state.original_context.chunk_length, 16, dtype=torch.bfloat16)
        pending = qkv.DeferredCP8QueryProjection.prepare(attention, state, hidden, mega)
        attention.index_weights.add_(0)
        state.plan.exchange.compact = Mock(side_effect=AssertionError("no changed protocol"))
        with self.assertRaisesRegex(RuntimeError, "epoch changed"):
            pending.finish(rope_indexer=_identity_rope, producer_complete=True)

    def test_finish_transports_small_raw_qr_and_full_weight_result_after_producers(self):
        state = _state()
        attention, mega = _attention(state)
        hidden = torch.randn(state.original_context.chunk_length, 16, dtype=torch.bfloat16)
        pending = qkv.DeferredCP8QueryProjection.prepare(attention, state, hidden, mega, defer_kv_norm=True)
        producer = pending.kv.clone()
        rows = state.plan.context.chunk_length
        selections = torch.arange(rows) % hidden.shape[0]
        calls = []

        def compact(value):
            calls.append(tuple(value.shape))
            return value.index_select(0, selections)

        state.plan.exchange.compact = compact
        expected_latent = attention.q_norm(pending.raw_latent.index_select(0, selections))
        expected_query = attention.wq_b(expected_latent)
        expected_index = attention.index_wq(expected_latent).reshape(rows, 2, 8)
        expected_weights = F.linear(hidden, attention.index_weights).index_select(0, selections).float() / 4
        with patch.object(qkv.F, "linear", wraps=F.linear) as weights:
            actual = pending.finish(rope_indexer=_identity_rope, producer_complete=True)
        self.assertEqual(calls, [(hidden.shape[0], 1280), (hidden.shape[0], 2), tuple(hidden.shape)])
        self.assertIs(weights.call_args.args[0], hidden, "full-M BF16 weight arithmetic must remain unchanged")
        self.assertTrue(torch.equal(actual.query, expected_query))
        self.assertTrue(torch.equal(actual.indexer_query, expected_index))
        self.assertTrue(torch.equal(actual.indexer_weights, expected_weights))
        self.assertTrue(torch.equal(pending.kv, producer))
        self.assertTrue(state.shared["cp_l20_projection"]["successful"])
        with self.assertRaisesRegex(RuntimeError, "cannot be retried"):
            pending.finish(rope_indexer=_identity_rope, producer_complete=True)

    def test_body_hook_runs_after_real_full_producer_writes_before_callback(self):
        # The real existing producer fixture executes SWA/global/Indexer/CSA
        # writes and Block carry. Only CP gathering and deferred projections
        # are synthetic here; actual WORLD8/native projection is tested below.
        from rtp_llm.models_py.modules.dsv41.ppu import attention_layer
        from rtp_llm.models_py.modules.dsv41.ppu import layers as layer_module

        results = []
        for candidate in (False, True):
            block, state, residual, previous, receipt = _component("cpu", 512)
            state.cache.sharded = True
            original = state.original_context
            original.cp_size = 8
            original.kv_cache_sharded = True
            state.context = original
            block.attn._cp_ctx = original
            mega = NS(owner=block.attn, project_q=lambda value: value,
                      prefill_reason=lambda *args, **kwargs: None,
                      prefill=lambda query, *args, **kwargs: query)
            block.attn._get_mega_attention = lambda _: mega

            class Pending:
                def __init__(self, hidden):
                    self.hidden = hidden
                    self.kv = hidden[:, :8].contiguous()

                def finish(self, *, rope_indexer, producer_complete):
                    if not producer_complete:
                        raise AssertionError("producer completion not declared")
                    for tag in ("global_kv_2", "indexer_kv", "csa_state"):
                        if tag not in receipt.writes:
                            raise AssertionError(f"{tag} publication must precede deferred queries")
                    receipt.events.append(("deferred_finish", self.hidden.shape[0]))
                    selected = self.hidden[-140:].contiguous()
                    query = selected[:, :8].contiguous()
                    return qkv.CP8DeferredQueries(selected, query, query,
                                                  query.reshape(-1, 1, 8),
                                                  query.new_zeros((140, 1), dtype=torch.float32), None)

            def prepare(attention, actual_state, hidden, actual_mega, **kwargs):
                return Pending(hidden) if candidate else None

            with ExitStack() as stack:
                # This CPU coordinator fixture supplies an independent norm
                # boundary. Actual PPUSharedRMSNorm dispatch is qualified by
                # CP8L20NativeProjectionRows, without a production CPU bypass.
                stack.enter_context(patch.object(layer_module, "rms_norm", side_effect=lambda x, w, eps:
                    (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)
                     * w.float()).to(x.dtype)))
                stack.enter_context(patch.object(attention_layer._step_prefill, "rms_norm",
                    side_effect=lambda x, w, eps:
                    (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)
                     * w.float()).to(x.dtype)))
                stack.enter_context(patch.dict(os.environ, {
                    attention_layer.FUSION_ENV: "0", attention_layer.STREAMS_ENV: "0",
                    "RTP_V41_PPU_NORM_STORE": "0", "RTP_V41_PPU_CP8_YOCO_Q8_ATTENTION": "0",
                    "RTP_V41_PPU_LOCAL_PREFILL_WORKSPACE": "0",
                    "RTP_V41_PPU_CP8_DENSE_INDEX_BOUNDS": "0",
                }))
                stack.enter_context(patch.object(attention_layer, "optimized_diagnostics_enabled", return_value=False))
                stack.enter_context(patch.object(attention_layer, "cp8_yoco_operator_eligible", return_value=False))
                stack.enter_context(patch.object(attention_layer, "cp_all_gather_full", side_effect=lambda value, *a, **kw: value))
                stack.enter_context(patch.object(attention_layer, "_local_prefill_dense_projections_enabled", return_value=False))
                stack.enter_context(patch.object(attention_layer, "build_cp_prefill_workspace",
                    side_effect=lambda **kwargs: (torch.zeros((1, 1, 8), dtype=torch.bfloat16),
                                                   torch.zeros((140, 1, 1), dtype=torch.int32),
                                                   torch.ones(140, dtype=torch.int32))))
                stack.enter_context(patch.object(qkv.DeferredCP8QueryProjection, "prepare", side_effect=prepare))
                # The fixture's original context is logical B1; do not claim
                # its synthetic CP tag proves a multi-rank first-token check.
                activate = state._activate_compact

                def local_activate(hidden):
                    original.cp_size, original.kv_cache_sharded = 1, False
                    return activate(hidden)

                state._activate_compact = local_activate
                output = block._forward_compact_l20(residual, state.inputs.input_ids, state, None)
            names = [name for name, _ in receipt.events]
            for tag in ("global_kv_2", "indexer_kv", "csa_state"):
                self.assertLess(names.index(tag), names.index("callback"))
                if candidate:
                    self.assertLess(names.index(tag), names.index("deferred_finish"))
            if candidate:
                self.assertLess(names.index("deferred_finish"), names.index("callback"))
                self.assertTrue(state.shared["dense_projection_calls"][0]["cp_raw_qr_transport"])
            self.assertEqual(receipt.events[0], ("swa", 512))
            results.append((output, state.cache.swa, state.cache.global_rows,
                            state.cache.index, state.cache.csa))
        for expected, actual in zip(*results):
            self.assertTrue(torch.equal(expected, actual))


@unittest.skipUnless(os.environ.get("RTP_V41_TEST_LOCAL_WORKSPACE_DEVICE") == "1", "explicit PPU device test")
class CP8L20NativeProjectionRows(unittest.TestCase):
    @torch.no_grad()
    def test_actual_shared_q_norm_strided_full_and_contiguous_compact_local_error(self):
        import json
        from rtp_llm.models_py.modules.dsv41.ppu.layers import PPUSharedRMSNorm
        from rtp_llm.models_py.modules.dsv41.ppu.test.test_mhc_mega_probe import _qualification_metric

        self.assertTrue(is_ppu())
        generator = torch.Generator().manual_seed(20261117)
        # Preserve the joined Q-a/KV-a output's actual gapped QR row stride.
        joined = torch.randn(4096, 1792, generator=generator).to(device="cuda:0", dtype=torch.bfloat16)
        raw_qr = joined[:, :1280]
        selected = torch.tensor([start + row for start in range(0, 4096, 512)
                                 for row in (*range(9), *range(503, 512))], device=joined.device)
        gamma = (torch.randn(1280, generator=generator) * 0.0625 + 1).to(
            device=joined.device, dtype=torch.bfloat16,
        )
        original, original_gamma = joined.clone(), gamma.clone()
        reports = []
        for eps in (1e-6, 1e-20):
            norm = PPUSharedRMSNorm(gamma, eps)
            expected = norm(raw_qr, allow_strided=True).index_select(0, selected)
            actual = norm(raw_qr.index_select(0, selected), allow_strided=True)
            torch.cuda.synchronize()
            metric = _qualification_metric(torch, actual, expected)
            exact = torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
            self.assertEqual(metric["qualified_status"], "PASS", (eps, exact, metric))
            reports.append({"epsilon": eps, "full_rows": 4096, "compact_rows": 144,
                            "full_row_stride": raw_qr.stride(0), "exact_bytes": exact, "metric": metric})
        self.assertTrue(torch.equal(joined, original))
        self.assertTrue(torch.equal(gamma, original_gamma))
        report = {"schema": "cp8-l20-native-q-norm-v1", "cases": reports}
        directory = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
        if directory:
            (Path(directory) / "cp8-l20-native-q-norm.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"schema": report["schema"], "cases": [
            {**case, "metric": {key: value for key, value in case["metric"].items() if key != "row_metrics"}}
            for case in reports
        ]}, sort_keys=True))

    @torch.no_grad()
    def test_reordered144_native_fp8_queries_match_full_row_bytes(self):
        self.assertTrue(is_ppu())
        generator = torch.Generator().manual_seed(20261117)
        latent = torch.randn(4096, 1280, generator=generator).to(device="cuda:0", dtype=torch.bfloat16)
        selected = torch.tensor([start + row for start in range(0, 4096, 512)
                                 for row in (*range(9), *range(503, 512))], device=latent.device)
        for width in (32768, 4096):
            linear = _linear(width, 1280, latent.device, generator)
            expected = linear(latent).index_select(0, selected)
            actual = linear(latent.index_select(0, selected))
            torch.cuda.synchronize()
            self.assertTrue(torch.equal(expected.view(torch.uint8), actual.view(torch.uint8)), width)


@patch.dict(os.environ, {DENSE_PROJECTIONS: "1", MEGA_ATTENTION: "1",
                         "RTP_V41_PPU_PTPC_FUSION": "1"})
def _world8_worker(rank, store):
    import torch.distributed as dist

    torch.cuda.set_device(rank)
    try:
        dist.init_process_group("nccl", init_method="file://" + store, rank=rank, world_size=8,
                                timeout=timedelta(seconds=180))
        with torch.no_grad():
            device = torch.device("cuda", rank)
            state = _state(rank, lengths=(512, 528), prefixes=(126976, 4096), device=device, group=dist.group.WORLD)
            attention, mega = _attention(state, hidden_width=5120, query_width=32768, index_heads=32, index_width=128)
            positions = state.original_context.global_positions.float()
            hidden = ((positions[:, None] % 97) * 0.0078125
                      + torch.arange(5120, device=device)[None] % 13 * 0.00390625).to(torch.bfloat16)
            exchange = state.plan.exchange
            raw_qr, raw_kv = attention._local_prefill_qkv.project(hidden)
            latent = attention.q_norm(raw_qr)
            expected_query = exchange.compact(attention.wq_b(latent))
            expected_index = exchange.compact(attention.index_wq(latent)).reshape(-1, 32, 128)
            expected_weights = exchange.compact(F.linear(hidden, attention.index_weights)).float() / 64
            expected_hidden = exchange.compact(hidden)
            pending = qkv.DeferredCP8QueryProjection.prepare(attention, state, hidden, mega, defer_kv_norm=True)
            producer_before = pending.kv.clone()
            result = pending.finish(rope_indexer=_identity_rope, producer_complete=True)
            torch.cuda.synchronize()
            for expected, actual in ((expected_query, result.query), (expected_index, result.indexer_query),
                                     (expected_weights, result.indexer_weights), (expected_hidden, result.hidden),
                                     (raw_kv, pending.kv), (producer_before, pending.kv)):
                if not torch.equal(expected.view(torch.uint8), actual.view(torch.uint8)):
                    raise AssertionError(f"rank {rank}: CP140 projection/producer bytes differ")
            dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@unittest.skipUnless(os.environ.get("RTP_V41_TEST_CP8_WORLD8") == "1", "explicit eight-rank CP collective test")
class CP8L20World8(unittest.TestCase):
    def test_raw_qr_transport_preserves_native_queries_weights_hidden_and_full_kv(self):
        self.assertTrue(is_ppu())
        import torch.multiprocessing as mp

        with tempfile.TemporaryDirectory(prefix="cp8-l20-query-") as directory:
            mp.spawn(_world8_worker, args=(str(Path(directory) / "store"),), nprocs=8, join=True)


if __name__ == "__main__":
    unittest.main()
