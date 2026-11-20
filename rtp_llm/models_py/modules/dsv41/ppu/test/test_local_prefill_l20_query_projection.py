"""L20 retained-query projection with unchanged full producer publication."""

from contextlib import ExitStack
import os
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace as NS

import torch

from rtp_llm.models_py.modules.dsv41.ppu import attention_layer, layers
from rtp_llm.models_py.modules.dsv41.ppu._quant import is_ppu
from rtp_llm.models_py.modules.dsv41.ppu.linear import PPUPTPCLinear
from rtp_llm.models_py.modules.dsv41.ppu.optimization_config import DENSE_PROJECTIONS
from rtp_llm.models_py.modules.dsv41.ppu.test.test_local_prefill_rows import public_ced_cpu_scope
from rtp_llm.models_py.modules.dsv41.ppu.test.test_local_prefill_split_l20 import _component, _state


ENV = attention_layer.LOCAL_PREFILL_DENSE_PROJECTIONS_ENV


def _stamp(state, *, sg_align=True):
    state.local_prefill_sg_align = sg_align
    state.plan_reason = "compact140"
    proof = state._local_compact_rows_key(state.plan.context, before_l20=True)
    if proof is None:
        raise AssertionError("fixture must satisfy the real local transport descriptor")
    state._l20_query_projection_proof = (state.plan, state.plan.context, proof[0], proof[1])


class L20QueryProjectionContracts(unittest.TestCase):
    @torch.no_grad()
    def test_global_dense_admission_does_not_require_legacy_flag_or_sg_alignment(self):
        state = NS(local_prefill_sg_align=False, prefill=True, idle=False)
        x = NS(is_cuda=True, dtype=torch.bfloat16, ndim=2, shape=(4096, 5120),
               is_contiguous=lambda: True, requires_grad=False)
        geometry = NS(q_lora_rank=1280, head_dim=512)
        with patch.dict(os.environ, {ENV: "0", DENSE_PROJECTIONS: "1"}), patch.object(
            torch.cuda, "is_current_stream_capturing", return_value=False,
        ):
            self.assertTrue(attention_layer._local_prefill_dense_projections_enabled(
                state, None, x, geometry, diagnostic=None, packed_plan=None, parallel=False,
            ))
            self.assertFalse(attention_layer._local_prefill_dense_projections_enabled(
                state, NS(cp_size=8, kv_cache_sharded=True), x, geometry,
                diagnostic=None, packed_plan=None, parallel=False,
            ))
            with patch.dict(os.environ, {DENSE_PROJECTIONS: "0"}):
                self.assertFalse(attention_layer._local_prefill_dense_projections_enabled(
                    state, None, x, geometry, diagnostic=None, packed_plan=None, parallel=False,
                ))

    @torch.no_grad()
    def test_suffix_requires_original_plan_and_unchanged_transport(self):
        with patch.dict(os.environ, {ENV: "1", DENSE_PROJECTIONS: "0"}):
            state = _state()
            _stamp(state)
            self.assertEqual(state.local_l20_query_rows(4096), slice(3956, 4096))
            for tensor_name in ("send_indices", "receive_positions"):
                state = _state()
                _stamp(state)
                getattr(state.plan.exchange, tensor_name).add_(0)
                self.assertIsNone(state.local_l20_query_rows(4096), tensor_name)
            for change in ("plan", "host", "ragged", "rows", "compacted", "sg", "off", "cp"):
                state = _state()
                _stamp(state)
                if change == "plan":
                    state.plan = type(state.plan)(**vars(state.plan))
                elif change == "host":
                    state.plan.exchange.candidate_rows_host = tuple(range(140))
                elif change == "ragged":
                    state.lengths = (2048, 2048)
                elif change == "compacted":
                    state.compacted = True
                elif change == "sg":
                    state.local_prefill_sg_align = False
                elif change == "cp":
                    state.plan.context.cp_size = 8
                with patch.dict(os.environ, {ENV: "0" if change == "off" else "1"}):
                    self.assertIsNone(state.local_l20_query_rows(4095 if change == "rows" else 4096), change)

    @torch.no_grad()
    def test_global_fusion_uses_versioned_suffix_without_sg_align(self):
        with patch.dict(os.environ, {ENV: "0", DENSE_PROJECTIONS: "1"}):
            state = _state()
            _stamp(state, sg_align=False)
            self.assertEqual(state.local_l20_query_rows(4096), slice(3956, 4096))
            for change in ("send_write", "send_replace", "mask_write", "positions_write",
                           "plan", "host", "graph", "verify", "proposal", "attn_verify",
                           "decode", "compacted", "cp", "off"):
                with self.subTest(change=change):
                    state = _state()
                    _stamp(state, sg_align=False)
                    if change == "send_write":
                        state.plan.exchange.send_indices.add_(0)
                    elif change == "send_replace":
                        state.plan.exchange.send_indices = state.plan.exchange.send_indices.clone()
                    elif change == "mask_write":
                        state.plan.context.local_is_real.fill_(True)
                    elif change == "positions_write":
                        state.plan.context.global_positions.add_(0)
                    elif change == "plan":
                        state.plan = type(state.plan)(**vars(state.plan))
                    elif change == "host":
                        state.plan.exchange.candidate_rows_host = tuple(range(140))
                    elif change in ("graph", "verify", "proposal"):
                        setattr(state, "is_graph" if change == "graph" else change, True)
                    elif change == "attn_verify":
                        state.attn.is_target_verify = True
                    elif change == "decode":
                        state.prefill = False
                    elif change == "compacted":
                        state.compacted = True
                    elif change == "cp":
                        state.plan.context.cp_size = 8
                    with patch.dict(os.environ, {DENSE_PROJECTIONS: "0" if change == "off" else "1"}):
                        self.assertIsNone(state.local_l20_query_rows(4096))
            with torch.inference_mode():
                state = _state()
                _stamp(state, sg_align=False)
                self.assertIsNone(state.local_l20_query_rows(4096), "unknown versionless transport")

    @torch.inference_mode()
    def test_prepare_plan_certifies_global_fusion_transport_at_birth(self):
        with patch.dict(os.environ, {
            ENV: "0", DENSE_PROJECTIONS: "1",
            "RTP_V41_PPU_LOCAL_PREFILL_HOST_METADATA": "1",
        }):
            state = _state()
            state.local_prefill_sg_align = False
            state.plan = None
            with public_ced_cpu_scope(state):
                # Exercise CEDPlan.create's real inference decorator and the
                # HOST_METADATA single-layout branch, which reuses its views.
                state.prepare_plan()
            self.assertEqual(state.local_l20_query_rows(4096), slice(3956, 4096))
            self.assertFalse(state.plan.exchange.send_indices.is_inference())
            state.plan.exchange.send_indices.add_(0)
            self.assertIsNone(state.local_l20_query_rows(4096))

    @torch.no_grad()
    def test_batched_plan_selects_each_request_tail_and_detects_mutation(self):
        with patch.dict(os.environ, {
            DENSE_PROJECTIONS: "1", "RTP_V41_PPU_LOCAL_PREFILL_HOST_METADATA": "1",
        }):
            for count in (2, 4):
                with self.subTest(count=count), torch.inference_mode():
                    state = _state(rows=4096 * count)
                    state.lengths = (4096,) * count
                    state.query_prefixes *= count
                    state.attn.cu_seqlens = torch.arange(count + 1, dtype=torch.int32) * 4096
                    state.plan = state.original_context = None
                    with public_ced_cpu_scope(state):
                        state.prepare_plan()
                    rows = 4096 * count
                    self.assertIsNone(state.local_l20_query_rows(rows))
                    retained = state.local_l20_query_rows(rows, allow_batch=True)
                    self.assertIs(retained, state.plan.exchange.send_indices)
                    expected = [row for end in range(4096, rows + 1, 4096) for row in range(end - 140, end)]
                    self.assertEqual(retained.tolist(), expected)
                    self.assertEqual(state.plan.context.chunk_length, 140 * count)
                    retained.add_(0)
                    self.assertIsNone(state.local_l20_query_rows(rows, allow_batch=True))

    @torch.no_grad()
    def test_mega_projector_and_indexer_share_retained_quantization_full_kv(self):
        x = torch.zeros((4096, 8), dtype=torch.bfloat16)
        joined = torch.zeros((4096, 1792), dtype=torch.bfloat16)
        raw_qr, raw_kv = joined[:, :1280], joined[:, 1280:]
        payload = torch.zeros((140, 1280), dtype=torch.float8_e4m3fn)
        scale = torch.ones((140, 1), dtype=torch.float32)
        pair = payload, scale
        q = torch.zeros((140, 8), dtype=torch.bfloat16)
        prepared = Mock(return_value=(raw_qr, raw_kv))
        norm = Mock(side_effect=lambda value, **kwargs: value.contiguous())
        projector = Mock(return_value=q)
        op = NS(_local_prefill_qkv=NS(project=prepared), q_norm=norm,
                wq_a=Mock(side_effect=AssertionError("separate full Q projection")),
                wkv=Mock(side_effect=AssertionError("separate full KV projection")),
                wq_b=Mock(side_effect=AssertionError("ordinary Q projector")), index_wq=object())
        with patch.dict(os.environ, {ENV: "0", DENSE_PROJECTIONS: "1"}), patch.object(
            layers, "prepare_shared_ptpc_input", return_value=pair,
        ) as quantize:
            qr, actual_q, kv, indexer_pair = layers.PPUAttentionProjections.project_query_kv(
                op, x, prepare_indexer=True, reuse_ptpc=True, defer_kv_norm=True,
                query_rows=slice(3956, 4096), q_projector=projector,
            )
        prepared.assert_called_once_with(x)
        self.assertEqual(norm.call_args.args[0].shape, (140, 1280))
        self.assertEqual(norm.call_args.args[0].data_ptr(), raw_qr[3956:].data_ptr())
        self.assertTrue(norm.call_args.kwargs["allow_strided"])
        quantize.assert_called_once()
        self.assertIs(quantize.call_args.args[0], qr)
        self.assertIs(projector.call_args.args[0], qr)
        self.assertIs(projector.call_args.kwargs["quantized_input"], pair)
        self.assertIs(indexer_pair, pair)
        self.assertIs(actual_q, q)
        self.assertIs(kv, raw_kv)
        self.assertEqual(kv.shape, (4096, 512))

        index_projector = Mock(return_value=q)
        table = torch.ones((4096, 4), dtype=torch.complex64)
        attn = NS(geometry=NS(index_n_heads=1, index_head_dim=8), layer_id=20,
                  runtime=NS(freqs_for=lambda layer: table),
                  project_indexer_query=index_projector,
                  index_wq=Mock(side_effect=AssertionError("indexer requantization")),
                  index_weights=torch.ones((1, 8), dtype=torch.bfloat16))
        with patch.object(attention_layer, "_rope_tail", side_effect=lambda value, freq: value), \
                patch.object(attention_layer.F, "linear", return_value=torch.zeros((4096, 1))) as weights:
            index_q, index_weights = attention_layer.SharedPPUAttention._project_indexer(
                attn, x[-140:], qr, torch.arange(3956, 4096), {}, quantized_input=indexer_pair,
                weight_input=x, weight_rows=slice(3956, 4096),
            )
        self.assertIs(index_projector.call_args.args[0], qr)
        self.assertIs(index_projector.call_args.kwargs["quantized_input"], pair)
        self.assertEqual(index_q.shape, (140, 1, 8))
        self.assertEqual(index_weights.shape, (140, 1))
        weights.assert_called_once()
        self.assertIs(weights.call_args.args[0], x, "BF16 weights projection retains full M4096 arithmetic")

    @torch.inference_mode()
    def test_full_cache_publication_and_selected_outputs_are_unchanged(self):
        results = []
        environment = {
            ENV: "1", "RTP_V41_PPU_LOCAL_PREFILL_WORKSPACE": "1",
            DENSE_PROJECTIONS: "0",
            "RTP_V41_PPU_LOCAL_PREFILL_AIU_PRENORM": "0",
            "RTP_V41_PPU_QUANT_MODE": "mxfp4_fp8_ptpc",
            attention_layer.FUSION_ENV: "0", attention_layer.STREAMS_ENV: "0",
        }
        for early in (False, True):
            with ExitStack() as stack:
                stack.enter_context(patch.dict(os.environ, environment))
                stack.enter_context(patch.object(attention_layer, "optimized_diagnostics_enabled", return_value=False))
                stack.enter_context(patch.object(attention_layer, "prepare_packed_attention", return_value=(None, "disabled")))
                stack.enter_context(patch.object(attention_layer, "_local_prefill_dense_projections_enabled", return_value=early))
                stack.enter_context(patch.object(attention_layer, "_local_sg_rope_positions", side_effect=lambda state, cp, rows, table: rows[0]))
                stack.enter_context(patch.object(attention_layer, "flash_mla_sparse_fwd_ppu", side_effect=lambda q, *a, **kw: (q.clone(), None, None)))
                block, state, residual, previous, receipt = _component("cpu", 4096)
                _stamp(state)
                projected = []

                def project(value, *, query_rows=None, **kwargs):
                    query = value if query_rows is None else value[query_rows]
                    projected.append(("qb", query.shape[0]))
                    return query[:, :8].contiguous(), query[:, :8].contiguous(), value[:, :8].contiguous(), None

                block.attn.project_query_kv = project
                original_indexer = block.attn._project_indexer

                def indexer(value, *args, **kwargs):
                    projected.append(("indexer", value.shape[0]))
                    return original_indexer(value, *args, **kwargs)

                block.attn._project_indexer = indexer
                output = block._forward_compact_l20(residual, state.inputs.input_ids, state, None)
                self.assertEqual(projected, [("qb", 140 if early else 4096), ("indexer", 140 if early else 4096)])
                events = [event for event, _ in receipt.events]
                for tag in ("global_kv_2", "indexer_kv", "csa_state"):
                    self.assertLess(events.index(tag), events.index("callback"))
                self.assertLess(events.index("callback"), events.index("select"))
                self.assertEqual(receipt.events[0], ("swa", 4096))
                self.assertEqual(receipt.ffn_rows[0][0], 140)
                results.append((output, state.cache.swa, state.cache.global_rows,
                                state.cache.index, state.cache.csa, block.ffn_hc.state.pre_mix))
        for control, candidate in zip(*results):
            self.assertTrue(torch.equal(control, candidate))


@unittest.skipUnless(os.environ.get("RTP_V41_TEST_LOCAL_WORKSPACE_DEVICE") == "1", "explicit PPU device test")
class L20NativeProjectionRows(unittest.TestCase):
    @torch.inference_mode()
    def test_qb_and_indexer_retained_rows_match_full_native_gemm_bytes(self):
        self.assertTrue(is_ppu())
        generator = torch.Generator().manual_seed(20261113)
        x = torch.randn((4096, 1280), generator=generator).to(device="cuda:0", dtype=torch.bfloat16)
        for width in (32768, 4096):
            with self.subTest(width=width):
                weight = (torch.randn((width, 1280), generator=generator) * 0.1).to(
                    device=x.device, dtype=torch.float8_e4m3fn,
                )
                scale = torch.full((width, 1), 0.125, device=x.device, dtype=torch.float32)
                linear = PPUPTPCLinear(weight, scale)
                full = linear(x)[-140:].contiguous()
                compact = linear(x[-140:])
                torch.cuda.synchronize()
                self.assertTrue(torch.equal(full.view(torch.uint8).cpu(), compact.view(torch.uint8).cpu()), width)

    @torch.inference_mode()
    def test_indexer_weight_projection_preserves_full_bf16_gemm_and_exact_bytes(self):
        self.assertTrue(is_ppu())
        torch.manual_seed(20261114)
        x = torch.randn((4096, 5120), device="cuda:0", dtype=torch.bfloat16)
        weight = torch.randn((32, 5120), device=x.device, dtype=torch.bfloat16) * 0.01
        full = torch.nn.functional.linear(x, weight)[-140:].float() * (32 * 128) ** -0.5
        attn = attention_layer.SharedPPUAttention.__new__(attention_layer.SharedPPUAttention)
        torch.nn.Module.__init__(attn)
        attn.geometry = NS(index_n_heads=32, index_head_dim=128)
        attn.index_weights = weight
        attn.index_wq = lambda qr: torch.zeros((qr.shape[0], 4096), device=x.device, dtype=x.dtype)
        attn.layer_id = 20
        frequencies = torch.ones((4096, 32), device=x.device, dtype=torch.complex64)
        attn.runtime = NS(freqs_for=lambda layer: frequencies)
        with patch.object(attention_layer, "_rope_tail", side_effect=lambda value, freq: value), patch.object(
            attention_layer.F, "linear", wraps=torch.nn.functional.linear,
        ) as projection:
            _, compact = attn._project_indexer(
                x[-140:], torch.empty((140, 1280), device=x.device, dtype=x.dtype),
                torch.arange(3956, 4096, device=x.device), {},
                weight_input=x, weight_rows=slice(3956, 4096),
            )
        self.assertEqual(projection.call_count, 1)
        self.assertIs(projection.call_args.args[0], x)
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(full.view(torch.uint8).cpu(), compact.view(torch.uint8).cpu()))


if __name__ == "__main__":
    unittest.main()
