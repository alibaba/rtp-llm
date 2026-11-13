"""L20 retained-query projection with unchanged full producer publication."""

from contextlib import ExitStack
import os
import unittest
from unittest.mock import patch
from types import SimpleNamespace as NS

import torch

from rtp_llm.models_py.modules.dsv41.ppu import attention_layer
from rtp_llm.models_py.modules.dsv41.ppu._quant import is_ppu
from rtp_llm.models_py.modules.dsv41.ppu.linear import PPUPTPCLinear
from rtp_llm.models_py.modules.dsv41.ppu.test.test_local_prefill_split_l20 import _component, _state


ENV = attention_layer.LOCAL_PREFILL_DENSE_PROJECTIONS_ENV


def _stamp(state):
    state.local_prefill_sg_align = True
    state.plan_reason = "compact140"
    proof = state._local_compact_rows_key(state.plan.context, before_l20=True)
    if proof is None:
        raise AssertionError("fixture must satisfy the real local transport descriptor")
    state._l20_query_projection_proof = (state.plan, state.plan.context, proof[0], proof[1])


class L20QueryProjectionContracts(unittest.TestCase):
    @torch.no_grad()
    def test_suffix_requires_original_plan_and_unchanged_transport(self):
        with patch.dict(os.environ, {ENV: "1"}):
            state = _state()
            _stamp(state)
            self.assertEqual(state.local_l20_query_slice(4096), slice(3956, 4096))
            for tensor_name in ("send_indices", "receive_positions"):
                state = _state()
                _stamp(state)
                getattr(state.plan.exchange, tensor_name).add_(0)
                self.assertIsNone(state.local_l20_query_slice(4096), tensor_name)
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
                    self.assertIsNone(state.local_l20_query_slice(4095 if change == "rows" else 4096), change)

    @torch.inference_mode()
    def test_full_cache_publication_and_selected_outputs_are_unchanged(self):
        results = []
        environment = {
            ENV: "1", "RTP_V41_PPU_LOCAL_PREFILL_WORKSPACE": "1",
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
