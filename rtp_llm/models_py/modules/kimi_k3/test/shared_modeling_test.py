"""Graph regressions for K3 residual and KDA output fusion."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import flashinfer
import torch
from torch import nn

from rtp_llm.models_py.model_desc.kimi_k3 import KimiK3DecoderLayer
from rtp_llm.models_py.model_desc.kimi_k3_mtp import KimiK3MtpModel
from rtp_llm.models_py.modules.hybrid.kda_attention import KdaAttention
from rtp_llm.models_py.modules.kimi_k3.residual import KimiK3AttentionResidual
from rtp_llm.models_py.triton_kernels.common.activation import (
    rmsnorm_sigmoid_gate_per_token_group_quant_fp8,
)
from rtp_llm.models_py.triton_kernels.common.layernorm_gated import (
    fused_q_kv_rmsnorm,
    rounded_add_rmsnorm,
)


def capture(run):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = run()
    torch.cuda.current_stream().wait_stream(stream)
    return graph, output


class _Attention(nn.Module):
    _fp8_collective = None
    _fp8_input_collective = None
    supports_fp8_activation = True

    def forward(self, hidden, *args):
        return hidden * 0.125


class _Mlp(nn.Module):
    def __init__(self, fused):
        super().__init__()
        self.fuses_layer_residual = fused

    def forward(self, hidden, valid_mask, residual=None, *, defer_output_add=False):
        routed, shared = hidden * 0.25, hidden * 0.0625
        if defer_output_add:
            return routed, shared
        combined = routed + shared
        return combined if residual is None else residual + combined


class _Norm(nn.Module):
    def __init__(self, weight, eps):
        super().__init__()
        self.weight = weight
        self.variance_epsilon = eps

    def forward(self, value):
        return flashinfer.norm.rmsnorm(value, self.weight, eps=self.variance_epsilon)


class SharedModelingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
            raise RuntimeError("K3 shared modeling graph tests require SM100 family")
        torch.cuda.set_device(0)
        torch.manual_seed(319)

    def random(self, *shape):
        return torch.randn(shape, dtype=torch.bfloat16, device="cuda")

    def residual(self):
        return KimiK3AttentionResidual(
            self.random(7168) * 0.02, self.random(7168) * 0.02, 1e-5
        )

    def test_packed_dual_norm_preserves_default_tuple_and_strides(self):
        for rows, q_width, kv_width in (
            (0, 1536, 512),
            (3, 1536, 512),
            (8, 7168, 7168),
        ):
            packed_input = self.random(rows, q_width + kv_width + 128)
            q = packed_input[:, :q_width]
            kv = packed_input[:, q_width : q_width + kv_width]
            qw, kvw = self.random(q_width), self.random(kv_width)
            separate = fused_q_kv_rmsnorm(q, kv, qw, kvw, 1e-5)
            self.assertIsInstance(separate, tuple)
            actual = fused_q_kv_rmsnorm(q, kv, qw, kvw, 1e-5, packed_output=True)
            self.assertTrue(actual.is_contiguous())
            torch.testing.assert_close(
                actual, torch.cat(separate, dim=-1), rtol=0, atol=0
            )

    def _mtp_fixture(self, rows, tp):
        model = KimiK3MtpModel.__new__(KimiK3MtpModel)
        nn.Module.__init__(model)
        model.tp_size, model.tp_rank = tp, tp - 1
        model._query_layout = None
        # Full TP input extent; the production method selects this rank's rows.
        physical_rows = rows * tp
        embedding = self.random(physical_rows, 7168 + 16)[:, :7168]
        inputs = SimpleNamespace(
            input_ids=torch.zeros(physical_rows, dtype=torch.int32, device="cuda"),
            input_hiddens=self.random(physical_rows, 7168),
        )
        model._embed_inputs = lambda *args, **kwargs: embedding
        model.enorm = _Norm(self.random(7168), 1e-5)
        model.hnorm = _Norm(self.random(7168), 1e-5)
        # Inspect the exact eh_proj input without adding GEMM rounding to the check.
        model.eh_proj = nn.Identity()
        return model, inputs, embedding

    def test_mtp_packed_input_graph_at_prefill_and_decode_shapes(self):
        # TP8 Prefill 32K/64K/128K; DP4TP4 Q1 and Verify4/update local rows.
        cases = [(p // 8, 8) for p in (32768, 65536, 131072)]
        cases += [(m, 4) for m in (1, 2, 4, 8, 16, 32)]
        for rows, tp in cases:
            model, inputs, embedding = self._mtp_fixture(rows, tp)
            with patch(
                "rtp_llm.models_py.triton_kernels.common.layernorm_gated.fused_q_kv_rmsnorm",
                wraps=fused_q_kv_rmsnorm,
            ) as fused:
                graph, actual = capture(lambda: model._project_local_mtp_input(inputs))
                self.assertTrue(fused.called)
                self.assertTrue(
                    all(
                        call.kwargs.get("packed_output")
                        for call in fused.call_args_list
                    )
                )
            for step in range(2):
                embedding.add_(step * 0.125)
                inputs.input_hiddens.add_(step * 0.0625)
                graph.replay()
                local = slice((tp - 1) * rows, tp * rows)
                expected = torch.cat(
                    (
                        model.enorm(embedding[local].clone()),
                        model.hnorm(inputs.input_hiddens[local]),
                    ),
                    dim=-1,
                )
                self.assertEqual(actual.shape, (rows, 14336))
                self.assertTrue(actual.is_contiguous())
                torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)

    def test_mtp_different_norm_eps_keeps_existing_path(self):
        model, inputs, embedding = self._mtp_fixture(8, 4)
        model.hnorm.variance_epsilon = 1e-6
        with patch(
            "rtp_llm.models_py.triton_kernels.common.layernorm_gated.fused_q_kv_rmsnorm",
            wraps=fused_q_kv_rmsnorm,
        ) as fused:
            graph, actual = capture(lambda: model._project_local_mtp_input(inputs))
            fused.assert_not_called()
        graph.replay()
        expected = torch.cat(
            (
                model.enorm(embedding[24:32].clone()),
                model.hnorm(inputs.input_hiddens[24:32]),
            ),
            dim=-1,
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_kda_composition_preserves_packed_qkv_stride(self):
        class Kernel(nn.Module):
            def __init__(self, tp):
                super().__init__()
                self.parallelism_config = SimpleNamespace(tp_size=1)
                self.local_num_k_heads = self.local_num_v_heads = 96 // tp
                self.head_k_dim = self.head_v_dim = 128

            def forward(self, qkv, forget, beta, *args):
                self.qkv_stride = qkv.stride(0)
                return qkv[:, : self.local_num_k_heads * 128]

        class Projection(nn.Module):
            def forward(self, latent):
                return latent

            def forward_quantized(self, values, scales):
                self.quantized = True
                return values

        class Norm(nn.Module):
            def forward(self, output, gate, *, row_indices=None, out=None):
                if row_indices is not None or out is not None:
                    raise AssertionError(
                        "Non-CP KDA must keep the ordinary norm layout"
                    )
                return output

        for is_prefill, tp, rows, packed_width in (
            (True, 8, 32768, 6400),
            (True, 8, 65536, 6400),
            (True, 8, 131072, 6400),
            *((False, 4, 4 * batch, 12544) for batch in (1, 8, 16, 32)),
        ):
            packed = self.random(rows, packed_width)
            prefill, decode, projection = Kernel(tp), Kernel(tp), Projection()
            module = KdaAttention(
                input_projection=lambda hidden: packed,
                forget_projection=projection,
                output_projection=nn.Identity(),
                fa_width=128,
                norm=Norm(),
                prefill=prefill,
                decode=decode,
                strided_fp8_forget=not is_prefill,
                strided_fp8_prefill_forget=True,
            )
            inputs = SimpleNamespace(
                is_prefill=is_prefill,
                is_mtp_draft_update=False,
                valid_token_mask=None,
            )
            metadata = SimpleNamespace(
                is_target_verify=not is_prefill,
                query_layout=None,
                outer_layout=None,
            )
            graph, output = capture(lambda: module(None, None, None, inputs, metadata))
            graph.replay()
            self.assertTrue(projection.quantized)
            self.assertEqual(
                (prefill if is_prefill else decode).qkv_stride, packed_width
            )
            self.assertEqual(output.shape, (rows, 96 // tp * 128))

    def test_rounded_add_norm_graph_at_draft_and_update_rows(self):
        # DP-local BS 1/8/16/32: Q1/SP4 and fixed Verify4/SP4.
        for rows in (1, 2, 4, 8, 16, 32):
            hidden, delta, weight = (
                self.random(rows, 7168),
                self.random(rows, 7168),
                self.random(7168),
            )
            graph, (residual, normalized) = capture(
                lambda: rounded_add_rmsnorm(hidden, delta, weight, 1e-5)
            )
            for step in range(2):
                delta.add_(step * 0.125)
                graph.replay()
                expected = hidden + delta
                torch.testing.assert_close(residual, expected, rtol=0, atol=0)
                torch.testing.assert_close(
                    normalized,
                    flashinfer.norm.rmsnorm(expected, weight, eps=1e-5),
                    rtol=1e-2,
                    atol=1e-2,
                )

    def test_fp8_attnres_shared_delta_graph_and_anchor_write(self):
        for rows in (1, 8, 16, 32):
            for active, write in ((0, 0), (2, -1), (8, -1), (8, 8)):
                module = self.residual()
                prefix, routed, shared = (self.random(rows, 7168) for _ in range(3))
                bank = self.random(rows, 9, 7168)
                source, original_bank = prefix.clone(), bank.clone()
                weight = self.random(7168)
                kwargs = dict(
                    output_norm_weight=weight,
                    output_norm_eps=1e-5,
                    num_blocks=active,
                    block_write_idx=write,
                )
                graph, actual = capture(
                    lambda: module.forward_fp8(
                        prefix, bank, delta=routed, shared_delta=shared, **kwargs
                    )
                )
                for step in range(2):
                    shared.add_(step * 0.125)
                    expected_prefix, expected_bank = (
                        source.clone(),
                        original_bank.clone(),
                    )
                    expected = module.forward_fp8(
                        expected_prefix, expected_bank, delta=routed + shared, **kwargs
                    )
                    prefix.copy_(source)
                    bank.copy_(original_bank)
                    graph.replay()
                    torch.testing.assert_close(prefix, expected_prefix, rtol=0, atol=0)
                    torch.testing.assert_close(bank, expected_bank, rtol=0, atol=0)
                    torch.testing.assert_close(
                        actual.values.float(), expected.values.float(), rtol=0, atol=0
                    )
                    torch.testing.assert_close(
                        actual.scale_wire.T[:rows],
                        expected.scale_wire.T[:rows],
                        rtol=0,
                        atol=0,
                    )

    def test_masked_norm_graph_reads_runtime_mask_and_ignores_invalid_nan(self):
        for batch in (1, 8, 16, 32):
            rows, heads = 4 * batch, 24
            packed = self.random(rows, 12544)
            gate = packed[:, 9216:12288].view(rows, heads, 128)
            output, weight = self.random(rows, heads, 128), self.random(128)
            valid = torch.ones(rows, device="cuda", dtype=torch.bool)
            graph, actual = capture(
                lambda: rmsnorm_sigmoid_gate_per_token_group_quant_fp8(
                    output, gate, weight, 1e-5, valid_rows=valid
                )
            )
            for step in (0, 1):
                valid.copy_(torch.arange(rows, device="cuda") % 2 == step)
                output.normal_()
                gate.normal_()
                output.masked_fill_(~valid[:, None, None], float("nan"))
                gate.masked_fill_(~valid[:, None, None], float("nan"))
                expected = rmsnorm_sigmoid_gate_per_token_group_quant_fp8(
                    torch.where(valid[:, None, None], output, 0),
                    torch.where(valid[:, None, None], gate, 0),
                    weight,
                    1e-5,
                )
                graph.replay()
                torch.testing.assert_close(
                    actual[0].float(), expected[0].float(), rtol=0, atol=0
                )
                torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)

    def test_deferred_decoder_chain_materializes_final_delta_and_anchors(self):
        layers = []
        for index in range(3):
            layer = KimiK3DecoderLayer.__new__(KimiK3DecoderLayer)
            nn.Module.__init__(layer)
            layer.index, layer.block_size = index, 2
            layer.attention = _Attention()
            layer.attention_residual, layer.mlp_residual = (
                self.residual(),
                self.residual(),
            )
            layer.attention_norm = SimpleNamespace(
                weight=self.random(7168), variance_epsilon=1e-5
            )
            layer.mlp_norm = SimpleNamespace(
                weight=self.random(7168), variance_epsilon=1e-5
            )
            layer.mlp = _Mlp(fused=index != 1)
            layer.apply_mlp = (
                layer._apply_fused_shared_mlp
                if layer.mlp.fuses_layer_residual
                else layer._apply_regular_mlp
            )
            layers.append(layer)
        final = self.residual()
        inputs = SimpleNamespace(is_prefill=False, is_mtp_draft_update=False)
        metadata = SimpleNamespace(
            is_target_verify=True, query_layout=None, outer_layout=None
        )
        for rows in (1, 8, 16, 32):
            hidden, bank = self.random(rows, 7168), self.random(rows, 2, 7168)
            original = hidden.clone()
            expected_bank = bank.clone()
            expected = hidden.clone()
            for layer in layers:
                expected = layer(
                    expected, expected_bank, None, None, inputs, metadata, None
                )
            expected = final(expected, expected_bank)

            def run():
                current, delta, shared = hidden, None, None
                for layer in layers:
                    current, delta, shared = layer(
                        current,
                        bank,
                        None,
                        None,
                        inputs,
                        metadata,
                        None,
                        mlp_delta=delta,
                        mlp_shared_delta=shared,
                        defer_mlp_residual=True,
                    )
                return final(current, bank, delta=delta, shared_delta=shared)

            graph, actual = capture(run)
            hidden.copy_(original)
            graph.replay()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(bank, expected_bank, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
