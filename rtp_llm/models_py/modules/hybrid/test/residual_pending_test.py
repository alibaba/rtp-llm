import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from rtp_llm.models_py.model_desc.qwen4_exp import Qwen4ExpDecoderLayer, Qwen4ExpModel
from rtp_llm.models_py.modules.hybrid.gated_residual import inject_into_residual
from rtp_llm.models_py.modules.hybrid.test import residual_fixtures as hc_fixture
from rtp_llm.ops.compute_ops import PyAttentionInputs, PyModelInputs


class _Attention(nn.Module):
    def forward(self, hidden_states, **kwargs):
        return hidden_states * 0.125


class _Mlp(nn.Module):
    def forward(self, hidden_states):
        return hidden_states * 0.25


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class ResidualPendingTest(unittest.TestCase):
    def layer(self):
        layer = Qwen4ExpDecoderLayer.__new__(Qwen4ExpDecoderLayer)
        nn.Module.__init__(layer)
        layer.attn_hyper_connection = hc_fixture.ResidualFp8Fixture().module()
        layer.mlp_hyper_connection = hc_fixture.ResidualFp8Fixture().module()
        layer.self_attn = _Attention()
        layer.mlp = _Mlp()
        return layer

    def test_model_business_rows_defer_cross_layer_injection(self):
        if torch.cuda.get_device_capability() != (12, 0):
            self.skipTest("model residual deferral is specific to SM120")
        torch.manual_seed(7383)
        first, second = self.layer(), self.layer()
        inputs = PyModelInputs()
        inputs.attention_inputs = PyAttentionInputs()
        for rows, graph, deferred in (
            (16384, False, True),
            (24576, False, True),
            (32768, False, True),
            (49152, False, True),
            (49153, False, False),
            (16, True, False),
        ):
            with self.subTest(rows=rows, graph=graph):
                x = torch.randn(rows, 10240, device="cuda", dtype=torch.bfloat16)
                expected = second(first(x, None), None)
                model = SimpleNamespace(
                    ple_layers={},
                    layers=[first, second],
                    kv_cache=None,
                    _capture_mtp_target_hidden=True,
                    _start_eager_ple_prefetch=lambda *_: {},
                    _try_fp8_hyper_embedding=lambda *_: None,
                    word_embedding=lambda _: x,
                    _initial_hyper_states=lambda value: value,
                    _build_attn_meta=lambda *_: None,
                    _layer_fmha_impl=lambda *_: None,
                    hyper_connection_mixer=lambda value: (value[:, :2560], None, None),
                )
                with patch(
                    "rtp_llm.models_py.model_desc.qwen4_exp._is_cuda_graph_forward",
                    return_value=graph,
                ), patch.object(
                    second.attn_hyper_connection,
                    "forward_with_injection",
                    wraps=second.attn_hyper_connection.forward_with_injection,
                ) as injected:
                    actual = Qwen4ExpModel._forward_with_ple_prefetch(
                        model, inputs, object()
                    )
                self.assertEqual(injected.call_count, int(deferred))
                torch.testing.assert_close(
                    actual.mtp_target_hidden_states, expected, rtol=0, atol=0
                )
                torch.testing.assert_close(
                    actual.hidden_states, expected[:, :2560], rtol=0, atol=0
                )

    def test_two_layers_and_materialized_ple_mtp_boundaries(self):
        torch.manual_seed(7382)
        first, second = self.layer(), self.layer()
        for rows in (4, 16, 4096, 12288, 16384, 24576, 32768, 49152):
            x = torch.randn(rows, 10240, device="cuda", dtype=torch.bfloat16)
            expected = second(first(x, None), None)
            base, output, gate = first.forward_deferred(x, None)
            base, output, gate = second.forward_deferred(
                base, None, pending_injection=(output, gate)
            )
            actual = inject_into_residual(base, output, gate)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            # A lexical injection consumes the materialized BF16 stream.
            lexical = torch.randn_like(x) * 0.01
            expected = second(first(x, None) + lexical, None)
            base, output, gate = first.forward_deferred(x, None)
            materialized = inject_into_residual(base, output, gate)
            base, output, gate = second.forward_deferred(materialized + lexical, None)
            torch.testing.assert_close(
                inject_into_residual(base, output, gate), expected, rtol=0, atol=0
            )


if __name__ == "__main__":
    unittest.main()
