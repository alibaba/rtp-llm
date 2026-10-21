import unittest

import torch
from torch import nn

from rtp_llm.models_py.model_desc.qwen4_exp import Qwen4ExpDecoderLayer
from rtp_llm.models_py.modules.hybrid.gated_residual import (
    inject_into_residual,
)
from rtp_llm.models_py.modules.qwen4_exp.test import (
    residual_norm_fp8_test as hc_fixture,
)


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
        layer.attn_hyper_connection = hc_fixture.ResidualNormFp8Test().module()
        layer.mlp_hyper_connection = hc_fixture.ResidualNormFp8Test().module()
        layer.self_attn = _Attention()
        layer.mlp = _Mlp()
        return layer

    def test_two_layers_and_materialized_ple_mtp_boundaries(self):
        torch.manual_seed(7382)
        first, second = self.layer(), self.layer()
        for rows in (4, 16, 4096, 12288):
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
