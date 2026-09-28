"""DFlash2 decoder math contract using real CUDA/HIP convolution kernels."""

import unittest

import torch
from torch import nn
from torch.nn import functional as F

from rtp_llm.models_py.model_desc.qwen3_dflash2_model import (
    DFlash2GroupedConv,
    _DFlash2DecoderLayer,
)


def _reference_conv(x, coefficients, base, width, group_size):
    # Per-request loops deliberately avoid the production flat-row kernel's
    # indexing and never let the prior request contribute to a causal tap.
    result = torch.zeros_like(x, dtype=torch.float32)
    for start in range(0, x.shape[0], width):
        for position in range(width):
            row = start + position
            for tap in range(min(position + 1, base.shape[0])):
                weight = base[tap].float() + coefficients[
                    row, tap
                ].float().repeat_interleave(group_size)
                result[row] += weight * x[row - tap].float()
    return result.to(x.dtype)


class _Attention(nn.Linear):
    def forward(self, x, **kwargs):
        return super().forward(x)


class DFlash2DecoderGpuTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise RuntimeError("DFlash2 model GPU tests require CUDA or ROCm")

    def _convolution(self, hidden=80, width=8):
        base = torch.randn(2, 2, hidden, device="cuda") * 0.1
        base[:, 0] += 1
        projection = torch.randn(4 * hidden // 16, hidden, device="cuda") * 0.01
        return DFlash2GroupedConv(base, projection, hidden, 2, 16, width)

    def _reference_sublayer(self, x, conv, sublayer):
        coefficients = F.linear(x, conv.kernel_projection).reshape(x.shape[0], 2, 2, -1)
        prepared = _reference_conv(
            x,
            coefficients[:, 0],
            conv.base_kernel[0],
            conv.query_width,
            conv.group_size,
        )
        return _reference_conv(
            sublayer(prepared),
            coefficients[:, 1],
            conv.base_kernel[1],
            conv.query_width,
            conv.group_size,
        )

    def test_prepare_finish_share_original_input_coefficients(self):
        conv = self._convolution()
        hidden = torch.randn(16, 80, device="cuda")
        prepared, coefficients = conv.prepare(hidden)
        output = prepared * 3 + 0.2
        actual = conv.finish(output, coefficients)
        expected = self._reference_sublayer(hidden, conv, lambda x: x * 3 + 0.2)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
        self.assertEqual(coefficients.stride(0), 2 * conv.taps * conv.groups)

    def test_decoder_norm_convolution_and_residual_order(self):
        # Isolate the decoder's composition from paged attention, which has its
        # own numerical tests. Non-commuting norms/linears catch misplaced
        # residuals and recomputing finish coefficients from sublayer output.
        layer = _DFlash2DecoderLayer.__new__(_DFlash2DecoderLayer)
        nn.Module.__init__(layer)
        layer.input_layernorm = nn.LayerNorm(80, device="cuda")
        layer.post_attention_layernorm = nn.LayerNorm(80, device="cuda")
        layer.self_attn = _Attention(80, 80, bias=False, device="cuda")
        layer.mlp = nn.Sequential(
            nn.Linear(80, 112, device="cuda"),
            nn.SiLU(),
            nn.Linear(112, 80, device="cuda"),
        )
        layer.attention_conv = self._convolution()
        layer.mlp_conv = self._convolution()
        hidden = torch.randn(16, 80, device="cuda")
        with torch.inference_mode():
            expected = hidden + self._reference_sublayer(
                layer.input_layernorm(hidden), layer.attention_conv, layer.self_attn
            )
            expected = expected + self._reference_sublayer(
                layer.post_attention_layernorm(expected), layer.mlp_conv, layer.mlp
            )
            actual = layer(hidden, query_width=8)
        torch.testing.assert_close(actual, expected, atol=4e-6, rtol=4e-6)
        with self.assertRaisesRegex(ValueError, "query widths"):
            layer(hidden, query_width=7)


if __name__ == "__main__":
    unittest.main()
