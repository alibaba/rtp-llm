import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
    sgl_per_token_group_quant_fp8,
)
from rtp_llm.models_py.modules.kimi_k3.residual import KimiK3AttentionResidual
from rtp_llm.models_py.model_desc.kimi_k3 import KimiK3DecoderLayer


class KimiK3Fp8AttnResTest(unittest.TestCase):
    def test_decode_fp8_producer_only_for_target_verify(self):
        residual = KimiK3AttentionResidual(torch.ones(4), torch.ones(4), 1e-6)
        residual.configure_verify_fp8(True)
        hidden = torch.ones((2, 4))
        anchors = torch.zeros((2, 1, 4))
        kwargs = dict(output_norm_weight=torch.ones(4), output_norm_eps=1e-6,
                      num_blocks=0, block_write_idx=-1)
        sentinel = object()
        with patch.object(residual, "forward_fp8", return_value=sentinel) as fused:
            actual = residual(hidden, anchors,
                              metadata=SimpleNamespace(is_target_verify=True), **kwargs)
            self.assertIs(actual, sentinel)
            self.assertEqual(fused.call_count, 1)
            ordinary = residual(hidden, anchors,
                                metadata=SimpleNamespace(is_target_verify=False), **kwargs)
            self.assertIsInstance(ordinary, torch.Tensor)
            self.assertEqual(fused.call_count, 1)

    def test_producer_falls_back_before_fp8_ag_capacity(self):
        class Hidden:
            is_cuda = True

            def __init__(self, rows):
                self.shape = (rows, 7168)

            def __add__(self, other):
                return self

        class Residual:
            def __init__(self):
                self.bf16_calls = 0
                self.fp8_calls = 0

            def __call__(self, hidden, *args, **kwargs):
                self.bf16_calls += 1
                return hidden

            def forward_fp8(self, hidden, *args, **kwargs):
                self.fp8_calls += 1
                return hidden

        class Attention:
            _fp8_collective = SimpleNamespace(
                enable_ag=True, can_run_ag=lambda rows: rows * 8 <= 65536)

            def __call__(self, hidden, *args):
                return hidden

        layer = KimiK3DecoderLayer.__new__(KimiK3DecoderLayer)
        torch.nn.Module.__init__(layer)
        layer.index = 0
        layer.block_size = 1
        layer.attention = Attention()
        layer.attention_residual = Residual()
        layer.mlp_residual = Residual()
        layer.attention_norm = SimpleNamespace(weight=None, variance_epsilon=1e-6)
        layer.mlp_norm = SimpleNamespace(weight=None, variance_epsilon=1e-6)
        layer.mlp = lambda hidden, mask: hidden
        inputs = SimpleNamespace(is_prefill=True, is_mtp_draft_update=False)
        with patch("torch.cuda.is_current_stream_capturing", return_value=False):
            layer.forward(Hidden(8192), None, None, None, inputs, None, None)
            layer.forward(Hidden(8193), None, None, None, inputs, None, None)
        self.assertEqual(layer.attention_residual.fp8_calls, 1)
        self.assertEqual(layer.attention_residual.bf16_calls, 1)

    @staticmethod
    def dequantize(values, scale_wire):
        m, k = values.shape
        packed = scale_wire.T[:m].contiguous()
        exponents = packed.view(torch.uint8).reshape(m, k // 128).float()
        scales = torch.exp2(exponents - 127)
        return (values.float().reshape(m, k // 128, 128) * scales[:, :, None]).reshape(m, k)

    def test_fused_producer_matches_bf16_residual_and_preserves_anchor(self):
        if not torch.cuda.is_available():
            self.skipTest("requires CUDA")
        torch.manual_seed(311)
        device = "cuda:0"
        m, k = 64, 7168
        norm = torch.randn(k, device=device, dtype=torch.bfloat16) * 0.02
        projection = torch.randn(k, device=device, dtype=torch.bfloat16) * 0.02
        output_norm = torch.ones(k, device=device, dtype=torch.bfloat16)
        residual = KimiK3AttentionResidual(norm, projection, 1e-6)

        for active_blocks, write_idx in ((0, 0), (1, -1)):
            with self.subTest(active_blocks=active_blocks):
                hidden = torch.randn((m, k), device=device, dtype=torch.bfloat16)
                anchors = torch.randn((m, 2, k), device=device, dtype=torch.bfloat16)
                reference_anchors = anchors.clone()
                reference = residual(
                    hidden.clone(), reference_anchors,
                    output_norm_weight=output_norm,
                    output_norm_eps=1e-6,
                    num_blocks=active_blocks,
                    block_write_idx=write_idx,
                )
                expected_values, expected_scales = sgl_per_token_group_quant_fp8(
                    reference,
                    group_size=128,
                    eps=1e-4,
                    column_major_scales=True,
                    scale_tma_aligned=True,
                    scale_ue8m0=True,
                )
                actual = residual.forward_fp8(
                    hidden.clone(), anchors,
                    output_norm_weight=output_norm,
                    output_norm_eps=1e-6,
                    num_blocks=active_blocks,
                    block_write_idx=write_idx,
                )
                torch.cuda.synchronize()
                torch.testing.assert_close(anchors, reference_anchors, rtol=0, atol=0)
                actual_dequant = self.dequantize(actual.values, actual.scale_wire)
                expected_wire = expected_scales.as_strided(
                    ((k + 511) // 512, (m + 3) // 4 * 4),
                    ((m + 3) // 4 * 4, 1),
                )
                expected_dequant = self.dequantize(expected_values, expected_wire)
                diff = (actual_dequant - expected_dequant).abs()
                print(
                    f"active_blocks={active_blocks} max_abs={diff.max().item()} "
                    f"mean_abs={diff.mean().item()} "
                    f"scale_mismatches={(actual.scale_wire != expected_wire).sum().item()}",
                    flush=True,
                )
                # The native BF16 AttnRes and Triton producer reduce the
                # multi-block softmax in different orders near FP8 boundaries.
                self.assertLess(diff.max().item(), 0.3)
                self.assertLess(diff.mean().item(), 0.002)
                self.assertLess((diff > 0.1).float().mean().item(), 0.01)


if __name__ == "__main__":
    unittest.main()
