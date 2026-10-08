import unittest

import torch

from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
    has_deep_gemm,
    is_deep_gemm_e8m0_used,
)
from rtp_llm.models_py.modules.factory.linear.impl.cuda.test import (
    fp8_deepgemm_linear_sm120_test as online_tests,
)


class CudaFp8DeepGEMMLinearSM100Test(
    online_tests.OnlineFp8LoaderTestBase, unittest.TestCase
):
    def test_sm100(self):
        self.assertTrue(has_deep_gemm())
        self.assertTrue(is_deep_gemm_e8m0_used())

    def test_skip_head_mid_matches_regular_fp8_gemm(self):
        head_splits = (128, 64, 128)
        left, gap, right = head_splits
        tokens, heads = 257, 12
        self.K = 512
        self.N = heads * (left + right)
        self.scale_K = self.K // 128
        self.scale_N = self.N // 128
        self.weight = torch.randn(
            self.K, self.N, dtype=torch.bfloat16, device=self.device
        ).to(torch.float8_e4m3fn)
        self.weight_scales = torch.rand(
            self.scale_K, self.scale_N, dtype=torch.float32, device=self.device
        )
        linear = self._create_cuda_fp8_linear()
        inputs = torch.randn(tokens, self.K, dtype=torch.bfloat16, device=self.device)
        output = torch.full(
            (tokens, heads * sum(head_splits)), 11.0,
            dtype=torch.bfloat16, device=self.device,
        )

        self.assertTrue(linear.supports_skip_head_mid(inputs, head_splits))
        expected = linear._deepgemm_linear(inputs).view(tokens, heads, left + right)
        actual = linear.forward_skip_head_mid(
            inputs, head_splits, output=output
        ).view(tokens, heads, left + gap + right)

        torch.testing.assert_close(actual[..., :left], expected[..., :left])
        torch.testing.assert_close(actual[..., -right:], expected[..., left:])
        self.assertTrue(torch.all(actual[..., left : left + gap] == 11.0))

    def test_skip_head_mid_mla_projection_preserves_rope_and_value(self):
        from rtp_llm.models_py.modules.kimi_k3.mla_prefill import KimiK3MlaPrefillOp

        tokens, heads = 257, 12
        self.K = 512
        self.N = heads * 256
        self.scale_K = self.K // 128
        self.scale_N = self.N // 128
        self.weight = torch.randn(
            self.K, self.N, dtype=torch.bfloat16, device=self.device
        ).to(torch.float8_e4m3fn)
        self.weight_scales = torch.rand(
            self.scale_K, self.scale_N, dtype=torch.float32, device=self.device
        )
        linear = self._create_cuda_fp8_linear()
        latent = torch.randn(tokens, self.K, dtype=torch.bfloat16, device=self.device)
        rope = torch.randn(tokens, 64, dtype=torch.bfloat16, device=self.device)
        op = KimiK3MlaPrefillOp.__new__(KimiK3MlaPrefillOp)
        op.num_heads = heads
        op.qk_nope_head_dim = 128
        op.qk_rope_head_dim = 64
        op.v_head_dim = 128

        key, value = op._project_kv(linear, latent, rope)
        regular = linear._deepgemm_linear(latent).view(tokens, heads, 256)
        expected_key = torch.cat(
            (regular[..., :128], rope[:, None, :].expand(-1, heads, -1)), dim=-1
        )
        torch.testing.assert_close(key, expected_key)
        torch.testing.assert_close(value, regular[..., 128:])


class OnlineLinearAttentionTPTest(online_tests.OnlineLinearAttentionTPTest):
    pass


if __name__ == "__main__":
    unittest.main()
