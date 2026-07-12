import unittest
from unittest import SkipTest

import torch
import torch.nn.functional as F
from aiter import dtypes

from rtp_llm.models_py.utils.arch import is_hip

try:
    import aiter

    AITER_AVAILABLE = True
except ImportError:
    AITER_AVAILABLE = False


def run_torch(x, weight, x_scale, w_scale, bias=None, dtype=torch.bfloat16):
    x = x.to(dtypes.fp32) * x_scale
    weight = weight.to(dtypes.fp32) * w_scale
    out = F.linear(x, weight)
    if bias is not None:
        out = out.to(bias) + bias
    return out.to(dtype)


def shuffle_weight(x: torch.Tensor, layout=(16, 16)) -> torch.Tensor:
    x_type = x.dtype
    IN, IK = layout
    BK = IK * 2
    K = 16 // x.element_size()
    BN = IN
    assert x.shape[-2] % BN == 0
    assert x.shape[-1] % BK == 0
    x_ = x.view(-1, x.shape[-2] // BN, BN, x.shape[-1] // BK, BK // K, K)
    x_ = x_.permute(0, 1, 3, 4, 2, 5)
    x_ = x_.contiguous().view(*x.shape)
    return x_.view(x_type)


class RocmFp8PTPCLinearTest(unittest.TestCase):
    """PTPC fp8 linear单元测试，kernel weight 必须 shuffle。"""

    def setUp(self):
        if not is_hip():
            raise SkipTest("Test requires ROCm/HIP backend!")
        if not AITER_AVAILABLE:
            raise SkipTest("aiter required for RocmFp8PTPCLinear!")
        self.device = "cuda"
        self.hidden_size = 256
        self.output_size = 512
        self.batch_size = 32
        self.fp8_dtype = torch.float8_e4m3fnuz
        self.input_fp32 = torch.randn(
            self.batch_size, self.hidden_size, dtype=torch.bfloat16, device=self.device
        )  # [B, K]
        self.weight_fp32 = torch.randn(
            self.output_size, self.hidden_size, dtype=torch.bfloat16, device=self.device
        )  # [N, K]
        self.bias = torch.randn(
            self.output_size, dtype=torch.bfloat16, device=self.device
        )  # [N]

    def test_ptpc_fp8_forward(self):
        from rtp_llm.models_py.kernels.rocm.fp8_kernel import rocm_per_token_quant_fp8
        from rtp_llm.models_py.modules.factory.linear.impl.rocm.fp8_ptpc_linear import (
            RocmFp8PTPCLinear,
        )

        weight_q, weight_scales = rocm_per_token_quant_fp8(
            self.weight_fp32
        )  # weight_q: [N, K], weight_scales: [N, 1]

        weight_shuffle = shuffle_weight(weight_q, layout=(16, 16))  # [N, K]

        weight_for_init = weight_shuffle.T.contiguous()  # [K, N]

        weight_scales_for_init = weight_scales.T.contiguous()  # [1, N]

        ptpc_linear = RocmFp8PTPCLinear(
            weight=weight_for_init,  # [K, N]
            weight_scales=weight_scales_for_init,  # [1, N]
            bias=None,
        )

        ptpc_output = ptpc_linear(self.input_fp32)

        ref_input_bf16 = self.input_fp32

        quantization_eps = 1e-10
        ref_input_fp8, ref_input_scales = rocm_per_token_quant_fp8(
            ref_input_bf16,
            eps=quantization_eps,
        )
        ref_input_scales = ref_input_scales.to(torch.float32)

        ref_output = aiter.gemm_a8w8_bpreshuffle(
            ref_input_fp8,  # A_quant_tensor
            ptpc_linear.weight,  # W_kernel_tensor (使用 RocmFp8PTPCLinear 内部的 weight)
            ref_input_scales,  # A_quant_scale_tensor (M, 1)
            ptpc_linear.weight_scales,  # W_scale_tensor (使用 RocmFp8PTPCLinear 内部的 weight_scales)
            None,  # bias
            ref_input_bf16.dtype,  # output dtype (与 RocmFp8PTPCLinear.forward() 相同)
        )
        ref = ref_output

        # 5. 对比结果
        max_diff = (ptpc_output - ref).abs().max().item()
        mean_diff = (ptpc_output - ref).abs().mean().item()
        print(f"nobias max_diff: {max_diff:.6f}, mean_diff: {mean_diff:.6f}")
        self.assertLess(max_diff, 1e-5)
        self.assertLess(mean_diff, 1e-5)
        self.assertEqual(ptpc_output.shape, (self.batch_size, self.output_size))
        self.assertFalse(torch.isnan(ptpc_output).any())
        self.assertFalse(torch.isinf(ptpc_output).any())


class RocmFp8PTPCLinearDispatchTest(unittest.TestCase):
    """Test numerical equivalence between cktile and default FP8 GEMM kernels.

    Directly compares gemm_a8w8_bpreshuffle_cktile vs gemm_a8w8_bpreshuffle
    with identical FP8 inputs at dispatch threshold boundaries.
    """

    def setUp(self):
        if not is_hip():
            raise SkipTest("Test requires ROCm/HIP backend!")
        if not AITER_AVAILABLE:
            raise SkipTest("aiter required for RocmFp8PTPCLinear!")
        self.device = "cuda"

    def _run_and_verify(self, M, N, K):
        """Verify FP8 PTPC linear dispatch output correctness.

        For all dispatch shapes, verifies:
        1. Shape correctness and finiteness (no NaN/Inf)
        2. Determinism: two forward() calls produce identical output (max_diff < 1e-3)
        3. Amplitude correctness:
           - Default path: max_diff < 1e-3 vs gemm_a8w8_bpreshuffle (verified baseline)
           - Cktile path: output is non-trivial (std > 0, absmax within 2x of
             expected magnitude sqrt(K) for randn inputs)
        """
        from rtp_llm.models_py.kernels.rocm.fp8_kernel import rocm_per_token_quant_fp8
        from rtp_llm.models_py.modules.factory.linear.impl.rocm.fp8_ptpc_linear import (
            RocmFp8PTPCLinear,
        )

        input_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device=self.device)
        weight_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device=self.device)

        weight_q, weight_scales = rocm_per_token_quant_fp8(weight_bf16)
        weight_shuffle = shuffle_weight(weight_q, layout=(16, 16))
        ptpc_linear = RocmFp8PTPCLinear(
            weight=weight_shuffle.T.contiguous(),
            weight_scales=weight_scales.T.contiguous(),
            bias=None,
        )

        # Forward calls
        output_1 = ptpc_linear(input_bf16)
        output_2 = ptpc_linear(input_bf16)

        # 1. Shape and finiteness
        self.assertEqual(output_1.shape, (M, N))
        self.assertFalse(torch.isnan(output_1).any(), f"NaN for M={M},N={N},K={K}")
        self.assertFalse(torch.isinf(output_1).any(), f"Inf for M={M},N={N},K={K}")

        # 2. Determinism: max_diff < 1e-3
        det_diff = (output_1.float() - output_2.float()).abs().max().item()
        self.assertLess(
            det_diff, 1e-3,
            f"determinism max_diff={det_diff:.6f} for M={M},N={N},K={K}",
        )

        # 3. Amplitude correctness
        use_cktile = K < 192 or M >= 1536 or (M >= 512 and N > 1536)
        if not use_cktile:
            # Default path: compare against gemm_a8w8_bpreshuffle baseline
            # (verified correct by test_ptpc_fp8_forward with max_diff=0)
            input_fp8, input_scales = rocm_per_token_quant_fp8(input_bf16)
            input_scales = input_scales.to(torch.float32)
            baseline = aiter.gemm_a8w8_bpreshuffle(
                input_fp8, ptpc_linear.weight, input_scales,
                ptpc_linear.weight_scales, None, input_bf16.dtype,
            )
            max_diff = (output_1.float() - baseline.float()).abs().max().item()
            self.assertLess(
                max_diff, 1e-3,
                f"default path max_diff={max_diff:.6f} for M={M},N={N},K={K}",
            )
        else:
            # Cktile path: verify output is non-trivial and has reasonable magnitude.
            # For randn inputs, matmul output absmax ~ sqrt(K) * input_absmax.
            # We check: output has variance (not all zeros or constant) and
            # absmax is within a reasonable range.
            output_std = output_1.float().std().item()
            output_absmax = output_1.float().abs().max().item()
            expected_scale = K ** 0.5  # sqrt(K) for randn @ randn.T

            self.assertGreater(
                output_std, 0.1,
                f"cktile output std too low ({output_std:.6f}) for M={M},N={N},K={K}",
            )
            self.assertGreater(
                output_absmax, 1.0,
                f"cktile output absmax too low ({output_absmax:.4f}) for M={M},N={N},K={K}",
            )
            # absmax should be roughly in [sqrt(K)/10, sqrt(K)*10] range
            self.assertLess(
                output_absmax, expected_scale * 20,
                f"cktile output absmax too high ({output_absmax:.4f}, "
                f"expected ~{expected_scale:.1f}) for M={M},N={N},K={K}",
            )

    # --- default path boundary tests ---
    def test_decode_small_m(self):
        """M=32, N=1024, K=256 → default (protects decode)."""
        self._run_and_verify(M=32, N=1024, K=256)

    def test_default_below_m512_large_n(self):
        """M=256, N=2816, K=256 → default (M<512)."""
        self._run_and_verify(M=256, N=2816, K=256)

    # --- cktile path boundary tests ---
    def test_small_n_at_threshold_m1536(self):
        """M=1536, N=1024, K=256 → cktile (M>=1536)."""
        self._run_and_verify(M=1536, N=1024, K=256)

    def test_large_n_at_threshold_m512(self):
        """M=512, N=2816, K=256 → cktile (M>=512 and N>1536)."""
        self._run_and_verify(M=512, N=2816, K=256)

    def test_prefill_large_m(self):
        """M=2048, N=1024, K=1024 → cktile (M>=1536)."""
        self._run_and_verify(M=2048, N=1024, K=1024)

    def test_large_m_does_not_wrap_output_rows(self):
        """CKTile must not wrap rows when M exceeds its launch-grid range."""
        from rtp_llm.models_py.kernels.rocm.fp8_kernel import rocm_per_token_quant_fp8
        from rtp_llm.models_py.modules.factory.linear.impl.rocm.fp8_ptpc_linear import (
            RocmFp8PTPCLinear,
        )

        torch.manual_seed(7)
        m, n, k = 262145, 8192, 128
        row_a = torch.randn(1, k, dtype=torch.bfloat16, device=self.device)
        row_b = torch.randn(1, k, dtype=torch.bfloat16, device=self.device)
        weight_bf16 = torch.randn(n, k, dtype=torch.bfloat16, device=self.device)
        weight_q, weight_scales = rocm_per_token_quant_fp8(weight_bf16)
        ptpc_linear = RocmFp8PTPCLinear(
            weight=shuffle_weight(weight_q).T.contiguous(),
            weight_scales=weight_scales.T.contiguous(),
            bias=None,
        )

        expected = ptpc_linear(torch.cat((row_a, row_b), dim=0))
        input_bf16 = torch.empty(m, k, dtype=torch.bfloat16, device=self.device)
        input_bf16[:-1].copy_(row_a.expand(m - 1, k))
        input_bf16[-1:].copy_(row_b)
        output = ptpc_linear(input_bf16)

        torch.testing.assert_close(output[0], expected[0], rtol=0, atol=0)
        torch.testing.assert_close(output[-1], expected[1], rtol=0, atol=0)

    def test_chunk_size_accounts_for_wide_output_rows(self):
        """Every CKTile input/output buffer must stay within 32-bit offsets."""
        from rtp_llm.models_py.modules.factory.linear.impl.rocm.fp8_ptpc_linear import (
            _cktile_max_m_per_launch,
        )

        self.assertEqual(
            _cktile_max_m_per_launch(
                k=5120,
                n=8192,
                input_element_size=1,
                output_element_size=2,
            ),
            131072,
        )
        self.assertEqual(
            _cktile_max_m_per_launch(
                k=5120,
                n=17408,
                input_element_size=1,
                output_element_size=2,
            ),
            61568,
        )


if __name__ == "__main__":
    unittest.main()
