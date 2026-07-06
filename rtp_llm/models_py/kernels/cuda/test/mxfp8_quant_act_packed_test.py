from unittest import SkipTest, TestCase, main

import torch

MX_BLOCK = 32


def _unpack_packed_scale(scale_packed: torch.Tensor, M: int, K: int) -> torch.Tensor:
    k_groups = K // MX_BLOCK
    k_packed = k_groups // 4
    packed = scale_packed[:, :k_packed].to(torch.int32)
    shifts = torch.tensor([0, 8, 16, 24], device=scale_packed.device, dtype=torch.int32)
    scale_u8 = ((packed[:, :, None] >> shifts) & 0xFF).reshape(M, k_groups)
    return torch.exp2(scale_u8.to(torch.float32) - 127.0)


def _dequant(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    M, K = q.shape
    q_fp32 = q.to(torch.float32).view(M, K // MX_BLOCK, MX_BLOCK)
    return (q_fp32 * scale[:, :, None]).view(M, K)


class Mxfp8QuantActPackedTest(TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is not available")
        try:
            import deep_gemm  # noqa: F401
            import flashinfer  # noqa: F401
        except Exception as e:
            raise SkipTest(f"flashinfer/deep_gemm unavailable: {e}")

    def _run_shape(self, M: int, K: int, dtype: torch.dtype) -> None:
        from rtp_llm.models_py.kernels.cuda.mxfp8_ops import mxfp8_quant_act_packed

        torch.manual_seed(20260706 + M + K)
        x = (torch.randn(M, K, device="cuda", dtype=torch.float32) * 2.0).to(dtype)
        x = x.contiguous()

        q, scale_packed = mxfp8_quant_act_packed(x)
        torch.cuda.synchronize()

        self.assertEqual(q.dtype, torch.float8_e4m3fn)
        self.assertEqual(scale_packed.dtype, torch.int32)
        self.assertEqual(tuple(q.shape), (M, K))
        self.assertEqual(scale_packed.shape[0], M)
        self.assertEqual(scale_packed.shape[1], K // (MX_BLOCK * 4))

        scale = _unpack_packed_scale(scale_packed, M, K)
        dequant = _dequant(q, scale)
        abs_err = (dequant - x.float()).abs()
        rel_err = abs_err / x.float().abs().clamp(min=1e-3)
        self.assertLess(abs_err.max().item(), 1.0)
        self.assertLess(rel_err.mean().item(), 0.08)

    def test_decode_and_prefill_shapes(self) -> None:
        cases = [
            (1, 128, torch.bfloat16),
            (8, 3072, torch.bfloat16),
            (32, 6144, torch.bfloat16),
            (1024, 128, torch.bfloat16),
            (1024, 6144, torch.bfloat16),
        ]
        for case in cases:
            with self.subTest(case=case):
                self._run_shape(*case)

    def test_dense_mlp_large_token_fallback_returns_packed_scale(self) -> None:
        from rtp_llm.models_py.triton_kernels.common.activation import (
            silu_and_mul_per_token_group_fp8_quant_dense_packed_fwd,
        )

        x = torch.randn(1024, 256, device="cuda", dtype=torch.bfloat16).contiguous()
        q, scale = silu_and_mul_per_token_group_fp8_quant_dense_packed_fwd(
            x,
            quant_group_size=MX_BLOCK,
            scale_ue8m0=False,
            round_to_pow2=True,
        )
        torch.cuda.synchronize()

        self.assertEqual(q.dtype, torch.float8_e4m3fn)
        self.assertEqual(scale.dtype, torch.int32)
        self.assertEqual(tuple(q.shape), (1024, 128))
        self.assertEqual(tuple(scale.shape), (1024, 1))


if __name__ == "__main__":
    main()
