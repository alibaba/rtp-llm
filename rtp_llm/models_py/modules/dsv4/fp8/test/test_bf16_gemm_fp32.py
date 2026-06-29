"""UT for BF16 x BF16 -> FP32 compressor GEMM."""

from __future__ import annotations

import os
import unittest

import torch

from rtp_llm.models_py.modules.dsv4.fp8.compressor import _linear_bf16_bf16_fp32
from rtp_llm.ops.compute_ops import rtp_llm_ops


class Bf16GemmFp32Test(unittest.TestCase):

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        if not hasattr(rtp_llm_ops, "cublas_gemm_bf16_bf16_fp32"):
            self.skipTest("cublas_gemm_bf16_bf16_fp32 op is not built")

    def test_op_returns_fp32(self) -> None:
        torch.manual_seed(0)
        x = (torch.randn(7, 13, device="cuda") * 0.1).to(torch.bfloat16)
        w = (torch.randn(5, 13, device="cuda") * 0.1).to(torch.bfloat16)

        out = rtp_llm_ops.cublas_gemm_bf16_bf16_fp32(x, w)
        ref = x.float().matmul(w.float().t())

        self.assertEqual(out.dtype, torch.float32)
        self.assertEqual(tuple(out.shape), (7, 5))
        self.assertTrue(torch.allclose(out, ref, rtol=1e-3, atol=1e-3))

    def test_helper_preserves_leading_dims(self) -> None:
        torch.manual_seed(1)
        x = (torch.randn(2, 3, 17, device="cuda") * 0.1).to(torch.bfloat16)
        w = (torch.randn(11, 17, device="cuda") * 0.1).to(torch.bfloat16)

        out = _linear_bf16_bf16_fp32(x, w)
        ref = x.float().matmul(w.float().t())

        self.assertEqual(out.dtype, torch.float32)
        self.assertEqual(tuple(out.shape), (2, 3, 11))
        self.assertTrue(torch.allclose(out, ref, rtol=1e-3, atol=1e-3))

    def test_aot_out_op_writes_caller_owned_output_when_built(self) -> None:
        if not hasattr(rtp_llm_ops, "cublas_gemm_bf16_bf16_fp32_out"):
            self.skipTest("cublas_gemm_bf16_bf16_fp32_out op is not built")
        torch.manual_seed(2)
        x = (torch.randn(7, 13, device="cuda") * 0.1).to(torch.bfloat16)
        w = (torch.randn(5, 13, device="cuda") * 0.1).to(torch.bfloat16)
        out = torch.empty(7, 5, device="cuda", dtype=torch.float32)

        actual = rtp_llm_ops.cublas_gemm_bf16_bf16_fp32_out(x, w, out)
        ref = x.float().matmul(w.float().t())

        self.assertEqual(actual.data_ptr(), out.data_ptr())
        self.assertEqual(actual.dtype, torch.float32)
        self.assertEqual(tuple(actual.shape), (7, 5))
        self.assertTrue(torch.allclose(actual, ref, rtol=1e-3, atol=1e-3))

    @unittest.skipUnless(
        os.environ.get("DSV4_TEST_BF16_GEMM_OUT_JIT", "0") == "1",
        "set DSV4_TEST_BF16_GEMM_OUT_JIT=1 to compile the experiment JIT op",
    )
    def test_helper_can_write_caller_owned_output(self) -> None:
        torch.manual_seed(2)
        x = (torch.randn(2, 3, 17, device="cuda") * 0.1).to(torch.bfloat16)
        w = (torch.randn(11, 17, device="cuda") * 0.1).to(torch.bfloat16)
        out = torch.empty(2, 3, 11, device="cuda", dtype=torch.float32)

        actual = _linear_bf16_bf16_fp32(x, w, out=out)
        ref = x.float().matmul(w.float().t())

        self.assertEqual(actual.data_ptr(), out.data_ptr())
        self.assertEqual(actual.dtype, torch.float32)
        self.assertEqual(tuple(actual.shape), (2, 3, 11))
        self.assertTrue(torch.allclose(actual, ref, rtol=1e-3, atol=1e-3))


if __name__ == "__main__":
    unittest.main()
