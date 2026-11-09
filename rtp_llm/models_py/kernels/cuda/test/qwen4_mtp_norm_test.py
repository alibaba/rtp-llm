import unittest

import torch

from rtp_llm.models_py.kernels.cuda.qwen4_mtp_norm import (
    _mtp_embedding_rms_norm,
    try_mtp_embedding_rms_norm,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class MtpEmbeddingNormTest(unittest.TestCase):
    def setUp(self):
        if torch.cuda.get_device_capability() != (12, 0):
            self.skipTest("SM120 pinned Torch reduction")
        torch.manual_seed(112201)

    def test_fp32_variance_and_bf16_bytes_match_native_mean(self):
        for rows in (
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            12,
            16,
            20,
            24,
            28,
            32,
            4093,
            4096,
            12288,
            32768,
        ):
            for dtype in (torch.bfloat16, torch.float32):
                x = torch.randn(rows, 2560, device="cuda", dtype=torch.bfloat16)
                gamma = torch.randn(2560, device="cuda", dtype=dtype) * 0.2
                fp = x.float()
                variance = fp.square().mean(-1)
                expected = (
                    (fp * torch.rsqrt(variance[:, None] + 1e-6)) * (1 + gamma.float())
                ).bfloat16()
                actual = try_mtp_embedding_rms_norm(x, gamma, 1e-6)
                torch.testing.assert_close(
                    actual.view(torch.uint8), expected.view(torch.uint8), rtol=0, atol=0
                )
                observed = torch.empty(rows, device="cuda")
                lanes = max(32, 512 // (1 << (rows.bit_length() - 1)))
                _mtp_embedding_rms_norm[(rows,)](
                    x,
                    gamma,
                    actual,
                    observed,
                    L=lanes,
                    EPS=1e-6,
                    WRITE_VARIANCE=True,
                    num_warps=4,
                    enable_fp_fusion=False,
                )
                torch.testing.assert_close(observed, variance, rtol=0, atol=0)

    def test_zero_nonfinite_and_fp32_raw_gain(self):
        for kind in ("zero", "wide", "nonfinite", "zero_gain"):
            x = torch.randn(4, 2560, device="cuda", dtype=torch.bfloat16)
            gamma = torch.randn(2560, device="cuda") * 0.07
            if kind == "zero":
                x.zero_()
            if kind == "wide":
                x.mul_(1000)
            if kind == "nonfinite":
                x[0, 0] = float("inf")
                x[1, 1] = float("nan")
            if kind == "zero_gain":
                gamma.fill_(-1.0)
            fp = x.float()
            expected = (
                (fp * torch.rsqrt(fp.square().mean(-1, keepdim=True) + 1e-6))
                * (1 + gamma)
            ).bfloat16()
            torch.testing.assert_close(
                try_mtp_embedding_rms_norm(x, gamma, 1e-6),
                expected,
                rtol=0,
                atol=0,
                equal_nan=True,
            )

    def test_unqualified_dtype_stride_grad_and_epsilon_decline(self):
        x = torch.empty(4, 2560, device="cuda", dtype=torch.bfloat16)
        gamma = torch.empty(2560, device="cuda", dtype=torch.bfloat16)
        self.assertIsNone(try_mtp_embedding_rms_norm(x.float(), gamma, 1e-6))
        self.assertIsNone(try_mtp_embedding_rms_norm(x[:, ::2], gamma, 1e-6))
        self.assertIsNone(try_mtp_embedding_rms_norm(x.requires_grad_(), gamma, 1e-6))
        self.assertIsNone(try_mtp_embedding_rms_norm(x.detach(), gamma, float("nan")))


if __name__ == "__main__":
    unittest.main()
