import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.model_desc.qwen4_exp_mtp import Qwen4ExpMTPInputProjection
from rtp_llm.models_py.modules.base.common.fp8_linear import Fp8Linear
from rtp_llm.models_py.modules.qwen4_exp.mtp_norm_triton import (
    _mtp_embedding_rms_norm,
    try_mtp_embedding_rms_norm,
)

MODULE = "rtp_llm.models_py.modules.qwen4_exp.mtp_norm_triton"


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
                    M=rows,
                    L=lanes,
                    EPS=1e-6,
                    WRITE_VARIANCE=True,
                    num_warps=4,
                    enable_fp_fusion=False,
                )
                torch.testing.assert_close(observed, variance, rtol=0, atol=0)

    def fixture(self):
        def projection():
            weight = (torch.randn(2560, 2560, device="cuda") * 0.1).to(
                torch.float8_e4m3fn
            )
            return Fp8Linear(weight, torch.tensor([0.119713], device="cuda"))

        return Qwen4ExpMTPInputProjection(
            torch.randn(2560, device="cuda", dtype=torch.bfloat16) * 0.07,
            torch.randn(10240, device="cuda", dtype=torch.bfloat16) * 0.07,
            projection(),
            projection(),
            hidden_size=2560,
            norm_eps=1e-6,
        )

    def test_complete_two_projection_consumer_and_live_graph(self):
        for rows in (1, 4, 32, 4096):
            model = self.fixture()
            embedding = torch.randn(rows, 2560, device="cuda", dtype=torch.bfloat16)
            hidden = torch.randn(rows, 10240, device="cuda", dtype=torch.bfloat16)
            for _ in range(3):
                model(embedding, hidden)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = model(embedding, hidden)
            for gain in (0.01, 0.2, 2.0):
                embedding.normal_()
                hidden.normal_()
                model.embedding_gamma.normal_(0, gain)
                graph.replay()
                with patch(MODULE + ".try_mtp_embedding_rms_norm", return_value=None):
                    expected = model(embedding, hidden)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            graph.reset()

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
