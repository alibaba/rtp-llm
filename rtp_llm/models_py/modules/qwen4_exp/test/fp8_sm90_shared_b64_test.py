"""Shared FFN B64 fusion must preserve dynamic scale, rounding and consumers."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import flashinfer
import torch
from torch import nn

from rtp_llm.models_py.kernels.cuda.fp8_kernel import scaled_fp8_per_tensor_quant
from rtp_llm.models_py.modules.base import FusedSiluAndMul
from rtp_llm.models_py.modules.base.common.fp8_linear import Fp8Linear
from rtp_llm.models_py.modules.hybrid.dense_mlp import DenseMLP
from rtp_llm.models_py.triton_kernels.common import fp8_small_quant as quant
from rtp_llm.ops import ActivationType


def make_mlp(bias=False):
    mlp = DenseMLP.__new__(DenseMLP)
    nn.Module.__init__(mlp)
    mlp.activation_type = ActivationType.Swiglu
    mlp.is_gated = True
    mlp.act_fn = FusedSiluAndMul()
    mlp.parallelism_config = SimpleNamespace(get_ffn_tp_size=lambda: 2)
    mlp.up_proj = Fp8Linear(
        (torch.randn(640, 2560, device="cuda") * 0.1).to(torch.float8_e4m3fn),
        torch.tensor([0.125], device="cuda"),
        role="PDFUSION",
    )
    mlp.down_proj = Fp8Linear(
        (torch.randn(2560, 320, device="cuda") * 0.1).to(torch.float8_e4m3fn),
        torch.tensor([0.03125], device="cuda"),
        torch.randn(2560, device="cuda", dtype=torch.bfloat16) if bias else None,
        role="PDFUSION",
    )
    return mlp


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class Fp8SM90SharedB64Test(unittest.TestCase):
    def setUp(self):
        if torch.cuda.get_device_capability() != (9, 0):
            self.skipTest("SM90 required")
        torch.manual_seed(20261116)

    def exact(self, a, b):
        self.assertTrue(
            torch.equal(
                a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
            )
        )

    def test_native_bytes_and_all_bf16_patterns(self):
        x = torch.randn(64, 640, device="cuda", dtype=torch.bfloat16)
        with patch.dict(os.environ, {"RTP_LLM_CUDA_GRAPH_WARMUP_FORWARD": "1"}):
            self.assertIsNotNone(quant.try_silu_mul_fp8_quant(x))
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = quant.try_silu_mul_fp8_quant(x)
        self.assertIsNotNone(actual)
        addresses = [t.data_ptr() for t in actual]

        def compare():
            graph.replay()
            expected = scaled_fp8_per_tensor_quant(
                flashinfer.activation.silu_and_mul(x)
            )
            for a, b in zip(actual, expected):
                self.exact(a, b)
            self.assertEqual(addresses, [t.data_ptr() for t in actual])

        for magnitude in (0.0, 1e-38, 1e-12, 0.01, 1.0, 100.0, 1e10):
            x.normal_().mul_(magnitude)
            compare()
        for value in (float("nan"), float("inf"), -float("inf"), -0.0):
            x.fill_(value)
            compare()
        bits = (
            torch.arange(65536, device="cuda", dtype=torch.int32)
            .to(torch.int16)
            .view(torch.bfloat16)
        )
        for half in (0, 1):
            for start in range(0, bits.numel(), 64 * 320):
                values = torch.zeros(64 * 320, device="cuda", dtype=torch.bfloat16)
                part = bits[start : start + values.numel()]
                values[: part.numel()].copy_(part)
                x.fill_(0.75)
                x[:, half * 320 : (half + 1) * 320].copy_(values.reshape(64, 320))
                compare()
        graph.reset()

    def test_full_mlp_replay_refresh_and_bias(self):
        for bias in (False, True):
            mlp = make_mlp(bias)
            x = torch.randn(64, 2560, device="cuda", dtype=torch.bfloat16)
            with patch.dict(os.environ, {"RTP_LLM_CUDA_GRAPH_WARMUP_FORWARD": "1"}):
                for _ in range(3):
                    mlp(x, skip_allreduce=True)
            torch.cuda.synchronize()
            baseline = torch.cuda.CUDAGraph()
            candidate = torch.cuda.CUDAGraph()
            with patch.object(Fp8Linear, "forward_silu_and_mul", return_value=None):
                with torch.cuda.graph(baseline):
                    a = mlp(x, skip_allreduce=True)
            with torch.cuda.graph(candidate):
                b = mlp(x, skip_allreduce=True)
            for mag in (0.0, 0.01, 1.0, 10.0):
                x.normal_().mul_(mag)
                baseline.replay()
                candidate.replay()
                torch.cuda.synchronize()
                self.exact(a, b)
            baseline.reset()
            candidate.reset()

    def test_eager_shape_arch_and_role_guards(self):
        x = torch.randn(64, 640, device="cuda", dtype=torch.bfloat16)
        mlp = make_mlp()
        self.assertIsNone(quant.try_silu_mul_fp8_quant(x))
        self.assertIsNone(mlp.down_proj.forward_silu_and_mul(x))
        with patch.dict(os.environ, {"RTP_LLM_CUDA_GRAPH_WARMUP_FORWARD": "1"}):
            self.assertIsNotNone(mlp.down_proj.forward_silu_and_mul(x))
        with patch.object(torch.cuda, "is_current_stream_capturing", return_value=True):
            for cap in ((8, 0), (10, 0), (12, 0)):
                with patch.object(
                    torch.cuda, "get_device_capability", return_value=cap
                ):
                    self.assertIsNone(quant.try_silu_mul_fp8_quant(x))
                    self.assertIsNone(mlp.down_proj.forward_silu_and_mul(x))
            for rows in (1, 4, 32, 64, 96, 128, 256):
                value = torch.randn(rows, 640, device="cuda", dtype=torch.bfloat16)
                self.assertIsNotNone(quant.try_silu_mul_fp8_quant(value))
                self.assertIsNotNone(mlp.down_proj.forward_silu_and_mul(value))
            for rows in (0, 257, 512):
                value = torch.randn(rows, 640, device="cuda", dtype=torch.bfloat16)
                self.assertIsNone(quant.try_silu_mul_fp8_quant(value))
                self.assertIsNone(mlp.down_proj.forward_silu_and_mul(value))
            for value in (x.float(), x.t(), x[:, ::2]):
                self.assertIsNone(quant.try_silu_mul_fp8_quant(value))
            mlp.down_proj.role = "PREFILL"
            self.assertIsNone(mlp.down_proj.forward_silu_and_mul(x))
            mlp.down_proj.role = "PDFUSION"
            mlp.down_proj.weight = mlp.down_proj.weight[:512]
            self.assertIsNone(mlp.down_proj.forward_silu_and_mul(x))

    def test_output_storage_validation(self):
        x = torch.randn(64, 640, device="cuda", dtype=torch.bfloat16)
        outputs = [
            torch.ones(64, 320, device="cuda", dtype=torch.bfloat16),
            torch.ones(65, 320, device="cuda", dtype=torch.float8_e4m3fn),
            x.view(torch.float8_e4m3fn).flatten()[:20480].reshape(64, 320),
        ]
        with patch.object(torch.cuda, "is_current_stream_capturing", return_value=True):
            for out in outputs:
                before = out.view(torch.uint8).clone()
                self.assertIsNone(quant.try_silu_mul_fp8_quant(x, output=out))
                self.exact(out, before.view(out.dtype))


if __name__ == "__main__":
    unittest.main()
