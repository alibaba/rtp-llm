"""Qualify upstream Q/KV projection helpers and native query quantization."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv4._v41_fused_qkv import (
    strided_q_rmsnorm,
    try_project_qr_kv,
)
from rtp_llm.models_py.modules.dsv4._v41_query_quant import try_project_quantized_qkv
from rtp_llm.models_py.modules.dsv4.test import test_v41_query_quant
from rtp_llm.models_py.modules.dsv4.utils import V41MXFP8Linear
from rtp_llm.ops.compute_ops import rtp_llm_ops


def reference_norm(x, weight, eps):
    flat = x.reshape(-1, x.shape[-1]).contiguous()
    output = torch.empty_like(flat)
    rtp_llm_ops.rmsnorm(
        output, flat, weight, eps, torch.cuda.current_stream().cuda_stream
    )
    return output.view(x.shape)


class QuerySM120Test(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4111)

    def make_linear(self, n, k):
        weight = (torch.randn(n, k, device="cuda") * 0.04).to(torch.float8_e4m3fn)
        scales = torch.randint(
            123, 128, (n // 32, k // 32), device="cuda", dtype=torch.uint8
        ).view(torch.float8_e8m0fnu)
        return V41MXFP8Linear(weight, scales)

    @torch.no_grad()
    def test_strided_norm_against_native_cuda(self):
        for rows in (1, 7, 128):
            for width in (128, 1280, 2560):
                with self.subTest(rows=rows, width=width):
                    x = torch.randn(
                        rows, width + 512, device="cuda", dtype=torch.bfloat16
                    )[:, :width]
                    weight = torch.randn(width, device="cuda", dtype=torch.bfloat16)
                    for eps in (1e-6, 1e-20):
                        torch.testing.assert_close(
                            strided_q_rmsnorm(x, weight, eps),
                            reference_norm(x, weight, eps),
                            rtol=0.008,
                            atol=1e-5,
                        )

    @torch.no_grad()
    def test_combined_projection_and_prequantized_input(self):
        q_rank, kv_dim, dim = 1280, 512, 5120
        linear = self.make_linear(q_rank + kv_dim, dim)
        # Independent linears are the unfused upstream execution path.
        scales = linear._sm120_linear.scales.view(torch.float8_e8m0fnu)
        query = V41MXFP8Linear(linear.weight[:q_rank], scales[: q_rank // 32])
        kv = V41MXFP8Linear(linear.weight[q_rank:], scales[q_rank // 32 :])
        norm = torch.randn(q_rank, device="cuda", dtype=torch.bfloat16)
        for shape in ((1, dim), (7, dim), (2, 6, dim)):
            x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
            expected_q = reference_norm(query(x), norm, 1e-20)
            expected_kv = kv(x)
            for quantized in (None, linear._quantize_input(x.reshape(-1, dim))):
                with self.subTest(shape=shape, prequantized=quantized is not None):
                    qr, kv_raw = try_project_qr_kv(
                        linear, x, norm, q_rank, 1e-20, quantized_input=quantized
                    )
                    torch.testing.assert_close(qr, expected_q, rtol=0.008, atol=1e-5)
                    torch.testing.assert_close(
                        kv_raw, expected_kv, rtol=0.008, atol=1e-5
                    )
                    self.assertEqual(kv_raw.stride(-2), q_rank + kv_dim)
            self.assertIsNone(try_project_qr_kv(linear, x.float(), norm, q_rank, 1e-20))

    @torch.no_grad()
    def test_fused_query_dispatch_and_graph_with_real_consumers(self):
        q_rank, kv_dim, dim = 1280, 512, 5120
        attn = SimpleNamespace(
            wq_a_wkv=self.make_linear(q_rank + kv_dim, dim),
            wq_b=self.make_linear(256, q_rank),
            q_norm=torch.randn(q_rank, device="cuda", dtype=torch.bfloat16),
            q_lora_rank=q_rank,
            eps=1e-20,
        )
        x = torch.randn(7, dim, device="cuda", dtype=torch.bfloat16)

        def fused():
            qr, raw_kv, quant = try_project_quantized_qkv(attn, x)
            return qr, raw_kv, attn.wq_b.forward_quantized(*quant)

        with patch.dict(os.environ, DSV4_FP8_QUANT_KERNEL="legacy"):
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    fused()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                result = fused()
            for scale in (0.0, 1e-10, 1.0, 10.0):
                x.normal_().mul_(scale)
                projected = attn.wq_a_wkv(x)
                qr = strided_q_rmsnorm(projected[:, :q_rank], attn.q_norm, attn.eps)
                expected = qr, projected[:, q_rank:], attn.wq_b(qr)
                graph.replay()
                for actual, reference in zip(result, expected):
                    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        for mode in ("v2", "invalid"):
            with patch.dict(os.environ, DSV4_FP8_QUANT_KERNEL=mode):
                self.assertIsNone(try_project_quantized_qkv(attn, x))


def load_tests(loader, tests, pattern):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        raise RuntimeError("Query helper qualification requires a real SM120 GPU")
    tests.addTests(loader.loadTestsFromModule(test_v41_query_quant))
    return tests


if __name__ == "__main__":
    unittest.main()
