"""Compare strided KDA activation quantization with the existing CUDA path."""

import unittest
from unittest.mock import patch

import torch

from rtp_llm.model_loader.per_block_fp8_quant_weight import per_block_cast_to_fp8
from rtp_llm.models_py.kernels.cuda.fp8_kernel.fp8_kernel import (
    _transform_scale_ue8m0,
    sgl_per_token_group_quant_fp8,
)
from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_deepgemm_linear import CudaFp8DeepGEMMLinear
from rtp_llm.models_py.triton_kernels.kimi_kda.fp8_quant import quantize_forget_latent_fp8


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class KdaFp8QuantTest(unittest.TestCase):
    def reference(self, x):
        return sgl_per_token_group_quant_fp8(
            x.contiguous(), 128, eps=1e-4, column_major_scales=True,
            scale_tma_aligned=True, scale_ue8m0=True,
        )

    def compare(self, x):
        actual, scales = quantize_forget_latent_fp8(x)
        expected, expected_scales = self.reference(x)
        torch.testing.assert_close(actual.view(torch.uint8), expected.view(torch.uint8), rtol=0, atol=0)
        torch.testing.assert_close(scales & 255, expected_scales & 255, rtol=0, atol=0)
        self.assertTrue(actual.is_contiguous())
        self.assertEqual(scales.stride(), (1, (x.shape[0] + 3) // 4 * 4))
        self.assertTrue(torch.all((scales >> 8) == 0x7F7F7F).item())

    def test_tp_layouts_and_tail_rows(self):
        torch.manual_seed(7)
        for width in (6368, 12512, 24800, 49376):
            for m in (1, 2, 3, 4, 5, 16, 128):
                with self.subTest(width=width, m=m):
                    fused = torch.randn(m, width, device="cuda", dtype=torch.bfloat16)
                    self.compare(fused[:, width - 224 : width - 96])
        fused = torch.randn(4096, 6368, device="cuda", dtype=torch.bfloat16)
        self.compare(fused[:, 6144:6272])

    def test_zero_tiny_and_scale_boundaries(self):
        values = [0.0, 1e-8, 1e-4]
        for exponent in range(-16, 17):
            # BF16 neighbors on either side of an exact UE8M0 boundary.
            values.extend([446.0 * 2**exponent, 448.0 * 2**exponent, 450.0 * 2**exponent])
        x = torch.zeros(len(values), 640, device="cuda", dtype=torch.bfloat16)[:, 128:256]
        x[:, 0] = torch.tensor(values, device="cuda", dtype=torch.bfloat16)
        x[:, 1] = -x[:, 0]
        self.compare(x)

    def test_large_prefill_v2(self):
        # The existing auto dispatcher selects v2 for matrices above 4M
        # elements; retain that path's exact activation/scale semantics too.
        for m in (65536, 131072):
            x = torch.randn(m, 256, device="cuda", dtype=torch.bfloat16)[:, 128:]
            with patch.dict("os.environ", {"DSV4_FP8_QUANT_KERNEL": "v2"}):
                self.compare(x)

    def test_fp8_gemm_and_graph_replay(self):
        torch.manual_seed(8)
        weight = torch.randn(1536, 128, device="cuda", dtype=torch.bfloat16) * 0.02
        w, scales = per_block_cast_to_fp8(weight, 128, use_ue8m0=True)
        linear = CudaFp8DeepGEMMLinear(w, _transform_scale_ue8m0(scales, mn=1536))
        for m in (1, 4, 16):
            fused = torch.randn(m, 6368, device="cuda", dtype=torch.bfloat16)
            x = fused[:, 6144:6272]
            def forward():
                return linear.forward_quantized(*quantize_forget_latent_fp8(x))
            for _ in range(3):
                forward()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = forward()
            for i in range(12):
                x.fill_(0.03125 * (i + 1))
                graph.replay()
                expected = linear(x.contiguous())
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_streams_have_independent_scales(self):
        streams = [torch.cuda.Stream(), torch.cuda.Stream()]
        inputs = [torch.randn(5, 6368, device="cuda", dtype=torch.bfloat16)[:, 6144:6272] * scale for scale in (0.01, 100.0)]
        torch.cuda.synchronize()
        outputs = []
        for stream, x in zip(streams, inputs):
            with torch.cuda.stream(stream):
                outputs.append(quantize_forget_latent_fp8(x))
        torch.cuda.synchronize()
        self.assertNotEqual(outputs[0][1].data_ptr(), outputs[1][1].data_ptr())
        for x, (q, s) in zip(inputs, outputs):
            rq, rs = self.reference(x)
            torch.testing.assert_close(q.view(torch.uint8), rq.view(torch.uint8), rtol=0, atol=0)
            torch.testing.assert_close(s & 255, rs & 255, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
