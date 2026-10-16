"""Verify resumed projections and CUDA graph replay on SM100."""

import os
import tempfile
import unittest
from unittest.mock import patch

import torch

# DeepGEMM's compiler changes working directory. A relative default cache path
# makes nvcc lose its generated source inside a Bazel test sandbox.
os.environ.setdefault(
    "DG_JIT_CACHE_DIR",
    os.path.join(
        os.environ.get("TEST_TMPDIR", tempfile.gettempdir()), "bf16_deepgemm_jit"
    ),
)
os.makedirs(os.environ["DG_JIT_CACHE_DIR"], exist_ok=True)

from rtp_llm.models_py.modules.factory.linear.impl.cuda.f16_linear import CudaF16Linear


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10,
    "requires SM100",
)
class Bf16DeepGemmLinearTest(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"RTP_BF16_LINEAR_BACKEND": "deepgemm"})
        self.env.start()
        self.addCleanup(self.env.stop)
        torch.manual_seed(20260921)
        torch.backends.cuda.matmul.allow_tf32 = False

    def test_resume_rows(self):
        for n, k in [(16, 4096), (128, 4096), (2048, 128), (4096, 2048), (6144, 4096)]:
            with self.subTest(n=n, k=k):
                weight = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
                layer = CudaF16Linear(weight)
                self.assertIsNotNone(layer._bf16_gemm)
                self.assertTrue(layer.weight.T.is_contiguous())
                x = torch.randn(3389, k, device="cuda", dtype=torch.bfloat16)
                full = layer(x)
                for m in [1, 13, 61, 129]:
                    suffix = layer(x[-m:])
                    self.assertTrue(torch.equal(full[-m:], suffix), (m, n, k))
                    reference = torch.nn.functional.linear(
                        x[-m:].float(), weight.T.float()
                    )
                    error = (suffix.float() - reference).norm() / reference.norm()
                    self.assertLess(error.item(), 0.005)
                self.assertEqual(tuple(layer(x[:0]).shape), (0, n))
                self.assertTrue(torch.equal(layer(x[-13:].unsqueeze(0))[0], full[-13:]))

    def test_fallback(self):
        for dtype, n, k, use_bias in [
            (torch.float16, 16, 128, False),
            (torch.bfloat16, 17, 127, False),
            (torch.bfloat16, 16, 128, True),
        ]:
            with self.subTest(dtype=dtype, n=n, k=k, bias=use_bias):
                weight = torch.randn(k, n, device="cuda", dtype=dtype)
                bias = torch.randn(n, device="cuda", dtype=dtype) if use_bias else None
                layer = CudaF16Linear(weight, bias=bias)
                self.assertIsNone(layer._bf16_gemm)
                x = torch.randn(13, k, device="cuda", dtype=dtype)
                self.assertTrue(
                    torch.equal(layer(x), torch.nn.functional.linear(x, weight.T, bias))
                )

    def test_strided_matrix_and_batched_input(self):
        layer = CudaF16Linear(
            torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
        )
        x = torch.randn(128, 26, device="cuda", dtype=torch.bfloat16).T[::2]
        self.assertFalse(x.is_contiguous())
        expected = layer(x.contiguous())
        self.assertTrue(torch.equal(layer(x), expected))
        self.assertTrue(torch.equal(layer(x.unsqueeze(0))[0], expected))

    def test_packed_kda_projection(self):
        """Check packed NN against the four existing NT projections exactly."""
        import json
        from pathlib import Path

        import deep_gemm

        widths = (6144, 16, 128, 128)
        layers = [
            CudaF16Linear(torch.randn(4096, n, device="cuda", dtype=torch.bfloat16))
            for n in widths
        ]
        padded_width = (sum(widths) + 127) // 128 * 128
        weight = torch.zeros(4096, padded_width, device="cuda", dtype=torch.bfloat16)
        offset = 0
        for layer, width in zip(layers, widths):
            weight[:, offset : offset + width].copy_(layer.weight.T)
            offset += width

        rows = []
        full_input = torch.randn(4093, 4096, device="cuda", dtype=torch.bfloat16)
        full_output = torch.empty(
            4093, padded_width, device="cuda", dtype=torch.bfloat16
        )
        deep_gemm.bf16_gemm_nn(full_input, weight, full_output)
        for m in (1, 13, 61, 129, 509, 4093):
            x = full_input[-m:]
            packed = torch.empty(m, padded_width, device="cuda", dtype=torch.bfloat16)
            separate = [
                torch.empty(m, n, device="cuda", dtype=torch.bfloat16) for n in widths
            ]

            def baseline():
                for layer, output in zip(layers, separate):
                    layer._bf16_gemm(x, layer._bf16_weight, output)

            def candidate():
                deep_gemm.bf16_gemm_nn(x, weight, packed)

            baseline()
            candidate()
            expected = torch.cat(separate, dim=1)
            actual = packed[:, : sum(widths)]
            row = {
                "tokens": m,
                "mismatches": int((expected != actual).sum().item()),
                "max_abs": float(
                    (expected.float() - actual.float()).abs().max().item()
                ),
                "suffix_mismatches": int((packed != full_output[-m:]).sum().item()),
            }
            for name, fn in (("separate", baseline), ("packed", candidate)):
                for _ in range(3):
                    fn()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                    enable_timing=True
                )
                start.record()
                for _ in range(50):
                    fn()
                end.record()
                end.synchronize()
                row[name + "_us"] = start.elapsed_time(end) * 1000 / 50
            rows.append(row)
        output_dir = Path(
            os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", tempfile.gettempdir())
        )
        (output_dir / "packed_projection.json").write_text(json.dumps(rows, indent=2))
        print(json.dumps(rows, indent=2), flush=True)
        self.assertTrue(
            all(
                row["mismatches"] == 0 and row["suffix_mismatches"] == 0 for row in rows
            ),
            rows,
        )

    def test_cuda_graph(self):
        layer = CudaF16Linear(
            torch.randn(4096, 128, device="cuda", dtype=torch.bfloat16)
        )
        x = torch.randn(13, 4096, device="cuda", dtype=torch.bfloat16)
        for _ in range(3):
            layer(x)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = layer(x)
        for _ in range(3):
            x.normal_()
            expected = layer(x)
            graph.replay()
            self.assertTrue(torch.equal(output, expected))


if __name__ == "__main__":
    unittest.main()
