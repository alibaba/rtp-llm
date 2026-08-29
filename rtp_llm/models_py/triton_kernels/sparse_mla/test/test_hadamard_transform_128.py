"""Correctness and CUDA Graph tests for the GLM sparse-indexer Hadamard path."""

import math
import unittest

import torch

from rtp_llm.models_py.triton_kernels.sparse_mla.fused_prefill_rope_hadamard import (
    hadamard_transform_128,
)


def _reference_hadamard_128(x: torch.Tensor) -> torch.Tensor:
    """Independent FP32 iterative Sylvester transform."""
    y = x.float().reshape(-1, 128)
    stride = 1
    while stride < 128:
        grouped = y.reshape(-1, 128 // (2 * stride), 2, stride)
        first = grouped[:, :, 0, :]
        second = grouped[:, :, 1, :]
        y = torch.stack((first + second, first - second), dim=2).reshape(-1, 128)
        stride *= 2
    return (y * (1.0 / math.sqrt(128))).to(torch.bfloat16).reshape_as(x)


class TestHadamardTransform128(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available")
        torch.manual_seed(20260828)

    def test_matches_independent_reference(self):
        for shape in ((1, 128), (37, 128), (5, 7, 128)):
            x = torch.randn(shape, dtype=torch.bfloat16, device="cuda").contiguous()
            actual = hadamard_transform_128(x)
            expected = _reference_hadamard_128(x)
            torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_empty_input(self):
        x = torch.empty((0, 128), dtype=torch.bfloat16, device="cuda")
        self.assertEqual(hadamard_transform_128(x).shape, x.shape)

    def test_cuda_graph_replay(self):
        x = torch.randn((16, 128), dtype=torch.bfloat16, device="cuda")
        static_x = torch.empty_like(x)
        hadamard_transform_128(static_x)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_out = hadamard_transform_128(static_x)

        static_x.copy_(x)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            graph_out, _reference_hadamard_128(x), rtol=0.0, atol=0.0
        )


if __name__ == "__main__":
    unittest.main()
