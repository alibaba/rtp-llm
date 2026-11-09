"""Live expert rows, NaN padding, column-major scales and dynamic GPU counts."""

from unittest import TestCase, main

import torch

from rtp_llm.models_py.kernels.ppu.fp8_quant import silu_mul_fp8_quantize_128
from rtp_llm.models_py.kernels.ppu.ragged_fp8_quant import (
    ragged_silu_mul_fp8_quantize_128,
)


class RaggedFp8QuantTest(TestCase):
    def test_live_values_match_existing_quantizer_with_nan_tail(self):
        torch.manual_seed(20261108)
        for capacity in [1, 17, 128, 1024, 8192]:
            for rows in sorted({0, 1, capacity // 8, capacity}):
                for hidden in [256, 2048]:
                    with self.subTest(capacity=capacity, rows=rows, hidden=hidden):
                        x = torch.full(
                            (capacity, hidden * 2),
                            float("nan"),
                            device="cuda",
                            dtype=torch.bfloat16,
                        )
                        x[:rows].normal_()
                        counts = torch.tensor(
                            [rows // 3, 0, rows - rows // 3, 0],
                            device="cuda",
                            dtype=torch.int32,
                        )
                        q, scales = ragged_silu_mul_fp8_quantize_128(x, counts)
                        expected_q, expected_scales = silu_mul_fp8_quantize_128(
                            x[:rows], gate_first=False
                        )
                        self.assertTrue(
                            torch.equal(
                                q[:rows].view(torch.uint8), expected_q.view(torch.uint8)
                            )
                        )
                        self.assertTrue(torch.equal(scales[:rows], expected_scales))
                        self.assertEqual(scales.stride(0), 1)
                        self.assertEqual(scales.stride(1) % 4, 0)

    def test_graph_replay_uses_changed_counts_and_activation(self):
        torch.manual_seed(7208)
        x = torch.randn(1024, 4096, device="cuda", dtype=torch.bfloat16)
        counts = torch.tensor([16, 0, 15, 1], device="cuda", dtype=torch.int32)
        for _ in range(3):
            ragged_silu_mul_fp8_quantize_128(x, counts)
        graph, stream = torch.cuda.CUDAGraph(), torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.graph(graph, stream=stream):
            q, scales = ragged_silu_mul_fp8_quantize_128(x, counts)
        for rows in [1, 129, 1024, 0, 17]:
            x.normal_()
            x[rows:].fill_(float("nan"))
            counts.copy_(
                torch.tensor(
                    [rows // 2, 0, rows - rows // 2, 0],
                    device="cuda",
                    dtype=torch.int32,
                )
            )
            graph.replay()
            torch.cuda.synchronize()
            expected_q, expected_scales = silu_mul_fp8_quantize_128(
                x[:rows], gate_first=False
            )
            self.assertTrue(
                torch.equal(q[:rows].view(torch.uint8), expected_q.view(torch.uint8))
            )
            self.assertTrue(torch.equal(scales[:rows], expected_scales))


if __name__ == "__main__":
    main()
