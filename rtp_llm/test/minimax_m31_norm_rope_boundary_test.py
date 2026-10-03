"""GPU regression for CP4/1M norm-RoPE projection address overflow."""

import os
import unittest

import torch

from rtp_llm.models_py.triton_kernels.minimax_m31_gemma_rope import (
    minimax_m31_gemma_norm_rope_,
)


@unittest.skipUnless(
    os.environ.get("RTP_TEST_M31_NORM_BOUNDARY_GPU") == "1",
    "opt-in GPU test allocates about 4.6 GiB",
)
class NormRopeBoundaryTest(unittest.TestCase):
    def test_fused_projection_stride_across_int32_boundary(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        stride = 9856
        boundary = (2**31 - 1) // stride + 1
        # The middle pair straddles the first overflowing row; 250K covers
        # production CP4/1M. Short -> long -> short also checks state reuse.
        for rows in (8192, boundary, boundary + 1, 250000, 7):
            with self.subTest(rows=rows):
                storage = torch.ones(
                    (rows, stride), device="cuda", dtype=torch.bfloat16
                )
                qkv = storage[:, :9216]
                index_q = storage[:, 9216:9728]
                index_k = storage[:, 9728:]
                weights = tuple(
                    torch.zeros(128, device="cuda", dtype=torch.bfloat16)
                    for _ in range(4)
                )
                positions = torch.zeros(rows, device="cuda", dtype=torch.int32)
                cache = torch.cat(
                    (
                        torch.ones((1, 32), device="cuda"),
                        torch.zeros((1, 32), device="cuda"),
                    ),
                    dim=1,
                )

                def run():
                    minimax_m31_gemma_norm_rope_(
                        qkv,
                        index_q,
                        index_k,
                        weights,
                        positions,
                        cache,
                        num_q_heads=64,
                        num_kv_heads=4,
                        num_index_heads=4,
                    )

                run()
                torch.cuda.synchronize()
                self.assertTrue(torch.all(storage == 1).item())
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    run()
                storage.fill_(1)
                graph.replay()
                torch.cuda.synchronize()
                # Includes all normalized rows, V (must remain unchanged),
                # and both index projections sharing the 9856-row allocation.
                self.assertTrue(torch.all(storage == 1).item())
                del graph, storage, qkv, index_q, index_k, weights, positions, cache


if __name__ == "__main__":
    unittest.main()
