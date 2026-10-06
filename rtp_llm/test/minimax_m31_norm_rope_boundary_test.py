"""GPU regression for CP4/1M norm-RoPE projection address overflow."""

import unittest

import torch

from rtp_llm.models_py.triton_kernels.minimax_m31_gemma_rope import (
    _gemma_norm_rope,
    minimax_m31_gemma_norm_rope_,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class NormRopeBoundaryTest(unittest.TestCase):
    def test_fused_projection_stride_across_int32_boundary(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        stride = 9856
        boundary = (2**31 - 1) // stride + 1
        # The middle pair straddles the first overflowing row; 250K covers
        # CP4/1,000,000 and CP4/1,048,576. Short -> long -> short checks reuse.
        for rows in (
            8192,
            32767,
            32768,
            32769,
            boundary,
            boundary + 1,
            250000,
            262144,
            7,
        ):
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

    def test_grouped_tail_matches_single_row_and_changed_graph_inputs(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        rows, stride = 32769, 9856
        generator = torch.Generator(device="cuda").manual_seed(20261004)
        source = torch.randn(rows, stride, device="cuda", generator=generator).to(
            torch.bfloat16
        )
        expected, actual = source.clone(), source.clone()
        weights = tuple(
            torch.randn(128, device="cuda", generator=generator).to(torch.bfloat16)
            for _ in range(4)
        )
        positions = torch.arange(rows, device="cuda", dtype=torch.int64) % 1024
        cache = torch.randn(1024, 64, device="cuda", generator=generator)

        def reference():
            _gemma_norm_rope[(rows, 73)](
                expected[:, :9216],
                expected[:, 9216:9728],
                expected[:, 9728:],
                *weights,
                positions,
                cache,
                stride,
                stride,
                stride,
                positions.stride(0),
                cache.stride(0),
                64,
                4,
                4,
                1e-6,
                rows,
                1,
                num_warps=4,
                enable_fp_fusion=False,
            )

        def grouped():
            minimax_m31_gemma_norm_rope_(
                actual[:, :9216],
                actual[:, 9216:9728],
                actual[:, 9728:],
                weights,
                positions,
                cache,
                num_q_heads=64,
                num_kv_heads=4,
                num_index_heads=4,
            )

        grouped()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            grouped()
        for replay in range(3):
            source.normal_(generator=generator)
            positions.add_(17).remainder_(1024)
            expected.copy_(source)
            actual.copy_(source)
            reference()
            graph.replay()
            torch.cuda.synchronize()
            self.assertTrue(torch.equal(actual, expected), f"replay={replay}")
            self.assertTrue(torch.equal(actual[:, 8704:9216], source[:, 8704:9216]))


if __name__ == "__main__":
    unittest.main()
