import unittest

import torch
import triton.language as tl

from rtp_llm.models_py.modules.qwen4_exp.moe.moe_align_triton import (
    moe_align_decode,
    sm120_decode_gemm_configs,
)
from rtp_llm.models_py.triton_kernels.moe.fused_moe_kernel import (
    get_default_config,
    invoke_fused_moe_kernel,
    moe_align_block_size_torch,
    sum_moe_routes,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class MoeAlignTritonTest(unittest.TestCase):
    def test_sm120_tuned_gemms_match_with_live_graph_routing(self):
        if torch.cuda.get_device_capability() != (12, 0):
            self.skipTest("SM120 tiles require SM120")
        torch.manual_seed(432)
        for rows in (3, 7, 17, 31):
            ids = torch.randint(512, (rows, 10), device="cuda", dtype=torch.int32)
            weights = torch.rand(rows * 10, device="cuda") * 0.1
            up = get_default_config(rows, 512, 160, 2560, 10)
            down = get_default_config(rows, 512, 2560, 80, 10)
            tuned_up, tuned_down = sm120_decode_gemm_configs(rows, up, down)
            for name, n, k, topk, old_config, new_config in (
                ("up", 160, 2560, 10, up, tuned_up),
                ("down", 2560, 80, 1, down, tuned_down),
            ):
                with self.subTest(rows=rows, gemm=name):
                    a = (
                        torch.randn(
                            rows if topk == 10 else rows * 10,
                            k,
                            device="cuda",
                            dtype=torch.bfloat16,
                        )
                        * 0.125
                    )
                    b = (
                        torch.randn(512, n, k, device="cuda", dtype=torch.bfloat16)
                        * 0.05
                    )
                    actual = torch.empty(
                        rows * 10, n, device="cuda", dtype=torch.bfloat16
                    )
                    expected = torch.empty_like(actual)

                    def run(config, out):
                        layout = moe_align_decode(ids, 16, 512)
                        invoke_fused_moe_kernel(
                            a,
                            b,
                            out,
                            weights,
                            ids.flatten(),
                            *layout,
                            name == "down",
                            topk,
                            config,
                            tl.bfloat16,
                        )

                    run(old_config, expected)
                    for _ in range(3):
                        run(new_config, actual)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        run(new_config, actual)
                    for spread in (1, 512, 3):
                        ids.copy_(
                            torch.randint(
                                spread, ids.shape, device="cuda", dtype=torch.int32
                            )
                        )
                        a.copy_(torch.randn_like(a) * 0.125)
                        graph.replay()
                        run(old_config, expected)
                        torch.testing.assert_close(
                            actual, expected, atol=0.0078125, rtol=0.008
                        )

    def test_nonlocal_routes_write_zero_during_graph_replay(self):
        rows, experts, top_k = 4, 3, 2
        n, k = 64, 64
        ids = torch.arange(rows * top_k, device="cuda", dtype=torch.int32).view(
            rows, top_k
        ) % (experts + 1)
        a = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(experts, n, k, device="cuda", dtype=torch.bfloat16)
        weights = torch.rand(rows * top_k, device="cuda")
        actual = torch.empty(rows * top_k, n, device="cuda", dtype=torch.bfloat16)
        expected = torch.empty_like(actual)
        config = get_default_config(rows, experts, n, k, top_k)

        def run(route_ids, route_weights, out, aligned_experts):
            layout = moe_align_block_size_torch(
                route_ids, config["BLOCK_SIZE_M"], aligned_experts
            )
            invoke_fused_moe_kernel(
                a,
                b,
                out,
                route_weights,
                route_ids.flatten(),
                *layout,
                True,
                top_k,
                config,
                tl.bfloat16,
                skip_remote_experts=aligned_experts > experts,
            )

        run(ids, weights, actual, experts + 1)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run(ids, weights, actual, experts + 1)
        for shift in range(2):
            ids.add_(shift).remainder_(experts + 1)
            graph.replay()
            local = ids != experts
            run(ids.clamp(max=experts - 1), weights * local.flatten(), expected, experts)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            self.assertEqual(torch.count_nonzero(actual[~local.flatten()]).item(), 0)

    def compare(self, ids, block, experts):
        actual = moe_align_decode(ids, block, experts)
        expected = moe_align_block_size_torch(ids, block, experts)
        for a, e in zip(actual, expected):
            torch.testing.assert_close(a, e, atol=0, rtol=0)

    def test_stable_layout_random_skewed_and_duplicate_routes(self):
        torch.manual_seed(42)
        for batch in (1, 3, 8, 17, 32):
            for block in (8, 16, 32, 64):
                for experts in (16, 512):
                    with self.subTest(batch=batch, block=block, experts=experts):
                        ids = torch.randint(
                            experts, (batch, 10), device="cuda", dtype=torch.int32
                        )
                        self.compare(ids, block, experts)
                        self.compare(torch.zeros_like(ids), block, experts)
                        self.compare(ids.remainder(3), block, experts)

    def test_cuda_graph_refreshes_routing_and_padding_in_both_directions(self):
        for batch, experts in ((1, 512), (8, 512), (32, 512), (32, 513), (64, 257)):
            ids = torch.randint(experts, (batch, 10), device="cuda", dtype=torch.int32)
            for _ in range(3):
                moe_align_decode(ids, 16, experts)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = moe_align_decode(ids, 16, experts)
            for first, last in ((0, 1), (0, experts), (experts - 1, experts)):
                ids.random_(first, last)
                graph.replay()
                expected = moe_align_block_size_torch(ids, 16, experts)
                for a, e in zip(actual, expected):
                    torch.testing.assert_close(a, e, atol=0, rtol=0)

    def test_rejects_unbounded_geometry(self):
        for batch, experts in ((65, 512), (8, 514)):
            ids = torch.zeros(batch, 10, device="cuda", dtype=torch.int32)
            with self.assertRaises(ValueError):
                moe_align_decode(ids, 16, experts)

    def test_prefill_route_sum_matches_torch(self):
        if torch.cuda.get_device_capability() != (9, 0):
            self.skipTest("SM90 route reduction requires SM90")
        for rows in (513, 8192):
            routed = torch.randn((rows, 10, 2560), device="cuda", dtype=torch.bfloat16)
            torch.testing.assert_close(
                sum_moe_routes(routed), routed.sum(dim=1), atol=0, rtol=0
            )


if __name__ == "__main__":
    unittest.main()
