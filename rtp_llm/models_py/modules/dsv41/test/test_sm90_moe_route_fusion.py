"""Strict old-expression oracles for SM90 non-GEMM MoE glue."""

import json
import os
import unittest

import torch

from rtp_llm.models_py.modules.dsv41._silu_mul_split_triton import silu_mul_split
from rtp_llm.models_py.modules.dsv41.moe._sm90_route_kernels import (
    reduce_routes,
    routed_swiglu,
)
from rtp_llm.models_py.modules.dsv41.moe.strategies.sm90_humming import (
    group_local_routes,
)


def old_activation(gate_up, weights, slots, clamp_limit, output_dtype):
    gate, up = gate_up.float().chunk(2, dim=-1)
    hidden = silu_mul_split(gate.contiguous(), up.contiguous(), clamp_limit=clamp_limit)
    hidden = hidden * weights.reshape(-1).index_select(0, slots).float().unsqueeze(1)
    return hidden.to(output_dtype)


def old_reduce(down, slots, rows, topk):
    slot_output = torch.zeros(
        (rows * topk, down.shape[1]), device=down.device, dtype=torch.float32
    )
    slot_output.index_copy_(0, slots, down.float())
    return slot_output.view(rows, topk, down.shape[1]).sum(dim=1)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class SM90MoeRouteFusionTest(unittest.TestCase):
    def setUp(self):
        if torch.cuda.get_device_capability() != (9, 0):
            self.skipTest("requires SM90")
        torch.manual_seed(4301)

    def _routes(self, rows, topk=8):
        indices = torch.randint(0, 48, (rows, topk), device="cuda", dtype=torch.int64)
        indices[:, min(6, topk) :] = -1
        # Uneven expert counts, duplicate expert in different slots, empty
        # experts, negative padding and nonlocal slots all remain meaningful.
        indices[:, 0] = 0
        if topk > 1:
            indices[::3, 1] = 0
        if rows > 1:
            indices[1::5].fill_(-1)
        return indices

    def test_weighted_swiglu_matches_original_chain_exactly(self):
        for rows, dim in ((1, 128), (13, 512), (1024, 2048), (1024, 2304)):
            routes = self._routes(rows)
            slots, _ = group_local_routes(routes, 0, 48)
            weights = torch.randn(rows, 8, device="cuda")
            gate_up = (
                torch.randn(slots.numel(), dim * 2, device="cuda") * 12
            ).bfloat16()
            for limit in (0.0, 10.0):
                for output_dtype in (torch.bfloat16, torch.float32):
                    with self.subTest(
                        rows=rows, dim=dim, limit=limit, dtype=output_dtype
                    ):
                        actual = routed_swiglu(
                            gate_up,
                            weights,
                            slots,
                            clamp_limit=limit,
                            output_dtype=output_dtype,
                        )
                        expected = old_activation(
                            gate_up, weights, slots, limit, output_dtype
                        )
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_activation_independent_torch_formula(self):
        # Controlled BF16 values avoid ambiguity from exp approximation while
        # independently checking clamp direction, gate/up order and weights.
        gate = torch.tensor(
            [-16.0, -2.0, -1.0, 0.0, 1.0, 2.0, 16.0], device="cuda"
        ).bfloat16()
        up = torch.tensor(
            [16.0, -16.0, 1.0, 2.0, -1.0, 16.0, -16.0], device="cuda"
        ).bfloat16()
        gate_up = torch.cat((gate, up)).repeat(4, 1)
        weights = torch.tensor([[0.125, -0.5, 2.0, 0.0]], device="cuda")
        slots = torch.tensor([2, 0, 3, 1], device="cuda", dtype=torch.int64)
        for limit in (0.0, 10.0):
            g, u = gate.float(), up.float()
            if limit > 0:
                g, u = g.clamp(max=limit), u.clamp(-limit, limit)
            expected = (
                (torch.nn.functional.silu(g) * u)[None] * weights.flatten()[slots, None]
            ).bfloat16()
            actual = routed_swiglu(
                gate_up, weights, slots, clamp_limit=limit, output_dtype=torch.bfloat16
            )
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_ordered_reduce_dense_decode_and_imbalanced_prefill_exact(self):
        for rows, dim, topk in ((1, 5120, 8), (7, 128, 6), (4096, 5120, 8)):
            indices = self._routes(rows, topk)
            slots, _ = group_local_routes(indices, 0, 48)
            down = torch.randn(slots.numel(), dim, device="cuda").bfloat16()
            expected = old_reduce(down, slots, rows, topk)
            actual = reduce_routes(down, slots, rows, topk)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(
                reduce_routes(down, slots, rows, topk), actual, rtol=0, atol=0
            )

    def test_reduction_tree_cancellation_extremes_and_missing_routes(self):
        rows, dim, topk = 4, 128, 8
        # A different association loses the small terms. Compare with actual
        # Torch sum, not another copy of the candidate's reduction expression.
        pattern = torch.tensor(
            [2.0**100, 1.0, -(2.0**100), 2.0, 3.0, -4.0, 5.0, -6.0], device="cuda"
        ).bfloat16()
        indices = torch.arange(8, device="cuda").expand(rows, -1).clone()
        indices[1, [1, 4, 7]] = -1
        indices[2].fill_(-1)
        indices[3] = indices[3].flip(0)
        slots, _ = group_local_routes(indices, 0, 8)
        down = pattern[slots % 8, None].expand(-1, dim).contiguous()
        expected = old_reduce(down, slots, rows, topk)
        torch.testing.assert_close(
            reduce_routes(down, slots, rows, topk), expected, rtol=0, atol=0
        )
        empty_slots, _ = group_local_routes(indices, 48, 96)
        empty_down = torch.empty((0, dim), device="cuda", dtype=torch.bfloat16)
        torch.testing.assert_close(
            reduce_routes(empty_down, empty_slots, rows, topk),
            old_reduce(empty_down, empty_slots, rows, topk),
            rtol=0,
            atol=0,
        )

    @unittest.skipUnless(
        os.environ.get("DSV41_SM90_ROUTE_BENCH") == "1", "explicit benchmark opt-in"
    )
    def test_real_shape_component_benchmark(self):
        import triton

        rows, dim, inter, topk = 4096, 5120, 2304, 8
        indices = self._routes(rows)
        slots, _ = group_local_routes(indices, 0, 48)
        weights = torch.rand(rows, topk, device="cuda")
        gate_up = torch.randn(
            slots.numel(), inter * 2, device="cuda", dtype=torch.bfloat16
        )
        down = torch.randn(slots.numel(), dim, device="cuda", dtype=torch.bfloat16)
        old_act = lambda: old_activation(gate_up, weights, slots, 10.0, torch.bfloat16)
        new_act = lambda: routed_swiglu(
            gate_up, weights, slots, clamp_limit=10.0, output_dtype=torch.bfloat16
        )
        old_sum = lambda: old_reduce(down, slots, rows, topk)
        new_sum = lambda: reduce_routes(down, slots, rows, topk)
        torch.testing.assert_close(old_act(), new_act(), rtol=0, atol=0)
        torch.testing.assert_close(old_sum(), new_sum(), rtol=0, atol=0)
        results = {
            "recv_rows": rows,
            "valid_routes": slots.numel(),
            "dim": dim,
            "inter": inter,
            "topk": topk,
            "experts": 48,
            "old_activation_ms": triton.testing.do_bench(old_act),
            "fused_activation_ms": triton.testing.do_bench(new_act),
            "old_route_reduce_ms": triton.testing.do_bench(old_sum),
            "inverse_route_reduce_ms": triton.testing.do_bench(new_sum),
            "old_slot_scratch_bytes": rows * topk * dim * 4,
            "inverse_map_scratch_bytes": rows * topk * 4,
        }
        print("SM90_ROUTE_FUSION_BENCH " + json.dumps(results), flush=True)


if __name__ == "__main__":
    unittest.main()
