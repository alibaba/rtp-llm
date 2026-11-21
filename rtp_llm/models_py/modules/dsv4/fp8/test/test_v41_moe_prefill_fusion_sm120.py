"""Exact byte and ordered-sum qualification for maintained prefill paths."""

import os
import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv4._v41_moe_combine_triton import combine_ordered_rows
from rtp_llm.models_py.modules.dsv4.fp8.test import test_v41_routed_hardware
from rtp_llm.models_py.modules.dsv4.v41_dispatch import V41DispatchLayout, pack_dispatch
from rtp_llm.models_py.modules.dsv4.v41_expert import quantize_swiglu
from rtp_llm.models_py.modules.dsv4.v41_moe import V41LocalMoE
from rtp_llm.models_py.modules.dsv4.v41_mxfp8 import quantize_group32
from rtp_llm.models_py.modules.dsv4.v41_return import V41ReturnLayout, pack_return
from rtp_llm.models_py.modules.dsv4.v41_routed import V41RoutedExperts


class V41MoEPrefillFusionTest(unittest.TestCase):
    def setUp(self):
        self.assertTrue(torch.cuda.is_available())
        self.assertEqual(torch.cuda.get_device_capability(), (12, 0))
        torch.manual_seed(41723)

    def test_route_builder_empty_experts_and_padding(self):
        from rtp_llm.models_py.modules.dsv4._v41_route_builder_triton import (
            route_builder,
        )

        device = torch.device("cuda")
        source = torch.arange(256, device=device).to(torch.uint8).view(2, 128)
        scales = torch.arange(8, device=device, dtype=torch.uint8).view(2, 4)
        weights = torch.tensor([[0.125, 0.25], [0.5, 1.0]], device=device)
        counts = torch.tensor([0, 2, 0, 1], device=device)
        starts = torch.tensor([0, 0, 2, 2], device=device)
        base = torch.tensor([0, 0, 128, 128], device=device)
        order = torch.tensor([3, 0, 2], device=device)
        route_map = torch.full((4,), -1, device=device)
        actual = route_builder(
            source,
            scales,
            weights,
            order,
            counts,
            starts,
            base,
            256,
            3,
            4,
            2,
            route_map,
        )
        dest = torch.tensor([0, 1, 128], device=device)
        expected = [torch.zeros_like(t) for t in actual]
        expected[0][dest] = source[order // 2]
        expected[1][dest] = scales[order // 2]
        expected[2][dest] = weights.flatten()[order]
        expected[3][dest] = True
        for result, reference in zip(actual, expected):
            self.assertTrue(
                torch.equal(result.view(torch.uint8), reference.view(torch.uint8))
            )
        self.assertEqual(route_map.tolist(), [1, -1, 128, 0])

    def test_whole_local_moe_matches_same_shape_control(self):
        tensors = test_v41_routed_hardware.V41RoutedHardwareTest.weights(96)
        operators = []
        for enabled in ("0", "1"):
            with patch.dict(
                os.environ,
                {
                    "DSV41_MOE_LOCAL_FUSE": enabled,
                    "DSV41_SWIGLU_FUSE": enabled,
                    "DSV41_MOE_MASK_SKIP": enabled,
                },
            ):
                operators.append(
                    V41LocalMoE(V41RoutedExperts(tensors), expert_start_id=96)
                )
        layout = V41DispatchLayout(5120, 6, 384)
        for rows in (129, 1024, 4096):
            x = torch.randn(rows, 5120, device="cuda", dtype=torch.bfloat16)
            weights = torch.rand(rows, 6, device="cuda") * 1.5
            ids = torch.arange(rows * 6, device="cuda").view(rows, 6) % 384
            weights[::13] = 0
            ids[::13] = -1
            x[::13] = float("nan")
            payload = pack_dispatch(
                x,
                weights,
                ids,
                torch.ones(rows, device="cuda", dtype=torch.bool),
                layout,
            )
            expected = operators[0].forward_received(payload, layout)
            actual = operators[1].forward_received(payload, layout)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_contiguous_activation_exact_with_padding_strides_and_clamps(self):
        for rows in (0, 1, 3, 31, 127, 128, 129, 257, 1021, 1024, 4093, 4096, 12288):
            merged = (torch.randn(rows, 4608, device="cuda") * 8).bfloat16()
            gate, up = merged.chunk(2, dim=-1)
            valid = torch.arange(rows, device="cuda") % 7 != 0
            route = torch.rand(rows, device="cuda") * 1.5
            route[::11] = 0
            merged[~valid] = float("nan")
            for clamp in (0.0, 10.0):
                expected = quantize_swiglu(
                    gate, up, valid, route_weights=route, clamp_limit=clamp
                )
                actual = quantize_swiglu(
                    gate,
                    up,
                    valid,
                    route_weights=route,
                    clamp_limit=clamp,
                    allow_contiguous=True,
                )
                torch.testing.assert_close(
                    actual[0].view(torch.uint8),
                    expected[0].view(torch.uint8),
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)

    def test_prefill_combine_preserves_order_and_strided_maps(self):
        for rows in (0, 1, 129, 1021, 4096):
            down = torch.randn(
                max(1, rows * 6), 5120, device="cuda", dtype=torch.bfloat16
            )
            if rows:
                down[:3] = torch.tensor(
                    [32768, -32768, 1], device="cuda", dtype=down.dtype
                )[:, None]
            storage = torch.full((rows, 12), -1, device="cuda", dtype=torch.int64)
            route_map = storage[:, ::2]
            if rows:
                route_map.copy_(torch.arange(rows * 6, device="cuda").view(rows, 6))
                route_map[::3, 4] = -1
            actual = combine_ordered_rows(down, route_map, allow_prefill=True)
            self.assertIsNotNone(actual)
            expected = torch.zeros(rows, 5120, device="cuda", dtype=torch.float32)
            for slot in range(6):
                index = route_map[:, slot]
                value = down.index_select(0, index.clamp_min(0)).float()
                expected.add_(torch.where(index[:, None] >= 0, value, 0))
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            if rows > 128:
                self.assertIsNone(combine_ordered_rows(down, route_map))

    def test_return_packets_match_existing_same_mode_pipeline(self):
        for rows in (0, 1, 3, 129, 1024, 4096):
            partial = torch.randn(rows, 5120, device="cuda")
            if rows:
                partial[0, :4] = torch.tensor(
                    [-0.0, 1e-8, 1.00390625, -1.00390625], device="cuda"
                )
            for mode in ("fp32", "mxfp8"):
                layout = V41ReturnLayout(5120, mode)
                actual = pack_return(partial, layout)
                if mode == "fp32":
                    expected = partial.view(torch.uint8)
                else:
                    values, scales = quantize_group32(partial.bfloat16())
                    expected = torch.cat((values.view(torch.uint8), scales), dim=-1)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
