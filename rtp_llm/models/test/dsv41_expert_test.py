"""Checkpoint expert operation order, scale ABI and shared-expert equations."""

import dataclasses
import math
import sys
import unittest
from unittest.mock import patch

import torch

from rtp_llm.models.test.dsv41_cache_codec_test import _f32, load_codec_source
from rtp_llm.models.test.dsv41_mxfp8_test import (
    grouped_reference,
    linear,
    quantization_reference,
    weights,
)

with patch.dict(sys.modules, {"rtp_llm.models_py.modules.dsv4.v41_mxfp8": linear}):
    expert = load_codec_source("_dsv41_expert_test", "v41_expert.py")


def scalar_swiglu(gate, up, valid, route=None, limit=10.0):
    output = torch.zeros_like(gate)
    flat_gate, flat_up = gate.reshape(-1, gate.shape[-1]), up.reshape(-1, up.shape[-1])
    result = output.reshape_as(flat_gate)
    route = torch.ones(valid.shape) if route is None else route
    for row in range(flat_gate.shape[0]):
        if not bool(valid.flatten()[row]) or not float(route.flatten()[row]):
            continue
        for column in range(flat_gate.shape[-1]):
            g, u = float(flat_gate[row, column]), float(flat_up[row, column])
            if limit:
                g, u = min(g, limit), max(-limit, min(u, limit))
            activation = _f32(g / _f32(1 + _f32(math.exp(-g))))
            result[row, column] = _f32(
                float(route.flatten()[row]) * _f32(activation * u)
            )
    return output


def shared_weights(dim=64, inter=96):
    w13, s13 = weights(n=2 * inter, k=dim)
    w2, s2 = weights(n=dim, k=inter)
    return dict(w13_w=w13, w13_s=s13, w2_w=w2, w2_s=s2)


def shared_reference(x, tensors, valid, route=None, limit=10.0):
    active = valid if route is None else valid & (route != 0)
    safe = torch.where(active[..., None], x, 0)
    raw, scales = quantization_reference(safe)
    gu = grouped_reference(
        raw.view(torch.float8_e4m3fn), scales, tensors["w13_w"], tensors["w13_s"]
    )
    gate, up = gu.chunk(2, -1)
    intermediate = scalar_swiglu(gate, up, valid, route, limit)
    raw, scales = quantization_reference(intermediate)
    output = grouped_reference(
        raw.view(torch.float8_e4m3fn), scales, tensors["w2_w"], tensors["w2_s"]
    )
    return torch.where(active[..., None], output, 0)


class V41ExpertTest(unittest.TestCase):
    def test_recipe_is_immutable_and_rejects_legacy_arithmetic(self):
        recipe = expert.V41_EXPERT_RECIPE
        self.assertEqual(recipe.activation_group, 32)
        self.assertEqual(recipe.router_weight_position, "before_down_input_bf16")
        self.assertEqual(hash(recipe), hash(expert.V41ExpertRecipe()))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            recipe.activation_group = 128
        for change in (
            {"activation_group": 128},
            {"amax_floor": 1e-10},
            {"gate_up_order": "up_then_gate"},
            {"router_weight_position": "after_down"},
        ):
            with self.assertRaisesRegex(ValueError, "recipe"):
                dataclasses.replace(recipe, **change)

    def test_swiglu_matches_scalar_checkpoint_order_and_asymmetric_clamps(self):
        gate = torch.tensor(
            [-18, -10, -3, 0, 1, 4, 10, 24], dtype=torch.bfloat16
        ).repeat(3, 4)
        up = torch.linspace(-20, 20, 96).reshape(3, 32).bfloat16()
        valid = torch.tensor([True, True, True])
        route = torch.tensor([0.37, 1.5, 0.001])
        for limit in (0, 10):
            result = expert.swiglu_bf16(
                gate, up, valid, route_weights=route, clamp_limit=limit
            )
            torch.testing.assert_close(
                result, scalar_swiglu(gate, up, valid, route, limit), rtol=0, atol=0
            )
        self.assertNotEqual(float(result[0, 0]), 0)

    def test_router_weight_precedes_bf16_rounding_and_changes_quantized_result(self):
        generator = torch.Generator().manual_seed(41)
        gate = (torch.randn(4, 128, generator=generator) * 5).bfloat16()
        up = (torch.randn(4, 128, generator=generator) * 4).bfloat16()
        valid, route = torch.ones(4, dtype=torch.bool), torch.full((4,), 0.37)
        correct = expert.swiglu_bf16(gate, up, valid, route_weights=route)
        late = (expert.swiglu_bf16(gate, up, valid).float() * route[:, None]).bfloat16()
        self.assertGreater(int((correct != late).sum()), 0)
        q, s = expert.quantize_swiglu(
            gate, up, valid, route_weights=route, backend="eager"
        )
        raw, expected_s = quantization_reference(correct)
        torch.testing.assert_close(q.view(torch.uint8), raw, rtol=0, atol=0)
        torch.testing.assert_close(s, expected_s, rtol=0, atol=0)
        op = linear.V41MXFP8Linear(*weights(n=64, k=128), backend="eager")
        wrong = (
            op(expert.swiglu_bf16(gate, up, valid)).float() * route[:, None]
        ).bfloat16()
        self.assertGreater(int((op(correct) != wrong).sum()), 0)

    def test_dead_and_zero_weight_routes_mask_nonfinite_inputs_before_arithmetic(self):
        gate = torch.ones(3, 64, dtype=torch.bfloat16)
        up = torch.ones_like(gate)
        gate[1:], up[1:] = float("nan"), float("inf")
        valid, route = torch.tensor([True, False, True]), torch.tensor(
            [0.5, float("nan"), 0]
        )
        values, scales = expert.quantize_swiglu(
            gate, up, valid, route_weights=route, backend="eager"
        )
        self.assertTrue(bool((values[1:].view(torch.uint8) == 0).all()))
        self.assertTrue(bool((scales[1:] == 0).all()))
        self.assertTrue(bool((scales[0] != 0).all()))

    def test_masked_expert_dimensions_strided_gates_and_empty_batches(self):
        gu = torch.randn(3, 5, 128).bfloat16()
        gate, up = gu.chunk(2, -1)
        valid = torch.arange(15).reshape(3, 5) % 2 == 0
        route = torch.full((3, 5), 0.7)
        expected = scalar_swiglu(gate, up, valid, route)
        actual = expert.swiglu_bf16(gate, up, valid, route_weights=route)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        values, scales = expert.quantize_swiglu(
            gate[:, :0], up[:, :0], valid[:, :0].contiguous(), backend="eager"
        )
        self.assertEqual(values.shape, (3, 0, 64))
        self.assertEqual(scales.shape, (0, 2))

    def test_scale_pack_row_and_tma_layouts_preserve_bytes_and_zero_padding(self):
        for rows in (0, 1, 5):
            for groups in (3, 72, 160):
                raw = (
                    torch.arange(2 * rows * groups).reshape(2, rows, groups) % 256
                ).to(torch.uint8)
                for column_major in (False, True):
                    packed = expert.pack_group32_scales(raw, column_major=column_major)
                    unpacked = (
                        torch.stack(
                            [
                                (packed.to(torch.int64) >> (8 * byte)) & 255
                                for byte in range(4)
                            ],
                            dim=-1,
                        )
                        .reshape(2, rows, (groups + 3) // 4 * 4)
                        .to(torch.uint8)
                    )
                    torch.testing.assert_close(
                        unpacked[..., :groups], raw, rtol=0, atol=0
                    )
                    self.assertTrue(bool((unpacked[..., groups:] == 0).all()))
                    if column_major:
                        self.assertEqual(packed.stride(-2), 1)
                        self.assertEqual(packed.stride(-1), max(4, (rows + 3) // 4 * 4))
                        backing = packed.as_strided(
                            (2, max(4, (rows + 3) // 4 * 4), (groups + 3) // 4),
                            packed.stride(),
                        )
                        self.assertTrue(bool((backing[:, rows:] == 0).all()))
        packed = expert.pack_group32_scales(torch.ones(1, 160, dtype=torch.uint8))
        self.assertEqual(5120 + packed.numel() * 4 + 6 * 4 * 2, 5328)

    def test_invalid_swiglu_and_scale_contracts(self):
        gate, up, valid = (
            torch.ones(2, 32, dtype=torch.bfloat16),
            torch.ones(2, 32, dtype=torch.bfloat16),
            torch.ones(2, dtype=torch.bool),
        )
        for route in (
            torch.ones(2, 1),
            torch.ones(2).bfloat16(),
            torch.tensor([-1.0, 1]),
            torch.tensor([float("nan"), 1]),
        ):
            with self.assertRaises(ValueError):
                expert.swiglu_bf16(gate, up, valid, route_weights=route)
        for limit in (-1, float("inf"), True):
            with self.assertRaises(ValueError):
                expert.swiglu_bf16(gate, up, valid, clamp_limit=limit)
        for bad_gate in (gate.float(), gate[:, :-1], gate.clone().requires_grad_()):
            with self.assertRaises(ValueError):
                expert.swiglu_bf16(bad_gate, up, valid)
        for scales in (
            torch.ones(2, 4, dtype=torch.int32),
            torch.ones(4, dtype=torch.uint8),
            torch.ones(4, 2, dtype=torch.uint8).T,
        ):
            with self.assertRaises(ValueError):
                expert.pack_group32_scales(scales)

    def test_shared_expert_matches_independent_grouped_checkpoint_equations(self):
        tensors = shared_weights()
        op = expert.V41SharedExpert(tensors, backend="eager")
        x = torch.randn(
            2, 3, 64, generator=torch.Generator().manual_seed(81)
        ).bfloat16()
        valid = torch.tensor([[True, False, True], [True, True, False]])
        x[~valid] = float("nan")
        for route in (None, torch.full((2, 3), 0.37)):
            expected = shared_reference(x, tensors, valid, route)
            out = torch.empty_like(x)
            self.assertIs(op(x, valid, route_weights=route, out=out), out)
            torch.testing.assert_close(out, expected, rtol=0, atol=0)
        self.assertIs(op.w13.weight, tensors["w13_w"])
        self.assertIs(op.w2.weight, tensors["w2_w"])

    def test_shared_expert_chunk_equivalence_empty_and_alias_contracts(self):
        op = expert.V41SharedExpert(shared_weights(), backend="eager")
        x, valid = torch.randn(7, 64).bfloat16(), torch.arange(7) % 3 != 0
        expected = op(x, valid)
        actual = torch.cat([op(x[:3], valid[:3]), op(x[3:], valid[3:])])
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(op(x[:0], valid[:0]).shape, (0, 64))
        with self.assertRaisesRegex(ValueError, "share"):
            op(x, valid, out=x)
        out = torch.full_like(x, 42)
        with self.assertRaises(ValueError):
            op(x, valid.long(), out=out)
        torch.testing.assert_close(out, torch.full_like(out, 42))


if __name__ == "__main__":
    unittest.main()
