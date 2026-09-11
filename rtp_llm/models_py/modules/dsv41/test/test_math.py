import os
import unittest

import torch
import torch.nn.functional as F
from standalone_load import load_component

_math = load_component("rtp_v41_math_test", "models_py/modules/dsv41/math.py")
dequantize_block32 = _math.dequantize_block32
engram_inject = _math.engram_inject
grouped_wo_a = _math.grouped_wo_a
hc_post = _math.hc_post
hc_pre = _math.hc_pre
identity_pre_mix = _math.identity_pre_mix
moe_gate = _math.moe_gate
swiglu_activation = _math.swiglu_activation


class MathTest(unittest.TestCase):
    def setUp(self):
        self.device = torch.device(os.environ.get("DSV41_TEST_DEVICE", "cuda"))

    def test_wo_a_preserves_bf16_group_boundaries(self):
        attention = torch.ones(2, 8, 4096, dtype=torch.bfloat16, device=self.device)
        weight = torch.zeros(8, 1024, 4096, dtype=torch.bfloat16, device=self.device)
        weight[:, :, 0] = torch.arange(1, 9, device=self.device)[:, None]
        result = grouped_wo_a(attention, weight)
        expected = (
            torch.arange(1, 9, device=self.device)
            .view(1, 8, 1)
            .expand(2, 8, 1024)
            .bfloat16()
        )
        self.assertTrue(torch.equal(result, expected))
        with self.assertRaises(ValueError):
            grouped_wo_a(attention.float(), weight)

    def test_dense_block32_uses_both_axes(self):
        weight = torch.ones(64, 96, dtype=torch.float8_e4m3fn, device=self.device)
        scale = torch.tensor(
            [[1, 2, 4], [8, 16, 32]], dtype=torch.float32, device=self.device
        )
        decoded = dequantize_block32(weight, scale)
        for row in range(2):
            for col in range(3):
                self.assertTrue(
                    torch.all(
                        decoded[row * 32 : (row + 1) * 32, col * 32 : (col + 1) * 32]
                        == scale[row, col]
                    ).item()
                )

    def test_swiglu_negative_gate_and_route_before_cast(self):
        gate = torch.tensor(
            [[-40, -12, -1.125, 0, 4.25, 11]], device=self.device
        ).bfloat16()
        up = torch.tensor(
            [[-40, 12, -2.25, 1, 1.25, 40]], device=self.device
        ).bfloat16()
        route = torch.tensor([0.3125], device=self.device)
        expected = (
            F.silu(gate.float().clamp(max=10))
            * up.float().clamp(-10, 10)
            * route[:, None]
        ).bfloat16()
        actual = swiglu_activation(gate, up, route)
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(actual[0, 3].item(), 0)
        self.assertGreater(actual[0, 0].item(), 0)

    def test_hc_source_destination_and_one_hot(self):
        hidden = torch.arange(4 * 5120, device=self.device).view(1, 4, 5120).bfloat16()
        self.assertTrue(
            torch.equal(hc_pre(hidden, identity_pre_mix(hidden)), hidden[:, 0])
        )
        comb = torch.zeros(1, 4, 4, device=self.device)
        comb[:, 0, 3] = 1
        result = hc_post(
            torch.zeros(1, 5120, device=self.device).bfloat16(),
            hidden,
            torch.zeros(1, 4, device=self.device),
            comb,
        )
        self.assertTrue(torch.equal(result[:, 3], hidden[:, 0]))
        self.assertTrue(torch.all(result[:, :3] == 0).item())

    def test_image_bias_selects_but_does_not_weight(self):
        hidden = torch.zeros(2, 5120, device=self.device).bfloat16()
        gate = torch.zeros(384, 5120, device=self.device).bfloat16()
        bias = torch.arange(384, device=self.device).float()
        weights, indices = moe_gate(
            hidden,
            gate,
            bias,
            -bias,
            torch.tensor([False, True], device=self.device),
            6,
        )
        self.assertEqual(indices[0].tolist(), [383, 382, 381, 380, 379, 378])
        self.assertEqual(indices[1].tolist(), [0, 1, 2, 3, 4, 5])
        self.assertTrue(torch.equal(weights[0], weights[1]))

    def test_image_mask_suppresses_engram_injection(self):
        hidden = torch.ones(2, 4, 5120, device=self.device).bfloat16()
        projected = torch.ones(2, 5 * 5120, device=self.device).bfloat16()
        qk = torch.ones(4, 5120, device=self.device).bfloat16()
        result = engram_inject(
            hidden, projected, qk, qk, torch.tensor([False, True], device=self.device)
        )
        self.assertTrue(torch.equal(result[0], hidden[0]))
        self.assertTrue(torch.all(result[1] > hidden[1]).item())


if __name__ == "__main__":
    unittest.main()
