"""Exact shared-expert wrapper boundaries, including native quantizer selection."""

import os
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8
from rtp_llm.models_py.modules.dsv4._silu_mul_split_triton import silu_mul_split
from rtp_llm.models_py.modules.dsv4.moe._v41_shared_act_quant_triton import (
    shared_act_quant,
)


def reference(gate_up, clamp):
    gate, up = gate_up.float().chunk(2, dim=-1)
    hidden = silu_mul_split(gate.contiguous(), up.contiguous(), clamp).bfloat16()
    return sgl_per_token_group_quant_fp8(
        hidden,
        group_size=32,
        eps=torch.finfo(torch.float32).tiny,
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=True,
    )


class V41SharedActQuantTest(unittest.TestCase):
    def setUp(self):
        self.assertTrue(torch.cuda.is_available())
        self.assertEqual(torch.cuda.get_device_capability(), (12, 0))
        torch.manual_seed(41724)

    def compare(self, gate_up, clamp, expected=None):
        expected = reference(gate_up, clamp) if expected is None else expected
        actual = shared_act_quant(gate_up, clamp_limit=clamp)
        self.assertIsNotNone(actual)
        torch.testing.assert_close(
            actual[0].view(torch.uint8), expected[0].view(torch.uint8), rtol=0, atol=0
        )
        torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
        self.assertEqual(
            actual[1].stride(), (1, max(1, (gate_up.shape[0] + 3) // 4 * 4))
        )
        if gate_up.shape[0] % 4:
            padding = actual[1]._base[:, gate_up.shape[0] :]
            self.assertEqual(torch.count_nonzero(padding).item(), 0)

    def test_random_and_strided_merged_projection(self):
        for rows in (
            0,
            1,
            3,
            31,
            127,
            128,
            129,
            257,
            1021,
            1024,
            1820,
            1821,
            4093,
            4096,
        ):
            storage = (torch.randn(rows, 4736, device="cuda") * 8).bfloat16()
            merged = storage[:, :4608]
            for mode in ("legacy", "v2", "auto"):
                with self.subTest(rows=rows, mode=mode), patch.dict(
                    os.environ, {"DSV4_FP8_QUANT_KERNEL": mode}
                ):
                    for clamp in (0.0, 10.0):
                        self.compare(merged, clamp)

    def test_rounding_zero_tiny_and_scale_transitions(self):
        gate_values = torch.tensor(
            [-100, -90, -20, -10, -1, -0.0, 0, 1, 10, 20],
            device="cuda",
            dtype=torch.bfloat16,
        )
        for mode in ("legacy", "v2"):
            with patch.dict(os.environ, {"DSV4_FP8_QUANT_KERNEL": mode}):
                for exponent in (-120, -30, -20, -8, 0, 8):
                    merged = torch.zeros(129, 4608, device="cuda", dtype=torch.bfloat16)
                    merged[:, :2304] = gate_values[
                        torch.arange(2304, device="cuda") % len(gate_values)
                    ]
                    merged[:, 2304:] = 2.0**exponent
                    with self.subTest(mode=mode, exponent=exponent):
                        self.compare(merged, 10.0)

    def test_captured_operands(self):
        directory = os.environ.get("DSV41_MOE_FIXTURES")
        if not directory:
            self.skipTest("Real model fixture path not supplied")
        fixtures = sorted(Path(directory).glob("rank*-layer*-chunk*.pt"))
        self.assertEqual(len(fixtures), 8)
        for path in fixtures:
            record = torch.load(path, map_location="cpu", weights_only=True)
            tensors = record["tensors"]
            merged = tensors["shared_gate_up"].cuda()
            expected = (
                tensors["shared_values"].cuda(),
                tensors["shared_packed_scales"].cuda(),
            )
            with self.subTest(fixture=path.name):
                self.compare(merged, 10.0, expected)

    def test_unsupported_inputs_leave_the_existing_path_available(self):
        self.assertIsNone(shared_act_quant(torch.empty(3, 4608)))
        self.assertIsNone(shared_act_quant(torch.empty(3, 4608, device="cuda")))
        self.assertIsNone(
            shared_act_quant(torch.empty(3, 64, device="cuda", dtype=torch.bfloat16))
        )

    def test_whole_shared_expert_same_shape(self):
        from rtp_llm.models_py.modules.dsv4.moe.shared_expert import W13SharedExpert

        weights = {}
        for name, n, k in (("w13", 4608, 5120), ("w2", 5120, 2304)):
            weights[name + "_w"] = (torch.randn(n, k, device="cuda") * 0.1).to(
                torch.float8_e4m3fn
            )
            weights[name + "_s"] = torch.full(
                (n // 32, k // 32), 127, dtype=torch.uint8, device="cuda"
            ).view(torch.float8_e8m0fnu)
        with patch.dict(os.environ, {"DSV41_SHARED_ACT_QUANT_FUSE": "1"}):
            expert = W13SharedExpert(5120, 2304, weights, swiglu_limit=10.0)
        for rows in (129, 1024):
            x = torch.randn(rows, 5120, dtype=torch.bfloat16, device="cuda")
            with patch.object(expert, "_v41_shared_act_quant", False):
                expected = expert(x)
            actual = expert(x)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
