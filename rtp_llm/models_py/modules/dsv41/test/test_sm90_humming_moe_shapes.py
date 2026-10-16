"""Actual V4.1 routed GEMM shapes, without loading full-model/Engram weights."""

import unittest

import torch

from rtp_llm.models_py.modules.dsv41.moe._sm90_humming_linear import (
    SM90HummingGroupedLinear,
)


def _quantized_input_reference(x):
    blocks = x.float().reshape(x.shape[0], -1, 128)
    scales = torch.pow(
        2.0, torch.ceil(torch.log2(blocks.abs().amax(-1).clamp_min(1e-4) / 448))
    )
    rounded = (blocks / scales.unsqueeze(-1)).to(torch.float8_e4m3fn).float()
    return (rounded * scales.unsqueeze(-1)).reshape_as(x)


def _weight_reference(weight, scale):
    lut = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
        dtype=torch.float32,
        device=weight.device,
    )
    packed = weight.to(torch.int16) & 255
    nibbles = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2)
    return lut[nibbles.long()] * scale.float().repeat_interleave(32, dim=-1)


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 9,
    "requires SM90",
)
class SM90ActualRoutedShapeTest(unittest.TestCase):
    def test_w13_and_down_actual_shapes_quantized_reference_and_repeatability(self):
        previous = torch.backends.cuda.matmul.allow_tf32
        self.addCleanup(setattr, torch.backends.cuda.matmul, "allow_tf32", previous)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.manual_seed(20261013)
        for name, n, k in (("w13", 4608, 5120), ("down", 5120, 2304)):
            weight = torch.randint(
                -128, 128, (2, n, k // 2), dtype=torch.int8, device="cuda"
            )
            # Cover the audited checkpoint's absolute [117, 127] range and
            # maximum per-expert span of nine, including guaranteed endpoints.
            scale_bytes = torch.stack(
                [
                    torch.randint(
                        117 + expert,
                        127 + expert,
                        (n, k // 32),
                        dtype=torch.uint8,
                        device="cuda",
                    )
                    for expert in range(2)
                ]
            )
            for expert in range(2):
                scale_bytes[expert, 0, 0] = 117 + expert
                scale_bytes[expert, 0, 1] = 126 + expert
            scale = scale_bytes.view(torch.float8_e8m0fnu)
            linear = SM90HummingGroupedLinear(weight, scale)
            # Decode one expert at a time, keeping this test below full-model
            # memory requirements and independent of Humming's weight repacking.
            for rows_per_expert in (1, 3):
                with self.subTest(projection=name, rows=rows_per_expert):
                    x = (
                        torch.randn(2 * rows_per_expert, k, device="cuda") * 12
                    ).bfloat16()
                    offsets = (
                        torch.arange(3, dtype=torch.int32, device="cuda")
                        * rows_per_expert
                    )
                    qx = _quantized_input_reference(x)
                    reference = torch.cat(
                        [
                            (
                                qx[
                                    expert
                                    * rows_per_expert : (expert + 1)
                                    * rows_per_expert
                                ]
                                @ _weight_reference(weight[expert], scale[expert]).T
                            ).bfloat16()
                            for expert in range(2)
                        ],
                        dim=0,
                    )
                    actual = linear(x, offsets)
                    difference = (actual.float() - reference.float()).abs()
                    print(
                        f"MoE actual_shape={name} E=2 M_per_expert={rows_per_expert} "
                        f"N={n} K={k} max_abs={difference.max().item():.8g} "
                        f"unequal={(actual != reference).sum().item()}",
                        flush=True,
                    )
                    # Same final tolerance as the small-shape and EP regression;
                    # no precision exemption for the production reduction size.
                    torch.testing.assert_close(actual, reference, rtol=0.02, atol=0.05)
                    for _ in range(10):
                        torch.testing.assert_close(
                            linear(x, offsets), actual, rtol=0, atol=0
                        )
            del linear, weight, scale


if __name__ == "__main__":
    unittest.main()
