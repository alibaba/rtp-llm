"""Bitwise PTPC activation/scale and grouped producer contracts."""

import os
import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv41.ppu.ptpc_quant import (
    pack_ptpc_grouped_output,
    ptpc_fusion_enabled,
    quantize_ptpc_activation,
    quantize_ptpc_grouped_activation,
)


def oracle(value):
    values = value.float()
    amax = values.abs().amax(-1, keepdim=True).clamp_min(1e-10)
    # Float32 reciprocal of 448, independently written as its exact value.
    scale = amax * 0.0022321429569274187
    quant = (values * (448.0 / amax)).clamp(-448, 448).to(torch.float8_e4m3fn)
    return quant.contiguous(), scale.contiguous()


def assert_quant_equal(test, actual, expected):
    aq, asc = actual
    eq, esc = expected
    test.assertEqual(aq.dtype, torch.float8_e4m3fn)
    test.assertEqual(asc.dtype, torch.float32)
    test.assertEqual(aq.shape, eq.shape)
    test.assertEqual(asc.shape, esc.shape)
    test.assertTrue(aq.is_contiguous())
    test.assertTrue(asc.is_contiguous())
    for name, actual_bits, expected_bits in (
        ("payload", aq.view(torch.uint8), eq.view(torch.uint8)),
        ("scale", asc.view(torch.int32), esc.view(torch.int32)),
    ):
        if not torch.equal(actual_bits, expected_bits):
            position = tuple((actual_bits != expected_bits).nonzero()[0].tolist())
            actual_value = int(actual_bits[position].item()) & 0xFFFFFFFF
            expected_value = int(expected_bits[position].item()) & 0xFFFFFFFF
            test.fail(
                f"{name} bit mismatch at {position}: actual=0x{actual_value:x}, "
                f"expected=0x{expected_value:x}"
            )


def adversarial_rows(width, device):
    # amax=7 and x=6.875 yield a theoretical E4M3 tie at440. The reciprocal
    # and multiplication ordering therefore affects payload bytes.
    base = torch.tensor(
        [7.0, 6.875, -7.0, -6.875, 0.0, -0.0, 0.001953125, -0.001953125],
        dtype=torch.bfloat16,
        device=device,
    )
    row = base.repeat((width + base.numel() - 1) // base.numel())[:width]
    multipliers = torch.tensor(
        [0.0, 2.0**-120, 2.0**-40, 2.0**-10, 1.0, 2.0**20, 2.0**100],
        dtype=torch.float32,
        device=device,
    )
    return (multipliers[:, None] * row.float()[None, :]).to(torch.bfloat16)


class PTPCQuantCPU(unittest.TestCase):
    def test_fusion_default_and_explicit_rollback(self):
        with patch.dict(os.environ):
            os.environ.pop("RTP_V41_PPU_PTPC_FUSION", None)
            self.assertTrue(ptpc_fusion_enabled())
        with patch.dict(os.environ, {"RTP_V41_PPU_PTPC_FUSION": "0"}):
            self.assertFalse(ptpc_fusion_enabled())
        with patch.dict(os.environ, {"RTP_V41_PPU_PTPC_FUSION": "1"}):
            self.assertTrue(ptpc_fusion_enabled())

    def test_cpu_exact_and_empty_shapes(self):
        for shape in ((129,), (7, 257), (2, 3, 128), (0, 5120), (2, 0, 128)):
            with self.subTest(shape=shape):
                value = torch.randn(shape, dtype=torch.bfloat16)
                assert_quant_equal(self, quantize_ptpc_activation(value), oracle(value))

    def test_grouped_gate_order_and_strides(self):
        value = torch.arange(4 * 3 * 128, dtype=torch.float32).reshape(4, 3, 128)
        value = ((value % 131) - 65).to(torch.bfloat16)[:, :, ::2]
        actual = quantize_ptpc_grouped_activation(value)
        for group in range(value.shape[1]):
            assert_quant_equal(
                self, (actual[0][group], actual[1][group]), oracle(value[:, group])
            )

    def test_rejects_empty_feature_dimension_and_wrong_group_rank(self):
        with self.assertRaisesRegex(ValueError, "nonempty feature"):
            quantize_ptpc_activation(torch.empty(4, 0))
        with self.assertRaisesRegex(ValueError, "requires \\[M, G, K\\]"):
            quantize_ptpc_grouped_activation(torch.empty(2, 128))

    def test_grouped_output_layout_preserves_all_bytes(self):
        raw = torch.tensor(
            [0x0000, 0x8000, 0x7FC1, 0xFFFF, 0x3F80, 0xBF80], dtype=torch.uint16
        ).reshape(3, 1, 2).expand(3, 4, 2).contiguous()
        value = raw.view(torch.bfloat16)
        actual = pack_ptpc_grouped_output(value)
        self.assertEqual(actual.shape, (4, 3, 2))
        self.assertTrue(actual.is_contiguous())
        self.assertTrue(torch.equal(actual.view(torch.uint16), raw.permute(1, 0, 2)))


@unittest.skipUnless(
    os.environ.get("RTP_V41_TEST_PTPC_DEVICE") == "1", "explicit PPU device test"
)
class PTPCQuantDevice(unittest.TestCase):
    def test_production_and_ragged_widths_exact(self):
        for width in (
            1, 127, 128, 257, 512, 1024, 1280, 2304, 4096, 5120, 8192, 9216, 15360
        ):
            with self.subTest(width=width):
                value = adversarial_rows(width, "cuda")
                assert_quant_equal(self, quantize_ptpc_activation(value), oracle(value))
                random = torch.randn((17, width), device="cuda", dtype=torch.bfloat16)
                assert_quant_equal(
                    self, quantize_ptpc_activation(random), oracle(random)
                )

    def test_float32_boundaries_and_subnormal_rows_exact(self):
        # A common BF16 input domain plus F32 values neighboring the FP8 tie
        # covers separate scalar reciprocal / multiply rounding boundaries.
        value = adversarial_rows(1280, "cuda").float()
        one = torch.ones_like(value)
        for values in (
            value, torch.nextafter(value, value + one), torch.nextafter(value, value - one)
        ):
            assert_quant_equal(self, quantize_ptpc_activation(values), oracle(values))
        ties = torch.tensor(
            [[amax, amax * (440.0 / 448.0), -amax * (440.0 / 448.0), 0.0]
             for amax in (3.0, 7.0, 49.0, 127.5, 255.0, 513.0, 1025.0, 3670.75)],
            dtype=torch.float32,
            device="cuda",
        ).repeat(1, 32)
        assert_quant_equal(self, quantize_ptpc_activation(ties), oracle(ties))
        extremes = torch.tensor(
            [[torch.finfo(torch.bfloat16).max, 2.0**126, -(2.0**125), 0.0]],
            dtype=torch.bfloat16,
            device="cuda",
        )
        assert_quant_equal(self, quantize_ptpc_activation(extremes), oracle(extremes))
        tiny = torch.tensor(
            [0.0, -0.0, 2.0**-133, -(2.0**-133)],
            dtype=torch.bfloat16,
            device="cuda",
        )
        assert_quant_equal(self, quantize_ptpc_activation(tiny), oracle(tiny))

    def test_strided_and_grouped_producers_exact(self):
        value = torch.randn((9, 8, 1024), device="cuda", dtype=torch.bfloat16)
        for values in (value, value[::2, :, ::2], value.transpose(0, 1)):
            with self.subTest(shape=values.shape, stride=values.stride()):
                assert_quant_equal(
                    self, quantize_ptpc_activation(values), oracle(values)
                )
                packed = quantize_ptpc_grouped_activation(values)
                for group in range(values.shape[1]):
                    assert_quant_equal(
                        self,
                        (packed[0][group], packed[1][group]),
                        oracle(values[:, group]),
                    )
        for shape in ((0, 8, 512), (3, 0, 512)):
            values = torch.empty(shape, device="cuda", dtype=torch.bfloat16)
            q, s = quantize_ptpc_grouped_activation(values)
            self.assertEqual(q.shape, (shape[1], shape[0], shape[2]))
            self.assertEqual(s.shape, (shape[1], shape[0], 1))

    def test_changed_input_graph_replay_exact(self):
        values = torch.randn((13, 8, 512), device="cuda", dtype=torch.bfloat16)
        quantize_ptpc_activation(values)
        quantize_ptpc_grouped_activation(values)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = quantize_ptpc_activation(values)
            grouped = quantize_ptpc_grouped_activation(values)
        for multiplier in (0.0, 0.125, 2.0, 0.00001):
            values.copy_(torch.randn_like(values) * multiplier)
            graph.replay()
            assert_quant_equal(self, actual, oracle(values))
            for group in range(values.shape[1]):
                assert_quant_equal(
                    self,
                    (grouped[0][group], grouped[1][group]),
                    oracle(values[:, group]),
                )

    def test_grouped_output_pack_and_graph_preserve_bytes(self):
        raw = torch.tensor(
            [0, -32768, 32705, -1, 16256, -16512],
            dtype=torch.int16,
            device="cuda",
        ).repeat(8 * 72 * 1024 // 6 + 1)[:8 * 72 * 1024].reshape(8, 72, 1024)
        source = raw.view(torch.bfloat16)
        pack_ptpc_grouped_output(source)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = pack_ptpc_grouped_output(source)
        for value in (0, -32768, 32705, 16256):
            raw.fill_(value)
            graph.replay()
            self.assertTrue(actual.is_contiguous())
            self.assertTrue(torch.equal(actual.view(torch.int16), raw.permute(1, 0, 2)))
        for shape in ((8, 0, 1024), (0, 72, 1024)):
            value = torch.empty(shape, device="cuda", dtype=torch.bfloat16)
            self.assertEqual(
                pack_ptpc_grouped_output(value).shape, (shape[1], shape[0], shape[2])
            )


if __name__ == "__main__":
    unittest.main()
