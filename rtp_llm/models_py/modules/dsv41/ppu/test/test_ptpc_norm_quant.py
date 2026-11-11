"""Exact component gates for the unpromoted ATen-tree norm/PTPC candidate."""

import os
import struct
import unittest

import torch

from rtp_llm.models_py.modules.dsv41.ppu.ptpc_norm_quant import (
    aten_mean_launch_config,
    is_supported,
    ptpc_rms_norm_candidate,
    ptpc_rms_norm_quant_candidate,
)


def canonical_norm(value, gamma, eps):
    values = value.float()
    inverse = torch.rsqrt(values.square().mean(-1, keepdim=True) + eps)
    return (values * inverse * gamma.float()).to(value.dtype)


def canonical_quant(value):
    values = value.float()
    maximum = values.abs().amax(-1, keepdim=True).clamp_min(1e-10)
    scale = maximum * 0.0022321429569274187
    payload = (values * (448.0 / maximum)).clamp(-448, 448).to(torch.float8_e4m3fn)
    return payload.contiguous(), scale.contiguous()


def exact_bits(test, actual, expected, label):
    test.assertIsNotNone(actual, label)
    test.assertEqual(actual.dtype, expected.dtype, label)
    test.assertEqual(actual.shape, expected.shape, label)
    left = actual.contiguous().view(torch.uint8).flatten()
    right = expected.contiguous().view(torch.uint8).flatten()
    if not torch.equal(left, right):
        offset = int((left != right).nonzero()[0].item())
        test.fail(
            f"{label}: byte {offset}, actual=0x{int(left[offset]):02x}, "
            f"reference=0x{int(right[offset]):02x}"
        )


def exact_producer(test, actual, value, gamma, eps):
    test.assertIsNotNone(actual)
    expected_norm = canonical_norm(value, gamma, eps)
    expected_payload, expected_scale = canonical_quant(expected_norm)
    for label, left, right in zip(
        ("BF16 norm", "E4M3 payload", "F32 scale"),
        actual, (expected_norm, expected_payload, expected_scale),
    ):
        exact_bits(test, left, right, label)


class PTPCNormQuantCPU(unittest.TestCase):
    def test_trace_launch_dimensions_and_project_factor(self):
        # Directly observed in C05 PPU traces, not inferred from template512.
        for width in (512, 1280, 5120):
            self.assertEqual(aten_mean_launch_config(8, width)[:2], (64, 8))
            for rows in (24, 32, 144, 4096, 16384):
                self.assertEqual(aten_mean_launch_config(rows, width)[:2], (32, 16))
        expected = {512: 0x3B000000, 1280: 0x3A4CCCCD, 5120: 0x394CCCCD}
        for width, bits in expected.items():
            factor = aten_mean_launch_config(144, width)[2]
            self.assertEqual(struct.unpack("<I", struct.pack("<f", factor))[0], bits)
        self.assertEqual(aten_mean_launch_config(1, 512)[:2], (128, 1))
        self.assertEqual(aten_mean_launch_config(1, 1280)[:2], (256, 1))
        self.assertEqual(aten_mean_launch_config(1, 5120)[:2], (512, 1))
        self.assertEqual(aten_mean_launch_config(3, 5120)[:2], (256, 2))

    def test_candidate_does_not_promote_unsupported_cpu_input(self):
        value = torch.ones(3, 5120, dtype=torch.bfloat16)
        gamma = torch.ones(5120, dtype=torch.bfloat16)
        self.assertFalse(is_supported(value, gamma, 1e-6))
        self.assertIsNone(ptpc_rms_norm_candidate(value, gamma))
        self.assertIsNone(ptpc_rms_norm_quant_candidate(value, gamma))
        for shape in ((0, 512), (2, 513), (2, 3, 1280)):
            value = torch.empty(shape, dtype=torch.bfloat16)
            gamma = torch.ones(shape[-1], dtype=torch.bfloat16)
            self.assertIsNone(ptpc_rms_norm_candidate(value, gamma))


@unittest.skipUnless(
    os.environ.get("RTP_V41_TEST_PTPC_DEVICE") == "1", "explicit PPU candidate test"
)
class PTPCNormQuantDevice(unittest.TestCase):
    @torch.no_grad()
    def test_tree_sensitive_shapes_and_single_kernel_norm(self):
        torch.manual_seed(4121)
        for width in (512, 1280, 5120):
            gamma = torch.randn(width, device="cuda", dtype=torch.bfloat16)
            for rows in (1, 2, 3, 4, 8, 15, 16, 24, 32, 72, 144, 4096, 16384):
                with self.subTest(rows=rows, width=width):
                    value = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16)
                    actual = ptpc_rms_norm_candidate(value, gamma)
                    exact_bits(self, actual, canonical_norm(value, gamma, 1e-6), "norm")

    @torch.no_grad()
    def test_combined_producer_preserves_bf16_landing_and_scales(self):
        torch.manual_seed(4122)
        for width in (512, 1280, 5120):
            gamma = torch.randn(width, device="cuda", dtype=torch.bfloat16)
            for rows in (1, 3, 8, 24, 144, 4096):
                with self.subTest(rows=rows, width=width):
                    value = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16)
                    exact_producer(self, ptpc_rms_norm_quant_candidate(value, gamma), value, gamma, 1e-6)
            value = torch.randn(2, 4, width, device="cuda", dtype=torch.bfloat16)
            exact_producer(self, ptpc_rms_norm_quant_candidate(value, gamma), value, gamma, 1e-6)

    @torch.no_grad()
    def test_signed_zero_subnormal_extremes_and_rounding_boundaries(self):
        raw = torch.tensor(
            [0, -32768, 1, -32767, 127, -32641, 128, -32640,
             16255, 16256, 16257, -16513, -16512, -16511],
            dtype=torch.int16,
        )
        for width in (512, 1280, 5120):
            value = raw.repeat((width + raw.numel() - 1) // raw.numel())[:width]
            value = value.view(torch.bfloat16)[None].repeat(8, 1).cuda()
            gamma = value[0].clone()
            for eps in (1e-6, 1e-5, 1e-12):
                exact_producer(self, ptpc_rms_norm_quant_candidate(value, gamma, eps), value, gamma, eps)
            gamma.fill_(1)
            for magnitude in (0.0, 2.0**-70, 2.0**-65, 2.0**-30, 2.0**30, torch.finfo(torch.bfloat16).max):
                value.fill_(magnitude)
                value[:, 1::2].neg_()
                exact_producer(self, ptpc_rms_norm_quant_candidate(value, gamma), value, gamma, 1e-6)

    @torch.no_grad()
    def test_graph_changes_input_and_gamma(self):
        for width in (512, 1280, 5120):
            for rows in (8, 24, 32, 144):
                value = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16)
                gamma = torch.randn(width, device="cuda", dtype=torch.bfloat16)
                ptpc_rms_norm_quant_candidate(value, gamma)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    actual = ptpc_rms_norm_quant_candidate(value, gamma)
                for magnitude in (0.0, 0.125, 1.0, 16.0):
                    value.copy_(torch.randn_like(value) * magnitude)
                    gamma.copy_(torch.randn_like(gamma))
                    graph.replay()
                    exact_producer(self, actual, value, gamma, 1e-6)

    @torch.no_grad()
    def test_empty_rows_and_guarded_metadata(self):
        for width in (512, 1280, 5120):
            gamma = torch.ones(width, dtype=torch.bfloat16, device="cuda")
            value = torch.empty((0, width), dtype=torch.bfloat16, device="cuda")
            actual = ptpc_rms_norm_quant_candidate(value, gamma)
            self.assertIsNotNone(actual)
            self.assertEqual(actual[0].shape, (0, width))
            self.assertEqual(actual[1].shape, (0, width))
            self.assertEqual(actual[2].shape, (0, 1))
        value = torch.ones(8, 5120, dtype=torch.bfloat16, device="cuda")
        gamma = torch.ones(5120, dtype=torch.bfloat16, device="cuda")
        self.assertIsNone(ptpc_rms_norm_candidate(value[:, ::2], gamma[::2]))
        self.assertIsNone(ptpc_rms_norm_candidate(value.float(), gamma))
        self.assertIsNone(ptpc_rms_norm_candidate(value, gamma, 0.0))


if __name__ == "__main__":
    unittest.main()
