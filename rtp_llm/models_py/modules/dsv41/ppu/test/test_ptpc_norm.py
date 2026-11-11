"""Exact PTPC norm boundaries and changed-input device Graph components."""

import os
import unittest

import torch

from rtp_llm.models_py.modules.dsv41.ppu.ptpc_norm import (
    can_fuse_ptpc_rms_norm,
    ptpc_rms_norm,
)


def canonical_norm(value, gamma, eps):
    values = value.float()
    variance = values.square().mean(dim=-1, keepdim=True)
    inverse = torch.rsqrt(variance + eps)
    normalized = values * inverse
    return (normalized * gamma.float()).to(value.dtype)


def assert_bf16_bits(case, actual, expected):
    case.assertEqual(actual.dtype, torch.bfloat16)
    case.assertEqual(expected.dtype, torch.bfloat16)
    case.assertEqual(actual.shape, expected.shape)
    case.assertTrue(torch.equal(actual.view(torch.int16), expected.view(torch.int16)))


class PTPCNormCPU(unittest.TestCase):
    def test_portable_reference_and_strided_fallback(self):
        generator = torch.Generator().manual_seed(4117)
        for width in (512, 513, 1280, 1281, 5120, 5121):
            base = torch.randn(3, width * 2, generator=generator).to(torch.bfloat16)
            gamma = torch.randn(width, generator=generator).to(torch.bfloat16)
            for value in (base[:, ::2], base[:, ::2].contiguous()):
                with self.subTest(width=width, contiguous=value.is_contiguous()):
                    self.assertFalse(can_fuse_ptpc_rms_norm(value, gamma))
                    assert_bf16_bits(
                        self,
                        ptpc_rms_norm(value, gamma, 1e-6),
                        canonical_norm(value, gamma, 1e-6),
                    )

    def test_zero_rows_signed_zero_and_subnormal(self):
        bits = torch.tensor(
            [0, -32768, 1, -32767, 127, -32641, 128, -32640],
            dtype=torch.int16,
        )
        value = bits.view(torch.bfloat16).repeat(64).reshape(1, 512)
        gamma = torch.ones(512, dtype=torch.bfloat16)
        assert_bf16_bits(
            self, ptpc_rms_norm(value, gamma, 1e-6), canonical_norm(value, gamma, 1e-6)
        )
        empty = torch.empty((0, 512), dtype=torch.bfloat16)
        assert_bf16_bits(
            self, ptpc_rms_norm(empty, gamma, 1e-6), canonical_norm(empty, gamma, 1e-6)
        )

    def test_autograd_uses_canonical_operations(self):
        value = torch.randn(3, 512, dtype=torch.bfloat16, requires_grad=True)
        gamma = torch.ones(512, dtype=torch.bfloat16, requires_grad=True)
        self.assertFalse(can_fuse_ptpc_rms_norm(value, gamma))
        ptpc_rms_norm(value, gamma, 1e-6).float().sum().backward()
        self.assertIsNotNone(value.grad)
        self.assertIsNotNone(gamma.grad)


@unittest.skipUnless(
    os.environ.get("RTP_V41_TEST_PTPC_DEVICE") == "1", "explicit PPU component run"
)
class PTPCNormDevice(unittest.TestCase):
    @torch.no_grad()
    def test_supported_widths_exact_reference(self):
        torch.manual_seed(4117)
        for width in (512, 1280, 5120):
            gamma = torch.randn(width, device="cuda", dtype=torch.bfloat16)
            for rows in (0, 1, 3, 31, 128, 4096):
                with self.subTest(width=width, rows=rows):
                    value = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16)
                    self.assertTrue(can_fuse_ptpc_rms_norm(value, gamma))
                    assert_bf16_bits(
                        self,
                        ptpc_rms_norm(value, gamma, 1e-6),
                        canonical_norm(value, gamma, 1e-6),
                    )
            value = torch.randn(2, 3, width, device="cuda", dtype=torch.bfloat16)
            assert_bf16_bits(
                self, ptpc_rms_norm(value, gamma, 1e-6), canonical_norm(value, gamma, 1e-6)
            )

    @torch.no_grad()
    def test_adversarial_bits_and_odd_width_fallback(self):
        bits = torch.tensor(
            [0, -32768, 1, -32767, 127, -32641, 128, -32640,
             16255, 16256, 16257, -16513, -16512, -16511],
            dtype=torch.int16,
        )
        for width in (512, 1280, 5120):
            repeat = (width + bits.numel() - 1) // bits.numel()
            value = bits.repeat(repeat)[:width].view(torch.bfloat16)[None].cuda()
            gamma = value[0].clone()
            for eps in (1e-6, 1e-5, 1e-12):
                assert_bf16_bits(
                    self, ptpc_rms_norm(value, gamma, eps), canonical_norm(value, gamma, eps)
                )
        for width in (513, 1281, 5121):
            value = torch.randn(3, width, device="cuda", dtype=torch.bfloat16)
            gamma = torch.randn(width, device="cuda", dtype=torch.bfloat16)
            self.assertFalse(can_fuse_ptpc_rms_norm(value, gamma))
            assert_bf16_bits(
                self, ptpc_rms_norm(value, gamma, 1e-6), canonical_norm(value, gamma, 1e-6)
            )

    @torch.no_grad()
    def test_graph_replays_changed_inputs_and_gamma(self):
        for width in (512, 1280, 5120):
            value = torch.randn(4, width, device="cuda", dtype=torch.bfloat16)
            gamma = torch.randn(width, device="cuda", dtype=torch.bfloat16)
            ptpc_rms_norm(value, gamma, 1e-6)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = ptpc_rms_norm(value, gamma, 1e-6)
            for magnitude in (0.0, 0.125, 1.0, 16.0):
                value.copy_(torch.randn_like(value) * magnitude)
                gamma.copy_(torch.randn_like(gamma))
                graph.replay()
                assert_bf16_bits(self, output, canonical_norm(value, gamma, 1e-6))


def load_tests(loader, tests, pattern):
    # Keep the experimental tree-preserving producer inside the existing
    # component entry; it is not selected by production norm dispatch yet.
    from rtp_llm.models_py.modules.dsv41.ppu.test import test_ptpc_norm_quant

    tests.addTests(loader.loadTestsFromModule(test_ptpc_norm_quant))
    return tests


if __name__ == "__main__":
    unittest.main()
