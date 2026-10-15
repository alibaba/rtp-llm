"""Accuracy tests for groupwise BF16/INT8 quantization and multi-source reduction."""

import itertools
import unittest

import torch

from rtp_llm.models_py.kernels.cuda.fast_bf16_int8 import (
    dequantize,
    dequantize_reduce,
    quantize,
)

GROUP_SIZES = (16, 32, 64, 128)


def reference_quantize(x, group_size):
    """Compute INT8 codes and BF16 group scales using arithmetic nearest-even rounding."""
    groups = x.double().reshape(x.shape[0], -1, group_size)
    # Scale arithmetic uses FP32; FP64 preserves BF16 operand products at integer rounding boundaries.
    maximum = groups.abs().amax(-1).float().clamp_min(1e-10)
    scales = (maximum / 127).to(torch.bfloat16)
    inverse = (1 / scales.float()).to(torch.bfloat16).double().unsqueeze(-1)
    q = (groups.abs() * inverse).round().clamp(0, 127) * groups.sign()
    return q.to(torch.int8).view_as(x), scales


def reference_dequantize(q, scales, group_size):
    """Multiply each INT8 code by its group scale and round the result to BF16."""
    groups = q.float().reshape(*scales.shape, group_size)
    return (groups * scales.float().unsqueeze(-1)).to(torch.bfloat16).view_as(q)


def allocate_codes(inputs, m, k, group_size, packed):
    """Allocate codes/scales for `inputs` sources, separately or as views of aligned packets."""
    if not packed:
        return (
            torch.empty((inputs, m, k), dtype=torch.int8, device="cuda"),
            torch.empty(
                (inputs, m, k // group_size), dtype=torch.bfloat16, device="cuda"
            ),
        )
    # [all codes][all scales][rank padding]; no padding within groups.
    n = m * k
    stride = ((n + 2 * n // group_size + 15) // 16) * 16
    packet = torch.empty((inputs, stride), dtype=torch.uint8, device="cuda")
    q = packet[:, :n].view(torch.int8).view(inputs, m, k)
    scales = packet[:, n : n + 2 * n // group_size].view(torch.bfloat16)
    return q, scales.view(inputs, m, k // group_size)


class FastBf16Int8Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Check that the current device supports NVIDIA SM90+ CUDA kernels."""
        if (
            torch.version.hip is not None
            or not torch.cuda.is_available()
            or torch.cuda.get_device_capability()[0] < 9
        ):
            raise unittest.SkipTest("NVIDIA CUDA SM90+ required")

    def assert_bits_equal(self, actual, expected):
        """Compare tensor shapes, dtypes and raw bits, including signed zero."""
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)
        dtype = torch.int16 if actual.dtype == torch.bfloat16 else torch.int8
        self.assertTrue(torch.equal(actual.view(dtype), expected.view(dtype)))

    def quantize_checked(self, x, q, scales, group_size):
        """Check quantization against the original input and verify input preservation."""
        original_x = x.clone()
        expected_q, expected_s = reference_quantize(original_x, group_size)
        quantize(x, q, scales, group_size=group_size)
        self.assert_bits_equal(x, original_x)
        self.assert_bits_equal(q, expected_q)
        self.assert_bits_equal(scales, expected_s)

    def dequantize_checked(self, q, scales, out, group_size):
        """Check decoding against the original codes/scales and verify input preservation."""
        original_q, original_s = q.clone(), scales.clone()
        expected = reference_dequantize(original_q, original_s, group_size)
        dequantize(q, scales, out, group_size=group_size)
        self.assert_bits_equal(q, original_q)
        self.assert_bits_equal(scales, original_s)
        self.assert_bits_equal(out, expected)

    def test_single_source_group_sizes(self):
        """Check groupwise quantization and single-source decoding across shapes and scale ranges."""
        cases = (
            (group_size, shape)
            for group_size in GROUP_SIZES
            for shape in (
                (1, group_size),
                (3, 3 * group_size),
                (1, 128),
                (7, 256),
                (127, 512),
                (4808, 5120),
            )
        )
        for group_size, (m, k) in cases:
            with self.subTest(group_size=group_size, shape=(m, k)):
                torch.manual_seed(9137)
                x = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
                groups = x.view(-1, group_size)
                if len(groups) > 1:
                    groups[0].zero_()
                    groups[1] *= 1e-12
                if len(groups) > 2:
                    groups[2, 0] = 100
                if len(groups) > 3:
                    groups[3].zero_()
                    groups[3, :5] = torch.tensor(
                        [127, 1.5, 2.5, -1.5, -2.5], device="cuda"
                    )
                q, scales = allocate_codes(1, m, k, group_size, packed=True)
                self.quantize_checked(x, q[0], scales[0], group_size)
                out = torch.empty_like(x)
                self.dequantize_checked(q[0], scales[0], out, group_size)
        for group_size in GROUP_SIZES:
            for max_index in (-1, 0, group_size // 2, group_size - 1):
                with self.subTest(group_size=group_size, max_index=max_index):
                    x = torch.zeros(
                        (1, group_size), dtype=torch.bfloat16, device="cuda"
                    )
                    if max_index >= 0:
                        x.copy_(
                            torch.tensor([1.5, 2.5, -1.5, -2.5], device="cuda")
                            .repeat(group_size // 4)
                            .view_as(x)
                        )
                        x[0, max_index] = 127 if max_index % 2 == 0 else -127
                    q, scales = allocate_codes(
                        1, 1, group_size, group_size, packed=True
                    )
                    self.quantize_checked(x, q[0], scales[0], group_size)
                    out = torch.empty_like(x)
                    self.dequantize_checked(q[0], scales[0], out, group_size)
        # Decoder accuracy covers every signed INT8 code and both scale paths.
        q = torch.arange(-128, 128, device="cuda").to(torch.int8).view(1, 256)
        out = torch.empty_like(q, dtype=torch.bfloat16)
        for group_size, scale in itertools.product(
            GROUP_SIZES, (0.0, 1e-38, 0.001, 1.0, 1000.0, 1e37)
        ):
            with self.subTest(group_size=group_size, scale=scale):
                scales = torch.full(
                    (1, 256 // group_size), scale, dtype=torch.bfloat16, device="cuda"
                )
                self.dequantize_checked(q, scales, out, group_size)

    def test_reduce_group_sizes(self):
        """Check ordered BF16 reduction across group sizes, source counts and storage layouts."""
        cases = itertools.product(
            GROUP_SIZES,
            (1, 2, 3, 4, 8, 16),
            ((1, 128), (7, 256), (127, 5120)),
            (False, True),
        )
        for group_size, inputs, (m, k), packed in cases:
            with self.subTest(
                group_size=group_size, inputs=inputs, shape=(m, k), packed=packed
            ):
                torch.manual_seed(9137)
                q, scales = allocate_codes(inputs, m, k, group_size, packed)
                expected = None
                for source in range(inputs):
                    x = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
                    x *= 2.0 ** (source % 5 - 2)
                    x.view(-1)[:group_size].zero_()
                    # Exact cancellation plus independent scales on other groups.
                    x.view(-1)[group_size : 2 * group_size] = (-1.0) ** source
                    self.quantize_checked(x, q[source], scales[source], group_size)
                    decoded = reference_dequantize(
                        q[source], scales[source], group_size
                    )
                    expected = (
                        decoded
                        if expected is None
                        else (expected.float() + decoded.float()).to(torch.bfloat16)
                    )
                out = torch.empty((m, k), dtype=torch.bfloat16, device="cuda")
                original_q, original_s = q.clone(), scales.clone()
                dequantize_reduce(q, scales, out, group_size=group_size)
                self.assert_bits_equal(q, original_q)
                self.assert_bits_equal(scales, original_s)
                self.assert_bits_equal(out, expected)


if __name__ == "__main__":
    unittest.main()
