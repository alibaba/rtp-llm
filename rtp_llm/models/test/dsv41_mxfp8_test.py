"""Portable group-32 equations and mocked provider ABI, not GPU qualification."""

import math
import os
import sys
import unittest
from contextlib import nullcontext
from types import ModuleType
from unittest.mock import Mock, patch

import torch

from rtp_llm.models.test.dsv41_cache_codec_test import (
    FP8_LEVELS,
    _f32,
    _nearest,
    load_codec_source,
)

linear = load_codec_source("_dsv41_mxfp8_test", "v41_mxfp8.py")


def quantization_reference(x):
    """Scalar arithmetic and nearest-even search, with no Torch FP8 conversion."""
    payload, encoded = [], []
    for group in x.float().reshape(-1, 32).tolist():
        maximum = max(max(abs(v) for v in group), _f32(1e-4))
        scaled = _f32(maximum * _f32(1.0 / 448.0))
        fraction, exponent = math.frexp(scaled)
        power = exponent - (fraction == 0.5)
        encoded.append(power + 127)
        for value in group:
            magnitude = min(_f32(abs(value) / 2.0**power), 448.0)
            sign = 128 if math.copysign(1.0, value) < 0 else 0
            payload.append(sign | _nearest(magnitude, FP8_LEVELS))
    return (
        torch.tensor(payload, dtype=torch.uint8).reshape(x.shape),
        torch.tensor(encoded, dtype=torch.uint8).reshape(-1, x.shape[-1] // 32),
    )


def grouped_reference(values, scales, weight, weight_scales):
    """Separate K groups and channel-by-channel checkpoint scale selection."""
    flat = values.float().reshape(-1, weight.shape[1])
    result = torch.zeros((flat.shape[0], weight.shape[0]), dtype=torch.float32)
    for channel in range(weight.shape[0]):
        for group in range(weight.shape[1] // 32):
            offset = group * 32
            partial = (
                flat[:, offset : offset + 32]
                * weight[channel, offset : offset + 32].float()
            ).sum(-1)
            result[:, channel] += (
                partial
                * torch.exp2(scales[:, group].float() - 127)
                * 2.0 ** (int(weight_scales[channel // 32, group]) - 127)
            )
    return result.to(torch.bfloat16).reshape(*values.shape[:-1], weight.shape[0])


def weights(n=160, k=160):
    generator = torch.Generator().manual_seed(4132)
    weight = torch.randint(-8, 9, (n, k), generator=generator).to(torch.float8_e4m3fn)
    scales = torch.randint(
        121, 128, ((n + 31) // 32, k // 32), generator=generator, dtype=torch.uint8
    )
    return weight, scales


class V41QuantizationTest(unittest.TestCase):
    def assert_recipe(self, x):
        values, scales = linear.quantize_group32(x, backend="eager")
        expected, expected_scales = quantization_reference(x)
        torch.testing.assert_close(values.view(torch.uint8), expected, rtol=0, atol=0)
        torch.testing.assert_close(scales, expected_scales, rtol=0, atol=0)

    def test_floor_signed_zero_and_finite_extremes(self):
        x = torch.zeros(5, 64)
        x[0, 1] = -0.0
        x[1] = 1e-5
        x[2] = -1e-4
        x[3, :32] = torch.finfo(torch.float32).max
        x[3, 32:] = -torch.finfo(torch.float32).max
        x[4] = torch.finfo(torch.float32).tiny
        self.assert_recipe(x)
        values, scales = linear.quantize_group32(x, backend="eager")
        self.assertEqual(int(scales[0, 0]), 105)
        self.assertEqual(int(values.view(torch.uint8)[0, 1]), 128)

    def test_all_fp8_rounding_midpoints_and_power_boundaries(self):
        midpoints = [(a + b) / 2 for a, b in zip(FP8_LEVELS, FP8_LEVELS[1:])]
        x = torch.zeros(len(midpoints) * 2, 32)
        x[:, -1] = 448  # Set scale to one while testing all halfway values.
        x[:, 0] = torch.tensor(midpoints + [-v for v in midpoints])
        self.assert_recipe(x)
        maxima = torch.tensor([448 * 2.0**p for p in (-20, -10, -1, 0, 1, 10, 100)])
        around = torch.stack(
            (
                torch.nextafter(maxima, torch.zeros_like(maxima)),
                maxima,
                torch.nextafter(maxima, torch.full_like(maxima, float("inf"))),
            )
        )
        self.assert_recipe(around.reshape(-1, 1).expand(-1, 32).contiguous())

    def test_group_independence_leading_dimensions_and_dtypes(self):
        x = torch.randn(3, 5, 96, generator=torch.Generator().manual_seed(32))
        x[..., :32] *= 0.00001
        x[..., 64:] *= 8192
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            self.assert_recipe(x.to(dtype))
        values, scales = linear.quantize_group32(torch.empty(2, 0, 96), backend="eager")
        self.assertEqual(values.shape, (2, 0, 96))
        self.assertEqual(scales.shape, (0, 3))

    def test_invalid_quantization_contracts(self):
        for x in (
            torch.ones(32),
            torch.ones(2, 33),
            torch.empty(2, 0),
            torch.ones(2, 32).double(),
            torch.ones(32, 2).T,
            torch.full((2, 32), float("nan")),
            torch.full((2, 32), float("inf")),
            torch.ones(2, 32, requires_grad=True),
        ):
            with self.assertRaises(ValueError):
                linear.quantize_group32(x, backend="eager")
        with self.assertRaisesRegex(ValueError, "CUDA"):
            linear.quantize_group32(torch.ones(2, 32))

    def test_triton_interpreter_representable_values_and_scale_boundaries(self):
        self.assertEqual(os.environ.get("TRITON_INTERPRET"), "1")
        kernel = load_codec_source("_dsv41_mxfp8_interpreter", "_v41_mxfp8_triton.py")
        x = torch.tensor([0, -0.0, 1, -2, 7, 448, 1e-4, 448 + 2**-15, 448 - 2**-15])
        x = x[:, None].expand(-1, 32).contiguous()
        # Interpreter arbitrary FP8 casts are not reliable; test that on hardware.
        x[6, 1:] = 0
        x[7:, 1:] = 0
        values, scales = kernel.quantize_rows(x)
        expected, expected_scales = quantization_reference(x)
        torch.testing.assert_close(scales, expected_scales, rtol=0, atol=0)
        torch.testing.assert_close(
            values[:6].view(torch.uint8), expected[:6], rtol=0, atol=0
        )


class V41LinearTest(unittest.TestCase):
    def test_grouped_gemm_channel_blocks_and_output_tiling(self):
        for n in (35, 160):
            weight, scales = weights(n=n, k=96)
            op = linear.V41MXFP8Linear(weight, scales, backend="eager")
            x = torch.randn(7, 5, 96, generator=torch.Generator().manual_seed(n))
            q, s = linear.quantize_group32(x, backend="eager")
            expected = grouped_reference(q, s, weight, scales)
            output = torch.empty(7, 5, n, dtype=torch.bfloat16)
            self.assertIs(op(x, out=output), output)
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
            torch.testing.assert_close(
                op.forward_quantized(q, s), expected, rtol=0, atol=0
            )
            self.assertIs(op.weight, weight)
            self.assertEqual(list(op.buffers()), [])

    def test_empty_rows_and_special_scale_bytes(self):
        weight = torch.ones(32, 32).to(torch.float8_e4m3fn)
        op = linear.V41MXFP8Linear(
            weight, torch.zeros(1, 1, dtype=torch.uint8), backend="eager"
        )
        empty = op(torch.empty(2, 0, 32))
        self.assertEqual(empty.shape, (2, 0, 32))
        q = torch.ones(2, 32).to(torch.float8_e4m3fn)
        s = torch.tensor([[127], [255]], dtype=torch.uint8)
        actual = op.forward_quantized(q, s)
        torch.testing.assert_close(
            actual[0].float(), torch.full((32,), 32 * 2.0**-127), rtol=0, atol=0
        )
        self.assertTrue(bool(torch.isnan(actual[1]).all()))

    def test_constructor_rejects_other_scale_abis(self):
        weight, scales = weights()
        for bad in (
            scales.int(),
            torch.zeros(2, 2, dtype=torch.uint8),
            scales.T,
            scales.flatten(),
        ):
            with self.assertRaises(ValueError):
                linear.V41MXFP8Linear(weight, bad, backend="eager")
        with self.assertRaisesRegex(ValueError, "SM120"):
            linear.V41MXFP8Linear(weight, scales)

    def test_input_validation_preserves_caller_output(self):
        weight, scales = weights()
        op = linear.V41MXFP8Linear(weight, scales, backend="eager")
        out = torch.full((2, 160), 42, dtype=torch.bfloat16)
        for x in (
            torch.ones(2, 128),
            torch.ones(2, 160).double(),
            torch.ones(160, 2).T,
            torch.ones(2, 160, requires_grad=True),
        ):
            with self.assertRaises(ValueError):
                op(x, out=out)
        q = torch.ones(2, 160).to(torch.float8_e4m3fn)
        for s in (
            torch.ones(2, 5, dtype=torch.int32),
            torch.ones(2, 4, dtype=torch.uint8),
        ):
            with self.assertRaises(ValueError):
                op.forward_quantized(q, s, out=out)
        torch.testing.assert_close(out, torch.full_like(out, 42))

    def test_output_aliases_and_dtypes_rejected(self):
        weight, scales = weights()
        op = linear.V41MXFP8Linear(weight, scales, backend="eager")
        x = torch.ones(2, 160, dtype=torch.bfloat16)
        for out in (x, x.float(), torch.empty(160, 2, dtype=torch.bfloat16).T):
            with self.assertRaises(ValueError):
                op(x, out=out)
        q = torch.ones(2, 160).to(torch.float8_e4m3fn)
        s = torch.ones(2, 5, dtype=torch.uint8)
        alias = weight.view(torch.bfloat16).flatten()[:320].view(2, 160)
        with self.assertRaisesRegex(ValueError, "share"):
            op.forward_quantized(q, s, out=alias)


class V41ProviderContractTest(unittest.TestCase):
    def setUp(self):
        self.capture = False
        self.flashinfer = ModuleType("flashinfer")
        self.gemm = ModuleType("flashinfer.gemm")
        self.flashinfer.block_scale_interleave = Mock(side_effect=self.interleave)
        self.gemm.mm_mxfp8 = Mock(side_effect=lambda *a, **kw: kw["out"].zero_())
        for patcher in (
            patch.dict(
                sys.modules,
                {"flashinfer": self.flashinfer, "flashinfer.gemm": self.gemm},
            ),
            patch.object(linear, "_require_sm120"),
            patch.object(torch.cuda, "device", side_effect=lambda *_: nullcontext()),
            patch.object(
                torch.cuda,
                "is_current_stream_capturing",
                side_effect=lambda: self.capture,
            ),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    @staticmethod
    def interleave(scales):
        rows, columns = scales.shape
        return torch.zeros(
            ((rows + 127) // 128 * 128) * ((columns + 3) // 4 * 4), dtype=torch.uint8
        )

    def test_weight_expansion_once_and_explicit_provider_contract(self):
        weight, scales = weights()
        op = linear.V41MXFP8Linear(weight, scales)
        expanded = self.flashinfer.block_scale_interleave.call_args.args[0]
        torch.testing.assert_close(expanded, scales.repeat_interleave(32, dim=0))
        q = torch.ones(3, 160).to(torch.float8_e4m3fn)
        s = torch.full((3, 5), 127, dtype=torch.uint8)
        for _ in range(2):
            op.forward_quantized(q, s)
        self.assertEqual(self.flashinfer.block_scale_interleave.call_count, 3)
        args, kwargs = self.gemm.mm_mxfp8.call_args
        self.assertEqual(args[1].shape, (160, 160))
        self.assertEqual(args[1].stride(), (1, 160))
        self.assertEqual(args[1].data_ptr(), weight.data_ptr())
        self.assertEqual(kwargs["backend"], "cutlass")
        self.assertFalse(kwargs["use_8x4_sf_layout"])
        self.assertEqual(kwargs["out_dtype"], torch.bfloat16)

    def test_provider_failure_does_not_fallback_or_mark_warmed(self):
        op = linear.V41MXFP8Linear(*weights())
        self.gemm.mm_mxfp8.side_effect = RuntimeError("provider failure")
        with self.assertRaisesRegex(RuntimeError, "provider failure"):
            op.forward_quantized(
                torch.ones(2, 160).to(torch.float8_e4m3fn),
                torch.ones(2, 5, dtype=torch.uint8),
            )
        self.assertEqual(op._warmed_rows, set())

    def test_invalid_interleave_output_is_rejected(self):
        op = linear.V41MXFP8Linear(*weights())
        self.flashinfer.block_scale_interleave.side_effect = lambda _: torch.zeros(
            1, dtype=torch.uint8
        )
        with self.assertRaisesRegex(ValueError, "swizzled"):
            op.forward_quantized(
                torch.ones(2, 160).to(torch.float8_e4m3fn),
                torch.ones(2, 5, dtype=torch.uint8),
            )
        self.gemm.mm_mxfp8.assert_not_called()

    def test_graph_admission_precedes_quantization_for_new_dtype_or_rows(self):
        op = linear.V41MXFP8Linear(*weights())
        eager = linear.quantize_group32
        with patch.object(
            linear,
            "quantize_group32",
            side_effect=lambda x, **_: eager(x, backend="eager"),
        ) as quantize:
            op(torch.ones(3, 160, dtype=torch.bfloat16))
            self.capture = True
            op(torch.ones(3, 160, dtype=torch.bfloat16))
            quantize.reset_mock()
            for x in (
                torch.ones(3, 160),
                torch.ones(4, 160, dtype=torch.bfloat16),
                torch.ones(481, dtype=torch.bfloat16)[1:].view(3, 160),
            ):
                with self.assertRaisesRegex(RuntimeError, "warmed"):
                    op(x)
            quantize.assert_not_called()
        with self.assertRaisesRegex(RuntimeError, "warmed"):
            op.forward_quantized(
                torch.ones(4, 160).to(torch.float8_e4m3fn),
                torch.ones(4, 5, dtype=torch.uint8),
            )
        with self.assertRaisesRegex(RuntimeError, "preparation"):
            linear.V41MXFP8Linear(*weights())


if __name__ == "__main__":
    unittest.main()
