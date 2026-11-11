"""Prepared whole-row PTPC inputs shared by real projection consumers."""

import os
import unittest
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules.dsv41.ppu import linear as linear_module
from rtp_llm.models_py.modules.dsv41.ppu.linear import (
    PPUResidentBF16Linear,
    PPUPTPCLinear,
    prepare_shared_ptpc_input,
)


def reference_quant(value):
    values = value.float()
    maximum = values.abs().amax(-1, keepdim=True).clamp_min(1e-10)
    scale = maximum * 0.0022321429569274187
    payload = (values * (448.0 / maximum)).clamp(-448, 448).to(torch.float8_e4m3fn)
    return payload.contiguous(), scale.contiguous()


def reference_linear(op, value):
    flat = value.reshape(-1, value.shape[-1])
    payload, scale = reference_quant(flat)
    out = (payload.float() @ op.weight.float().T) * scale * op.scale.T
    return out.to(torch.bfloat16).reshape((*value.shape[:-1], op.weight.shape[0]))


def assert_bits(test, actual, expected):
    test.assertEqual(actual.dtype, expected.dtype)
    test.assertEqual(actual.shape, expected.shape)
    if not actual.numel():
        return  # Empty tensors have no bytes; their valid last stride can be zero.
    test.assertTrue(torch.equal(
        actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)
    ))


def make_linear(n=16, k=128, device="cpu"):
    weight = torch.randn(n, k, dtype=torch.bfloat16, device=device).to(torch.float8_e4m3fn)
    scale = torch.rand(n, 1, device=device, dtype=torch.float32) * 0.01 + 0.001
    return PPUPTPCLinear(weight, scale)


class PTPCPreparedCPU(unittest.TestCase):
    def test_pair_bits_leading_shapes_and_independent_reference(self):
        torch.manual_seed(4121)
        op = make_linear()
        for shape in ((128,), (4, 128), (2, 3, 128), (0, 128), (2, 0, 128)):
            with self.subTest(shape=shape):
                value = torch.randn(shape, dtype=torch.bfloat16)
                payload, scale = op.quantize_input(value)
                expected_payload, expected_scale = reference_quant(value)
                assert_bits(self, payload, expected_payload)
                assert_bits(self, scale, expected_scale)
                self.assertTrue(payload.is_contiguous())
                self.assertTrue(scale.is_contiguous())
                expected = reference_linear(op, value)
                assert_bits(self, op(value), expected)
                with patch.object(
                    linear_module, "quantize_ptpc_activation",
                    side_effect=AssertionError("prepared consumer must not quantize"),
                ):
                    assert_bits(self, op.forward_quantized(payload, scale), expected)

    def test_noncontiguous_unquantized_input_preserves_logical_rows(self):
        op = make_linear()
        values = (
            torch.randn(256, dtype=torch.bfloat16)[::2],
            torch.randn(4, 256, dtype=torch.bfloat16)[:, ::2],
            torch.randn(3, 2, 256, dtype=torch.bfloat16).transpose(0, 1)[..., ::2],
        )
        for value in values:
            with self.subTest(shape=value.shape, stride=value.stride()):
                self.assertFalse(value.is_contiguous())
                actual = op.quantize_input(value)
                expected = reference_quant(value)
                for left, right in zip(actual, expected):
                    assert_bits(self, left, right)
                assert_bits(self, op.forward_quantized(*actual), reference_linear(op, value))

    def test_prepared_contract_rejects_dtype_shape_stride_and_device(self):
        op = make_linear()
        payload, scale = op.quantize_input(torch.randn(4, 128, dtype=torch.bfloat16))
        for invalid_payload, invalid_scale, error, message in (
            (payload.float(), scale, TypeError, "E4M3 payload and FP32 scale"),
            (payload, scale.half(), TypeError, "E4M3 payload and FP32 scale"),
            (payload[:, :127].contiguous(), scale, ValueError, "input shape/device"),
            (payload, scale.reshape(4), ValueError, "scale shape"),
            (payload, torch.ones(3, 1), ValueError, "scale shape"),
            (torch.empty((), dtype=torch.float8_e4m3fn), scale, ValueError, "nonempty feature"),
            (torch.empty(4, 128, dtype=torch.float8_e4m3fn, device="meta"), scale,
             ValueError, "input shape/device"),
            (payload, torch.empty(4, 1, device="meta"), ValueError, "share the weight device"),
            (torch.empty(4, 256, dtype=torch.float8_e4m3fn)[:, ::2], scale,
             ValueError, "must be contiguous"),
            (payload, torch.ones(4, 2)[:, ::2], ValueError, "must be contiguous"),
        ):
            with self.subTest(message=message, shape=invalid_payload.shape):
                with self.assertRaisesRegex(error, message):
                    op.forward_quantized(invalid_payload, invalid_scale)
        for value in (torch.empty(()), torch.empty(4, 0), torch.empty(4, 127)):
            with self.assertRaises(ValueError):
                op.quantize_input(value)

    def test_compatible_consumers_quantize_once_and_pair_is_not_cached(self):
        torch.manual_seed(4122)
        first, second = make_linear(), make_linear(n=8)
        value = torch.randn(2, 3, 128, dtype=torch.bfloat16)
        with patch.dict(os.environ, {"RTP_V41_PPU_PTPC_FUSION": "1"}), patch.object(
            first, "quantize_input", wraps=first.quantize_input,
        ) as produce, patch.object(
            second, "quantize_input", side_effect=AssertionError("second consumer quantized"),
        ):
            pair = prepare_shared_ptpc_input(value, (first, second))
            self.assertIsNotNone(pair)
            self.assertEqual(produce.call_count, 1)
            for op in (first, second):
                assert_bits(self, op.forward_quantized(*pair), reference_linear(op, value))
            value.zero_()
            second_pair = prepare_shared_ptpc_input(value, (first, second))
            self.assertEqual(produce.call_count, 2)
            for actual, expected in zip(second_pair, reference_quant(value)):
                assert_bits(self, actual, expected)
            self.assertFalse(torch.equal(pair[0].view(torch.uint8), second_pair[0].view(torch.uint8)))

    def test_helper_declines_rollback_legacy_incompatible_and_empty(self):
        op = make_linear()
        other = make_linear(n=8)
        value = torch.randn(4, 128, dtype=torch.bfloat16)
        legacy = PPUResidentBF16Linear(torch.ones(8, 128, dtype=torch.bfloat16))
        wrong_width = make_linear(k=64)
        wrong_device = PPUPTPCLinear(
            torch.empty(8, 128, dtype=torch.float8_e4m3fn, device="meta"),
            torch.empty(8, 1, device="meta"),
        )
        with patch.dict(os.environ, {"RTP_V41_PPU_PTPC_FUSION": "0"}), patch.object(
            op, "quantize_input", side_effect=AssertionError("rollback must not prepare"),
        ):
            self.assertIsNone(prepare_shared_ptpc_input(value, (op, other)))
        with patch.dict(os.environ, {"RTP_V41_PPU_PTPC_FUSION": "1"}), patch.object(
            op, "quantize_input", side_effect=AssertionError("ineligible consumers must not prepare"),
        ):
            for inputs, consumers in (
                (value, ()), (value, (op,)), (value, (op, legacy)),
                (value, (legacy, op)), (value, (op, wrong_width)),
                (value, (op, wrong_device)), (torch.empty(4, 127), (op, other)),
                (torch.empty(()), (op, other)), (torch.empty(0, 128), (op, other)),
            ):
                self.assertIsNone(prepare_shared_ptpc_input(inputs, consumers))


@unittest.skipUnless(
    os.environ.get("RTP_V41_TEST_PTPC_DEVICE") == "1", "explicit PPU device test"
)
class PTPCPreparedDevice(unittest.TestCase):
    @torch.no_grad()
    def test_shared_pair_native_and_changed_input_graph_exact(self):
        torch.manual_seed(4123)
        first, second = make_linear(1280, 5120, "cuda"), make_linear(512, 5120, "cuda")

        def original(op, value):
            payload, scale = reference_quant(value)
            result = torch.empty(
                value.shape[0], op.weight.shape[0], device=value.device, dtype=torch.bfloat16
            )
            op._gemm((payload, scale), (op.weight, op.scale), result)
            return result

        with patch.dict(os.environ, {"RTP_V41_PPU_PTPC_FUSION": "1"}):
            for rows in (8, 24, 32):
                with self.subTest(rows=rows):
                    value = torch.randn(rows, 5120, device="cuda", dtype=torch.bfloat16)
                    pair = prepare_shared_ptpc_input(value, (first, second))
                    for actual, expected in zip(pair, reference_quant(value)):
                        assert_bits(self, actual, expected)
                    for op in (first, second):
                        assert_bits(self, op.forward_quantized(*pair), original(op, value))
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        prepared = prepare_shared_ptpc_input(value, (first, second))
                        outputs = tuple(op.forward_quantized(*prepared) for op in (first, second))
                    for magnitude in (0.0, 0.125, 2.0):
                        value.copy_(torch.randn_like(value) * magnitude)
                        graph.replay()
                        for actual, expected in zip(prepared, reference_quant(value)):
                            assert_bits(self, actual, expected)
                        for op, actual in zip((first, second), outputs):
                            assert_bits(self, actual, original(op, value))


if __name__ == "__main__":
    unittest.main()
