"""Qualify the upstream packed-scale dense interface on real SM120 hardware."""

import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from rtp_llm.models_py.kernels.cuda.fp8_kernel.fp8_kernel import (
    create_per_token_group_quant_fp8_output_scale,
)
from rtp_llm.models_py.modules.dsv4 import utils
from rtp_llm.models_py.modules.dsv4.v41_mxfp8 import quantize_group32


class V41UpstreamLinearSM120Test(unittest.TestCase):
    def setUp(self):
        self.assertTrue(torch.cuda.is_available())
        self.assertEqual(torch.cuda.get_device_capability(), (12, 0))
        torch.manual_seed(4108)
        previous = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        self.addCleanup(setattr, torch.backends.cuda.matmul, "allow_tf32", previous)

    @staticmethod
    def weights(n=160, k=160):
        weight = torch.randn(n, k, device="cuda").to(torch.float8_e4m3fn)
        scales = torch.randint(
            116, 121, (n // 32, k // 32), device="cuda", dtype=torch.uint8
        ).view(torch.float8_e8m0fnu)
        return weight, scales

    @staticmethod
    def unpack(scales, k):
        # Arithmetic extraction is independent of the adapter's byte view.
        return (
            torch.stack([(scales >> (8 * byte)) & 255 for byte in range(4)], dim=-1)
            .reshape(scales.shape[0], -1)[:, : k // 32]
            .to(torch.uint8)
        )

    def assert_projection(self, op, values, scales, result):
        m = values.numel() // op.K
        a = values.reshape(m, op.K).float()
        a *= torch.exp2(self.unpack(scales, op.K).float() - 127).repeat_interleave(
            32, dim=1
        )
        b = op.weight.float() * torch.exp2(
            op._sm120_linear.scales.float() - 127
        ).repeat_interleave(32, 0).repeat_interleave(32, 1)
        expected = (a @ b.T).reshape(result.shape).bfloat16()
        self.assertTrue(bool(torch.isfinite(result).all()))
        torch.testing.assert_close(result, expected, rtol=8e-3, atol=2e-3)

    def test_scale_allocation_respects_group_size(self):
        for rows in (0, 1, 5):
            for group in (16, 32, 64, 128):
                scales = create_per_token_group_quant_fp8_output_scale(
                    (rows, 640), "cpu", group, True, True, True
                )
                self.assertEqual(scales.shape, (rows, (640 // group + 3) // 4))
                self.assertEqual(scales.stride(), (1, max(1, (rows + 3) // 4 * 4)))

    @torch.no_grad()
    def test_factory_selects_group32_without_mutating_checkpoint(self):
        weight, scales = self.weights()
        weights = {"w": weight, "s": scales}
        for op in (
            utils._v4_fp8_linear(weight, scales),
            utils._v4_fp8_linear_from_dict(weights, "w", "s"),
        ):
            self.assertIsInstance(op, utils.V41MXFP8Linear)
            self.assertIs(op.weight, weight)
            self.assertIsNotNone(op._sm120_linear)
            self.assertFalse(hasattr(op, "quantize_input"))
        self.assertIs(weights["s"], scales)
        v4_weight, _ = self.weights(256, 256)
        v4_scales = torch.ones(2, 2, device="cuda").to(torch.float8_e8m0fnu)
        with patch.object(utils.LinearFactory, "create_linear_from_weights") as factory:
            factory.return_value = object()
            utils._v4_fp8_linear(v4_weight, v4_scales)
            self.assertIs(factory.call_args.args[0]["_s"], v4_scales)

    @torch.no_grad()
    def test_native_group32_bytes_for_both_quantizers(self):
        op = utils.V41MXFP8Linear(*self.weights())
        for kernel in ("legacy", "v2"):
            with patch.dict(os.environ, DSV4_FP8_QUANT_KERNEL=kernel):
                for rows in (1, 5, 129):
                    x = torch.randn(rows, op.K, device="cuda", dtype=torch.bfloat16)
                    expected, expected_scales = quantize_group32(
                        x.cpu(), backend="eager"
                    )
                    values, scales = op._quantize_input(x)
                    torch.testing.assert_close(
                        values.view(torch.uint8).cpu(),
                        expected.view(torch.uint8),
                        rtol=0,
                        atol=0,
                    )
                    torch.testing.assert_close(
                        self.unpack(scales, op.K).cpu(), expected_scales, rtol=0, atol=0
                    )
                    self.assert_projection(
                        op, values, scales, op.forward_quantized(values, scales)
                    )

    @torch.no_grad()
    def test_single_packed_word_uses_dense_byte_strides(self):
        op = utils.V41MXFP8Linear(*self.weights(n=128, k=128))
        for rows in (1, 5):
            x = torch.randn(rows, op.K, device="cuda", dtype=torch.bfloat16)
            values, scales = op._quantize_input(x)
            self.assertTrue(scales.is_contiguous())
            self.assertNotEqual(scales.stride(-1), 1)
            self.assert_projection(op, values, scales, op(x))

    @torch.no_grad()
    def test_decode_prefill_leading_shapes_and_noncontiguous_input(self):
        op = utils.V41MXFP8Linear(*self.weights())
        for shape in ((160,), (1, 160), (2, 4, 160), (129, 160)):
            x = torch.randn(*shape, 2, device="cuda", dtype=torch.bfloat16)[..., 0]
            values, scales = op._quantize_input(x)
            output = torch.empty(*shape[:-1], op.N, device="cuda", dtype=torch.bfloat16)
            self.assertIs(op(x, out=output), output)
            self.assert_projection(op, values, scales, output)

    @torch.no_grad()
    def test_prequantized_input_validation_and_empty_output(self):
        op = utils.V41MXFP8Linear(*self.weights())
        x = torch.randn(5, op.K, device="cuda", dtype=torch.bfloat16)
        values, scales = op._quantize_input(x)
        for bad_scales in (
            scales.contiguous(),
            self.unpack(scales, op.K),
            scales[:, :1],
        ):
            with self.assertRaisesRegex(ValueError, "packed UE8M0"):
                op.forward_quantized(values, bad_scales)
        with self.assertRaisesRegex(ValueError, "share storage"):
            op.forward_quantized(values, scales, out=values.view(torch.bfloat16))
        empty = torch.empty(0, op.K, device="cuda", dtype=torch.float32)
        output = op(empty)
        self.assertEqual(output.shape, (0, op.N))
        self.assertEqual(output.dtype, torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "contiguous BF16"):
            op(empty, out=torch.empty_like(empty))

    @torch.no_grad()
    def test_small_and_zero_inputs_are_finite_in_both_quantizers(self):
        op = utils.V41MXFP8Linear(*self.weights())
        for kernel in ("legacy", "v2"):
            with patch.dict(os.environ, DSV4_FP8_QUANT_KERNEL=kernel):
                for factor in (0, 1e-20, 1e-8, 1e-5):
                    with self.subTest(kernel=kernel, factor=factor):
                        x = (
                            torch.randn(5, op.K, device="cuda", dtype=torch.bfloat16)
                            * factor
                        )
                        values, scales = op._quantize_input(x)
                        self.assertTrue(bool(torch.isfinite(values.float()).all()))
                        self.assert_projection(op, values, scales, op(x))

    @torch.no_grad()
    def test_native_recipe_remains_distinct_from_checkpoint_reference_floor(self):
        op = utils.V41MXFP8Linear(*self.weights())
        x = torch.full((1, op.K), 1e-8, device="cuda", dtype=torch.bfloat16)
        _, reference_scales = quantize_group32(x.cpu(), backend="eager")
        self.assertTrue(bool((reference_scales == 105).all()))
        for kernel, exponent in (("legacy", 94), ("v2", 92)):
            with patch.dict(os.environ, DSV4_FP8_QUANT_KERNEL=kernel):
                values, scales = op._quantize_input(x)
                self.assertTrue(bool((self.unpack(scales, op.K) == exponent).all()))
                self.assert_projection(op, values, scales, op(x))

    @unittest.skipUnless(
        os.environ.get("DSV41_TEST_CHECKPOINT"), "NAS checkpoint opt-in"
    )
    @torch.no_grad()
    def test_checkpoint_attention_and_engram_projections(self):
        from safetensors import safe_open

        root = Path(os.environ["DSV41_TEST_CHECKPOINT"])
        index = json.loads((root / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]

        def load(name):
            with safe_open(root / index[name], framework="pt", device="cpu") as file:
                return file.get_tensor(name).cuda()

        for prefix in ("layers.0.attn.wq_a", "layers.1.engram.wkv"):
            op = utils.V41MXFP8Linear(load(prefix + ".weight"), load(prefix + ".scale"))
            for rows in (1, 17):
                with self.subTest(projection=prefix, rows=rows):
                    x = torch.randn(rows, op.K, device="cuda", dtype=torch.bfloat16)
                    values, scales = op._quantize_input(x)
                    self.assert_projection(op, values, scales, op(x))

    @torch.no_grad()
    def test_graph_replay_requantizes_changed_inputs(self):
        op = utils.V41MXFP8Linear(*self.weights())
        x = torch.randn(5, op.K, device="cuda", dtype=torch.bfloat16)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                op(x)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = op(x)
        initial = x.clone()
        for factor in (0, 1e-8, -8, 2):
            x.copy_(initial * factor)
            expected = op(x)
            graph.replay()
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
            self.assert_projection(op, *op._quantize_input(x), output)


if __name__ == "__main__":
    unittest.main()
