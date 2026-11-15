"""Dedicated accelerator target: no backend mocks and no missing-GPU skips."""

import unittest

import torch

from rtp_llm.models_py.quant_methods import fp8
from rtp_llm.models_py.quant_methods.test.test_fp8_per_tensor import (
    config,
    load_tiny,
    reference_quantize,
)


class Fp8PerTensorGpuTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise RuntimeError("Dedicated FP8 GPU test requires an accelerator")
        cls.device = torch.device("cuda:0")
        cls.fp8_dtype = fp8._runtime_dtype(cls.device)

    def test_native_quantization_against_reference(self):
        for dtype in (torch.float16, torch.bfloat16):
            for rows in (1, 7, 129):
                for static in (False, True):
                    with self.subTest(dtype=dtype, rows=rows, static=static):
                        x = torch.linspace(
                            -3, 3, rows * 32, device=self.device, dtype=dtype
                        ).reshape(rows, 32)
                        scale = (
                            torch.tensor([0.025], device=self.device)
                            if static
                            else None
                        )
                        expected, expected_scale = reference_quantize(
                            x, self.fp8_dtype, scale
                        )
                        actual, actual_scale = fp8._quantize(x, self.fp8_dtype, scale)
                        torch.testing.assert_close(
                            actual_scale, expected_scale, rtol=1e-5, atol=1e-7
                        )
                        # Allow one FP8 rounding step at a tie, not arbitrary
                        # errors hidden by comparing only shape or scale.
                        torch.testing.assert_close(
                            actual.float() * actual_scale,
                            expected.float() * expected_scale,
                            rtol=0.07,
                            atol=0.002,
                        )
                        self.assertEqual(actual.dtype, self.fp8_dtype)
                        self.assertEqual(actual.shape, x.shape)
                        self.assertTrue(torch.isfinite(actual.float()).all())

    def test_real_load_and_scaled_mm_online_prequantized_static(self):
        # Exercise the complete checkpoint -> device conversion -> native GEMM path.
        # Both runtime output types use their actual construction dtype.
        for output_dtype in (torch.float16, torch.bfloat16):
            for mode in ("online", "dynamic", "static"):
                with self.subTest(dtype=output_dtype, mode=mode):
                    source = (torch.arange(256).reshape(16, 16).remainder(11) - 5).to(
                        torch.bfloat16
                    ) / 8
                    if mode == "online":
                        weights = {"proj.weight": source}
                    else:
                        qweight, scale = reference_quantize(source, torch.float8_e4m3fn)
                        weights = {"proj.weight": qweight, "proj.weight_scale": scale}
                        if mode == "static":
                            weights["proj.input_scale"] = torch.tensor([0.025])
                    model = load_tiny(
                        weights,
                        config(online=mode == "online", dynamic=mode != "static"),
                        str(self.device),
                        compute_dtype=output_dtype,
                    )
                    x = torch.linspace(
                        -2, 2, 5 * 16, device=self.device, dtype=output_dtype
                    ).reshape(5, 16)
                    scale = model.proj.input_scale if mode == "static" else None
                    qx, sx = reference_quantize(x, self.fp8_dtype, scale)
                    reference = (qx.float() * sx) @ (
                        model.proj.weight.float() * model.proj.weight_scale
                    ).t()
                    actual = model(x)
                    self.assertEqual(actual.dtype, output_dtype)
                    torch.testing.assert_close(
                        actual.float(), reference, rtol=0.08, atol=0.03
                    )
                    torch.testing.assert_close(model(x[:0]), x.new_empty(0, 16))

    def test_zero_and_empty_inputs(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                zeros = torch.zeros(2, 16, dtype=dtype, device=self.device)
                output, scale = fp8._quantize(zeros, self.fp8_dtype)
                self.assertTrue(torch.isfinite(output.float()).all())
                self.assertTrue(torch.isfinite(scale).all())
                torch.testing.assert_close(output.float() * scale, zeros.float())
                empty, empty_scale = fp8._quantize(zeros[:0], self.fp8_dtype)
                self.assertEqual(empty.shape, (0, 16))
                self.assertEqual(empty.dtype, self.fp8_dtype)
                self.assertEqual(empty_scale.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
