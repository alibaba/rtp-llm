import itertools
from typing import Optional, Tuple
from unittest import SkipTest, TestCase, main

import torch

import rtp_llm.ops  # isort:skip
from rtp_llm.ops.compute_ops import per_tensor_quant_fp8  # isort:skip


class PerTensorFp8QuantFusedTest(TestCase):
    NUM_TOKENS = [128, 256, 512]
    HIDDEN_SIZES = [128, 768, 769, 770, 5120, 8192]
    INPUT_DTYPES = [torch.float16, torch.bfloat16, torch.float32]
    # Allow FP32 division/reciprocal rounding differences; zero remains exact.
    SCALE_RTOL = 2 * torch.finfo(torch.float32).eps

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is not available")
        torch.manual_seed(2026)

    def torch_scaled_fp8_quant(
        self, tensor: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        finfo = torch.finfo(torch.float8_e4m3fn)
        multiplier = scale.clamp_min(1e-9).reciprocal()
        return (
            (tensor.float() * multiplier)
            .clamp(min=finfo.min, max=finfo.max)
            .to(torch.float8_e4m3fn)
        )

    def call_per_tensor_quant_fp8(
        self,
        input: torch.Tensor,
        scale: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        output = torch.empty_like(input, dtype=torch.float8_e4m3fn)
        is_static = scale is not None
        if scale is None:
            # Dynamic quantization must initialize its own scale.
            scale = torch.full(
                (1,), float("nan"), device=input.device, dtype=torch.float32
            )
        per_tensor_quant_fp8(input, output, scale, is_static)
        return output, scale

    def assert_dynamic_scale(self, input: torch.Tensor, scale: torch.Tensor) -> None:
        expected = input.detach().cpu().float().abs().max().reshape(1) / 448.0
        torch.testing.assert_close(
            scale.detach().cpu().reshape(1),
            expected,
            rtol=self.SCALE_RTOL,
            atol=0,
        )

    def assert_quantized_output(
        self, input: torch.Tensor, output: torch.Tensor, scale: torch.Tensor
    ) -> None:
        # Use the returned scale, matching the original test reference.
        expected = self.torch_scaled_fp8_quant(input, scale)
        torch.testing.assert_close(
            output.contiguous().view(torch.uint8).cpu(),
            expected.contiguous().view(torch.uint8).cpu(),
            rtol=0,
            atol=0,
            msg="FP8 output bytes differ",
        )

    def test_per_tensor_fp8_quant(self):
        for num_tokens, hidden_size, dtype in itertools.product(
            self.NUM_TOKENS, self.HIDDEN_SIZES, self.INPUT_DTYPES
        ):
            with self.subTest(
                num_tokens=num_tokens, hidden_size=hidden_size, dtype=dtype
            ):
                x = torch.randn(
                    (num_tokens, hidden_size), device="cuda", dtype=dtype
                )
                output, scale = self.call_per_tensor_quant_fp8(x)
                self.assert_dynamic_scale(x, scale)
                self.assert_quantized_output(x, output, scale)
                static_scale = torch.tensor([0.1], device="cuda")
                original_scale = static_scale.clone()
                output, scale = self.call_per_tensor_quant_fp8(x, static_scale)
                torch.testing.assert_close(scale, original_scale, rtol=0, atol=0)
                self.assert_quantized_output(x, output, scale)

    def test_dynamic_small_inputs_and_vector_tail(self):
        # All lengths satisfy the binding's 16-byte input alignment constraint;
        # 8/24 BF16 or FP16 and 4/12 FP32 also exercise the quantizer's tail.
        for dtype in self.INPUT_DTYPES:
            alignment = 16 // torch.empty((), dtype=dtype).element_size()
            for length in (alignment, 3 * alignment, 264):
                with self.subTest(dtype=dtype, length=length):
                    x = torch.zeros(length, device="cuda", dtype=dtype)
                    # Ensure the maximum is in the tail, not the vector prefix.
                    x[-1] = -7.0
                    output, scale = self.call_per_tensor_quant_fp8(x)
                    self.assert_dynamic_scale(x, scale)
                    self.assert_quantized_output(x, output, scale)

    def test_dynamic_scale_reciprocal_rounding(self):
        x = torch.full((8,), 255 / 256, device="cuda", dtype=torch.bfloat16)
        output, scale = self.call_per_tensor_quant_fp8(x)
        self.assert_dynamic_scale(x, scale)
        self.assert_quantized_output(x, output, scale)

    def test_dynamic_scale_overwrites_previous_value(self):
        x = torch.randn((8, 1024), device="cuda", dtype=torch.bfloat16)
        output = torch.empty_like(x, dtype=torch.float8_e4m3fn)
        scale = torch.full((1,), float("nan"), device="cuda")
        # Exercise repeated calls with decreasing, zero, and increasing inputs.
        for magnitude in (16.0, 0.25, 0.0, 8.0):
            with self.subTest(magnitude=magnitude):
                value = x * magnitude
                per_tensor_quant_fp8(value, output, scale, False)
                self.assert_dynamic_scale(value, scale)
                self.assert_quantized_output(value, output, scale)

    def test_per_tensor_fp8_static_quant_precision(self):
        values = [
            -500,
            -3.5,
            -2.5,
            -1.5,
            -0.5,
            0.0,
            0.5,
            1.5,
            2.5,
            3.5,
            500,
            2**-10,
            2**-9,
            3 * 2**-10,
            -2**-9,
            448,
        ]
        for dtype, scale_value in itertools.product(
            self.INPUT_DTYPES, (0.1, 2.5, 22.5, 222.0, 5012.0)
        ):
            with self.subTest(dtype=dtype, scale=scale_value):
                # Bound FP16 inputs to finite values while preserving saturation
                # coverage through the other input dtypes and smaller scales.
                x = torch.tensor(values, device="cuda", dtype=torch.float32)
                x = x * scale_value
                if dtype == torch.float16:
                    x = x.clamp(-65504, 65504)
                x = x.to(dtype)
                scale = torch.tensor([scale_value], device="cuda")
                original_scale = scale.clone()
                output, scale = self.call_per_tensor_quant_fp8(x, scale)
                torch.testing.assert_close(scale, original_scale, rtol=0, atol=0)
                self.assert_quantized_output(x, output, scale)

        x = torch.tensor([2**31, -(2**31), 0, 1], device="cuda", dtype=torch.float32)
        scale = torch.ones(1, device="cuda")
        output, scale = self.call_per_tensor_quant_fp8(x, scale)
        self.assert_quantized_output(x, output, scale)

    def test_static_scale_safety_clamp(self):
        x = torch.tensor(
            [0, 1e-10, -1e-10, 1e-9, -1e-9, 1, -1, 2],
            device="cuda",
            dtype=torch.float32,
        )
        for scale_value in (0.0, 1e-12, 1e-9):
            with self.subTest(scale=scale_value):
                scale = torch.tensor([scale_value], device="cuda")
                original_scale = scale.clone()
                output, scale = self.call_per_tensor_quant_fp8(x, scale)
                torch.testing.assert_close(scale, original_scale, rtol=0, atol=0)
                self.assert_quantized_output(x, output, scale)

    def test_dynamic_scale_cuda_graph_replay(self):
        x = torch.ones((8, 1024), device="cuda", dtype=torch.bfloat16)
        output = torch.empty_like(x, dtype=torch.float8_e4m3fn)
        scale = torch.empty(1, device="cuda")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                per_tensor_quant_fp8(x, output, scale, False)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            per_tensor_quant_fp8(x, output, scale, False)
        for magnitude in (8.0, 0.125, 0.0, -16.0):
            with self.subTest(magnitude=magnitude):
                x.fill_(magnitude)
                graph.replay()
                self.assert_dynamic_scale(x, scale)
                self.assert_quantized_output(x, output, scale)


if __name__ == "__main__":
    main()
