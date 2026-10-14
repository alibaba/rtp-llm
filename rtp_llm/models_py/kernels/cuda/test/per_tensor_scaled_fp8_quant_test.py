import itertools
from typing import Optional, Tuple
from unittest import SkipTest, TestCase, main

import torch
from torch import dtype as _dtype

import rtp_llm.ops  # isort:skip
from rtp_llm.ops.compute_ops import per_tensor_quant_fp8  # isort:skip


class PerTensorFp8QuantTest(TestCase):
    NUM_TOKENS = [128, 256, 512]
    HIDDEN_SIZES = [128, 768, 769, 770, 5120, 8192]

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is not available")
        torch.set_default_device("cuda")

    def torch_scaled_fp8_quant(self, tensor, inv_scale):
        # The reference implementation that fully aligns to
        # the kernel being tested.
        finfo = torch.finfo(torch.float8_e4m3fn)
        scale = inv_scale.reciprocal()
        q_tensor = (tensor.to(torch.float32) * scale).clamp(
            min=finfo.min, max=finfo.max
        )
        return q_tensor.to(torch.float8_e4m3fn)

    def call_per_tensor_quant_fp8(
        self,
        input: torch.Tensor,
        scale: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        output = torch.empty_like(input, device=input.device, dtype=torch.float8_e4m3fn)
        is_static = True
        if scale is None:
            scale = torch.zeros(1, device=input.device, dtype=torch.float32)
            is_static = False
        # kernel call
        per_tensor_quant_fp8(input, output, scale, is_static)
        return output, scale

    def _run_per_tensor_fp8_quant_test(self, num_tokens: int, hidden_size: int):
        device = torch.device("cuda")
        x = torch.rand((num_tokens, hidden_size), dtype=torch.bfloat16, device=device)

        # dynamic quant
        q_out, scale = self.call_per_tensor_quant_fp8(x)
        torch_out = self.torch_scaled_fp8_quant(x, scale)

        self.assertTrue(torch.equal(q_out.float(), torch_out.float()))
        # static quant
        scale = torch.rand(1, dtype=torch.float32, device=device)
        q_out, scale = self.call_per_tensor_quant_fp8(x, scale)
        torch_out = self.torch_scaled_fp8_quant(x, scale)
        self.assertTrue(torch.equal(q_out.float(), torch_out.float()))

    def _run_per_tensor_fp8_static_quant_precision_test(self):
        test_scales = [0.1, 2.5, 22.5, 222, 5012]
        device = torch.device("cuda")
        for scale in test_scales:
            x1 = (
                torch.tensor(
                    [-0.5, -1.5, -2.5, -3.5, 0.5, 1.5, 2.5, 3.5],
                    dtype=torch.bfloat16,
                    device=device,
                )
                * scale
            )
            x2 = torch.tensor([-500, 500], dtype=torch.bfloat16, device=device) * scale
            x3 = (
                torch.tensor(
                    [2**-9 - 2**-10, 2**-9, 2**-9 + 2**-10],
                    dtype=torch.bfloat16,
                    device=device,
                )
                * scale
            )
            for x in (x1, x2, x3):
                scale = torch.tensor(scale, device=device, dtype=torch.float32)
                q_out, scale = self.call_per_tensor_quant_fp8(x, scale)
                torch_out = self.torch_scaled_fp8_quant(x, scale)
                self.assertTrue(torch.equal(q_out.float(), torch_out.float()))

        x = torch.tensor([2**31, -(2**31)], dtype=torch.float32, device=device)
        scale = torch.tensor([1.0], device=device, dtype=torch.float32)
        q_out, scale = self.call_per_tensor_quant_fp8(x, scale)
        torch_out = self.torch_scaled_fp8_quant(x, scale)
        self.assertTrue(torch.equal(q_out.float(), torch_out.float()))

    def test_per_tensor_fp8_quant(self):
        for params in itertools.product(
            self.NUM_TOKENS,
            self.HIDDEN_SIZES,
        ):
            with self.subTest(
                num_tokens=params[0],
                hidden_size=params[1],
            ):
                self._run_per_tensor_fp8_quant_test(*params)
            self._run_per_tensor_fp8_static_quant_precision_test()

    def test_padded_rows_preserve_quantized_bytes_and_scale(self):
        for rows, width in itertools.product((1, 3, 8, 17, 4093), (320, 2560)):
            x = torch.randn(rows, width, dtype=torch.bfloat16, device="cuda")
            output = torch.empty(
                ((rows + 15) // 16 * 16, width),
                dtype=torch.float8_e4m3fn,
                device="cuda",
            )
            scale = torch.zeros(1, dtype=torch.float32, device="cuda")
            per_tensor_quant_fp8(x, output, scale, False)
            expected, expected_scale = self.call_per_tensor_quant_fp8(x)
            self.assertTrue(torch.equal(scale, expected_scale))
            self.assertTrue(
                torch.equal(output[:rows].view(torch.uint8), expected.view(torch.uint8))
            )
            self.assertEqual(output[rows:].view(torch.uint8).count_nonzero().item(), 0)
            # Static scaling uses the same padded-output contract.
            output.view(torch.uint8).fill_(255)
            per_tensor_quant_fp8(x, output, expected_scale, True)
            self.assertTrue(
                torch.equal(output[:rows].view(torch.uint8), expected.view(torch.uint8))
            )
            self.assertEqual(output[rows:].view(torch.uint8).count_nonzero().item(), 0)


if __name__ == "__main__":
    main()
