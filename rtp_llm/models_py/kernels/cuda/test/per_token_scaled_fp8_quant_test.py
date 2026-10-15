import itertools
from typing import Optional, Tuple
from unittest import SkipTest, TestCase, main

import torch
from torch import dtype as _dtype

import rtp_llm.ops  # isort:skip
from rtp_llm.ops.compute_ops import per_token_quant_fp8  # isort:skip


class PerTokenFp8QuantTest(TestCase):
    NUM_TOKENS = [128, 256, 512]
    HIDDEN_SIZES = [128, 768, 1024, 2048, 4096, 8192]

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is not available")
        torch.set_default_device("cuda")

    def torch_scaled_fp8_quant(self, tensor, inv_scale):
        finfo = torch.finfo(torch.float8_e4m3fn)
        inv_scale = inv_scale.view(-1, 1)
        scale = inv_scale.reciprocal()
        q_tensor = (tensor.to(torch.float32) * scale).clamp(
            min=finfo.min, max=finfo.max
        )
        return q_tensor.to(torch.float8_e4m3fn)

    def call_per_token_quant_fp8(
        self,
        input: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        output = torch.empty_like(input, device=input.device, dtype=torch.float8_e4m3fn)
        scale = torch.zeros(input.size(0), device=input.device, dtype=torch.float32)
        per_token_quant_fp8(input, output, scale)
        scale = scale.reshape(-1, 1)
        return output, scale

    def _run_per_token_fp8_quant_test(self, num_tokens: int, hidden_size: int):
        device = torch.device("cuda")
        x = torch.rand((num_tokens, hidden_size), dtype=torch.bfloat16, device=device)

        # dynamic quant
        q_out, scale = self.call_per_token_quant_fp8(x)
        torch_out = self.torch_scaled_fp8_quant(x, scale)
        self.assertTrue(torch.equal(q_out.float(), torch_out.float()))

    def test_per_token_fp8_quant(self):
        for params in itertools.product(
            self.NUM_TOKENS,
            self.HIDDEN_SIZES,
        ):
            with self.subTest(
                num_tokens=params[0],
                hidden_size=params[1],
            ):
                self._run_per_token_fp8_quant_test(*params)

    def test_zero_rows_and_independent_warp_scales(self):
        # This size forces the eight-warps-per-CTA path. Adjacent tokens have
        # deliberately different ranges so a CTA-shared scale race is visible.
        sm_count = torch.cuda.get_device_properties(0).multi_processor_count
        num_tokens = sm_count * 16
        hidden_size = 128
        token_scales = torch.pow(
            torch.tensor(2.0, device="cuda"),
            (torch.arange(num_tokens, device="cuda") % 15 - 7).float(),
        )
        x = torch.linspace(-1.0, 1.0, hidden_size, device="cuda").repeat(num_tokens, 1)
        x = (x * token_scales.unsqueeze(1)).to(torch.bfloat16)
        x[0].zero_()
        x[9].zero_()

        q_out, scales = self.call_per_token_quant_fp8(x)
        max_abs = x.float().abs().amax(dim=1, keepdim=True)
        expected_scales = torch.where(
            max_abs == 0, torch.ones_like(max_abs), max_abs / 448.0
        )
        expected_q = self.torch_scaled_fp8_quant(x, expected_scales)

        torch.testing.assert_close(scales, expected_scales, rtol=1e-6, atol=0)
        self.assertTrue(torch.equal(q_out.float(), expected_q.float()))
        self.assertEqual(scales[0].item(), 1.0)
        self.assertEqual(scales[9].item(), 1.0)


if __name__ == "__main__":
    main()
