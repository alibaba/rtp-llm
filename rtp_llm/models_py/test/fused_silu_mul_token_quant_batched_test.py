import itertools
import random
from typing import Optional, Tuple
from unittest import SkipTest, TestCase, main

import torch
from torch import dtype as _dtype

from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.util import (
    moe_kernel_quantize_input,
)
from rtp_llm.models_py.triton_kernels.common.activation import (
    silu_and_mul,
    silu_mul_fp8_per_token_quant_batched,
)

import rtp_llm.ops  # isort:skip
from rtp_llm.ops.compute_ops import per_token_quant_fp8  # isort:skip


class FusedSiluMulPerTokenQuantBatchedTest(TestCase):
    MAX_NUM_TOKENS = [128, 256, 512]
    HIDDEN_SIZES = [128, 768, 1024, 2048, 4096, 8192]
    NUM_EXPERTS = [16, 20, 64]

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is not available")
        torch.set_default_device("cuda")

    def ref_silu_mul_quant_no_fused(self, input_x, expert_num_tokens):
        E, T, H2 = input_x.shape
        input_x = input_x.view(-1, H2)

        output = torch.empty((E * T, H2 // 2), dtype=input_x.dtype)
        silu_and_mul(output, input_x)
        q_x, q_s = moe_kernel_quantize_input(output, None, torch.float8_e4m3fn, True)
        return q_x, q_s

    def fused_silu_mul_quant_batched(self, input_x, expert_num_tokens):
        q_x, q_s = silu_mul_fp8_per_token_quant_batched(input_x, expert_num_tokens)
        return q_x, q_s

    def _run_silu_mul_per_token_fp8_quant_batched_test(
        self, max_num_tokens: int, hidden_size: int, num_experts: int
    ):
        device = torch.device("cuda")
        x = torch.rand(
            (num_experts, max_num_tokens, hidden_size),
            dtype=torch.bfloat16,
            device=device,
        )
        expert_num_tokens = torch.empty([num_experts], dtype=torch.int32, device=device)
        for i in range(num_experts):
            expert_num_tokens[i] = random.randint(0, max_num_tokens)
        expert_num_tokens[0] = max(1, expert_num_tokens[0])

        ref_q_out, ref_q_scale = self.ref_silu_mul_quant_no_fused(x, expert_num_tokens)
        ref_q_out = ref_q_out.view(num_experts, max_num_tokens, -1)
        ref_q_scale = ref_q_scale.view(num_experts, -1)

        q_out, q_scale = self.fused_silu_mul_quant_batched(x, expert_num_tokens)
        self.assertTrue(
            torch.allclose(
                ref_q_scale[i, 0 : expert_num_tokens[i]],
                q_scale[i, 0 : expert_num_tokens[i]],
                atol=1e-5,
                rtol=1e-5,
            )
        )

    def test_silu_mul_per_token_fp8_quant_batched(self):
        for params in itertools.product(
            self.MAX_NUM_TOKENS,
            self.HIDDEN_SIZES,
            self.NUM_EXPERTS,
        ):
            with self.subTest(
                max_num_tokens=params[0],
                hidden_size=params[1],
                num_experts=params[2],
            ):
                self._run_silu_mul_per_token_fp8_quant_batched_test(*params)


if __name__ == "__main__":
    main()
