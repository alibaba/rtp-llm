import itertools
import random
from unittest import TestCase, main

import pytest
import torch

pytestmark = [pytest.mark.gpu(type="H20")]
from rtp_llm.models_py.modules.factory.fused_moe.utils.common import (
    moe_kernel_quantize_input,
)
from rtp_llm.models_py.triton_kernels.common.activation import (
    silu_and_mul,
    silu_mul_fp8_per_token_quant_batched,
)

import rtp_llm.ops  # isort:skip

def _ordered_fp8_codes(values: torch.Tensor) -> torch.Tensor:
    """Map finite E4M3 bit patterns to monotonically increasing integer codes."""
    raw = values.contiguous().view(torch.uint8)
    return torch.where(
        (raw & 0x80).bool(),
        torch.bitwise_not(raw),
        raw | 0x80,
    ).to(torch.int16)


class FusedSiluMulPerTokenQuantBatchedTest(TestCase):
    MAX_NUM_TOKENS = [128, 256, 512]
    HIDDEN_SIZES = [128, 768, 1024, 2048, 4096, 8192]
    NUM_EXPERTS = [16, 20, 64]

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.fail("CUDA is required for this GPU test")
        random.seed(42)
        torch.manual_seed(42)
        torch.set_default_device("cuda")

    def ref_silu_mul_quant_no_fused(self, input_x):
        E, T, H2 = input_x.shape
        values, gates = input_x.float().chunk(2, dim=-1)
        output = torch.nn.functional.silu(gates) * values
        output = output.view(E * T, H2 // 2)
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
        expert_num_tokens = (
            torch.arange(num_experts, dtype=torch.int32, device=device)
            .mul_(37)
            .remainder_(max_num_tokens + 1)
        )
        expert_num_tokens[0] = 0
        expert_num_tokens[1] = max_num_tokens

        ref_q_out, ref_q_scale = self.ref_silu_mul_quant_no_fused(x)
        ref_q_out = ref_q_out.view(num_experts, max_num_tokens, -1)
        ref_q_scale = ref_q_scale.view(num_experts, max_num_tokens, -1)

        q_out, q_scale = self.fused_silu_mul_quant_batched(x, expert_num_tokens)
        q_out = q_out.view(num_experts, max_num_tokens, -1)
        q_scale = q_scale.view(num_experts, max_num_tokens, -1)
        for i in range(num_experts):
            n = int(expert_num_tokens[i].item())
            if n == 0:
                continue
            self.assertTrue(
                torch.allclose(
                    ref_q_scale[i, :n].float(),
                    q_scale[i, :n].float(),
                    atol=1e-5,
                    rtol=1e-5,
                ),
                f"q_scale mismatch at expert {i}",
            )
            # PyTorch and Triton may round to opposite neighbors at an exact
            # E4M3 bin boundary. Accept only that one-ULP ambiguity; larger
            # differences remain a hard failure.
            ref_codes = _ordered_fp8_codes(ref_q_out[i, :n])
            actual_codes = _ordered_fp8_codes(q_out[i, :n])
            code_distance = torch.abs(actual_codes - ref_codes)
            self.assertLessEqual(
                int(code_distance.max().item()),
                1,
                f"q_out differs by more than one E4M3 value at expert {i}",
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
