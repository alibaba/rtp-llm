"""SwiGLU clamp coverage for the framework SiLU-and-mul kernels.

The DeepSeek-V4 routed-expert contract clamps the SwiGLU branches (gate
capped at the limit, the linear branch clamped symmetrically) before the
activation. The DeepGemm FP8-per-block executors forward the model's
``swiglu_limit`` into these kernels; ``clamp_limit=0`` must stay a
bit-identical no-op for every other caller.
"""

import unittest

import torch

from rtp_llm.models_py.triton_kernels.common.activation import (
    MaskedSiluInputLayout,
    silu_and_mul,
    silu_mul_masked_bf16_no_post_quant_fwd,
    silu_mul_masked_fp8_post_quant_fwd,
)
from rtp_llm.test.utils.numeric_util import calc_diff, per_token_cast_back


def _clamped_reference(up_gate: torch.Tensor, clamp_limit: float) -> torch.Tensor:
    """Eager reference; the input layout is [up, gate] on the last axis."""
    n = up_gate.shape[-1]
    up = up_gate[..., : n // 2].to(torch.float32)
    gate = up_gate[..., n // 2 :].to(torch.float32)
    if clamp_limit > 0:
        up = torch.clamp(up, min=-clamp_limit, max=clamp_limit)
        gate = torch.clamp(gate, max=clamp_limit)
    gate = gate * (1.0 / (1.0 + torch.exp(-gate)))
    return (gate * up).to(torch.bfloat16)


class SiluClampTest(unittest.TestCase):
    CLAMP_LIMIT = 7.0
    NUM_EXPERTS = 4
    TOKENS = 256
    INTER = 512
    QUANT_GROUP = 128

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is not available")
        torch.manual_seed(0)

    def _masked_inputs(self, scale: float):
        shape = (self.NUM_EXPERTS, self.TOKENS, self.INTER * 2)
        up_gate = torch.randn(shape, device="cuda", dtype=torch.bfloat16) * scale
        masked_m = torch.full(
            (self.NUM_EXPERTS,), self.TOKENS, device="cuda", dtype=torch.int
        )
        return up_gate, masked_m

    def test_silu_and_mul_applies_clamp(self):
        # scale 6.0 puts a meaningful tail beyond the 7.0 clamp limit.
        up_gate = torch.randn(
            (64, self.INTER * 2), device="cuda", dtype=torch.bfloat16
        ) * 6.0
        for clamp in (0.0, self.CLAMP_LIMIT):
            with self.subTest(clamp=clamp):
                output = torch.empty(
                    (64, self.INTER), device="cuda", dtype=torch.bfloat16
                )
                silu_and_mul(output, up_gate, clamp_limit=clamp)
                reference = _clamped_reference(up_gate, clamp)
                self.assertLess(calc_diff(output, reference), 1e-3)

    def test_masked_fp8_post_quant_applies_clamp(self):
        up_gate, masked_m = self._masked_inputs(scale=6.0)
        output = torch.empty(
            (self.NUM_EXPERTS, self.TOKENS, self.INTER),
            device="cuda",
            dtype=torch.float8_e4m3fn,
        )
        output_scale = torch.empty(
            (self.NUM_EXPERTS, self.TOKENS, self.INTER // self.QUANT_GROUP),
            device="cuda",
            dtype=torch.float32,
        )
        silu_mul_masked_fp8_post_quant_fwd(
            input=up_gate,
            output=output,
            output_scale=output_scale,
            quant_group_size=self.QUANT_GROUP,
            masked_m=masked_m,
            expected_m=self.TOKENS,
            input_layout=MaskedSiluInputLayout.PER_EXPERT_CAPACITY,
            clamp_limit=self.CLAMP_LIMIT,
        )
        dequantized = per_token_cast_back(
            output.flatten(0, -2), output_scale.flatten(0, -2)
        ).view(output.shape)
        reference = _clamped_reference(up_gate, self.CLAMP_LIMIT)
        self.assertLess(calc_diff(dequantized, reference), 1e-3)

    def test_masked_bf16_no_post_quant_applies_clamp(self):
        up_gate, masked_m = self._masked_inputs(scale=6.0)
        output = torch.empty(
            (self.NUM_EXPERTS, self.TOKENS, self.INTER),
            device="cuda",
            dtype=torch.bfloat16,
        )
        silu_mul_masked_bf16_no_post_quant_fwd(
            input=up_gate,
            output=output,
            masked_m=masked_m,
            expected_m=self.TOKENS,
            input_layout=MaskedSiluInputLayout.PER_EXPERT_CAPACITY,
            group_size=self.QUANT_GROUP,
            clamp_limit=self.CLAMP_LIMIT,
        )
        reference = _clamped_reference(up_gate, self.CLAMP_LIMIT)
        self.assertLess(calc_diff(output, reference), 1e-3)

    def test_packed_post_quant_clamp_contract(self):
        from rtp_llm.models_py.triton_kernels.common.activation import (
            create_packed_scale_tensor,
            silu_and_mul_masked_post_quant_packed_fwd,
        )

        up_gate, masked_m = self._masked_inputs(scale=6.0)

        def run(clamp_limit: float) -> torch.Tensor:
            output = torch.empty(
                (self.NUM_EXPERTS, self.TOKENS, self.INTER),
                device="cuda",
                dtype=torch.float8_e4m3fn,
            )
            output_scale = create_packed_scale_tensor(
                expert_num=self.NUM_EXPERTS,
                token_num_padded=self.TOKENS,
                hidden_dim=self.INTER * 2,
                quant_group_size=self.QUANT_GROUP,
                device=torch.device("cuda"),
            )
            silu_and_mul_masked_post_quant_packed_fwd(
                up_gate,
                output,
                output_scale,
                self.QUANT_GROUP,
                masked_m,
                clamp_limit=clamp_limit,
            )
            return output

        unclamped = run(0.0).view(torch.uint8)
        # A clamp far beyond the data range is a bit-identical no-op.
        self.assertTrue(torch.equal(run(1.0e4).view(torch.uint8), unclamped))
        # An in-range clamp changes the quantized activations.
        self.assertFalse(
            torch.equal(run(self.CLAMP_LIMIT).view(torch.uint8), unclamped)
        )


if __name__ == "__main__":
    unittest.main()
