"""Minimal gpt-oss MoE executor reproduction with synthetic weights.

Isolates the Fp8Fp4MoeLayer (grouped-fp4 executor) construction + forward
from the loader: synthetic tensors with the exact diagnosed shapes
(gpt-oss 20b: E=32, D=inter=2880 -> K padded to 3072). Crashes here
reproduce without the 13.8GB checkpoint load; passing here plus a crash in
the weights-compare test would point at loader-context interaction.
"""

import unittest

import torch

from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.layer import (
    Fp8Fp4MoeLayer,
)
from rtp_llm.utils.model_weight import W


class GptOssMoeSynthTest(unittest.TestCase):
    def _layer_weights(self, E, D, inter):
        torch.manual_seed(0)
        # scale bytes use the real gpt-oss distribution (observed 117-126):
        # a 0x00 byte (2^-127) or 0xFF (NaN) in the REAL region faults
        # deep_gemm's transform kernel — see RUN issues/
        # b300-deepgemm-launch-fail-in-loader-test.md
        return {
            W.moe_w1: torch.randint(
                -128, 128, (E, 2 * inter, D // 2), dtype=torch.int8, device="cuda"
            ),
            W.moe_s1: torch.randint(
                110, 135, (E, 2 * inter, D // 32), dtype=torch.uint8, device="cuda"
            ),
            W.moe_b1: torch.randn(E, 2 * inter, dtype=torch.bfloat16, device="cuda"),
            W.moe_w2: torch.randint(
                -128, 128, (E, D, inter // 2), dtype=torch.int8, device="cuda"
            ),
            W.moe_s2: torch.randint(
                110, 135, (E, D, inter // 32), dtype=torch.uint8, device="cuda"
            ),
            W.moe_b2: torch.randn(E, D, dtype=torch.bfloat16, device="cuda"),
            W.moe_gate: torch.randn(E, D, dtype=torch.bfloat16, device="cuda"),
            W.moe_gate_bias: torch.randn(E, dtype=torch.bfloat16, device="cuda"),
        }

    def test_construct_and_forward(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        E, D, inter = 32, 2880, 2880
        lw = self._layer_weights(E, D, inter)
        moe = Fp8Fp4MoeLayer(
            layer_id=0,
            dim=D,
            moe_inter_dim=inter,
            n_routed_experts=E,
            n_activated_experts=4,
            n_shared_experts=0,
            score_func="topk_softmax",
            route_scale=1.0,
            swiglu_limit=7.0,
            n_hash_layers=0,
            vocab_size=201088,
            layer_weights=lw,
            ep_size=1,
            ep_rank=0,
            max_tokens_per_rank=512,
            strategy="auto",
            model_type="gpt_oss",
            moe_w1_layout="gate_up",
            swiglu_alpha=1.702,
            swiglu_up_shift=1.0,
        )
        print(f"[synth] strategy={moe.strategy_name}", flush=True)
        x = (torch.randn(16, D, device="cuda") * 0.5).bfloat16()
        out = moe(x, None).float()
        torch.cuda.synchronize()
        self.assertEqual(out.shape, (16, D))
        self.assertTrue(torch.isfinite(out).all())
        print(f"[synth] forward OK mean|out|={out.abs().mean().item():.4f}", flush=True)


if __name__ == "__main__":
    unittest.main()
