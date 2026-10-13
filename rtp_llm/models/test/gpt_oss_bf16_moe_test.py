"""gpt-oss BF16-dequant MoE lane (sm90/sm12x fallback) contract test.

sm90 (H20) and sm12x (RTX Pro 5000) have no native MXFP4 GEMM; their
bring-up dequantizes the packed experts to BF16 and runs the generic
Triton fused-MoE executor. This test pins the fallback contract with
synthetic checkpoints (no 13.8GB load):

  1. ``dequant_mxfp4_stacked`` decodes nibbles + UE8M0 group-32 scales
     bit-exactly against the fp32 oracle from gpt_oss_math_test.
  2. The dequantized layer (``_dequant_expert_weights`` layout: w1
     [up; gate], triton executor, gpt-oss gated activation + expert
     biases) matches the HF-math fp32 oracle (envelope, as with the
     packed lane — BF16 compute).
  3. The legacy no-bias SiLU*up lane is unchanged (route weight folded
     in GEMM2) — regression guard for existing Qwen/DeepSeek models.
"""

import unittest

import torch

from rtp_llm.models.gpt_oss.gpt_oss_dequant import dequant_mxfp4_stacked
from rtp_llm.models_py.model_desc.gpt_oss import _dequant_expert_weights
from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.layer import (
    Fp8Fp4MoeLayer,
)
from rtp_llm.utils.model_weight import W

E2M1_MAGS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _random_mxfp4(e: int, n: int, k: int, seed: int):
    """Ground-truth stacked tensors: packed [e, n, k/2] uint8 (gate rows
    first after deinterleave), scales [e, n, k/32] uint8, decoded fp32
    [e, n, k]. Mirrors gpt_oss_math_test._random_mxfp4 + the loader's
    deinterleave_stack_moe_w1 output layout."""
    torch.manual_seed(seed)
    assert k % 32 == 0
    g = k // 32
    codes = torch.randint(0, 16, (e, n, k), dtype=torch.uint8)
    exps = torch.randint(120, 135, (e, n, g), dtype=torch.uint8)
    # HF interleaves gate/up rows (even gate, odd up); the loader's
    # deinterleave_stack_moe_w1 reorders to [gate rows; up rows].
    gate_rows = codes[:, 0::2]
    up_rows = codes[:, 1::2]
    reordered = torch.cat([gate_rows, up_rows], dim=1)
    packed = (
        (reordered[..., 1::2].to(torch.int16) << 4) | reordered[..., 0::2].to(
            torch.int16
        )
    ).to(torch.uint8)
    reordered_exps = torch.cat(
        [exps[:, 0::2], exps[:, 1::2]], dim=1
    )  # scales follow their rows
    mags = E2M1_MAGS
    sign = torch.where((reordered & 8) > 0, -1.0, 1.0)
    decoded = mags[(reordered & 7).long()] * sign
    scales = torch.pow(2.0, reordered_exps.to(torch.float32) - 127.0)
    w = decoded.reshape(e, n, g, 32) * scales.unsqueeze(-1)
    return packed, reordered_exps, w.reshape(e, n, k)


def _hf_oracle(x, w_gu, w_d, b_gu, b_d, gate_w, gate_b, limit=7.0):
    """gpt-oss MoE forward in fp32 from the deinterleaved [gate; up] view."""
    logits = x @ gate_w.T + gate_b
    top_value, indices = logits.topk(4, dim=-1)
    weights = torch.softmax(top_value, dim=-1)
    y = torch.zeros_like(x, dtype=torch.float32)
    n = w_gu.shape[1]
    inter = n // 2
    for t in range(x.shape[0]):
        for k in range(4):
            eid = int(indices[t, k])
            gu = x[t] @ w_gu[eid].T + b_gu[eid]
            gate, up = gu[:inter], gu[inter:]
            gate = gate.clamp(max=limit)
            up = up.clamp(min=-limit, max=limit)
            act = (gate * torch.sigmoid(1.702 * gate)) * (up + 1.0)
            out = act @ w_d[eid].T + b_d[eid]
            y[t] += float(weights[t, k]) * out
    return y


class GptOssBf16MoeTest(unittest.TestCase):
    def test_dequant_bit_exact(self):
        torch.manual_seed(0)
        e, n, k = 5, 16, 64
        packed, exps, w_ref = _random_mxfp4(e, n, k, seed=11)
        got = dequant_mxfp4_stacked(packed, exps)
        self.assertEqual(got.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(got.float(), w_ref))

    def _bf16_layer_weights(self, E, D, inter):
        w1_packed, s1, w_gu = _random_mxfp4(E, 2 * inter, D, seed=21)
        w2_packed, s2, w_d = _random_mxfp4(E, D, inter, seed=22)
        gate_w = (torch.randn(E, D) * 0.05).bfloat16()
        gate_b = (torch.randn(E) * 0.05).bfloat16()
        b_gu = (torch.randn(E, 2 * inter) * 0.05).bfloat16()
        b_d = (torch.randn(E, D) * 0.05).bfloat16()
        lw = {
            W.moe_w1: w1_packed,
            W.moe_s1: s1,
            W.moe_w2: w2_packed,
            W.moe_s2: s2,
            W.moe_b1: b_gu,
            W.moe_b2: b_d,
            W.moe_gate: gate_w,
            W.moe_gate_bias: gate_b,
        }
        lw = _dequant_expert_weights(lw)
        oracle_w = (w_gu, w_d, b_gu.float(), b_d.float(), gate_w.float(), gate_b.float())
        return lw, oracle_w

    def test_bf16_lane_matches_oracle(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        E, D, inter = 8, 128, 96
        lw, (w_gu, w_d, b_gu, b_d, gate_w, gate_b) = self._bf16_layer_weights(
            E, D, inter
        )
        # _dequant_expert_weights mutates on CPU tensors here; move after.
        lw = {k: v.cuda() for k, v in lw.items()}
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
            strategy="no_quant_cpp",
            model_type="gpt_oss",
            moe_w1_layout="up_gate",
            swiglu_alpha=1.702,
            swiglu_up_shift=1.0,
            moe_quant_method=None,
        )
        print(f"[bf16] strategy={moe.strategy_name}", flush=True)
        self.assertEqual(moe.strategy_name, "no_quant_cpp")
        x = (torch.randn(16, D) * 0.5).bfloat16().cuda()
        out = moe(x, None).float()
        torch.cuda.synchronize()
        ref = _hf_oracle(x.float().cpu(), w_gu, w_d, b_gu, b_d, gate_w, gate_b)
        rel = (out.cpu() - ref).norm() / ref.norm()
        print(f"[bf16] rel-l2={rel:.4f}", flush=True)
        self.assertLess(rel, 0.05)
        self.assertTrue(torch.isfinite(out).all())

    def test_legacy_silu_lane_unchanged(self):
        """No-bias lane keeps legacy SiLU*up math (Qwen-style weights)."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        E, D, inter = 4, 64, 48
        torch.manual_seed(3)
        lw = {
            W.moe_w1: torch.randn(E, 2 * inter, D, dtype=torch.bfloat16).cuda(),
            W.moe_w2: torch.randn(E, D, inter, dtype=torch.bfloat16).cuda(),
            W.moe_gate: torch.randn(E, D, dtype=torch.bfloat16).cuda(),
            W.moe_gate_bias: torch.randn(E, dtype=torch.bfloat16).cuda(),
        }
        moe = Fp8Fp4MoeLayer(
            layer_id=0,
            dim=D,
            moe_inter_dim=inter,
            n_routed_experts=E,
            n_activated_experts=4,
            n_shared_experts=0,
            score_func="topk_softmax",
            route_scale=1.0,
            swiglu_limit=0.0,
            n_hash_layers=0,
            vocab_size=201088,
            layer_weights=lw,
            ep_size=1,
            ep_rank=0,
            max_tokens_per_rank=512,
            strategy="no_quant_cpp",
            model_type="gpt_oss",
            moe_w1_layout="up_gate",
            moe_quant_method=None,
        )
        x = (torch.randn(12, D) * 0.5).bfloat16().cuda()
        out = moe(x, None).float()
        torch.cuda.synchronize()
        # fp32 oracle: Qwen math, w1 rows [up; gate] per stack_moe_w1.
        logits = x.float().cpu() @ lw[W.moe_gate].float().cpu().T + lw[
            W.moe_gate_bias
        ].float().cpu()
        top_value, indices = logits.topk(4, dim=-1)
        weights = torch.softmax(top_value, dim=-1)
        w1 = lw[W.moe_w1].float().cpu()
        w2 = lw[W.moe_w2].float().cpu()
        y = torch.zeros(12, D)
        for t in range(12):
            for k in range(4):
                eid = int(indices[t, k])
                gu = x[t].float().cpu() @ w1[eid].T
                up, gate = gu[:inter], gu[inter:]
                act = torch.nn.functional.silu(gate) * up
                y[t] += float(weights[t, k]) * (act @ w2[eid].T)
        rel = (out.cpu() - y).norm() / y.norm()
        print(f"[legacy] rel-l2={rel:.4f}", flush=True)
        self.assertLess(rel, 0.05)
        self.assertTrue(torch.isfinite(out).all())


if __name__ == "__main__":
    unittest.main()
