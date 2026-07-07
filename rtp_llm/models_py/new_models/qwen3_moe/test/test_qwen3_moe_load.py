# type: ignore
"""CPU unit tests for Qwen3Experts.load_weights.

Covers the per-expert -> stacked [E, ...] packing contract, including:
- up_proj    -> rows [0, moe_inter_tp) of w13[expert_id]   (= SiLU "value")
- gate_proj  -> rows [moe_inter_tp, 2*moe_inter_tp) of w13[expert_id] (= SiLU "gate")
- down_proj  -> w2[expert_id]
- TP slicing: gate/up along dim-0, down along dim-1

The actual FusedMoe build (which requires CUDA + MoEConfigAdapter wiring) is
patched out, so these tests run on CPU.
"""
import unittest

import torch

from rtp_llm.models_py.new_models.qwen3_moe.language import Qwen3Experts


def _make_experts(
    num_experts: int,
    hidden_size: int,
    moe_intermediate_size: int,
    tp_size: int = 1,
    tp_rank: int = 0,
) -> Qwen3Experts:
    e = Qwen3Experts(
        num_experts=num_experts,
        hidden_size=hidden_size,
        moe_intermediate_size=moe_intermediate_size,
        tp_size=tp_size,
        tp_rank=tp_rank,
        params_dtype=torch.bfloat16,
        model_config=None,
        parallelism_config=None,
        moe_config=None,
        quant_config=None,
        layer_idx=0,
    )
    # Skip FusedMoe build — that path needs CUDA + a fully-populated
    # MoEConfigAdapter, neither of which we want in a CPU load test.
    e._maybe_build_fused_moe = lambda: None
    return e


class TestQwen3ExpertsStacking(unittest.TestCase):

    def test_gate_up_down_pack_into_correct_rows(self):
        E, H, M = 4, 16, 32
        experts = _make_experts(num_experts=E, hidden_size=H, moe_intermediate_size=M)

        # Build per-expert ckpt with deterministic values so we can check
        # the exact rows after stacking.
        weights = {}
        gate_refs, up_refs, down_refs = {}, {}, {}
        for i in range(E):
            gate = torch.full((M, H), float(i) + 0.1, dtype=torch.bfloat16)
            up = torch.full((M, H), float(i) + 0.2, dtype=torch.bfloat16)
            down = torch.full((H, M), float(i) + 0.3, dtype=torch.bfloat16)
            weights[f"{i}.gate_proj.weight"] = gate
            weights[f"{i}.up_proj.weight"] = up
            weights[f"{i}.down_proj.weight"] = down
            gate_refs[i], up_refs[i], down_refs[i] = gate, up, down

        experts.load_weights(weights)

        for i in range(E):
            # up rows go first, gate rows go second.
            torch.testing.assert_close(
                experts.w13.data[i, :M], up_refs[i], rtol=0, atol=0
            )
            torch.testing.assert_close(
                experts.w13.data[i, M:], gate_refs[i], rtol=0, atol=0
            )
            torch.testing.assert_close(experts.w2.data[i], down_refs[i], rtol=0, atol=0)

        self.assertEqual(experts._loaded_count, E * 3)

    def test_ignores_non_weight_and_unknown_proj(self):
        E, H, M = 2, 8, 16
        experts = _make_experts(num_experts=E, hidden_size=H, moe_intermediate_size=M)

        weights = {
            "0.gate_proj.weight": torch.zeros(M, H, dtype=torch.bfloat16),
            "0.gate_proj.weight_scale": torch.tensor([0.1]),  # ignored in Phase 1
            "0.unknown_proj.weight": torch.zeros(M, H, dtype=torch.bfloat16),
            # out-of-range expert id gets dropped silently
            "99.gate_proj.weight": torch.zeros(M, H, dtype=torch.bfloat16),
        }
        experts.load_weights(weights)
        self.assertEqual(experts._loaded_count, 1)


class TestQwen3ExpertsTPSlicing(unittest.TestCase):

    def test_tp_slices_gate_up_along_dim0_and_down_along_dim1(self):
        E, H, M, TP = 2, 16, 32, 2
        rank0 = _make_experts(E, H, M, tp_size=TP, tp_rank=0)
        rank1 = _make_experts(E, H, M, tp_size=TP, tp_rank=1)
        M_tp = M // TP  # 16

        weights = {}
        full_gate, full_up, full_down = {}, {}, {}
        for i in range(E):
            # use arange so each row is unique and we can verify the slice.
            full_gate[i] = torch.arange(M * H, dtype=torch.bfloat16).reshape(M, H) + (
                100 * i
            )
            full_up[i] = torch.arange(M * H, dtype=torch.bfloat16).reshape(M, H) + (
                200 * i
            )
            full_down[i] = torch.arange(H * M, dtype=torch.bfloat16).reshape(H, M) + (
                300 * i
            )
            weights[f"{i}.gate_proj.weight"] = full_gate[i]
            weights[f"{i}.up_proj.weight"] = full_up[i]
            weights[f"{i}.down_proj.weight"] = full_down[i]

        rank0.load_weights(weights)
        rank1.load_weights(weights)

        for i in range(E):
            # First half of w13 = up_proj. Each rank gets its TP slice along dim 0.
            torch.testing.assert_close(
                rank0.w13.data[i, :M_tp], full_up[i][:M_tp], rtol=0, atol=0
            )
            torch.testing.assert_close(
                rank1.w13.data[i, :M_tp],
                full_up[i][M_tp : 2 * M_tp],
                rtol=0,
                atol=0,
            )
            # Second half of w13 = gate_proj.
            torch.testing.assert_close(
                rank0.w13.data[i, M_tp:], full_gate[i][:M_tp], rtol=0, atol=0
            )
            torch.testing.assert_close(
                rank1.w13.data[i, M_tp:],
                full_gate[i][M_tp : 2 * M_tp],
                rtol=0,
                atol=0,
            )
            # down: TP-sliced along dim-1.
            torch.testing.assert_close(
                rank0.w2.data[i], full_down[i][:, :M_tp], rtol=0, atol=0
            )
            torch.testing.assert_close(
                rank1.w2.data[i], full_down[i][:, M_tp : 2 * M_tp], rtol=0, atol=0
            )

    def test_rejects_misaligned_moe_inter(self):
        # moe_intermediate_size not divisible by tp_size should raise at __init__.
        with self.assertRaises(ValueError):
            _make_experts(
                num_experts=2, hidden_size=8, moe_intermediate_size=15, tp_size=2
            )

    def test_rejects_wrong_dim0_for_gate_or_up(self):
        E, H, M = 1, 8, 16
        experts = _make_experts(E, H, M)
        bad = torch.zeros(M + 1, H, dtype=torch.bfloat16)
        with self.assertRaises(ValueError):
            experts.load_weights({"0.gate_proj.weight": bad})

    def test_rejects_wrong_dim1_for_down(self):
        E, H, M = 1, 8, 16
        experts = _make_experts(E, H, M)
        bad = torch.zeros(H, M + 1, dtype=torch.bfloat16)
        with self.assertRaises(ValueError):
            experts.load_weights({"0.down_proj.weight": bad})


class TestQwen3ExpertsStreamingLoad(unittest.TestCase):
    """Verify that calling load_weights once per tensor (streaming dispatch)
    produces the same result as a single batched call."""

    def test_per_tensor_matches_batched(self):
        E, H, M = 3, 16, 32
        batched = _make_experts(E, H, M)
        streamed = _make_experts(E, H, M)

        weights = {}
        for i in range(E):
            weights[f"{i}.gate_proj.weight"] = torch.randn(M, H, dtype=torch.bfloat16)
            weights[f"{i}.up_proj.weight"] = torch.randn(M, H, dtype=torch.bfloat16)
            weights[f"{i}.down_proj.weight"] = torch.randn(H, M, dtype=torch.bfloat16)

        batched.load_weights(weights)

        for name, tensor in weights.items():
            streamed.load_weights({name: tensor})

        torch.testing.assert_close(streamed.w13.data, batched.w13.data, rtol=0, atol=0)
        torch.testing.assert_close(streamed.w2.data, batched.w2.data, rtol=0, atol=0)
        self.assertEqual(streamed._loaded_count, batched._loaded_count)


if __name__ == "__main__":
    unittest.main()
