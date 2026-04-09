"""
Unit test for MoE layer with mixtbstars model parameters.
Tests both BF16 and FP8 per-channel paths against torch reference.
"""

import unittest

import torch
import torch.nn.functional as F
from aiter.ops.shuffle import shuffle_weight

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    ExpertForwardPayload,
    ExpertTokensMetadata,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.rocm.executors.deepep_normal_fused_moe_executor import (
    torch_moe_ref,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.rocm.executors.rocm_moe import (
    RocmExpertsBf16,
    RocmExpertsFp8PerChannel,
)
from rtp_llm.ops import MoeConfig, ParallelismConfig
from rtp_llm.utils.model_weight import W


def make_config(expert_num, top_k, inter_dim, hidden_dim):
    model_config = ModelConfig()
    model_config.attn_config.head_num = 4
    model_config.attn_config.size_per_head = 64
    model_config.num_layers = 2
    model_config.max_seq_len = 2048
    model_config.vocab_size = 32000
    model_config.expert_num = expert_num
    model_config.moe_k = top_k
    model_config.inter_size = inter_dim
    model_config.activation_type = "silu"

    parallelism_config = ParallelismConfig()
    parallelism_config.ep_size = 1
    parallelism_config.ep_rank = 0
    parallelism_config.tp_size = 1
    parallelism_config.tp_rank = 0
    parallelism_config.dp_size = 1
    parallelism_config.dp_rank = 0
    parallelism_config.world_size = 1
    parallelism_config.world_rank = 0
    parallelism_config.local_rank = 0
    parallelism_config.local_world_size = 1

    moe_config = MoeConfig()
    config_adapter = MoEConfigAdapter(
        model_config=model_config,
        parallelism_config=parallelism_config,
        moe_config=moe_config,
    )
    return config_adapter


def pad_to_align(tensor, dim, align):
    """Pad tensor along dim to be divisible by align."""
    size = tensor.size(dim)
    if size % align == 0:
        return tensor
    pad_size = align - (size % align)
    # F.pad expects padding in reverse dim order: (last_dim_left, last_dim_right, ...)
    # We want to pad at the back of the target dim
    ndim = tensor.dim()
    pad_list = [0] * (2 * ndim)
    # For dim d in an ndim tensor, the F.pad index for "right side" is:
    # 2 * (ndim - 1 - d) + 1  (left is +0, right is +1, but F.pad order is left,right)
    # Actually F.pad order: [last_left, last_right, second_last_left, second_last_right, ...]
    rev_dim = ndim - 1 - dim
    pad_list[2 * rev_dim + 1] = pad_size  # right side padding
    return F.pad(tensor, pad_list)


def make_payload(hidden_states, topk_ids, topk_weights, expert_x_scale=None):
    return ExpertForwardPayload(
        expert_x=hidden_states,
        expert_x_scale=expert_x_scale,
        expert_topk_ids=topk_ids,
        expert_topk_weights=topk_weights,
        expert_tokens_meta=ExpertTokensMetadata(),
    )


class TestMoeMixtbstars(unittest.TestCase):
    """Test MoE with mixtbstars-like parameters"""

    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        torch.set_default_device("cuda")
        torch.manual_seed(42)

    def test_bf16_small(self):
        """BF16 MoE with small aligned dimensions (sanity check)"""
        self._run_bf16_test(M=4, E=8, K=2, D=256, N=512, atol=1e-2, rtol=1e-2)

    def test_bf16_mixtbstars_dims(self):
        """BF16 MoE with mixtbstars dimensions (96 experts, topk=8, hidden=2048, inter=1560)"""
        self._run_bf16_test(M=4, E=96, K=8, D=2048, N=1560, atol=1e-1, rtol=1e-1)

    @unittest.skip(
        "aiter moe_stage1_g1u1 doesn't support BF16 input + FP8 weight combo"
    )
    def test_fp8_small(self):
        """FP8 per-channel MoE with small aligned dimensions (sanity check)"""
        self._run_fp8_test(M=4, E=8, K=2, D=256, N=512, atol=5e-1, rtol=5e-1)

    @unittest.skip(
        "aiter moe_stage1_g1u1 doesn't support BF16 input + FP8 weight combo"
    )
    def test_fp8_mixtbstars_dims(self):
        """FP8 per-channel MoE with mixtbstars dimensions"""
        self._run_fp8_test(M=4, E=96, K=8, D=2048, N=1560, atol=5e-1, rtol=5e-1)

    def _run_bf16_test(self, M, E, K, D, N, atol, rtol):
        """Run BF16 MoE test: aiter fused_moe vs torch reference"""
        dtype = torch.bfloat16

        hidden_states = torch.randn(M, D, dtype=dtype) * 0.03
        topk_ids = torch.topk(torch.rand(M, E), K, dim=1).indices.to(torch.int32)
        topk_weights = torch.softmax(torch.randn(M, K), dim=-1).to(torch.float32)

        # Raw weights (unpadded) for reference
        # w1: [E, N*2, D] where first half is gate, second half is up
        # w2: [E, D, N]
        w1_raw = torch.randn(E, N * 2, D, dtype=dtype) * 0.02
        w2_raw = torch.randn(E, D, N, dtype=dtype) * 0.02

        # Reference: pure torch matmul with raw weights
        payload_ref = make_payload(hidden_states, topk_ids, topk_weights)
        ref_out = torch_moe_ref(
            payload=payload_ref,
            activation="silu",
            global_num_experts=E,
            expert_map=None,
            a2_scale=None,
            apply_router_weight_on_input=False,
            extra_expert_args=None,
            w1=w1_raw,
            w2=w2_raw,
        )

        # aiter: pad inter_dim consistently, then shuffle
        # Pad inter_dim to next multiple of 256 (matching production moe_stack_w1_pad)
        N_padded = ((N + 255) // 256) * 256
        # Pad w1: [E, N*2, D] -> [E, N_padded*2, D]
        w1_padded = F.pad(w1_raw.clone(), (0, 0, 0, N_padded * 2 - N * 2))
        # Pad w2: [E, D, N] -> [E, D, N_padded]
        w2_padded = F.pad(w2_raw.clone(), (0, N_padded - N))

        w1_shuffled = shuffle_weight(w1_padded, layout=(16, 16))
        w2_shuffled = shuffle_weight(w2_padded, layout=(16, 16))

        config = make_config(E, K, N_padded * 2, D)
        weights = {W.moe_w1: w1_shuffled, W.moe_w2: w2_shuffled}
        executor = RocmExpertsBf16(config, FusedMoEQuantConfig(), weights)

        payload_test = make_payload(
            hidden_states.clone(), topk_ids.clone(), topk_weights.clone()
        )
        result = executor.execute(
            payload=payload_test,
            activation="silu",
            expert_map=None,
            a2_scale=None,
            apply_router_weight_on_input=False,
            extra_expert_args=None,
        )

        print(f"\n[BF16 E={E} K={K} D={D} N={N} N_padded={N_padded}]")
        print(
            f"  ref:   mean={ref_out.float().mean():.6f} std={ref_out.float().std():.6f}"
        )
        print(
            f"  aiter: mean={result.fused_expert_output.float().mean():.6f} std={result.fused_expert_output.float().std():.6f}"
        )
        diff = (result.fused_expert_output.float() - ref_out.float()).abs()
        print(f"  diff:  max={diff.max():.6f} mean={diff.mean():.6f}")

        torch.testing.assert_close(
            result.fused_expert_output, ref_out, atol=atol, rtol=rtol
        )

    def _run_fp8_test(self, M, E, K, D, N, atol, rtol):
        """Run FP8 per-channel MoE test: CK stage1+stage2 vs torch reference"""
        dtype = torch.bfloat16

        hidden_states = torch.randn(M, D, dtype=dtype) * 0.03
        topk_ids = torch.topk(torch.rand(M, E), K, dim=1).indices.to(torch.int32)
        topk_weights = torch.softmax(torch.randn(M, K), dim=-1).to(torch.float32)

        # Raw BF16 weights for reference
        w1_raw = torch.randn(E, N * 2, D, dtype=dtype) * 0.02
        w2_raw = torch.randn(E, D, N, dtype=dtype) * 0.02

        # Reference output (pure torch, no quantization)
        payload_ref = make_payload(hidden_states, topk_ids, topk_weights)
        ref_out = torch_moe_ref(
            payload=payload_ref,
            activation="silu",
            global_num_experts=E,
            expert_map=None,
            a2_scale=None,
            apply_router_weight_on_input=False,
            extra_expert_args=None,
            w1=w1_raw,
            w2=w2_raw,
        )

        # Quantize weights to FP8 with per-channel scales
        def quantize_to_fp8(tensor):
            abs_max = tensor.float().abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
            max_fp8 = 240.0
            scale = abs_max / max_fp8
            quantized = (
                (tensor.float() / scale)
                .clamp(-max_fp8, max_fp8)
                .to(torch.float8_e4m3fnuz)
            )
            return quantized, scale

        # Pad before quantize, then shuffle
        w1_padded = pad_to_align(w1_raw.clone(), dim=1, align=32)
        w2_padded = pad_to_align(w2_raw.clone(), dim=2, align=32)

        w1_fp8, w1_scale = quantize_to_fp8(w1_padded)
        w2_fp8, w2_scale = quantize_to_fp8(w2_padded)

        w1_fp8_shuffled = shuffle_weight(w1_fp8, layout=(16, 16))
        w2_fp8_shuffled = shuffle_weight(w2_fp8, layout=(16, 16))

        # Per-token quantize input
        from aiter import dynamic_per_token_scaled_quant

        input_fp8 = torch.empty_like(hidden_states, dtype=torch.float8_e4m3fnuz)
        input_scale = torch.empty(
            (M, 1), dtype=torch.float32, device=hidden_states.device
        )
        dynamic_per_token_scaled_quant(input_fp8, hidden_states, input_scale)

        config = make_config(E, K, N * 2, D)
        weights = {
            W.moe_w1: w1_fp8_shuffled,
            W.moe_w2: w2_fp8_shuffled,
            W.moe_s1: w1_scale,
            W.moe_s2: w2_scale,
        }
        executor = RocmExpertsFp8PerChannel(config, FusedMoEQuantConfig(), weights)

        payload_test = make_payload(
            hidden_states.clone(),
            topk_ids.clone(),
            topk_weights.clone(),
            expert_x_scale=input_scale,
        )
        result = executor.execute(
            payload=payload_test,
            activation="silu",
            expert_map=None,
            a2_scale=None,
            apply_router_weight_on_input=False,
            extra_expert_args=None,
        )

        print(f"\n[FP8 E={E} K={K} D={D} N={N}]")
        print(
            f"  ref:   mean={ref_out.float().mean():.6f} std={ref_out.float().std():.6f}"
        )
        print(
            f"  fp8:   mean={result.fused_expert_output.float().mean():.6f} std={result.fused_expert_output.float().std():.6f}"
        )
        diff = (result.fused_expert_output.float() - ref_out.float()).abs()
        print(f"  diff:  max={diff.max():.6f} mean={diff.mean():.6f}")
        cos_sim = F.cosine_similarity(
            result.fused_expert_output.float().flatten().unsqueeze(0),
            ref_out.float().flatten().unsqueeze(0),
        )
        print(f"  cosine_sim: {cos_sim.item():.6f}")

        self.assertGreater(
            cos_sim.item(), 0.9, f"Cosine similarity too low: {cos_sim.item()}"
        )


if __name__ == "__main__":
    unittest.main()
