"""Unit test for Mxfp8DeepepExecutor (DeepEP non-expand dispatch + MXFP8 grouped GEMM).

Mirrors deepep_normal_executor_test.py but for the non-expand dispatch path:
- expert_x: [N_recv, K] BF16 unique tokens (not per-expert duplicated)
- expert_topk_ids: [N_recv, top_k] int32 local expert IDs + -1
- expert_num_tokens: [E_local] padded to expert_alignment (128)

Reference output is computed via pure PyTorch BF16 matmul + SwiGLU + weighted sum,
compared against Mxfp8DeepepExecutor.execute() output.

Requires SM100 (Blackwell) + deep_gemm + mxfp8_grouped_gemm.
Skips gracefully on unsupported hardware.
"""

import random
import unittest

import torch

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
from rtp_llm.ops import MoeConfig, ParallelismConfig
from rtp_llm.utils.model_weight import W


def _has_mxfp8_support():
    """Check if mxfp8_grouped_gemm is available (SM100 + deep_gemm with FP4)."""
    if not torch.cuda.is_available():
        return False
    try:
        from rtp_llm.models_py.utils.arch import get_sm

        if get_sm()[0] < 10:
            return False
        from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import has_deep_gemm

        if not has_deep_gemm():
            return False
        # Check per_token_cast_to_fp4 (needed by test data generation)
        from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (  # noqa: F401
            per_token_cast_to_fp4,
        )
        from rtp_llm.models_py.kernels.cuda.mxfp8_ops import (  # noqa: F401
            mxfp8_grouped_gemm,
        )

        return True
    except (ImportError, RuntimeError):
        return False


def generate_deepep_normal_payload(
    config: MoEConfigAdapter,
):
    """Generate DeepEP non-expand dispatch payload + weights for Mxfp8DeepepExecutor.

    Returns:
        payload: ExpertForwardPayload with unique-token dispatch format
        weights: dict with BF16 w1/w2 + fp32 w1_scale/w2_scale
        topk_ids: [N_recv, top_k] for reference computation
        topk_weights: [N_recv, top_k] for reference computation
    """
    K = config.hidden_size
    N = config.model_config.moe_inter_size * 2  # gate + up concatenated
    E_global = config.expert_num
    E_local = config.expert_num // config.ep_size
    top_k = 8
    torch_dtype = torch.bfloat16
    device = "cuda"

    N_recv = (
        (config.ll_num_max_token + config.tp_size - 1)
        // config.tp_size
        * config.ep_size
    )

    # Unique received tokens (BF16)
    hidden = (
        torch.rand(N_recv, K, device=device, dtype=torch.float32).to(torch_dtype) * 0.1
        - 0.05
    )

    # Routing: each token picks top_k experts, map to local IDs via expert_map
    global_topk_ids = torch.randint(
        0, E_global, (N_recv, top_k), device=device, dtype=torch.int32
    )
    topk_weights = torch.softmax(
        torch.randn(N_recv, top_k, device=device, dtype=torch.float32), dim=-1
    )

    # expert_map: global expert → local expert ID (or -1 if not local)
    ep_rank = 0
    start_expert = ep_rank * E_local
    expert_map = torch.full((E_global,), -1, dtype=torch.int32, device=device)
    expert_map[start_expert : start_expert + E_local] = torch.arange(
        E_local, dtype=torch.int32, device=device
    )

    local_topk_ids = expert_map[
        global_topk_ids.long()
    ]  # [N_recv, top_k], -1 for non-local

    # Compute per-expert counts padded to alignment
    ALIGN = 128  # deep_gemm contiguous alignment
    flat_ids = local_topk_ids.reshape(-1)
    valid_mask = flat_ids >= 0
    counts = (
        torch.bincount(flat_ids[valid_mask], minlength=E_local)
        if valid_mask.any()
        else torch.zeros(E_local, device=device, dtype=torch.int64)
    )
    padded_counts = [((int(c) + ALIGN - 1) // ALIGN * ALIGN) for c in counts.tolist()]

    payload = ExpertForwardPayload(
        expert_x=hidden,
        expert_x_origin_dtype=torch.bfloat16,
        expert_x_scale=None,
        expert_topk_ids=local_topk_ids,
        expert_topk_weights=topk_weights,
        expert_tokens_meta=ExpertTokensMetadata(
            expert_num_tokens=torch.tensor(
                padded_counts, dtype=torch.int32, device=device
            ),
            expert_num_tokens_cpu=padded_counts,
        ),
    )

    # BF16 weights → quantize to MXFP4 (packed int8 + UE8M0 scale → packed int32)
    # DeepGEMM's m_grouped_fp8_fp4_gemm_nt_contiguous requires:
    #   weight: [E, N, K//2] int8 (packed FP4 e2m1)
    #   scale:  fp32 raw scale (executor calls pack_mxfp8_scale internally)
    from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import per_token_cast_to_fp4

    MX_BLOCK = 32

    w1_bf16 = (
        torch.rand(E_local, N, K, device=device, dtype=torch.float32).to(torch_dtype)
        * 2
        - 1
    ) * 0.01
    w2_bf16 = (
        torch.rand(E_local, K, N // 2, device=device, dtype=torch.float32).to(
            torch_dtype
        )
        * 2
        - 1
    ) * 0.01

    # Quantize w1 to packed FP4
    w1_packed = torch.empty((E_local, N, K // 2), dtype=torch.int8, device=device)
    w1_scale_raw = torch.empty(
        (E_local, N, K // MX_BLOCK), dtype=torch.float32, device=device
    )
    for i in range(E_local):
        w1_packed[i], w1_scale_raw[i] = per_token_cast_to_fp4(
            w1_bf16[i], use_ue8m0=True, gran_k=MX_BLOCK
        )

    # Quantize w2 to packed FP4
    w2_packed = torch.empty(
        (E_local, K, (N // 2) // 2), dtype=torch.int8, device=device
    )
    w2_scale_raw = torch.empty(
        (E_local, K, (N // 2) // MX_BLOCK), dtype=torch.float32, device=device
    )
    for i in range(E_local):
        w2_packed[i], w2_scale_raw[i] = per_token_cast_to_fp4(
            w2_bf16[i], use_ue8m0=True, gran_k=MX_BLOCK
        )

    weights = {
        W.moe_w1: w1_packed,
        W.moe_w2: w2_packed,
        W.moe_s1: w1_scale_raw,  # executor calls pack_mxfp8_scale internally
        W.moe_s2: w2_scale_raw,
    }

    # Keep original BF16 weights for reference computation
    ref_weights = {
        W.moe_w1: w1_bf16,
        W.moe_w2: w2_bf16,
    }

    return payload, weights, ref_weights, local_topk_ids, topk_weights


def generate_ref_output_deepep_normal(
    payload: ExpertForwardPayload,
    ref_weights: dict,
):
    """Pure PyTorch BF16 reference for Mxfp8DeepepExecutor.

    Uses original BF16 weights (not FP4 quantized) for high-precision reference.
    """
    hidden = payload.expert_x  # [N_recv, K]
    topk_ids = payload.expert_topk_ids  # [N_recv, top_k]
    topk_weights = payload.expert_topk_weights  # [N_recv, top_k]
    w1 = ref_weights[W.moe_w1]  # [E_local, 2*inter, K] BF16
    w2 = ref_weights[W.moe_w2]  # [E_local, K, inter] BF16
    E_local = w1.size(0)
    N = w1.size(1)
    K = hidden.size(1)
    N_recv = hidden.size(0)
    inter = N // 2

    ref_output = torch.zeros(N_recv, K, device="cuda", dtype=torch.bfloat16)

    for e in range(E_local):
        # Find all (token_i, slot_k) assigned to expert e
        mask = topk_ids == e
        if not mask.any():
            continue

        token_indices = torch.where(mask)[0]  # source token row indices
        slot_indices = torch.where(mask)[1]  # which top_k slot

        # Deduplicate: multiple slots of same token may go to same expert
        # (unlikely but possible) — accumulate all contributions
        x = hidden[token_indices]  # [num_assignments, K]
        w = topk_weights[token_indices, slot_indices]  # [num_assignments]

        # Expert FFN: gate+up → SwiGLU → down
        gate_up = x @ w1[e].T  # [num_assignments, 2*inter]
        gate = gate_up[:, :inter].float()
        up = gate_up[:, inter:].float()
        act = (gate * torch.sigmoid(gate)) * up  # SwiGLU in fp32
        down = act.to(torch.bfloat16) @ w2[e].T  # [num_assignments, K]

        # Weighted accumulation
        ref_output.index_add_(
            0, token_indices, (down * w.unsqueeze(1)).to(torch.bfloat16)
        )

    return ref_output


class Mxfp8DeepepExecutorTest(unittest.TestCase):
    DP_SIZE = 4
    TP_SIZE = 1
    EP_SIZE = 4
    NUM_EXPERTS = 128
    MAX_GENERATE_BATCH_SIZE = 128
    HIDDEN_SIZE = 2048
    MOE_INTERMEDIATE_SIZE = 768

    def setUp(self):
        if not _has_mxfp8_support():
            self.skipTest(
                "MXFP8 DeepEP executor requires SM100 + deep_gemm + mxfp8_grouped_gemm"
            )
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        random.seed(42)

    def _generate_config(self) -> MoEConfigAdapter:
        model_config = ModelConfig()
        model_config.attn_config.head_num = 2
        model_config.attn_config.size_per_head = 128
        model_config.num_layers = 2
        model_config.max_seq_len = 2048
        model_config.vocab_size = 500000
        model_config.expert_num = self.NUM_EXPERTS
        model_config.hidden_size = self.HIDDEN_SIZE
        model_config.moe_inter_size = self.MOE_INTERMEDIATE_SIZE

        parallelism_config = ParallelismConfig()
        parallelism_config.world_size = self.DP_SIZE * self.EP_SIZE
        parallelism_config.dp_size = self.DP_SIZE
        parallelism_config.tp_size = self.TP_SIZE
        parallelism_config.ep_size = self.EP_SIZE
        parallelism_config.dp_rank = 0
        parallelism_config.tp_rank = 0
        parallelism_config.ep_rank = 0
        parallelism_config.world_rank = 0
        parallelism_config.local_world_size = 1

        moe_config = MoeConfig()
        moe_config.ll_num_max_token = self.MAX_GENERATE_BATCH_SIZE
        return MoEConfigAdapter(
            model_config=model_config,
            parallelism_config=parallelism_config,
            moe_config=moe_config,
        )

    def test_mxfp8_deepep_executor_correctness(self):
        """End-to-end correctness: Mxfp8DeepepExecutor vs BF16 reference."""
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mxfp8_deepep_executor import (
            Mxfp8DeepepExecutor,
        )
        from rtp_llm.test.utils.numeric_util import calc_diff

        config = self._generate_config()
        payload, weights, ref_weights, _, _ = generate_deepep_normal_payload(config)

        ref_output = generate_ref_output_deepep_normal(payload, ref_weights)

        executor = Mxfp8DeepepExecutor(
            config,
            FusedMoEQuantConfig(
                quant_dtype=None,  # MXFP8 quantizes internally
                per_act_token_quant=False,
                per_out_ch_quant=False,
                block_shape=None,
            ),
            weights,
        )

        result = executor.execute(payload, "silu", None, None, False, None)

        diff = calc_diff(result.fused_expert_output, ref_output)
        # MXFP8 (1x32) quantization introduces some numerical noise;
        # 0.02 tolerance is generous for e4m3 + UE8M0 scales on small weights
        self.assertLess(diff, 0.02, f"calc_diff={diff:.6f} exceeds tolerance 0.02")

    def test_mxfp8_deepep_executor_perf(self):
        """Latency benchmark: Mxfp8DeepepExecutor expand block timing."""
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.mxfp8_deepep_executor import (
            Mxfp8DeepepExecutor,
        )

        config = self._generate_config()
        payload, weights, _, _, _ = generate_deepep_normal_payload(config)

        executor = Mxfp8DeepepExecutor(
            config,
            FusedMoEQuantConfig(
                quant_dtype=None,
                per_act_token_quant=False,
                per_out_ch_quant=False,
                block_shape=None,
            ),
            weights,
        )

        # Warmup (includes triton JIT compilation + deep_gemm warmup)
        iters = 50
        warmup = 10
        for _ in range(warmup):
            executor.execute(payload, "silu", None, None, False, None)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            executor.execute(payload, "silu", None, None, False, None)
        end.record()
        torch.cuda.synchronize()

        avg_ms = start.elapsed_time(end) / iters
        # Just print — no assertion on latency (varies by hardware)
        print(
            f"\n  Mxfp8DeepepExecutor: {avg_ms:.3f} ms/iter "
            f"(N_recv={payload.expert_x.size(0)}, E={config.expert_num // config.ep_size})"
        )


if __name__ == "__main__":
    unittest.main()
