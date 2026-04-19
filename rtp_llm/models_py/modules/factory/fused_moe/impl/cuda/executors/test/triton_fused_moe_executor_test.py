"""Unit tests for the sglang TritonRunnerCore port (Triton fused MoE).

Covers two layers:

1. Direct test of ``fused_experts_impl`` for the BF16 (no quant) and FP8
   per-block W8A8 paths against a per-token reference implementation. This is
   the actual code path observed in sglang's MTP profiling timeline:
   ``fused_moe_kernel`` + ``moe_align_block_size_kernel`` + (custom triton
   reduce / torch.compile reduce for small tokens).

2. End-to-end executor test through ``TritonFusedMoeExecutor`` with a single
   GPU / single expert-parallel rank ``MoEConfigAdapter``.

The tests assume a CUDA-capable device and skip otherwise.
"""

import random
import unittest
from typing import Tuple

import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models_py.kernels.cuda.fp8_kernel.fp8_kernel import per_block_cast_to_fp8
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    ExpertForwardPayload,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.triton_fused_moe_executor import (
    TritonFusedMoeExecutor,
)
from rtp_llm.models_py.triton_kernels.moe.fused_moe_triton import fused_experts_impl
from rtp_llm.ops import MoeConfig, ParallelismConfig
from rtp_llm.test.utils.numeric_util import calc_diff
from rtp_llm.utils.model_weight import W


# ---------------------------------------------------------------------------
# Reference (pure torch, per-token expert dispatch).
# ---------------------------------------------------------------------------
def _ref_fused_experts(
    hidden_states: torch.Tensor,  # (M, K)  bf16
    w1: torch.Tensor,  # (E, 2N, K) bf16
    w2: torch.Tensor,  # (E, K, N)  bf16
    topk_weights: torch.Tensor,  # (M, top_k) fp32
    topk_ids: torch.Tensor,  # (M, top_k) int
    activation: str = "silu",
) -> torch.Tensor:
    M, K = hidden_states.shape
    E, two_n, _ = w1.shape
    N = two_n // 2
    out = torch.zeros((M, K), device=hidden_states.device, dtype=torch.bfloat16)

    hs = hidden_states.to(torch.float32)
    w1f = w1.to(torch.float32)
    w2f = w2.to(torch.float32)

    for tok in range(M):
        acc = torch.zeros((K,), device=hidden_states.device, dtype=torch.float32)
        for k in range(topk_ids.shape[1]):
            eid = int(topk_ids[tok, k].item())
            if eid < 0 or eid >= E:
                continue
            weight = float(topk_weights[tok, k].item())
            x = hs[tok]
            up = x @ w1f[eid].t()  # (2N,)
            # RTP-LLM convention: value = first half, gate = second half
            value = up[:N]
            gate = up[N:]
            if activation == "silu":
                act = gate * torch.sigmoid(gate)
            else:
                raise NotImplementedError(activation)
            inter = act * value  # (N,)
            down = inter @ w2f[eid].t()  # (K,)
            acc = acc + weight * down
        out[tok] = acc.to(torch.bfloat16)
    return out


def _make_random_routing(
    M: int, top_k: int, num_experts: int, device: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (topk_weights fp32, topk_ids int32) with normalized weights."""
    logits = torch.randn((M, num_experts), device=device, dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(logits, k=top_k, dim=-1)
    topk_weights = torch.softmax(topk_weights, dim=-1).to(torch.float32)
    return topk_weights.contiguous(), topk_ids.to(torch.int32).contiguous()


# ---------------------------------------------------------------------------
# Direct fused_experts_impl tests.
# ---------------------------------------------------------------------------
class FusedExpertsImplTestBase:
    M = 16  # small token count -> exercises torch.compile reduce branch
    K = 256
    N = 384  # intermediate (single side)
    NUM_EXPERTS = 8
    TOP_K = 4  # > 2 so reduce kernels (not the torch.add fast path) run

    def setUp(self):
        torch.manual_seed(123)
        torch.cuda.manual_seed(123)
        random.seed(123)

    def _gen_inputs(self, dtype: torch.dtype = torch.bfloat16):
        device = "cuda"
        hs = (
            torch.randn((self.M, self.K), device=device, dtype=torch.float32) * 0.1
        ).to(dtype)
        w1 = (
            torch.randn(
                (self.NUM_EXPERTS, 2 * self.N, self.K),
                device=device,
                dtype=torch.float32,
            )
            * (1.0 / (self.K**0.5))
        ).to(dtype)
        w2 = (
            torch.randn(
                (self.NUM_EXPERTS, self.K, self.N),
                device=device,
                dtype=torch.float32,
            )
            * (1.0 / (self.N**0.5))
        ).to(dtype)
        topk_weights, topk_ids = _make_random_routing(
            self.M, self.TOP_K, self.NUM_EXPERTS, device
        )
        return hs, w1, w2, topk_weights, topk_ids

    def test_bf16(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        hs, w1, w2, topk_weights, topk_ids = self._gen_inputs(torch.bfloat16)

        ref = _ref_fused_experts(hs, w1, w2, topk_weights, topk_ids, "silu")
        out = fused_experts_impl(
            hidden_states=hs,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=False,
            activation="silu",
            apply_router_weight_on_input=False,
            use_fp8_w8a8=False,
            per_channel_quant=False,
            w1_scale=None,
            w2_scale=None,
            a1_scale=None,
            a2_scale=None,
            block_shape=None,
            filter_expert=False,
        )
        diff = calc_diff(out, ref)
        self.assertLess(diff, 5e-3, f"BF16 fused_experts diff too high: {diff}")

    def test_bf16_large_tokens_triton_reduce(self):
        """Token count > 32 forces the triton moe_sum_reduce branch."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        # Override M for this test.
        saved_M = self.M
        try:
            self.M = 64
            hs, w1, w2, topk_weights, topk_ids = self._gen_inputs(torch.bfloat16)
            ref = _ref_fused_experts(hs, w1, w2, topk_weights, topk_ids, "silu")
            out = fused_experts_impl(
                hidden_states=hs,
                w1=w1,
                w2=w2,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                inplace=False,
                activation="silu",
                apply_router_weight_on_input=False,
                use_fp8_w8a8=False,
                per_channel_quant=False,
                w1_scale=None,
                w2_scale=None,
                a1_scale=None,
                a2_scale=None,
                block_shape=None,
                filter_expert=False,
            )
            diff = calc_diff(out, ref)
            self.assertLess(diff, 5e-3, f"BF16 (triton reduce) diff too high: {diff}")
        finally:
            self.M = saved_M

    def test_fp8_per_block(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        # Per-block fp8 requires K and N divisible by 128.
        block = 128
        if self.K % block != 0 or self.N % block != 0:
            self.skipTest(f"K/N not divisible by {block}")
        hs, w1_bf16, w2_bf16, topk_weights, topk_ids = self._gen_inputs(torch.bfloat16)

        # Quantize weights per [128, 128].
        w1_fp8 = torch.zeros_like(w1_bf16, dtype=torch.float8_e4m3fn)
        w2_fp8 = torch.zeros_like(w2_bf16, dtype=torch.float8_e4m3fn)
        w1_scale = torch.zeros(
            (self.NUM_EXPERTS, (2 * self.N) // block, self.K // block),
            device="cuda",
            dtype=torch.float32,
        )
        w2_scale = torch.zeros(
            (self.NUM_EXPERTS, self.K // block, self.N // block),
            device="cuda",
            dtype=torch.float32,
        )
        for i in range(self.NUM_EXPERTS):
            w1_fp8[i], w1_scale[i] = per_block_cast_to_fp8(w1_bf16[i], use_ue8m0=False)
            w2_fp8[i], w2_scale[i] = per_block_cast_to_fp8(w2_bf16[i], use_ue8m0=False)

        # Reference uses the quantized weights dequantized back, to get an
        # apples-to-apples target for the fp8 kernel.
        def _dequant(w_fp8, w_s):
            E, R, C = w_fp8.shape
            wf = w_fp8.to(torch.float32)
            wf = wf.view(E, R // block, block, C // block, block)
            ws = w_s.unsqueeze(2).unsqueeze(4)  # (E, R/b, 1, C/b, 1)
            wf = wf * ws
            return wf.view(E, R, C).to(torch.bfloat16)

        w1_dq = _dequant(w1_fp8, w1_scale)
        w2_dq = _dequant(w2_fp8, w2_scale)
        ref = _ref_fused_experts(hs, w1_dq, w2_dq, topk_weights, topk_ids, "silu")

        out = fused_experts_impl(
            hidden_states=hs,
            w1=w1_fp8,
            w2=w2_fp8,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=False,
            activation="silu",
            apply_router_weight_on_input=False,
            use_fp8_w8a8=True,
            per_channel_quant=False,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            a1_scale=None,
            a2_scale=None,
            block_shape=[block, block],
            filter_expert=False,
        )
        diff = calc_diff(out, ref)
        # FP8 quant introduces ~1-3% error.
        self.assertLess(
            diff, 5e-2, f"FP8 per-block fused_experts diff too high: {diff}"
        )


class FusedExpertsImplTest(FusedExpertsImplTestBase, unittest.TestCase):
    pass


# ---------------------------------------------------------------------------
# End-to-end executor test (single GPU, EP=1, TP=1).
# ---------------------------------------------------------------------------
class TritonFusedMoeExecutorE2ETest(unittest.TestCase):
    NUM_EXPERTS = 8
    HIDDEN_SIZE = 256
    MOE_INTERMEDIATE_SIZE = 384
    TOP_K = 4
    M = 16

    def setUp(self):
        torch.manual_seed(7)
        torch.cuda.manual_seed(7)

    def _make_config(self) -> MoEConfigAdapter:
        model_config = ModelConfig()
        model_config.attn_config.head_num = 2
        model_config.attn_config.size_per_head = 128
        model_config.num_layers = 2
        model_config.max_seq_len = 2048
        model_config.vocab_size = 1000
        model_config.expert_num = self.NUM_EXPERTS
        model_config.hidden_size = self.HIDDEN_SIZE
        model_config.moe_inter_size = self.MOE_INTERMEDIATE_SIZE
        model_config.moe_k = self.TOP_K

        parallelism_config = ParallelismConfig()
        parallelism_config.world_size = 1
        parallelism_config.dp_size = 1
        parallelism_config.tp_size = 1
        parallelism_config.ep_size = 1
        parallelism_config.dp_rank = 0
        parallelism_config.tp_rank = 0
        parallelism_config.ep_rank = 0
        parallelism_config.world_rank = 0
        parallelism_config.local_world_size = 1

        moe_config = MoeConfig()
        moe_config.ll_num_max_token = self.M
        return MoEConfigAdapter(
            model_config=model_config,
            parallelism_config=parallelism_config,
            moe_config=moe_config,
        )

    def test_executor_bf16(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        config = self._make_config()
        device = "cuda"

        N2 = self.MOE_INTERMEDIATE_SIZE * 2
        hs = (
            torch.randn((self.M, self.HIDDEN_SIZE), device=device, dtype=torch.float32)
            * 0.1
        ).to(torch.bfloat16)
        w1 = (
            torch.randn(
                (self.NUM_EXPERTS, N2, self.HIDDEN_SIZE),
                device=device,
                dtype=torch.float32,
            )
            * (1.0 / (self.HIDDEN_SIZE**0.5))
        ).to(torch.bfloat16)
        w2 = (
            torch.randn(
                (self.NUM_EXPERTS, self.HIDDEN_SIZE, self.MOE_INTERMEDIATE_SIZE),
                device=device,
                dtype=torch.float32,
            )
            * (1.0 / (self.MOE_INTERMEDIATE_SIZE**0.5))
        ).to(torch.bfloat16)
        topk_weights, topk_ids = _make_random_routing(
            self.M, self.TOP_K, self.NUM_EXPERTS, device
        )

        ref = _ref_fused_experts(hs, w1, w2, topk_weights, topk_ids, "silu")

        weights = {W.moe_w1: w1, W.moe_w2: w2}
        executor = TritonFusedMoeExecutor(
            config,
            FusedMoEQuantConfig(quant_dtype=None),
            weights,
        )
        payload = ExpertForwardPayload(
            expert_x=hs,
            expert_x_scale=None,
            expert_x_origin_dtype=torch.bfloat16,
            expert_topk_ids=topk_ids,
            expert_topk_weights=topk_weights,
            expert_tokens_meta=None,
        )
        combine = executor.execute(payload, "silu", None, None, False, None)
        diff = calc_diff(combine.fused_expert_output, ref)
        self.assertLess(diff, 5e-3, f"Executor BF16 diff too high: {diff}")


if __name__ == "__main__":
    unittest.main()
