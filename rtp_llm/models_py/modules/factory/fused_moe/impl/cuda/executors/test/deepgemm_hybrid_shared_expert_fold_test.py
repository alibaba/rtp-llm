"""Shared-expert fold contract for DeepGemmHybridExecutor (pure TP, FP8).

GenericMoeLayer hands the shared-expert output to the executor through
``SHARED_EXPERT_ACCUMULATE_ARG`` so the routed combine kernel (ep_gather)
accumulates it in-store instead of a separate bf16 add launch.  This test
pins, on the real deepgemm FP8 pipeline in both execute_masked and
execute_contiguous layouts:

1. the executor requests the fold (ep_gather receives ``accumulate_input``);
2. the folded output is bitwise identical to the unfolded output plus the
   shared tensor via the plain aten add (the exact numerics claim);
3. the unfolded routed output still matches a bf16 reference GEMM within the
   FP8 tolerance, so the equivalence is checked on meaningful values;
4. without the request, ep_gather runs with no accumulate input at all.

Run with bazel:
    bazel test //rtp_llm/models_py/modules/factory/fused_moe/impl/cuda/executors/test:deepgemm_hybrid_shared_expert_fold_test --config=cuda13
"""

import unittest
from typing import Any, Dict, Tuple
from unittest.mock import Mock, patch

import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import is_deep_gemm_e8m0_used
from rtp_llm.models_py.kernels.cuda.fp8_kernel.fp8_kernel import (
    per_block_cast_to_fp8,
    requant_weight_ue8m0,
    sgl_per_token_group_quant_fp8,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    ExpertForwardPayload,
    ExpertTokensMetadata,
    SHARED_EXPERT_ACCUMULATE_ARG,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.deepgemm_hybrid_executor import (
    DeepGemmHybridExecutor,
    ep_gather as real_ep_gather,
)
from rtp_llm.ops import MoeConfig, ParallelismConfig
from rtp_llm.test.utils.numeric_util import calc_diff
from rtp_llm.utils.model_weight import W

_EP_GATHER_TARGET = (
    "rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors."
    "deepgemm_hybrid_executor.ep_gather"
)

NUM_EXPERTS = 8
NUM_TOKENS = 64
TOPK = 4
HIDDEN_SIZE = 2048
MOE_INTERMEDIATE_SIZE = 512
DIFF_THRESHOLD = 0.003


def _generate_config(masked_max_token_num: int) -> MoEConfigAdapter:
    model_config = ModelConfig()
    model_config.attn_config.head_num = 2
    model_config.attn_config.size_per_head = 128
    model_config.num_layers = 2
    model_config.max_seq_len = 2048
    model_config.vocab_size = 500000
    model_config.expert_num = NUM_EXPERTS
    model_config.moe_k = TOPK
    model_config.hidden_size = HIDDEN_SIZE
    model_config.moe_inter_size = MOE_INTERMEDIATE_SIZE

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
    moe_config.ll_num_max_token = NUM_TOKENS
    moe_config.masked_max_token_num = masked_max_token_num
    return MoEConfigAdapter(
        model_config=model_config,
        parallelism_config=parallelism_config,
        moe_config=moe_config,
        enable_cuda_graph=False,
    )


def _quantize_input(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mirror PureTpRouterFp8PerBlock._do_quant for the bf16 stream."""
    if is_deep_gemm_e8m0_used():
        return sgl_per_token_group_quant_fp8(
            x,
            128,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=True,
        )
    from rtp_llm.ops.compute_ops import trt_fp8_quantize_128

    return trt_fp8_quantize_128(x, False)


def _make_routing_inputs() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Deterministic bf16 stream, routing ids and weights."""
    generator = torch.Generator(device="cpu").manual_seed(2024)
    x = (
        (torch.randn(NUM_TOKENS, HIDDEN_SIZE, generator=generator) * 0.1)
        .cuda()
        .to(torch.bfloat16)
    )
    topk_ids = (
        torch.randint(0, NUM_EXPERTS, (NUM_TOKENS, TOPK), generator=generator)
        .to(torch.int64)
        .cuda()
    )
    raw_weights = torch.rand((NUM_TOKENS, TOPK), generator=generator).cuda()
    topk_weights = raw_weights / raw_weights.sum(dim=-1, keepdim=True)
    return x, topk_ids, topk_weights


def _make_payload(
    x: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor
) -> ExpertForwardPayload:
    """Fresh pure-TP payload; eager execute() disposes expert_x per run."""
    expert_x, expert_x_scale = _quantize_input(x)
    expert_num_tokens = (
        torch.bincount(topk_ids.flatten(), minlength=NUM_EXPERTS)
        .to(torch.int32)
        .cuda()
    )
    return ExpertForwardPayload(
        expert_x=expert_x,
        expert_x_scale=expert_x_scale,
        expert_x_origin_dtype=torch.bfloat16,
        expert_topk_ids=topk_ids,
        expert_topk_weights=topk_weights,
        expert_tokens_meta=ExpertTokensMetadata(
            expert_num_tokens=expert_num_tokens,
            expert_num_tokens_cpu=expert_num_tokens.tolist(),
        ),
    )


def _make_weights() -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """FP8 block-quantized expert weights (plus the bf16 originals for the ref)."""
    generator = torch.Generator(device="cpu").manual_seed(7)
    n = MOE_INTERMEDIATE_SIZE * 2
    w1 = (
        (torch.randn(NUM_EXPERTS, n, HIDDEN_SIZE, generator=generator) * 0.1)
        .cuda()
        .to(torch.bfloat16)
    )
    w2 = (
        (torch.randn(NUM_EXPERTS, HIDDEN_SIZE, n // 2, generator=generator) * 0.1)
        .cuda()
        .to(torch.bfloat16)
    )
    weights = {W.moe_w1: w1, W.moe_w2: w2}
    new_w1 = torch.empty_like(w1, dtype=torch.float8_e4m3fn)
    new_w2 = torch.empty_like(w2, dtype=torch.float8_e4m3fn)
    weights[W.moe_s1] = torch.empty(
        (NUM_EXPERTS, n // 128, HIDDEN_SIZE // 128),
        device="cuda",
        dtype=torch.float32,
    )
    weights[W.moe_s2] = torch.empty(
        (NUM_EXPERTS, HIDDEN_SIZE // 128, n // 2 // 128),
        device="cuda",
        dtype=torch.float32,
    )
    for e in range(NUM_EXPERTS):
        new_w1[e], weights[W.moe_s1][e] = per_block_cast_to_fp8(w1[e], use_ue8m0=False)
        new_w2[e], weights[W.moe_s2][e] = per_block_cast_to_fp8(w2[e], use_ue8m0=False)
    if is_deep_gemm_e8m0_used():
        new_w1, weights[W.moe_s1] = requant_weight_ue8m0(new_w1, weights[W.moe_s1])
        new_w2, weights[W.moe_s2] = requant_weight_ue8m0(new_w2, weights[W.moe_s2])
    weights[W.moe_w1] = new_w1
    weights[W.moe_w2] = new_w2
    return weights, w1, w2


def _reference_routed_output(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
) -> torch.Tensor:
    """bf16 reference: per-expert SiGLU GEMMs, fp32 topk accumulation."""
    n = MOE_INTERMEDIATE_SIZE * 2
    out_fp32 = torch.zeros(NUM_TOKENS, HIDDEN_SIZE, device=x.device, dtype=torch.float32)
    for e in range(NUM_EXPERTS):
        token_idx, slot = (topk_ids == e).nonzero(as_tuple=True)
        if token_idx.numel() == 0:
            continue
        workspace = x[token_idx] @ w1[e].transpose(0, 1)
        gate = workspace[..., n // 2 :].to(torch.float32)
        value = workspace[..., : n // 2].to(torch.float32)
        activated = (gate * torch.sigmoid(gate) * value).to(torch.bfloat16)
        down = activated @ w2[e].transpose(0, 1)
        contribution = down.to(torch.float32) * topk_weights[token_idx, slot].unsqueeze(1)
        out_fp32[token_idx] += contribution
    return out_fp32.to(torch.bfloat16)


class DeepGemmHybridSharedExpertFoldTest(unittest.TestCase):
    def setUp(self) -> None:
        if not torch.cuda.is_available():
            self.skipTest("CUDA required for deepgemm executor test")
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        self.shared_expert_output = (
            (torch.randn(NUM_TOKENS, HIDDEN_SIZE) * 0.1).cuda().to(torch.bfloat16)
        )
        self.weights, self.ref_w1, self.ref_w2 = _make_weights()
        self.x, self.topk_ids, self.topk_weights = _make_routing_inputs()

    def _make_executor(self, config: MoEConfigAdapter) -> DeepGemmHybridExecutor:
        return DeepGemmHybridExecutor(
            config,
            FusedMoEQuantConfig(
                quant_dtype=torch.float8_e4m3fn,
                per_act_token_quant=False,
                per_out_ch_quant=False,
                block_shape=[128, 128],
            ),
            self.weights,
        )

    def _execute(
        self,
        executor: DeepGemmHybridExecutor,
        extra_expert_args: Dict[str, Any],
    ) -> torch.Tensor:
        payload = _make_payload(self.x, self.topk_ids, self.topk_weights)
        combine_payload = executor.execute(
            payload, "SiGLU", None, None, False, extra_expert_args
        )
        return combine_payload.fused_expert_output

    def _spy_ep_gather(self):
        return patch(_EP_GATHER_TARGET, new=Mock(wraps=real_ep_gather))

    def _assert_fold_contract(self, masked_max_token_num: int) -> None:
        config = _generate_config(masked_max_token_num)
        executor = self._make_executor(config)

        base = self._execute(executor, None)
        self.assertEqual(base.shape, (NUM_TOKENS, HIDDEN_SIZE))
        self.assertEqual(base.dtype, torch.bfloat16)

        # The unfolded routed output must be a meaningful MoE result before
        # the fold equivalence is worth anything.
        ref = _reference_routed_output(
            self.x, self.topk_ids, self.topk_weights, self.ref_w1, self.ref_w2
        )
        diff = calc_diff(base, ref)
        self.assertLess(
            diff,
            DIFF_THRESHOLD,
            f"routed output deviates from reference (diff={diff})",
        )

        # Folded execution: ep_gather must receive the shared tensor, and the
        # result must be bitwise identical to the unfolded output + aten add.
        with self._spy_ep_gather() as ep_gather_spy:
            folded = self._execute(
                executor, {SHARED_EXPERT_ACCUMULATE_ARG: self.shared_expert_output}
            )
            self.assertGreaterEqual(ep_gather_spy.call_count, 1)
            accumulate_seen = [
                call.kwargs.get("accumulate_input")
                for call in ep_gather_spy.call_args_list
            ]
            self.assertTrue(
                any(t is self.shared_expert_output for t in accumulate_seen),
                "executor did not request the shared-expert fold in ep_gather",
            )

        expected = base + self.shared_expert_output
        self.assertTrue(
            torch.equal(folded, expected),
            "folded executor output must be bitwise identical to the "
            "unfolded output plus the plain shared-expert add",
        )
        # Without the request, ep_gather must run with no accumulate input.
        with self._spy_ep_gather() as ep_gather_spy:
            self._execute(executor, None)
            for call in ep_gather_spy.call_args_list:
                self.assertIsNone(call.kwargs.get("accumulate_input"))

    def test_masked_path_folds_shared_expert_bitwise(self) -> None:
        self._assert_fold_contract(masked_max_token_num=128)

    def test_contiguous_path_folds_shared_expert_bitwise(self) -> None:
        self._assert_fold_contract(masked_max_token_num=0)


if __name__ == "__main__":
    unittest.main()
