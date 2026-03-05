# Adapt from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/ep_moe/kernels.py
# but make some modifications for RTP-LLM
# Licensed under the Apache License, Version 2.0
import logging
import math
import os
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)

_MOE_DEBUG_SYNC = os.environ.get("MOE_DEBUG_SYNC", "1") == "1"


def _cuda_sync_check(stage: str):
    """Sync CUDA and raise with stage info if a previous kernel had an error."""
    try:
        torch.cuda.synchronize()
    except RuntimeError as e:
        raise RuntimeError(
            f"CUDA error detected after '{stage}' in DeepGemmContinousExecutor: {e}"
        ) from e

from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
    configure_deep_gemm_num_sms,
    is_deep_gemm_e8m0_used,
    m_grouped_fp8_gemm_nt_contiguous,
    m_grouped_fp8_gemm_nt_masked,
)
from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    CombineForwardPayload,
    ExpertForwardPayload,
    FusedMoeExpertExecutor,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.type import ExecutorType
from rtp_llm.models_py.triton_kernels.common.activation import (
    create_packed_scale_tensor,
    silu_and_mul,
    silu_and_mul_masked_post_quant_packed_fwd,
    silu_mul_masked_fp8_post_quant_fwd,
)
from rtp_llm.models_py.triton_kernels.moe.ep_kernels import (
    ep_gather,
    ep_scatter,
    ep_scatter_v2,
    tma_align_input_scale,
)
from rtp_llm.models_py.utils.arch import get_num_device_sms, get_sm
from rtp_llm.models_py.utils.math import align, ceil_div
from rtp_llm.models_py.utils.memory import dispose_tensor
from rtp_llm.ops.compute_ops import trt_fp8_quantize_128
from rtp_llm.utils.model_weight import W


def align_up_math(n: int, alignment: int = 128) -> int:
    return int(math.ceil(n / alignment)) * alignment


class DeepGemmHybridExecutor(FusedMoeExpertExecutor):
    BLOCK_SIZE = 128
    EXPERT_ALIGNMENT = 128
    DEEPGEMM_BLOCK_SHAPE: list[int] = [128, 128]

    @classmethod
    def executor_type(cls) -> ExecutorType:
        return ExecutorType.DEEPGEMM_CONTINUOUS

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        """Check if DeepGemmHybridExecutor can handle the configuration"""
        from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import has_deep_gemm
        from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
            MoeConfigResolver,
        )

        resolver = MoeConfigResolver()
        quant_method = resolver.get_quant_method(config)
        checker.check(quant_method == "FP8_PER_BLOCK")
        checker.check(resolver.is_bf16(config))
        checker.check(has_deep_gemm())
        checker.check(get_sm()[0] >= 9)
        checker.check(not config.enable_cuda_graph)

    def __init__(
        self,
        config: MoEConfigAdapter,
        quant_config: FusedMoEQuantConfig,
        weights: Dict[str, torch.Tensor],
    ):
        super().__init__(config, quant_config, weights)

        self.ep_size = config.ep_size
        self.ep_rank = config.ep_rank
        self.num_experts = config.expert_num

        assert self.num_experts % self.ep_size == 0
        self.num_experts_per_partition = self.num_experts // self.ep_size
        self.start_expert_id = self.ep_rank * self.num_experts_per_partition
        self.end_expert_id = self.start_expert_id + self.num_experts_per_partition - 1

        self.top_k = config.moe_k
        self.activation = config.activation_type
        self.renormalize = True
        self.use_fp8_w8a8 = True
        self.use_block_quant = True

        self.masked_max_token_num = config.masked_max_token_num

        # 权重初始化
        self.w13_weight = weights[W.moe_w1]
        self.w2_weight = weights[W.moe_w2]
        self.w13_weight_scale_inv = weights[W.moe_s1]
        self.w2_weight_scale_inv = weights[W.moe_s2]
        self.w13_weight_scale = None
        self.w2_weight_scale = None

        self.E, self.N, self.K = self.w13_weight.size()
        assert self.N % 2 == 0
        assert self.w2_weight.size(0) == self.E
        assert self.w2_weight.size(1) == self.K
        assert self.w2_weight.size(2) == self.N // 2

        self.w13_weight_fp8 = (
            self.w13_weight,
            self.w13_weight_scale_inv,
        )
        self.w2_weight_fp8 = (
            self.w2_weight,
            self.w2_weight_scale_inv,
        )

        self.num_gemm_sms = get_num_device_sms()

    def execute(
        self,
        payload: ExpertForwardPayload,
        activation: str,
        expert_map: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        extra_expert_args: Optional[dict[str, Any]],
    ) -> CombineForwardPayload:
        assert payload.expert_x is not None, "hidden_states_fp8 is not initialized"
        token_num = payload.expert_x.shape[0]
        if token_num <= self.masked_max_token_num:
            return self.execute_masked(
                payload,
                activation,
                expert_map,
                a2_scale,
                apply_router_weight_on_input,
                extra_expert_args,
            )
        else:
            return self.execute_contiguous(
                payload,
                activation,
                expert_map,
                a2_scale,
                apply_router_weight_on_input,
                extra_expert_args,
            )

    def execute_masked(
        self,
        payload: ExpertForwardPayload,
        activation: str,
        expert_map: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        extra_expert_args: Optional[dict[str, Any]],
    ) -> CombineForwardPayload:
        assert payload.expert_x is not None, "hidden_states_fp8 is not initialized"
        assert (
            payload.expert_x_scale is not None
        ), "hidden_states_scale is not initialized"
        assert payload.expert_topk_ids is not None, "expert_topk_ids is not initialized"
        assert (
            payload.expert_topk_weights is not None
        ), "expert_topk_weights is not initialized"
        assert (
            payload.expert_tokens_meta is not None
        ), "expert_tokens_meta is not initialized"
        assert payload.expert_tokens_meta.expert_num_tokens is not None

        with configure_deep_gemm_num_sms(self.num_gemm_sms):
            hidden_states_fp8 = payload.expert_x
            hidden_states_scale = payload.expert_x_scale
            topk_idx = payload.expert_topk_ids
            topk_weights = payload.expert_topk_weights
            num_recv_tokens_per_expert = payload.expert_tokens_meta.expert_num_tokens

            token_num = hidden_states_fp8.shape[0]
            num_experts = num_recv_tokens_per_expert.shape[0]
            max_token_num = token_num * self.top_k
            token_num_mean_per_expert = ceil_div(max_token_num, num_experts)
            alignment = align(token_num, self.EXPERT_ALIGNMENT)
            expected_m = min(alignment, token_num_mean_per_expert)

            _, K = hidden_states_fp8.size()
            assert K == self.K, f"K: {K} != self.K: {self.K}"
            hidden_states_fp8_shape = hidden_states_fp8.shape
            hidden_states_fp8_device = hidden_states_fp8.device
            input_tensor = [
                torch.empty(
                    (self.num_experts_per_partition, alignment, self.K),
                    device=hidden_states_fp8_device,
                    dtype=hidden_states_fp8.dtype,
                ),
                (
                    torch.zeros(
                        [
                            self.num_experts_per_partition,
                            ceil_div(self.K // self.BLOCK_SIZE, 4),
                            alignment,
                        ],
                        device=hidden_states_fp8_device,
                        dtype=torch.int,
                    ).transpose(1, 2)
                    if is_deep_gemm_e8m0_used()
                    else torch.empty(
                        (
                            self.num_experts_per_partition,
                            alignment,
                            self.K // self.BLOCK_SIZE,
                        ),
                        device=hidden_states_fp8_device,
                        dtype=torch.float32,
                    )
                ),
            ]
            output_index = torch.empty_like(topk_idx)
            expert_start_loc = torch.empty_like(num_recv_tokens_per_expert)
            ep_scatter_v2(
                hidden_states_fp8,
                hidden_states_scale,
                topk_idx,
                alignment,
                expert_start_loc,
                input_tensor[0].view(
                    self.num_experts_per_partition * alignment, self.K
                ),
                input_tensor[1],
                output_index,
                scale_ue8m0=is_deep_gemm_e8m0_used(),
            )
            dispose_tensor(hidden_states_fp8)

            upgate_output = torch.empty(
                (self.num_experts_per_partition, alignment, self.N),
                device=hidden_states_fp8_device,
                dtype=torch.bfloat16,
            )
            # Gate and Up GroupGEMM-0
            m_grouped_fp8_gemm_nt_masked(
                (input_tensor[0], input_tensor[1]),
                self.w13_weight_fp8,
                upgate_output,
                num_recv_tokens_per_expert,
                expected_m,
                disable_ue8m0_cast=not is_deep_gemm_e8m0_used(),
            )

            del input_tensor
            # Allocate down_input
            down_input = torch.empty(
                (self.num_experts_per_partition, alignment, self.N // 2),
                device=hidden_states_fp8_device,
                dtype=torch.float8_e4m3fn,
            )

            # SM100 (compute capability 10.x) uses fused packed kernel for better performance
            # when UE8M0 scale format is enabled
            sm_major = torch.cuda.get_device_capability()[0]
            if (
                sm_major == 10
                and is_deep_gemm_e8m0_used()
                and self.N % (self.DEEPGEMM_BLOCK_SHAPE[0] * 2 * 4) == 0
            ):
                # Create packed scale tensor with proper layout for deep_gemm
                # Shape: (E, T, G // 4) where G = hidden_dim // 2 // group_size
                down_input_scale = create_packed_scale_tensor(
                    expert_num=self.num_experts_per_partition,
                    token_num_padded=alignment,
                    hidden_dim=self.N,
                    quant_group_size=self.DEEPGEMM_BLOCK_SHAPE[0],
                    device=hidden_states_fp8_device,
                )
                # Fused SiLU-and-mul + FP8 quantization with UE8M0 scale packing
                silu_and_mul_masked_post_quant_packed_fwd(
                    upgate_output,
                    down_input,
                    down_input_scale,
                    self.DEEPGEMM_BLOCK_SHAPE[0],
                    num_recv_tokens_per_expert,
                )
            else:
                # Standard path for other SM versions
                down_input_scale = torch.empty(
                    (
                        self.num_experts_per_partition,
                        alignment,
                        self.N // 2 // self.DEEPGEMM_BLOCK_SHAPE[0],
                    ),
                    device=hidden_states_fp8_device,
                    dtype=torch.float32,
                )
                # SiLU Activation
                silu_mul_masked_fp8_post_quant_fwd(
                    input=upgate_output,
                    output=down_input,
                    output_scale=down_input_scale,
                    quant_group_size=self.DEEPGEMM_BLOCK_SHAPE[0],
                    masked_m=num_recv_tokens_per_expert,
                    expected_m=expected_m,
                    scale_ue8m0=is_deep_gemm_e8m0_used(),
                )

            # Free upgate_output
            dispose_tensor(upgate_output)
            down_output = torch.empty(
                (self.num_experts_per_partition, alignment, self.K),
                device=hidden_states_fp8_device,
                dtype=torch.bfloat16,
            )

            # Down GroupGEMM-1
            m_grouped_fp8_gemm_nt_masked(
                (
                    down_input,
                    down_input_scale,
                ),
                self.w2_weight_fp8,
                down_output,
                num_recv_tokens_per_expert,
                expected_m,
                disable_ue8m0_cast=not is_deep_gemm_e8m0_used(),
            )

            # Free down_input and down_input_scale
            dispose_tensor(down_input)
            dispose_tensor(down_input_scale)

            gather_out = torch.empty(
                hidden_states_fp8_shape,
                device=hidden_states_fp8_device,
                dtype=torch.bfloat16,
            )
            ep_gather(
                down_output.view(self.num_experts_per_partition * alignment, self.K),
                topk_idx,
                topk_weights,
                output_index,
                gather_out,
            )
            return CombineForwardPayload(fused_expert_output=gather_out)

    def execute_contiguous(
        self,
        payload: ExpertForwardPayload,
        activation: str,
        expert_map: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        extra_expert_args: Optional[dict[str, Any]],
    ) -> CombineForwardPayload:
        assert payload.expert_x is not None, "hidden_states_fp8 is not initialized"
        assert (
            payload.expert_x_scale is not None
        ), "hidden_states_scale is not initialized"
        assert payload.expert_topk_ids is not None, "expert_topk_ids is not initialized"
        assert (
            payload.expert_topk_weights is not None
        ), "expert_topk_weights is not initialized"
        assert (
            payload.expert_tokens_meta is not None
        ), "expert_tokens_meta is not initialized"
        hidden_states_fp8 = payload.expert_x
        hidden_states_scale = payload.expert_x_scale
        topk_idx = payload.expert_topk_ids
        topk_weights = payload.expert_topk_weights
        if payload.expert_tokens_meta.expert_num_tokens_cpu is not None:
            num_recv_tokens_per_expert = (
                payload.expert_tokens_meta.expert_num_tokens_cpu
            )
        elif payload.expert_tokens_meta.expert_num_tokens is not None:
            num_recv_tokens_per_expert = (
                payload.expert_tokens_meta.expert_num_tokens.cpu().tolist()
            )
        else:
            raise ValueError(
                "expert_tokens_meta.expert_num_tokens or expert_tokens_meta.expert_num_tokens_cpu should be not None"
            )
        if isinstance(num_recv_tokens_per_expert, torch.Tensor):
            num_recv_tokens_per_expert = num_recv_tokens_per_expert.tolist()

        raw_tokens_per_expert = list(num_recv_tokens_per_expert)
        num_recv_tokens_per_expert = [
            align_up_math(x, self.EXPERT_ALIGNMENT) for x in num_recv_tokens_per_expert
        ]
        all_tokens: int = sum(num_recv_tokens_per_expert)

        num_experts_local = len(num_recv_tokens_per_expert)

        has_nan = torch.isnan(hidden_states_fp8.float()).any().item()
        has_inf = torch.isinf(hidden_states_fp8.float()).any().item()
        if has_nan or has_inf:
            logger.error(
                f"[DeepGemm SKIP] hidden_states_fp8 contains {'NaN' if has_nan else ''}{'&Inf' if has_inf else ''}, "
                f"M={hidden_states_fp8.shape[0]}, returning zeros to prevent CUDA OOB"
            )
            return CombineForwardPayload(
                fused_expert_output=torch.zeros(
                    hidden_states_fp8.shape,
                    device=hidden_states_fp8.device,
                    dtype=torch.bfloat16,
                ),
            )

        topk_max = topk_idx.max().item()
        topk_min = topk_idx.min().item()
        if topk_max >= num_experts_local or topk_min < -1:
            logger.error(
                f"[DeepGemm CLAMP] topk_ids out of range [{topk_min}, {topk_max}], "
                f"num_experts={num_experts_local}, clamping to valid range"
            )
            topk_idx = topk_idx.clamp(min=-1, max=num_experts_local - 1)

        if _MOE_DEBUG_SYNC:
            actual_counts = []
            for e in range(num_experts_local):
                actual_counts.append(int((topk_idx == e).sum().item()))
            logger.warning(
                f"[DeepGemm DEBUG] M={hidden_states_fp8.shape[0]}, K={hidden_states_fp8.shape[1]}, "
                f"num_experts={num_experts_local}, topk_ids range=[{topk_idx.min().item()}, {topk_idx.max().item()}], "
                f"raw_tokens_per_expert={raw_tokens_per_expert}, "
                f"aligned_tokens_per_expert={num_recv_tokens_per_expert}, "
                f"actual_counts_from_topk={actual_counts}, "
                f"all_tokens={all_tokens}, "
                f"w13_shape={list(self.w13_weight.shape)}, w2_shape={list(self.w2_weight.shape)}"
            )
            for e in range(num_experts_local):
                if actual_counts[e] > num_recv_tokens_per_expert[e]:
                    logger.error(
                        f"[DeepGemm OVERFLOW] expert {e}: actual_count={actual_counts[e]} > "
                        f"aligned_alloc={num_recv_tokens_per_expert[e]} (raw={raw_tokens_per_expert[e]}). "
                        f"Buffer will overflow!"
                    )

        if all_tokens <= 0:
            return CombineForwardPayload(
                fused_expert_output=torch.zeros(
                    hidden_states_fp8.shape,
                    device=hidden_states_fp8.device,
                    dtype=torch.bfloat16,
                ),
            )
        _, K = hidden_states_fp8.size()
        N = self.w13_weight.size(1)
        hidden_states_fp8_shape = hidden_states_fp8.shape
        hidden_states_fp8_device = hidden_states_fp8.device
        input_tensor = [
            torch.empty(
                (all_tokens, K),
                device=hidden_states_fp8.device,
                dtype=hidden_states_fp8.dtype,
            ),
            (
                torch.zeros(
                    [ceil_div(K // self.BLOCK_SIZE, 4), all_tokens],
                    device=hidden_states_fp8.device,
                    dtype=torch.int,
                ).transpose(0, 1)
                if is_deep_gemm_e8m0_used()
                else torch.empty(
                    (all_tokens, K // self.BLOCK_SIZE),
                    device=hidden_states_fp8.device,
                    dtype=torch.float32,
                )
            ),
        ]
        m_indices = torch.empty(
            all_tokens, device=hidden_states_fp8.device, dtype=torch.int32
        )
        output_index = torch.empty_like(topk_idx)
        num_recv_tokens_per_expert_gpu = torch.tensor(
            num_recv_tokens_per_expert,
            dtype=torch.int32,
            pin_memory=True,
            device="cpu",
        ).cuda(non_blocking=True)
        expert_start_loc = torch.empty_like(num_recv_tokens_per_expert_gpu)
        ep_scatter(
            hidden_states_fp8,
            hidden_states_scale,
            topk_idx,
            num_recv_tokens_per_expert_gpu,
            expert_start_loc,
            input_tensor[0],
            input_tensor[1],
            m_indices,
            output_index,
            scale_ue8m0=is_deep_gemm_e8m0_used(),
        )
        m_indices.clamp_(min=0, max=self.num_experts_per_partition - 1)
        if _MOE_DEBUG_SYNC:
            _cuda_sync_check("ep_scatter")
            final_locs = expert_start_loc.cpu().tolist()
            m_min, m_max = m_indices.min().item(), m_indices.max().item()
            logger.warning(
                f"[DeepGemm DEBUG] after ep_scatter: expert_start_loc={final_locs}, "
                f"m_indices range=[{m_min}, {m_max}], all_tokens={all_tokens}, "
                f"input_data shape={list(input_tensor[0].shape)}, "
                f"input_scale shape={list(input_tensor[1].shape)}, "
                f"input_scale stride={input_tensor[1].stride()}, "
                f"input_scale contiguous={input_tensor[1].is_contiguous()}"
            )
            if m_max >= self.num_experts_per_partition:
                logger.error(
                    f"[DeepGemm OOB] m_indices max={m_max} >= num_experts_per_partition="
                    f"{self.num_experts_per_partition}, will cause weight OOB!"
                )
            if final_locs and max(final_locs) > all_tokens:
                logger.error(
                    f"[DeepGemm OVERFLOW] expert_start_loc max={max(final_locs)} > "
                    f"all_tokens={all_tokens}, ep_scatter wrote beyond buffer!"
                )
        dispose_tensor(hidden_states_fp8)
        gateup_output = torch.empty(
            (all_tokens, N),
            device=hidden_states_fp8_device,
            dtype=torch.bfloat16,
        )
        if not is_deep_gemm_e8m0_used():
            input_tensor[1] = tma_align_input_scale(input_tensor[1])
        if _MOE_DEBUG_SYNC:
            logger.warning(
                f"[DeepGemm DEBUG] before gate_up GEMM: "
                f"a_data={list(input_tensor[0].shape)}, a_scale={list(input_tensor[1].shape)}, "
                f"a_scale_stride={input_tensor[1].stride()}, "
                f"b_data={list(self.w13_weight_fp8[0].shape)}, b_scale={list(self.w13_weight_fp8[1].shape)}, "
                f"output={list(gateup_output.shape)}, m_indices len={m_indices.shape[0]}"
            )
        m_grouped_fp8_gemm_nt_contiguous(
            (input_tensor[0], input_tensor[1]),
            self.w13_weight_fp8,
            gateup_output,
            m_indices,
            disable_ue8m0_cast=not is_deep_gemm_e8m0_used(),
        )
        if _MOE_DEBUG_SYNC:
            _cuda_sync_check("m_grouped_fp8_gemm_nt_contiguous(gate_up)")
        del input_tensor
        down_input = torch.empty(
            (
                all_tokens,
                N // 2,
            ),
            device=gateup_output.device,
            dtype=torch.bfloat16,
        )
        gateup_output = gateup_output.view(-1, N)
        silu_and_mul(down_input, gateup_output)
        if _MOE_DEBUG_SYNC:
            _cuda_sync_check("silu_and_mul")
        del gateup_output
        down_output = torch.empty(
            (all_tokens, K),
            device=hidden_states_fp8_device,
            dtype=torch.bfloat16,
        )
        if is_deep_gemm_e8m0_used():
            down_input_fp8, down_input_scale = sgl_per_token_group_quant_fp8(
                down_input,
                group_size=self.BLOCK_SIZE,
                column_major_scales=True,
                scale_tma_aligned=True,
                scale_ue8m0=is_deep_gemm_e8m0_used(),
            )
        else:
            down_input_fp8, down_input_scale = trt_fp8_quantize_128(down_input, False)
        if _MOE_DEBUG_SYNC:
            _cuda_sync_check("fp8_quantize(down_input)")
        del down_input
        if not is_deep_gemm_e8m0_used():
            down_input_scale = tma_align_input_scale(down_input_scale)
        if _MOE_DEBUG_SYNC:
            logger.warning(
                f"[DeepGemm DEBUG] before down GEMM: "
                f"a_data={list(down_input_fp8.shape)}, a_scale={list(down_input_scale.shape)}, "
                f"a_scale_stride={down_input_scale.stride()}, "
                f"b_data={list(self.w2_weight_fp8[0].shape)}, b_scale={list(self.w2_weight_fp8[1].shape)}, "
                f"output={list(down_output.shape)}, m_indices len={m_indices.shape[0]}"
            )
        m_grouped_fp8_gemm_nt_contiguous(
            (down_input_fp8, down_input_scale),
            self.w2_weight_fp8,
            down_output,
            m_indices,
            disable_ue8m0_cast=not is_deep_gemm_e8m0_used(),
        )
        if _MOE_DEBUG_SYNC:
            _cuda_sync_check("m_grouped_fp8_gemm_nt_contiguous(down)")
        del down_input_fp8, down_input_scale
        gather_out = torch.empty(
            hidden_states_fp8_shape,
            device=hidden_states_fp8_device,
            dtype=torch.bfloat16,
        )
        ep_gather(down_output, topk_idx, topk_weights, output_index, gather_out)
        return CombineForwardPayload(fused_expert_output=gather_out)
