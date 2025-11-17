# Adapt from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/moe/ep_moe/kernels.py
# but make some modifications for RTP-LLM
# Licensed under the Apache License, Version 2.0
import math
from typing import Any, Dict, Optional

import torch

from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.models_py.modules.common.moe.fused_moe import (
    ExpertForwardPayload,
    FusedMoeExpertExecutor,
)
from rtp_llm.models_py.modules.factory.fused_moe.quant_config import FusedMoEQuantConfig
from rtp_llm.models_py.modules.factory.fused_moe.type import ExecutorType
from rtp_llm.models_py.modules.quantization.deepgemm_wrapper import (
    is_deep_gemm_e8m0_used,
    m_grouped_fp8_gemm_nt_contiguous,
)
from rtp_llm.models_py.modules.utils import ceil_div, dispose_tensor
from rtp_llm.models_py.triton_kernels.common.activation import silu_and_mul
from rtp_llm.models_py.triton_kernels.moe.ep_kernels import (
    ep_gather,
    ep_scatter,
    tma_align_input_scale,
)
from rtp_llm.ops.compute_ops import trt_fp8_quantize_128
from rtp_llm.utils.model_weight import W

BLOCK_SIZE = 128
EXPERT_ALIGNMENT = 128


def align_up_math(n: int, alignment: int = 128) -> int:
    return int(math.ceil(n / alignment)) * alignment


class DeepGemmContinousExecutor(FusedMoeExpertExecutor):
    @classmethod
    def executor_type(cls):
        return ExecutorType.DEEPGEMM_CONTINUOUS

    @classmethod
    def check_conditions(cls, checker: Any, config: GptInitModelParameters) -> None:
        """Check if DeepGemmContinousExecutor can handle the configuration"""
        from rtp_llm.models_py.modules.factory.fused_moe.config_resolver import (
            MoeConfigResolver,
        )
        from rtp_llm.models_py.modules.quantization.deepgemm_wrapper import (
            has_deep_gemm,
        )

        resolver = MoeConfigResolver()
        quant_method = resolver.get_quant_method(config)
        checker.check(quant_method == "FP8_PER_BLOCK")
        checker.check(has_deep_gemm())

    def __init__(
        self,
        config: GptInitModelParameters,
        weights: Dict[str, torch.Tensor],
    ):
        super().__init__(FusedMoEQuantConfig())
        self.config = config
        self.ep_size = config.ep_size
        self.ep_rank = config.ep_rank
        self.num_experts = config.expert_num
        assert self.num_experts % self.ep_size == 0
        self.num_experts_per_partition = self.num_experts // self.ep_size
        self.start_expert_id = self.ep_rank * self.num_experts_per_partition
        self.end_expert_id = self.start_expert_id + self.num_experts_per_partition - 1
        self.top_k = config.moe_k
        self.intermediate_size = config.moe_inter_padding_size
        self.activation = config.activation_type.lower()
        self.renormalize = True
        self.use_fp8_w8a8 = True
        self.use_block_quant = True
        # 权重初始化
        self.w13_weight = weights[W.moe_w1]
        self.w2_weight = weights[W.moe_w2]
        self.w13_weight_scale_inv = weights[W.moe_s1]
        self.w2_weight_scale_inv = weights[W.moe_s2]
        self.w13_weight_scale = None
        self.w2_weight_scale = None
        self.w13_weight_fp8 = (
            self.w13_weight,
            self.w13_weight_scale_inv,
        )
        self.w2_weight_fp8 = (
            self.w2_weight,
            self.w2_weight_scale_inv,
        )

    @property
    def local_num_experts(self) -> int:
        return self.num_experts_per_partition

    def execute(
        self,
        payload: ExpertForwardPayload,
        activation: str,
        expert_map: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        extra_expert_args: Optional[dict[str, Any]],
    ) -> torch.Tensor:
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
        num_recv_tokens_per_expert = [
            align_up_math(x, EXPERT_ALIGNMENT) for x in num_recv_tokens_per_expert
        ]
        all_tokens: int = sum(num_recv_tokens_per_expert)
        if all_tokens <= 0:
            return torch.zeros(
                hidden_states_fp8.shape,
                device=hidden_states_fp8.device,
                dtype=torch.bfloat16,
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
                    [ceil_div(K // BLOCK_SIZE, 4), all_tokens],
                    device=hidden_states_fp8.device,
                    dtype=torch.int,
                ).transpose(0, 1)
                if is_deep_gemm_e8m0_used()
                else torch.empty(
                    (all_tokens, K // BLOCK_SIZE),
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
        )
        dispose_tensor(hidden_states_fp8)
        gateup_output = torch.empty(
            (all_tokens, N),
            device=hidden_states_fp8_device,
            dtype=torch.bfloat16,
        )
        if not is_deep_gemm_e8m0_used():
            input_tensor[1] = tma_align_input_scale(input_tensor[1])
        m_grouped_fp8_gemm_nt_contiguous(
            (input_tensor[0], input_tensor[1]),
            self.w13_weight_fp8,
            gateup_output,
            m_indices,
            disable_ue8m0_cast=True,
        )
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
        del gateup_output
        down_output = torch.empty(
            (all_tokens, K),
            device=hidden_states_fp8_device,
            dtype=torch.bfloat16,
        )
        down_input_fp8, down_input_scale = trt_fp8_quantize_128(down_input, False)
        del down_input
        if not is_deep_gemm_e8m0_used():
            down_input_scale = tma_align_input_scale(down_input_scale)
        m_grouped_fp8_gemm_nt_contiguous(
            (down_input_fp8, down_input_scale),
            self.w2_weight_fp8,
            down_output,
            m_indices,
            disable_ue8m0_cast=True,
        )
        del down_input_fp8, down_input_scale
        gather_out = torch.empty(
            hidden_states_fp8_shape,
            device=hidden_states_fp8_device,
            dtype=torch.bfloat16,
        )
        ep_gather(down_output, topk_idx, topk_weights, output_index, gather_out)
        return gather_out
