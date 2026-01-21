from typing import Any, Optional

import torch

from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import is_deep_gemm_e8m0_used
from rtp_llm.models_py.kernels.cuda.fp8_kernel import (
    per_token_cast_to_fp8,
    sgl_per_token_group_quant_fp8,
)
from rtp_llm.models_py.distributed.collective_torch import Group, all_reduce, all_gather
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    ExpertForwardPayload,
    ExpertTokensMetadata,
    FusedMoeDataRouter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.type import RouterType

from rtp_llm.models_py.kernels.cuda.fp8_kernel import scaled_fp8_per_token_quant
from rtp_llm.models_py.triton_kernels.moe.ep_kernels import (
    recompute_topk_ids_sum_expert_count,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import MoEConfigAdapter
from rtp_llm.ops.compute_ops import trt_fp8_quantize_128


class PureTpRouter(FusedMoeDataRouter):
    @classmethod
    def router_type(cls):
        return RouterType.PURE_TP

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        """Check if PureTpRouter can handle the configuration"""
        from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
            MoeConfigResolver,
        )

        resolver = MoeConfigResolver()
        checker.check(resolver.is_single_gpu(config) or resolver.is_tp_equal_ep(config))

    def __init__(
        self,
        config: MoEConfigAdapter,
        use_fp8: bool = True,
        async_mode: bool = False,
        expert_alignment: int = 128,
    ):
        self.config = config
        self.tp_size = config.tp_size
        self.tp_rank = config.tp_rank
        self.dp_size = config.dp_size
        self.dp_rank = config.dp_rank
        self.ep_size = config.ep_size
        self.ep_rank = config.ep_rank
        self.expert_num = config.expert_num
        self.expert_num_per_rank = self.expert_num // self.ep_size
        self.expert_start_id = self.ep_rank * self.expert_num_per_rank
        # self.top_k = config.moe_topk_group
        self.top_k = 8
        self.use_fp8 = use_fp8
        self.async_mode = async_mode
        self.expert_alignment = expert_alignment
        self.save_first = False
        if self.async_mode:
            raise ValueError("DeepEPNormal not supports async mode now")

    def prepare(
        self,
        a1: torch.Tensor,
        a1_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        quant_config: FusedMoEQuantConfig,
    ) -> ExpertForwardPayload:
        # recompute top_k ids to current expert, mask out of range expert to -1
        if self.use_fp8 and is_deep_gemm_e8m0_used():
            expert_x, expert_x_scale = sgl_per_token_group_quant_fp8(
                a1,
                128,
                column_major_scales=True,
                scale_tma_aligned=True,
                scale_ue8m0=True,
            )
        elif self.use_fp8:
            if quant_config.is_per_act_token:
                expert_x, expert_x_scale = scaled_fp8_per_token_quant(a1, None)
            else:
                expert_x, expert_x_scale = trt_fp8_quantize_128(a1, False)
        else:
            expert_x = a1
            expert_x_scale = None
        adjusted_topk_ids, num_recv_tokens_per_expert = (
            recompute_topk_ids_sum_expert_count(
                topk_ids, self.expert_start_id, self.expert_num_per_rank
            )
        )
        # if not self.save_first:
        #     my_dict = {
        #         "hidden_states": a1,
        #         "topk_ids": adjusted_topk_ids,
        #         "expert_num_tokens": num_recv_tokens_per_expert,
        #         "topk_weights": topk_weights
        #     }
        #     torch.save(my_dict, "cutedsl_data/cutedsl_nodp_prepare.pt")
            #self.save_first = True
        # self.top_k
        if a1.dim() == 2:
            num_tokens, hidden_size = a1.shape
            self.num_tokens = num_tokens
            self.hidden_size = hidden_size
            hidden_states_expanded = (
                a1.view(num_tokens, -1, hidden_size)
                .repeat(1, self.top_k, 1)
                .reshape(-1, hidden_size)
            )
            hidden_states_3d = torch.empty(
                (self.expert_num, max(num_recv_tokens_per_expert), hidden_states_expanded.shape[1]), dtype=a1.dtype,
                device=a1.device
            )
            for i in range(self.expert_num):
                hidden_states_3d[i, : num_recv_tokens_per_expert[i], :] = hidden_states_expanded[adjusted_topk_ids.view(-1) == i]
            expert_x = hidden_states_3d
            self.expert_num_tokens = num_recv_tokens_per_expert
        
        return ExpertForwardPayload(
            expert_x,
            a1.dtype,
            expert_x_scale,
            ExpertTokensMetadata(
                expert_num_tokens=num_recv_tokens_per_expert,
                expert_num_tokens_cpu=None,
            ),
            adjusted_topk_ids,
            topk_weights,
        )
        return ExpertForwardPayload(
            expert_x,
            a1.dtype,
            expert_x_scale,
            ExpertTokensMetadata(num_recv_tokens_per_expert, None),
            adjusted_topk_ids,
            topk_weights,
        )

    def finalize(
        self,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        extra_finalize_args: Optional[dict[str, Any]],
    ) -> torch.Tensor:
        # combine
        topk_ids = topk_ids.to(torch.int64)

        combined_x = fused_expert_output
        # if not self.save_first:
        #     haha_dict = {
        #         "topk_weights": topk_weights,
        #         "topk_ids": topk_ids,
        #         "fused_expert_output": fused_expert_output
        #     }
        #     torch.save(haha_dict, "cutedsl_data/finalize_cutedsl_moe.pt")
        #     self.save_first = True
        num_local_experts = combined_x.shape[0]
        for expert_id in range(num_local_experts):
            num_valid_tokens = self.expert_num_tokens[expert_id].item()
            if num_valid_tokens < combined_x.shape[1]:
                combined_x[expert_id, num_valid_tokens:, :] = 0
        output_aggregated = torch.zeros(
            self.num_tokens, self.hidden_size, device=combined_x.device, dtype=combined_x.dtype
        )
        expert_positions = torch.zeros(self.expert_num, dtype=torch.long, device="cuda")
        
        for batch_idx in range(self.num_tokens):
            for k_pos in range(self.top_k):
                expert_id = topk_ids[batch_idx, k_pos].item()
                if expert_id < self.expert_num:
                    weight = topk_weights[batch_idx, k_pos].item()
                    expert_pos = expert_positions[expert_id].item()
                    if expert_pos < self.expert_num_tokens[expert_id]:
                        output_aggregated[batch_idx] += (
                            combined_x[expert_id, expert_pos, :] * weight
                        )
                        expert_positions[expert_id] += 1

        return output_aggregated
        if self.tp_size > 1:
            fused_expert_output = all_reduce(fused_expert_output, group=Group.TP)
        return fused_expert_output
