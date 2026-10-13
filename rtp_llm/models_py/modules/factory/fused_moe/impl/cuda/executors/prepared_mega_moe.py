"""Factory executor for prepared FP4/FP8 MegaMoE weights and graph workspaces."""

import copy

from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    CombineForwardPayload,
    FusedMoeExpertExecutor,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.type import ExecutorType


class PreparedMegaMoeExecutor(FusedMoeExpertExecutor):
    execute_empty_inputs = True

    @classmethod
    def executor_type(cls):
        return ExecutorType.FP8_FP4

    @classmethod
    def check_conditions(cls, checker, config):
        checker.check(getattr(config, "mega_moe_policy", None) is not None)
        checker.check(not config.has_redundant_experts)
        checker.check(config.ep_size == config.world_size)
        checker.check(config.ep_rank == config.world_rank)

    def __init__(self, config, quant_config, weights):
        super().__init__(config, quant_config, weights)
        from .mega_moe_runtime_factory import create_mega_moe_wrapper

        self.runtime = create_mega_moe_wrapper(
            config.model_config,
            config.parallelism_config,
            weights,
            config.moe_config,
            layer_idx=config.layer_id,
            max_generate_batch_size=config.max_generate_batch_size,
            execution_policy=config.mega_moe_policy,
        )
        self.includes_shared_expert = self.runtime.includes_shared_expert

    def execute(
        self,
        payload,
        activation,
        expert_map,
        a2_scale,
        apply_router_weight_on_input,
        extra_expert_args,
    ):
        output = self.runtime(
            payload.expert_x,
            payload.expert_topk_weights,
            payload.expert_topk_ids,
            activation=activation,
            expert_map=expert_map,
            a1_scale=payload.expert_x_scale,
            a2_scale=a2_scale,
            apply_router_weight_on_input=apply_router_weight_on_input,
            extra_expert_args=extra_expert_args,
        )
        return CombineForwardPayload(fused_expert_output=output)

    def clone_for_cuda_graph(self, *, share_mega_buf=False):
        clone = copy.copy(self)
        clone.runtime = self.runtime.clone_for_cuda_graph(share_mega_buf=share_mega_buf)
        return clone

    def prepacked_input_views(self, tokens):
        return self.runtime.prepacked_input_views(tokens)

    def forward_prepacked(self, hidden_states):
        return self.runtime.forward_prepacked(hidden_states)
