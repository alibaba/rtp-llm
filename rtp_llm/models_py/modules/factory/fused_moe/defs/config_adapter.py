"""
Adapter to provide a unified interface from individual config objects.
This allows Router and Executor classes to work with specific config objects.
"""

from typing import Optional

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.quant_config import QuantizationConfig
from rtp_llm.ops import MoeConfig, ParallelismConfig


class MoEConfigAdapter:
    """
    Adapter class that provides a unified interface
    from individual configuration objects.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        parallelism_config: ParallelismConfig,
        moe_config: Optional[MoeConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        enable_cuda_graph: bool = False,
    ):
        self.model_config = model_config
        self.parallelism_config = parallelism_config
        self.moe_config = moe_config or MoeConfig()
        self.quant_config = quant_config

        # Provide shortcut access to commonly used attributes
        self.ep_size = parallelism_config.ep_size
        self.ep_rank = parallelism_config.ep_rank
        # tp_size/tp_rank reflect the attention/MoE-input view: when CP is
        # enabled, get_attn_tp_size() returns 1, so MoE input slicing
        # (deepep narrow/allgather) stays a no-op. Router selectors that
        # need the physical TP topology (e.g. pure_cp_router) read raw
        # parallelism_config.tp_size via is_cp_equal_ep().
        self.tp_size = parallelism_config.get_attn_tp_size()
        self.tp_rank = parallelism_config.get_attn_tp_rank()
        self.dp_size = parallelism_config.dp_size
        self.dp_rank = parallelism_config.dp_rank
        self.world_size = parallelism_config.world_size
        self.world_rank = parallelism_config.world_rank
        # Calculate local_rank from world_rank and local_world_size
        self.local_rank = parallelism_config.local_rank

        self.expert_num = model_config.expert_num
        self.moe_k = model_config.moe_k
        self.moe_topk_group = model_config.moe_topk_group
        self.hidden_size = model_config.hidden_size
        self.dim = self.hidden_size
        self.layer_id = int(getattr(model_config, "layer_id", -1))
        self.moe_inter_dim = int(
            getattr(model_config, "moe_inter_size", model_config.inter_size)
        )
        self.n_routed_experts = self.expert_num
        self.n_activated_experts = self.moe_k
        self.n_shared_experts = int(getattr(model_config, "n_shared_experts", 0))
        if self.n_shared_experts == 0 and getattr(model_config, "moe_style", 0) == 2:
            routed_inter_size = int(getattr(model_config, "moe_inter_size", 0))
            shared_inter_size = int(getattr(model_config, "inter_size", 0))
            if routed_inter_size > 0 and shared_inter_size % routed_inter_size == 0:
                self.n_shared_experts = shared_inter_size // routed_inter_size
        self.has_shared_expert_gate = False
        self.swiglu_limit = float(getattr(model_config, "swiglu_limit", 0.0))
        self.n_local_experts = self.expert_num // max(self.ep_size, 1)
        self.local_expert_start = self.ep_rank * self.n_local_experts
        self.local_expert_end = self.local_expert_start + self.n_local_experts
        self.max_tokens_per_rank = int(
            getattr(moe_config, "ll_num_max_token", 0) or model_config.max_seq_len or 1
        )
        self.moe_quant_method = getattr(model_config, "moe_quant_method", None)
        self.data_type = model_config.data_type
        self.head_num = model_config.attn_config.head_num
        self.ll_num_max_token = moe_config.ll_num_max_token
        self.masked_max_token_num = moe_config.masked_max_token_num
        self.moe_strategy = moe_config.moe_strategy
        self.use_mori_ep = moe_config.use_mori_ep
        self.use_deepep_moe = moe_config.use_deepep_moe
        self.enable_cuda_graph = enable_cuda_graph

    @property
    def activation_type(self):
        """Access activation_type from model_config when needed."""
        return self.model_config.activation_type
