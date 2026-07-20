"""
Adapter to provide a unified interface from individual config objects.
This allows Router and Executor classes to work with specific config objects.
"""

from typing import Optional

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.ops import MoeConfig, ParallelismConfig

_UNSET_QUANT_CONFIG = object()


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
        quant_config=_UNSET_QUANT_CONFIG,
        enable_cuda_graph: bool = False,
    ):
        if not isinstance(enable_cuda_graph, bool):
            raise TypeError("enable_cuda_graph must be a bool")
        self.model_config = model_config
        self.parallelism_config = parallelism_config
        self.moe_config = moe_config if moe_config is not None else MoeConfig()
        self.quant_config = (
            getattr(model_config, "quant_config", None)
            if quant_config is _UNSET_QUANT_CONFIG
            else quant_config
        )

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
        # Calculate local_rank from world_rank and local_world_size
        self.local_rank = parallelism_config.local_rank

        self.expert_num = model_config.expert_num
        self.moe_k = model_config.moe_k
        self.moe_topk_group = model_config.moe_topk_group
        self.hidden_size = model_config.hidden_size
        self.data_type = model_config.data_type
        self.head_num = model_config.attn_config.head_num
        self.ll_num_max_token = self.moe_config.ll_num_max_token
        self.masked_max_token_num = self.moe_config.masked_max_token_num
        self.moe_strategy = self.moe_config.moe_strategy
        self.use_mori_ep = self.moe_config.use_mori_ep
        self.use_deepep_moe = self.moe_config.use_deepep_moe
        self.enable_cuda_graph = enable_cuda_graph

    @property
    def activation_type(self):
        """Access activation_type from model_config when needed."""
        return self.model_config.activation_type
