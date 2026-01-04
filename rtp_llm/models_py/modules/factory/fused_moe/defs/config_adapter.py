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
        max_generate_batch_size: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        enable_cuda_graph: bool = False,
        ll_num_max_token_per_rank: int = 0,
    ):
        self.model_config = model_config
        self.parallelism_config = parallelism_config
        self.moe_config = moe_config or MoeConfig()
        self.quant_config = quant_config

        # Provide shortcut access to commonly used attributes
        self.ep_size = parallelism_config.ep_size
        self.ep_rank = parallelism_config.ep_rank
        self.tp_size = parallelism_config.tp_size
        self.tp_rank = parallelism_config.tp_rank
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

        self.max_generate_batch_size = max_generate_batch_size
        self.enable_cuda_graph = enable_cuda_graph
        self.ll_num_max_token_per_rank = ll_num_max_token_per_rank

    @property
    def activation_type(self):
        """Access activation_type from model_config when needed."""
        return self.model_config.activation_type
