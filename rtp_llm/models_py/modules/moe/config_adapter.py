"""
Adapter to provide a unified interface from individual config objects.
This allows Router and Executor classes to work with specific config objects.
"""

from typing import Optional
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.quant_config import QuantizationConfig
from rtp_llm.ops import ParallelismConfig, MoeConfig, RuntimeConfig


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
        runtime_config: Optional[RuntimeConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        self.model_config = model_config
        self.parallelism_config = parallelism_config
        self.moe_config = moe_config or MoeConfig()
        self.runtime_config = runtime_config or RuntimeConfig()
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
        self.moe_inter_padding_size = model_config.moe_inter_padding_size
        self.activation_type = model_config.activation_type
        self.hidden_size = model_config.hidden_size
        
        self.max_generate_batch_size = runtime_config.max_generate_batch_size if runtime_config else 0
        
        # For compatibility with distributed init, provide nccl_ip and th_nccl_port
        # These are typically in parallelism_config
        self.nccl_ip = parallelism_config.nccl_ip
        # Note: th_nccl_port may not be in ParallelismConfig, but tp_nccl_port is
        # For now, use tp_nccl_port as fallback, but this should be passed separately if needed
        self.th_nccl_port = parallelism_config.tp_nccl_port
        

