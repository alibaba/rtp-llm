"""
Adapter to provide a unified interface from individual config objects.
This allows Router and Executor classes to work with specific config objects.
"""

from typing import Optional
from rtp_llm.config.model_config import ModelConfig as PyModelConfig
from rtp_llm.config.quant_config import QuantizationConfig
from rtp_llm.ops import ParallelismConfig, MoeConfig, RuntimeConfig


class MoEConfigAdapter:
    """
    Adapter class that provides a unified interface
    from individual configuration objects.
    """
    
    def __init__(
        self,
        py_model_config: PyModelConfig,
        parallelism_config: ParallelismConfig,
        moe_config: Optional[MoeConfig] = None,
        runtime_config: Optional[RuntimeConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        self.py_model_config = py_model_config
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
        self.local_rank = getattr(parallelism_config, 'local_rank', 
                                 parallelism_config.world_rank % parallelism_config.local_world_size if parallelism_config.local_world_size > 0 else parallelism_config.world_rank)
        
        self.expert_num = py_model_config.expert_num
        self.moe_k = py_model_config.moe_k
        self.moe_topk_group = getattr(py_model_config, 'moe_topk_group', py_model_config.moe_k)
        self.moe_inter_padding_size = py_model_config.moe_inter_padding_size
        self.activation_type = py_model_config.activation_type
        self.hidden_size = py_model_config.hidden_size
        
        self.max_generate_batch_size = runtime_config.max_generate_batch_size if runtime_config else 0
        
        # For compatibility with distributed init, provide nccl_ip and th_nccl_port
        # These are typically in parallelism_config
        self.nccl_ip = parallelism_config.nccl_ip if hasattr(parallelism_config, 'nccl_ip') and parallelism_config.nccl_ip else ""
        # Note: th_nccl_port may not be in ParallelismConfig, but tp_nccl_port is
        # For now, use tp_nccl_port as fallback, but this should be passed separately if needed
        self.th_nccl_port = getattr(parallelism_config, 'tp_nccl_port', 0)
        
        # For compatibility with deepep_wrapper, provide access to nested configs
        # Create a minimal structure to access ffn_disaggregate_config
        self._ffn_disaggregate_config = None

