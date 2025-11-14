"""ConfigWrapper class for providing unified config access to C++ operations.

This wrapper class provides a unified interface for C++ operations (RtpLLMOp, EmbeddingOp)
to access configuration objects from BaseModel without directly accessing model_config
and engine_config separately.
"""

from typing import Optional
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.engine_config import EngineConfig
from rtp_llm.config.py_config_modules import VitConfig
from rtp_llm.ops import (
    ParallelismConfig,
    RuntimeConfig,
    PDSepConfig,
    ConcurrencyConfig,
    FMHAConfig,
    KVCacheConfig,
    ProfilingDebugLoggingConfig,
    HWKernelConfig,
    DeviceResourceConfig,
    MoeConfig,
    ModelSpecificConfig,
    SpeculativeExecutionConfig,
    CacheStoreConfig,
    MiscellaneousConfig,
    ArpcConfig,
    FfnDisAggregateConfig,
)


class ConfigWrapper:
    """Wrapper class that provides unified access to model and engine configurations.
    
    This class aggregates configuration objects from BaseModel's model_config and
    engine_config, providing a single interface for C++ operations to access all
    necessary configuration.
    
    Attributes:
        model_config: Model configuration (ModelConfig)
        parallelism_config: Parallelism configuration (ParallelismConfig)
        runtime_config: Runtime configuration (RuntimeConfig)
        pd_sep_config: PD separation configuration (PDSepConfig)
        concurrency_config: Concurrency configuration (ConcurrencyConfig)
        fmha_config: FMHA configuration (FMHAConfig)
        kv_cache_config: KV cache configuration (KVCacheConfig)
        profiling_debug_logging_config: Profiling and debug logging configuration (ProfilingDebugLoggingConfig)
        hw_kernel_config: Hardware kernel configuration (HWKernelConfig)
        device_resource_config: Device resource configuration (DeviceResourceConfig)
        moe_config: MoE configuration (MoeConfig)
        model_specific_config: Model-specific configuration (ModelSpecificConfig)
        sp_config: Speculative execution configuration (SpeculativeExecutionConfig)
        cache_store_config: Cache store configuration (CacheStoreConfig)
        misc_config: Miscellaneous configuration (MiscellaneousConfig)
        arpc_config: ARPC configuration (ArpcConfig)
        vit_config: Optional VitConfig for multimodal models
        
    Note:
        ffn_disaggregate_config is available via parallelism_config.ffn_disaggregate_config
    """
    
    def __init__(
        self,
        model_config: ModelConfig,
        engine_config: EngineConfig,
        vit_config: Optional[VitConfig] = None,
    ):
        """Initialize ConfigWrapper with model and engine configurations.
        
        Args:
            model_config: Model configuration
            engine_config: Engine configuration
            vit_config: Optional VitConfig for multimodal models
        """
        self.model_config = model_config
        self.parallelism_config = engine_config.parallelism_config
        self.runtime_config = engine_config.runtime_config
        self.pd_sep_config = engine_config.pd_sep_config
        self.concurrency_config = engine_config.concurrency_config
        self.fmha_config = engine_config.fmha_config
        self.kv_cache_config = engine_config.kv_cache_config
        self.profiling_debug_logging_config = engine_config.profiling_debug_logging_config
        self.hw_kernel_config = engine_config.hw_kernel_config
        self.device_resource_config = engine_config.device_resource_config
        self.moe_config = engine_config.moe_config
        self.model_specific_config = engine_config.model_specific_config
        self.sp_config = engine_config.sp_config
        self.cache_store_config = engine_config.cache_store_config
        self.misc_config = engine_config.misc_config
        self.arpc_config = engine_config.arpc_config
        # ffn_disaggregate_config is available via parallelism_config.ffn_disaggregate_config
        self.vit_config = vit_config

