from dataclasses import dataclass
from typing import Any, Dict, Optional
import logging

from rtp_llm.config.kv_cache_config import KVCacheConfig
from rtp_llm.config.py_config_modules import PyEnvConfigs, WORKER_INFO_PORT_NUM
from rtp_llm.distribute.worker_info import (
    g_master_info,
    g_parallel_info,
    g_worker_info,
    ParallelInfo,
)
from rtp_llm.ops import (
    ArpcConfig,
    CacheStoreConfig,
    ConcurrencyConfig,
    DeviceResourceConfig,
    FMHAConfig,
    FfnDisAggregateConfig,
    GrpcConfig,
    HWKernelConfig,
    MiscellaneousConfig,
    ModelSpecificConfig,
    MoeConfig,
    ParallelismConfig,
    PDSepConfig,
    ProfilingDebugLoggingConfig,
    RoleType,
    RuntimeConfig,
    SpeculativeExecutionConfig,
    VitSeparation,
)

@dataclass
class EngineConfig:
    """Engine configuration collection created from py_env_configs.
    
    It contains only configuration objects that are related to py_env_configs,
    not model-specific configs like PyModelConfig or MMModelConfig.
    """
    # Parallelism and runtime configs
    parallelism_config: ParallelismConfig
    runtime_config: RuntimeConfig
    
    # Specialized configs from py_env_configs
    pd_sep_config: PDSepConfig
    concurrency_config: ConcurrencyConfig
    fmha_config: FMHAConfig
    kv_cache_config: KVCacheConfig
    profiling_debug_logging_config: ProfilingDebugLoggingConfig
    hw_kernel_config: HWKernelConfig
    device_resource_config: DeviceResourceConfig
    moe_config: MoeConfig
    model_specific_config: ModelSpecificConfig
    sp_config: SpeculativeExecutionConfig
    cache_store_config: CacheStoreConfig
    misc_config: MiscellaneousConfig
    arpc_config: ArpcConfig
    grpc_config: GrpcConfig
    
    def to_string(self) -> str:
        """Return a formatted string representation of EngineConfig for debugging.
        
        Returns:
            A multi-line string containing all configuration information.
        """
        lines = []
        lines.append("=" * 80)
        lines.append("EngineConfig")
        lines.append("=" * 80)
        
        # Parallelism and runtime configs
        lines.append("\n[ParallelismConfig]")
        if hasattr(self.parallelism_config, 'to_string'):
            lines.append(self.parallelism_config.to_string())
        else:
            lines.append(str(self.parallelism_config))
        
        lines.append("\n[RuntimeConfig]")
        if hasattr(self.runtime_config, 'to_string'):
            lines.append(self.runtime_config.to_string())
        else:
            lines.append(str(self.runtime_config))
        
        # Specialized configs
        lines.append("\n[PDSepConfig]")
        if hasattr(self.pd_sep_config, 'to_string'):
            lines.append(self.pd_sep_config.to_string())
        else:
            lines.append(str(self.pd_sep_config))
        
        lines.append("\n[ConcurrencyConfig]")
        if hasattr(self.concurrency_config, 'to_string'):
            lines.append(self.concurrency_config.to_string())
        else:
            lines.append(str(self.concurrency_config))
        
        lines.append("\n[FMHAConfig]")
        if hasattr(self.fmha_config, 'to_string'):
            lines.append(self.fmha_config.to_string())
        else:
            lines.append(str(self.fmha_config))
        
        lines.append("\n[KVCacheConfig]")
        if hasattr(self.kv_cache_config, 'to_string'):
            lines.append(self.kv_cache_config.to_string())
        else:
            lines.append(str(self.kv_cache_config))
        
        lines.append("\n[ProfilingDebugLoggingConfig]")
        if hasattr(self.profiling_debug_logging_config, 'to_string'):
            lines.append(self.profiling_debug_logging_config.to_string())
        else:
            lines.append(str(self.profiling_debug_logging_config))
        
        lines.append("\n[HWKernelConfig]")
        if hasattr(self.hw_kernel_config, 'to_string'):
            lines.append(self.hw_kernel_config.to_string())
        else:
            lines.append(str(self.hw_kernel_config))
        
        lines.append("\n[DeviceResourceConfig]")
        if hasattr(self.device_resource_config, 'to_string'):
            lines.append(self.device_resource_config.to_string())
        else:
            lines.append(str(self.device_resource_config))
        
        lines.append("\n[MoeConfig]")
        if hasattr(self.moe_config, 'to_string'):
            lines.append(self.moe_config.to_string())
        else:
            lines.append(str(self.moe_config))
        
        lines.append("\n[ModelSpecificConfig]")
        if hasattr(self.model_specific_config, 'to_string'):
            lines.append(self.model_specific_config.to_string())
        else:
            lines.append(str(self.model_specific_config))
        
        lines.append("\n[SpeculativeExecutionConfig]")
        if hasattr(self.sp_config, 'to_string'):
            lines.append(self.sp_config.to_string())
        else:
            lines.append(str(self.sp_config))
        
        lines.append("\n[CacheStoreConfig]")
        if hasattr(self.cache_store_config, 'to_string'):
            lines.append(self.cache_store_config.to_string())
        else:
            lines.append(str(self.cache_store_config))
        
        lines.append("\n[MiscellaneousConfig]")
        if hasattr(self.misc_config, 'to_string'):
            lines.append(self.misc_config.to_string())
        else:
            lines.append(str(self.misc_config))
        
        lines.append("\n[ArpcConfig]")
        if hasattr(self.arpc_config, 'to_string'):
            lines.append(self.arpc_config.to_string())
        else:
            lines.append(str(self.arpc_config))
        
        lines.append("\n" + "=" * 80)
        return "\n".join(lines)
    
    @staticmethod
    def create(py_env_configs: PyEnvConfigs) -> 'EngineConfig':
        """Create and fully initialize EngineConfig from py_env_configs.
        
        This method creates the EngineConfig dataclass and performs necessary
        initialization including parallelism setup, runtime config setup, and 
        PD separation config setup.
        
        Note: Worker address updates (via update_worker_addrs) should be called 
        separately after this method, when gang_info is available.
        
        Args:
            py_env_configs: PyEnvConfigs instance containing all configuration
        
        Returns:
            Initialized EngineConfig instance
        """
        
        # Create ParallelismConfig and setup from parallel_info
        parallelism_config = ParallelismConfig()
        setup_parallelism_config(
            parallelism_config,
            g_parallel_info,
            py_env_configs.ffn_disaggregate_config,
        )
        
        runtime_config = py_env_configs.runtime_config
        
        # Directly use C++ binding objects from py_env_configs
        pd_sep_config = py_env_configs.pd_separation_config
        concurrency_config = py_env_configs.concurrency_config
        fmha_config = py_env_configs.fmha_config
        kv_cache_config = py_env_configs.kv_cache_config
        profiling_debug_logging_config = py_env_configs.profiling_debug_logging_config
        hw_kernel_config = py_env_configs.py_hw_kernel_config
        device_resource_config = py_env_configs.device_resource_config
        model_specific_config = py_env_configs.model_specific_config
        misc_config = py_env_configs.misc_config.misc_config
        moe_config = py_env_configs.moe_config
        sp_config = py_env_configs.sp_config
        cache_store_config = py_env_configs.cache_store_config
        arpc_config = py_env_configs.arpc_config
        grpc_config = py_env_configs.grpc_config
        
        # Setup pd_sep_config role_type based on vit_separation
        if py_env_configs.vit_config.vit_separation == VitSeparation.VIT_SEPARATION_ROLE:
            pd_sep_config.role_type = RoleType.VIT
        else:
            # role_config.role_type property automatically converts string to RoleType enum
            pd_sep_config.role_type = py_env_configs.role_config.role_type
        
        # Create EngineConfig instance
        engine_config = EngineConfig(
            parallelism_config=parallelism_config,
            runtime_config=runtime_config,
            pd_sep_config=pd_sep_config,
            concurrency_config=concurrency_config,
            fmha_config=fmha_config,
            kv_cache_config=kv_cache_config,
            profiling_debug_logging_config=profiling_debug_logging_config,
            hw_kernel_config=hw_kernel_config,
            device_resource_config=device_resource_config,
            moe_config=moe_config,
            model_specific_config=model_specific_config,
            sp_config=sp_config,
            cache_store_config=cache_store_config,
            misc_config=misc_config,
            arpc_config=arpc_config,
            grpc_config=grpc_config,
        )
        
        runtime_config.max_generate_batch_size = concurrency_config.concurrency_limit
        
        # Setup PD separation config
        setup_pd_sep_config(
            engine_config.pd_sep_config,
            cache_store_config,
        )
 
        return engine_config

# ============================================================================
# EngineConfig setup and initialization functions
# ============================================================================

def setup_parallelism_config(
    parallelism_config: ParallelismConfig,
    parallel_info: ParallelInfo = g_parallel_info,
    py_ffn_disaggregate_config: Optional[FfnDisAggregateConfig] = None,
) -> None:
    """Setup ParallelismConfig from parallel_info and master/worker info.
    
    Also sets up FfnDisAggregateConfig if it's a member of ParallelismConfig.
    
    Args:
        parallelism_config: ParallelismConfig instance to setup
        parallel_info: ParallelInfo for parallelism setup
        py_ffn_disaggregate_config: Optional FfnDisAggregateConfig from py_env_configs
    """
    parallelism_config.tp_size = parallel_info.tp_size
    parallelism_config.tp_rank = parallel_info.tp_rank
    parallelism_config.ep_size = parallel_info.ep_size
    parallelism_config.ep_rank = parallel_info.ep_rank
    parallelism_config.dp_size = parallel_info.dp_size
    parallelism_config.dp_rank = parallel_info.dp_rank
    parallelism_config.ffn_tp_rank = parallel_info.ffn_tp_rank
    parallelism_config.ffn_tp_size = parallel_info.ffn_tp_size
    parallelism_config.enable_sp = parallel_info.ffn_sp_size > 1
    # Note: local_rank is a computed property in ParallelInfo, not a field in ParallelismConfig
    parallelism_config.world_size = parallel_info.world_size
    parallelism_config.world_rank = parallel_info.world_rank
    parallelism_config.local_world_size = parallel_info.local_world_size
    parallelism_config.local_rank = parallel_info.local_rank
    parallelism_config.pp_size = parallel_info.pp_size
    parallelism_config.ffn_sp_size = parallel_info.ffn_sp_size
    
    # Set port and IP related fields
    parallelism_config.nccl_ip = g_master_info.ip
    parallelism_config.tp_nccl_port = g_master_info.tp_nccl_port
    parallelism_config.dp_tp_nccl_port = g_master_info.dp_tp_nccl_port
    parallelism_config.ffn_tp_nccl_port = g_master_info.ffn_tp_nccl_port
    parallelism_config.model_rpc_port = g_worker_info.rpc_server_port
    parallelism_config.embedding_rpc_server_port = g_worker_info.embedding_rpc_server_port
    parallelism_config.http_port = g_worker_info.http_port
    parallelism_config.th_nccl_port = g_master_info.th_nccl_port
    
    # Setup FfnDisAggregateConfig if it's a member of ParallelismConfig
    # Note: This assumes ParallelismConfig has ffn_disaggregate_config as a member
    # If not, the C++ code needs to be updated first
    if py_ffn_disaggregate_config and py_ffn_disaggregate_config.enable_ffn_disaggregate:
        # 暂时先限制tp=1, 更多支持在python版本实现
        assert (
            parallel_info.tp_size == 1 and parallel_info.world_size > 1
        ), "enable_ffn_disaggregate must be used in dp = 1 world_size > 1"
        attention_dp_size = parallel_info.world_size - 1
        attention_tp_size = 1
        ffn_tp_size = 1
        assert (
            attention_tp_size == ffn_tp_size
        ), "attention_tp_size must be equal to ffn_tp_size"
        parallelism_config.ffn_disaggregate_config.enable_ffn_disaggregate = True
        parallelism_config.ffn_disaggregate_config.attention_tp_size = attention_tp_size
        parallelism_config.ffn_disaggregate_config.attention_dp_size = attention_dp_size
        parallelism_config.ffn_disaggregate_config.ffn_tp_size = ffn_tp_size
        # TODO: remove it, ffn dp is stupid
        parallelism_config.ffn_disaggregate_config.ffn_dp_size = 1
        parallelism_config.ffn_disaggregate_config.is_ffn_rank = (
            parallel_info.world_rank >= attention_tp_size * attention_dp_size
        )
    
    logging.info(f"th_nccl_port: {parallelism_config.th_nccl_port}")
    

def update_worker_addrs(
    runtime_config: RuntimeConfig,
    parallelism_config: ParallelismConfig,
    gang_info) -> None:
    """Update worker addresses in runtime_config based on gang info."""
    if gang_info is None:
        # For standalone mode, skip worker address updates
        logging.warning("gang_info is None, skipping worker address updates (standalone mode)")
        return
    worker_addrs = []
    worker_grpc_addrs = []
    local_rank = parallelism_config.local_rank
    for member in gang_info.members:
        if int((member.world_rank / parallelism_config.tp_size) % parallelism_config.dp_size) == parallelism_config.dp_rank:
            worker_addrs.append(
                f"{member.ip}:{member.cache_store_listen_port}:{member.cache_store_rdma_listen_port}"
            )
            worker_grpc_addrs.append(f"{member.ip}:{member.rpc_server_port}")
            logging.info(
                f"append member for pd sep "
                f"{member.ip}:{member.rpc_server_port}, {member.cache_store_listen_port}, "
                f"{member.cache_store_rdma_listen_port} to local rank {local_rank}, world rank {member.world_rank}"
            )
    runtime_config.worker_grpc_addrs = worker_grpc_addrs
    runtime_config.worker_addrs = worker_addrs


def setup_pd_sep_config(
    pd_sep_config: PDSepConfig,
    cache_store_config,
) -> None:
    """Setup PDSepConfig from worker info and cache_store_config."""
    # Update pd_sep_config fields
    pd_sep_config.cache_store_listen_port = g_worker_info.cache_store_listen_port
    pd_sep_config.cache_store_connect_port = g_worker_info.cache_store_connect_port
    pd_sep_config.cache_store_rdma_listen_port = g_worker_info.cache_store_rdma_listen_port
    pd_sep_config.cache_store_rdma_connect_port = g_worker_info.cache_store_rdma_connect_port
    pd_sep_config.remote_rpc_server_port = g_worker_info.remote_rpc_server_port
    pd_sep_config.worker_port_offset = WORKER_INFO_PORT_NUM
    
    # Override with values from other sources
    if pd_sep_config.role_type in [RoleType.PREFILL, RoleType.DECODE]:
        pd_sep_config.cache_store_rdma_mode = (
            cache_store_config.cache_store_rdma_mode
        )


def finalize_scheduler_config(
    fifo_scheduler_config: Any,  # FIFOSchedulerConfig
    max_seq_len: int,
) -> None:
    """Finalize fifo_scheduler_config with computed values.
    
    Args:
        fifo_scheduler_config: FIFOSchedulerConfig instance to finalize
        max_seq_len: Maximum sequence length from model config
    """
    # fast_gen_max_context_len uses fast_gen_context_budget from fifo_scheduler_config
    if fifo_scheduler_config.fast_gen_context_budget == -1:
        fifo_scheduler_config.fast_gen_max_context_len = 1024
    else:
        fifo_scheduler_config.fast_gen_max_context_len = fifo_scheduler_config.fast_gen_context_budget
    logging.info(f"fast_gen_max_context_len: {fifo_scheduler_config.fast_gen_max_context_len}")

    # Set max_batch_tokens_size if not set from py_runtime_config
    if fifo_scheduler_config.max_batch_tokens_size == 0:
        fifo_scheduler_config.max_batch_tokens_size = (
            fifo_scheduler_config.max_context_batch_size * max_seq_len
        )
    logging.info(f"max_batch_tokens_size: {fifo_scheduler_config.max_batch_tokens_size}")

