import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

from rtp_llm.config.kv_cache_config import KVCacheConfig
from rtp_llm.config.py_config_modules import (
    MIN_WORKER_INFO_PORT_NUM,
    WORKER_INFO_PORT_NUM,
    LoadConfig,
    PyEnvConfigs,
    ServerConfig,
)
from rtp_llm.ops import (
    ArpcConfig,
    CacheStoreConfig,
    ConcurrencyConfig,
    DeviceResourceConfig,
    FfnDisAggregateConfig,
    FMHAConfig,
    GrpcConfig,
    HWKernelConfig,
    MiscellaneousConfig,
    ModelSpecificConfig,
    MoeConfig,
    NcclCommConfig,
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
    # C++ initDevices uses this for NCCL ip/ports
    nccl_comm_config: NcclCommConfig
    # C++ reads rpc_server_port, embedding_rpc_server_port, http_port from this
    server_config: ServerConfig

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
    load_config: LoadConfig

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
        if hasattr(self.parallelism_config, "to_string"):
            lines.append(self.parallelism_config.to_string())
        else:
            lines.append(str(self.parallelism_config))

        lines.append("\n[RuntimeConfig]")
        if hasattr(self.runtime_config, "to_string"):
            lines.append(self.runtime_config.to_string())
        else:
            lines.append(str(self.runtime_config))

        # Specialized configs
        lines.append("\n[PDSepConfig]")
        if hasattr(self.pd_sep_config, "to_string"):
            lines.append(self.pd_sep_config.to_string())
        else:
            lines.append(str(self.pd_sep_config))

        lines.append("\n[ConcurrencyConfig]")
        if hasattr(self.concurrency_config, "to_string"):
            lines.append(self.concurrency_config.to_string())
        else:
            lines.append(str(self.concurrency_config))

        lines.append("\n[FMHAConfig]")
        if hasattr(self.fmha_config, "to_string"):
            lines.append(self.fmha_config.to_string())
        else:
            lines.append(str(self.fmha_config))

        lines.append("\n[KVCacheConfig]")
        if hasattr(self.kv_cache_config, "to_string"):
            lines.append(self.kv_cache_config.to_string())
        else:
            lines.append(str(self.kv_cache_config))

        lines.append("\n[ProfilingDebugLoggingConfig]")
        if hasattr(self.profiling_debug_logging_config, "to_string"):
            lines.append(self.profiling_debug_logging_config.to_string())
        else:
            lines.append(str(self.profiling_debug_logging_config))

        lines.append("\n[HWKernelConfig]")
        if hasattr(self.hw_kernel_config, "to_string"):
            lines.append(self.hw_kernel_config.to_string())
        else:
            lines.append(str(self.hw_kernel_config))

        lines.append("\n[DeviceResourceConfig]")
        if hasattr(self.device_resource_config, "to_string"):
            lines.append(self.device_resource_config.to_string())
        else:
            lines.append(str(self.device_resource_config))

        lines.append("\n[MoeConfig]")
        if hasattr(self.moe_config, "to_string"):
            lines.append(self.moe_config.to_string())
        else:
            lines.append(str(self.moe_config))

        lines.append("\n[ModelSpecificConfig]")
        if hasattr(self.model_specific_config, "to_string"):
            lines.append(self.model_specific_config.to_string())
        else:
            lines.append(str(self.model_specific_config))

        lines.append("\n[SpeculativeExecutionConfig]")
        if hasattr(self.sp_config, "to_string"):
            lines.append(self.sp_config.to_string())
        else:
            lines.append(str(self.sp_config))

        lines.append("\n[CacheStoreConfig]")
        if hasattr(self.cache_store_config, "to_string"):
            lines.append(self.cache_store_config.to_string())
        else:
            lines.append(str(self.cache_store_config))

        lines.append("\n[MiscellaneousConfig]")
        if hasattr(self.misc_config, "to_string"):
            lines.append(self.misc_config.to_string())
        else:
            lines.append(str(self.misc_config))

        lines.append("\n[ArpcConfig]")
        if hasattr(self.arpc_config, "to_string"):
            lines.append(self.arpc_config.to_string())
        else:
            lines.append(str(self.arpc_config))

        lines.append("\n[LoadConfig]")
        if hasattr(self.load_config, "to_string"):
            lines.append(self.load_config.to_string())
        else:
            lines.append(str(self.load_config))

        lines.append("\n" + "=" * 80)
        return "\n".join(lines)

    @staticmethod
    def create(
        py_env_configs: PyEnvConfigs,
        nccl_comm_config: Optional[NcclCommConfig] = None,
    ) -> "EngineConfig":
        """Create and fully initialize EngineConfig from py_env_configs.

        This method creates the EngineConfig dataclass and performs necessary
        initialization including parallelism setup, runtime config setup, and
        PD separation config setup. Ports are read from server_config and
        distribute_config (already adjusted for current rank in backend).

        Note: Worker address updates (via update_worker_addrs) should be called
        separately after this method, when world is available.

        Args:
            py_env_configs: PyEnvConfigs instance containing all configuration
            nccl_comm_config: Optional NcclCommConfig from DistributedServer.get_nccl_comm_config().
                When provided, NCCL ip/ports are taken from it.

        Returns:
            Initialized EngineConfig instance
        """
        server_config = py_env_configs.server_config
        distribute_config = py_env_configs.distribute_config

        parallelism_config = py_env_configs.parallelism_config
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
        load_config = py_env_configs.load_config

        # Setup pd_sep_config role_type based on vit_separation
        if (
            py_env_configs.vit_config.vit_separation
            == VitSeparation.VIT_SEPARATION_ROLE
        ):
            pd_sep_config.role_type = RoleType.VIT
        else:
            # role_config.role_type property automatically converts string to RoleType enum
            pd_sep_config.role_type = py_env_configs.role_config.role_type

        if nccl_comm_config is None:
            nccl_comm_config = NcclCommConfig(
                nccl_ip="",
                tp_nccl_port=0,
                dp_tp_nccl_port=0,
                ffn_tp_nccl_port=0,
            )

        # Create EngineConfig instance
        engine_config = EngineConfig(
            parallelism_config=parallelism_config,
            runtime_config=runtime_config,
            nccl_comm_config=nccl_comm_config,
            server_config=server_config,
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
            load_config=load_config,
        )

        runtime_config.max_generate_batch_size = concurrency_config.concurrency_limit

        # Setup PD separation config
        setup_pd_sep_config(
            engine_config.pd_sep_config,
            cache_store_config,
            server_config,
            distribute_config,
        )

        return engine_config


# ============================================================================
# EngineConfig setup and initialization functions
# ============================================================================


def update_worker_addrs(
    runtime_config: RuntimeConfig, parallelism_config: ParallelismConfig, world_info
) -> None:
    """Update worker addresses in runtime_config based on gang info."""
    if world_info is None:
        # For standalone mode, skip worker address updates
        logging.warning(
            "world_info is None, skipping worker address updates (standalone mode)"
        )
        return
    worker_addrs = []
    worker_grpc_addrs = []
    local_rank = parallelism_config.local_rank
    for member in world_info.members:
        if (
            int(
                (member.world_rank / parallelism_config.tp_size)
                % parallelism_config.dp_size
            )
            == parallelism_config.dp_rank
        ):
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
    server_config,
    distribute_config,
) -> None:
    """Setup PDSepConfig from server/distribute config and cache_store_config."""
    # Update pd_sep_config fields from config
    pd_sep_config.cache_store_listen_port = server_config.cache_store_listen_port
    pd_sep_config.cache_store_connect_port = distribute_config.cache_store_connect_port
    pd_sep_config.cache_store_rdma_listen_port = (
        server_config.cache_store_rdma_listen_port
    )
    pd_sep_config.cache_store_rdma_connect_port = (
        distribute_config.cache_store_rdma_connect_port
    )
    pd_sep_config.remote_rpc_server_port = distribute_config.remote_rpc_server_port
    pd_sep_config.worker_port_offset = WORKER_INFO_PORT_NUM

    # Override with values from other sources
    if pd_sep_config.role_type in [RoleType.PREFILL, RoleType.DECODE]:
        pd_sep_config.cache_store_rdma_mode = cache_store_config.cache_store_rdma_mode


def finalize_scheduler_config(
    fifo_scheduler_config: Any,  # FIFOSchedulerConfig
    max_seq_len: int,
) -> None:
    """Finalize fifo_scheduler_config with computed values.

    Args:
        fifo_scheduler_config: FIFOSchedulerConfig instance to finalize
        max_seq_len: Maximum sequence length from model config
    """

    # Set max_batch_tokens_size if not set from py_runtime_config
    if fifo_scheduler_config.max_batch_tokens_size == 0:
        fifo_scheduler_config.max_batch_tokens_size = (
            fifo_scheduler_config.max_context_batch_size * max_seq_len
        )
    logging.info(
        f"max_batch_tokens_size: {fifo_scheduler_config.max_batch_tokens_size}"
    )
