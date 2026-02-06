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
)
from rtp_llm.distribute.worker_info import CoordinatorInfo, WorkerInfo
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
        coordinator_info: Optional[CoordinatorInfo] = None,
        worker_info: Optional[WorkerInfo] = None,
    ) -> "EngineConfig":
        """Create and fully initialize EngineConfig from py_env_configs.

        This method creates the EngineConfig dataclass and performs necessary
        initialization including parallelism setup, runtime config setup, and
        PD separation config setup.

        Note: Worker address updates (via update_worker_addrs) should be called
        separately after this method, when world is available.

        Args:
            py_env_configs: PyEnvConfigs instance containing all configuration
            coordinator_info: Optional CoordinatorInfo from DistributedServer.get_coordinator_info().
                When provided, port/ip fields are taken from it.
            worker_info: WorkerInfo for model/pd_sep ports. When None, created from env
                (current process rank). Caller should pass adjusted worker_info in backend
                so each rank uses its own ports.

        Returns:
            Initialized EngineConfig instance
        """
        if worker_info is None:
            worker_info = WorkerInfo.from_env(
                py_env_configs.server_config.start_port,
                py_env_configs.distribute_config.remote_server_port,
                py_env_configs.server_config.worker_info_port_num,
            )

        # Create ParallelismConfig: fill from env, then set ports and ffn_disaggregate
        parallelism_config = ParallelismConfig()
        parallelism_config_from_env(
            parallelism_config,
            py_env_configs.server_config.worker_info_port_num,
        )
        setup_parallelism_config(
            parallelism_config,
            py_env_configs.ffn_disaggregate_config,
            coordinator_info=coordinator_info,
            worker_info=worker_info,
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
            load_config=load_config,
        )

        runtime_config.max_generate_batch_size = concurrency_config.concurrency_limit

        # Setup PD separation config
        setup_pd_sep_config(
            engine_config.pd_sep_config,
            cache_store_config,
            worker_info,
        )

        return engine_config


# ============================================================================
# EngineConfig setup and initialization functions
# ============================================================================


def _apply_parallelism_side_effects(local_rank: int) -> None:
    """Apply CUDA device and ACCL env side effects (same as ParallelInfo.from_params)."""
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if os.environ.get("ACCL_SELECT_PATH") == "1":
        select_port = str(local_rank % 2)
        os.environ["ACCL_SELECT_PORT"] = select_port
        logging.info(f"local rank {local_rank} set accl select port to {select_port} ")

    if (
        os.environ.get("ACCL_USE_NICS") is None
        and os.environ.get("ACCL_NIC_GPU_AFFINITY") is not None
    ):
        content = os.environ.get("ACCL_NIC_GPU_AFFINITY")
        try:
            gpu_nic_affinity = json.loads(content)
            if str(local_rank) in gpu_nic_affinity:
                affinity_nic = gpu_nic_affinity[str(local_rank)]
                os.environ["ACCL_USE_NICS"] = affinity_nic
                logging.info(
                    f"local rank {local_rank} use cuda device {local_rank} set ACCL_USE_NICS to {affinity_nic}"
                )
            else:
                logging.info(
                    f"local rank {local_rank} use cuda device {local_rank} get affinity nic failed, content is {content}"
                )
        except json.JSONDecodeError:
            logging.info(
                f"try decode ACCL_NIC_GPU_AFFINITY failed, content is {content}"
            )


def parallelism_config_from_params(
    parallelism_config: ParallelismConfig,
    params: Dict[str, str],
    worker_info_port_num: int,
) -> None:
    """Update ParallelismConfig from params (e.g. os.environ), mirroring ParallelInfo.from_params.

    Parses WORLD_SIZE, LOCAL_WORLD_SIZE, TP_SIZE, EP_SIZE, PP_SIZE, DP_SIZE,
    FFN_SP_SIZE, WORLD_RANK/WORLD_INDEX and writes sizes and derived ranks to
    parallelism_config. worker_info_port_num is only used for validation, not stored.

    Also applies the same side effects as ParallelInfo.from_params: torch.cuda.set_device,
    ACCL_SELECT_PORT, ACCL_USE_NICS when applicable.

    Args:
        parallelism_config: ParallelismConfig to update in place
        params: Dict of env-like key/values (e.g. dict(os.environ))
        worker_info_port_num: Used for validation (>= MIN_WORKER_INFO_PORT_NUM), not stored
    """
    if worker_info_port_num < MIN_WORKER_INFO_PORT_NUM:
        raise Exception(
            f"worker info port num {worker_info_port_num} "
            f"is smaller than min worker info port num {MIN_WORKER_INFO_PORT_NUM}"
        )

    world_size = int(params.get("WORLD_SIZE", "1"))
    if "LOCAL_WORLD_SIZE" in params:
        local_world_size = int(params["LOCAL_WORLD_SIZE"])
    else:
        local_world_size = (
            min(torch.cuda.device_count(), world_size)
            if torch.cuda.is_available()
            else world_size
        )
    local_world_size = max(local_world_size, 1)

    tp_size = int(params.get("TP_SIZE", "1"))
    ep_size = int(params.get("EP_SIZE", params.get("WORLD_SIZE", "1")))
    pp_size = int(params.get("PP_SIZE", "1"))
    dp_size = int(params.get("DP_SIZE", 1))
    ffn_sp_size = int(params.get("FFN_SP_SIZE", "1"))
    world_rank = int(params.get("WORLD_RANK", "0"))
    if ("WORLD_INDEX" in params) and ("WORLD_RANK" not in params):
        world_index = int(params["WORLD_INDEX"])
        world_rank = world_index * local_world_size

    # Validate (same as ParallelInfo.from_params)
    if ep_size > world_size or world_size % ep_size != 0:
        raise Exception(
            f"ep_size:{ep_size} <= world_size:{world_size} and "
            f"world_size:{world_size} % ep_size:{ep_size} != 0"
        )
    if world_size != tp_size * dp_size * pp_size:
        raise Exception(
            f"world_size:{world_size} != tp_size:{tp_size} * dp_size:{dp_size} * pp_size:{pp_size}"
        )
    if torch.cuda.is_available() and local_world_size > torch.cuda.device_count():
        raise Exception(
            f"local_world_size:{local_world_size} > cuda device count:{torch.cuda.device_count()}"
        )
    if tp_size * pp_size * dp_size != world_size or world_rank >= world_size:
        raise Exception(
            f"tp_size:{tp_size}, ep_size:{ep_size}, pp_size:{pp_size}, world_size:{world_size}, "
            f"world_rank:{world_rank} ffn_sp_size:{ffn_sp_size} invalid world config"
        )
    if tp_size % ffn_sp_size != 0:
        raise Exception(
            f"tp_size:{tp_size} % ffn_sp_size:{ffn_sp_size} != 0 invalid world config"
        )
    if world_size % local_world_size != 0:
        raise Exception(
            f"not support world_size:[{world_size}] mod local_world_size:[{local_world_size}] != 0"
        )

    # Derived values (same formulas as ParallelInfo)
    ffn_tp_size = tp_size // ffn_sp_size
    local_rank = world_rank % local_world_size
    tp_rank = world_rank % tp_size
    dp_rank = world_rank // tp_size
    ep_rank = world_rank % ep_size
    ffn_tp_rank = tp_rank % ffn_tp_size

    # Write to parallelism_config
    parallelism_config.tp_size = tp_size
    parallelism_config.ep_size = ep_size
    parallelism_config.pp_size = pp_size
    parallelism_config.dp_size = dp_size
    parallelism_config.ffn_sp_size = ffn_sp_size
    parallelism_config.ffn_tp_size = ffn_tp_size
    parallelism_config.world_size = world_size
    parallelism_config.world_rank = world_rank
    parallelism_config.local_world_size = local_world_size
    parallelism_config.local_rank = local_rank
    parallelism_config.tp_rank = tp_rank
    parallelism_config.dp_rank = dp_rank
    parallelism_config.ep_rank = ep_rank
    parallelism_config.ffn_tp_rank = ffn_tp_rank
    parallelism_config.enable_sp = ffn_sp_size > 1

    _apply_parallelism_side_effects(local_rank)

    logging.info(
        f"parallelism_config_from_params: tp_size={tp_size} ep_size={ep_size} pp_size={pp_size} "
        f"world_size={world_size} world_rank={world_rank} local_world_size={local_world_size} "
        f"ffn_sp_size={ffn_sp_size} ffn_tp_size={ffn_tp_size}"
    )


def parallelism_config_from_env(
    parallelism_config: ParallelismConfig,
    worker_info_port_num: int,
) -> None:
    """Update ParallelismConfig from os.environ, mirroring ParallelInfo.from_env.

    Args:
        parallelism_config: ParallelismConfig to update in place
        worker_info_port_num: Passed to parallelism_config_from_params for validation, not stored
    """
    parallelism_config_from_params(
        parallelism_config, dict(os.environ), worker_info_port_num
    )


def is_master_rank(parallelism_config: ParallelismConfig) -> bool:
    """Return True iff this rank is master (world_rank == 0)."""
    return parallelism_config.world_rank == 0


def setup_parallelism_config(
    parallelism_config: ParallelismConfig,
    py_ffn_disaggregate_config: Optional[FfnDisAggregateConfig] = None,
    coordinator_info: Optional[CoordinatorInfo] = None,
    worker_info: Optional[WorkerInfo] = None,
) -> None:
    """Set port/worker and FfnDisAggregate fields on an already-filled ParallelismConfig.

    Expects parallelism_config to already have sizes and ranks set (e.g. via
    parallelism_config_from_env or parallelism_config_from_params). Sets nccl/ports
    from coordinator_info and worker_info, and ffn_disaggregate from py_ffn_disaggregate_config.

    Args:
        parallelism_config: ParallelismConfig instance (already filled with sizes/ranks)
        py_ffn_disaggregate_config: Optional FfnDisAggregateConfig from py_env_configs
        coordinator_info: CoordinatorInfo (e.g. from DistributedServer.get_coordinator_info()).
        worker_info: WorkerInfo for model/embedding/http ports; when None, port fields are not set.
    """
    if coordinator_info is not None:
        parallelism_config.nccl_ip = coordinator_info.ip
        parallelism_config.tp_nccl_port = coordinator_info.tp_nccl_port
        parallelism_config.dp_tp_nccl_port = coordinator_info.dp_tp_nccl_port
        parallelism_config.ffn_tp_nccl_port = coordinator_info.ffn_tp_nccl_port
        parallelism_config.th_nccl_port = coordinator_info.th_nccl_port

    if worker_info is not None:
        parallelism_config.model_rpc_port = worker_info.rpc_server_port
        parallelism_config.embedding_rpc_server_port = (
            worker_info.embedding_rpc_server_port
        )
        parallelism_config.http_port = worker_info.http_port

    if (
        py_ffn_disaggregate_config
        and py_ffn_disaggregate_config.enable_ffn_disaggregate
    ):
        assert (
            parallelism_config.tp_size == 1 and parallelism_config.world_size > 1
        ), "enable_ffn_disaggregate must be used in dp = 1 world_size > 1"
        attention_dp_size = parallelism_config.world_size - 1
        attention_tp_size = 1
        ffn_tp_size = 1
        assert (
            attention_tp_size == ffn_tp_size
        ), "attention_tp_size must be equal to ffn_tp_size"
        parallelism_config.ffn_disaggregate_config.enable_ffn_disaggregate = True
        parallelism_config.ffn_disaggregate_config.attention_tp_size = attention_tp_size
        parallelism_config.ffn_disaggregate_config.attention_dp_size = attention_dp_size
        parallelism_config.ffn_disaggregate_config.ffn_tp_size = ffn_tp_size
        parallelism_config.ffn_disaggregate_config.ffn_dp_size = 1
        parallelism_config.ffn_disaggregate_config.is_ffn_rank = (
            parallelism_config.world_rank >= attention_tp_size * attention_dp_size
        )

    logging.info(f"th_nccl_port: {parallelism_config.th_nccl_port}")


def adjust_parallelism_config_for_world_rank(
    parallelism_config: ParallelismConfig,
    world_rank: int,
) -> None:
    """Update rank-related fields in ParallelismConfig from a given world_rank.

    Uses the same derivation as ParallelInfo: local_rank, tp_rank, dp_rank, ep_rank,
    ffn_tp_rank are computed from world_rank and the existing size fields.

    Args:
        parallelism_config: ParallelismConfig to update in place
        world_rank: World rank to apply (local_rank = world_rank % local_world_size, etc.)
    """
    parallelism_config.world_rank = world_rank
    parallelism_config.local_rank = world_rank % parallelism_config.local_world_size
    parallelism_config.tp_rank = world_rank % parallelism_config.tp_size
    parallelism_config.dp_rank = world_rank // parallelism_config.tp_size
    parallelism_config.ep_rank = world_rank % parallelism_config.ep_size
    parallelism_config.ffn_tp_rank = (
        parallelism_config.tp_rank % parallelism_config.ffn_tp_size
    )


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
    worker_info: WorkerInfo,
) -> None:
    """Setup PDSepConfig from worker info and cache_store_config."""
    # Update pd_sep_config fields
    pd_sep_config.cache_store_listen_port = worker_info.cache_store_listen_port
    pd_sep_config.cache_store_connect_port = worker_info.cache_store_connect_port
    pd_sep_config.cache_store_rdma_listen_port = (
        worker_info.cache_store_rdma_listen_port
    )
    pd_sep_config.cache_store_rdma_connect_port = (
        worker_info.cache_store_rdma_connect_port
    )
    pd_sep_config.remote_rpc_server_port = worker_info.remote_rpc_server_port
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