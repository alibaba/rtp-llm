import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

from rtp_llm.config.kv_cache_config import KVCacheConfig
from rtp_llm.config.py_config_modules import (
    MIN_WORKER_INFO_PORT_NUM,
    LoadConfig,
    PyEnvConfigs,
    ServerConfig,
)
from rtp_llm.ops import (
    ArpcConfig,
    CacheStoreConfig,
    ConcurrencyConfig,
    DashScGrpcConfig,
    DeviceResourceConfig,
    FfnDisAggregateConfig,
    FMHAConfig,
    GrammarConfig,
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
    # C++ uses this for NCCL ip/ports
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
    dash_sc_grpc_config: DashScGrpcConfig
    grammar_config: GrammarConfig
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
        lines.append(self.parallelism_config.to_string())

        lines.append("\n[RuntimeConfig]")
        lines.append(self.runtime_config.to_string())

        # Specialized configs
        lines.append("\n[PDSepConfig]")
        lines.append(self.pd_sep_config.to_string())

        lines.append("\n[ConcurrencyConfig]")
        lines.append(self.concurrency_config.to_string())

        lines.append("\n[FMHAConfig]")
        lines.append(self.fmha_config.to_string())

        lines.append("\n[KVCacheConfig]")
        lines.append(self.kv_cache_config.to_string())

        lines.append("\n[ProfilingDebugLoggingConfig]")
        lines.append(self.profiling_debug_logging_config.to_string())

        lines.append("\n[HWKernelConfig]")
        lines.append(self.hw_kernel_config.to_string())

        lines.append("\n[DeviceResourceConfig]")
        lines.append(self.device_resource_config.to_string())

        lines.append("\n[MoeConfig]")
        lines.append(self.moe_config.to_string())

        lines.append("\n[ModelSpecificConfig]")
        lines.append(self.model_specific_config.to_string())

        lines.append("\n[SpeculativeExecutionConfig]")
        lines.append(self.sp_config.to_string())

        lines.append("\n[CacheStoreConfig]")
        lines.append(self.cache_store_config.to_string())

        lines.append("\n[MiscellaneousConfig]")
        lines.append(self.misc_config.to_string())

        lines.append("\n[ArpcConfig]")
        lines.append(self.arpc_config.to_string())

        lines.append("\n[GrpcConfig]")
        lines.append(self.grpc_config.to_string())

        lines.append("\n[DashScGrpcConfig]")
        lines.append(self.dash_sc_grpc_config.to_string())

        lines.append("\n[GrammarConfig]")
        lines.append(self.grammar_config.to_string())
        lines.append("\n[LoadConfig]")
        lines.append(self.load_config.to_string())

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
        dash_sc_grpc_config = py_env_configs.dash_sc_grpc_config
        grammar_config = py_env_configs.grammar_config
        load_config = py_env_configs.load_config

        derive_grammar_compile_threads(grammar_config, parallelism_config)

        # Setup pd_sep_config role_type based on vit_separation
        if (
            py_env_configs.vit_config.vit_separation
            == VitSeparation.VIT_SEPARATION_ROLE
        ):
            pd_sep_config.role_type = RoleType.VIT
        else:
            # role_config.role_type property automatically converts string to RoleType enum
            pd_sep_config.role_type = py_env_configs.role_config.role_type

        # Mirror role into parallelism_config so model construction can read it
        # via parallelism_config.role_type instead of os.environ["ROLE_TYPE"].
        # Mirrors the vit_separation override above so VIT rank also surfaces
        # as RoleType::VIT to model code.
        parallelism_config.role_type = pd_sep_config.role_type

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
            dash_sc_grpc_config=dash_sc_grpc_config,
            grammar_config=grammar_config,
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
    runtime_config: RuntimeConfig,
    parallelism_config: ParallelismConfig,
    world_info,
    decode_entrance: bool = False,
) -> None:
    """Update worker addresses in runtime_config based on gang info.

    `worker_addrs` keeps the legacy cache-store address list in normal PD mode:
    `ip:cache_store_listen_port:cache_store_rdma_listen_port`.
    In decode_entrance mode, P2P needs an extra transfer port and rpc port, so we
    publish the P2P-specific `ip:p2p_transfer_port:grpc_port` format there.
    The P2P transfer port is reserved as `cache_store_listen_port + 1`
    (worker base + 3) and is validated before publishing.
    """
    if world_info is None:
        # For standalone mode, skip worker address updates
        logging.warning(
            "world_info is None, skipping worker address updates (standalone mode)"
        )
        return
    worker_addrs = []
    worker_grpc_addrs = []
    p2p_transfer_ports_by_ip = {}
    local_rank = parallelism_config.local_rank
    # P2P rank0 builds CP-local plans by worker_grpc_addrs index. Canonicalize
    # the address lists so index order is the world-rank order regardless of
    # gang discovery/refresh ordering.
    for member in sorted(world_info.members, key=lambda item: item.world_rank):
        p2p_transfer_port = (
            member.cache_store_listen_port + 1
            if member.cache_store_listen_port > 0
            else member.cache_store_listen_port
        )
        if decode_entrance:
            _validate_p2p_transfer_port(member, p2p_transfer_port)
            p2p_transfer_ports = p2p_transfer_ports_by_ip.setdefault(member.ip, set())
            if p2p_transfer_port in p2p_transfer_ports:
                raise ValueError(
                    f"duplicate p2p_transfer_port={p2p_transfer_port} on ip={member.ip}"
                )
            p2p_transfer_ports.add(p2p_transfer_port)
        p2p_worker_addr = f"{member.ip}:{p2p_transfer_port}:{member.rpc_server_port}"
        if (
            int(
                (member.world_rank / parallelism_config.tp_size)
                % parallelism_config.dp_size
            )
            == parallelism_config.dp_rank
        ):
            worker_addr = (
                p2p_worker_addr
                if decode_entrance
                else (
                    f"{member.ip}:{member.cache_store_listen_port}:"
                    f"{member.cache_store_rdma_listen_port}"
                )
            )
            worker_addrs.append(worker_addr)
            worker_grpc_addrs.append(f"{member.ip}:{member.rpc_server_port}")
            logging.info(
                f"append member for pd sep "
                f"{member.ip}:{member.rpc_server_port}, worker_addr={worker_addr}, "
                f"p2p_transfer_port={p2p_transfer_port}, cache_store_port={member.cache_store_listen_port}, "
                f"cache_store_rdma_port={member.cache_store_rdma_listen_port} "
                f"to local rank {local_rank}, world rank {member.world_rank}"
            )
    runtime_config.worker_grpc_addrs = worker_grpc_addrs
    runtime_config.worker_addrs = worker_addrs


def _valid_tcp_port(port: int) -> bool:
    return 1 <= port <= 65535


def _validate_p2p_transfer_port(member, p2p_transfer_port: int) -> None:
    if not _valid_tcp_port(p2p_transfer_port):
        raise ValueError(
            f"invalid p2p_transfer_port={p2p_transfer_port} for member {member}"
        )

    reserved_start = getattr(member, "server_port", None)
    worker_info_port_num = getattr(member, "_worker_info_port_num", 0)
    if reserved_start is not None and worker_info_port_num > 0:
        reserved_end = reserved_start + worker_info_port_num
        if not reserved_start <= p2p_transfer_port < reserved_end:
            raise ValueError(
                f"p2p_transfer_port={p2p_transfer_port} for member {member} "
                f"is outside worker reserved port block [{reserved_start}, {reserved_end})"
            )

    occupied_ports = {
        "server_port": getattr(member, "server_port", None),
        "rpc_server_port": getattr(member, "rpc_server_port", None),
        "cache_store_listen_port": getattr(member, "cache_store_listen_port", None),
        "cache_store_rdma_listen_port": getattr(
            member, "cache_store_rdma_listen_port", None
        ),
    }
    for port_name, port in occupied_ports.items():
        if port is not None and port > 0 and port == p2p_transfer_port:
            raise ValueError(
                f"p2p_transfer_port={p2p_transfer_port} for member {member} "
                f"conflicts with {port_name}"
            )


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
    pd_sep_config.worker_port_offset = server_config.worker_info_port_num

    # Always sync cache_store_rdma_mode between cache_store_config and
    # pd_sep_config, regardless of role_type. The two fields drive different
    # downstream consumers but must agree on whether RDMA is in use:
    #   * PDSepConfig.cache_store_rdma_mode (C++ default true)
    #     -> P2PConnectorWorker selects TransferBackend
    #        (kBarexRdma vs kTcp) for the decode_entrance push path;
    #     -> server_config_setup gates ACCL env hints; also referenced
    #        by RemoteRpcServer::initCacheStore and other PDSep RDMA-port wiring.
    #   * CacheStoreConfig.cache_store_rdma_mode (C++ default false)
    #     -> only used for non-P2P CacheStore paths (decode_entrance=0).
    # The C++ defaults disagree, so without an unconditional sync we can
    # end up with PDSepConfig=true while CacheStoreConfig=false or vice versa.
    # CACHE_STORE_RDMA_MODE env binds to cache_store_config.* via argparse,
    # so cache_store_config is the source of truth.
    pd_sep_before = pd_sep_config.cache_store_rdma_mode
    pd_sep_config.cache_store_rdma_mode = cache_store_config.cache_store_rdma_mode
    # Always log so any future divergence (or anyone reading the boot log)
    # can see whether the sync ran and what values it observed. The 5/25
    # incident showed CacheStoreConfig=0 + PDSepConfig=1 in the same
    # process, which is impossible if this log appears with both values
    # equal — so logging here pins down whether setup_pd_sep_config was
    # actually called on the live config instances.
    logging.info(
        "[PDSep-RdmaSync] role_type=%s, "
        "cache_store_config.cache_store_rdma_mode=%s (source of truth, "
        "from CACHE_STORE_RDMA_MODE env / --cache_store_rdma_mode cmdline), "
        "pd_sep_config.cache_store_rdma_mode: %s -> %s",
        pd_sep_config.role_type,
        cache_store_config.cache_store_rdma_mode,
        pd_sep_before,
        pd_sep_config.cache_store_rdma_mode,
    )


GRAMMAR_MIN_COMPILE_THREADS = 8
GRAMMAR_MAX_COMPILE_THREADS = 32


def derive_grammar_compile_threads(
    grammar_config: Any,  # GrammarConfig
    parallelism_config: Any,  # ParallelismConfig
) -> None:
    """Resolve automatic grammar compile fanout from this rank's CPU share."""
    if grammar_config.num_workers > 0:
        return

    cores = len(os.sched_getaffinity(0))
    ranks = max(
        1, min(parallelism_config.local_world_size, parallelism_config.world_size)
    )
    grammar_config.num_workers = min(
        max(
            GRAMMAR_MIN_COMPILE_THREADS,
            min(GRAMMAR_MAX_COMPILE_THREADS, cores // ranks),
        ),
        max(1, cores),
    )
    logging.info(
        f"grammar compile fanout derived: num_workers={grammar_config.num_workers} "
        f"(affinity_cores={cores}, ranks_on_node={ranks})"
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

    # Set max_batch_tokens_size if not set from py_runtime_config
    if fifo_scheduler_config.max_batch_tokens_size == 0:
        fifo_scheduler_config.max_batch_tokens_size = (
            fifo_scheduler_config.max_context_batch_size * max_seq_len
        )
    logging.info(
        f"max_batch_tokens_size: {fifo_scheduler_config.max_batch_tokens_size}"
    )
