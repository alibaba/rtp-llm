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

        # Legacy default: the decode running-batch cap follows concurrency_limit.
        # An explicit --max_generate_batch_size overrides it as a scheduler-side
        # cap: excess streams queue in the scheduler instead of being rejected
        # at the HTTP frontend.
        runtime_config.max_generate_batch_size = concurrency_config.concurrency_limit
        if py_env_configs.max_generate_batch_size_override is not None:
            if py_env_configs.max_generate_batch_size_override <= 0:
                raise ValueError(
                    "--max_generate_batch_size must be positive, got "
                    f"{py_env_configs.max_generate_batch_size_override}"
                )
            runtime_config.max_generate_batch_size = (
                py_env_configs.max_generate_batch_size_override
            )

        apply_deterministic_inference_config(
            py_env_configs.deterministic_config,
            runtime_config,
            hw_kernel_config,
            parallelism_config,
            explicit_max_generate_batch_size=py_env_configs.max_generate_batch_size_override,
        )

        # Setup PD separation config
        setup_pd_sep_config(
            engine_config.pd_sep_config,
            cache_store_config,
            server_config,
            distribute_config,
        )

        return engine_config


def apply_deterministic_inference_config(
    deterministic_config,
    runtime_config,
    hw_kernel_config,
    parallelism_config,
    explicit_max_generate_batch_size=None,
) -> None:
    """Apply the deterministic-inference switch to engine configs.

    Default (enable=False) is a strict no-op: this function returns before
    touching any config so the default serving path is unchanged.

    level="decode" (mechanism A): force single-size decode CUDA graph so every
    decode step runs at one fixed geometry (batch padded to B_det with dummy
    rows).

    level="batched" (mechanisms A + C + F): production determinism with batched
    decode. Every request still prefills alone (one stream per prefill forward,
    so cuBLASLt heuristics never see a batch-total M that varies with traffic),
    but decode keeps batching up to B_det streams (extras queue in the
    scheduler). The ratio scheduler schedules the exclusive prefill rounds
    promptly instead of starving behind the running decode batch. Reported
    top-1/top-5 logprobs can carry ~1e-6 batch-composition noise while tokens
    stay identical to solo.

    level="full" (mechanisms A + C + F): additionally force single-request
    serial serving -- every request prefills alone and decodes as the only real
    row of the padded batch, which is exactly the solo composition, so each
    request reproduces the solo output bitwise regardless of concurrent traffic.

    explicit_max_generate_batch_size carries the tri-state --max_generate_batch_size
    override (None = not provided). The batched/full presets pin the value
    (decode graph size / 1) because their determinism contract requires it;
    when they override a user-provided value a warning is logged.
    """
    if not deterministic_config.enable:
        return

    level = deterministic_config.level
    if level not in ("decode", "batched", "full"):
        raise ValueError(
            f"invalid deterministic_level: {level!r} (expected 'decode', 'batched' or 'full')"
        )
    b_det = int(deterministic_config.decode_batch_size)
    if b_det <= 0:
        raise ValueError(
            f"deterministic_decode_batch_size must be positive, got {b_det}"
        )

    # Mechanism A: fixed decode geometry via a single-size decode CUDA graph.
    if not hw_kernel_config.enable_cuda_graph:
        logging.info(
            "deterministic_inference: force enable_cuda_graph=True for fixed decode geometry"
        )
    hw_kernel_config.enable_cuda_graph = True
    hw_kernel_config.decode_capture_batch_sizes = [b_det]

    if level == "batched":
        # Mechanism F: one request per prefill forward (M = its own length)
        # while decode keeps batching.
        runtime_config.fifo_scheduler_config.force_single_prefill = True
        # Cap the running decode batch at the single captured graph size;
        # extra streams queue in the scheduler (never rejected).
        if (
            explicit_max_generate_batch_size is not None
            and explicit_max_generate_batch_size != b_det
        ):
            logging.warning(
                "deterministic_level=batched pins max_generate_batch_size=%d to "
                "match the single decode graph size; overriding explicit "
                "--max_generate_batch_size=%d",
                b_det,
                explicit_max_generate_batch_size,
            )
        runtime_config.max_generate_batch_size = b_det
        # Ratio scheduler: PREFILL and DECODE run as separate rounds on a fixed
        # cadence, so exclusive prefill forwards are scheduled promptly instead
        # of starving behind the running decode batch (decode-first FIFO).
        # Cadence 0 = prefill-first while streams are waiting; tune with
        # --decode_prefill_ratio.
        runtime_config.fifo_scheduler_config.pdfusion_scheduler_mode = "ratio"
        runtime_config.fifo_scheduler_config.decode_prefill_ratio = "0"

    if level == "full":
        # Mechanism F: single-request serial serving. The scheduler admits
        # at most one running stream, so each request prefills alone (M equals
        # its own length) and decodes as the only real row of the B_det batch.
        if (
            explicit_max_generate_batch_size is not None
            and explicit_max_generate_batch_size != 1
        ):
            logging.warning(
                "deterministic_level=full is single-request serial serving; "
                "overriding explicit --max_generate_batch_size=%d with 1",
                explicit_max_generate_batch_size,
            )
        runtime_config.max_generate_batch_size = 1

    # Mechanism C: pin the NCCL algorithm for TP>1 so all-reduce reduction
    # order does not depend on the NCCL communicator's algorithm selection.
    tp_size = getattr(parallelism_config, "tp_size", 1) or 1
    if tp_size > 1:
        os.environ.setdefault("NCCL_ALGO", "Ring")

    logging.info(
        "deterministic_inference enabled: level=%s, decode graph batch size=%d, "
        "max_generate_batch_size=%d, force_single_prefill=%s, pdfusion=%s/%s%s",
        level,
        b_det,
        runtime_config.max_generate_batch_size,
        runtime_config.fifo_scheduler_config.force_single_prefill,
        runtime_config.fifo_scheduler_config.pdfusion_scheduler_mode,
        runtime_config.fifo_scheduler_config.decode_prefill_ratio,
        ", NCCL_ALGO=Ring" if tp_size > 1 else "",
    )


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
    pd_sep_config.worker_port_offset = server_config.worker_info_port_num

    # Override with values from other sources
    if pd_sep_config.role_type in [RoleType.PREFILL, RoleType.DECODE]:
        pd_sep_config.cache_store_rdma_mode = cache_store_config.cache_store_rdma_mode


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
