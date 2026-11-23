import json
import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional

import torch

from rtp_llm.config.collective_config import CollectiveConfig
from rtp_llm.config.kv_cache_config import KVCacheConfig
from rtp_llm.config.py_config_modules import (
    MIN_WORKER_INFO_PORT_NUM,
    LoadConfig,
    PyEnvConfigs,
    ServerConfig,
    resolve_max_generate_batch_size,
)
from rtp_llm.device.device_type import is_cuda
from rtp_llm.ops import (
    ArpcConfig,
    CacheStoreConfig,
    ConcurrencyConfig,
    DashScGrpcConfig,
    DataType,
    FMHAConfig,
    GrammarConfig,
    GrpcConfig,
    HWKernelConfig,
    KvCacheDataType,
    MiscellaneousConfig,
    MoeConfig,
    NcclCommConfig,
    ParallelismConfig,
    PDSepConfig,
    ProfilingDebugLoggingConfig,
    RoleType,
    RuntimeConfig,
    SpeculativeExecutionConfig,
    SpeculativeType,
    TaskType,
    VitSeparation,
)
from rtp_llm.utils.util import get_config_from_path

if TYPE_CHECKING:
    from rtp_llm.config.model_config import ModelConfig


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
    # C++ reads rpc_server_port, embedding_rpc_server_port from this
    server_config: ServerConfig

    # Specialized configs from py_env_configs
    pd_sep_config: PDSepConfig
    concurrency_config: ConcurrencyConfig
    fmha_config: FMHAConfig
    kv_cache_config: KVCacheConfig
    profiling_debug_logging_config: ProfilingDebugLoggingConfig
    hw_kernel_config: HWKernelConfig
    moe_config: MoeConfig
    sp_config: SpeculativeExecutionConfig
    cache_store_config: CacheStoreConfig
    misc_config: MiscellaneousConfig
    arpc_config: ArpcConfig
    grpc_config: GrpcConfig
    dash_sc_grpc_config: DashScGrpcConfig
    grammar_config: GrammarConfig
    load_config: LoadConfig
    collective_config: CollectiveConfig = field(default_factory=CollectiveConfig)

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

        lines.append("\n[CollectiveConfig]")
        lines.append(self.collective_config.to_string())

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

        lines.append("\n[MoeConfig]")
        lines.append(self.moe_config.to_string())

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
        misc_config = py_env_configs.misc_config.misc_config
        moe_config = py_env_configs.moe_config
        sp_config = py_env_configs.sp_config
        cache_store_config = py_env_configs.cache_store_config
        arpc_config = py_env_configs.arpc_config
        grpc_config = py_env_configs.grpc_config
        dash_sc_grpc_config = py_env_configs.dash_sc_grpc_config
        grammar_config = py_env_configs.grammar_config
        load_config = py_env_configs.load_config

        collective_config = CollectiveConfig(
            all_reduce_backend=py_env_configs.all_reduce_backend,
            large_prefill=py_env_configs.all_reduce_large_prefill,
            batch_invariant=hw_kernel_config.batch_invariant,
        )
        # Also validate single-rank servers, which do not initialize Torch groups.
        collective_config.validate_topology(parallelism_config)

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
            moe_config=moe_config,
            sp_config=sp_config,
            cache_store_config=cache_store_config,
            misc_config=misc_config,
            arpc_config=arpc_config,
            grpc_config=grpc_config,
            dash_sc_grpc_config=dash_sc_grpc_config,
            grammar_config=grammar_config,
            load_config=load_config,
            collective_config=collective_config,
        )

        # Legacy default: the decode running-batch cap follows concurrency_limit.
        # An explicit --max_generate_batch_size overrides it as a scheduler-side
        # cap: excess streams queue in the scheduler instead of being rejected
        # at the HTTP frontend. The resolution helper is shared with
        # setup_and_configure_server so the DeepEP ll_num_max_token derived
        # there stays identical to this value: DeepepWrapperConfig.equal
        # compares both derivations and would fail at startup on any drift.
        runtime_config.max_generate_batch_size = resolve_max_generate_batch_size(
            py_env_configs
        )

        # Setup PD separation config
        setup_pd_sep_config(
            engine_config.pd_sep_config,
            cache_store_config,
            server_config,
            distribute_config,
        )

        return engine_config


_BATCH_INVARIANT_QWEN_MODELS = {
    # Model type, layers, hidden size, Q heads, KV heads, head size, FFN shape.
    ("qwen_3", 36, 4096, 32, 8, 128, (12288,)): ("Qwen3-8B", 1),
    ("qwen_3_moe", 48, 2048, 32, 4, 128, (128, 8, 768)): ("Qwen3-30B-A3B", 1),
    ("qwen35_moe", 40, 2048, 16, 2, 256, (256, 8, 512)): ("Qwen3.6-35B-A3B", 1),
    ("qwen35_moe", 60, 4096, 32, 2, 256, (512, 10, 1024)): ("Qwen3.5-397B-A17B", 4),
}


def configure_batch_invariant_model(
    engine_config: EngineConfig, model_config: "ModelConfig"
) -> None:
    """Validate the supported CUDA eager configuration before loading weights."""
    hw = engine_config.hw_kernel_config
    if not hw.batch_invariant:
        return

    parallel = engine_config.parallelism_config
    cache = engine_config.kv_cache_config
    moe = engine_config.moe_config
    attn = model_config.attn_config
    model_type = model_config.model_type
    hybrid = model_type == "qwen35_moe"
    uses_moe = hybrid or model_type == "qwen_3_moe"
    dp_ep = parallel.tp_size == 1 and parallel.dp_size == parallel.ep_size > 1
    conflicts = []
    if not is_cuda() or torch.cuda.get_device_capability() != (10, 3):
        conflicts.append("requires an SM103 CUDA GPU")
    if parallel.role_type != RoleType.PDFUSION or (
        model_config.vit_separation != VitSeparation.VIT_SEPARATION_LOCAL
    ):
        conflicts.append("requires a colocated PDFUSION worker with no separate ViT")
    ffn_shape = (
        (model_config.expert_num, model_config.moe_k, model_config.moe_inter_size)
        if uses_moe
        else (model_config.inter_size,)
    )
    geometry = (
        model_type,
        model_config.num_layers,
        model_config.hidden_size,
        attn.head_num,
        attn.kv_head_num,
        attn.size_per_head,
        ffn_shape,
    )
    model = _BATCH_INVARIANT_QWEN_MODELS.get(geometry)
    if model is None:
        supported = ", ".join(name for name, _ in _BATCH_INVARIANT_QWEN_MODELS.values())
        conflicts.append(f"requires one of these architectures: {supported}")
    else:
        name, tp = model
        topology = (parallel.world_size, parallel.dp_size, parallel.ep_size)
        supported_topologies = [(tp, 1, 1)]
        if uses_moe:
            supported_topologies.append((4, 4 // tp, 4))
        if (
            topology not in supported_topologies
            or parallel.local_world_size != parallel.world_size
            or (parallel.tp_size, parallel.ffn_tp_size, parallel.pp_size) != (tp, tp, 1)
            or parallel.prefill_cp_config.is_enabled()
            or parallel.prefill_cp_config.is_prefill_enabled()
        ):
            ep4 = f" or DP_SIZE={4 // tp}/EP_SIZE=4 on one node" if uses_moe else ""
            conflicts.append(
                f"{name} requires TP_SIZE={tp}, PP_SIZE=1, CP disabled and "
                f"DP_SIZE=EP_SIZE=1{ep4}"
            )
    quant = model_config.quant_config
    quant_method = quant.get_method() if quant is not None else None
    fp8_block = (
        quant_method == "FP8_PER_BLOCK"
        and quant.bits == 8
        and quant.group_size() == 128
    )
    if model_config.data_type != DataType.TYPE_BF16:
        conflicts.append("requires BF16 activations (ACT_TYPE=bf16)")
    if quant is not None and (not fp8_block or not hybrid or model is None):
        conflicts.append(
            f"unsupported quantization {quant_method}: FP8_PER_BLOCK with 128x128 "
            "weight blocks requires Qwen3.6-35B-A3B TP1 or Qwen3.5-397B-A17B TP4"
        )
    if model is not None:
        checkpoint_config = get_config_from_path(model_config.ckpt_path) or {}
        text_config = (
            (checkpoint_config.get("text_config") or {})
            if hybrid
            else checkpoint_config
        )
        checkpoint_dtype = text_config.get("dtype") or text_config.get("torch_dtype")
        if checkpoint_dtype not in ("bfloat16", "bf16"):
            config_name = "text_config" if hybrid else "top-level config"
            conflicts.append(
                f"requires a BF16 checkpoint ({config_name} has dtype {checkpoint_dtype!r})"
            )
    if model_config.task_type != TaskType.LANGUAGE_MODEL:
        conflicts.append("requires a text generation task")
    if model_config.lora_infos:
        conflicts.append("LoRA adapters are unsupported")
    if hw.enable_decode_cascade_attention:
        conflicts.append("ENABLE_DECODE_CASCADE_ATTENTION must be disabled")
    if engine_config.sp_config.type != SpeculativeType.NONE:
        conflicts.append("speculative decoding and MTP must be disabled")
    if engine_config.runtime_config.use_batch_decode_scheduler:
        conflicts.append("USE_BATCH_DECODE_SCHEDULER must be disabled")
    if dp_ep:
        moe_strategies = (
            ("auto", "fp8_per_block_pure_dp", "fp8_per_block_ep_normal")
            if fp8_block
            else ("auto", "no_auant_dp_normal", "no_quant_pure_dp")
        )
    else:
        moe_strategies = (
            ("auto", "fp8_per_block_no_dp") if fp8_block else ("auto", "no_quant_cpp")
        )
    needs_all_gather = fp8_block or parallel.ep_size > 1
    uses_unstable_router = not dp_ep and (
        moe.use_deepep_moe or (needs_all_gather and not moe.use_all_gather)
    )
    if uses_moe and (
        moe.moe_strategy not in moe_strategies
        or uses_unstable_router
        or moe.use_mori_ep
        or moe.fake_balance_expert
        or model_config.eplb_config.phy_exp_num(model_config.expert_num)
        != model_config.expert_num
    ):
        if dp_ep:
            conflicts.append(
                "requires the eager all-gather MoE path for DP/EP without expert remapping"
            )
        elif fp8_block:
            conflicts.append(
                "FP8_PER_BLOCK requires the local DeepGEMM MoE strategy "
                "(MOE_STRATEGY=auto or fp8_per_block_no_dp, USE_ALL_GATHER enabled) "
                "and no expert remapping"
            )
        else:
            conflicts.append(
                "requires the local Triton MoE strategy and no expert remapping"
            )
    if attn.kv_cache_dtype != KvCacheDataType.BASE:
        conflicts.append("requires BF16 KV cache with FP8_KV_CACHE disabled")
    if (
        hybrid
        and model_config.linear_attention_config.ssm_state_dtype != DataType.TYPE_FP32
    ):
        conflicts.append("requires FP32 SSM state (SSM_STATE_DTYPE=auto or fp32)")
    if attn.tokens_per_block != 64 or attn.kernel_tokens_per_block != 64:
        pages = "KV and SSM" if hybrid else "KV"
        conflicts.append(f"requires 64-token {pages} pages")
    if (
        cache.reuse_cache
        or cache.enable_remote_cache
        or cache.enable_disk_cache
        or cache.enable_memory_cache
        or cache.multi_task_prompt
        or cache.multi_task_prompt_str
        or cache.multi_task_prompt_tokens
    ):
        conflicts.append(
            "prefix reuse, task prompts and external KV caches must be disabled"
        )
    if conflicts:
        raise ValueError("RTP_BATCH_INVARIANT=1:\n" + "\n".join(conflicts))

    if dp_ep:
        # Ordinary DP/EP auto-selection uses DeepEP. Move the same shards over
        # the stage group in BI, where the combine order is fixed.
        moe.moe_strategy = "fp8_per_block_pure_dp" if fp8_block else "no_quant_pure_dp"
        moe.use_all_gather = True
        moe.use_deepep_moe = False
        moe.use_deepep_low_latency = False
        moe.use_deepep_internode = False
    if hybrid:
        # Keep the text weights and partial MRoPE parameters of the hybrid checkpoint.
        model_config.mm_model_config.is_multimodal = False
    # Keep the checkpoint's BF16 LM head and accumulate its products in FP32.
    model_config.enable_fp32_lm_head = False


# ============================================================================
# EngineConfig setup and initialization functions
# ============================================================================


def update_worker_addrs(
    runtime_config: RuntimeConfig,
    parallelism_config: ParallelismConfig,
    world_info,
    role_type: RoleType = RoleType.PDFUSION,
) -> None:
    """Update worker addresses in runtime_config based on gang info.

    PD workers publish `ip:p2p_transfer_port:grpc_port`. Other roles keep the
    CacheStore address format. The P2P transfer port uses the reserved slot
    `cache_store_listen_port + 1` and is validated before publishing.
    """
    p2p_enabled = role_type in (RoleType.PREFILL, RoleType.DECODE)
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
        if p2p_enabled:
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
                if p2p_enabled
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

    if pd_sep_config.role_type in (RoleType.PREFILL, RoleType.DECODE):
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
    role_type: RoleType,
    use_batch_decode_scheduler: bool = False,
    seq_size_per_block: int = 0,
) -> None:
    """Finalize fifo_scheduler_config with computed values.

    Args:
        fifo_scheduler_config: FIFOSchedulerConfig instance to finalize
        max_seq_len: Maximum sequence length from model config
        role_type: Engine role; chunked prefill only applies to PREFILL and PDFUSION.
        use_batch_decode_scheduler: Whether BatchDecodeScheduler is enabled, which is incompatible with chunked prefill.
        seq_size_per_block: KV block size used to align the chunk budget; 0 skips normalization.
    """

    # Set max_batch_tokens_size if not set from py_runtime_config
    if fifo_scheduler_config.max_batch_tokens_size == 0:
        fifo_scheduler_config.max_batch_tokens_size = (
            fifo_scheduler_config.max_context_batch_size * max_seq_len
        )
    logging.info(
        f"max_batch_tokens_size: {fifo_scheduler_config.max_batch_tokens_size}"
    )

    # Chunked prefill normalization + engine-level hard gates.
    chunk_size = fifo_scheduler_config.prefill_chunk_size
    if chunk_size <= 0:
        return  # chunked prefill disabled; nothing to validate.

    if role_type not in (RoleType.PREFILL, RoleType.PDFUSION):
        logging.info(
            f"prefill_chunk_size only applies to PREFILL / PDFUSION roles; "
            f"disabling for role_type={role_type}"
        )
        fifo_scheduler_config.prefill_chunk_size = 0
        return

    if use_batch_decode_scheduler:
        raise ValueError(
            "prefill_chunk_size > 0 is not supported with use_batch_decode_scheduler=True; "
            "BatchDecodeScheduler forces streams onto the decode path. Disable chunked prefill "
            "or use the normal FIFO/PDFusion scheduler."
        )

    max_stream_chunk_size = 2**31 - 1
    if chunk_size > max_stream_chunk_size:
        raise ValueError(
            f"prefill_chunk_size ({chunk_size}) must not exceed "
            f"INT_MAX ({max_stream_chunk_size})"
        )

    # Require at least one KV block, then floor-align so every chunk starts on a block boundary
    # without exceeding the activation-memory budget. Log a warning on adjust.
    if seq_size_per_block > 0:
        if chunk_size < seq_size_per_block:
            raise ValueError(
                f"prefill_chunk_size ({chunk_size}) must be at least seq_size_per_block "
                f"({seq_size_per_block})"
            )

        aligned = (chunk_size // seq_size_per_block) * seq_size_per_block
        if aligned != chunk_size:
            logging.warning(
                f"prefill_chunk_size ({chunk_size}) is not a multiple of seq_size_per_block "
                f"({seq_size_per_block}); adjusted to {aligned}"
            )
        chunk_size = aligned
        fifo_scheduler_config.prefill_chunk_size = chunk_size

    logging.info(f"chunked prefill enabled: prefill_chunk_size={chunk_size}")
