import json
import logging
import os
import shutil
import socket
import time
from typing import Optional

import torch

from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.model_factory_register import ModelDict
from rtp_llm.ops import (
    FfnDisAggregateConfig,
    ParallelismConfig,
    PrefillCPConfig,
    RoleType,
    SpeculativeType,
)
from rtp_llm.utils.fuser import MountRwMode, fetch_remote_file_to_local

JIT_CACHE_ENV_NAMES = ("DG_JIT_CACHE_DIR", "TRITON_CACHE_DIR", "TILELANG_CACHE_DIR")
JIT_REMOTE_DONE_MARKER = "WRITE_DONE"
JIT_REMOTE_MANIFEST = "MANIFEST"
JIT_REMOTE_CONTROL_FILES = (JIT_REMOTE_DONE_MARKER, JIT_REMOTE_MANIFEST)
JIT_REMOTE_STAGING_PREFIX = ".staging-"
LEGACY_JIT_ENV_NAMES = ("REMOTE_JIT_DIR", "DG_JIT_REMOTE_CACHE_DIR")


def auto_configure_deepep(
    moe_config,
    deep_ep_config,
    parallelism_config: ParallelismConfig,
    role_type: RoleType,
    ll_num_max_token: int = 0,
):
    """
    Automatically configure DeepEP settings based on deployment scenario.

    If user has explicitly set any DeepEP configuration values (in deep_ep_config),
    those values will be used. Otherwise, automatic configuration will be applied
    based on the deployment scenario.

    Args:
        moe_config: MoeConfig object to modify
        deep_ep_config: DeepEPConfig object containing user-specified values (may be None)
        parallelism_config: ParallelismConfig containing tp_size, ep_size, world_size, local_world_size
        role_type: Role type (PREFILL, DECODE, or PDFUSION)

    Note: USE_ALL_GATHER should be enabled for pure TP scenarios (ep_size == tp_size).
    When USE_ALL_GATHER is enabled, DeepEP should not be used.

    Configuration rules (for 8-GPU machine):
    - Non-PD separation + Inference node + Single GPU (1TP): 0, 0, 0
    - Non-PD separation + Inference node + Single-node multi-GPU (>1TP): 1, 0, 0
    - Non-PD separation + Inference node + Multi-node multi-GPU: 1, 0, 1
    - PD separation + Prefill node + Single-node single-GPU (1TP): 0, 0, 0
    - PD separation + Decode node + Single-node single-GPU (1TP): 0, 0, 0
    - PD separation + Prefill node + Single-node multi-GPU (2-8 GPUs): 1, 0, 0
    - PD separation + Decode node + Single-node multi-GPU (2-8 GPUs): 1, 1, 0
    - PD separation + Prefill node + Multi-node multi-GPU (>=9 GPUs): 1, 0, 1
    - PD separation + Decode node + Multi-node multi-GPU (>=9 GPUs): 1, 1, 1
    """

    # in cp mode, do not use all gather, tp_size set to 1
    tp_size = parallelism_config.get_attn_tp_size()
    ep_size = parallelism_config.ep_size
    moe_config.ll_num_max_token = ll_num_max_token
    moe_config.use_all_gather = (
        moe_config.use_all_gather
        and not deep_ep_config.use_deepep_low_latency
        and (ep_size == tp_size or ep_size == 1)
    )
    if moe_config.use_all_gather:
        moe_config.use_deepep_moe = False
        moe_config.use_deepep_low_latency = False
        moe_config.use_deepep_internode = False
        logging.info(
            f"USE_ALL_GATHER is enabled (use_all_gather={moe_config.use_all_gather}), "
            f"all DeepEP settings are disabled (0, 0, 0)"
        )
        return

    # Check if user has explicitly set DeepEP configuration
    if (
        deep_ep_config.use_deepep_moe is None
        and deep_ep_config.use_deepep_internode is None
        and deep_ep_config.use_deepep_low_latency is None
    ):
        # All are None, use auto configuration
        _apply_auto_deepep_config(
            moe_config=moe_config,
            world_size=parallelism_config.world_size,
            local_world_size=parallelism_config.local_world_size,
            role_type=role_type,
        )
    else:
        # User has set at least one value, copy them to moe_config
        if deep_ep_config.use_deepep_moe is not None:
            moe_config.use_deepep_moe = deep_ep_config.use_deepep_moe
        if deep_ep_config.use_deepep_internode is not None:
            moe_config.use_deepep_internode = deep_ep_config.use_deepep_internode
        if deep_ep_config.use_deepep_low_latency is not None:
            moe_config.use_deepep_low_latency = deep_ep_config.use_deepep_low_latency
        logging.info(
            f"Using user-specified DeepEP configuration:\n"
            f"  USE_DEEPEP_MOE: {moe_config.use_deepep_moe}\n"
            f"  USE_DEEPEP_INTERNODE: {moe_config.use_deepep_internode}\n"
            f"  USE_DEEPEP_LOW_LATENCY: {moe_config.use_deepep_low_latency}"
            f"  ll_num_max_token: {moe_config.ll_num_max_token}"
        )


def _apply_auto_deepep_config(
    moe_config,
    world_size: int,
    local_world_size: int,
    role_type: RoleType,
):
    """
    Internal function to apply automatic DeepEP configuration based on deployment scenario.
    """
    logging.info("auto configure deepep work")

    # Determine if PD separation is enabled
    is_pd_separation = role_type in [RoleType.PREFILL, RoleType.DECODE]
    is_inference = role_type == RoleType.PDFUSION
    is_decode = role_type == RoleType.DECODE

    # Determine GPU configuration
    is_single_gpu = world_size == 1
    is_multi_gpu = world_size > 1
    is_multi_node = world_size > local_world_size

    # Apply configuration rules
    use_deepep_moe = False
    use_deepep_low_latency = False
    use_deepep_internode = False

    if is_inference:
        # Non-PD separation + Inference node
        if is_single_gpu:
            # Single GPU (1TP): 0, 0, 0
            use_deepep_moe = False
            use_deepep_low_latency = False
            use_deepep_internode = False
        elif is_multi_gpu and not is_multi_node:
            # Single-node multi-GPU (>1TP): 1, 0, 0
            use_deepep_moe = True
            use_deepep_low_latency = False
            use_deepep_internode = False
        elif is_multi_node:
            # Multi-node multi-GPU: 1, 0, 1
            use_deepep_moe = True
            use_deepep_low_latency = False
            use_deepep_internode = True
    elif is_pd_separation:
        # PD separation
        if is_single_gpu:
            # Single-node single-GPU: 0, 0, 0
            use_deepep_moe = False
            use_deepep_low_latency = False
            use_deepep_internode = False
        elif is_multi_gpu and not is_multi_node:
            # Single-node multi-GPU (2-8 GPUs)
            use_deepep_moe = True
            if is_decode:
                use_deepep_low_latency = True
        elif is_multi_node:
            # Multi-node multi-GPU (>=9 GPUs)
            use_deepep_moe = True
            use_deepep_internode = True
            if is_decode:
                use_deepep_low_latency = True

    # Set moe_config members directly
    moe_config.use_deepep_moe = use_deepep_moe
    moe_config.use_deepep_low_latency = use_deepep_low_latency
    moe_config.use_deepep_internode = use_deepep_internode

    logging.info(
        f"Auto-configured DeepEP settings based on deployment scenario:\n"
        f"  Role Type: {role_type}\n"
        f"  World Size: {world_size}\n"
        f"  Local World Size: {local_world_size}\n"
        f"  PD Separation: {is_pd_separation}\n"
        f"  USE_DEEPEP_MOE: {use_deepep_moe}\n"
        f"  USE_DEEPEP_LOW_LATENCY: {use_deepep_low_latency}\n"
        f"  USE_DEEPEP_INTERNODE: {use_deepep_internode}"
    )


def set_parallelism_config(
    parallelism_config: ParallelismConfig,
    world_rank: Optional[int] = None,
    py_ffn_disaggregate_config: Optional[FfnDisAggregateConfig] = None,
    py_prefill_cp_config: Optional[PrefillCPConfig] = None,
) -> None:
    """Update rank-related fields in ParallelismConfig from a given world_rank.

    Uses the same derivation as ParallelInfo: local_rank, tp_rank, dp_rank, ep_rank,
    ffn_tp_rank are computed from world_rank and the existing size fields.

    Args:
        parallelism_config: ParallelismConfig to update in place
        world_rank: World rank to apply (local_rank = world_rank % local_world_size, etc.). Default 0.
        py_ffn_disaggregate_config: Optional FFN disaggregate config to apply when enabled.
    """
    world_size = parallelism_config.world_size
    need_local = world_size > 1 and parallelism_config.local_world_size == 1
    if need_local:
        if torch.cuda.is_available():
            n = min(torch.cuda.device_count(), world_size)
        else:
            n = world_size
        parallelism_config.local_world_size = max(n, 1)

    # Resolve and validate parallelism configuration.
    # ep_size default is 0, which triggers automatic derivation.
    # Three supported modes:
    # 1. Single GPU: tp_size == 1, dp_size == 1, ep_size == 0 (default) → ep_size set to 1
    # 2. Pure TP:    ep_size explicitly set to 1, tp_size > 1, dp_size == 1
    # 3. EP mode:    ep_size == 0 (default), ep_size auto-derived as tp_size * dp_size
    if parallelism_config.ep_size == 1:
        assert (
            parallelism_config.tp_size >= 1
        ), f"Pure TP mode (ep_size=1) requires tp_size >= 1, got tp_size={parallelism_config.tp_size}"
        assert (
            parallelism_config.dp_size == 1
        ), f"Pure TP mode (ep_size=1) requires dp_size == 1, got dp_size={parallelism_config.dp_size}"
    elif parallelism_config.ep_size == 0:
        logging.info("parallelism_config.ep_size == 0, auto set to world size")
        parallelism_config.ep_size = (
            parallelism_config.tp_size * parallelism_config.dp_size
        )
    else:
        assert (
            parallelism_config.ep_size
            == parallelism_config.tp_size * parallelism_config.dp_size
        ), f"ep_size must be equal to 1 or tp_size * dp_size, got ep_size={parallelism_config.ep_size}, tp_size={parallelism_config.tp_size}, dp_size={parallelism_config.dp_size}"

    ffn_tp_size = parallelism_config.tp_size // parallelism_config.ffn_sp_size
    parallelism_config.ffn_tp_size = ffn_tp_size
    parallelism_config.enable_sp = parallelism_config.ffn_sp_size > 1
    if world_rank is not None:
        parallelism_config.world_rank = world_rank
    parallelism_config.local_rank = (
        parallelism_config.world_rank % parallelism_config.local_world_size
    )
    parallelism_config.tp_rank = (
        parallelism_config.world_rank % parallelism_config.tp_size
    )
    parallelism_config.dp_rank = (
        parallelism_config.world_rank // parallelism_config.tp_size
    )
    parallelism_config.ep_rank = (
        parallelism_config.world_rank % parallelism_config.ep_size
    )
    parallelism_config.ffn_tp_rank = (
        parallelism_config.tp_rank % parallelism_config.ffn_tp_size
    )

    # FfnDisAggregate
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

    if py_prefill_cp_config:
        parallelism_config.prefill_cp_config.method = py_prefill_cp_config.method
        parallelism_config.prefill_cp_config.comm_buffer_size = (
            py_prefill_cp_config.comm_buffer_size
        )
        parallelism_config.prefill_cp_config.kv_cache_sharded = (
            py_prefill_cp_config.kv_cache_sharded
        )
        if hasattr(py_prefill_cp_config, "prefill_cp_size") and hasattr(
            parallelism_config.prefill_cp_config, "prefill_cp_size"
        ):
            parallelism_config.prefill_cp_config.prefill_cp_size = (
                py_prefill_cp_config.prefill_cp_size
            )
        elif py_prefill_cp_config.kv_cache_sharded:
            logging.warning(
                "PREFILL_CP_SIZE is not available in this rtp_llm.ops build; "
                "prefill_cp_kv_cache_sharded was enabled but explicit CP size "
                "cannot be propagated."
            )
    logging.info(
        f"set_parallelism_config: rank {world_rank}\nparallelism_config={parallelism_config.to_string()}world_rank={world_rank}\n"
    )


def _infer_model_type(ckpt_path: str) -> Optional[str]:
    """Infer ``model_type`` by reading config.json from a local checkpoint directory.

    ModelDict owns lightweight architecture / repo mappings, so this does not
    import every model implementation during startup.
    """
    if not ckpt_path or not os.path.isdir(ckpt_path):
        return None
    config_path = os.path.join(ckpt_path, "config.json")
    if not os.path.isfile(config_path):
        return None
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        return ModelDict.get_ft_model_type_by_config(config)
    except Exception as e:
        logging.warning(f"Failed to infer model_type from {config_path}: {e}")
        return None


def setup_default_args(py_env_configs):
    set_parallelism_config(
        py_env_configs.parallelism_config,
        py_prefill_cp_config=py_env_configs.prefill_cp_config,
    )
    if not py_env_configs.model_args.tokenizer_path:
        py_env_configs.model_args.tokenizer_path = py_env_configs.model_args.ckpt_path

    if not py_env_configs.profiling_debug_logging_config.ft_alog_conf_path:
        py_env_configs.profiling_debug_logging_config.ft_alog_conf_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "alog.conf"
        )

    if not py_env_configs.model_args.model_type:
        py_env_configs.model_args.model_type = _infer_model_type(
            py_env_configs.model_args.ckpt_path
        )
    if not py_env_configs.model_args.model_type:
        raise ValueError(
            f"model_type is not set and could not be inferred from checkpoint path: {py_env_configs.model_args.ckpt_path}. Please provide --model_type or MODEL_TYPE environment variable."
        )

    # add rocm env config, if using default value, change it to optimize version
    # 这些特殊处理仍然需要设置环境变量（因为可能被 C++ 代码读取）
    if os.path.exists("/dev/kfd") and os.getenv("FT_DISABLE_CUSTOM_AR") is None:
        py_env_configs.py_hw_kernel_config.ft_disable_custom_ar = False
        logging.info(
            "[MI308X] enable FT_DISABLE_CUSTOM_AR by default, as amd has own implementation."
        )

    if (
        os.path.exists("/dev/kfd")
        and py_env_configs.kv_cache_config.seq_size_per_block == 0
    ):
        py_env_configs.kv_cache_config.seq_size_per_block = 16
        logging.info(
            "[MI308X] set SEQ_SIZE_PER_BLOCK 16 by default, as it just support 16 now."
        )
    if (
        os.path.exists("/dev/alixpu")
        and py_env_configs.kv_cache_config.seq_size_per_block == 0
    ):
        py_env_configs.kv_cache_config.seq_size_per_block = 256
        logging.info("set SEQ_SIZE_PER_BLOCK 256 by default")
    if py_env_configs.kv_cache_config.seq_size_per_block == 0:
        py_env_configs.kv_cache_config.seq_size_per_block = 64

    # Set NCCL_P2P_DISABLE for RTX GPUs or when CUDA is not available
    # Frontend doesn't need this setting
    if py_env_configs.role_config.role_type != RoleType.FRONTEND:
        if torch.cuda.is_available():
            if (
                "NCCL_P2P_DISABLE" not in os.environ
                and "RTX" in torch.cuda.get_device_name(0)
            ):
                os.environ["NCCL_P2P_DISABLE"] = "1"
                logging.info("set NCCL_P2P_DISABLE to 1")

    if (
        py_env_configs.role_config.role_type == RoleType.PREFILL
        or py_env_configs.role_config.role_type == RoleType.DECODE
    ):
        if py_env_configs.pd_separation_config.cache_store_rdma_mode == True:
            # AcclBarex envs in cache store, can be replaced by user config
            if os.getenv("ACCL_MAX_USER_MR_GB") is None:
                os.environ["ACCL_MAX_USER_MR_GB"] = "2000"
            if os.getenv("ACCL_SOFT_TX_DEPTH") is None:
                os.environ["ACCL_SOFT_TX_DEPTH"] = "8192"
            # for mlx
            if os.getenv("ACCL_RX_DEPTH") is None:
                os.environ["ACCL_RX_DEPTH"] = "32"
            if os.getenv("ACCL_TX_DEPTH") is None:
                os.environ["ACCL_TX_DEPTH"] = "512"
            # for eic
            if os.getenv("ACCL_RX_CONN_DEPTH") is None:
                os.environ["ACCL_RX_CONN_DEPTH"] = "32"
            if os.getenv("ACCL_TX_CONN_DEPTH") is None:
                os.environ["ACCL_TX_CONN_DEPTH"] = "512"
            logging.info(
                f"role type is {py_env_configs.role_config.role_type}, cache store rdma mode is {py_env_configs.pd_separation_config.cache_store_rdma_mode}, set ACCL_MAX_USER_MR_GB to {os.getenv('ACCL_MAX_USER_MR_GB')}, ACCL_SOFT_TX_DEPTH to {os.getenv('ACCL_SOFT_TX_DEPTH')}, ACCL_RX_DEPTH to {os.getenv('ACCL_RX_DEPTH')}, ACCL_TX_DEPTH to {os.getenv('ACCL_TX_DEPTH')}, ACCL_RX_CONN_DEPTH to {os.getenv('ACCL_RX_CONN_DEPTH')}, ACCL_TX_CONN_DEPTH to {os.getenv('ACCL_TX_CONN_DEPTH')}"
            )

    return py_env_configs


def fetch_model_files_to_local(py_env_configs: PyEnvConfigs):
    """Fetch remote model files to local and update py_env_configs in place."""
    # Fetch checkpoint_path from model_args
    model_args = py_env_configs.model_args
    if model_args.ckpt_path:
        model_args.ckpt_path = fetch_remote_file_to_local(model_args.ckpt_path)

    # Fetch tokenizer_path from model_args
    tokenizer_path = model_args.tokenizer_path
    if tokenizer_path:
        model_args.tokenizer_path = fetch_remote_file_to_local(tokenizer_path)

    # Fetch extra_data_path from vit_config
    vit_config = py_env_configs.vit_config
    if vit_config.extra_data_path:
        local_extra_data_path = fetch_remote_file_to_local(vit_config.extra_data_path)
        vit_config.local_extra_data_path = local_extra_data_path

    # Fetch ptuning_path from model_args
    if model_args.ptuning_path:
        model_args.ptuning_path = fetch_remote_file_to_local(model_args.ptuning_path)

    if model_args.phy2log_path:
        model_args.phy2log_path = fetch_remote_file_to_local(model_args.phy2log_path)

    # Fetch lora paths
    lora_config = py_env_configs.lora_config
    if lora_config.lora_info:
        try:
            lora_infos = json.loads(lora_config.lora_info)
            for lora_name, lora_path in lora_infos.items():
                lora_infos[lora_name] = fetch_remote_file_to_local(lora_path)
            # Update lora_info back to string format
            lora_config.lora_info = json.dumps(lora_infos)
        except (json.JSONDecodeError, TypeError) as e:
            logging.warning(
                f"Failed to parse lora_info: {e}, skipping lora path fetching"
            )

    # Fetch checkpoint_path if exists
    sp_config = py_env_configs.sp_config
    if sp_config.checkpoint_path:
        sp_config.checkpoint_path = fetch_remote_file_to_local(
            sp_config.checkpoint_path
        )

    logging.info(
        f"Fetched model files - checkpoint_path: {model_args.ckpt_path}, "
        f"tokenizer_path: {model_args.tokenizer_path}, "
        f"ptuning_path: {model_args.ptuning_path}, "
        f"extra_data_path: {vit_config.local_extra_data_path}, "
        f"phy2log_path: {model_args.phy2log_path}"
    )


def get_cuda_device_id_for_local_rank(local_rank: int) -> int:
    """Map logical local rank to CUDA device id.

    RTP_LLM_LOCAL_DEVICE_OFFSET is only intended for local multi-part smoke tests
    that simulate multiple nodes in separate server processes on one host.
    """
    return local_rank + int(os.environ.get("RTP_LLM_LOCAL_DEVICE_OFFSET", "0"))


def setup_cuda_device_and_accl_env(local_rank: int) -> None:
    """Apply CUDA device and ACCL env side effects (same as ParallelInfo.from_params)."""
    cuda_device_id = get_cuda_device_id_for_local_rank(local_rank)
    if torch.cuda.is_available():
        torch.cuda.set_device(cuda_device_id)
        logging.info(
            "local rank %s mapped to cuda device %s", local_rank, cuda_device_id
        )

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


def _local_jit_cache_dir() -> Optional[str]:
    workdir = os.environ.get("HIPPO_APP_WORKDIR") or os.environ.get(
        "HIPPO_PROC_WORKDIR"
    )
    if not workdir:
        return None
    return os.path.join(workdir, "jit_cache")


def _reset_local_jit_cache_dir(path: str) -> None:
    if os.path.islink(path) or os.path.isfile(path):
        os.remove(path)
    elif os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def _remote_staging_path(dst_path: str) -> str:
    return os.path.join(
        os.path.dirname(dst_path),
        f"{JIT_REMOTE_STAGING_PREFIX}{os.path.basename(dst_path)}."
        f"{os.getpid()}.{time.time_ns()}.tmp",
    )


def _copy_file_for_remote_publish(src_path: str, dst_path: str) -> None:
    try:
        if os.path.exists(dst_path) and os.path.getsize(src_path) == os.path.getsize(
            dst_path
        ):
            return
    except OSError:
        pass
    tmp_path = _remote_staging_path(dst_path)
    try:
        shutil.copy2(src_path, tmp_path)
        os.replace(tmp_path, dst_path)
    finally:
        try:
            os.remove(tmp_path)
        except FileNotFoundError:
            pass


def _replace_text_atomic(dst_path: str, content: str) -> None:
    tmp_path = _remote_staging_path(dst_path)
    try:
        with open(tmp_path, "w") as f:
            f.write(content)
        os.replace(tmp_path, dst_path)
    finally:
        try:
            os.remove(tmp_path)
        except FileNotFoundError:
            pass


def _copytree_into(
    src: str,
    dst: str,
    *,
    raise_on_error: bool = False,
    skip_remote_control_files: bool = False,
    remote_publish_copy: bool = False,
    copied_relpaths: Optional[list[str]] = None,
    progress_log_prefix: Optional[str] = None,
    progress_log_interval: int = 100,
) -> int:
    """Recursive copy mirroring `cp -a src/. dst/`."""
    src_abs = os.path.abspath(src)
    dst_abs = os.path.abspath(dst)
    if src_abs == dst_abs:
        logging.info("_copytree_into: source and destination are same: %s", src_abs)
        return 0
    if os.path.commonpath([src_abs, dst_abs]) == src_abs:
        raise RuntimeError(f"refuse to copy {src_abs} into its child {dst_abs}")

    files_copied = 0
    for root, dirs, files in os.walk(src_abs):
        if skip_remote_control_files:
            dirs[:] = [
                dirname
                for dirname in dirs
                if not dirname.startswith(JIT_REMOTE_STAGING_PREFIX)
            ]
        rel = os.path.relpath(root, src_abs)
        target_root = dst_abs if rel == "." else os.path.join(dst_abs, rel)
        os.makedirs(target_root, exist_ok=True)
        for filename in files:
            if skip_remote_control_files and (
                filename in JIT_REMOTE_CONTROL_FILES
                or filename.startswith(JIT_REMOTE_STAGING_PREFIX)
            ):
                continue
            src_path = os.path.join(root, filename)
            dst_path = os.path.join(target_root, filename)
            try:
                if remote_publish_copy:
                    _copy_file_for_remote_publish(src_path, dst_path)
                else:
                    shutil.copy2(src_path, dst_path)
                files_copied += 1
                if copied_relpaths is not None:
                    copied_relpaths.append(
                        filename if rel == "." else os.path.join(rel, filename)
                    )
                if (
                    progress_log_prefix
                    and progress_log_interval > 0
                    and files_copied % progress_log_interval == 0
                ):
                    logging.info(
                        "%s: copied %d files", progress_log_prefix, files_copied
                    )
            except Exception as e:
                if raise_on_error:
                    raise
                logging.warning(
                    "_copytree_into: failed to copy %s -> %s: %s",
                    src_path,
                    dst_path,
                    e,
                )
    return files_copied


def _safe_manifest_relpath(line: str) -> Optional[str]:
    relpath = line.strip()
    if not relpath or os.path.isabs(relpath):
        return None
    normalized = os.path.normpath(relpath)
    if normalized == "." or normalized.startswith(".."):
        return None
    return normalized


def _copy_manifest_files(
    src: str,
    dst: str,
    manifest_path: str,
    *,
    progress_log_prefix: Optional[str] = None,
    progress_log_interval: int = 100,
) -> int:
    src_abs = os.path.abspath(src)
    dst_abs = os.path.abspath(dst)
    files_copied = 0
    with open(manifest_path, "r") as manifest:
        for line in manifest:
            relpath = _safe_manifest_relpath(line)
            if not relpath:
                logging.warning("skip invalid JIT cache manifest entry: %r", line)
                continue
            src_path = os.path.join(src_abs, relpath)
            if not os.path.isfile(src_path):
                logging.warning("skip missing JIT cache manifest file: %s", src_path)
                continue
            dst_path = os.path.join(dst_abs, relpath)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy2(src_path, dst_path)
            files_copied += 1
            if (
                progress_log_prefix
                and progress_log_interval > 0
                and files_copied % progress_log_interval == 0
            ):
                logging.info("%s: copied %d files", progress_log_prefix, files_copied)
    return files_copied


def _dir_has_copyable_files(path: str) -> bool:
    for _, dirs, files in os.walk(path):
        dirs[:] = [
            dirname
            for dirname in dirs
            if not dirname.startswith(JIT_REMOTE_STAGING_PREFIX)
        ]
        if any(filename not in JIT_REMOTE_CONTROL_FILES for filename in files):
            return True
    return False


def _iter_jit_cache_dirs(*, require_files: bool = False):
    seen = set()
    missing_env_names = []
    for env_name in JIT_CACHE_ENV_NAMES:
        cache_dir = os.environ.get(env_name)
        if not cache_dir:
            missing_env_names.append(env_name)
            continue
        cache_dir = os.path.abspath(cache_dir)
        if cache_dir in seen:
            continue
        seen.add(cache_dir)
        if not os.path.isdir(cache_dir):
            logging.warning(
                "JIT cache env %s points to non-existing dir %s; skip",
                env_name,
                cache_dir,
            )
            continue
        if require_files and not _dir_has_copyable_files(cache_dir):
            logging.info(
                "JIT cache env %s points to empty dir %s; skip remote publishing",
                env_name,
                cache_dir,
            )
            continue
        yield env_name, cache_dir
    if missing_env_names:
        logging.info("JIT cache envs not set: %s", missing_env_names)


def _warn_legacy_jit_envs() -> None:
    for env_name in LEGACY_JIT_ENV_NAMES:
        if os.environ.get(env_name):
            logging.warning(
                "legacy JIT env %s is ignored; use REMOTE_JIT_READ_DIR or "
                "WARM_UP_JIT_AND_WRITE_REMOTE instead",
                env_name,
            )


def setup_jit_cache_envs(py_env_configs: PyEnvConfigs) -> None:
    """Set local JIT cache envs before any subprocess can trigger JIT.

    JIT compilation writes locally first; the explicit warm-up writer copies the
    finished artifacts to a remote RW mount after service startup succeeds.
    """
    _warn_legacy_jit_envs()
    jit_config = py_env_configs.jit_config
    remote_read_dir = (jit_config.remote_jit_read_dir or "").strip()
    remote_write_dir = (jit_config.warm_up_jit_and_write_remote or "").strip()
    target_local_dir = _local_jit_cache_dir()

    if remote_read_dir:
        if not target_local_dir:
            raise RuntimeError(
                f"REMOTE_JIT_READ_DIR={remote_read_dir} requires HIPPO_APP_WORKDIR or HIPPO_PROC_WORKDIR"
            )
        local_read_dir = fetch_remote_file_to_local(
            remote_read_dir, MountRwMode.RWMODE_RO
        )
        if not local_read_dir:
            raise RuntimeError(
                f"REMOTE_JIT_READ_DIR={remote_read_dir} fuse returned empty local path"
            )
        if not os.path.isdir(local_read_dir):
            raise RuntimeError(
                f"REMOTE_JIT_READ_DIR={remote_read_dir} resolved to non-existing dir {local_read_dir}"
            )
        os.makedirs(target_local_dir, exist_ok=True)
        done_marker = os.path.join(local_read_dir, JIT_REMOTE_DONE_MARKER)
        if not os.path.exists(done_marker):
            logging.warning(
                "REMOTE_JIT_READ_DIR=%s (%s) has no %s marker; skip seeding and "
                "use local JIT cache dir %s",
                remote_read_dir,
                local_read_dir,
                JIT_REMOTE_DONE_MARKER,
                target_local_dir,
            )
            for env_name in JIT_CACHE_ENV_NAMES:
                os.environ[env_name] = target_local_dir
            return
        copy_begin = time.time()
        manifest_path = os.path.join(local_read_dir, JIT_REMOTE_MANIFEST)
        if os.path.exists(manifest_path):
            n_files = _copy_manifest_files(
                local_read_dir,
                target_local_dir,
                manifest_path,
                progress_log_prefix="setup_jit_cache_envs: manifest seed",
            )
            seed_source = f"manifest {manifest_path}"
        else:
            n_files = _copytree_into(
                local_read_dir,
                target_local_dir,
                skip_remote_control_files=True,
                progress_log_prefix="setup_jit_cache_envs: tree seed",
            )
            seed_source = "remote tree"
        for env_name in JIT_CACHE_ENV_NAMES:
            os.environ[env_name] = target_local_dir
        logging.info(
            "setup_jit_cache_envs: seeded %d files from REMOTE_JIT_READ_DIR=%s (%s, %s) "
            "into %s in %.2fs; set %s=%s",
            n_files,
            remote_read_dir,
            local_read_dir,
            seed_source,
            target_local_dir,
            time.time() - copy_begin,
            list(JIT_CACHE_ENV_NAMES),
            target_local_dir,
        )
        return

    if remote_write_dir:
        if not target_local_dir:
            raise RuntimeError(
                f"WARM_UP_JIT_AND_WRITE_REMOTE={remote_write_dir} requires HIPPO_APP_WORKDIR or HIPPO_PROC_WORKDIR"
            )
        _reset_local_jit_cache_dir(target_local_dir)
        for env_name in JIT_CACHE_ENV_NAMES:
            os.environ[env_name] = target_local_dir
        logging.info(
            "setup_jit_cache_envs: WARM_UP_JIT_AND_WRITE_REMOTE=%s uses clean local "
            "JIT cache dir %s; set %s=%s",
            remote_write_dir,
            target_local_dir,
            list(JIT_CACHE_ENV_NAMES),
            target_local_dir,
        )
        return

    if not target_local_dir:
        logging.info(
            "setup_jit_cache_envs: HIPPO_APP_WORKDIR/HIPPO_PROC_WORKDIR not set; leaving JIT envs as-is"
        )
        return

    os.makedirs(target_local_dir, exist_ok=True)
    for env_name in JIT_CACHE_ENV_NAMES:
        existing = os.environ.get(env_name)
        if existing:
            logging.info(
                "setup_jit_cache_envs: %s already set to %s; not overriding",
                env_name,
                existing,
            )
            continue
        os.environ[env_name] = target_local_dir
        logging.info("setup_jit_cache_envs: %s=%s", env_name, target_local_dir)


def maybe_write_jit_cache_to_remote(
    py_env_configs: PyEnvConfigs, startup_warmup_succeeded: bool
) -> None:
    remote_write_dir = (
        py_env_configs.jit_config.warm_up_jit_and_write_remote or ""
    ).strip()
    if not remote_write_dir:
        return

    if not startup_warmup_succeeded:
        logging.warning(
            "WARM_UP_JIT_AND_WRITE_REMOTE=%s is set but startup real warmup did not "
            "run successfully; skip remote JIT cache publishing",
            remote_write_dir,
        )
        return

    source_dirs = list(_iter_jit_cache_dirs(require_files=True))
    if not source_dirs:
        logging.warning(
            "WARM_UP_JIT_AND_WRITE_REMOTE=%s is set but no local JIT cache artifacts "
            "exist; skip remote fuse and publishing",
            remote_write_dir,
        )
        return

    local_write_dir = fetch_remote_file_to_local(
        remote_write_dir, MountRwMode.RWMODE_RW
    )
    if not local_write_dir:
        raise RuntimeError(
            f"WARM_UP_JIT_AND_WRITE_REMOTE={remote_write_dir} fuse returned empty local path"
        )
    os.makedirs(local_write_dir, exist_ok=True)

    # JIT cache file paths are content-addressed. Existing files can remain in
    # place while this publish refreshes them; matching files are not overwritten,
    # and readers only trust the manifest after it is atomically replaced below.
    copy_begin = time.time()
    total_files = 0
    manifest_entries: list[str] = []
    source_dir_desc = []
    for env_name, cache_dir in source_dirs:
        source_dir_desc.append(f"{env_name}={cache_dir}")
        copied_relpaths: list[str] = []
        copied = _copytree_into(
            cache_dir,
            local_write_dir,
            raise_on_error=True,
            skip_remote_control_files=True,
            remote_publish_copy=True,
            copied_relpaths=copied_relpaths,
            progress_log_prefix=(
                f"maybe_write_jit_cache_to_remote: publishing {env_name}"
            ),
        )
        total_files += copied
        manifest_entries.extend(copied_relpaths)
        logging.info(
            "maybe_write_jit_cache_to_remote: copied %d files from %s (%s) to %s",
            copied,
            env_name,
            cache_dir,
            local_write_dir,
        )

    if total_files == 0:
        logging.warning(
            "WARM_UP_JIT_AND_WRITE_REMOTE=%s found source dirs %s but no files; "
            "skip publishing",
            remote_write_dir,
            source_dir_desc,
        )
        return

    manifest_path = os.path.join(local_write_dir, JIT_REMOTE_MANIFEST)
    manifest_content = "".join(
        f"{relpath}\n" for relpath in sorted(set(manifest_entries))
    )
    _replace_text_atomic(manifest_path, manifest_content)

    done_marker = os.path.join(local_write_dir, JIT_REMOTE_DONE_MARKER)
    marker_content = (
        f"pid={os.getpid()}\n"
        f"files={total_files}\n"
        f"manifest={JIT_REMOTE_MANIFEST}\n"
    )
    _replace_text_atomic(done_marker, marker_content)
    logging.info(
        "maybe_write_jit_cache_to_remote: published %d files from %s to "
        "WARM_UP_JIT_AND_WRITE_REMOTE=%s (%s) in %.2fs",
        total_files,
        source_dir_desc,
        remote_write_dir,
        local_write_dir,
        time.time() - copy_begin,
    )


def setup_and_configure_server(py_env_configs: PyEnvConfigs):
    """
    Build parallelism_config from env and run auto_configure_deepep.
    Caller should run setup_default_args and fetch_model_files_to_local before this.

    Args:
        py_env_configs: PyEnvConfigs object to configure
    """
    setup_default_args(py_env_configs)
    setup_jit_cache_envs(py_env_configs)
    fetch_model_files_to_local(py_env_configs)
    ll_num_max_token = py_env_configs.concurrency_config.concurrency_limit
    if py_env_configs.role_config.role_type == RoleType.DECODE:
        # DeepEP low-latency dispatch allocates per routed expert row.
        model_config = getattr(py_env_configs, "model_config", None)
        moe_k = getattr(model_config, "moe_k", 0) if model_config is not None else 0
        if (
            not moe_k
            and getattr(py_env_configs.model_args, "model_type", "") == "minimax_m3"
        ):
            moe_k = 4
        ll_num_max_token *= max(1, int(moe_k or 1))
    sp_type = py_env_configs.sp_config.type  # Get SpeculativeType enum value
    if py_env_configs.sp_config.type != SpeculativeType.NONE:
        ll_num_max_token *= py_env_configs.sp_config.gen_num_per_cycle + 1

    auto_configure_deepep(
        moe_config=py_env_configs.moe_config,
        deep_ep_config=py_env_configs.deep_ep_config,
        parallelism_config=py_env_configs.parallelism_config,
        role_type=py_env_configs.role_config.role_type,
        ll_num_max_token=ll_num_max_token,
    )

    # Set local ip if not already set (e.g. for world_info / distributed_server)
    if not py_env_configs.server_config.ip:
        py_env_configs.server_config.ip = socket.gethostbyname(socket.gethostname())
