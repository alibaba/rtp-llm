import json
import logging
import os
import socket
from typing import Optional

import torch

from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.model_factory_register import ModelDict
from rtp_llm.ops import (
    FfnDisAggregateConfig,
    ParallelismConfig,
    RoleType,
    SpeculativeType,
)
from rtp_llm.utils.fuser import fetch_remote_file_to_local


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

    tp_size = parallelism_config.tp_size
    ep_size = parallelism_config.ep_size
    world_size = parallelism_config.world_size
    moe_config.ll_num_max_token = ll_num_max_token
    moe_config.use_all_gather = (
        moe_config.use_all_gather
        and not deep_ep_config.use_deepep_low_latency
        and ep_size == tp_size
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
        if deep_ep_config.support_dual_mode is not None:
            moe_config.support_dual_mode = deep_ep_config.support_dual_mode
        logging.info(
            f"Using user-specified DeepEP configuration:\n"
            f"  USE_DEEPEP_MOE: {moe_config.use_deepep_moe}\n"
            f"  USE_DEEPEP_INTERNODE: {moe_config.use_deepep_internode}\n"
            f"  USE_DEEPEP_LOW_LATENCY: {moe_config.use_deepep_low_latency}\n"
            f"  SUPPORT_DUAL_MODE: {moe_config.support_dual_mode}\n"
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
    world_rank: int = 0,
    py_ffn_disaggregate_config: Optional[FfnDisAggregateConfig] = None,
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

    expected_ep = parallelism_config.tp_size * parallelism_config.dp_size
    need_ep = expected_ep > 1 and parallelism_config.ep_size == 1
    if need_ep:
        parallelism_config.ep_size = expected_ep
    ffn_tp_size = parallelism_config.tp_size // parallelism_config.ffn_sp_size
    parallelism_config.ffn_tp_size = ffn_tp_size
    parallelism_config.enable_sp = parallelism_config.ffn_sp_size > 1
    parallelism_config.world_rank = world_rank
    parallelism_config.local_rank = world_rank % parallelism_config.local_world_size
    parallelism_config.tp_rank = world_rank % parallelism_config.tp_size
    parallelism_config.dp_rank = world_rank // parallelism_config.tp_size
    parallelism_config.ep_rank = world_rank % parallelism_config.ep_size
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
    logging.info(
        f"set_parallelism_config : rank {world_rank}, parallelism_config={parallelism_config.to_string()}, world_rank={world_rank}"
    )


def setup_default_args(py_env_configs):
    set_parallelism_config(py_env_configs.parallelism_config)
    if not py_env_configs.model_args.tokenizer_path:
        py_env_configs.model_args.tokenizer_path = py_env_configs.model_args.ckpt_path

    if not py_env_configs.profiling_debug_logging_config.ft_alog_conf_path:
        py_env_configs.profiling_debug_logging_config.ft_alog_conf_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "alog.conf"
        )

    if not py_env_configs.model_args.model_type:
        py_env_configs.model_args.model_type = ModelDict.get_ft_model_type_by_config(
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

    if os.path.exists("/dev/kfd") and os.getenv("SEQ_SIZE_PER_BLOCK") is None:
        py_env_configs.kv_cache_config.seq_size_per_block = 16
        logging.info(
            "[MI308X] set SEQ_SIZE_PER_BLOCK 16 by default, as it just support 16 now."
        )
    if os.path.exists("/dev/alixpu") and os.getenv("SEQ_SIZE_PER_BLOCK") is None:
        py_env_configs.kv_cache_config.seq_size_per_block = 256
        logging.info("set SEQ_SIZE_PER_BLOCK 256 by default")

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

    # Fetch extra_data_path from model_args
    if model_args.extra_data_path:
        local_extra_data_path = fetch_remote_file_to_local(model_args.extra_data_path)
        model_args.local_extra_data_path = local_extra_data_path

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
        f"extra_data_path: {model_args.local_extra_data_path}, "
        f"phy2log_path: {model_args.phy2log_path}"
    )


def setup_cuda_device_and_accl_env(local_rank: int) -> None:
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


def setup_and_configure_server(py_env_configs: PyEnvConfigs):
    """
    Build parallelism_config from env and run auto_configure_deepep.
    Caller should run setup_default_args and fetch_model_files_to_local before this.

    Args:
        py_env_configs: PyEnvConfigs object to configure
    """
    setup_default_args(py_env_configs)
    fetch_model_files_to_local(py_env_configs)
    ll_num_max_token = py_env_configs.concurrency_config.concurrency_limit
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
