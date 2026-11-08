import json
import logging
import os
import socket
import subprocess
from typing import Optional

import torch

from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.model_factory_register import ModelDict
from rtp_llm.models_py.distributed.rank_layout import Group, RankLayout
from rtp_llm.ops import (
    ParallelismConfig,
    PrefillCPConfig,
    RoleType,
    SpeculativeType,
)
from rtp_llm.utils.fuser import fetch_remote_file_to_local

RUN_AFFINITY_TIMEOUT_SEC = 10


def finalize_rank_local_deepep_internode(
    moe_config,
    deep_ep_config,
    parallelism_config: ParallelismConfig,
) -> None:
    """Correct auto-derived internode mode after the worker rank is known.

    A node launcher may host ranks from different PP stages; explicit user
    settings are left untouched.
    """
    if (
        parallelism_config.pp_size <= 1
        or not moe_config.use_deepep_moe
        or deep_ep_config.use_deepep_moe is not None
        or deep_ep_config.use_deepep_internode is not None
        or deep_ep_config.use_deepep_low_latency is not None
        or deep_ep_config.use_mori_ep is not None
    ):
        return
    layout = RankLayout.from_parallelism_config(parallelism_config)
    stage_group = layout.group_of(Group.STAGE, parallelism_config.world_rank)
    moe_config.use_deepep_internode = (
        min(stage_group) // parallelism_config.local_world_size
        != max(stage_group) // parallelism_config.local_world_size
    )


def _auto_deepep_supported_on_visible_devices(local_world_size: int = 0) -> bool:
    """Whether automatic DeepEP selection is valid for participating devices.

    Consumer Blackwell SM120 uses the generic SM120 collective strategy;
    DeepEP does not ship compatible kernels there. Models on other SM12x
    architectures retain main's startup behavior and validate their concrete
    backends later.
    Devices outside the local world do not participate and must not affect
    selection.
    """
    if not torch.cuda.is_available():
        # Preserve CPU-only config parsing and unit-test behavior.  Runtime
        # hardware validation occurs later when CUDA is initialized.
        return True
    device_count = torch.cuda.device_count()
    participating_device_count = min(
        device_count,
        local_world_size if local_world_size > 0 else device_count,
    )
    capabilities = [
        torch.cuda.get_device_capability(device_index)
        for device_index in range(participating_device_count)
    ]
    return all(capability != (12, 0) for capability in capabilities)


def auto_configure_deepep(
    moe_config,
    deep_ep_config,
    parallelism_config: ParallelismConfig,
    role_type: RoleType,
    ll_num_max_token: int = 0,
    model_type: str = "",
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

    # MoE uses physical TP topology, not attention TP. get_attn_tp_size()
    # returns 1 when CP is enabled, which would incorrectly disable
    # use_all_gather for ep_size == tp_size configurations.
    tp_size = parallelism_config.tp_size
    ep_size = parallelism_config.ep_size
    dp_size = parallelism_config.dp_size
    moe_config.ll_num_max_token = ll_num_max_token

    # Explicit MoriEP is incompatible with use_all_gather (PURE_TP router).
    # Disable use_all_gather when user explicitly requests MoriEP.
    if deep_ep_config.use_mori_ep:
        moe_config.use_all_gather = False
        logging.info(
            "use_mori_ep is explicitly enabled; disabling use_all_gather "
            "to allow MoriEP router selection"
        )

    # allgather default applies only to single GPU and pure TP (no CP).
    # PureCP / PureDP routers exist but must be opted in via --moe_strategy
    # (auto-selection falls back to DeepEP). CP-enabled topologies share the
    # tp>1 / dp==1 / ep==tp shape with pure TP, so they must be excluded here
    # to avoid silently disabling DeepEP without selecting PureCP.
    prefill_cp_enabled = parallelism_config.prefill_cp_config.is_enabled()
    is_single_gpu = ep_size == 1
    is_pure_tp = (
        tp_size > 1 and dp_size == 1 and ep_size == tp_size and not prefill_cp_enabled
    )
    # Explicit opt-in via --moe_strategy must preserve use_all_gather, otherwise
    # the matching strategy's check_conditions (which requires use_all_gather)
    # will never be satisfied. Topology is validated here to keep the auto path
    # falling back to DeepEP when --moe_strategy and shape disagree.
    explicit_pure_dp = (
        moe_config.moe_strategy == "fp8_per_block_pure_dp"
        and tp_size == 1
        and dp_size > 1
        and ep_size == dp_size
    )
    explicit_pure_cp = (
        moe_config.moe_strategy == "fp8_per_block_pure_cp"
        and tp_size > 1
        and dp_size == 1
        and ep_size == tp_size
        and prefill_cp_enabled
    )

    # use_deepep_moe disables use_all_gather only when a viable DeepEP path
    # exists (ep_size > 1). On single-GPU configs there is no DeepEP path,
    # so honor the implicit allgather fallback even if --use_deepep_moe was set.
    moe_config.use_all_gather = (
        moe_config.use_all_gather
        and not deep_ep_config.use_deepep_low_latency
        and (is_single_gpu or not deep_ep_config.use_deepep_moe)
        and (is_single_gpu or is_pure_tp or explicit_pure_dp or explicit_pure_cp)
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
        and deep_ep_config.use_mori_ep is None
    ):
        # All are None, use auto configuration
        is_model_worker = role_type in {
            RoleType.PREFILL,
            RoleType.DECODE,
            RoleType.PDFUSION,
        }
        if (
            is_model_worker
            and model_type == "deepseek_v4"
            and parallelism_config.pp_size == 1
            and parallelism_config.ep_size > 1
            and not _auto_deepep_supported_on_visible_devices(
                parallelism_config.local_world_size
            )
        ):
            moe_config.use_deepep_moe = False
            moe_config.use_deepep_low_latency = False
            moe_config.use_deepep_internode = False
            logging.info(
                "Automatic DeepEP selection is disabled on the participating "
                "GPU architecture; a concrete MoE backend will validate any "
                "required fallback after the model configuration is loaded"
            )
        else:
            _apply_auto_deepep_config(
                moe_config=moe_config,
                parallelism_config=parallelism_config,
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
        if deep_ep_config.use_mori_ep is not None:
            moe_config.use_mori_ep = deep_ep_config.use_mori_ep

        # MoriEP does not support low-latency mode. When use_mori_ep is
        # explicitly enabled and use_deepep_low_latency was not explicitly set
        # by the user, disable the default True value so that MoriEP strategy
        # can pass its check_conditions.
        if moe_config.use_mori_ep and deep_ep_config.use_deepep_low_latency is None:
            moe_config.use_deepep_low_latency = False
            logging.info(
                "use_mori_ep is enabled and use_deepep_low_latency was not "
                "explicitly set; disabling the default low-latency mode so "
                "that MoriEP strategy is selectable"
            )

        logging.info(
            f"Using user-specified DeepEP/MoriEP configuration:\n"
            f"  USE_DEEPEP_MOE: {moe_config.use_deepep_moe}\n"
            f"  USE_DEEPEP_INTERNODE: {moe_config.use_deepep_internode}\n"
            f"  USE_DEEPEP_LOW_LATENCY: {moe_config.use_deepep_low_latency}\n"
            f"  USE_MORI_EP: {moe_config.use_mori_ep}\n"
            f"  ll_num_max_token: {moe_config.ll_num_max_token}"
        )


def _apply_auto_deepep_config(
    moe_config,
    parallelism_config,
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

    # Determine GPU configuration via RankLayout (stage-local view under PP).
    local_world_size = parallelism_config.local_world_size
    world_size = parallelism_config.world_size
    if parallelism_config.pp_size > 1:
        layout = RankLayout.from_parallelism_config(parallelism_config)
        stage_size = layout.size_of(Group.STAGE)
        is_single_gpu = stage_size == 1
        is_multi_gpu = stage_size > 1
        stage_group = layout.group_of(Group.STAGE, parallelism_config.world_rank)
        is_multi_node = (
            min(stage_group) // local_world_size
            != max(stage_group) // local_world_size
        )
    else:
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


def _validate_sm120_dsv4_worker_profile(py_env_configs) -> None:
    """Fail fast on unsupported exact-SM120 DSv4 worker topologies.

    Applies when any participating visible device reports the exact
    ``sm120`` architecture and the role is a DSv4 model role. Other
    architectures and roles keep main behavior. Accepted profiles:

      PREFILL : PP1, physical TP4 (CP4), DP1, EP4, world4, prefill CP enabled
      DECODE  : PP1, TP1, DP4, EP4, world4, no local CP; eager execution
                only (CUDA graphs are rejected)
      PDFUSION: PP1, TP1, DP == EP == world, no local CP; over DP > 1 the
                sm120_a2a_decode backend runs eagerly only (CUDA graphs are
                rejected), and non-speculative sm120_a2a_decode PDFUSION over
                DP > 1 is unsupported (use active MTP or separate roles)

    Keep this in sync with the C++ twin in
    ``rtp_llm/cpp/normal_engine/Dsv4DpPrefillPolicy.h`` (engine-side
    admission, needed because a wrong topology can hang inside NCCL before
    Python error handling runs) and with the auto-selection table in
    ``rtp_llm/models_py/modules/factory/fused_moe/impl/cuda/strategy/fp8_fp4.py``.
    """
    role = py_env_configs.role_config.role_type
    if role not in {RoleType.PREFILL, RoleType.DECODE, RoleType.PDFUSION}:
        return
    p = py_env_configs.parallelism_config
    if _auto_deepep_supported_on_visible_devices(p.local_world_size):
        # No participating device is exact SM120; not the target lane.
        return

    cp_cfg = getattr(p, "prefill_cp_config", None)
    cp_on = bool(cp_cfg and cp_cfg.is_enabled())
    spec_on = py_env_configs.sp_config.type != SpeculativeType.NONE
    graph_on = bool(
        getattr(
            getattr(py_env_configs, "py_hw_kernel_config", None),
            "enable_cuda_graph",
            False,
        )
    )
    actual = (
        f"role={role}, pp={p.pp_size}, tp={p.tp_size}, dp={p.dp_size}, "
        f"ep={p.ep_size}, world={p.world_size}, prefill_cp={cp_on}, "
        f"speculative={spec_on}"
    )

    required = None
    ok = False
    if p.pp_size != 1:
        required = "pp_size=1 (non-PP)"
    elif role == RoleType.PREFILL:
        ok = (
            p.tp_size == 4
            and p.dp_size == 1
            and p.ep_size == 4
            and p.world_size == 4
            and cp_on
        )
        required = "TP4/CP4/DP1/EP4/world4 PREFILL"
    elif role == RoleType.DECODE:
        ok = (
            p.tp_size == 1
            and p.dp_size == 4
            and p.ep_size == 4
            and p.world_size == 4
            and not cp_on
        )
        required = "TP1/DP4/EP4/world4 DECODE (CP off)"
    else:  # PDFUSION
        base = p.tp_size == 1 and p.dp_size == p.ep_size == p.world_size and not cp_on
        ok = base
        required = "TP1 DP=EP=world PDFUSION (PP1, no local CP)"
    if not ok:
        raise ValueError(
            f"Unsupported exact-SM120 DSv4 worker profile: {actual}. "
            f"Required: {required}."
        )

    # Resolve the required MoE strategy at startup instead of letting an
    # unsupported `auto` surface during execution.
    moe = py_env_configs.moe_config
    canonical = {
        RoleType.PREFILL: "sm120_fused_moe",
        RoleType.DECODE: "sm120_a2a_decode",
        RoleType.PDFUSION: "sm120_a2a_decode" if p.dp_size > 1 else None,
    }[role]
    current = getattr(moe, "moe_strategy", "auto") or "auto"
    if canonical is not None:
        if current == "auto":
            moe.moe_strategy = canonical
            logging.info(
                "Resolved exact-SM120 DSv4 %s moe_strategy to %s", role, canonical
            )
        elif current != canonical:
            raise ValueError(
                f"Unsupported exact-SM120 DSv4 {role} moe_strategy={current}. "
                f"Required: {canonical}."
            )

    # The exact retired profile: resolved sm120_a2a_decode PDFUSION over DP>1
    # without speculation has no phase protocol that keeps EP peers inside
    # the same collective order. Reject before warmup or graph capture.
    if (
        role == RoleType.PDFUSION
        and p.pp_size == 1
        and p.tp_size == 1
        and p.dp_size > 1
        and p.dp_size == p.ep_size == p.world_size
        and not spec_on
        and not cp_on
        and moe.moe_strategy == "sm120_a2a_decode"
    ):
        raise ValueError(
            "Unsupported exact-SM120 DSv4 non-speculative sm120_a2a_decode "
            f"PDFUSION profile: {actual}. Use active MTP or separate "
            "PREFILL/DECODE roles."
        )

    # sm120_a2a_decode decode execution is eager-only: its DP=EP collective
    # protocol derives per-round payload extents from exchanged row counts,
    # which captured CUDA graphs cannot reproduce. Reject a resolved
    # sm120_a2a_decode profile that still enables CUDA graphs, before warmup.
    if moe.moe_strategy == "sm120_a2a_decode" and graph_on:
        raise ValueError(
            "sm120_a2a_decode requires eager execution: CUDA graphs are "
            "unsupported for the exact-SM120 DSv4 DECODE and "
            "PDFUSION(DP>1) collective protocol; set enable_cuda_graph=false"
        )


def set_parallelism_config(
    parallelism_config: ParallelismConfig,
    world_rank: Optional[int] = None,
    py_prefill_cp_config: Optional[PrefillCPConfig] = None,
) -> None:
    """Update rank-related fields in ParallelismConfig from a given world_rank.

    Uses the same derivation as ParallelInfo: local_rank, tp_rank, dp_rank, ep_rank,
    ffn_tp_rank are computed from world_rank and the existing size fields.

    Args:
        parallelism_config: ParallelismConfig to update in place
        world_rank: World rank to apply (local_rank = world_rank % local_world_size, etc.). Default 0.
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

    ffn_tp_size = parallelism_config.tp_size
    parallelism_config.ffn_tp_size = ffn_tp_size
    if world_rank is not None:
        parallelism_config.world_rank = world_rank
    parallelism_config.local_rank = (
        parallelism_config.world_rank % parallelism_config.local_world_size
    )
    layout = RankLayout.from_parallelism_config(parallelism_config)
    coord = layout.coord_of(parallelism_config.world_rank)
    parallelism_config.tp_rank = coord.tp
    parallelism_config.dp_rank = coord.dp
    parallelism_config.pp_rank = coord.pp
    parallelism_config.ep_rank = layout.ep_rank_of(
        parallelism_config.world_rank, parallelism_config.ep_size
    )
    parallelism_config.ffn_tp_rank = (
        parallelism_config.tp_rank % parallelism_config.ffn_tp_size
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


def load_gpu_nic_affinity() -> bool:
    if os.environ.get("ACCL_NIC_GPU_AFFINITY") is not None:
        return True

    run_affinity_path = "/usr/local/bin/run_affinity"
    if not os.path.exists(run_affinity_path):
        logging.info("get gpu nic affinity failed, %s not exist", run_affinity_path)
        return False

    try:
        subprocess.run(
            [run_affinity_path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=RUN_AFFINITY_TIMEOUT_SEC,
        )
    except subprocess.TimeoutExpired:
        logging.warning(
            "get gpu nic affinity timed out after %ss while running %s; continue without affinity",
            RUN_AFFINITY_TIMEOUT_SEC,
            run_affinity_path,
        )
        return False
    except (subprocess.CalledProcessError, OSError) as e:
        logging.warning(
            "get gpu nic affinity failed, run %s failed, exception is %s",
            run_affinity_path,
            e,
        )
        return False
    except Exception as e:
        logging.warning(
            "get gpu nic affinity failed unexpectedly while running %s: %s",
            run_affinity_path,
            e,
        )
        return False

    json_path = "npu_nic_affinity.json"
    if not os.path.exists(json_path):
        logging.info("get gpu nic affinity failed, %s does not exist", json_path)
        return False

    try:
        with open(json_path) as affinity_file:
            content = affinity_file.read().strip()
        os.environ["ACCL_NIC_GPU_AFFINITY"] = content
        logging.info(
            "get gpu nic affinity success, set env ACCL_NIC_GPU_AFFINITY to %s",
            content,
        )
        return True
    except Exception as e:
        logging.info(
            "get gpu nic affinity failed, load %s failed, exception is %s",
            json_path,
            e,
        )
        return False


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


def configure_kv_cache_event_host_ip_port(py_env_configs: PyEnvConfigs) -> None:
    """Resolve the KVCM host identity after this backend rank is known."""
    kv_cache_config = py_env_configs.kv_cache_config
    parallelism_config = py_env_configs.parallelism_config
    if (
        kv_cache_config.kv_cache_event_publisher_type != "kvcm"
        or parallelism_config.tp_rank != 0
    ):
        return

    server_config = py_env_configs.server_config
    derived = not kv_cache_config.kv_cache_event_host_ip_port
    if derived:
        server_ip = server_config.ip or socket.gethostbyname(socket.gethostname())
        kv_cache_config.kv_cache_event_host_ip_port = (
            f"{server_ip}:{server_config.server_port}"
        )
        logging.info(
            "KV_CACHE_EVENT_HOST_IP_PORT was empty; derived per-rank endpoint "
            "%s from start_port=%s rank_id=%s worker_info_port_num=%s",
            kv_cache_config.kv_cache_event_host_ip_port,
            server_config.start_port,
            server_config.rank_id,
            server_config.worker_info_port_num,
        )

    if parallelism_config.dp_size > 1:
        logging.warning(
            "dp_size=%s requires a distinct KV_CACHE_EVENT_HOST_IP_PORT for every DP replica; "
            "using %s value %s for dp_rank=%s. Sharing this value lets KVCM snapshots "
            "overwrite another replica's host state.",
            parallelism_config.dp_size,
            "auto-derived" if derived else "explicit",
            kv_cache_config.kv_cache_event_host_ip_port,
            parallelism_config.dp_rank,
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
        model_type=py_env_configs.model_args.model_type,
    )

    # Set local ip if not already set (e.g. for world_info / distributed_server)
    if not py_env_configs.server_config.ip:
        py_env_configs.server_config.ip = socket.gethostbyname(socket.gethostname())

    # Frontend wheels omit model implementations. Model-specific settings must
    # also leave unrelated models unaffected when launch environments are shared.
    if (
        py_env_configs.role_config.role_type == RoleType.FRONTEND
        or py_env_configs.model_args.model_type != "deepseek_v4"
    ):
        return

    # Exact-SM120 DSv4 workers only accept the validated deployment profiles.
    # Fail before backend construction/capture rather than letting an
    # unsupported auto configuration die inside the runtime.
    _validate_sm120_dsv4_worker_profile(py_env_configs)
