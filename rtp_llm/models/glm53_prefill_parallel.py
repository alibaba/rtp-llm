"""Resolve GLM MoE weight layout and Prefill parallelism before loading."""

from contextlib import nullcontext
from typing import Any

from rtp_llm.ops import RoleType


def use_dedicated_mla_cp(config, parallelism) -> bool:
    cp = parallelism.prefill_cp_config
    return bool(
        glm53_moe_enabled(config)
        and config.mega_moe_policy is not None
        and parallelism.role_type == RoleType.PREFILL
        and parallelism.tp_size > 1
        and cp.kv_cache_sharded
        and not cp.is_enabled()
        and config.glm53_hc_mult > 1
    )


def glm53_moe_enabled(config: Any) -> bool:
    return config.model_type in ("glm5_3_flash", "glm5_3_flash_mtp") or (
        config.model_type == "glm_5_mtp" and getattr(config, "is_glm53_mtp", False)
    )


def resolve_moe_policy(config, parallelism, moe_config):
    """Choose a checkpoint-compatible executor before any weights are sharded."""
    import torch
    import torch.distributed as dist

    from rtp_llm.config.mega_moe import (
        FP8_MEGA_MOE_STRATEGIES,
        FUSED_SHARED_STRATEGIES,
        MEGA_MOE_STRATEGIES,
        MegaMoeExecutionPolicy,
    )
    from rtp_llm.models_py.modules.factory.fused_moe.utils.mega_moe.buffer import (
        _mega_moe_unavailable_reason,
    )

    requested = moe_config.moe_strategy or "auto"
    if not glm53_moe_enabled(config) or not config.moe_layer_index:
        return None
    if requested != "auto" and requested not in MEGA_MOE_STRATEGIES:
        return None
    if requested == "auto" and (
        config.quant_config is None
        or config.quant_config.get_method() != "FP8_PER_BLOCK"
        or not config.quant_config.is_quanted()
        or config.quant_config.group_size() != 128
    ):
        return None

    shared = config.n_shared_experts > 0
    local = (
        shared
        and parallelism.role_type == RoleType.PREFILL
        and parallelism.get_attn_tp_size() > 1
        and not parallelism.prefill_cp_config.is_enabled()
    )
    full_shared = local or parallelism.get_ffn_tp_size() == 1
    candidate = requested
    if requested == "auto":
        candidate = (
            "mega_moe_fp8_se"
            if shared and full_shared and config.moe_k <= 31
            else "mega_moe_fp8"
        )
    error = None
    shared_supported = candidate in FUSED_SHARED_STRATEGIES
    if (
        parallelism.ep_size <= 1
        or parallelism.ep_size != parallelism.world_size
        or parallelism.ep_rank != parallelism.world_rank
        or config.eplb_config.phy_exp_num(config.expert_num) != config.expert_num
    ):
        error = "MegaMoE requires EP == WORLD with no redundant experts"
    elif candidate in FUSED_SHARED_STRATEGIES and (
        not shared or not full_shared or config.moe_k > 31
    ):
        error = "fused shared experts require full shared weights and at most 31 routed choices"
    elif candidate in FP8_MEGA_MOE_STRATEGIES:
        error = _mega_moe_unavailable_reason(
            kernel_name="fp8_fp8_mega_moe",
            allocator_name="get_symm_buffer_for_mega_moe_fp8",
            device_capabilities=((10, 0), (10, 3)),
        )
        if error is None:
            import deep_gemm

            if not callable(
                getattr(deep_gemm, "transform_weights_for_mega_moe_fp8", None)
            ):
                error = "DeepGEMM lacks the FP8 weight transform"
            if candidate == "mega_moe_fp8_se":
                shared_supported = callable(
                    getattr(deep_gemm, "get_block_m_for_mega_moe_fp8", None)
                )
                if not shared_supported and requested != "auto":
                    error = "DeepGEMM lacks the shared-expert FP8 API"
    # Every worker must make the same layout decision before the loader creates
    # rank-local tensors. The WORLD collective runs once per target/draft model.
    if (
        dist.is_initialized()
        and parallelism.ep_size == parallelism.world_size
        and parallelism.world_size > 1
    ):
        device = (
            torch.device("cuda", parallelism.local_rank)
            if dist.get_backend() == "nccl"
            else torch.device("cpu")
        )
        available = torch.tensor(
            [error is None, shared_supported], dtype=torch.int32, device=device
        )
        dist.all_reduce(available, op=dist.ReduceOp.MIN)
        routed_supported, shared_supported = available.tolist()
        if not routed_supported:
            error = error or "MegaMoE unavailable on another EP rank"
    if requested == "auto" and not shared_supported:
        candidate = "mega_moe_fp8"
    if error:
        if requested != "auto":
            raise ValueError(f"{requested}: {error}")
        return None
    moe_config.moe_strategy = candidate
    from rtp_llm.models_py.modules.dsv4.chunk_env import (
        chunked_moe_enabled,
        moe_chunk_tokens_from_env,
    )

    return MegaMoeExecutionPolicy(
        activation_clamp=(
            float(config.swiglu_limit) if config.swiglu_limit > 0 else None
        ),
        prefill_chunk_tokens=(
            moe_chunk_tokens_from_env() if chunked_moe_enabled() else 0
        ),
        preserve_fp32_scales=True,
        replicate_shared_weights=local,
    )


def glm_prefill_moe_chunk_plan(layers, tokens, device, *, token_sharded=False):
    from rtp_llm.models_py.modules.factory.fused_moe.utils.fp8_fp4.chunked_layer import (
        synchronized_moe_chunk_plan,
    )

    moes = [
        layer.mlp for layer in layers if getattr(layer.mlp, "requires_ep_sync", False)
    ]
    layouts = {(moe.routed_tp_size, moe.route_local_tokens) for moe in moes}
    if len(layouts) != 1:
        return nullcontext()
    tp_size, local_router = next(iter(layouts))
    if not token_sharded and tp_size > 1:
        tokens = (tokens + tp_size - 1) // tp_size
        if not local_router:
            tokens = max(tokens, 1)
    return synchronized_moe_chunk_plan(moes, tokens, device)


class _SequenceCPView:
    def __init__(self, config):
        self._config = config
        if not config.kv_cache_sharded:
            raise ValueError("GLM53 MLA CP requires PREFILL_CP_KV_CACHE_SHARDED=true")

    def is_enabled(self):
        return True

    def __getattr__(self, name):
        return getattr(self._config, name)


class MlaCPParallelismView:
    """Use full MLA heads while retaining the real TP group and page-RR geometry."""

    def __init__(self, parallelism):
        if parallelism.tp_size <= 1:
            raise ValueError("GLM53 MLA CP requires TP size > 1")
        if parallelism.prefill_cp_config.is_enabled():
            raise ValueError(
                "GLM53 dedicated MLA CP requires cp_rotate_method=DISABLED for the surrounding model"
            )
        self._parallelism = parallelism
        self.prefill_cp_config = _SequenceCPView(parallelism.prefill_cp_config)

    def get_attn_tp_size(self):
        return 1

    def get_attn_tp_rank(self):
        return 0

    def __getattr__(self, name):
        return getattr(self._parallelism, name)
