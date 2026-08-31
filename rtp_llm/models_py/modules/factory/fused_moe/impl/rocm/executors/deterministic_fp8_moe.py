"""Opt-in deterministic ROCm FP8-per-channel MoE path for Qwen3.5 TP2.

The current AITER CK stage-2 kernel atomically scatters every TopK route into
the token output.  On TP2 this makes the final BF16 value depend on CTA arrival
order.  This module reuses the same per-token-FP8 CK GEMMs, but gives every
route a private output row and performs one fixed-order FP32 reduction after
stage2.
"""

import dataclasses
import functools
import importlib
import logging
import os
from typing import Optional

import aiter
import torch

from rtp_llm.models_py.triton_kernels.moe.fixed_order_route_reduce import (
    fixed_order_fp32_route_reduce,
    make_route_local_ids,
)

_LOGGER = logging.getLogger(__name__)
_LOGGED_MESSAGES: set[str] = set()

_ENABLE_ENV = "RTP_LLM_ROCM_FP8_MOE_DETERMINISTIC_REDUCE"
_MAX_TOKENS_ENV = "RTP_LLM_ROCM_FP8_MOE_DETERMINISTIC_MAX_TOKENS"
_GRAPH_ENVS = ("ENABLE_CUDA_GRAPH", "ENABLE_NATIVE_CUDA_GRAPH")

_SUPPORTED_EXPERTS = 256
_SUPPORTED_HIDDEN_SIZE = 2048
_SUPPORTED_INTER_SIZE = 256
_SUPPORTED_TOPK = 8
_DEFAULT_MAX_TOKENS = 128
_BLOCK_M = 32

_STAGE1_KERNEL = (
    "moe_ck2stages_gemm1_256x32x64x256_1x4_MulABScale_v1_"
    "Nswizzle0_Quant2_MulRoutedWeight0_silu_F8_F8_B16"
)
_STAGE2_KERNEL = (
    "moe_ck2stages_gemm2_256x32x64x256_1x4_MulABScaleExpertWeight_v1_"
    "Nswizzle0_Quant2_MulRoutedWeight1_F8_F8_B16"
)


def _log_once(level: int, key: str, message: str) -> None:
    if key in _LOGGED_MESSAGES:
        return
    _LOGGED_MESSAGES.add(key)
    _LOGGER.log(level, message)


def _enabled() -> bool:
    return os.environ.get(_ENABLE_ENV, "0") == "1"


def _validate_runtime_mode() -> None:
    enabled_graph_envs = [name for name in _GRAPH_ENVS if os.environ.get(name) == "1"]
    if enabled_graph_envs:
        raise RuntimeError(
            f"{_ENABLE_ENV}=1 requires CUDA/HIP graph mode to be disabled; "
            f"set {', '.join(enabled_graph_envs)}=0"
        )


def _max_tokens() -> int:
    value = int(os.environ.get(_MAX_TOKENS_ENV, str(_DEFAULT_MAX_TOKENS)))
    return max(1, value)


def _unsupported_reason(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    activation: aiter.ActivationType,
    expert_mask: Optional[torch.Tensor],
) -> Optional[str]:
    if torch.version.hip is None:
        return "ROCm is unavailable"
    if expert_mask is not None:
        return "expert-parallel masking is unsupported"
    if activation != aiter.ActivationType.Silu:
        return f"activation {activation} is unsupported"
    if hidden_states.dtype != torch.bfloat16:
        return f"hidden dtype {hidden_states.dtype} is unsupported"
    if hidden_states.dim() != 2 or hidden_states.shape[1] != _SUPPORTED_HIDDEN_SIZE:
        return f"hidden shape {tuple(hidden_states.shape)} is unsupported"
    if not 0 < hidden_states.shape[0] <= _max_tokens():
        return f"token count {hidden_states.shape[0]} exceeds {_max_tokens()}"
    if topk_ids.shape != (hidden_states.shape[0], _SUPPORTED_TOPK):
        return f"topk id shape {tuple(topk_ids.shape)} is unsupported"
    if topk_weights.shape != topk_ids.shape:
        return "topk weight shape does not match topk ids"
    if topk_ids.dtype != torch.int32 or topk_weights.dtype != torch.float32:
        return "topk ids/weights must be int32/float32"
    if w1.shape != (
        _SUPPORTED_EXPERTS,
        _SUPPORTED_INTER_SIZE * 2,
        _SUPPORTED_HIDDEN_SIZE,
    ):
        return f"w1 shape {tuple(w1.shape)} is unsupported"
    if w2.shape != (
        _SUPPORTED_EXPERTS,
        _SUPPORTED_HIDDEN_SIZE,
        _SUPPORTED_INTER_SIZE,
    ):
        return f"w2 shape {tuple(w2.shape)} is unsupported"
    if w1.dtype != torch.float8_e4m3fnuz or w2.dtype != torch.float8_e4m3fnuz:
        return f"weight dtypes {w1.dtype}/{w2.dtype} are unsupported"
    if w1_scale.dtype != torch.float32 or w2_scale.dtype != torch.float32:
        return "weight scales must use FP32 per-output-channel layout"
    if w1_scale.shape != (_SUPPORTED_EXPERTS, _SUPPORTED_INTER_SIZE * 2, 1):
        return f"w1 scale shape {tuple(w1_scale.shape)} is unsupported"
    if w2_scale.shape != (_SUPPORTED_EXPERTS, _SUPPORTED_HIDDEN_SIZE, 1):
        return f"w2 scale shape {tuple(w2_scale.shape)} is unsupported"
    if not getattr(w1, "is_shuffled", False) or not getattr(w2, "is_shuffled", False):
        return "weights are not marked as AITER-preshuffled"
    return None


def _route_local_stage2(
    ck_stage2,
    expected_topk: int,
    inter_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    output: torch.Tensor,
    topk: int,
    **kwargs,
) -> torch.Tensor:
    if topk != expected_topk:
        raise RuntimeError(f"unexpected TopK in deterministic stage2: {topk}")

    token_num = inter_states.shape[0]
    route_num = token_num * topk
    route_inter_states = inter_states.reshape(route_num, inter_states.shape[-1])
    route_token_ids = make_route_local_ids(sorted_token_ids, token_num, topk)

    # CK stage2 still uses AtomicAdd, so its private route rows must start at
    # zero.  With one route per row there are no conflicting atomic writers.
    route_output = torch.zeros(
        (route_num, output.shape[-1]), dtype=output.dtype, device=output.device
    )
    ck_stage2(
        route_inter_states,
        w1,
        w2,
        route_token_ids,
        sorted_expert_ids,
        num_valid_ids,
        route_output,
        1,
        **kwargs,
    )
    fixed_order_fp32_route_reduce(route_output, output, topk)
    return output


@functools.lru_cache(maxsize=4)
def _make_metadata_transform(output_dtype: torch.dtype, topk: int):
    fused_moe_module = importlib.import_module("aiter.fused_moe")

    def transform(metadata):
        use_non_temporal_load = metadata.use_non_temporal_load
        ck_stage2 = functools.partial(
            aiter.ck_moe_stage2_fwd,
            kernelName=_STAGE2_KERNEL,
            activation=aiter.ActivationType.Silu,
            quant_type=aiter.QuantType.per_Token,
            use_non_temporal_load=use_non_temporal_load,
        )
        route_local_stage2 = functools.partial(
            _route_local_stage2,
            ck_stage2,
            topk,
        )
        stage1 = functools.partial(
            fused_moe_module.ck_moe_stage1,
            kernelName=_STAGE1_KERNEL,
            activation=aiter.ActivationType.Silu,
            quant_type=aiter.QuantType.per_Token,
            dtype=output_dtype,
            splitk=0,
            use_non_temporal_load=use_non_temporal_load,
        )
        # get_2stage_cfgs is cached; never mutate its metadata object in place.
        return dataclasses.replace(
            metadata,
            stage1=stage1,
            stage2=route_local_stage2,
            block_m=_BLOCK_M,
            ksplit=0,
            run_1stage=False,
            has_bias=False,
            stage2_has_bias=False,
            flat=False,
            output_aux=False,
            prequant=True,
            skip_inter_quant=False,
            fuse_quant="",
            route_bucket="",
            expected_sorted_blocks=None,
            min_sorted_blocks=None,
            max_sorted_blocks=None,
        )

    return transform


def try_deterministic_fp8_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    activation: aiter.ActivationType,
    expert_mask: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    """Return deterministic output when enabled/supported, otherwise ``None``."""

    if not _enabled():
        return None
    _validate_runtime_mode()

    reason = _unsupported_reason(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        w1_scale,
        w2_scale,
        activation,
        expert_mask,
    )
    if reason is not None:
        _log_once(
            logging.WARNING,
            f"fallback:{reason}",
            f"{_ENABLE_ENV}=1 but deterministic ROCm FP8 MoE is falling back: {reason}",
        )
        return None

    _log_once(
        logging.INFO,
        "enabled",
        "Using route-local CK stage2 plus fixed-order FP32 reduction for ROCm "
        "FP8-per-channel MoE",
    )
    fused_moe_module = importlib.import_module("aiter.fused_moe")
    return fused_moe_module._fused_moe_impl(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weight=topk_weights,
        topk_ids=topk_ids,
        activation=activation.value,
        quant_type=aiter.QuantType.per_Token.value,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        block_size_M=-1,
        _metadata_transform=_make_metadata_transform(
            hidden_states.dtype, _SUPPORTED_TOPK
        ),
    )
