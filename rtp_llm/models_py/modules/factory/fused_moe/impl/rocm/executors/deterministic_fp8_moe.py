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
import importlib.metadata
import logging
import os
from typing import Optional

import aiter
import torch

from rtp_llm.models_py.kernel_tuning import (
    ROCM_FP8_MOE_DETERMINISTIC_REDUCE_ENV,
    is_rocm_fp8_moe_deterministic_reduce_enabled,
)
from rtp_llm.models_py.kernel_tuning.aiter import AITER_FMOE_SUPPORTED_VERSION
from rtp_llm.models_py.triton_kernels.moe.fixed_order_route_reduce import (
    fixed_order_fp32_route_reduce,
    make_route_local_ids,
)

_LOGGER = logging.getLogger(__name__)
_LOGGED_MESSAGES: set[str] = set()

_MAX_TOKENS_ENV = "RTP_LLM_ROCM_FP8_MOE_DETERMINISTIC_MAX_TOKENS"

_SUPPORTED_EXPERTS = 256
_SUPPORTED_HIDDEN_SIZE = 2048
_SUPPORTED_INTER_SIZE = 256
_SUPPORTED_TOPK = 8
_DEFAULT_MAX_TOKENS = 128
_BLOCK_M = 32
_SUPPORTED_GFX = "gfx942"
_SUPPORTED_CU_COUNT = 80

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


def _max_tokens() -> int:
    value = int(os.environ.get(_MAX_TOKENS_ENV, str(_DEFAULT_MAX_TOKENS)))
    return max(1, value)


@functools.lru_cache(maxsize=8)
def _runtime_unsupported_reason(device_name: str) -> Optional[str]:
    device = torch.device(device_name)
    properties = torch.cuda.get_device_properties(device)
    gfx = str(getattr(properties, "gcnArchName", "")).split(":", 1)[0]
    if gfx != _SUPPORTED_GFX:
        return f"GPU architecture {gfx or '<unknown>'} is unsupported"
    cu_count = int(getattr(properties, "multi_processor_count", 0))
    if cu_count != _SUPPORTED_CU_COUNT:
        return f"GPU CU count {cu_count} is unsupported"

    try:
        aiter_version = importlib.metadata.version("aiter")
    except importlib.metadata.PackageNotFoundError:
        return "AITER distribution metadata is unavailable"
    if aiter_version != AITER_FMOE_SUPPORTED_VERSION:
        return (
            f"AITER version {aiter_version} is unsupported; expected "
            f"{AITER_FMOE_SUPPORTED_VERSION}"
        )

    try:
        fused_moe_module = importlib.import_module("aiter.fused_moe")
    except ImportError as error:
        return f"AITER fused_moe module is unavailable: {error}"
    required_symbols = {
        "aiter.ck_moe_stage2_fwd": getattr(aiter, "ck_moe_stage2_fwd", None),
        "aiter.fused_moe.ck_moe_stage1": getattr(
            fused_moe_module, "ck_moe_stage1", None
        ),
        "aiter.fused_moe._fused_moe_impl": getattr(
            fused_moe_module, "_fused_moe_impl", None
        ),
    }
    missing_symbols = [
        name for name, symbol in required_symbols.items() if not callable(symbol)
    ]
    if missing_symbols:
        return f"required AITER symbols are unavailable: {missing_symbols}"
    return None


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
    tensors = (w1, w2, topk_weights, topk_ids, w1_scale, w2_scale)
    if any(tensor.device != hidden_states.device for tensor in tensors):
        return "input, routing, weight, and scale tensors must share one device"
    runtime_reason = _runtime_unsupported_reason(str(hidden_states.device))
    if runtime_reason is not None:
        return runtime_reason
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

    if not is_rocm_fp8_moe_deterministic_reduce_enabled():
        return None

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
            f"{ROCM_FP8_MOE_DETERMINISTIC_REDUCE_ENV}=1 but deterministic ROCm "
            f"FP8 MoE is falling back: {reason}",
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
