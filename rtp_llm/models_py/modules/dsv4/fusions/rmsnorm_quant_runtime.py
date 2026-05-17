from __future__ import annotations

import logging
import math
import os
import weakref
from typing import Optional

import torch

from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8
from rtp_llm.models_py.kernels.cuda.fp8_kernel.fp8_kernel import (
    create_per_token_group_quant_fp8_output_scale,
    fp8_max,
    fp8_min,
)
from rtp_llm.models_py.kernels.cuda.fused_rmsnorm_fp8_quant import (
    is_supported as is_fused_rmsnorm_fp8_quant_supported,
    fused_rmsnorm_bf16_fp8_quant,
    fused_rmsnorm_fp8_quant,
)

logger = logging.getLogger(__name__)

_RMSNORM_TOKEN_REGISTRY: dict[
    int,
    tuple[
        weakref.ReferenceType[torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        float,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ],
] = {}
_RMSNORM_TOKEN_STORAGE_REGISTRY: dict[
    tuple,
    tuple[
        weakref.ReferenceType[torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        float,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ],
] = {}
_RMSNORM_TOKEN_DATA_REGISTRY: dict[
    tuple,
    tuple[
        weakref.ReferenceType[torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        float,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ],
] = {}
_RMSNORM_TOKEN_ORDER: list[int] = []
_MAX_RMSNORM_TOKENS = 4096


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


def _debug_enabled() -> bool:
    return _env_flag("DSV4_RMSNORM_QUANT_DEBUG")


def _debug_tensor(label: str, tensor: torch.Tensor | None) -> str:
    if tensor is None:
        return f"{label}=None"
    try:
        return (
            f"{label}:shape={tuple(int(v) for v in tensor.shape)} "
            f"stride={tuple(int(v) for v in tensor.stride())} "
            f"dtype={tensor.dtype} device={tensor.device} "
            f"data={int(tensor.data_ptr())}"
        )
    except Exception as exc:
        return f"{label}:unavailable({exc})"


def _tensor_storage_key(tensor: torch.Tensor) -> tuple | None:
    if tensor.is_meta:
        return None
    try:
        return (
            int(tensor.data_ptr()),
            tuple(int(v) for v in tensor.shape),
            tuple(int(v) for v in tensor.stride()),
            str(tensor.dtype),
            str(tensor.device),
        )
    except Exception:
        return None


def _tensor_data_key(tensor: torch.Tensor) -> tuple | None:
    if tensor.is_meta:
        return None
    try:
        return (
            int(tensor.data_ptr()),
            int(tensor.numel()),
            int(tensor.shape[-1]) if tensor.dim() > 0 else 1,
            str(tensor.dtype),
            str(tensor.device),
        )
    except Exception:
        return None


def _remember_rmsnorm_token(
    token: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    norm_eps: float,
    q: Optional[torch.Tensor] = None,
    scale: Optional[torch.Tensor] = None,
) -> None:
    key = id(token)
    entry = (weakref.ref(token), x, weight, float(norm_eps), q, scale)
    _RMSNORM_TOKEN_REGISTRY[key] = entry
    storage_key = _tensor_storage_key(token)
    if storage_key is not None:
        _RMSNORM_TOKEN_STORAGE_REGISTRY[storage_key] = entry
    data_key = _tensor_data_key(token)
    if data_key is not None:
        _RMSNORM_TOKEN_DATA_REGISTRY[data_key] = entry
    _RMSNORM_TOKEN_ORDER.append(key)
    if _debug_enabled():
        logger.info(
            "DSV4 RMSNorm quant remember token key=%s storage_key=%s data_key=%s %s %s %s",
            key,
            storage_key,
            data_key,
            _debug_tensor("token", token),
            _debug_tensor("x", x),
            _debug_tensor("q", q),
        )
    overflow = len(_RMSNORM_TOKEN_ORDER) - _MAX_RMSNORM_TOKENS
    if overflow <= 0:
        return
    for old_key in _RMSNORM_TOKEN_ORDER[:overflow]:
        old_entry = _RMSNORM_TOKEN_REGISTRY.pop(old_key, None)
        if old_entry is None:
            continue
        old_token = old_entry[0]()
        old_storage_key = _tensor_storage_key(old_token) if old_token is not None else None
        if old_storage_key is not None:
            _RMSNORM_TOKEN_STORAGE_REGISTRY.pop(old_storage_key, None)
        old_data_key = _tensor_data_key(old_token) if old_token is not None else None
        if old_data_key is not None:
            _RMSNORM_TOKEN_DATA_REGISTRY.pop(old_data_key, None)
    del _RMSNORM_TOKEN_ORDER[:overflow]


def _lookup_rmsnorm_token(
    y: torch.Tensor,
) -> Optional[
    tuple[torch.Tensor, torch.Tensor, float, Optional[torch.Tensor], Optional[torch.Tensor]]
]:
    y_id = id(y)
    y_storage_key = _tensor_storage_key(y)
    y_data_key = _tensor_data_key(y)
    candidates = [
        ("id", _RMSNORM_TOKEN_REGISTRY.get(y_id)),
        (
            "storage",
            _RMSNORM_TOKEN_STORAGE_REGISTRY.get(y_storage_key)
            if y_storage_key is not None
            else None,
        ),
        (
            "data",
            _RMSNORM_TOKEN_DATA_REGISTRY.get(y_data_key)
            if y_data_key is not None
            else None,
        ),
    ]
    for lookup_kind, entry in candidates:
        if entry is None:
            continue
        token_ref, x, weight, norm_eps, q, scale = entry
        token = token_ref()
        if token is None:
            _RMSNORM_TOKEN_REGISTRY.pop(y_id, None)
            if lookup_kind == "storage" and y_storage_key is not None:
                _RMSNORM_TOKEN_STORAGE_REGISTRY.pop(y_storage_key, None)
            if lookup_kind == "data" and y_data_key is not None:
                _RMSNORM_TOKEN_DATA_REGISTRY.pop(y_data_key, None)
            continue
        if (
            token is y
            or _tensor_storage_key(token) == y_storage_key
            or _tensor_data_key(token) == y_data_key
        ):
            return x, weight, norm_eps, q, scale
        if lookup_kind == "id":
            # Python may reuse tensor object ids across compiled segments.
            # A stale id hit must not hide a valid storage/data hit.
            _RMSNORM_TOKEN_REGISTRY.pop(y_id, None)
            continue
        if lookup_kind == "storage" and y_storage_key is not None:
            _RMSNORM_TOKEN_STORAGE_REGISTRY.pop(y_storage_key, None)
        if lookup_kind == "data" and y_data_key is not None:
            _RMSNORM_TOKEN_DATA_REGISTRY.pop(y_data_key, None)
    return None


def _ceil_align(value: int, align: int) -> int:
    return ((value + align - 1) // align) * align


def _product(values: tuple[int, ...]) -> int:
    return math.prod(values) if values else 1


def _view_precomputed_scale_like_quant_input(
    scale: torch.Tensor,
    *,
    flat_m: int,
    quant_shape: tuple[int, ...],
    group_size: int,
    scale_ue8m0: bool,
) -> torch.Tensor | None:
    if len(quant_shape) < 2:
        return None
    m_shape = tuple(int(v) for v in quant_shape[:-1])
    k = int(quant_shape[-1])
    if _product(m_shape) != int(flat_m) or k % int(group_size) != 0:
        return None

    if scale_ue8m0:
        k_groups = k // int(group_size)
        packed_k = _ceil_align(k_groups, 4) // 4
        expected_shape = m_shape + (packed_k,)
        if scale.numel() != _product(expected_shape):
            return None
        if len(m_shape) == 1:
            return scale.as_strided(expected_shape, (1, _ceil_align(flat_m, 4)))
        strides = []
        suffix = 1
        for dim in reversed(m_shape[1:]):
            strides.append(suffix)
            suffix *= dim
        strides.append(suffix)
        m_strides = tuple(reversed(strides))
        return scale.as_strided(expected_shape, m_strides + (_ceil_align(flat_m, 4),))

    expected_shape = m_shape + (k // int(group_size),)
    if scale.numel() != _product(expected_shape):
        return None
    return scale.reshape(expected_shape)


def _view_precomputed_quant_like_input(
    q: torch.Tensor,
    scale: torch.Tensor,
    quant_input: torch.Tensor,
    *,
    group_size: int,
    scale_ue8m0: bool,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if tuple(q.shape) == tuple(quant_input.shape):
        return q, scale
    if (
        q.dim() == 2
        and quant_input.dim() >= 2
        and q.numel() == quant_input.numel()
        and int(q.shape[-1]) == int(quant_input.shape[-1])
    ):
        q_view = q.reshape(tuple(int(v) for v in quant_input.shape))
        scale_view = _view_precomputed_scale_like_quant_input(
            scale,
            flat_m=int(q.shape[0]),
            quant_shape=tuple(int(v) for v in quant_input.shape),
            group_size=int(group_size),
            scale_ue8m0=bool(scale_ue8m0),
        )
        if scale_view is not None:
            return q_view, scale_view
    return None


def dsv4_rmsnorm_quant_producer_token(
    x: torch.Tensor, weight: torch.Tensor, norm_eps: float
) -> torch.Tensor:
    """GraphFX-inserted producer for cross-graph RMSNorm -> quant fusion.

    This function replaces the original RMSNorm node in the FX producer graph.
    By default it preserves original behavior and records provenance.  With
    ``DSV4_RMSNORM_QUANT_PRECOMPUTE_FP8=1`` it runs the BF16+FP8 fused CUDA
    kernel, returns the BF16 RMSNorm output for normal consumers, and stores
    the FP8/scale outputs for the downstream quant consumer rewrite.
    """
    if _env_flag("DSV4_RMSNORM_QUANT_PRECOMPUTE_FP8"):
        y, q, scale = fused_rmsnorm_bf16_fp8_quant(
            x,
            weight,
            norm_eps=float(norm_eps),
            quant_eps=1.0e-4,
            group_size=128,
        )
        _remember_rmsnorm_token(y, x, weight, float(norm_eps), q, scale)
        return y

    from rtp_llm.models_py.modules.base.cuda.norm import rmsnorm

    y = rmsnorm(x, weight, float(norm_eps))
    _remember_rmsnorm_token(y, x, weight, float(norm_eps))
    return y


def dsv4_rmsnorm_quant_mutating_producer_token(
    output: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    norm_eps: float,
) -> None:
    """GraphFX-inserted producer for mutating RMSNorm callsites.

    Direct ``rtp_llm_ops.rmsnorm(output, x, weight, ...)`` callsites often
    cross a Dynamo graph boundary: one segment allocates ``output``, another
    mutates it, and a later segment views/quantizes it.  This helper keeps the
    same ``output`` tensor object for downstream consumers, but replaces the
    side-effect RMSNorm launch with the fused BF16+FP8 path and records
    provenance on ``output``.
    """
    if _env_flag("DSV4_RMSNORM_QUANT_PRECOMPUTE_FP8"):
        if not is_fused_rmsnorm_fp8_quant_supported(x, weight, 128):
            raise ValueError(
                "unsupported dsv4_rmsnorm_quant_mutating_producer_token input: "
                f"x_shape={tuple(x.shape)} x_dtype={x.dtype} x_stride={tuple(x.stride())} "
                f"output_shape={tuple(output.shape)} output_dtype={output.dtype} "
                f"output_stride={tuple(output.stride())} "
                f"weight_shape={tuple(weight.shape)} weight_dtype={weight.dtype}"
            )
        if tuple(output.shape) != tuple(x.shape) or not output.is_contiguous():
            raise ValueError(
                "mutating RMSNorm producer requires contiguous output with same shape as input: "
                f"x_shape={tuple(x.shape)} output_shape={tuple(output.shape)} "
                f"output_stride={tuple(output.stride())}"
            )
        output_q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
        output_s = create_per_token_group_quant_fp8_output_scale(
            x_shape=output_q.shape,
            device=x.device,
            group_size=128,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=True,
        )
        from rtp_llm.ops.compute_ops import rtp_llm_ops

        rtp_llm_ops.fused_rmsnorm_bf16_fp8_quant(
            x,
            weight,
            output,
            output_q,
            output_s,
            float(norm_eps),
            1.0e-4,
            fp8_min,
            fp8_max,
        )
        _remember_rmsnorm_token(output, x, weight, float(norm_eps), output_q, output_s)
        return None

    from rtp_llm.models_py.modules.base.cuda.stream import current_cuda_stream_id
    from rtp_llm.ops.compute_ops import rtp_llm_ops

    rtp_llm_ops.rmsnorm(output, x, weight, float(norm_eps), current_cuda_stream_id())
    _remember_rmsnorm_token(output, x, weight, float(norm_eps))
    return None


def dsv4_rmsnorm_quant_provenance_token(
    y: torch.Tensor, x: torch.Tensor, weight: torch.Tensor, norm_eps: float
) -> torch.Tensor:
    """Runtime provenance bridge for legacy RMSNorm -> quant producer graphs.

    New producer rewrites should use ``dsv4_rmsnorm_quant_producer_token`` so
    DCE can remove the original RMSNorm node. This helper remains for older
    graphs and tests that already materialized the RMSNorm output.
    """
    if _env_flag("DSV4_RMSNORM_QUANT_SKIP_RMSNORM"):
        token = torch.empty_like(y)
    else:
        token = y
    _remember_rmsnorm_token(token, x, weight, float(norm_eps))
    return token


def dsv4_fused_rmsnorm_fp8_quant_from_provenance(
    y: torch.Tensor,
    *,
    fallback_y: torch.Tensor | None = None,
    group_size: int = 128,
    eps: float = 1e-4,
    column_major_scales: bool = True,
    scale_tma_aligned: bool = True,
    scale_ue8m0: bool = True,
    fuse_silu_and_mul: bool = False,
    masked_m: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    provenance = _lookup_rmsnorm_token(y)
    quant_input = fallback_y if fallback_y is not None else y
    if provenance is not None:
        x, weight, norm_eps, q, scale = provenance
        same_logical_input = tuple(x.shape) == tuple(quant_input.shape) or (
            x.numel() == quant_input.numel()
            and x.dim() == 2
            and quant_input.dim() >= 2
            and int(x.shape[-1]) == int(quant_input.shape[-1])
        )
        if same_logical_input and x.device == quant_input.device:
            if (
                q is not None
                and scale is not None
                and int(group_size) == 128
                and float(eps) == 1.0e-4
                and bool(column_major_scales)
                and bool(scale_tma_aligned)
                and bool(scale_ue8m0)
                and not bool(fuse_silu_and_mul)
                and masked_m is None
            ):
                precomputed = _view_precomputed_quant_like_input(
                    q,
                    scale,
                    quant_input,
                    group_size=int(group_size),
                    scale_ue8m0=bool(scale_ue8m0),
                )
                if precomputed is not None:
                    return precomputed
                if _debug_enabled():
                    logger.info(
                        "DSV4 RMSNorm quant precomputed shape mismatch: %s %s %s %s group=%s",
                        _debug_tensor("q", q),
                        _debug_tensor("scale", scale),
                        _debug_tensor("y", y),
                        _debug_tensor("quant_input", quant_input),
                        group_size,
                    )
            return fused_rmsnorm_fp8_quant(
                x,
                weight,
                norm_eps=float(norm_eps),
                quant_eps=float(eps),
                group_size=int(group_size),
            )
        if _debug_enabled():
            logger.info(
                "DSV4 RMSNorm quant provenance rejected: same_logical=%s x_device=%s quant_device=%s %s %s %s",
                same_logical_input,
                x.device,
                quant_input.device,
                _debug_tensor("x", x),
                _debug_tensor("y", y),
                _debug_tensor("quant_input", quant_input),
            )
    elif _debug_enabled():
        logger.info(
            "DSV4 RMSNorm quant provenance miss: registry_ids=%d storage=%d data=%d %s %s y_storage_key=%s y_data_key=%s",
            len(_RMSNORM_TOKEN_REGISTRY),
            len(_RMSNORM_TOKEN_STORAGE_REGISTRY),
            len(_RMSNORM_TOKEN_DATA_REGISTRY),
            _debug_tensor("y", y),
            _debug_tensor("fallback_y", fallback_y),
            _tensor_storage_key(y),
            _tensor_data_key(y),
        )
    if _env_flag("DSV4_RMSNORM_QUANT_REQUIRE_PROVENANCE"):
        raise RuntimeError(
            "DSV4 RMSNorm+quant consumer rewrite did not find valid RMSNorm provenance"
        )
    return sgl_per_token_group_quant_fp8(
        quant_input,
        group_size=group_size,
        eps=eps,
        column_major_scales=column_major_scales,
        scale_tma_aligned=scale_tma_aligned,
        scale_ue8m0=scale_ue8m0,
        fuse_silu_and_mul=fuse_silu_and_mul,
        masked_m=masked_m,
    )
