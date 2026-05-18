from __future__ import annotations

import logging
import os
import weakref
from typing import Optional

import torch

logger = logging.getLogger(__name__)

_KvRopeQuantEntry = tuple[
    weakref.ReferenceType[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]
_KV_ROPE_QUANT_REGISTRY: dict[int, _KvRopeQuantEntry] = {}
_KV_ROPE_QUANT_STORAGE_REGISTRY: dict[tuple, _KvRopeQuantEntry] = {}
_KV_ROPE_QUANT_ORDER: list[int] = []
_MAX_KV_ROPE_QUANT_TOKENS = 4096


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


def _debug_enabled() -> bool:
    return _env_flag("DSV4_KV_ROPE_QUANT_DEBUG")


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


def _remember_kv_rope_quant_token(
    token: torch.Tensor,
    q: Optional[torch.Tensor] = None,
    scale: Optional[torch.Tensor] = None,
) -> None:
    entry = (weakref.ref(token), q, scale)
    key = id(token)
    _KV_ROPE_QUANT_REGISTRY[key] = entry
    storage_key = _tensor_storage_key(token)
    if storage_key is not None:
        _KV_ROPE_QUANT_STORAGE_REGISTRY[storage_key] = entry
    _KV_ROPE_QUANT_ORDER.append(key)
    if _debug_enabled():
        logger.info(
            "DSV4 KV RoPE quant remember token key=%s storage_key=%s "
            "shape=%s q_shape=%s scale_shape=%s",
            key,
            storage_key,
            tuple(token.shape),
            None if q is None else tuple(q.shape),
            None if scale is None else tuple(scale.shape),
        )
    overflow = len(_KV_ROPE_QUANT_ORDER) - _MAX_KV_ROPE_QUANT_TOKENS
    if overflow <= 0:
        return
    for old_key in _KV_ROPE_QUANT_ORDER[:overflow]:
        old_entry = _KV_ROPE_QUANT_REGISTRY.pop(old_key, None)
        if old_entry is None:
            continue
        old_token = old_entry[0]()
        old_storage_key = _tensor_storage_key(old_token) if old_token is not None else None
        if old_storage_key is not None:
            _KV_ROPE_QUANT_STORAGE_REGISTRY.pop(old_storage_key, None)
    del _KV_ROPE_QUANT_ORDER[:overflow]


def remember_dsv4_kv_rope_quant_payload(
    y: torch.Tensor,
    q: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Record producer-side FP8 payload for a KV RoPE BF16 tensor.

    This mirrors the GraphFX producer-token provenance used by the RMSNorm
    quant passes while allowing a producer wrapper that already computed
    ``q`` and ``scale`` to register them without going through a rewritten FX
    token node.
    """
    _remember_kv_rope_quant_token(y, q, scale)
    return y


def _lookup_kv_rope_quant_token(
    y: torch.Tensor,
) -> Optional[tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]:
    storage_key = _tensor_storage_key(y)
    candidates = [
        _KV_ROPE_QUANT_REGISTRY.get(id(y)),
        _KV_ROPE_QUANT_STORAGE_REGISTRY.get(storage_key) if storage_key is not None else None,
    ]
    for entry in candidates:
        if entry is None:
            continue
        token_ref, q, scale = entry
        token = token_ref()
        if token is None:
            continue
        if token is y or _tensor_storage_key(token) == storage_key:
            return q, scale
    return None


def dsv4_kv_rope_quant_producer_token(
    y: torch.Tensor,
    q: torch.Tensor | None = None,
    scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """GraphFX provenance token for Path2 KV-compress/RoPE producers.

    Today this token is primarily a safe bridge for graph shape and provenance:
    it preserves the original BF16 tensor and records optional precomputed
    FP8/scale outputs.  A future producer-side dual-output CUDA kernel should
    pass those precomputed tensors here so the consumer rewrite can remove the
    standalone FP8 quant launch.
    """
    if q is None and scale is None and _env_flag("DSV4_KV_ROPE_QUANT_PRECOMPUTE_FP8"):
        from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8

        q, scale = sgl_per_token_group_quant_fp8(
            y,
            group_size=128,
            eps=1e-4,
            column_major_scales=True,
            scale_tma_aligned=True,
            scale_ue8m0=True,
        )
    _remember_kv_rope_quant_token(y, q, scale)
    return y


def dsv4_kv_rope_fp8_quant_from_provenance(
    y: torch.Tensor,
    *,
    fallback_y: torch.Tensor | None = None,
    group_size: int = 128,
    eps: float = 1e-4,
    column_major_scales: bool = True,
    scale_tma_aligned: bool = True,
    scale_ue8m0: bool = True,
    fuse_silu_and_mul: bool = False,
    masked_m: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    quant_input = fallback_y if fallback_y is not None else y
    provenance = _lookup_kv_rope_quant_token(y)
    if provenance is not None:
        q, scale = provenance
        if (
            q is not None
            and scale is not None
            and tuple(q.shape) == tuple(quant_input.shape)
            and q.device == quant_input.device
            and int(group_size) == 128
            and bool(column_major_scales)
            and bool(scale_tma_aligned)
            and bool(scale_ue8m0)
            and not bool(fuse_silu_and_mul)
            and masked_m is None
        ):
            return q, scale
        if _debug_enabled():
            logger.info(
                "DSV4 KV RoPE quant provenance has no reusable FP8 payload: "
                "input_shape=%s q_shape=%s scale_shape=%s",
                tuple(quant_input.shape),
                None if q is None else tuple(q.shape),
                None if scale is None else tuple(scale.shape),
            )
    elif _debug_enabled():
        logger.info(
            "DSV4 KV RoPE quant provenance miss: input_shape=%s registry=%d storage=%d",
            tuple(quant_input.shape),
            len(_KV_ROPE_QUANT_REGISTRY),
            len(_KV_ROPE_QUANT_STORAGE_REGISTRY),
        )
    if _env_flag("DSV4_KV_ROPE_QUANT_REQUIRE_PROVENANCE"):
        raise RuntimeError(
            "DSV4 KV RoPE quant consumer rewrite did not find valid producer provenance"
        )
    from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8

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
