"""Dedicated symmetric-buffer cache and capability gates for Mega MoE SE.

DeepGEMM 2.6 exposes fused shared-expert execution through optional arguments
on ``fp8_fp4_mega_moe``. Its symmetric-buffer layout depends on the configured
shared-expert count and is not compatible with the routed-only buffer cached by
:mod:`mega_buf`.
"""

from __future__ import annotations

import inspect
import logging

import torch

from .buffer import _mega_moe_unavailable_reason

_MEGA_SE_BUF_CACHE: dict = {}
_MEGA_SE_OUTPUT_CACHE: dict = {}

_MMA_TYPE = "fp8xfp4"


def estimate_mega_moe_se_symm_buffer_bytes(
    group_size: int,
    num_experts: int,
    num_max_tokens_per_rank: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
    num_shared_experts: int,
    activation: str = "swiglu",
) -> int | None:
    """Best-effort estimate using the DeepGEMM 2.6 private API."""

    try:
        import deep_gemm

        return int(
            deep_gemm._C.get_symm_buffer_size_for_mega_moe(
                group_size,
                num_experts,
                num_max_tokens_per_rank,
                num_topk,
                hidden,
                intermediate_hidden,
                _MMA_TYPE,
                activation,
                num_shared_experts,
            )[0]
        )
    except Exception:
        return None


def _get_or_create_mega_se_buf(
    group,
    num_experts: int,
    num_max_tokens_per_rank: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
    num_shared_experts: int,
    activation: str,
):
    """Collectively allocate/cache the expanded Mega-SE symmetric buffer."""

    import deep_gemm

    key = (
        id(group),
        num_experts,
        num_max_tokens_per_rank,
        num_topk,
        hidden,
        intermediate_hidden,
        num_shared_experts,
        _MMA_TYPE,
        activation,
    )
    buf = _MEGA_SE_BUF_CACHE.get(key)
    if buf is not None:
        return buf

    try:
        group_size = int(group.size())
    except Exception:
        group_size = 0
    estimated_bytes = (
        estimate_mega_moe_se_symm_buffer_bytes(
            group_size=group_size,
            num_experts=num_experts,
            num_max_tokens_per_rank=num_max_tokens_per_rank,
            num_topk=num_topk,
            hidden=hidden,
            intermediate_hidden=intermediate_hidden,
            num_shared_experts=num_shared_experts,
            activation=activation,
        )
        if group_size > 0
        else None
    )
    buf = deep_gemm.get_symm_buffer_for_mega_moe(
        group=group,
        num_experts=num_experts,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        num_topk=num_topk,
        hidden=hidden,
        intermediate_hidden=intermediate_hidden,
        num_shared_experts=num_shared_experts,
        mma_type=_MMA_TYPE,
        activation=activation,
    )
    shared_sf = getattr(buf, "shared_l1_acts_sf", None)
    if shared_sf is None:
        raise RuntimeError(
            "DeepGEMM returned a Mega-SE buffer without shared_l1_acts_sf"
        )

    actual_bytes = None
    try:
        actual_bytes = int(buf.buffer.numel() * buf.buffer.element_size())
    except Exception:
        pass
    details = (
        group_size,
        num_experts,
        num_max_tokens_per_rank,
        num_topk,
        hidden,
        intermediate_hidden,
    )
    if actual_bytes is not None and estimated_bytes is not None:
        logging.info(
            "[MegaMoESE] allocated symm buffer: group_size=%d "
            "num_experts=%d max_tokens_per_rank=%d topk=%d hidden=%d "
            "intermediate=%d shared=%d actual=%.3f GiB estimated=%.3f GiB",
            *details,
            num_shared_experts,
            actual_bytes / (1024**3),
            estimated_bytes / (1024**3),
        )
    elif actual_bytes is not None:
        logging.info(
            "[MegaMoESE] allocated symm buffer: group_size=%d "
            "num_experts=%d max_tokens_per_rank=%d topk=%d hidden=%d "
            "intermediate=%d shared=%d actual=%.3f GiB",
            *details,
            num_shared_experts,
            actual_bytes / (1024**3),
        )
    elif estimated_bytes is not None:
        logging.info(
            "[MegaMoESE] allocated symm buffer: group_size=%d "
            "num_experts=%d max_tokens_per_rank=%d topk=%d hidden=%d "
            "intermediate=%d shared=%d actual=unavailable estimated=%.3f GiB",
            *details,
            num_shared_experts,
            estimated_bytes / (1024**3),
        )

    _MEGA_SE_BUF_CACHE[key] = buf
    return buf


def _get_or_create_mega_se_output(capacity, hidden, dtype, device):
    """Return a stable BF16 output allocation dedicated to the SE path."""

    key = (device, hidden, dtype)
    cached = _MEGA_SE_OUTPUT_CACHE.get(key)
    if cached is not None and cached.size(0) >= capacity:
        return cached
    cached = torch.empty((max(capacity, 1), hidden), dtype=dtype, device=device)
    _MEGA_SE_OUTPUT_CACHE[key] = cached
    return cached


def _signature_has(callable_obj, required: tuple[str, ...]) -> str | None:
    try:
        parameters = inspect.signature(callable_obj).parameters
    except (TypeError, ValueError) as exc:
        return f"cannot inspect DeepGEMM API signature: {exc}"
    missing = [name for name in required if name not in parameters]
    if missing:
        return f"DeepGEMM API is missing parameters: {missing}"
    return None


def _mega_moe_se_unavailable_reason() -> str | None:
    """Return ``None`` only when the installed Mega API supports fused SE."""

    base = _mega_moe_unavailable_reason()
    if base is not None:
        return base
    try:
        import deep_gemm

        reason = _signature_has(
            deep_gemm.fp8_fp4_mega_moe,
            ("shared_l1_weights", "shared_l2_weights", "shared_recipe"),
        )
        if reason is not None:
            return reason
        reason = _signature_has(
            deep_gemm.get_symm_buffer_for_mega_moe,
            ("num_shared_experts",),
        )
        if reason is not None:
            return reason
        if not hasattr(deep_gemm, "get_block_m_for_mega_moe"):
            return "deep_gemm.get_block_m_for_mega_moe is missing"
        if not hasattr(deep_gemm, "transform_weights_for_mega_moe"):
            return "deep_gemm.transform_weights_for_mega_moe is missing"
        if not hasattr(deep_gemm, "transform_sf_into_required_layout"):
            return "deep_gemm.transform_sf_into_required_layout is missing"
    except Exception as exc:
        return f"failed to inspect DeepGEMM Mega-SE capability: {exc}"
    return None


def _mega_moe_se_available() -> bool:
    return _mega_moe_se_unavailable_reason() is None
