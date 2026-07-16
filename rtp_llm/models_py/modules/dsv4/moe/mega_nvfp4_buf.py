"""Symmetric-memory buffers for DeepGEMM NVFP4 Mega MoE."""

from __future__ import annotations

import logging
import os

import torch

_USE_MEGA_MOE_NVFP4_ENV = "DSV4_USE_MEGA_MOE_NVFP4"
_MEGA_NVFP4_BUF_CACHE: dict = {}
_MEGA_NVFP4_OUTPUT_CACHE: dict = {}


def mega_moe_nvfp4_requested() -> bool:
    """Whether the NVFP4 Mega path was explicitly requested."""
    return os.environ.get(_USE_MEGA_MOE_NVFP4_ENV, "0") == "1"


def estimate_mega_moe_nvfp4_symm_buffer_bytes(
    group_size: int,
    num_experts: int,
    num_max_tokens_per_rank: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
    activation: str = "swiglu",
) -> int | None:
    try:
        import deep_gemm

        return int(
            deep_gemm._C.get_symm_buffer_size_for_mega_moe_nvfp4(
                group_size,
                num_experts,
                num_max_tokens_per_rank,
                num_topk,
                hidden,
                intermediate_hidden,
                activation,
            )[0]
        )
    except Exception:
        return None


def _get_or_create_mega_nvfp4_buf(
    group,
    num_experts: int,
    num_max_tokens_per_rank: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
    activation: str,
):
    import deep_gemm

    key = (
        id(group),
        num_experts,
        num_max_tokens_per_rank,
        num_topk,
        hidden,
        intermediate_hidden,
        activation,
    )
    buf = _MEGA_NVFP4_BUF_CACHE.get(key)
    if buf is not None:
        return buf

    try:
        group_size = int(group.size())
    except Exception:
        group_size = 0
    estimated_bytes = (
        estimate_mega_moe_nvfp4_symm_buffer_bytes(
            group_size,
            num_experts,
            num_max_tokens_per_rank,
            num_topk,
            hidden,
            intermediate_hidden,
            activation,
        )
        if group_size > 0
        else None
    )
    buf = deep_gemm.get_symm_buffer_for_mega_moe_nvfp4(
        group=group,
        num_experts=num_experts,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        num_topk=num_topk,
        hidden=hidden,
        intermediate_hidden=intermediate_hidden,
        activation=activation,
    )
    actual_bytes = None
    try:
        actual_bytes = int(buf.buffer.numel() * buf.buffer.element_size())
    except Exception:
        pass
    logging.info(
        "[DSV4 MegaMoE NVFP4] allocated symm buffer: group_size=%d "
        "num_experts=%d max_tokens_per_rank=%d topk=%d hidden=%d "
        "intermediate=%d actual=%s estimated=%s",
        group_size,
        num_experts,
        num_max_tokens_per_rank,
        num_topk,
        hidden,
        intermediate_hidden,
        (
            f"{actual_bytes / (1024**3):.3f} GiB"
            if actual_bytes is not None
            else "unavailable"
        ),
        (
            f"{estimated_bytes / (1024**3):.3f} GiB"
            if estimated_bytes is not None
            else "unavailable"
        ),
    )
    _MEGA_NVFP4_BUF_CACHE[key] = buf
    return buf


def _get_or_create_mega_nvfp4_output(
    capacity: int,
    hidden: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    key = (device, hidden, dtype)
    cached = _MEGA_NVFP4_OUTPUT_CACHE.get(key)
    if cached is not None and cached.size(0) >= capacity:
        return cached
    cached = torch.empty((max(capacity, 1), hidden), dtype=dtype, device=device)
    _MEGA_NVFP4_OUTPUT_CACHE[key] = cached
    return cached


def _mega_moe_nvfp4_unavailable_reason() -> str | None:
    try:
        import deep_gemm

        required = (
            "nvfp4_nvfp4_mega_moe",
            "get_symm_buffer_for_mega_moe_nvfp4",
            "transform_weights_for_mega_moe_nvfp4",
        )
        missing = [name for name in required if not hasattr(deep_gemm, name)]
        if missing:
            return f"deep_gemm is missing NVFP4 Mega APIs: {', '.join(missing)}"
    except Exception as exc:
        return f"failed to import deep_gemm: {exc}"

    try:
        import torch.distributed as dist

        if not dist.is_initialized():
            return "torch.distributed is not initialized"
        if dist.get_world_size() <= 1:
            return f"distributed world_size={dist.get_world_size()} is not > 1"
    except Exception as exc:
        return f"failed to query torch.distributed: {exc}"

    if not torch.cuda.is_available():
        return "CUDA is not available"
    capability = torch.cuda.get_device_capability()
    if capability[0] != 10:
        return (
            f"CUDA device capability sm{capability[0]}{capability[1]} is unsupported; "
            "NVFP4 Mega MoE requires SM100/SM103"
        )
    return None


def _mega_moe_nvfp4_available() -> bool:
    return _mega_moe_nvfp4_unavailable_reason() is None


def _mega_moe_nvfp4_enabled() -> bool:
    return mega_moe_nvfp4_requested() and _mega_moe_nvfp4_available()


def _mega_moe_nvfp4_disabled_or_unavailable_reason() -> str:
    if not mega_moe_nvfp4_requested():
        return f"{_USE_MEGA_MOE_NVFP4_ENV}=1 was not set"
    return (
        _mega_moe_nvfp4_unavailable_reason()
        or "unknown NVFP4 Mega MoE availability failure"
    )
