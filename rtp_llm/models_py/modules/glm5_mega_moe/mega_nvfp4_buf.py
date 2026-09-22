"""Symmetric-memory buffer cache for DeepGEMM NVFP4xNVFP4 MegaMoE."""

from __future__ import annotations

import logging

import torch

_MEGA_NVFP4_BUF_CACHE: dict = {}


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


def get_or_create_mega_buf_nvfp4(
    group,
    num_experts: int,
    num_max_tokens_per_rank: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
    activation: str = "swiglu",
):
    """Collectively create or reuse the NVFP4 MegaMoE symmetric buffer."""
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
        "[MegaMoE NVFP4] allocated symm buffer: group_size=%d "
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
            "NVFP4 MegaMoE requires SM100/SM103"
        )
    return None


def mega_moe_nvfp4_available() -> bool:
    return _mega_moe_nvfp4_unavailable_reason() is None
