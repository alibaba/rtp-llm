"""Independent symmetric-buffer caches for FP8xFP4 MegaMoE with SE."""

from __future__ import annotations

import logging
import os

import torch

_MEGA_SE_BUF_CACHE: dict[tuple, object] = {}
_MEGA_SE_OUTPUT_CACHE: dict[tuple, torch.Tensor] = {}
_MEGA_SE_CLONE_BUF_CACHE: dict[tuple, object] = {}


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
                "fp8xfp4",
                activation,
                num_shared_experts,
            )[0]
        )
    except Exception:
        return None


def _create_mega_moe_se_buf(
    *,
    group,
    num_experts: int,
    num_max_tokens_per_rank: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
    num_shared_experts: int,
    activation: str,
):
    import deep_gemm

    return deep_gemm.get_symm_buffer_for_mega_moe(
        group=group,
        num_experts=num_experts,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        num_topk=num_topk,
        hidden=hidden,
        intermediate_hidden=intermediate_hidden,
        num_shared_experts=num_shared_experts,
        mma_type="fp8xfp4",
        activation=activation,
    )


def get_or_create_mega_moe_se_buf(
    *,
    group,
    num_experts: int,
    num_max_tokens_per_rank: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
    num_shared_experts: int,
    activation: str = "swiglu",
):
    if num_shared_experts <= 0:
        raise ValueError(
            "mega_moe_se requires num_shared_experts > 0, got " f"{num_shared_experts}"
        )
    key = (
        id(group),
        num_experts,
        num_max_tokens_per_rank,
        num_topk,
        hidden,
        intermediate_hidden,
        num_shared_experts,
        activation,
    )
    cached = _MEGA_SE_BUF_CACHE.get(key)
    if cached is not None:
        return cached

    cached = _create_mega_moe_se_buf(
        group=group,
        num_experts=num_experts,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        num_topk=num_topk,
        hidden=hidden,
        intermediate_hidden=intermediate_hidden,
        num_shared_experts=num_shared_experts,
        activation=activation,
    )
    _MEGA_SE_BUF_CACHE[key] = cached
    try:
        actual_bytes = int(cached.buffer.numel() * cached.buffer.element_size())
        group_size = int(group.size())
        estimated = estimate_mega_moe_se_symm_buffer_bytes(
            group_size,
            num_experts,
            num_max_tokens_per_rank,
            num_topk,
            hidden,
            intermediate_hidden,
            num_shared_experts,
            activation,
        )
        estimate_text = (
            f" estimated={estimated / (1024**3):.3f} GiB" if estimated else ""
        )
        logging.info(
            "[GLM5 MegaMoE SE] allocated symm buffer: group_size=%d "
            "experts=%d max_tokens=%d topk=%d hidden=%d intermediate=%d "
            "shared_experts=%d actual=%.3f GiB%s",
            group_size,
            num_experts,
            num_max_tokens_per_rank,
            num_topk,
            hidden,
            intermediate_hidden,
            num_shared_experts,
            actual_bytes / (1024**3),
            estimate_text,
        )
    except Exception:
        pass
    return cached


def get_or_create_mega_moe_se_output(
    capacity: int,
    hidden: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    key = (device, hidden, dtype)
    cached = _MEGA_SE_OUTPUT_CACHE.get(key)
    if cached is not None and cached.size(0) >= capacity:
        return cached
    cached = torch.empty((max(capacity, 1), hidden), dtype=dtype, device=device)
    _MEGA_SE_OUTPUT_CACHE[key] = cached
    return cached


def get_or_create_mega_moe_se_clone_buf(src_buf, group, cfg, num_shared_experts: int):
    """Allocate clone-only mutable symmetric state for MTP CUDA Graph."""
    if src_buf is None or group is None:
        return src_buf
    key = (
        id(src_buf),
        id(group),
        cfg.n_routed_experts,
        cfg.max_tokens_per_rank,
        cfg.n_activated_experts,
        cfg.dim,
        cfg.moe_inter_dim,
        num_shared_experts,
    )
    cached = _MEGA_SE_CLONE_BUF_CACHE.get(key)
    if cached is not None:
        return cached
    cached = _create_mega_moe_se_buf(
        group=group,
        num_experts=cfg.n_routed_experts,
        num_max_tokens_per_rank=max(cfg.max_tokens_per_rank, 1),
        num_topk=cfg.n_activated_experts,
        hidden=cfg.dim,
        intermediate_hidden=cfg.moe_inter_dim,
        num_shared_experts=num_shared_experts,
        activation="swiglu",
    )
    _MEGA_SE_CLONE_BUF_CACHE[key] = cached
    logging.info(
        "[GLM5 MegaMoE SE] allocated CUDA graph clone symm buffer: "
        "layer=%d max_tokens_per_rank=%d shared_experts=%d",
        cfg.layer_id,
        cfg.max_tokens_per_rank,
        num_shared_experts,
    )
    return cached


def _mega_moe_se_unavailable_reason() -> str | None:
    try:
        import deep_gemm

        for symbol in (
            "fp8_fp4_mega_moe",
            "get_symm_buffer_for_mega_moe",
            "get_block_m_for_mega_moe",
            "transform_weights_for_mega_moe",
        ):
            if not hasattr(deep_gemm, symbol):
                return f"deep_gemm.{symbol} is missing"
    except Exception as error:
        return f"failed to import deep_gemm: {error}"
    try:
        import torch.distributed as dist

        if not dist.is_initialized():
            return "torch.distributed is not initialized"
        if dist.get_world_size() <= 1:
            return f"distributed world_size={dist.get_world_size()} is not > 1"
    except Exception as error:
        return f"failed to query torch.distributed: {error}"
    if not torch.cuda.is_available():
        return "CUDA is not available"
    capability = torch.cuda.get_device_capability()
    if capability[0] < 10:
        return f"CUDA device capability sm{capability[0]}{capability[1]} is below SM100"
    return None


def mega_moe_se_available() -> bool:
    return _mega_moe_se_unavailable_reason() is None


def mega_moe_se_enabled() -> bool:
    default = os.environ.get("GLM5_USE_MEGA_MOE", "1")
    if os.environ.get("GLM5_USE_MEGA_MOE_SE", default) == "0":
        return False
    return mega_moe_se_available()


__all__ = [
    "estimate_mega_moe_se_symm_buffer_bytes",
    "get_or_create_mega_moe_se_buf",
    "get_or_create_mega_moe_se_clone_buf",
    "get_or_create_mega_moe_se_output",
    "mega_moe_se_available",
    "mega_moe_se_enabled",
]
