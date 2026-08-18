"""Independent buffers for Kimi K3 FP8xFP4 MegaMoE with fused SE."""

from __future__ import annotations

import logging

import torch

_MEGA_SE_BUF_CACHE: dict[tuple, object] = {}
_MEGA_SE_ROUTED_OUTPUT_CACHE: dict[tuple, torch.Tensor] = {}
_MEGA_SE_SHARED_INPUT_CACHE: dict[tuple, torch.Tensor] = {}
_MEGA_SE_SHARED_OUTPUT_CACHE: dict[tuple, torch.Tensor] = {}


def get_or_create_kimi_k3_mega_moe_se_buf(
    *,
    group,
    num_experts: int,
    num_max_tokens_per_rank: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
    shared_intermediate_hidden: int,
    activation: str,
):
    """Create a buffer whose cache cannot alias routed-only MegaMoE."""

    if shared_intermediate_hidden <= 0:
        raise ValueError(
            "Kimi K3 mega_moe_se requires shared_intermediate_hidden > 0, got "
            f"{shared_intermediate_hidden}"
        )
    key = (
        id(group),
        num_experts,
        num_max_tokens_per_rank,
        num_topk,
        hidden,
        intermediate_hidden,
        shared_intermediate_hidden,
        activation,
    )
    cached = _MEGA_SE_BUF_CACHE.get(key)
    if cached is not None:
        return cached

    import deep_gemm

    cached = deep_gemm.get_symm_buffer_for_mega_moe(
        group=group,
        num_experts=num_experts,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        num_topk=num_topk,
        hidden=hidden,
        intermediate_hidden=intermediate_hidden,
        mma_type="fp8xfp4",
        activation=activation,
        shared_intermediate_hidden=shared_intermediate_hidden,
    )
    workspace = getattr(cached, "shared_l2_acts", None)
    expected_workspace = (
        int(cached.num_max_tokens_per_rank),
        shared_intermediate_hidden,
    )
    if workspace is None or tuple(workspace.shape) != expected_workspace:
        raise RuntimeError(
            "DeepGEMM did not allocate the Kimi K3 fused-SE BF16 workspace: "
            f"got={None if workspace is None else tuple(workspace.shape)} "
            f"expected={expected_workspace}"
        )
    _MEGA_SE_BUF_CACHE[key] = cached
    logging.info(
        "[KimiK3 DeepGEMM MegaMoE SE] allocated buffer: experts=%d "
        "max_tokens_per_rank=%d aligned_capacity=%d topk=%d hidden=%d "
        "intermediate=%d shared_intermediate=%d shared_workspace=%.3f MiB",
        num_experts,
        num_max_tokens_per_rank,
        int(cached.num_max_tokens_per_rank),
        num_topk,
        hidden,
        intermediate_hidden,
        shared_intermediate_hidden,
        workspace.numel() * workspace.element_size() / (1024**2),
    )
    return cached


def _get_or_create_storage(
    cache: dict[tuple, torch.Tensor],
    capacity: int,
    hidden: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    key = (device, hidden, dtype)
    cached = cache.get(key)
    if cached is not None and int(cached.size(0)) >= capacity:
        return cached
    cached = torch.empty(
        (max(capacity, 1), hidden),
        dtype=dtype,
        device=device,
    )
    cache[key] = cached
    return cached


def get_or_create_kimi_k3_mega_moe_se_storages(
    *,
    capacity: int,
    routed_hidden: int,
    shared_hidden: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return routed output and caller-owned shared input/output storages."""

    routed_y = _get_or_create_storage(
        _MEGA_SE_ROUTED_OUTPUT_CACHE,
        capacity,
        routed_hidden,
        torch.bfloat16,
        device,
    )
    shared_x = _get_or_create_storage(
        _MEGA_SE_SHARED_INPUT_CACHE,
        capacity,
        shared_hidden,
        torch.bfloat16,
        device,
    )
    shared_y = _get_or_create_storage(
        _MEGA_SE_SHARED_OUTPUT_CACHE,
        capacity,
        shared_hidden,
        torch.bfloat16,
        device,
    )
    return routed_y, shared_x, shared_y


__all__ = [
    "get_or_create_kimi_k3_mega_moe_se_buf",
    "get_or_create_kimi_k3_mega_moe_se_storages",
]
