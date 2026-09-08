"""Exact PPU MoE gather with LocalLoop-compatible accumulation order.

The fused kernel accumulates a token's routes in stable ``(expert_id,
original_slot)`` order -- the expert-major order used by the eager LocalLoop
implementation -- reading each ``BF16 down`` row once and forming the
``down * FP32 route_weight`` product in FP32 registers.  This is the layout
SGLang reaches with ``_fwd_kernel_ep_gather``.

The historical two-kernel path materialized every product into an FP32
``[routes, dim]`` workspace before reducing it.  It is retained behind
``DSV4_MOE_GATHER_FUSED=0`` as the rollback arm: the workspace store/load is
lossless FP32, so both paths agree to FP32 rounding.

Both paths are asynchronous on the current stream.  This leaf performs no host
readback and contains no Python loop over experts.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

_SUPPORTED_INDEX_DTYPES = (torch.int32, torch.int64)
_MAX_TOPK = 16
_FUSED_ENV = "DSV4_MOE_GATHER_FUSED"
# Measured on ZW-M890P at dim=4096, topk=6: the reduction is index-math
# bound, not bandwidth bound, so a wide BLOCK_D and a
# top-k-sized lane count matter far more than warp count.  8.46x over the
# workspace path at 8192 tokens, 7.08x at 2048.
_FUSED_BLOCK_D = 2048
_FUSED_NUM_WARPS = 2


@triton.jit
def _materialize_route_product_kernel(
    down_ptr,
    down_stride_m,
    route_weight_ptr,
    output_index_ptr,
    workspace_ptr,
    dim,
    TOPK: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    route = tl.program_id(0).to(tl.int64)
    offsets_d = tl.program_id(1).to(tl.int64) * BLOCK_D + tl.arange(0, BLOCK_D)
    grouped_row = tl.load(output_index_ptr + route).to(tl.int64)
    valid = (grouped_row >= 0) & (offsets_d < dim)
    down = tl.load(
        down_ptr + grouped_row * down_stride_m + offsets_d,
        mask=valid,
        other=0.0,
    ).to(tl.float32)
    route_weight = tl.load(route_weight_ptr + route).to(tl.float32)
    route_product = down * route_weight
    # The FP32 store is required: folding this product into the reduction
    # changes the result even when the reduction order itself is identical.
    tl.store(
        workspace_ptr + route * dim + offsets_d,
        route_product,
        mask=offsets_d < dim,
    )


@triton.jit
def _stable_expert_accumulate_kernel(
    workspace_ptr,
    expert_ids_ptr,
    output_index_ptr,
    output_ptr,
    dim,
    TOPK: tl.constexpr,
    MAX_TOPK: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    offsets_d = tl.program_id(1).to(tl.int64) * BLOCK_D + tl.arange(0, BLOCK_D)
    slots = tl.arange(0, MAX_TOPK)
    slot_mask = slots < TOPK
    route_offsets = token * TOPK + slots
    expert_ids = tl.load(expert_ids_ptr + route_offsets, mask=slot_mask, other=-1).to(
        tl.int64
    )
    grouped_rows = tl.load(
        output_index_ptr + route_offsets, mask=slot_mask, other=-1
    ).to(tl.int64)
    remaining = slot_mask & (expert_ids >= 0) & (grouped_rows >= 0)

    # Expert id is the primary key and the original top-k slot is the stable
    # tie-break.  Selection is fully unrolled by Triton for the constexpr top-k.
    stable_keys = expert_ids * (TOPK + 1) + slots.to(tl.int64)
    sentinel: tl.constexpr = 0x7FFFFFFFFFFFFFFF
    accumulated = tl.zeros([BLOCK_D], dtype=tl.float32)
    for _ in tl.static_range(0, TOPK):
        selected_key = tl.min(tl.where(remaining, stable_keys, sentinel), axis=0)
        selected = remaining & (stable_keys == selected_key)
        selected_slot = tl.sum(tl.where(selected, slots, 0), axis=0).to(tl.int64)
        selected_valid = selected_key != sentinel
        product = tl.load(
            workspace_ptr + (token * TOPK + selected_slot) * dim + offsets_d,
            mask=selected_valid & (offsets_d < dim),
            other=0.0,
        ).to(tl.float32)
        accumulated += product
        remaining = remaining & ~selected

    tl.store(
        output_ptr + token * dim + offsets_d,
        accumulated,
        mask=offsets_d < dim,
    )


@triton.jit
def _fused_route_gather_kernel(
    down_ptr,
    down_stride_m,
    route_weight_ptr,
    expert_ids_ptr,
    output_index_ptr,
    output_ptr,
    dim,
    TOPK: tl.constexpr,
    MAX_TOPK: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Reduce one token's routes without materializing the FP32 products."""

    token = tl.program_id(0).to(tl.int64)
    offsets_d = tl.program_id(1).to(tl.int64) * BLOCK_D + tl.arange(0, BLOCK_D)
    dim_mask = offsets_d < dim
    slots = tl.arange(0, MAX_TOPK)
    slot_mask = slots < TOPK
    route_offsets = token * TOPK + slots
    expert_ids = tl.load(expert_ids_ptr + route_offsets, mask=slot_mask, other=-1).to(
        tl.int64
    )
    grouped_rows = tl.load(
        output_index_ptr + route_offsets, mask=slot_mask, other=-1
    ).to(tl.int64)
    remaining = slot_mask & (expert_ids >= 0) & (grouped_rows >= 0)

    # Identical key construction and selection order to the two-kernel path;
    # only the FP32 workspace round trip is removed.
    stable_keys = expert_ids * (TOPK + 1) + slots.to(tl.int64)
    sentinel: tl.constexpr = 0x7FFFFFFFFFFFFFFF
    accumulated = tl.zeros([BLOCK_D], dtype=tl.float32)
    for _ in tl.static_range(0, TOPK):
        selected_key = tl.min(tl.where(remaining, stable_keys, sentinel), axis=0)
        selected = remaining & (stable_keys == selected_key)
        selected_slot = tl.sum(tl.where(selected, slots, 0), axis=0).to(tl.int64)
        selected_row = tl.sum(tl.where(selected, grouped_rows, 0), axis=0).to(tl.int64)
        selected_valid = selected_key != sentinel
        down = tl.load(
            down_ptr + selected_row * down_stride_m + offsets_d,
            mask=selected_valid & dim_mask,
            other=0.0,
        ).to(tl.float32)
        # A scalar load cannot be masked, so an exhausted slot reads route 0 and
        # is zeroed afterwards; ``down`` is already zero in that case.
        safe_slot = tl.where(selected_valid, selected_slot, 0)
        route_weight = tl.load(route_weight_ptr + token * TOPK + safe_slot).to(
            tl.float32
        )
        route_product = down * route_weight
        accumulated += route_product
        remaining = remaining & ~selected

    tl.store(
        output_ptr + token * dim + offsets_d,
        accumulated,
        mask=dim_mask,
    )


def _fused_gather_enabled() -> bool:
    """Fused reduction is the default; ``0`` selects the workspace rollback."""

    requested = os.environ.get(_FUSED_ENV, "").strip().lower()
    if requested in ("", "1", "on", "true", "yes", "fused"):
        return True
    if requested in ("0", "off", "false", "no", "workspace"):
        return False
    raise ValueError(
        f"invalid {_FUSED_ENV}={requested!r}; expected 1 (fused) or 0 (workspace)"
    )


def _fused_launch_config(dim: int, topk: int) -> tuple[int, int]:
    """Return ``(BLOCK_D, MAX_TOPK)`` for the fused reduction.

    ``MAX_TOPK`` only pads the selection lanes to a power of two, so keeping it
    at the real top-k instead of the ``_MAX_TOPK`` ceiling removes wasted lanes.
    """

    block_d = min(_FUSED_BLOCK_D, triton.next_power_of_2(dim))
    return block_d, triton.next_power_of_2(topk)


def _require_tensor(name: str, value: torch.Tensor) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")


def gather_local_loop_compatible(
    down: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    output_index: torch.Tensor,
    output: torch.Tensor,
    *,
    fused: bool | None = None,
) -> None:
    """Gather grouped BF16 expert rows into a caller-owned FP32 output.

    ``output_index[token, slot]`` maps an original route to its row in
    expert-grouped ``down``; ``-1`` marks a route that must be ignored.  The
    function is deliberately fail-closed on shape, dtype, layout, and device
    mismatches.  Device values are trusted from the upstream nopad compactor so
    validation never synchronizes the current stream.
    """

    tensors = {
        "down": down,
        "topk_ids": topk_ids,
        "topk_weights": topk_weights,
        "output_index": output_index,
        "output": output,
    }
    for name, tensor in tensors.items():
        _require_tensor(name, tensor)

    if down.ndim != 2 or down.dtype != torch.bfloat16:
        raise TypeError("down must be a BF16 rank-2 tensor [routes, dim]")
    if not down.is_contiguous():
        raise ValueError("down must be contiguous")
    if topk_ids.ndim != 2 or topk_ids.dtype not in _SUPPORTED_INDEX_DTYPES:
        raise TypeError("topk_ids must be a signed int32/int64 rank-2 tensor")
    if topk_weights.dtype != torch.float32 or topk_weights.shape != topk_ids.shape:
        raise TypeError("topk_weights must be FP32 with the same shape as topk_ids")
    if (
        output_index.dtype not in _SUPPORTED_INDEX_DTYPES
        or output_index.shape != topk_ids.shape
    ):
        raise TypeError(
            "output_index must be int32/int64 with the same shape as topk_ids"
        )
    if (
        not topk_ids.is_contiguous()
        or not topk_weights.is_contiguous()
        or not output_index.is_contiguous()
    ):
        raise ValueError("route ids, weights, and output_index must be contiguous")

    tokens, topk = topk_ids.shape
    if tokens <= 0 or not 1 <= topk <= _MAX_TOPK:
        raise ValueError(f"route shape must be [tokens>0, 1<=topk<={_MAX_TOPK}]")
    routes = tokens * topk
    if down.shape[0] != routes or down.shape[1] <= 0:
        raise ValueError("down must have exactly tokens * topk rows and positive dim")
    if output.dtype != torch.float32 or output.shape != (tokens, down.shape[1]):
        raise TypeError("output must be FP32 [tokens, dim]")
    if not output.is_contiguous():
        raise ValueError("output must be contiguous")
    if not down.is_cuda or any(
        tensor.device != down.device for tensor in tensors.values()
    ):
        raise ValueError(
            "all tensors must be CUDA-compatible PPU tensors on one device"
        )

    output_pointer = output.untyped_storage().data_ptr()
    if any(
        tensor.untyped_storage().data_ptr() == output_pointer
        for tensor in (down, topk_ids, topk_weights, output_index)
    ):
        raise ValueError("output must not alias an input tensor")

    dim = down.shape[1]
    block_d = 256
    gather_grid = (tokens, triton.cdiv(dim, block_d))
    use_fused = _fused_gather_enabled() if fused is None else fused
    if use_fused:
        fused_block_d, fused_max_topk = _fused_launch_config(dim, topk)
        _fused_route_gather_kernel[(tokens, triton.cdiv(dim, fused_block_d))](
            down,
            down.stride(0),
            topk_weights,
            topk_ids,
            output_index,
            output,
            dim,
            TOPK=topk,
            MAX_TOPK=fused_max_topk,
            BLOCK_D=fused_block_d,
            num_warps=_FUSED_NUM_WARPS,
        )
        return

    workspace = torch.empty((routes, dim), dtype=torch.float32, device=down.device)
    grid = (routes, triton.cdiv(dim, block_d))
    _materialize_route_product_kernel[grid](
        down,
        down.stride(0),
        topk_weights,
        output_index,
        workspace,
        dim,
        TOPK=topk,
        BLOCK_D=block_d,
        num_warps=4,
    )
    _stable_expert_accumulate_kernel[gather_grid](
        workspace,
        topk_ids,
        output_index,
        output,
        dim,
        TOPK=topk,
        MAX_TOPK=_MAX_TOPK,
        BLOCK_D=block_d,
        num_warps=4,
    )


__all__ = ["gather_local_loop_compatible"]
