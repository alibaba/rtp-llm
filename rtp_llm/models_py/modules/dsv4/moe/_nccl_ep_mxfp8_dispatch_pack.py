"""Pack quantized activation/scale bytes with per-peer masked router weights and IDs."""

from __future__ import annotations

import os

import torch

try:  # CPU tests use the semantic oracle below; no Triton execution is required.
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - import must remain CPU-safe
    triton = None
    tl = None

_PACKET_FLAG = "DSV4_NCCL_EP_MXFP8_DISPATCH_PACK"
_WEIGHT_BYTES = 4
_ID_BYTES = 4


def dispatch_pack_experimental_enabled() -> bool:
    """Strict default-off hook proposal for the parent-owned strategy seam."""
    return os.environ.get(_PACKET_FLAG, "0") == "1"


def _validate(q, scale, weights, ids, experts_per_rank: int, world: int):
    if not isinstance(world, int) or world != 4:
        raise ValueError("dispatch packet kernel currently requires integer world=4")
    if not isinstance(experts_per_rank, int) or experts_per_rank <= 0:
        raise ValueError("experts_per_rank must be a positive integer")
    if q.dtype != torch.uint8 or scale.dtype != torch.uint8:
        raise ValueError("q and scale must be uint8")
    if weights.dtype != torch.float32 or ids.dtype != torch.int64:
        raise ValueError("weights must be float32 and ids must be int64")
    if q.dim() != 2 or scale.dim() != 2 or weights.dim() != 2 or ids.dim() != 2:
        raise ValueError("dispatch packet inputs must be 2D")
    n, hidden = q.shape
    if scale.shape != (n, hidden // 32):
        raise ValueError("scale must be [N, D/32]")
    if weights.shape != ids.shape or weights.shape[0] != n:
        raise ValueError("weights/ids must be matching [N, K]")
    # The legacy typed FP32 router view starts after D + D/32 bytes.  D % 128
    # keeps that field 4-byte aligned and matches the MXFP8 dispatch ABI.
    if hidden % 128:
        raise ValueError("D must be divisible by 128")
    if weights.shape[1] <= 0:
        raise ValueError("topk K must be positive")
    if not (
        q.is_contiguous()
        and scale.is_contiguous()
        and weights.is_contiguous()
        and ids.is_contiguous()
    ):
        raise ValueError("dispatch packet inputs must be contiguous")
    if not (q.device == scale.device == weights.device == ids.device):
        raise ValueError("dispatch packet inputs must share a device")
    return n, hidden, weights.shape[1]


if triton is not None:

    @triton.jit
    def _dispatch_packet_write_kernel(
        q_ptr,
        scale_ptr,
        weight_ptr,
        id_ptr,
        out_ptr,
        n_rows,
        D: tl.constexpr,
        S: tl.constexpr,
        K: tl.constexpr,
        payload_cols: tl.constexpr,
        experts_per_rank: tl.constexpr,
        WORLD: tl.constexpr,
        BLOCK_BYTES: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        row_dst = tl.program_id(0).to(tl.int64)
        byte_block = tl.program_id(1)
        # Output rows are peer-major exactly like legacy `[dst, row, payload]`:
        # flattening is `dst * N + row`, not the tempting token-major mapping.
        dst = row_dst // n_rows
        row = row_dst % n_rows
        byte_offsets = byte_block * BLOCK_BYTES + tl.arange(0, BLOCK_BYTES)
        active_mask = (row < n_rows) & (byte_offsets < D)
        scale_mask = (row < n_rows) & (byte_offsets >= D) & (byte_offsets < D + S)
        out_base = row_dst * payload_cols
        q = tl.load(q_ptr + row * D + byte_offsets, mask=active_mask, other=0)
        tl.store(out_ptr + out_base + byte_offsets, q, mask=active_mask)
        scale_offsets = byte_offsets - D
        s = tl.load(scale_ptr + row * S + scale_offsets, mask=scale_mask, other=0)
        tl.store(out_ptr + out_base + byte_offsets, s, mask=scale_mask)

        # A single first-byte-block also emits the K router fields.  Encode raw
        # IEEE-754 / int32 bytes explicitly so NaN payloads are copied rather
        # than numerically normalized by a conversion or a typed output view.
        if byte_block == 0:
            k = tl.arange(0, BLOCK_K)
            k_mask = (row < n_rows) & (k < K)
            idx = tl.load(id_ptr + row * K + k, mask=k_mask, other=-1).to(tl.int64)
            weight = tl.load(weight_ptr + row * K + k, mask=k_mask, other=0.0).to(
                tl.float32
            )
            owner = (idx >= 0) & ((idx // experts_per_rank) == dst)
            selected_weight = tl.where(owner, weight, 0.0)
            selected_id = tl.where(owner, idx, -1).to(tl.int32)
            weight_bits = selected_weight.to(tl.int32, bitcast=True)
            for byte in tl.static_range(4):
                tl.store(
                    out_ptr + out_base + D + S + k * 4 + byte,
                    (weight_bits >> (byte * 8)).to(tl.uint8),
                    mask=k_mask,
                )
                tl.store(
                    out_ptr + out_base + D + S + K * 4 + k * 4 + byte,
                    (selected_id >> (byte * 8)).to(tl.uint8),
                    mask=k_mask,
                )


def pack_dispatch_packet_torch(
    q: torch.Tensor,
    scale: torch.Tensor,
    weights: torch.Tensor,
    ids: torch.Tensor,
    experts_per_rank: int,
    world: int = 4,
) -> torch.Tensor:
    """CPU byte oracle for packet layout, including NaN bits and int64-to-int32 narrowing."""
    n, d, k = _validate(q, scale, weights, ids, experts_per_rank, world)
    s = scale.shape[1]
    payload_cols = d + s + 8 * k
    out = torch.empty((world, n, payload_cols), dtype=torch.uint8, device=q.device)
    if n == 0:
        return out.view(0, payload_cols)
    out[:, :, :d].copy_(q.unsqueeze(0))
    out[:, :, d : d + s].copy_(scale.unsqueeze(0))
    destinations = torch.div(ids, experts_per_rank, rounding_mode="floor")
    for dst in range(world):
        owned = (ids >= 0) & (destinations == dst)
        # torch.where selects raw float lanes; assigning through a float view
        # preserves the selected IEEE bytes, including non-canonical NaNs.
        out[dst, :, d + s : d + s + 4 * k].view(torch.float32).copy_(
            torch.where(owned, weights, torch.zeros_like(weights))
        )
        out[dst, :, d + s + 4 * k :].view(torch.int32).copy_(
            torch.where(owned, ids, torch.full_like(ids, -1)).to(torch.int32)
        )
    return out.view(world * n, payload_cols)


def pack_dispatch_packet(
    q: torch.Tensor,
    scale: torch.Tensor,
    weights: torch.Tensor,
    ids: torch.Tensor,
    experts_per_rank: int,
    world: int = 4,
) -> torch.Tensor:
    """Launch the sidecar packet kernel.  This is not wired into the strategy."""
    n, d, k = _validate(q, scale, weights, ids, experts_per_rank, world)
    if triton is None:
        raise RuntimeError("Triton is unavailable for dispatch packet kernel")
    if not q.is_cuda:
        raise RuntimeError(
            "dispatch packet kernel requires CUDA; use only the CPU oracle in tests"
        )
    payload_cols = d + scale.shape[1] + 8 * k
    out = torch.empty((world * n, payload_cols), dtype=torch.uint8, device=q.device)
    if n == 0:
        return out
    block_bytes = 256
    block_k = triton.next_power_of_2(k)
    _dispatch_packet_write_kernel[
        (world * n, triton.cdiv(d + scale.shape[1], block_bytes))
    ](
        q,
        scale,
        weights,
        ids,
        out,
        n,
        d,
        scale.shape[1],
        k,
        payload_cols,
        experts_per_rank,
        WORLD=world,
        BLOCK_BYTES=block_bytes,
        BLOCK_K=block_k,
        num_warps=4,
    )
    return out


__all__ = [
    "dispatch_pack_experimental_enabled",
    "pack_dispatch_packet",
    "pack_dispatch_packet_torch",
]
