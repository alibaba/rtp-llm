"""Dequantize MXFP8 returns in FP32 and accumulate in deterministic peer order.

UE8M0 byte b represents the scale 2**(b - 127)."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

SCALE_BLOCK = 32
UE8M0_BIAS = 127.0


@triton.jit
def _mxfp8_peer_sum_kernel(
    payload_ptr,
    output_ptr,
    n_rows: tl.constexpr,
    hidden_size: tl.constexpr,
    payload_cols: tl.constexpr,
    scale_cols: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    block = tl.program_id(1)
    cols = block * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = cols < hidden_size
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for peer in tl.static_range(world_size):
        peer_row = peer * n_rows + row
        base = peer_row.to(tl.int64) * payload_cols
        q_u8 = tl.load(payload_ptr + base + cols, mask=mask, other=0)
        q = q_u8.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        scale_idx = cols // 32
        encoded_scale = tl.load(
            payload_ptr + base + hidden_size + scale_idx,
            mask=mask,
            other=127,
        )
        scale = tl.exp2(encoded_scale.to(tl.float32) - 127.0)
        acc += q * scale
    tl.store(output_ptr + row.to(tl.int64) * hidden_size + cols, acc, mask=mask)


def mxfp8_dequant_peer_sum(
    returned_payload: torch.Tensor,
    n_rows: int,
    hidden_size: int,
    world_size: int,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Sum peer-major [world_size * n_rows, hidden_size + hidden_size//32] byte rows."""
    if returned_payload.dtype != torch.uint8 or not returned_payload.is_contiguous():
        raise ValueError("returned_payload must be contiguous uint8")
    scale_cols = hidden_size // SCALE_BLOCK
    payload_cols = hidden_size + scale_cols
    if tuple(returned_payload.shape) != (world_size * n_rows, payload_cols):
        raise ValueError(
            "unexpected payload shape %s, expected %s"
            % (tuple(returned_payload.shape), (world_size * n_rows, payload_cols))
        )
    if n_rows == 0:
        return torch.empty(
            (0, hidden_size), dtype=out_dtype, device=returned_payload.device
        )
    output = torch.empty(
        (n_rows, hidden_size), dtype=out_dtype, device=returned_payload.device
    )
    block_d = 256
    _mxfp8_peer_sum_kernel[(n_rows, triton.cdiv(hidden_size, block_d))](
        returned_payload,
        output,
        n_rows=n_rows,
        hidden_size=hidden_size,
        payload_cols=payload_cols,
        scale_cols=scale_cols,
        world_size=world_size,
        BLOCK_D=block_d,
        num_warps=8,
    )
    return output


__all__ = [
    "SCALE_BLOCK",
    "mxfp8_dequant_peer_sum",
    "mxfp8_peer_sum_shared_epilogue",
]


@triton.jit
def _mxfp8_peer_sum_add_cast_kernel(
    payload_ptr,
    shared_ptr,
    output_ptr,
    n_rows: tl.constexpr,
    hidden_size: tl.constexpr,
    payload_cols: tl.constexpr,
    scale_cols: tl.constexpr,
    world_size: tl.constexpr,
    BLOCK_D: tl.constexpr,
    shared_stride_m,
    out_stride_m,
):
    """E8: _mxfp8_peer_sum_kernel and _add_cast_kernel in one launch.

    Op order is preserved EXACTLY: fp32 accumulation over peers in
    ascending peer order (same dequant: fp8e4nv bits x 2**(byte-127)),
    then + shared in fp32, then the out_dtype rounding happens in the
    store — bitwise identical to the two-kernel tail.
    """
    row = tl.program_id(0)
    block = tl.program_id(1)
    cols = block * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = cols < hidden_size
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for peer in tl.static_range(world_size):
        peer_row = peer * n_rows + row
        base = peer_row.to(tl.int64) * payload_cols
        q_u8 = tl.load(payload_ptr + base + cols, mask=mask, other=0)
        q = q_u8.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        scale_idx = cols // 32
        encoded_scale = tl.load(
            payload_ptr + base + hidden_size + scale_idx,
            mask=mask,
            other=127,
        )
        scale = tl.exp2(encoded_scale.to(tl.float32) - 127.0)
        acc += q * scale
    shared = tl.load(
        shared_ptr + row.to(tl.int64) * shared_stride_m + cols,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    tl.store(
        output_ptr + row.to(tl.int64) * out_stride_m + cols,
        acc + shared,
        mask=mask,
    )


def mxfp8_peer_sum_shared_epilogue(
    returned_payload: torch.Tensor,
    shared: torch.Tensor,
    n_rows: int,
    hidden_size: int,
    world_size: int,
    out_dtype: torch.dtype,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused EP combine tail: peer-sum dequant + shared add + cast, one kernel.

    Validation mirrors both parents (mxfp8_dequant_peer_sum payload checks;
    fused_moe_epilogue shape checks). Bitwise-equal to the sequential pair
    by construction (identical per-element op order and rounding points).
    """
    if returned_payload.dtype != torch.uint8 or not returned_payload.is_contiguous():
        raise ValueError("returned_payload must be contiguous uint8")
    scale_cols = hidden_size // SCALE_BLOCK
    payload_cols = hidden_size + scale_cols
    if tuple(returned_payload.shape) != (world_size * n_rows, payload_cols):
        raise ValueError(
            "unexpected payload shape %s, expected %s"
            % (tuple(returned_payload.shape), (world_size * n_rows, payload_cols))
        )
    if tuple(shared.shape) != (n_rows, hidden_size):
        raise ValueError(
            "shared shape %s does not match (%d, %d)"
            % (tuple(shared.shape), n_rows, hidden_size)
        )
    if shared.stride(1) != 1:
        raise ValueError("shared must have unit column stride")
    if out is None:
        out = torch.empty(
            (n_rows, hidden_size), dtype=out_dtype, device=returned_payload.device
        )
    elif (
        tuple(out.shape) != (n_rows, hidden_size)
        or out.dtype != out_dtype
        or out.device != returned_payload.device
        or out.stride(1) != 1
    ):
        raise ValueError(
            "out must match (n_rows, hidden), out_dtype, device, unit stride"
        )
    if n_rows == 0 or hidden_size == 0:
        return out
    block_d = 256
    _mxfp8_peer_sum_add_cast_kernel[(n_rows, triton.cdiv(hidden_size, block_d))](
        returned_payload,
        shared,
        out,
        n_rows=n_rows,
        hidden_size=hidden_size,
        payload_cols=payload_cols,
        scale_cols=scale_cols,
        world_size=world_size,
        BLOCK_D=block_d,
        shared_stride_m=shared.stride(0),
        out_stride_m=out.stride(0),
        num_warps=8,
    )
    return out
