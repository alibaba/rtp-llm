"""Caller-buffered Q-only split/RoPE for head-sharded GLM5 CMP.

Positions must address cos_sin rows; the caller owns this device-only bounds
contract. The latent Q prefix is left untouched for absorbed_q_nope_bmm.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _split_q_rope_kernel(
    Projected,
    CosSin,
    Positions,
    Nope,
    Query,
    HEADS: tl.constexpr,
    IS_NEOX: tl.constexpr,
    HEAD_TILE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1) * HEAD_TILE + tl.arange(0, HEAD_TILE)
    token_head = row * HEADS + head
    column = tl.arange(0, 256)
    valid = (head[:, None] < HEADS) & (column[None, :] < 192)
    nope = tl.load(
        Projected + token_head[:, None] * 256 + column[None, :], valid, other=0
    )
    tl.store(Nope + token_head[:, None] * 192 + column[None, :], nope, valid)

    pair = tl.arange(0, 32)
    position = tl.load(Positions + row).to(tl.int64)
    cosine = tl.load(CosSin + position * 64 + pair)
    sine = tl.load(CosSin + position * 64 + 32 + pair)
    if IS_NEOX:
        low, high = pair, pair + 32
    else:
        low, high = 2 * pair, 2 * pair + 1
    mask = head[:, None] < HEADS
    x0 = tl.load(
        Projected + token_head[:, None] * 256 + 192 + low[None, :], mask, other=0
    ).to(tl.float32)
    x1 = tl.load(
        Projected + token_head[:, None] * 256 + 192 + high[None, :], mask, other=0
    ).to(tl.float32)
    # Match fused_qk_rope_cat_cache_mla's rounded multiply + explicit FMA.
    y0 = tl.extra.libdevice.fma_rn(x0, cosine[None, :], -x1 * sine[None, :])
    y1 = tl.extra.libdevice.fma_rn(x1, cosine[None, :], x0 * sine[None, :])
    tl.store(
        Query + token_head[:, None] * 576 + 512 + low[None, :], y0.to(tl.bfloat16), mask
    )
    tl.store(
        Query + token_head[:, None] * 576 + 512 + high[None, :],
        y1.to(tl.bfloat16),
        mask,
    )


def split_q_rope(
    projected: torch.Tensor,
    cos_sin: torch.Tensor,
    positions: torch.Tensor,
    *,
    q_nope_out: torch.Tensor,
    q_out: torch.Tensor,
    is_neox_style: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split BF16 [M,H*256] into NoPE [M,H,192] and rotated Q[...,512:].

    All tensors must be contiguous on one CUDA device. Output storage is
    caller-owned and must not overlap inputs or each other. No KV is read or
    written, and q_out[..., :512] is never initialized or modified here.
    """
    if (
        projected.ndim != 2
        or projected.shape[1] <= 0
        or projected.shape[1] % 256
        or projected.dtype != torch.bfloat16
        or not projected.is_cuda
        or not projected.is_contiguous()
    ):
        raise ValueError("projected must be contiguous CUDA BF16 [M,H*256]")
    rows, width = projected.shape
    heads = width // 256
    if cos_sin.ndim != 2:
        raise ValueError("cos_sin must be contiguous FP32 [S,64]")
    for name, tensor, shape, dtype in (
        ("q_nope_out", q_nope_out, (rows, heads, 192), torch.bfloat16),
        ("q_out", q_out, (rows, heads, 576), torch.bfloat16),
        ("cos_sin", cos_sin, (cos_sin.shape[0], 64), torch.float32),
    ):
        if (
            tuple(tensor.shape) != shape
            or tensor.dtype != dtype
            or tensor.device != projected.device
            or not tensor.is_contiguous()
        ):
            raise ValueError(
                f"{name} must be contiguous {dtype} {shape} on the Q device"
            )
    if (
        positions.shape != (rows,)
        or positions.dtype not in (torch.int32, torch.int64)
        or positions.device != projected.device
        or not positions.is_contiguous()
    ):
        raise ValueError("positions must be contiguous device int32/int64 [M]")
    if not isinstance(is_neox_style, bool):
        raise ValueError("is_neox_style must be bool")
    if rows:
        if cos_sin.shape[0] == 0:
            raise ValueError("cos_sin must contain position rows")
        _split_q_rope_kernel[(rows, triton.cdiv(heads, 4))](
            projected,
            cos_sin,
            positions,
            q_nope_out,
            q_out,
            HEADS=heads,
            IS_NEOX=is_neox_style,
            HEAD_TILE=4,
            num_warps=4,
        )
    return q_nope_out, q_out
