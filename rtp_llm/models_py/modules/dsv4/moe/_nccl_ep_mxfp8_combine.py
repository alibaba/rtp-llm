"""Ordered per-peer dequant-and-sum for the NCCL MXFP8 expert return.

Ported from the user's fork donor
``Chiiisen/rtp-llm@f1f85a914b4077742e48b1aea7af968cd6b1bdd2``
``rtp_llm/models_py/modules/dsv4/moe/_nccl_ep_combine_triton.py`` and kept
byte-comparable in behaviour.  Only the docstring and the error text are ours.

The kernel dequantizes each peer's block32-MXFP8 partial output in FP32 and
accumulates in PEER ORDER, so the reduction is deterministic: the summation
order is `for peer in 0..world-1`, not whatever order NCCL completed in.  A
non-deterministic combine would make an already-unstable reference worse and
would hide transport bugs behind run-to-run drift.

UE8M0 scale decode: the stored byte is a biased exponent, `value = 2^(b-127)`.
"""
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
    """Sum `world_size` peers' block32-MXFP8 partials for `n_rows` owner rows.

    `returned_payload` is `[world_size * n_rows, hidden_size + hidden_size//32]`
    uint8, peer-major: rows `[p*n_rows, (p+1)*n_rows)` came from peer `p`.
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
    if n_rows == 0:
        return torch.empty((0, hidden_size), dtype=out_dtype, device=returned_payload.device)
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


__all__ = ["SCALE_BLOCK", "mxfp8_dequant_peer_sum"]
