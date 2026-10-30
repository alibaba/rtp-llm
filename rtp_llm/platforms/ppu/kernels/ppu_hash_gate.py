"""Fuse hash lookup, finite-row containment, normalization and route padding."""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from .ppu_gate_padding import supports_active_mask


@triton.jit
def _hash_routes(
    RAW,
    ACTIVE,
    IDS,
    TABLE,
    OUT_I,
    OUT_W,
    VALID,
    MASK,
    E: tl.constexpr,
    K: tl.constexpr,
    VOCAB: tl.constexpr,
    PAD: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_E: tl.constexpr,
    HAS_MASK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    experts = tl.arange(0, BLOCK_E)
    raw = tl.load(RAW + row * E + experts, experts < E, 0)
    active = tl.load(ACTIVE + row * E + experts, experts < E, 0)
    finite = (
        tl.sum(
            (
                (experts < E)
                & (
                    (tl.abs(raw) >= float("inf"))
                    | (raw != raw)
                    | (tl.abs(active) >= float("inf"))
                    | (active != active)
                )
            ).to(tl.int32),
            0,
        )
        == 0
    )
    slots = tl.arange(0, PAD)
    token = tl.load(IDS + row).to(tl.int64)
    token_ok = (token >= -VOCAB) & (token < VOCAB)
    token = tl.where(token < 0, token + VOCAB, token)
    indices = tl.load(TABLE + token * K + slots, token_ok & (slots < K), 0).to(tl.int64)
    indices = tl.where(finite, indices, slots.to(tl.int64))
    index_ok = (indices >= 0) & (indices < E)
    selected = tl.load(ACTIVE + row * E + indices, (slots < K) & index_ok, 0)
    denom = tl.sum(selected, 0) + 1.0e-12
    weights = tl.div_rn(selected, denom) * SCALE
    weights = tl.where(finite, weights, SCALE / K)
    keep = slots < K
    if HAS_MASK:
        keep = keep & tl.load(MASK + row)
    tl.store(OUT_I + row * PAD + slots, tl.where(keep, indices, -1))
    tl.store(OUT_W + row * PAD + slots, tl.where(keep, weights, 0.0))
    tl.store(
        VALID + row, token_ok & (tl.sum(((slots < K) & ~index_ok).to(tl.int32), 0) == 0)
    )


def is_supported(scores, input_ids, table):
    return (
        scores.is_cuda
        and scores.dtype == torch.float32
        and scores.ndim == 2
        and scores.is_contiguous()
        and input_ids is not None
        and input_ids.numel() == scores.shape[0]
        and input_ids.dtype in (torch.int32, torch.int64)
        and table.ndim == 2
        and 1 <= table.shape[1] <= min(16, scores.shape[1])
        and table.shape[0] > 0
        and table.dtype in (torch.int32, torch.int64)
        and all(
            t.device == scores.device and t.is_contiguous() for t in (input_ids, table)
        )
    )


def hash_gate_with_padding(
    scores, input_ids, table, route_scale=1.0, *, active_token_mask=None
):
    if not is_supported(scores, input_ids, table):
        raise ValueError(
            "PPU hash routing requires FP32 scores and contiguous integer IDs/table"
        )
    if not supports_active_mask(scores, active_token_mask):
        raise ValueError("PPU gate mask must be one contiguous device bool per token")
    m, e = scores.shape
    k = table.shape[1]
    pad = max(2, triton.next_power_of_2(k))
    indices = torch.empty((m, pad), device=scores.device, dtype=torch.int64)
    weights = torch.empty((m, pad), device=scores.device, dtype=torch.float32)
    if m:
        # Retain PyTorch's subnormal softplus/sqrt behavior. The PPU libdevice
        # fused activation flushes some tiny intermediates and is not bit-exact.
        active = F.softplus(scores).sqrt()
        valid = torch.empty((m,), device=scores.device, dtype=torch.bool)
        _hash_routes[(m,)](
            scores,
            active,
            input_ids,
            table,
            indices,
            weights,
            valid,
            active_token_mask,
            e,
            k,
            table.shape[0],
            pad,
            float(route_scale),
            triton.next_power_of_2(e),
            active_token_mask is not None,
            num_warps=4,
            enable_fp_fusion=False,
        )
        # Keep the baseline's asynchronous bounds check for finite hash rows.
        torch._assert_async(valid.all(), "hash token/expert index out of bounds")
    return weights, indices
