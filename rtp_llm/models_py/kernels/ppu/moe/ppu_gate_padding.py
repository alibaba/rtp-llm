"""Fuse the existing sqrt-softplus router with DeepEP's inactive columns."""

import torch
import triton
import triton.language as tl


@triton.jit
def _gate_padded(
    scores,
    bias,
    indices,
    weights,
    mask,
    E: tl.constexpr,
    K: tl.constexpr,
    PAD: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_E: tl.constexpr,
    HAS_MASK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    experts = tl.arange(0, BLOCK_E)
    raw = tl.load(scores + row * E + experts, experts < E, 0).to(tl.float32)
    correction = tl.load(bias + experts, experts < E, 0).to(tl.float32)
    finite_raw = tl.abs(raw) < float("inf")
    finite_bias = tl.abs(correction) < float("inf")
    finite = tl.sum(((experts < E) & (~finite_raw | ~finite_bias)).to(tl.int32), 0) == 0
    raw = tl.where(finite_raw, raw, 0.0)
    correction = tl.where(finite_bias, correction, 0.0)
    active = tl.sqrt(tl.where(raw > 20.0, raw, tl.log(1.0 + tl.exp(raw))))
    ranking = tl.where(experts < E, active + correction, -float("inf"))
    slots = tl.arange(0, PAD)
    selected_weights = tl.full((PAD,), 0.0, tl.float32)
    selected_indices = tl.full((PAD,), -1, tl.int64)
    for k in tl.static_range(K):
        idx = tl.argmax(ranking, 0)
        weight = tl.sum(tl.where(experts == idx, active, 0.0), 0)
        selected_weights = tl.where(slots == k, weight, selected_weights)
        selected_indices = tl.where(
            slots == k, tl.where(finite, idx, k).to(tl.int64), selected_indices
        )
        ranking = tl.where(experts == idx, -float("inf"), ranking)
    normalized = selected_weights / (tl.sum(selected_weights, 0) + 1.0e-12) * SCALE
    normalized = tl.where(finite, normalized, SCALE / K)
    normalized = tl.where(slots < K, normalized, 0.0)
    if HAS_MASK:
        is_active = tl.load(mask + row)
        selected_indices = tl.where(is_active, selected_indices, -1)
        normalized = tl.where(is_active, normalized, 0.0)
    tl.store(indices + row * PAD + slots, selected_indices)
    tl.store(weights + row * PAD + slots, normalized)


def is_supported(scores, bias, topk):
    return (
        scores.is_cuda
        and scores.dtype == torch.float32
        and scores.ndim == 2
        and scores.is_contiguous()
        and bias.device == scores.device
        and bias.dtype == torch.float32
        and bias.shape == (scores.shape[1],)
        and bias.is_contiguous()
        and 1 <= topk <= min(16, scores.shape[1])
    )


def supports_active_mask(scores, mask):
    return mask is None or (
        mask.device == scores.device
        and mask.dtype == torch.bool
        and mask.shape == (scores.shape[0],)
        and mask.is_contiguous()
    )


def gate_with_padding(scores, bias, topk, route_scale=1.0, *, active_token_mask=None):
    """Same ordered routes and FP32 bits, with exactly -1/0 inactive slots."""
    if not is_supported(scores, bias, topk):
        raise ValueError("PPU padded gate requires contiguous FP32 scores/bias")
    if not supports_active_mask(scores, active_token_mask):
        raise ValueError("PPU gate mask must be one contiguous device bool per token")
    pad = max(2, triton.next_power_of_2(topk))
    m, e = scores.shape
    indices = torch.empty((m, pad), device=scores.device, dtype=torch.int64)
    weights = torch.empty((m, pad), device=scores.device, dtype=torch.float32)
    if m:
        _gate_padded[(m,)](
            scores,
            bias,
            indices,
            weights,
            active_token_mask,
            e,
            topk,
            pad,
            float(route_scale),
            triton.next_power_of_2(e),
            active_token_mask is not None,
            num_warps=4,
            num_stages=2,
        )
    return weights, indices
