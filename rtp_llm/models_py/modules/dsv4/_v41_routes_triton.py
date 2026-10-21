"""SM120 active-route masks and exact validation for small MoE batches."""

import torch
import triton
import triton.language as tl


@triton.jit
def _validate_kernel(W, I, V, A, L, N: tl.constexpr, K: tl.constexpr, E: tl.constexpr,
                     WS0: tl.constexpr, WS1: tl.constexpr, IS0: tl.constexpr,
                     IS1: tl.constexpr, VS: tl.constexpr, R: tl.constexpr, C: tl.constexpr):
    row = tl.arange(0, R)
    col = tl.arange(0, C)
    live = (row[:, None] < N) & (col[None, :] < K)
    valid = tl.load(V + row * VS, row < N, 0)
    weights = tl.load(W + row[:, None] * WS0 + col[None, :] * WS1, live, 0)
    ids = tl.load(I + row[:, None] * IS0 + col[None, :] * IS1, live, 0).to(tl.int64)
    active = live & valid[:, None] & (weights != 0)
    good_w = ~live | ~valid[:, None] | ((tl.abs(weights) < float("inf")) & (weights >= 0))
    good_i = ~active | ((ids >= 0) & (ids < E))
    duplicate = (
        (ids[:, :, None] == ids[:, None, :])
        & active[:, :, None] & active[:, None, :]
        & (col[:, None] < col[None, :])[None, :, :]
    )
    bad = tl.sum(tl.sum((~good_w | ~good_i).to(tl.int32), 1), 0)
    duplicates = tl.sum(tl.sum(tl.sum(duplicate.to(tl.int32), 2), 1), 0)
    tl.store(L, (bad == 0) & (duplicates == 0))
    tl.store(A + row[:, None] * K + col[None, :], active, live)


def can_validate_routes(weights, indices, valid_rows, experts):
    return weights.device.type == 'cuda' and torch.cuda.get_device_capability(weights.device) == (12, 0) and (weights.dtype == torch.float32) and (weights.ndim == 2) and (0 < weights.shape[0] <= 32) and (0 < weights.shape[1] <= 32) and (indices.shape == weights.shape) and (indices.dtype in (torch.int32, torch.int64)) and (indices.device == weights.device) and (valid_rows.shape == (weights.shape[0],)) and (valid_rows.dtype == torch.bool) and (valid_rows.device == weights.device) and (type(experts) is int) and (0 < experts <= torch.iinfo(torch.int32).max)


def validate_routes(weights, indices, valid_rows, experts):
    (rows, topk) = weights.shape
    active = torch.empty(weights.shape, dtype=torch.bool, device=weights.device)
    legal = torch.empty((), dtype=torch.bool, device=weights.device)
    _validate_kernel[1,](weights, indices, valid_rows, active, legal, rows, topk, experts, weights.stride(0), weights.stride(1), indices.stride(0), indices.stride(1), valid_rows.stride(0), R=triton.next_power_of_2(rows), C=triton.next_power_of_2(topk), num_warps=4, enable_fp_fusion=False)
    torch._assert_async(legal, 'Invalid V4.1 dispatch route values')
    return active
