# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM 0.29.0 arithmetic for RMSNorm and selected-token log probabilities.

Adapted from model_executor/determinism/batch_invariant.py; see UPSTREAM.md.
Keep the 1024-wide reduction, sqrt/reciprocal and subtraction order intact.
The normalizer is a pair, not their rounded sum: x - max - log_sum_exp.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _rms_norm_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0).to(tl.int64)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    sum_sq = tl.zeros([1], dtype=tl.float32)
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols
        vals = tl.load(row_start_ptr + col_idx, mask=mask, other=0.0)
        vals_f32 = vals.to(tl.float32)
        sq_vals = vals_f32 * vals_f32
        sum_sq += tl.sum(tl.where(mask, sq_vals, 0.0))
    mean_sq = sum_sq / n_cols
    rms = tl.sqrt(mean_sq + eps)
    inv_rms = 1.0 / rms
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols
        vals = tl.load(row_start_ptr + col_idx, mask=mask, other=0.0)
        vals_f32 = vals.to(tl.float32)
        output_f32 = vals_f32 * inv_rms
        weight = tl.load(weight_ptr + col_idx, mask=mask, other=1.0)
        output_f32 = output_f32 * weight.to(tl.float32)
        output = output_f32.to(vals.dtype)
        tl.store(output_row_start_ptr + col_idx, output, mask=mask)


def rmsnorm(x, weight, eps):
    if x.ndim == 0 or x.shape[-1] == 0 or not x.is_cuda:
        raise ValueError("batch invariant RMSNorm requires nonempty CUDA rows")
    h = x.shape[-1]
    if weight.shape != (h,) or x.device != weight.device:
        raise ValueError("invalid batch invariant RMSNorm weight")
    shape = x.shape
    x = x.reshape(-1, h).contiguous()
    weight = weight.contiguous()
    out = torch.empty_like(x)
    if x.shape[0]:
        _rms_norm_kernel[(x.shape[0],)](
            x, weight, out, x.stride(0), out.stride(0), h, eps, BLOCK_SIZE=1024
        )
    return out.view(shape)


@triton.jit
def _logprob_normalizers_kernel(
    input_ptr,
    output_ptr,
    input_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0).to(tl.int64)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    max_val = -float("inf")
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols
        vals = tl.load(row_start_ptr + col_idx, mask=mask, other=-float("inf"))
        max_val = tl.max(tl.maximum(vals, max_val))
    sum_exp = 0.0
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < n_cols
        vals = tl.load(row_start_ptr + col_idx, mask=mask, other=0.0)
        exp_vals = tl.exp(vals - max_val)
        sum_exp += tl.sum(tl.where(mask, exp_vals, 0.0))
    log_sum_exp = tl.log(sum_exp)
    # Preserve the two FP32 values through native prompt/decode dispatch.
    # Combining them into an LSE changes the final subtraction's rounding.
    tl.store(output_ptr + row_idx * 2, max_val)
    tl.store(output_ptr + row_idx * 2 + 1, log_sum_exp)


def logprob_normalizers(logits):
    """Return [rows, 2] FP32 (row maximum, log shifted exponential sum)."""
    if logits.ndim != 2 or logits.shape[1] == 0 or not logits.is_cuda:
        raise ValueError(
            "batch invariant logprobs require nonempty CUDA vocabulary rows"
        )
    x = logits.float().contiguous()
    rows, vocab = x.shape
    out = torch.empty((rows, 2), device=x.device, dtype=torch.float32)
    if rows:
        _logprob_normalizers_kernel[(rows,)](
            x, out, x.stride(0), vocab, BLOCK_SIZE=1024
        )
    return out


def logsumexp(logits):
    """Convenience LSE; selected-token scoring must retain both components."""
    normalizers = logprob_normalizers(logits)
    return normalizers[:, :1] + normalizers[:, 1:]


def selected_logprobs(logits, token_ids):
    if token_ids.numel() != logits.shape[0]:
        raise ValueError("one target token is required per logits row")
    normalizers = logprob_normalizers(logits)
    values = logits.float().gather(1, token_ids.reshape(-1, 1).long())
    return (values - normalizers[:, :1]) - normalizers[:, 1:]
