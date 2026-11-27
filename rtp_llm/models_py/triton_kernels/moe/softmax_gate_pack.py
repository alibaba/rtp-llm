# SPDX-License-Identifier: Apache-2.0
"""Softmax top-k and MXFP8 staging for routed-only and SE MegaMoE.

Adapted from rtp-omni/rtp-omni 1e65797d2679399a2197b882d0bfc94e3143ca39,
python/rtp_omni/layers/mega_moe_pack.py.
"""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["M"])
def _mega_moe_softmax_gate_pack_kernel(
    x_ptr,
    logits_ptr,
    out_fp8_ptr,
    out_sf_ptr,
    out_weights_ptr,
    out_indices_ptr,
    M,
    N: tl.constexpr,
    E: tl.constexpr,
    K: tl.constexpr,
    x_stride_m: tl.constexpr,
    logits_stride_m: tl.constexpr,
    out_stride_m: tl.constexpr,
    sf_stride_m: tl.constexpr,
    out_weights_stride_m: tl.constexpr,
    out_indices_stride_m: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    # Route and quantize in separate CTAs of the same launch. This lets the
    # quantization tile cover several rows without serializing their routing.
    if pid < M:
        row = pid.to(tl.int64)
        expert = tl.arange(0, BLOCK_E)
        slot = tl.arange(0, BLOCK_K)
        valid = expert < E
        logits = tl.load(
            logits_ptr + row * logits_stride_m + expert, valid, -float("inf")
        ).to(tl.float32)
        invalid = valid & ~(tl.abs(logits) < float("inf"))
        row_is_finite = tl.sum(invalid.to(tl.int32), axis=0) == 0
        logits = tl.where(invalid, 0.0, logits)
        exp_logits = tl.exp(logits - tl.max(logits, axis=0))
        probs = exp_logits / tl.sum(exp_logits, axis=0)

        # Rank rounded softmax probabilities and break ties by expert ID.
        # Positive float bits sort as uint32. Add one so zero-probability
        # experts still rank above padding and already-selected experts.
        keys = tl.where(valid, probs.to(tl.uint32, bitcast=True) + 1, 0)
        selected = tl.full((BLOCK_K,), 0, tl.uint32)
        indices = tl.full((BLOCK_K,), 0, tl.int32)
        for k in tl.static_range(K):
            best = tl.max(keys, axis=0)
            best_idx = tl.min(tl.where(keys == best, expert, BLOCK_E), axis=0)
            selected = tl.where(slot == k, best - 1, selected)
            indices = tl.where(slot == k, best_idx, indices)
            keys = tl.where(expert == best_idx, 0, keys)

        values = selected.to(tl.float32, bitcast=True)
        weights = values / tl.sum(values, axis=0)
        weights = tl.where(row_is_finite, weights, float("nan"))
        indices = tl.where(row_is_finite, indices, slot)
        tl.store(
            out_indices_ptr + row * out_indices_stride_m + slot,
            indices.to(tl.int64),
            mask=slot < K,
        )
        tl.store(
            out_weights_ptr + row * out_weights_stride_m + slot,
            weights,
            mask=slot < K,
        )
    else:
        pid_pack = pid - M
        pid_row = pid_pack // tl.cdiv(N, BLOCK_N)
        pid_col = pid_pack % tl.cdiv(N, BLOCK_N)
        rows = pid_row.to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = pid_col * BLOCK_N + tl.arange(0, BLOCK_N)
        row_mask = rows < M
        mask = row_mask[:, None] & (cols[None, :] < N)
        x = tl.load(x_ptr + rows[:, None] * x_stride_m + cols[None, :], mask, 0.0).to(
            tl.float32
        )
        x = tl.where(tl.abs(x) < float("inf"), x, 0.0)
        groups = tl.reshape(x, (BLOCK_M, BLOCK_N // 32, 32))
        absmax = tl.maximum(tl.max(tl.abs(groups), axis=2), 1.0e-4)
        bits = (absmax / 448.0).to(tl.int32, bitcast=True)
        exp = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0)
        exp = tl.minimum(tl.maximum(exp, 1), 254)
        scale = (exp << 23).to(tl.float32, bitcast=True)
        quant = tl.clamp(groups / scale[:, :, None], -448.0, 448.0).to(tl.float8e4nv)
        quant = tl.reshape(quant, (BLOCK_M, BLOCK_N))
        tl.store(
            out_fp8_ptr + rows[:, None] * out_stride_m + cols[None, :], quant, mask
        )

        exp = tl.reshape(exp, (BLOCK_M, BLOCK_N // 128, 4))
        packed = tl.sum(exp << (tl.arange(0, 4)[None, None, :] * 8), axis=2)
        sf_cols = pid_col * (BLOCK_N // 128) + tl.arange(0, BLOCK_N // 128)
        tl.store(
            out_sf_ptr + rows[:, None] * sf_stride_m + sf_cols[None, :],
            packed,
            mask=row_mask[:, None] & (sf_cols[None, :] < N // 128),
        )


def fused_pack_mega_moe_softmax_inputs(
    hidden: torch.Tensor,
    logits: torch.Tensor,
    out_fp8: torch.Tensor,
    out_sf: torch.Tensor,
    out_indices: torch.Tensor,
    out_weights: torch.Tensor,
    *,
    topk: int,
) -> None:
    from rtp_llm.models_py.triton_kernels.moe.mega_moe_input_pack import (
        _validate_gate_pack_inputs,
    )

    tokens, dim, experts = _validate_gate_pack_inputs(
        hidden, logits, out_fp8, out_sf, out_indices, out_weights, topk
    )
    if not 1 <= topk <= min(experts, 32):
        raise ValueError("Softmax gate pack requires 1 <= topk <= min(experts, 32)")
    if tokens == 0 or dim == 0:
        return

    # M11: spread decode rows over small CTAs, then use larger quantization
    # tiles to keep CTA overhead from dominating long prefill chunks.
    if tokens <= 64:
        block_m, block_n, num_warps = 1, 128, 1
    elif tokens <= 2048:
        block_m, block_n, num_warps = 4, 256, 2
    else:
        block_m, block_n, num_warps = 16, 512, 4
    grid = (tokens + triton.cdiv(tokens, block_m) * triton.cdiv(dim, block_n),)
    _mega_moe_softmax_gate_pack_kernel[grid](
        hidden,
        logits,
        out_fp8,
        out_sf,
        out_weights,
        out_indices,
        tokens,
        dim,
        experts,
        topk,
        hidden.stride(0),
        logits.stride(0),
        out_fp8.stride(0),
        out_sf.stride(0),
        out_weights.stride(0),
        out_indices.stride(0),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_E=triton.next_power_of_2(experts),
        BLOCK_K=triton.next_power_of_2(topk),
        num_warps=num_warps,
    )
