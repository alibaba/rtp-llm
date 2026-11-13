# SPDX-License-Identifier: Apache-2.0
"""Softmax top-k + MXFP8 packing adapted from rtp-omni.

Source: rtp-omni/rtp-omni 1e65797d2679399a2197b882d0bfc94e3143ca39,
python/rtp_omni/layers/mega_moe_pack.py. RTP uses probability-order top-k,
zero-epsilon renormalization and propagates invalid router logits as NaNs.
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
    first_expert,
    num_local_experts,
    owners_ptr,
    norm_eps: tl.constexpr,
    eps: tl.constexpr,
    fp8_max: tl.constexpr,
    HAS_OWNERS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m_blk = tl.program_id(0).to(tl.int64)
    pid_blk = tl.program_id(1)
    offs_m = pid_m_blk * BLOCK_M + tl.arange(0, BLOCK_M).to(tl.int64)
    row_mask = offs_m < M
    offs_32 = tl.arange(0, 32)
    packed = tl.zeros((BLOCK_M,), dtype=tl.int32)
    for pack_idx in tl.static_range(4):
        cols = pid_blk * 128 + pack_idx * 32 + offs_32
        mask = row_mask[:, None] & (cols[None, :] < N)
        x = tl.load(
            x_ptr + offs_m[:, None] * x_stride_m + cols[None, :], mask=mask, other=0.0
        ).to(tl.float32)
        x = tl.where(tl.abs(x) < float("inf"), x, 0.0)
        block_absmax = tl.maximum(tl.max(tl.abs(x), axis=1), eps)
        scale_raw = block_absmax / fp8_max
        scale_raw_bits = scale_raw.to(tl.int32, bitcast=True)
        exp = ((scale_raw_bits >> 23) & 0xFF) + ((scale_raw_bits & 0x7FFFFF) != 0)
        exp = tl.minimum(tl.maximum(exp, 1), 254)
        scale_bits = exp << 23
        scale = scale_bits.to(tl.float32, bitcast=True)
        q = tl.clamp(x / scale[:, None], -fp8_max, fp8_max).to(tl.float8e4nv)
        tl.store(
            out_fp8_ptr + offs_m[:, None] * out_stride_m + cols[None, :], q, mask=mask
        )
        packed = packed | (exp << (pack_idx * 8))
    tl.store(out_sf_ptr + offs_m * sf_stride_m + pid_blk, packed, mask=row_mask)

    if pid_blk == 0:
        offs_e = tl.arange(0, BLOCK_E)
        e_mask = offs_e < E
        k_offs = tl.arange(0, BLOCK_K)
        k_mask_base = k_offs < K
        for row_i in tl.static_range(BLOCK_M):
            row = pid_m_blk * BLOCK_M + row_i
            r_mask = row < M
            if HAS_OWNERS:
                own = tl.load(owners_ptr + row, mask=r_mask, other=0) != 0
            else:
                own = True
            scores = tl.load(
                logits_ptr + row * logits_stride_m + offs_e,
                mask=r_mask & e_mask,
                other=-float("inf"),
            ).to(tl.float32)
            score_is_finite = tl.abs(scores) < float("inf")
            row_is_finite = (
                tl.sum((r_mask & e_mask & ~score_is_finite).to(tl.int32), axis=0) == 0
            )
            # Match RTP: softmax first, then stable lower-index top-k and
            # renormalization. In particular, rounded softmax ties remain ties.
            safe = tl.where(e_mask, scores, -float("inf"))
            probs = tl.exp(safe - tl.max(safe, axis=0))
            probs = probs / tl.sum(probs, axis=0)
            cur = tl.where(e_mask, probs, -1.0)
            selected = tl.zeros((BLOCK_K,), dtype=tl.float32)
            kept = tl.zeros((BLOCK_K,), dtype=tl.int32)
            for k in tl.static_range(K):
                idx = tl.argmax(cur, axis=0)
                val = tl.sum(tl.where(offs_e == idx, cur, 0.0), axis=0)
                safe_idx = tl.where(row_is_finite, idx, k)
                keep = (
                    (safe_idx >= first_expert)
                    & (safe_idx < first_expert + num_local_experts)
                    & own
                )
                tl.store(
                    out_indices_ptr + row * out_indices_stride_m + k,
                    tl.where(keep, safe_idx, -1).to(tl.int64),
                    mask=r_mask,
                )
                selected = tl.where(k_offs == k, val, selected)
                kept = tl.where(k_offs == k, keep.to(tl.int32), kept)
                cur = tl.where(offs_e == idx, -1.0, cur)
            k_mask = r_mask & k_mask_base
            selected_sum = tl.sum(tl.where(k_mask_base, selected, 0.0), axis=0)
            weights = selected / selected_sum
            # Invalid logits must stay visible to correctness checks.
            weights = tl.where(row_is_finite, weights, float("nan"))
            weights = tl.where(kept != 0, weights, 0.0)
            tl.store(
                out_weights_ptr + row * out_weights_stride_m + k_offs,
                weights,
                mask=k_mask,
            )


def fused_pack_mega_moe_softmax_inputs(
    hidden, logits, out_fp8, out_sf, out_indices, out_weights, *, topk
):
    from rtp_llm.models_py.triton_kernels.moe.mega_moe_input_pack import (
        _validate_gate_pack_inputs,
    )

    tokens, dim, experts = _validate_gate_pack_inputs(
        hidden, logits, out_fp8, out_sf, out_indices, out_weights, topk
    )
    if not 1 <= topk <= min(experts, 32):
        raise ValueError("Softmax gate pack requires 1 <= topk <= min(experts, 32)")
    if tokens == 0:
        return
    # Upstream's decode geometry. Tune phase buckets only after profiling.
    block_m = 1
    _mega_moe_softmax_gate_pack_kernel[
        (triton.cdiv(tokens, block_m), triton.cdiv(dim, 128))
    ](
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
        0,
        experts,
        hidden,
        0.0,
        1.0e-4,
        448.0,
        HAS_OWNERS=False,
        BLOCK_M=block_m,
        BLOCK_E=triton.next_power_of_2(experts),
        BLOCK_K=triton.next_power_of_2(topk),
        num_warps=4,
    )
