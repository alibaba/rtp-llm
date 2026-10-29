"""GQA block-sparse FMHA triton kernel (prefill shape).

Takes the indexer's ``selected_indices [B, S, K]`` (int32, -1 padded), gathers
the corresponding K and V entries from the dense KV tensors, and computes
softmax attention with per-row selected sets.  GQA head mapping:
``H_q // H_kv`` query heads share one KV head (qwen4: 24 / 2 = 12).

This is the **triton production kernel** which replaces :mod:`sparse_attention`
once validated against it.

Contract::

    q [B, H_q, S, D],  k [B, H_kv, T, D],  v [B, H_kv, T, D]
    selected [B, S, K] int32   (-1 = padded)
    -> o [B, H_q, S, D]

Kernel: one program per (b, h_q, s) triplet.  K entries are processed in
BLK_K-sized chunks so that programs with large K do not exhaust registers.
"""

import torch
import triton
import triton.language as tl

_BLK_K = 64


@triton.jit
def _sparse_prefill_online_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    idx_ptr,
    out_ptr,
    stride_q_b,
    stride_q_h,
    stride_q_s,
    stride_k_b,
    stride_k_h,
    stride_k_t,
    stride_v_b,
    stride_v_h,
    stride_v_t,
    stride_idx_b,
    stride_idx_s,
    K,
    T,
    D: tl.constexpr,
    SCALE: tl.constexpr,
    q_per_kv: tl.constexpr,
    MAX_BLKS: tl.constexpr,
    BLK_K: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)
    h_kv = pid_h // q_per_kv
    rd = tl.arange(0, D)
    q = tl.load(
        q_ptr + pid_b * stride_q_b + pid_h * stride_q_h + pid_s * stride_q_s + rd
    ).to(tl.float32)
    idx_base = idx_ptr + pid_b * stride_idx_b + pid_s * stride_idx_s
    k_base = k_ptr + pid_b * stride_k_b + h_kv * stride_k_h
    v_base = v_ptr + pid_b * stride_v_b + h_kv * stride_v_h

    max_score = tl.full((1,), float("-inf"), dtype=tl.float32)
    sum_exp = tl.zeros((1,), dtype=tl.float32)
    acc_o = tl.zeros((D,), dtype=tl.float32)
    for blk in range(MAX_BLKS):
        tr = blk * BLK_K + tl.arange(0, BLK_K)
        in_range = tr < K
        idx = tl.load(idx_base + tr, mask=in_range, other=-1)
        valid = in_range & (idx >= 0) & (idx < T)
        offsets = idx[:, None] * stride_k_t + rd[None, :]
        k_chunk = tl.load(k_base + offsets, mask=valid[:, None], other=0.0).to(
            tl.float32
        )
        score = tl.sum(q[None, :] * k_chunk, axis=1) * SCALE
        score = tl.where(valid, score, float("-inf"))
        next_max = tl.maximum(max_score, tl.max(score, axis=0))
        # All-masked chunks have next_max=-inf. Avoid -inf - -inf and keep
        # their numerator and denominator exactly zero.
        rescale = tl.where(next_max == float("-inf"), 0.0, tl.exp(max_score - next_max))
        weights = tl.where(valid, tl.exp(score - next_max), 0.0)
        v_offsets = idx[:, None] * stride_v_t + rd[None, :]
        v_chunk = tl.load(v_base + v_offsets, mask=valid[:, None], other=0.0).to(
            tl.float32
        )
        acc_o = acc_o * rescale + tl.sum(weights[:, None] * v_chunk, axis=0)
        sum_exp = sum_exp * rescale + tl.sum(weights)
        max_score = next_max

    out = tl.where(sum_exp > 0, acc_o / sum_exp, 0.0)
    tl.store(
        out_ptr + pid_b * stride_q_b + pid_h * stride_q_h + pid_s * stride_q_s + rd,
        out.to(q_ptr.dtype.element_ty),
    )


@triton.jit
def _sparse_prefill_grouped_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    idx_ptr,
    out_ptr,
    stride_q_b,
    stride_q_h,
    stride_q_s,
    stride_k_b,
    stride_k_h,
    stride_k_t,
    stride_v_b,
    stride_v_h,
    stride_v_t,
    stride_idx_b,
    stride_idx_s,
    K,
    T,
    D: tl.constexpr,
    SCALE: tl.constexpr,
    q_per_kv: tl.constexpr,
    MAX_BLKS: tl.constexpr,
    BLK_K: tl.constexpr,
    BLK_H: tl.constexpr,
    SM120_GQA_PV: tl.constexpr,
):
    """Share selected K/V across GQA heads and use tensor-core reductions."""
    pid_b = tl.program_id(0)
    pid_kv = tl.program_id(1)
    pid_s = tl.program_id(2)
    rh = tl.arange(0, BLK_H)
    rd = tl.arange(0, D)
    q = tl.load(
        q_ptr
        + pid_b * stride_q_b
        + (pid_kv * q_per_kv + rh[:, None]) * stride_q_h
        + pid_s * stride_q_s
        + rd[None, :],
        mask=rh[:, None] < q_per_kv,
        other=0.0,
    )
    idx_base = idx_ptr + pid_b * stride_idx_b + pid_s * stride_idx_s
    k_base = k_ptr + pid_b * stride_k_b + pid_kv * stride_k_h
    v_base = v_ptr + pid_b * stride_v_b + pid_kv * stride_v_h
    max_score = tl.full((BLK_H,), float("-inf"), tl.float32)
    sum_exp = tl.zeros((BLK_H,), tl.float32)
    acc = tl.zeros((BLK_H, D), tl.float32)
    for blk in range(MAX_BLKS):
        tr = blk * BLK_K + tl.arange(0, BLK_K)
        idx = tl.load(idx_base + tr, mask=tr < K, other=-1)
        valid = (tr < K) & (idx >= 0) & (idx < T)
        if not SM120_GQA_PV or tl.sum(valid.to(tl.int32), 0) > 0:
            k = tl.load(
                k_base + idx[:, None] * stride_k_t + rd[None, :],
                mask=valid[:, None],
                other=0.0,
            )
            score = tl.dot(q, tl.trans(k)) * SCALE
            score = tl.where(valid[None, :], score, float("-inf"))
            next_max = tl.maximum(max_score, tl.max(score, axis=1))
            rescale = tl.where(
                next_max == float("-inf"), 0.0, tl.exp(max_score - next_max)
            )
            weights = tl.where(valid[None, :], tl.exp(score - next_max[:, None]), 0.0)
            v = tl.load(
                v_base + idx[:, None] * stride_v_t + rd[None, :],
                mask=valid[:, None],
                other=0.0,
            )
            # The SM120 specialization uses two BF16 probability components;
            # both products accumulate in FP32. Other hardware and geometries
            # retain the FP32-probability tensor-core product.
            if SM120_GQA_PV:
                # Preserve the first BF16 component's rounding residual in
                # the second component (approximately 14 mantissa bits).
                high = weights.to(tl.bfloat16)
                low = (weights - high.to(tl.float32)).to(tl.bfloat16)
                acc = acc * rescale[:, None]
                acc = tl.dot(high, v, acc)
                acc = tl.dot(low, v, acc)
            else:
                acc = acc * rescale[:, None] + tl.dot(
                    weights, v.to(tl.float32), input_precision="tf32x3"
                )
            sum_exp = sum_exp * rescale + tl.sum(weights, axis=1)
            max_score = next_max
    out = tl.where(sum_exp[:, None] > 0, acc / sum_exp[:, None], 0.0)
    tl.store(
        out_ptr
        + pid_b * stride_q_b
        + (pid_kv * q_per_kv + rh[:, None]) * stride_q_h
        + pid_s * stride_q_s
        + rd[None, :],
        out.to(q_ptr.dtype.element_ty),
        mask=rh[:, None] < q_per_kv,
    )


def sparse_prefill_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    selected: torch.Tensor,
    *,
    blk_k: int = _BLK_K,
    validate_selected: bool = True,
) -> torch.Tensor:
    """Triton sparse FMHA (one program per query position).

    Args see module docstring.  ``scale=1/sqrt(D)`` is specialised from the
    runtime head dimension for each compiled kernel variant.
    """
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4 or selected.dim() != 3:
        raise ValueError("q/k/v must be rank 4 and selected must be rank 3")
    B, H_q, S, D = q.shape
    H_kv = k.shape[1]
    T = k.shape[2]
    K = selected.shape[-1]
    if k.shape != v.shape:
        raise ValueError(f"k and v shapes must match, got {k.shape} and {v.shape}")
    if k.shape[0] != B or k.shape[3] != D:
        raise ValueError("q, k, and v batch/head-dim geometry must match")
    if selected.shape[:2] != (B, S):
        raise ValueError(f"selected must have leading shape [{B}, {S}]")
    if H_kv <= 0 or H_q % H_kv:
        raise ValueError(f"query heads {H_q} must be divisible by KV heads {H_kv}")
    if D <= 0 or D & (D - 1):
        raise ValueError(f"head dimension must be a positive power of two, got {D}")
    if blk_k <= 0 or blk_k & (blk_k - 1):
        raise ValueError(f"blk_k must be a positive power of two, got {blk_k}")
    q_per_kv = H_q // H_kv
    dev = q.device

    if not q.is_cuda:
        raise ValueError("sparse_prefill_attn requires CUDA tensors")
    if any(tensor.device != dev for tensor in (k, v, selected)):
        raise ValueError("q, k, v, and selected must share one CUDA device")
    if q.stride(3) != 1 or k.stride(3) != 1 or v.stride(3) != 1:
        raise ValueError("contiguous last dim required")
    if selected.stride(2) != 1:
        raise ValueError("selected must have a contiguous last dim")
    if (
        q.dtype != torch.bfloat16
        or k.dtype != torch.bfloat16
        or v.dtype != torch.bfloat16
    ):
        raise ValueError("bf16 tensors required")
    if selected.dtype != torch.int32:
        raise ValueError("selected must be int32")
    # The internal causal indexer constructs every nonnegative id from a
    # completed block or the current sequence tail, so it can call with
    # validate_selected=False. Public and injected selections retain this
    # strict preflight; the kernel masks invalid ids on either path.
    if validate_selected:
        max_selected = int(selected.max().item()) if selected.numel() else -1
        if max_selected >= T:
            raise ValueError(
                f"selected index {max_selected} is outside KV sequence length {T}"
            )

    from .sparse_prefill_sm90 import try_sm90_sparse_prefill

    result = try_sm90_sparse_prefill(q, k, v, selected, blk_k=blk_k)
    if result is not None:
        return result

    from .sparse_prefill_bitmap import try_bitmap_prefill

    result = try_bitmap_prefill(q, k, v, selected)
    if result is not None:
        return result

    out = torch.empty_like(q)

    def _s(name, t):
        return t.stride(0), t.stride(1), t.stride(2)

    stride_q_b, stride_q_h, stride_q_s = _s("q", q)
    stride_k_b, stride_k_h, stride_k_t = _s("k", k)
    stride_v_b, stride_v_h, stride_v_t = _s("v", v)
    stride_idx_b, stride_idx_s = selected.stride(0), selected.stride(1)

    grid = (B, H_q, S)
    grouped = (
        ((S >= 128 and K >= 256 and q_per_kv >= 3))
        and D >= 32
        and blk_k >= 32
        and q_per_kv <= 16
    )
    capability = None
    if grouped and torch.version.hip is None:
        capability = torch.cuda.get_device_capability(dev)
    grouped = grouped and capability is not None and capability[0] >= 8
    kernel = (
        _sparse_prefill_grouped_kernel if grouped else (_sparse_prefill_online_kernel)
    )
    if grouped:
        grid = (B, H_kv, S)
        # The FP32 PV product needs three tensor-core products. A 32-key tile
        # fits the shared-memory budget and avoids padding work for TP8's
        # three query heads per KV head.
        blk_k = 32
    # This is the measured SM120 prefill support domain. It selects the
    # adopted implementation without exposing a runtime tuning switch.
    sm120_gqa_pv = (
        grouped and capability == (12, 0) and D == 256 and q_per_kv in (6, 12)
    )
    grouped_options = {"BLK_H": 16, "SM120_GQA_PV": sm120_gqa_pv} if grouped else {}
    if sm120_gqa_pv:
        grouped_options["num_stages"] = 1
    kernel[grid](
        q,
        k,
        v,
        selected,
        out,
        stride_q_b,
        stride_q_h,
        stride_q_s,
        stride_k_b,
        stride_k_h,
        stride_k_t,
        stride_v_b,
        stride_v_h,
        stride_v_t,
        stride_idx_b,
        stride_idx_s,
        K,
        T,
        D=D,
        SCALE=1.0 / float(D) ** 0.5,
        q_per_kv=q_per_kv,
        MAX_BLKS=triton.cdiv(K, blk_k),
        BLK_K=blk_k,
        **grouped_options,
    )
    return out


# ---------------------------------------------------------------------------
# Torch reference (for validation)
# ---------------------------------------------------------------------------
