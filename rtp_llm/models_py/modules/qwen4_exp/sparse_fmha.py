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
def _sparse_prefill_kernel(
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

    # Load q ----------------------------------------------------------------
    q = tl.load(
        q_ptr + pid_b * stride_q_b + pid_h * stride_q_h + pid_s * stride_q_s + rd
    ).to(tl.float32)

    idx_base = idx_ptr + pid_b * stride_idx_b + pid_s * stride_idx_s
    k_base = k_ptr + pid_b * stride_k_b + h_kv * stride_k_h
    v_base = v_ptr + pid_b * stride_v_b + h_kv * stride_v_h

    # Two-pass online softmax: first pass gets max(score),
    # second pass accumulates exp(score - max) and exp*V.
    max_score = tl.full((1,), float("-inf"), dtype=tl.float32)
    # Use a constant upper bound (MAX_BLKS) and mask out-of-range iterations.
    for blk in tl.static_range(MAX_BLKS):
        tr = blk * BLK_K + tl.arange(0, BLK_K)
        in_range = tr < K
        idx = tl.load(idx_base + tr, mask=in_range, other=-1)
        valid = in_range & (idx >= 0) & (idx < T)
        k_off = idx[:, None] * stride_k_t + rd[None, :]
        k_chunk = tl.load(
            k_base + k_off,
            mask=(in_range[:, None]) & (valid[:, None]),
            other=0.0,
        ).to(tl.float32)
        scores = tl.sum(q[None, :] * k_chunk, axis=1) * SCALE
        scores = tl.where(valid, scores, float("-inf"))
        blk_max = tl.max(scores, axis=0)
        max_score = tl.maximum(max_score, blk_max)

    # Second pass: accumulate exp(score - max) and exp*V.
    sum_exp = tl.zeros((1,), dtype=tl.float32)
    acc_o = tl.zeros((D,), dtype=tl.float32)

    for blk in tl.static_range(MAX_BLKS):
        tr = blk * BLK_K + tl.arange(0, BLK_K)
        in_range = tr < K
        idx = tl.load(idx_base + tr, mask=in_range, other=-1)
        valid = in_range & (idx >= 0) & (idx < T)
        col_mask = (in_range[:, None]) & (valid[:, None])

        k_off = idx[:, None] * stride_k_t + rd[None, :]
        k_chunk = tl.load(k_base + k_off, mask=col_mask, other=0.0).to(tl.float32)
        scores = tl.sum(q[None, :] * k_chunk, axis=1) * SCALE
        scores = tl.where(valid, scores, float("-inf"))

        v_off = idx[:, None] * stride_v_t + rd[None, :]
        v_chunk = tl.load(v_base + v_off, mask=col_mask, other=0.0).to(tl.float32)

        w = tl.exp(scores - max_score)
        sum_exp += tl.sum(w)
        acc_o += tl.sum(w[:, None] * v_chunk, axis=0)

    # Normalise; avoid division by zero when all idx are -1.
    out = tl.where(sum_exp > 0, acc_o / sum_exp, 0.0)

    tl.store(
        out_ptr + pid_b * stride_q_b + pid_h * stride_q_h + pid_s * stride_q_s + rd,
        out.to(q_ptr.dtype.element_ty),
    )


def sparse_prefill_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    selected: torch.Tensor,
    *,
    blk_k: int = _BLK_K,
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
    # Correctness-first synchronization: a bad positive index is a metadata
    # contract violation, not padding, and must not become an OOB device read.
    max_selected = int(selected.max().item()) if selected.numel() else -1
    if max_selected >= T:
        raise ValueError(
            f"selected index {max_selected} is outside KV sequence length {T}"
        )

    out = torch.empty_like(q)

    def _s(name, t):
        return t.stride(0), t.stride(1), t.stride(2)

    stride_q_b, stride_q_h, stride_q_s = _s("q", q)
    stride_k_b, stride_k_h, stride_k_t = _s("k", k)
    stride_v_b, stride_v_h, stride_v_t = _s("v", v)
    stride_idx_b, stride_idx_s = selected.stride(0), selected.stride(1)

    grid = (B, H_q, S)
    _sparse_prefill_kernel[grid](
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
    )
    return out


# ---------------------------------------------------------------------------
# Torch reference (for validation)
# ---------------------------------------------------------------------------
def sparse_prefill_attn_torch_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    selected: torch.Tensor,
) -> torch.Tensor:
    """Matches :func:`sparse_prefill_attn` up to fp rounding."""
    from rtp_llm.models_py.modules.qwen4_exp.sparse_attention import gather_and_attend

    return gather_and_attend(q, k, v, selected)
