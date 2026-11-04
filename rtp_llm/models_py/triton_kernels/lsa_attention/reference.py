"""PyTorch reference implementation for LongCat-Flash-Lite-Sparse LSA attention.

This module is the semantic oracle for the Triton kernels in this package. It
implements the three stages of the LongCat Sparse Attention (LSA) operator:

1. Lightning-indexer scoring (DSA-style, MQA, ReLU)::

       I[t, s] = sum_j w[t, j] * ReLU(q_idx[t, j] . k_idx[s])

2. Streaming-Aware Indexing (SI) budget selection. Mirrors SGLang
   ``Indexer._mask_init_and_local_tokens`` followed by ``topk_transform``:
   sink (``index_init_tokens``) and sliding-window (``index_local_tokens``)
   positions are forced into the budget, and the remaining budget is filled by
   a global top-k over the scores. The total budget is ``index_topk``
   (K = K_sink + K_swa + K_sparse).

3. Sparse MLA attention in absorbed MQA mode over the selected slots::

       q_abs = [q_nope @ W_uk ; q_rope]      (per head, kv_lora_rank + rope)
       k     = [c_kv ; k_rope]               (shared across heads)
       v     = c_kv
       o_h   = softmax(q_abs_h . k^T * scale) @ v

All functions take and return plain torch tensors so they can run on CPU (unit
tests) and GPU (kernel validation). Shapes follow the RTP FlashMLA sparse
contract (``flashmla_sparse_impl.py``): q ``[T, H, kv_lora + rope]``, kv
``[N, kv_lora + rope]``, indices ``[T, 1, topk]`` or ``[T, topk]``, out
``[T, H, kv_lora]`` plus natural-log LSE ``[T, H]``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

DEFAULT_INDEX_TOPK = 2048
DEFAULT_INIT_TOKENS = 16
DEFAULT_LOCAL_TOKENS = 1024
DEFAULT_KV_LORA_RANK = 512


def indexer_scores(
    q_idx: torch.Tensor,
    k_idx: torch.Tensor,
    weights: torch.Tensor,
    *,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Lightning-indexer scores.

    Args:
        q_idx: ``[T, HI, DI]`` indexer query vectors (one per indexer head).
        k_idx: ``[N, DI]`` shared indexer keys (MQA, single key per token).
        weights: ``[T, HI]`` learned per-head weights.
        scale: dot-product scale; defaults to ``1 / sqrt(DI)``.

    Returns:
        ``[T, N]`` float32 scores. Row ``t`` may only attend to ``s <= t``;
        causal masking is applied by :func:`select_slots` via ``valid_lens``.
    """
    if scale is None:
        scale = q_idx.shape[-1] ** -0.5
    dots = torch.einsum("thd,nd->thn", q_idx.float(), k_idx.float()) * scale
    return (torch.relu(dots) * weights.float().unsqueeze(-1)).sum(dim=1)


def select_slots(
    scores: torch.Tensor,
    valid_lens: torch.Tensor,
    *,
    topk: int = DEFAULT_INDEX_TOPK,
    init_tokens: int = DEFAULT_INIT_TOKENS,
    local_tokens: int = DEFAULT_LOCAL_TOKENS,
) -> torch.Tensor:
    """SI budget selection returning the top-k slot index set per query row.

    Sink positions ``[0, init_tokens)`` and local positions
    ``[len - local_tokens, len)`` are set to ``+inf`` before the global top-k,
    so the fixed streaming part consumes part of the shared ``topk`` budget.
    Note: ties (including the ``+inf`` anchors) are resolved by the underlying
    top-k implementation; callers must treat the returned set as unordered.

    Args:
        scores: ``[T, N]`` indexer scores.
        valid_lens: ``[T]`` number of valid keys per row (``t + 1`` in prefill).
        topk: total attention budget (``index_topk``).
        init_tokens: sink size (``index_init_tokens``).
        local_tokens: sliding-window size (``index_local_tokens``).

    Returns:
        ``[T, min(topk, N)]`` int64 slot indices, sorted by descending score,
        with ``-1`` marking padding for rows whose valid length is below the
        budget.
    """
    if scores.dim() != 2:
        raise ValueError(f"scores must be [T, N], got {tuple(scores.shape)}")
    if valid_lens.dim() != 1 or valid_lens.shape[0] != scores.shape[0]:
        raise ValueError("valid_lens must be [T] matching scores")

    t_count, n_count = scores.shape
    device = scores.device
    masked = scores.float().clone()

    cols = torch.arange(n_count, device=device)[None, :]
    valid = cols < valid_lens.to(device)[:, None]
    masked = masked.masked_fill(~valid, float("-inf"))

    if init_tokens > 0:
        init_idx = torch.arange(init_tokens, device=device)[None, :].expand(t_count, -1)
        init_idx = init_idx.clamp_max(n_count - 1)
        masked.scatter_(1, init_idx, float("inf"))

    if local_tokens > 0:
        last = (valid_lens.to(device) - 1).clamp_min(0)[:, None]
        local_idx = last - torch.arange(local_tokens, device=device)[None, :]
        local_idx = local_idx.clamp_min(0)
        masked.scatter_(1, local_idx, float("inf"))

    k = min(topk, n_count)
    values, indices = torch.topk(masked, k, dim=1)
    indices = indices.masked_fill(values == float("-inf"), -1)
    return indices


def absorb_queries(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    w_uk: torch.Tensor,
) -> torch.Tensor:
    """Absorbed MLA query construction: ``[q_nope @ W_uk ; q_rope]``.

    Args:
        q_nope: ``[T, H, qk_nope_head_dim]``.
        q_rope: ``[T, H, qk_rope_head_dim]``.
        w_uk: ``[H, qk_nope_head_dim, kv_lora_rank]`` per-head up-projection.

    Returns:
        ``[T, H, kv_lora_rank + qk_rope_head_dim]`` absorbed queries.
    """
    absorbed = torch.einsum("thd,hdk->thk", q_nope.float(), w_uk.float())
    return torch.cat([absorbed, q_rope.float()], dim=-1)


def sparse_mla_attention(
    q_abs: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    *,
    scale: Optional[float] = None,
    kv_lora_rank: int = DEFAULT_KV_LORA_RANK,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Vectorized reference of absorbed-MQA sparse MLA attention.

    Args:
        q_abs: ``[T, H, DQ]`` absorbed queries (``DQ = kv_lora_rank + rope``).
        kv: ``[N, DQ]`` latent KV: ``[c_kv ; k_rope]``.
        indices: ``[T, 1, S]`` or ``[T, S]`` slot indices; ``-1`` = padding.
        scale: softmax scale; defaults to ``1 / sqrt(DQ)``.
        kv_lora_rank: width of ``c_kv`` / attention output.

    Returns:
        ``(out, lse)`` with ``out`` in ``q_abs.dtype`` of shape
        ``[T, H, kv_lora_rank]`` and float32 natural-log LSE ``[T, H]``.
        Rows with no valid slot produce zeros and ``-inf`` LSE.
    """
    if q_abs.dim() != 3:
        raise ValueError(f"q_abs must be [T, H, DQ], got {tuple(q_abs.shape)}")
    if kv.dim() != 2 or kv.shape[1] != q_abs.shape[-1]:
        raise ValueError("kv must be [N, DQ] matching q_abs' last dim")
    if indices.dim() == 3:
        if indices.shape[1] != 1:
            raise ValueError("indices with 3 dims must be [T, 1, S] (MQA broadcast)")
        indices = indices[:, 0, :]
    if indices.dim() != 2 or indices.shape[0] != q_abs.shape[0]:
        raise ValueError("indices must be [T, S] or [T, 1, S]")

    if scale is None:
        scale = q_abs.shape[-1] ** -0.5

    t_count, heads, _ = q_abs.shape
    n_count = kv.shape[0]
    slots = indices.shape[1]

    safe = indices.clamp_min(0).clamp_max(max(n_count - 1, 0))
    gathered = kv[safe]  # [T, S, DQ]
    valid = (indices >= 0) & (indices < n_count)  # [T, S]

    logits = torch.einsum("thd,tsd->ths", q_abs.float(), gathered.float()) * scale
    logits = logits.masked_fill(~valid[:, None, :], float("-inf"))
    lse = torch.logsumexp(logits, dim=-1)  # [T, H]
    probs = torch.softmax(logits, dim=-1)
    out = torch.einsum("ths,tsd->thd", probs, gathered[:, :, :kv_lora_rank].float())
    out = out.to(q_abs.dtype)
    # Empty rows (no valid slot) yield NaN softmax; normalize to zeros.
    empty = lse == float("-inf")
    out = out.masked_fill(empty[:, :, None], 0.0)
    return out, lse


def sparse_mla_attention_naive(
    q_abs: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    *,
    scale: Optional[float] = None,
    kv_lora_rank: int = DEFAULT_KV_LORA_RANK,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Scalar-loop reference used to cross-check :func:`sparse_mla_attention`.

    Deliberately unoptimized (explicit python loops) so that a bug in the
    vectorized path cannot hide behind a shared implementation detail.
    """
    if indices.dim() == 3:
        indices = indices[:, 0, :]
    if scale is None:
        scale = q_abs.shape[-1] ** -0.5

    t_count, heads, dq = q_abs.shape
    n_count = kv.shape[0]
    slots = indices.shape[1]
    out = torch.zeros(t_count, heads, kv_lora_rank, dtype=torch.float32)
    lse = torch.full((t_count, heads), float("-inf"), dtype=torch.float32)

    for t in range(t_count):
        for h in range(heads):
            best = float("-inf")
            acc = [0.0] * kv_lora_rank
            total = 0.0
            found = False
            for s in range(slots):
                slot = int(indices[t, s].item())
                if slot < 0 or slot >= n_count:
                    continue
                found = True
                dot = 0.0
                for d in range(dq):
                    dot += float(q_abs[t, h, d].float().item()) * float(kv[slot, d].float().item())
                logit = dot * scale
                if logit > best:
                    # renormalize previously accumulated terms
                    ratio = 1.0 if best == float("-inf") else float(torch.exp(torch.tensor(best - logit)))
                    total = total * ratio + 1.0
                    acc = [x * ratio for x in acc]
                    for d in range(kv_lora_rank):
                        acc[d] += float(kv[slot, d].float().item())
                    best = logit
                else:
                    w = float(torch.exp(torch.tensor(logit - best)))
                    total += w
                    for d in range(kv_lora_rank):
                        acc[d] += w * float(kv[slot, d].float().item())
            if found and total > 0.0:
                lse[t, h] = best + float(torch.log(torch.tensor(total)))
                for d in range(kv_lora_rank):
                    out[t, h, d] = acc[d] / total
    return out.to(q_abs.dtype), lse


def lsa_forward(
    q_abs: torch.Tensor,
    kv: torch.Tensor,
    q_idx: torch.Tensor,
    k_idx: torch.Tensor,
    weights: torch.Tensor,
    valid_lens: torch.Tensor,
    *,
    topk: int = DEFAULT_INDEX_TOPK,
    init_tokens: int = DEFAULT_INIT_TOKENS,
    local_tokens: int = DEFAULT_LOCAL_TOKENS,
    kv_lora_rank: int = DEFAULT_KV_LORA_RANK,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """End-to-end LSA reference: scores -> SI selection -> sparse MLA.

    Returns ``(out, lse, indices)`` where ``indices`` is the selected
    ``[T, topk]`` slot set (useful for kernel-level A/B checks).
    """
    scores = indexer_scores(q_idx, k_idx, weights)
    indices = select_slots(
        scores,
        valid_lens,
        topk=topk,
        init_tokens=init_tokens,
        local_tokens=local_tokens,
    )
    out, lse = sparse_mla_attention(
        q_abs, kv, indices, kv_lora_rank=kv_lora_rank
    )
    return out, lse, indices


# ---------------------------------------------------------------------------
# FP8 KV helpers (e4m3 with per-128-block amax scales)
# ---------------------------------------------------------------------------

FP8_E4M3_MAX = 448.0
FP8_KV_BLOCK = 128


def quantize_kv_fp8(
    kv: torch.Tensor, block: int = FP8_KV_BLOCK
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize a ``[N, DQ]`` latent KV buffer to e4m3 with per-block scales.

    The rope slice (``DQ - kv_lora_rank`` columns) occupies its own trailing
    scale block, so a row of 576 dims yields ``ceil(576 / 128) = 5`` scales.

    Returns:
        ``(kv_fp8 [N, DQ] float8_e4m3fn, scales [N, NB] float32)``.
    """
    if kv.dim() != 2:
        raise ValueError(f"kv must be [N, DQ], got {tuple(kv.shape)}")
    n_count, dq = kv.shape
    # The rope slice makes DQ (576 = 4 x 128 + 64) ragged against the block, so
    # the last block is zero-padded for the amax and sliced back afterwards.
    nb = (dq + block - 1) // block
    padded = torch.nn.functional.pad(kv.float(), (0, nb * block - dq))
    shaped = padded.reshape(n_count, nb, block)
    amax = shaped.abs().amax(dim=-1).clamp_min(1e-6)
    scales = (amax / FP8_E4M3_MAX).to(torch.float32)
    quantized = (shaped / scales[:, :, None]).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)
    quantized = quantized.reshape(n_count, nb * block)[:, :dq]
    return quantized.to(torch.float8_e4m3fn), scales


def dequantize_kv_fp8(
    kv_fp8: torch.Tensor, scales: torch.Tensor, block: int = FP8_KV_BLOCK
) -> torch.Tensor:
    """Inverse of :func:`quantize_kv_fp8` in float32."""
    n_count, dq = kv_fp8.shape
    nb = (dq + block - 1) // block
    if scales.shape != (n_count, nb):
        raise ValueError(f"scales must be [{n_count}, {nb}], got {tuple(scales.shape)}")
    scale_cols = scales.repeat_interleave(block, dim=1)[:, :dq]
    return kv_fp8.float() * scale_cols
