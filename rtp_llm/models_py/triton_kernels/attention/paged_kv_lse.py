"""Natural-log LSE over paged/ragged KV caches (Triton).

Backends that return attention output but no usable LSE (their internal
log-space semantics are undocumented and shape-dependent) can still apply
per-head attention sinks post-hoc — ``y' = y * sigmoid(lse - sink_h)`` —
using this module's own natural-log logsumexp. The KV layout is the
RTP-LLM paged format ``[block_num, 2, num_kv_heads, page_size, head_dim]``
with per-sequence page tables (indptr/indices/last_page_len).

Three entries:
  - paged_kv_lse: decode — each query row attends its sequence's full KV.
  - paged_kv_lse_prefill: causal prefill — row r in seq s attends the
    first ``cutoff(r)`` KV tokens (prefix + row's own position + 1).
  - ragged_kv_lse: causal over contiguous K/V (the hybrid impl's ragged
    section), same cutoff semantics.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _paged_kv_lse_kernel(
    q_ptr,  # [rows, H, D]
    kv_ptr,  # [blocks, 2, H_kv, PS, D]
    kv_indptr_ptr,  # [rows + 1] page span per row
    indices_ptr,  # [total_pages]
    last_len_ptr,  # [rows] valid tokens in the row's last page
    lse_ptr,  # [rows, H]
    sm_scale,
    H: tl.constexpr,
    H_KV: tl.constexpr,
    D: tl.constexpr,
    PS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)

    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < D
    q = tl.load(
        q_ptr + row * H * D + head * D + offs_d, mask=d_mask, other=0.0
    ).to(tl.float32)

    start = tl.load(kv_indptr_ptr + row)
    end = tl.load(kv_indptr_ptr + row + 1)
    n_pages = end - start

    kv_head = head * H_KV // H

    m = -1e30
    d_sum = 0.0

    offs_t = tl.arange(0, PS)
    for i in range(0, n_pages):
        page = tl.load(indices_ptr + start + i)
        tokens = PS
        if i == n_pages - 1:
            tokens = tl.load(last_len_ptr + row)
        t_mask = offs_t < tokens
        k_base = (page * 2 + 0) * H_KV * PS * D + kv_head * PS * D
        k = tl.load(
            kv_ptr + k_base + offs_t[:, None] * D + offs_d[None, :],
            mask=t_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        logits = tl.sum(q[None, :] * k, axis=1) * sm_scale
        logits = tl.where(t_mask, logits, -1e30)
        m_page = tl.max(logits)
        m_new = tl.maximum(m, m_page)
        d_sum = d_sum * tl.exp(m - m_new) + tl.sum(tl.exp(logits - m_new))
        m = m_new

    lse = m + tl.log(d_sum)
    tl.store(lse_ptr + row * H + head, lse)


@triton.jit
def _paged_kv_lse_prefill_kernel(
    q_ptr,
    kv_ptr,
    row_page_start_ptr,  # [rows] page span start (from the row's seq)
    row_page_end_ptr,  # [rows] page span end
    row_tokens_ptr,  # [rows] total KV tokens of the row's seq
    row_cutoff_ptr,  # [rows] causal cutoff token count for this row
    indices_ptr,
    lse_ptr,
    sm_scale,
    H: tl.constexpr,
    H_KV: tl.constexpr,
    D: tl.constexpr,
    PS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)

    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < D
    q = tl.load(
        q_ptr + row * H * D + head * D + offs_d, mask=d_mask, other=0.0
    ).to(tl.float32)

    start = tl.load(row_page_start_ptr + row)
    end = tl.load(row_page_end_ptr + row)
    cutoff = tl.load(row_cutoff_ptr + row)
    kv_head = head * H_KV // H

    m = -1e30
    d_sum = 0.0

    offs_t = tl.arange(0, PS)
    remaining = cutoff
    for i in range(0, end - start):
        if remaining <= 0:
            break
        page = tl.load(indices_ptr + start + i)
        tokens = tl.minimum(remaining, PS)
        t_mask = offs_t < tokens
        k_base = (page * 2 + 0) * H_KV * PS * D + kv_head * PS * D
        k = tl.load(
            kv_ptr + k_base + offs_t[:, None] * D + offs_d[None, :],
            mask=t_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        logits = tl.sum(q[None, :] * k, axis=1) * sm_scale
        logits = tl.where(t_mask, logits, -1e30)
        m_page = tl.max(logits)
        m_new = tl.maximum(m, m_page)
        d_sum = d_sum * tl.exp(m - m_new) + tl.sum(tl.exp(logits - m_new))
        m = m_new
        remaining -= PS

    tl.store(lse_ptr + row * H + head, m + tl.log(d_sum))


@triton.jit
def _ragged_kv_lse_kernel(
    q_ptr,
    k_ptr,  # [total_kv, H_kv, D]
    row_kv_start_ptr,  # [rows] kv span start (row's seq)
    row_cutoff_ptr,  # [rows] causal cutoff within the span (pos + 1)
    lse_ptr,
    sm_scale,
    H: tl.constexpr,
    H_KV: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)

    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < D
    q = tl.load(
        q_ptr + row * H * D + head * D + offs_d, mask=d_mask, other=0.0
    ).to(tl.float32)

    start = tl.load(row_kv_start_ptr + row)
    cutoff = tl.load(row_cutoff_ptr + row)
    kv_head = head * H_KV // H

    m = -1e30
    d_sum = 0.0

    offs_n = tl.arange(0, BLOCK_N)
    for base in range(0, cutoff, BLOCK_N):
        n_mask = (base + offs_n) < cutoff
        k = tl.load(
            k_ptr
            + (start + base + offs_n[:, None]) * H_KV * D
            + kv_head * D
            + offs_d[None, :],
            mask=n_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        logits = tl.sum(q[None, :] * k, axis=1) * sm_scale
        logits = tl.where(n_mask, logits, -1e30)
        m_page = tl.max(logits)
        m_new = tl.maximum(m, m_page)
        d_sum = d_sum * tl.exp(m - m_new) + tl.sum(tl.exp(logits - m_new))
        m = m_new

    tl.store(lse_ptr + row * H + head, m + tl.log(d_sum))


def _row_seq_map(qo_indptr: torch.Tensor, device) -> torch.Tensor:
    rows = int(qo_indptr[-1])
    seqs = qo_indptr.numel() - 1
    row_seq = torch.empty(rows, dtype=torch.int32, device=device)
    for s in range(seqs):
        row_seq[int(qo_indptr[s]):int(qo_indptr[s + 1])] = s
    return row_seq


def paged_kv_lse(
    q, paged_kv_cache, kv_indptr, kv_indices, kv_last_page_len,
    num_qo_heads, num_kv_heads, head_dim, page_size, sm_scale,
):
    rows = q.shape[0]
    lse = torch.empty((rows, num_qo_heads), dtype=torch.float32, device=q.device)
    if rows == 0:
        return lse
    BLOCK_D = triton.next_power_of_2(head_dim)
    _paged_kv_lse_kernel[(rows, num_qo_heads)](
        q, paged_kv_cache, kv_indptr, kv_indices, kv_last_page_len, lse,
        sm_scale,
        H=num_qo_heads, H_KV=num_kv_heads, D=head_dim, PS=page_size,
        BLOCK_D=BLOCK_D, num_warps=1, num_stages=2,
    )
    return lse


def paged_kv_lse_prefill(
    q, paged_kv_cache, qo_indptr, kv_indptr, kv_indices, kv_last_page_len,
    num_qo_heads, num_kv_heads, head_dim, page_size, sm_scale,
):
    """Causal prefill: row r of seq s attends the first
    ``kv_tokens(s) - (seq_q_len(s) - 1 - pos(r))`` KV tokens, i.e. everything
    written up to and including its own position."""
    rows = q.shape[0]
    lse = torch.empty((rows, num_qo_heads), dtype=torch.float32, device=q.device)
    if rows == 0:
        return lse
    row_seq = _row_seq_map(qo_indptr, q.device)
    kv_indptr = kv_indptr.to(torch.int32)
    row_page_start = kv_indptr[row_seq]
    row_page_end = kv_indptr[row_seq + 1]
    n_pages_per_seq = (row_page_end - row_page_start).clamp(min=1)
    kv_tokens = (
        (n_pages_per_seq - 1) * page_size
        + kv_last_page_len.to(torch.int32)[row_seq]
    )
    # cutoff = tokens up to the row's own position within its sequence
    row_offsets = torch.arange(rows, device=q.device, dtype=torch.int32) - qo_indptr[
        row_seq
    ].to(torch.int32)
    seq_q_len = (qo_indptr[1:] - qo_indptr[:-1]).to(torch.int32)[row_seq]
    cutoff = kv_tokens - (seq_q_len - 1 - row_offsets)
    BLOCK_D = triton.next_power_of_2(head_dim)
    _paged_kv_lse_prefill_kernel[(rows, num_qo_heads)](
        q, paged_kv_cache,
        row_page_start.contiguous(), row_page_end.contiguous(),
        kv_tokens.contiguous(), cutoff.contiguous(),
        kv_indices, lse, sm_scale,
        H=num_qo_heads, H_KV=num_kv_heads, D=head_dim, PS=page_size,
        BLOCK_D=BLOCK_D, num_warps=1, num_stages=2,
    )
    return lse


def ragged_kv_lse(
    q, k, qo_indptr, kv_indptr,
    num_qo_heads, num_kv_heads, head_dim, sm_scale,
):
    """Causal over the contiguous ragged K (hybrid impl's new-token section):
    row r of seq s attends the first pos(r)+1 new tokens of its sequence."""
    rows = q.shape[0]
    lse = torch.empty((rows, num_qo_heads), dtype=torch.float32, device=q.device)
    if rows == 0:
        return lse
    row_seq = _row_seq_map(qo_indptr, q.device)
    kv_indptr = kv_indptr.to(torch.int32)
    row_kv_start = kv_indptr[row_seq]
    row_offsets = torch.arange(rows, device=q.device, dtype=torch.int32) - qo_indptr[
        row_seq
    ].to(torch.int32)
    cutoff = (row_offsets + 1).clamp(min=0)
    BLOCK_D = triton.next_power_of_2(head_dim)
    BLOCK_N = 128
    _ragged_kv_lse_kernel[(rows, num_qo_heads)](
        q, k, row_kv_start.contiguous(), cutoff.contiguous(), lse, sm_scale,
        H=num_qo_heads, H_KV=num_kv_heads, D=head_dim,
        BLOCK_D=BLOCK_D, BLOCK_N=BLOCK_N, num_warps=1, num_stages=2,
    )
    return lse
