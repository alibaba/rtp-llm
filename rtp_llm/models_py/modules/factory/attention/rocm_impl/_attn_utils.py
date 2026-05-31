"""Pure-torch helpers for ROCm attention impls.

Kept free of ``aiter`` / kernel imports so unit tests can exercise the layout
math on CPU without a ROCm wheel installed. All functions in this module are
geometric tensor manipulations (split / reshape / gather) — the actual
attention kernels live in ``aiter.py``.
"""

import torch


def compute_kv_unpad_indices(cu_seqlens_k, total_kv_tokens=None):
    """Compute per-token ``(batch_idx, pos_idx)`` gather indices used by
    :func:`unpad_kv_vectorized` to scatter ``[B, H_kv, max_seqlen_k, D]``
    padded K/V into packed ``[total_kv, H_kv, D]``.

    ``cu_seqlens_k`` is a ``[B+1]`` int tensor. The per-token ``batch_idx`` is
    recovered via ``searchsorted`` on ``cu_seqlens_k`` (data-independent
    output shape — no host sync), and ``pos_idx`` is the residual within each
    batch.

    ``total_kv_tokens`` (Python int) sizes the per-token index buffers. The
    hot caller (``AiterPrefillAttnOp.prepare``) already has
    ``fmha_params.token_kv_num`` and should pass it; otherwise we read
    ``cu_seqlens_k[-1]``, which forces one host-device sync. The previous
    ``repeat_interleave(arange, kv_lengths)`` formulation always synced (its
    output length is the GPU-resident sum), which serialised this op against
    the CPU stream.

    The indices depend only on the per-batch sequence-length layout — not on
    K/V data — so the prefill op materialises them on ``FMHAParams`` once per
    request and reuses them for every attention layer.
    """
    device = cu_seqlens_k.device
    if total_kv_tokens is None:
        total_kv_tokens = int(cu_seqlens_k[-1])

    pos_lin = torch.arange(total_kv_tokens, device=device, dtype=cu_seqlens_k.dtype)
    batch_idx = torch.searchsorted(cu_seqlens_k, pos_lin, right=True) - 1
    pos_idx = pos_lin - cu_seqlens_k[batch_idx]
    return batch_idx.long(), pos_idx.long()


def unpad_kv_vectorized(k_padded, v_padded, batch_idx_long, pos_idx_long):
    """Gather ``[B, H_kv, max_seqlen_k, D]`` padded K/V into packed
    ``[total_kv, H_kv, D]`` using precomputed per-token indices from
    :func:`compute_kv_unpad_indices`.
    """
    key_packed = k_padded[batch_idx_long, :, pos_idx_long, :].contiguous()
    value_packed = v_padded[batch_idx_long, :, pos_idx_long, :].contiguous()
    return key_packed, value_packed


def split_qkv_fp8(qkv_fp8, head_num, head_num_kv, head_dim):
    """Split a packed FP8 QKV buffer ``[token_num, (H_q + 2*H_kv) * D]`` into
    separate Q / K / V views ``[token_num, H_q, D]`` / ``[token_num, H_kv, D]``.

    The C++ FP8 path returns Q, K, V concatenated along the last dim; this is
    the inverse split used by ``flash_attn_varlen_fp8_pertensor_func``. The
    returned tensors are views into ``qkv_fp8`` — no copy.
    """
    token_num = qkv_fp8.shape[0]
    qkv_reshaped = qkv_fp8.reshape(token_num, head_num + 2 * head_num_kv, head_dim)
    query = qkv_reshaped[:, :head_num, :]
    key = qkv_reshaped[:, head_num : head_num + head_num_kv, :]
    value = qkv_reshaped[:, head_num + head_num_kv : head_num + 2 * head_num_kv, :]
    return query, key, value


def split_raw_qkv(qkv, head_num, head_num_kv, head_dim, token_q_num, token_kv_num):
    """Split a flat concatenated QKV tensor ``[token_num, (H_q + 2*H_kv) * D]``
    into separate Q / K / V tensors and slice to the actual token counts.

    Used by the ``kv_cache is None`` path (encoder-only models, e.g. BERT)
    where Q and K/V may have different active token counts (token_q_num /
    token_kv_num). Returns views (not contiguous copies): ``flash_attn_varlen_func``
    only requires ``stride(-1) == 1`` which the views already satisfy after
    ``split + view``, so forcing ``.contiguous()`` would emit redundant
    ``direct_copy`` kernels per layer (3 × ~19µs on bs=64 visionbert).
    """
    token_num = qkv.size(0)
    q_size = head_num * head_dim
    kv_size = head_num_kv * head_dim
    query, key, value = torch.split(qkv, [q_size, kv_size, kv_size], dim=-1)
    query = query.view(token_num, head_num, head_dim)[:token_q_num]
    key = key.view(token_num, head_num_kv, head_dim)[:token_kv_num]
    value = value.view(token_num, head_num_kv, head_dim)[:token_kv_num]
    assert query.stride(-1) == key.stride(-1) == value.stride(-1) == 1
    return query, key, value
