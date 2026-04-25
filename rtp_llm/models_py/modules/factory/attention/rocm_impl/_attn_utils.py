"""Pure-torch helpers for ROCm attention impls.

Kept free of ``aiter`` / kernel imports so unit tests can exercise the layout
math on CPU without a ROCm wheel installed. All functions in this module are
geometric tensor manipulations (split / reshape / gather) — the actual
attention kernels live in ``aiter.py``.
"""

import torch


def unpad_kv_vectorized(k_padded, v_padded, cu_seqlens_k):
    """Gather ``[B, H_kv, max_seqlen_k, D]`` padded K/V into packed
    ``[total_kv, H_kv, D]`` without Python-side iteration.

    ``cu_seqlens_k`` is a ``[B+1]`` int tensor on the same device as the K/V
    buffers. The function builds per-token ``(batch_idx, pos_idx)`` on device
    via ``repeat_interleave`` + ``arange`` and gathers via advanced indexing —
    one kernel chain, no ``.item()`` per batch.
    """
    device = k_padded.device
    batch_size = cu_seqlens_k.shape[0] - 1
    kv_lengths = cu_seqlens_k[1:] - cu_seqlens_k[:-1]

    batch_idx = torch.repeat_interleave(
        torch.arange(batch_size, device=device, dtype=cu_seqlens_k.dtype),
        kv_lengths,
    )
    pos_idx = (
        torch.arange(batch_idx.shape[0], device=device, dtype=cu_seqlens_k.dtype)
        - cu_seqlens_k[batch_idx]
    )
    b_long = batch_idx.long()
    p_long = pos_idx.long()
    key_packed = k_padded[b_long, :, p_long, :].contiguous()
    value_packed = v_padded[b_long, :, p_long, :].contiguous()
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
    token_kv_num). Returns contiguous tensors so the downstream
    ``flash_attn_varlen_func`` sees clean memory.
    """
    token_num = qkv.size(0)
    q_size = head_num * head_dim
    kv_size = head_num_kv * head_dim
    query, key, value = torch.split(qkv, [q_size, kv_size, kv_size], dim=-1)
    query = query.view(token_num, head_num, head_dim)[:token_q_num]
    key = key.view(token_num, head_num_kv, head_dim)[:token_kv_num]
    value = value.view(token_num, head_num_kv, head_dim)[:token_kv_num]
    return query.contiguous(), key.contiguous(), value.contiguous()


def reshape_kv_cache_vectorized(
    kv_cache_base, head_num_kv, tokens_per_block, head_dim, v1_kv_layout
):
    """Reshape a packed 2D ``kv_cache_base`` into the 5D VECTORIZED_LAYOUT
    expected by ``mha_batch_prefill_func``.

    Returns ``(k_cache_5d, v_cache_5d)``::

        K: [num_blocks, H_kv, head_dim/vs, page_size, vs]
        V: [num_blocks, H_kv, page_size/vs, head_dim, vs]   (ASM)
        V: [num_blocks, H_kv, page_size/vs, head_dim, vs]   (V1, after permute)

    For the V1 path the kernel writes V in linear ``[hd, ps]`` layout
    (non-template ``getVLocalIdx``); we reshape and permute to match the ASM
    target. The K layout is already vectorized in both kernels and only needs
    a ``view``.
    """
    block_num = kv_cache_base.shape[0]
    hk = head_num_kv
    ps = tokens_per_block
    hd = head_dim
    vs = 16 // kv_cache_base.element_size()
    expected_elems = 2 * hk * ps * hd

    flat = kv_cache_base[:, :expected_elems].reshape(block_num, 2, hk, ps * hd)

    k_cache = flat[:, 0, :, :].view(block_num, hk, hd // vs, ps, vs)

    if v1_kv_layout:
        v_linear = flat[:, 1, :, :].view(block_num, hk, hd, ps)
        v_cache = (
            v_linear.reshape(block_num, hk, hd, ps // vs, vs)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )
    else:
        v_cache = flat[:, 1, :, :].view(block_num, hk, ps // vs, hd, vs)

    return k_cache, v_cache
