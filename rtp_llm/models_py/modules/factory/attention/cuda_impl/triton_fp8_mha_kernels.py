import triton
import triton.language as tl


@triton.jit
def _find_batch(cu_q, token_idx, batch_size):
    left = 0
    right = batch_size
    while left < right:
        mid = (left + right) // 2
        if tl.load(cu_q + mid + 1) <= token_idx:
            left = mid + 1
        else:
            right = mid
    return left


@triton.jit
def _triton_fp8_paged_mha_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    k_scale_ptr,
    v_scale_ptr,
    block_table_ptr,
    cu_q_ptr,
    seq_lens_ptr,
    out_ptr,
    total_tokens: tl.constexpr,
    batch_size: tl.constexpr,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_size: tl.constexpr,
    block_size: tl.constexpr,
    block_table_stride: tl.constexpr,
    q_stride_t: tl.constexpr,
    q_stride_h: tl.constexpr,
    q_stride_d: tl.constexpr,
    k_stride_b: tl.constexpr,
    k_stride_h: tl.constexpr,
    k_stride_s: tl.constexpr,
    k_stride_d: tl.constexpr,
    v_stride_b: tl.constexpr,
    v_stride_h: tl.constexpr,
    v_stride_s: tl.constexpr,
    v_stride_d: tl.constexpr,
    ks_stride_b: tl.constexpr,
    ks_stride_s: tl.constexpr,
    ks_stride_h: tl.constexpr,
    vs_stride_b: tl.constexpr,
    vs_stride_s: tl.constexpr,
    vs_stride_h: tl.constexpr,
    out_stride_t: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_d: tl.constexpr,
    softmax_scale: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
):
    token_idx = tl.program_id(0)
    q_head = tl.program_id(1)

    batch_idx = _find_batch(cu_q_ptr, token_idx, batch_size)
    q_start = tl.load(cu_q_ptr + batch_idx)
    q_end = tl.load(cu_q_ptr + batch_idx + 1)
    q_len = q_end - q_start
    local_q_pos = token_idx - q_start
    seq_len = tl.load(seq_lens_ptr + batch_idx)
    context_len = seq_len - q_len
    max_kv_pos = context_len + local_q_pos + 1

    kv_group_size = num_heads // num_kv_heads
    kv_head = q_head // kv_group_size
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    dim_mask = offs_d < head_size

    q = tl.load(
        q_ptr + token_idx * q_stride_t + q_head * q_stride_h + offs_d * q_stride_d,
        mask=dim_mask,
        other=0.0,
    ).to(tl.float32)

    m_i = tl.full((), -float("inf"), dtype=tl.float32)
    l_i = tl.full((), 0.0, dtype=tl.float32)
    acc = tl.zeros((HEAD_SIZE_PADDED,), dtype=tl.float32)
    offs_n = tl.arange(0, BLOCK_N)
    start_n = 0

    while start_n < max_kv_pos:
        kv_pos = start_n + offs_n
        kv_mask = kv_pos < max_kv_pos
        physical_block = tl.load(
            block_table_ptr + batch_idx * block_table_stride + kv_pos // block_size,
            mask=kv_mask,
            other=0,
        )
        slot = kv_pos % block_size
        k_scale = tl.load(
            k_scale_ptr
            + physical_block * ks_stride_b
            + slot * ks_stride_s
            + kv_head * ks_stride_h,
            mask=kv_mask,
            other=1.0,
        )
        v_scale = tl.load(
            v_scale_ptr
            + physical_block * vs_stride_b
            + slot * vs_stride_s
            + kv_head * vs_stride_h,
            mask=kv_mask,
            other=1.0,
        )
        k = tl.load(
            k_cache_ptr
            + physical_block[:, None] * k_stride_b
            + kv_head * k_stride_h
            + slot[:, None] * k_stride_s
            + offs_d[None, :] * k_stride_d,
            mask=kv_mask[:, None] & dim_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        v = tl.load(
            v_cache_ptr
            + physical_block[:, None] * v_stride_b
            + kv_head * v_stride_h
            + slot[:, None] * v_stride_s
            + offs_d[None, :] * v_stride_d,
            mask=kv_mask[:, None] & dim_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        scores = tl.sum(k * q[None, :], axis=1) * (softmax_scale * k_scale)
        scores = tl.where(kv_mask, scores, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(scores, axis=0))
        p = tl.exp(scores - m_new)
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        acc = acc * alpha + tl.sum((p * v_scale)[:, None] * v, axis=0)
        m_i = m_new
        start_n += BLOCK_N

    acc = acc / l_i
    tl.store(
        out_ptr
        + token_idx * out_stride_t
        + q_head * out_stride_h
        + offs_d * out_stride_d,
        acc,
        mask=dim_mask,
    )


@triton.jit
def _triton_fp8_paged_mha_split_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    k_scale_ptr,
    v_scale_ptr,
    block_table_ptr,
    cu_q_ptr,
    seq_lens_ptr,
    partial_acc_ptr,
    partial_m_ptr,
    partial_l_ptr,
    total_tokens: tl.constexpr,
    batch_size: tl.constexpr,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_size: tl.constexpr,
    block_size: tl.constexpr,
    block_table_stride: tl.constexpr,
    q_stride_t: tl.constexpr,
    q_stride_h: tl.constexpr,
    q_stride_d: tl.constexpr,
    k_stride_b: tl.constexpr,
    k_stride_h: tl.constexpr,
    k_stride_s: tl.constexpr,
    k_stride_d: tl.constexpr,
    v_stride_b: tl.constexpr,
    v_stride_h: tl.constexpr,
    v_stride_s: tl.constexpr,
    v_stride_d: tl.constexpr,
    ks_stride_b: tl.constexpr,
    ks_stride_s: tl.constexpr,
    ks_stride_h: tl.constexpr,
    vs_stride_b: tl.constexpr,
    vs_stride_s: tl.constexpr,
    vs_stride_h: tl.constexpr,
    partial_stride_t: tl.constexpr,
    partial_stride_h: tl.constexpr,
    partial_stride_s: tl.constexpr,
    partial_stride_d: tl.constexpr,
    stats_stride_t: tl.constexpr,
    stats_stride_h: tl.constexpr,
    stats_stride_s: tl.constexpr,
    softmax_scale: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SPLIT_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
):
    token_idx = tl.program_id(0)
    q_head = tl.program_id(1)
    split_idx = tl.program_id(2)

    batch_idx = _find_batch(cu_q_ptr, token_idx, batch_size)
    q_start = tl.load(cu_q_ptr + batch_idx)
    q_end = tl.load(cu_q_ptr + batch_idx + 1)
    q_len = q_end - q_start
    local_q_pos = token_idx - q_start
    seq_len = tl.load(seq_lens_ptr + batch_idx)
    context_len = seq_len - q_len
    max_kv_pos = context_len + local_q_pos + 1

    kv_group_size = num_heads // num_kv_heads
    kv_head = q_head // kv_group_size
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    dim_mask = offs_d < head_size

    q = tl.load(
        q_ptr + token_idx * q_stride_t + q_head * q_stride_h + offs_d * q_stride_d,
        mask=dim_mask,
        other=0.0,
    ).to(tl.float32)

    split_start = split_idx * SPLIT_SIZE
    split_end = tl.minimum(split_start + SPLIT_SIZE, max_kv_pos)
    m_i = tl.full((), -float("inf"), dtype=tl.float32)
    l_i = tl.full((), 0.0, dtype=tl.float32)
    acc = tl.zeros((HEAD_SIZE_PADDED,), dtype=tl.float32)
    offs_n = tl.arange(0, BLOCK_N)
    start_n = split_start

    while start_n < split_end:
        kv_pos = start_n + offs_n
        kv_mask = kv_pos < split_end
        physical_block = tl.load(
            block_table_ptr + batch_idx * block_table_stride + kv_pos // block_size,
            mask=kv_mask,
            other=0,
        )
        slot = kv_pos % block_size
        k_scale = tl.load(
            k_scale_ptr
            + physical_block * ks_stride_b
            + slot * ks_stride_s
            + kv_head * ks_stride_h,
            mask=kv_mask,
            other=1.0,
        )
        v_scale = tl.load(
            v_scale_ptr
            + physical_block * vs_stride_b
            + slot * vs_stride_s
            + kv_head * vs_stride_h,
            mask=kv_mask,
            other=1.0,
        )
        k = tl.load(
            k_cache_ptr
            + physical_block[:, None] * k_stride_b
            + kv_head * k_stride_h
            + slot[:, None] * k_stride_s
            + offs_d[None, :] * k_stride_d,
            mask=kv_mask[:, None] & dim_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        v = tl.load(
            v_cache_ptr
            + physical_block[:, None] * v_stride_b
            + kv_head * v_stride_h
            + slot[:, None] * v_stride_s
            + offs_d[None, :] * v_stride_d,
            mask=kv_mask[:, None] & dim_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        scores = tl.sum(k * q[None, :], axis=1) * (softmax_scale * k_scale)
        scores = tl.where(kv_mask, scores, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(scores, axis=0))
        p = tl.exp(scores - m_new)
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        acc = acc * alpha + tl.sum((p * v_scale)[:, None] * v, axis=0)
        m_i = m_new
        start_n += BLOCK_N

    tl.store(
        partial_acc_ptr
        + token_idx * partial_stride_t
        + q_head * partial_stride_h
        + split_idx * partial_stride_s
        + offs_d * partial_stride_d,
        acc,
        mask=dim_mask,
    )
    tl.store(
        partial_m_ptr
        + token_idx * stats_stride_t
        + q_head * stats_stride_h
        + split_idx * stats_stride_s,
        m_i,
    )
    tl.store(
        partial_l_ptr
        + token_idx * stats_stride_t
        + q_head * stats_stride_h
        + split_idx * stats_stride_s,
        l_i,
    )


@triton.jit
def _triton_fp8_paged_mha_split_combine_kernel(
    partial_acc_ptr,
    partial_m_ptr,
    partial_l_ptr,
    out_ptr,
    num_splits: tl.constexpr,
    head_size: tl.constexpr,
    partial_stride_t: tl.constexpr,
    partial_stride_h: tl.constexpr,
    partial_stride_s: tl.constexpr,
    partial_stride_d: tl.constexpr,
    stats_stride_t: tl.constexpr,
    stats_stride_h: tl.constexpr,
    stats_stride_s: tl.constexpr,
    out_stride_t: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_d: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
):
    token_idx = tl.program_id(0)
    q_head = tl.program_id(1)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    dim_mask = offs_d < head_size

    m = tl.full((), -float("inf"), dtype=tl.float32)
    split_idx = 0
    while split_idx < num_splits:
        m_s = tl.load(
            partial_m_ptr
            + token_idx * stats_stride_t
            + q_head * stats_stride_h
            + split_idx * stats_stride_s
        )
        m = tl.maximum(m, m_s)
        split_idx += 1

    l = tl.full((), 0.0, dtype=tl.float32)
    acc = tl.zeros((HEAD_SIZE_PADDED,), dtype=tl.float32)
    split_idx = 0
    while split_idx < num_splits:
        m_s = tl.load(
            partial_m_ptr
            + token_idx * stats_stride_t
            + q_head * stats_stride_h
            + split_idx * stats_stride_s
        )
        l_s = tl.load(
            partial_l_ptr
            + token_idx * stats_stride_t
            + q_head * stats_stride_h
            + split_idx * stats_stride_s
        )
        alpha = tl.where(l_s > 0.0, tl.exp(m_s - m), 0.0)
        acc_s = tl.load(
            partial_acc_ptr
            + token_idx * partial_stride_t
            + q_head * partial_stride_h
            + split_idx * partial_stride_s
            + offs_d * partial_stride_d,
            mask=dim_mask,
            other=0.0,
        )
        l += l_s * alpha
        acc += acc_s * alpha
        split_idx += 1

    acc = acc / l
    tl.store(
        out_ptr
        + token_idx * out_stride_t
        + q_head * out_stride_h
        + offs_d * out_stride_d,
        acc,
        mask=dim_mask,
    )
