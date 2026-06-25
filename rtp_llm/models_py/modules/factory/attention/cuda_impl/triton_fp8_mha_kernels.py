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
    BLOCK_ALIGNED: tl.constexpr,
    SCALE_CONTIGUOUS: tl.constexpr,
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
        if BLOCK_ALIGNED:
            physical_block = tl.load(
                block_table_ptr + batch_idx * block_table_stride + start_n // block_size
            )
            slot = offs_n
        else:
            physical_block = tl.load(
                block_table_ptr + batch_idx * block_table_stride + kv_pos // block_size,
                mask=kv_mask,
                other=0,
            )
            slot = kv_pos % block_size
        k_scale_offset = (
            physical_block * ks_stride_b + slot * ks_stride_s + kv_head * ks_stride_h
        )
        if SCALE_CONTIGUOUS:
            v_scale_offset = k_scale_offset
        else:
            v_scale_offset = (
                physical_block * vs_stride_b
                + slot * vs_stride_s
                + kv_head * vs_stride_h
            )
        k_scale = tl.load(
            k_scale_ptr + k_scale_offset,
            mask=kv_mask,
            other=1.0,
            cache_modifier=".ca",
        )
        v_scale = tl.load(
            v_scale_ptr + v_scale_offset,
            mask=kv_mask,
            other=1.0,
            cache_modifier=".ca",
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
    DECODE_ONE_TOKEN: tl.constexpr,
    SPLIT_BLOCK_ALIGNED: tl.constexpr,
    SCALE_CONTIGUOUS: tl.constexpr,
):
    token_idx = tl.program_id(0)
    q_head = tl.program_id(1)
    split_idx = tl.program_id(2)

    if DECODE_ONE_TOKEN:
        batch_idx = token_idx
        max_kv_pos = tl.load(seq_lens_ptr + batch_idx)
    else:
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
        if SPLIT_BLOCK_ALIGNED:
            physical_block = tl.load(
                block_table_ptr + batch_idx * block_table_stride + start_n // block_size
            )
            slot = offs_n
        else:
            physical_block = tl.load(
                block_table_ptr + batch_idx * block_table_stride + kv_pos // block_size,
                mask=kv_mask,
                other=0,
            )
            slot = kv_pos % block_size
        k_scale_offset = (
            physical_block * ks_stride_b + slot * ks_stride_s + kv_head * ks_stride_h
        )
        if SCALE_CONTIGUOUS:
            v_scale_offset = k_scale_offset
        else:
            v_scale_offset = (
                physical_block * vs_stride_b
                + slot * vs_stride_s
                + kv_head * vs_stride_h
            )
        k_scale = tl.load(
            k_scale_ptr + k_scale_offset,
            mask=kv_mask,
            other=1.0,
            cache_modifier=".ca",
        )
        v_scale = tl.load(
            v_scale_ptr + v_scale_offset,
            mask=kv_mask,
            other=1.0,
            cache_modifier=".ca",
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
def _triton_fp8_paged_xqa_split_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    k_scale_ptr,
    v_scale_ptr,
    block_table_ptr,
    seq_lens_ptr,
    partial_acc_ptr,
    partial_m_ptr,
    partial_l_ptr,
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
    SPLIT_BLOCK_ALIGNED: tl.constexpr,
    SCALE_CONTIGUOUS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    q_group = tl.program_id(1)
    split_idx = tl.program_id(2)

    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    dim_mask = offs_d < head_size
    kv_group_size = num_heads // num_kv_heads
    q_head_base = q_group * GROUP_SIZE
    kv_head = q_head_base // kv_group_size

    q0 = tl.load(
        q_ptr
        + token_idx * q_stride_t
        + (q_head_base + 0) * q_stride_h
        + offs_d * q_stride_d,
        mask=dim_mask & (q_head_base + 0 < num_heads),
        other=0.0,
    ).to(tl.float32)
    if GROUP_SIZE >= 2:
        q1 = tl.load(
            q_ptr
            + token_idx * q_stride_t
            + (q_head_base + 1) * q_stride_h
            + offs_d * q_stride_d,
            mask=dim_mask & (q_head_base + 1 < num_heads),
            other=0.0,
        ).to(tl.float32)
    if GROUP_SIZE >= 3:
        q2 = tl.load(
            q_ptr
            + token_idx * q_stride_t
            + (q_head_base + 2) * q_stride_h
            + offs_d * q_stride_d,
            mask=dim_mask & (q_head_base + 2 < num_heads),
            other=0.0,
        ).to(tl.float32)
    if GROUP_SIZE >= 4:
        q3 = tl.load(
            q_ptr
            + token_idx * q_stride_t
            + (q_head_base + 3) * q_stride_h
            + offs_d * q_stride_d,
            mask=dim_mask & (q_head_base + 3 < num_heads),
            other=0.0,
        ).to(tl.float32)

    split_start = split_idx * SPLIT_SIZE
    max_kv_pos = tl.load(seq_lens_ptr + token_idx)
    split_end = tl.minimum(split_start + SPLIT_SIZE, max_kv_pos)

    m0 = tl.full((), -float("inf"), dtype=tl.float32)
    l0 = tl.full((), 0.0, dtype=tl.float32)
    acc0 = tl.zeros((HEAD_SIZE_PADDED,), dtype=tl.float32)
    if GROUP_SIZE >= 2:
        m1 = tl.full((), -float("inf"), dtype=tl.float32)
        l1 = tl.full((), 0.0, dtype=tl.float32)
        acc1 = tl.zeros((HEAD_SIZE_PADDED,), dtype=tl.float32)
    if GROUP_SIZE >= 3:
        m2 = tl.full((), -float("inf"), dtype=tl.float32)
        l2 = tl.full((), 0.0, dtype=tl.float32)
        acc2 = tl.zeros((HEAD_SIZE_PADDED,), dtype=tl.float32)
    if GROUP_SIZE >= 4:
        m3 = tl.full((), -float("inf"), dtype=tl.float32)
        l3 = tl.full((), 0.0, dtype=tl.float32)
        acc3 = tl.zeros((HEAD_SIZE_PADDED,), dtype=tl.float32)

    offs_n = tl.arange(0, BLOCK_N)
    start_n = split_start
    while start_n < split_end:
        kv_pos = start_n + offs_n
        kv_mask = kv_pos < split_end
        if SPLIT_BLOCK_ALIGNED:
            physical_block = tl.load(
                block_table_ptr + token_idx * block_table_stride + start_n // block_size
            )
            slot = offs_n
        else:
            physical_block = tl.load(
                block_table_ptr + token_idx * block_table_stride + kv_pos // block_size,
                mask=kv_mask,
                other=0,
            )
            slot = kv_pos % block_size

        k_scale_offset = (
            physical_block * ks_stride_b + slot * ks_stride_s + kv_head * ks_stride_h
        )
        if SCALE_CONTIGUOUS:
            v_scale_offset = k_scale_offset
        else:
            v_scale_offset = (
                physical_block * vs_stride_b
                + slot * vs_stride_s
                + kv_head * vs_stride_h
            )
        k_scale = tl.load(
            k_scale_ptr + k_scale_offset,
            mask=kv_mask,
            other=1.0,
            cache_modifier=".ca",
        )
        v_scale = tl.load(
            v_scale_ptr + v_scale_offset,
            mask=kv_mask,
            other=1.0,
            cache_modifier=".ca",
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

        scores0 = tl.sum(k * q0[None, :], axis=1) * (softmax_scale * k_scale)
        scores0 = tl.where(kv_mask, scores0, -float("inf"))
        m0_new = tl.maximum(m0, tl.max(scores0, axis=0))
        p0 = tl.exp(scores0 - m0_new)
        alpha0 = tl.exp(m0 - m0_new)
        l0 = l0 * alpha0 + tl.sum(p0, axis=0)
        acc0 = acc0 * alpha0 + tl.sum((p0 * v_scale)[:, None] * v, axis=0)
        m0 = m0_new

        if GROUP_SIZE >= 2:
            scores1 = tl.sum(k * q1[None, :], axis=1) * (softmax_scale * k_scale)
            scores1 = tl.where(kv_mask, scores1, -float("inf"))
            m1_new = tl.maximum(m1, tl.max(scores1, axis=0))
            p1 = tl.exp(scores1 - m1_new)
            alpha1 = tl.exp(m1 - m1_new)
            l1 = l1 * alpha1 + tl.sum(p1, axis=0)
            acc1 = acc1 * alpha1 + tl.sum((p1 * v_scale)[:, None] * v, axis=0)
            m1 = m1_new

        if GROUP_SIZE >= 3:
            scores2 = tl.sum(k * q2[None, :], axis=1) * (softmax_scale * k_scale)
            scores2 = tl.where(kv_mask, scores2, -float("inf"))
            m2_new = tl.maximum(m2, tl.max(scores2, axis=0))
            p2 = tl.exp(scores2 - m2_new)
            alpha2 = tl.exp(m2 - m2_new)
            l2 = l2 * alpha2 + tl.sum(p2, axis=0)
            acc2 = acc2 * alpha2 + tl.sum((p2 * v_scale)[:, None] * v, axis=0)
            m2 = m2_new

        if GROUP_SIZE >= 4:
            scores3 = tl.sum(k * q3[None, :], axis=1) * (softmax_scale * k_scale)
            scores3 = tl.where(kv_mask, scores3, -float("inf"))
            m3_new = tl.maximum(m3, tl.max(scores3, axis=0))
            p3 = tl.exp(scores3 - m3_new)
            alpha3 = tl.exp(m3 - m3_new)
            l3 = l3 * alpha3 + tl.sum(p3, axis=0)
            acc3 = acc3 * alpha3 + tl.sum((p3 * v_scale)[:, None] * v, axis=0)
            m3 = m3_new

        start_n += BLOCK_N

    tl.store(
        partial_acc_ptr
        + token_idx * partial_stride_t
        + (q_head_base + 0) * partial_stride_h
        + split_idx * partial_stride_s
        + offs_d * partial_stride_d,
        acc0,
        mask=dim_mask & (q_head_base + 0 < num_heads),
    )
    tl.store(
        partial_m_ptr
        + token_idx * stats_stride_t
        + (q_head_base + 0) * stats_stride_h
        + split_idx * stats_stride_s,
        m0,
        mask=q_head_base + 0 < num_heads,
    )
    tl.store(
        partial_l_ptr
        + token_idx * stats_stride_t
        + (q_head_base + 0) * stats_stride_h
        + split_idx * stats_stride_s,
        l0,
        mask=q_head_base + 0 < num_heads,
    )

    if GROUP_SIZE >= 2:
        tl.store(
            partial_acc_ptr
            + token_idx * partial_stride_t
            + (q_head_base + 1) * partial_stride_h
            + split_idx * partial_stride_s
            + offs_d * partial_stride_d,
            acc1,
            mask=dim_mask & (q_head_base + 1 < num_heads),
        )
        tl.store(
            partial_m_ptr
            + token_idx * stats_stride_t
            + (q_head_base + 1) * stats_stride_h
            + split_idx * stats_stride_s,
            m1,
            mask=q_head_base + 1 < num_heads,
        )
        tl.store(
            partial_l_ptr
            + token_idx * stats_stride_t
            + (q_head_base + 1) * stats_stride_h
            + split_idx * stats_stride_s,
            l1,
            mask=q_head_base + 1 < num_heads,
        )
    if GROUP_SIZE >= 3:
        tl.store(
            partial_acc_ptr
            + token_idx * partial_stride_t
            + (q_head_base + 2) * partial_stride_h
            + split_idx * partial_stride_s
            + offs_d * partial_stride_d,
            acc2,
            mask=dim_mask & (q_head_base + 2 < num_heads),
        )
        tl.store(
            partial_m_ptr
            + token_idx * stats_stride_t
            + (q_head_base + 2) * stats_stride_h
            + split_idx * stats_stride_s,
            m2,
            mask=q_head_base + 2 < num_heads,
        )
        tl.store(
            partial_l_ptr
            + token_idx * stats_stride_t
            + (q_head_base + 2) * stats_stride_h
            + split_idx * stats_stride_s,
            l2,
            mask=q_head_base + 2 < num_heads,
        )
    if GROUP_SIZE >= 4:
        tl.store(
            partial_acc_ptr
            + token_idx * partial_stride_t
            + (q_head_base + 3) * partial_stride_h
            + split_idx * partial_stride_s
            + offs_d * partial_stride_d,
            acc3,
            mask=dim_mask & (q_head_base + 3 < num_heads),
        )
        tl.store(
            partial_m_ptr
            + token_idx * stats_stride_t
            + (q_head_base + 3) * stats_stride_h
            + split_idx * stats_stride_s,
            m3,
            mask=q_head_base + 3 < num_heads,
        )
        tl.store(
            partial_l_ptr
            + token_idx * stats_stride_t
            + (q_head_base + 3) * stats_stride_h
            + split_idx * stats_stride_s,
            l3,
            mask=q_head_base + 3 < num_heads,
        )


@triton.jit
def _triton_fp8_paged_xqa_dot_split_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    k_scale_ptr,
    v_scale_ptr,
    block_table_ptr,
    seq_lens_ptr,
    partial_acc_ptr,
    partial_m_ptr,
    partial_l_ptr,
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
    BLOCK_M: tl.constexpr,
    SPLIT_BLOCK_ALIGNED: tl.constexpr,
    SCALE_CONTIGUOUS: tl.constexpr,
):
    token_idx = tl.program_id(0)
    kv_head = tl.program_id(1)
    split_idx = tl.program_id(2)

    kv_group_size = num_heads // num_kv_heads
    q_head_base = kv_head * kv_group_size
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    offs_n = tl.arange(0, BLOCK_N)
    q_mask = offs_m < kv_group_size
    dim_mask = offs_d < head_size

    q = tl.load(
        q_ptr
        + token_idx * q_stride_t
        + (q_head_base + offs_m[:, None]) * q_stride_h
        + offs_d[None, :] * q_stride_d,
        mask=q_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )

    split_start = split_idx * SPLIT_SIZE
    max_kv_pos = tl.load(seq_lens_ptr + token_idx)
    split_end = tl.minimum(split_start + SPLIT_SIZE, max_kv_pos)

    log2e: tl.constexpr = 1.4426950408889634
    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.full((BLOCK_M,), 0.0, dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_SIZE_PADDED), dtype=tl.float32)

    start_n = split_start
    while start_n < split_end:
        kv_pos = start_n + offs_n
        kv_mask = kv_pos < split_end
        if SPLIT_BLOCK_ALIGNED:
            physical_block = tl.load(
                block_table_ptr + token_idx * block_table_stride + start_n // block_size
            )
            slot = offs_n
        else:
            physical_block = tl.load(
                block_table_ptr + token_idx * block_table_stride + kv_pos // block_size,
                mask=kv_mask,
                other=0,
            )
            slot = kv_pos % block_size

        k_scale_offset = (
            physical_block * ks_stride_b + slot * ks_stride_s + kv_head * ks_stride_h
        )
        if SCALE_CONTIGUOUS:
            v_scale_offset = k_scale_offset
        else:
            v_scale_offset = (
                physical_block * vs_stride_b
                + slot * vs_stride_s
                + kv_head * vs_stride_h
            )
        k_scale = tl.load(
            k_scale_ptr + k_scale_offset,
            mask=kv_mask,
            other=1.0,
            cache_modifier=".ca",
        )
        v_scale = tl.load(
            v_scale_ptr + v_scale_offset,
            mask=kv_mask,
            other=1.0,
            cache_modifier=".ca",
        )
        k_t = tl.load(
            k_cache_ptr
            + physical_block[None, :] * k_stride_b
            + kv_head * k_stride_h
            + slot[None, :] * k_stride_s
            + offs_d[:, None] * k_stride_d,
            mask=dim_mask[:, None] & kv_mask[None, :],
            other=0.0,
        )
        v = tl.load(
            v_cache_ptr
            + physical_block[:, None] * v_stride_b
            + kv_head * v_stride_h
            + slot[:, None] * v_stride_s
            + offs_d[None, :] * v_stride_d,
            mask=kv_mask[:, None] & dim_mask[None, :],
            other=0.0,
        )

        scores = tl.dot(q, k_t.to(tl.float16), out_dtype=tl.float32) * (
            log2e * softmax_scale * k_scale[None, :]
        )
        scores = tl.where(q_mask[:, None] & kv_mask[None, :], scores, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        p = tl.exp2(scores - m_new[:, None])
        alpha = tl.exp2(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        pv = tl.dot(
            (p * v_scale[None, :]).to(tl.float16),
            v.to(tl.float16),
            out_dtype=tl.float32,
        )
        acc = acc * alpha[:, None] + pv
        m_i = m_new
        start_n += BLOCK_N

    tl.store(
        partial_acc_ptr
        + token_idx * partial_stride_t
        + (q_head_base + offs_m[:, None]) * partial_stride_h
        + split_idx * partial_stride_s
        + offs_d[None, :] * partial_stride_d,
        acc,
        mask=q_mask[:, None] & dim_mask[None, :],
    )
    tl.store(
        partial_m_ptr
        + token_idx * stats_stride_t
        + (q_head_base + offs_m) * stats_stride_h
        + split_idx * stats_stride_s,
        m_i,
        mask=q_mask,
    )
    tl.store(
        partial_l_ptr
        + token_idx * stats_stride_t
        + (q_head_base + offs_m) * stats_stride_h
        + split_idx * stats_stride_s,
        l_i,
        mask=q_mask,
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

    l = tl.full((), 0.0, dtype=tl.float32)
    m = tl.full((), -float("inf"), dtype=tl.float32)
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
        m_new = tl.maximum(m, m_s)
        old_scale = tl.exp(m - m_new)
        split_scale = tl.where(l_s > 0.0, tl.exp(m_s - m_new), 0.0)
        acc_s = tl.load(
            partial_acc_ptr
            + token_idx * partial_stride_t
            + q_head * partial_stride_h
            + split_idx * partial_stride_s
            + offs_d * partial_stride_d,
            mask=dim_mask,
            other=0.0,
        )
        l = l * old_scale + l_s * split_scale
        acc = acc * old_scale + acc_s * split_scale
        m = m_new
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
