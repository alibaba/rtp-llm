# Copyright 2025 XunhaoLai. All rights reserved.

from typing import Optional

import torch
import triton
import triton.language as tl

from ..common.utils import get_cu_seqblocks, robust_allocator


_HEUR_gqa_share_sparse_fwd_kernel = {
        "BLOCK_SIZE_KD": lambda args: triton.next_power_of_2(args["qk_head_dim"]),
        "BLOCK_SIZE_VD": lambda args: triton.next_power_of_2(args["v_head_dim"]),
        "BLOCK_SIZE_H": lambda args: triton.next_power_of_2(
            max(
                16 // args["BLOCK_SIZE_Q"],
                triton.next_power_of_2(args["gqa_group_size"]),
            )
        ),
        "BLOCK_SIZE_T": lambda args: triton.next_power_of_2(args["max_topk"]),
        "BLOCK_SIZE_QH": lambda args: args["BLOCK_SIZE_Q"] * args["BLOCK_SIZE_H"],
        "HAS_SINK": lambda args: args["sink_ptr"] is not None,
    }


@triton.heuristics(_HEUR_gqa_share_sparse_fwd_kernel)
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=nw, num_stages=ns)
        for nw in [2, 4, 8]
        for ns in [2, 3, 4]
    ],
    key=[
        "BLOCK_SIZE_Q",
        "BLOCK_SIZE_K",
        "qk_head_dim",
        "v_head_dim",
        "gqa_group_size",
        "SAFE_SLOTS",
    ],
)
@triton.jit
def _gqa_share_sparse_fwd_kernel(
    q_ptr,  # Q: n x h x d
    k_cache_ptr,  # K paged: max_slots x kh x d
    v_cache_ptr,  # V paged: max_slots x kh x d
    sink_ptr,  # Sink: h x d
    t_ptr,  # topk_idx: kh x n x k
    o_ptr,  # O: n x h x d
    req_to_token_ptr,  # req_to_token: max_reqs x max_kv_len
    # seqlens
    cu_seqlens_q,
    cu_seqblocks_q,
    seq_lens,
    prefix_lens,
    slot_ids,
    # shape
    max_slots,
    num_kv_heads,
    gqa_group_size,
    qk_head_dim,
    v_head_dim,
    max_topk,
    # q loop num
    num_q_loop,
    # sm_scale
    sm_scale,
    # stride
    stride_qn,
    stride_qh,
    stride_qd,
    stride_ks,
    stride_kh,
    stride_kd,
    stride_vs,
    stride_vh,
    stride_vd,
    stride_sh,
    stride_sd,
    stride_th,
    stride_tn,
    stride_tk,
    stride_on,
    stride_oh,
    stride_od,
    stride_r2t_b,
    # META parameters
    BLOCK_SIZE_Q: tl.constexpr,  # q block size
    BLOCK_SIZE_K: tl.constexpr,  # k block size
    BLOCK_SIZE_KD: tl.constexpr,
    BLOCK_SIZE_VD: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    BLOCK_SIZE_QH: tl.constexpr,
    # has sink
    HAS_SINK: tl.constexpr,
    USE_TMA: tl.constexpr,
    # Skip the `(x + max_slots) % max_slots` safety wrap when the caller
    # guarantees slot_ids and req_to_token contain only valid non-negative
    # slots. The wrap is a non-trivial 128-element i64 mod on the hot path.
    SAFE_SLOTS: tl.constexpr,
):
    sm_scale_log2e = sm_scale * 1.4426950409
    # get batch id and head id
    pid_q = tl.program_id(0)
    pid_kh = tl.program_id(1)
    pid_b = tl.program_id(2)
    pid_h = pid_kh * gqa_group_size
    # get q k start and len after rmpad
    q_start = tl.load(cu_seqlens_q + pid_b)
    q_len = tl.load(cu_seqlens_q + pid_b + 1) - q_start
    q_block_start = tl.load(cu_seqblocks_q + pid_b)
    q_block_len = tl.load(cu_seqblocks_q + pid_b + 1) - q_block_start
    seq_len = tl.load(seq_lens + pid_b)
    prefix_len = tl.load(prefix_lens + pid_b)
    sid = tl.load(slot_ids + pid_b).to(tl.int64)
    if SAFE_SLOTS:
        sid = (sid + max_slots) % max_slots  # safety against negative
    if pid_q * num_q_loop >= q_block_len:
        return
    real_q_loop = min(num_q_loop, q_block_len - pid_q * num_q_loop)
    if HAS_SINK:
        sink_ptrs = tl.make_block_ptr(
            base=sink_ptr + pid_h * stride_sh,
            shape=(gqa_group_size, qk_head_dim),
            strides=(stride_sh, stride_sd),
            offsets=(0, 0),
            block_shape=(BLOCK_SIZE_H, BLOCK_SIZE_KD),
            order=(1, 0),
        )
        sink = tl.load(sink_ptrs, boundary_check=(0, 1), padding_option="zero").to(
            tl.float32
        )
    # offsets for paged K/V load
    off_n = tl.arange(0, BLOCK_SIZE_K)
    off_kd = tl.arange(0, BLOCK_SIZE_KD)
    off_vd = tl.arange(0, BLOCK_SIZE_VD)
    kd_mask = off_kd < qk_head_dim
    vd_mask = off_vd < v_head_dim
    for j in range(real_q_loop):
        pid_q_j = pid_q * num_q_loop + j
        # init topk idx pointer
        t_ptr_j = t_ptr + (q_block_start + pid_q_j) * stride_tn + pid_kh * stride_th
        # we assume that the topk_idx is right padded with -1
        off_t = tl.arange(0, BLOCK_SIZE_T)
        topk_idx = tl.load(t_ptr_j + off_t * stride_tk, mask=off_t < max_topk, other=-1)
        valid_idx = tl.where(topk_idx >= 0, off_t, -1)
        real_topk = tl.sum(valid_idx != -1, axis=0)
        # init qkv pointer
        q_ptrs = tl.make_block_ptr(
            base=q_ptr + q_start * stride_qn + pid_h * stride_qh,
            shape=(q_len, gqa_group_size, qk_head_dim),
            strides=(stride_qn, stride_qh, stride_qd),
            offsets=(pid_q_j * BLOCK_SIZE_Q, 0, 0),
            block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_KD),
            order=(2, 1, 0),
        )
        # load q, shape: [BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_D] -> [BLOCK_SIZE_QH, BLOCK_SIZE_D]
        q = tl.load(q_ptrs, boundary_check=(0, 1, 2), padding_option="zero")
        # init statistics
        off_q_k = (
            tl.arange(0, BLOCK_SIZE_Q)[:, None]
            + pid_q_j * BLOCK_SIZE_Q
            + prefix_len
            - tl.arange(0, BLOCK_SIZE_K)[None, :]
        )
        # Online softmax in `l_i` (running denominator) form instead of the
        # log-sum-exp form. Saves one tl.exp2 + one tl.log2 per inner iter on
        # a [BLOCK_SIZE_QH] vector — SFU-bound ops that the loop has to wait
        # on. Mathematically equivalent: sum_i tracks
        #     sum_i = sum_{k seen so far} exp2(qk_k - m_current)
        # which is exactly the denominator that the final scale needs.
        if HAS_SINK:
            m_i = tl.zeros((BLOCK_SIZE_Q, BLOCK_SIZE_H), dtype=tl.float32)
            qsink = (
                tl.sum(q.to(tl.float32) * sink[None, :, :], axis=2) * sm_scale_log2e
            )  # (BLOCK_SIZE_Q, BLOCK_SIZE_H)
            m_i += qsink
            m_i = tl.reshape(m_i, BLOCK_SIZE_QH)
            # Sink contributes a virtual position with score qsink and weight 1.
            sum_i = tl.full((BLOCK_SIZE_QH,), 1.0, dtype=tl.float32)
        else:
            m_i = tl.full((BLOCK_SIZE_QH,), float("-inf"), dtype=tl.float32)
            sum_i = tl.zeros((BLOCK_SIZE_QH,), dtype=tl.float32)
        acc_o = tl.full((BLOCK_SIZE_QH, BLOCK_SIZE_VD), 0, dtype=tl.float32)
        q = tl.reshape(q, BLOCK_SIZE_QH, BLOCK_SIZE_KD)
        # sparse attention
        for i in range(real_topk):
            # get current block start index (absolute K position)
            c = tl.load(t_ptr_j).to(tl.int32) * BLOCK_SIZE_K
            t_ptr_j = t_ptr_j + stride_tk
            # paged load K via req_to_token: pos -> slot -> k_cache
            pos = c + off_n
            pos_mask = pos < seq_len
            slots = tl.load(
                req_to_token_ptr + sid * stride_r2t_b + pos,
                mask=pos_mask,
                other=0,
            ).to(tl.int64)
            # Hint the compiler that the slot vector is a contiguous run of
            # 16+ consecutive lanes so K / V loads can issue wider vectors.
            slots = tl.max_contiguous(tl.multiple_of(slots, 16), 16)
            if SAFE_SLOTS:
                slots = (slots + max_slots) % max_slots  # safety against negative
            # k shape: [BLOCK_SIZE_KD, BLOCK_SIZE_K] (transposed for tl.dot)
            k = tl.load(
                k_cache_ptr
                + slots[None, :] * stride_ks
                + pid_kh * stride_kh
                + off_kd[:, None] * stride_kd,
                mask=kd_mask[:, None] & pos_mask[None, :],
                other=0.0,
            )
            # Issue V load right after K so they share in-flight bandwidth and
            # the scheduler can hide V's latency behind the QK dot + softmax
            # work below. Both addresses depend only on `slots`, computed once.
            v = tl.load(
                v_cache_ptr
                + slots[:, None] * stride_vs
                + pid_kh * stride_vh
                + off_vd[None, :] * stride_vd,
                mask=pos_mask[:, None] & vd_mask[None, :],
                other=0.0,
            )
            # [BLOCK_SIZE_QH, qk_head_dim] @ [qk_head_dim, BLOCK_SIZE_K]
            #   -> [BLOCK_SIZE_QH, BLOCK_SIZE_K]
            qk = tl.dot(q, k) * sm_scale_log2e
            # causal mask: broadcast a [Q,1,K] mask out to [Q,H,K] then reshape
            qk += tl.reshape(
                tl.broadcast_to(
                    tl.where(off_q_k[:, None, :] >= c, 0.0, float("-inf")),
                    (BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_K),
                ),
                BLOCK_SIZE_QH,
                BLOCK_SIZE_K,
            )
            # K boundary mask: positions beyond seq_len contribute -inf
            qk += tl.where(pos_mask[None, :], 0, float("-inf"))
            # compute m_ij and l_ij
            m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
            p = tl.exp2(qk - m_ij[:, None])
            l_ij = tl.sum(p, axis=1)
            # scale acc_o and running denominator together
            acc_o_scale = tl.exp2(m_i - m_ij)
            acc_o = acc_o * acc_o_scale[:, None]
            sum_i = sum_i * acc_o_scale + l_ij
            p = p.to(v.dtype)
            acc_o += tl.dot(p, v)
            # update statistics
            m_i = m_ij
        # final scale: divide by running denominator. Guard against the
        # all-empty case (real_topk == 0 -> sum_i == 0) to emit a clean zero
        # instead of NaN, mirroring the partition-empty handling in the decode
        # kernel.
        inv_sum = tl.where(sum_i > 0, 1.0 / sum_i, 0.0)
        acc_o = acc_o * inv_sum[:, None]
        # save output
        acc_o = tl.reshape(acc_o, BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_VD)
        o_ptrs = tl.make_block_ptr(
            base=o_ptr + q_start * stride_on + pid_h * stride_oh,
            shape=(q_len, gqa_group_size, v_head_dim),
            strides=(stride_on, stride_oh, stride_od),
            offsets=(pid_q_j * BLOCK_SIZE_Q, 0, 0),
            block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_H, BLOCK_SIZE_VD),
            order=(2, 1, 0),
        )
        tl.store(o_ptrs, acc_o.to(o_ptr.dtype.element_ty), boundary_check=(0, 1, 2))


@torch.no_grad()
def flash_prefill_with_gqa_share_sparse(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    sink: Optional[torch.Tensor],
    req_to_token: torch.Tensor,
    slot_ids: torch.Tensor,
    topk_idx: torch.Tensor,
    block_size_q: int,
    block_size_k: int,
    cu_seqlens: torch.Tensor,
    seq_lens: torch.Tensor,
    prefix_lens: torch.Tensor,
    max_seqlen_q: int,
    sm_scale: Optional[float] = None,
    use_tma: bool = True,
    safe_slots: bool = False,
) -> torch.Tensor:
    triton.set_allocator(robust_allocator)
    # dtype check
    assert k_cache.dtype == q.dtype and v_cache.dtype == q.dtype
    assert block_size_q in {1, 2, 4, 8, 16, 32, 64}
    assert block_size_k in {16, 32, 64, 128}
    # shape
    total_q, num_q_heads, qk_head_dim = q.shape
    max_slots, num_k_heads, _ = k_cache.shape
    _, num_v_heads, v_head_dim = v_cache.shape
    batch_size = cu_seqlens.shape[0] - 1
    topk = topk_idx.shape[-1]
    assert topk_idx.shape[0] == num_k_heads
    # gqa
    assert num_k_heads == num_v_heads
    assert num_q_heads % num_k_heads == 0
    gqa_group_size = num_q_heads // num_k_heads
    assert gqa_group_size * block_size_q <= 128
    if sm_scale is None:
        sm_scale = qk_head_dim**-0.5
    cu_seqblocks_q, max_seqblock_q, _, _, _, _ = get_cu_seqblocks(
        cu_seqlens, max_seqlen_q, block_size_q, block_size_k
    )
    # output tensor
    o = torch.empty(total_q, num_q_heads, v_head_dim, device=q.device, dtype=q.dtype)
    # launch kernel
    num_q_loop = (
        max_seqblock_q // 131072 + 1
    )  # calculate multiple querys in one kernel if seqlence length is too long
    BLOCK_SIZE_Q = triton.next_power_of_2(block_size_q)
    BLOCK_SIZE_K = triton.next_power_of_2(block_size_k)
    grid = (
        triton.cdiv(triton.cdiv(max_seqlen_q, block_size_q), num_q_loop),
        num_k_heads,
        batch_size,
    )
    _gqa_share_sparse_fwd_kernel[grid](
        q,
        k_cache,
        v_cache,
        sink,
        topk_idx,
        o,
        req_to_token,
        cu_seqlens,
        cu_seqblocks_q,
        seq_lens,
        prefix_lens,
        slot_ids,
        max_slots,
        num_k_heads,
        gqa_group_size,
        qk_head_dim,
        v_head_dim,
        topk,
        num_q_loop,
        sm_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        sink.stride(0) if sink is not None else 0,
        sink.stride(1) if sink is not None else 0,
        topk_idx.stride(0),
        topk_idx.stride(1),
        topk_idx.stride(2),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        req_to_token.stride(0),
        BLOCK_SIZE_Q=BLOCK_SIZE_Q,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        USE_TMA=use_tma,
        SAFE_SLOTS=safe_slots,
    )
    return o
