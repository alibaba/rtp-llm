# Adapted from SGLang (Apache-2.0).
# Source: https://raw.githubusercontent.com/sgl-project/sglang/c5367fa964af92d8c253cc129ea8a86e97cab2c5/python/sglang/kernels/ops/attention/dsa/tilelang_kernel.py
# Local changes: CUDA/BF16-only entry point; invalid-index bounds and empty-row
# normalization; no SGLang imports or global TileLang monkey patches.
import tilelang
import tilelang.language as T
import torch


@tilelang.jit(
    out_idx=[-1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def sparse_attention_fwd_kernel_v1(
    num_heads,
    dim,
    tail_dim,
    topk,
    *,
    kv_group=1,
    sm_scale=None,
    is_causal=True,
    block_I=64,
    num_stages=2,
    threads=256,
):
    assert (
        dim == tilelang.math.next_power_of_2(dim) or dim % 64 == 0
    ), f"dim={dim} must be a power of 2 or a multiple of 64"
    assert tail_dim == 0 or tail_dim == tilelang.math.next_power_of_2(
        tail_dim
    ), f"tail_dim={tail_dim} must be 0 or a power of 2"
    has_tail = tail_dim > 0
    assert is_causal == True, "non-casual is not supported"
    assert (
        topk % block_I == 0
    ), "otherwise will load some index=0 thus causing wrong kv to be loaded"
    if sm_scale is None:
        sm_scale = (1.0 / (dim + tail_dim)) ** 0.5 * 1.44269504  # log2(e)
    else:
        sm_scale = sm_scale * 1.44269504  # log2(e)

    batch = T.symbolic("batch")
    seq_len = T.symbolic("seq_len")
    seq_len_kv = T.symbolic("seq_len_kv")

    head_kv = num_heads // kv_group
    q_shape = [batch, seq_len, num_heads, dim + tail_dim]
    kv_shape = [batch, seq_len_kv, kv_group, dim + tail_dim]
    o_shape = [batch, seq_len, num_heads, dim]
    indices_shape = [batch, seq_len, kv_group, topk]
    indices_dtype = "int32"
    dtype = "bfloat16"
    accum_dtype = "float"

    H = head_kv
    padded_H = max(tilelang.math.next_power_of_2(head_kv), 16)
    if padded_H != H:
        assert kv_group == 1
    BI = block_I
    NI = tilelang.cdiv(topk, block_I)
    D = dim
    D_tail = tail_dim

    if head_kv > 64:
        assert head_kv % 64 == 0, "head_kv should be a multiple of 64"
        REPLICATE_H = head_kv // 64
    else:
        REPLICATE_H = 1

    H_per_block = padded_H if REPLICATE_H == 1 else 64

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),  # type: ignore
        KV: T.Tensor(kv_shape, dtype),  # type: ignore
        Indices: T.Tensor(indices_shape, indices_dtype),  # type: ignore
        Output: T.Tensor(o_shape, dtype),  # type: ignore
    ):
        with T.Kernel(seq_len * REPLICATE_H, batch, kv_group, threads=threads) as (
            bx,
            by,
            bz,
        ):
            Q_shared = T.alloc_shared([H_per_block, D], dtype)
            if has_tail:
                Q_tail_shared = T.alloc_shared([H_per_block, D_tail], dtype)
            KV_shared = T.alloc_shared([BI, D], dtype)
            if has_tail:
                K_tail_shared = T.alloc_shared([BI, D_tail], dtype)
            O_shared = T.alloc_shared([H_per_block, D], dtype)
            mask = T.alloc_fragment([BI], "bool")

            acc_o = T.alloc_fragment([H_per_block, D], accum_dtype)
            acc_s = T.alloc_fragment([H_per_block, BI], accum_dtype)
            S_shared = T.alloc_shared([H_per_block, BI], dtype)
            sumexp = T.alloc_fragment([H_per_block], accum_dtype)
            sumexp_i = T.alloc_fragment([H_per_block], accum_dtype)
            alpha = T.alloc_fragment([H_per_block], accum_dtype)
            m_i = T.alloc_fragment([H_per_block], accum_dtype)
            m_i_prev = T.alloc_fragment([H_per_block], accum_dtype)

            T.fill(acc_o, 0)
            T.fill(sumexp, 0)
            T.fill(m_i, -(2**30))  # avoid -inf - inf to cause nan

            b_i, g_i = by, bz
            s_i = bx if REPLICATE_H == 1 else (bx // REPLICATE_H)
            q_i = s_i
            max_kv_i = q_i

            H0 = g_i * padded_H + (0 if REPLICATE_H == 1 else (bx % REPLICATE_H) * 64)
            H1 = H0 + H_per_block

            T.copy(Q[b_i, s_i, H0:H1, :D], Q_shared)
            if has_tail:
                T.copy(Q[b_i, s_i, H0:H1, D:], Q_tail_shared)

            for i_i in T.Pipelined(NI, num_stages=num_stages):
                for bi_i in T.Parallel(BI):
                    mask[bi_i] = (Indices[b_i, s_i, g_i, i_i * BI + bi_i] >= 0) & (
                        Indices[b_i, s_i, g_i, i_i * BI + bi_i] < seq_len_kv
                    )

                for bi_i, d_i in T.Parallel(BI, D):
                    KV_shared[bi_i, d_i] = KV[
                        b_i, Indices[b_i, s_i, g_i, i_i * BI + bi_i], g_i, d_i
                    ]
                if has_tail:
                    for bi_i, d_i in T.Parallel(BI, D_tail):
                        K_tail_shared[bi_i, d_i] = KV[
                            b_i,
                            Indices[b_i, s_i, g_i, i_i * BI + bi_i],
                            g_i,
                            D + d_i,
                        ]

                for h_i, bi_i in T.Parallel(H_per_block, BI):
                    acc_s[h_i, bi_i] = T.if_then_else(
                        mask[bi_i], 0, -T.infinity(acc_s.dtype)
                    )
                T.gemm(
                    Q_shared,
                    KV_shared,
                    acc_s,
                    transpose_B=True,
                    policy=T.GemmWarpPolicy.FullCol,
                )
                if has_tail:
                    T.gemm(
                        Q_tail_shared,
                        K_tail_shared,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullCol,
                    )
                T.copy(m_i, m_i_prev)
                T.reduce_max(acc_s, m_i, dim=1, clear=False)
                for h_i in T.Parallel(H_per_block):
                    alpha[h_i] = T.exp2((m_i_prev[h_i] - m_i[h_i]) * sm_scale)
                for h_i, bi_i in T.Parallel(H_per_block, BI):
                    acc_s[h_i, bi_i] = T.exp2(
                        acc_s[h_i, bi_i] * sm_scale - m_i[h_i] * sm_scale
                    )
                T.reduce_sum(acc_s, sumexp_i, dim=1)  # is this a accumulate operator?
                for h_i in T.Parallel(H_per_block):
                    sumexp[h_i] = sumexp[h_i] * alpha[h_i] + sumexp_i[h_i]
                for h_i, d_i in T.Parallel(H_per_block, D):
                    acc_o[h_i, d_i] = acc_o[h_i, d_i] * alpha[h_i]

                T.copy(acc_s, S_shared)
                T.gemm(S_shared, KV_shared, acc_o, policy=T.GemmWarpPolicy.FullCol)

            # Rescale
            for h_i, d_i in T.Parallel(H_per_block, D):
                acc_o[h_i, d_i] = T.if_then_else(
                    sumexp[h_i] > 0, acc_o[h_i, d_i] / sumexp[h_i], 0
                )
            for h_i in T.Parallel(H_per_block):
                sumexp[h_i] = T.log2(sumexp[h_i]) + m_i[h_i] * sm_scale

            T.copy(acc_o, O_shared)
            T.copy(acc_o, Output[b_i, s_i, H0:H1, :])

    return main


def tilelang_sparse_fwd(q, kv, indices, sm_scale, d_v=512):
    if q.ndim != 3 or q.shape[1:] != (8, 512) or d_v != 512:
        raise ValueError("GLM TileLang path requires Q[T,8,512] and d_v=512")
    if q.dtype != torch.bfloat16 or kv.dtype != torch.bfloat16:
        raise ValueError("GLM TileLang path requires BF16 Q and KV")
    if indices.shape[-1] % 64:
        raise ValueError("top-k must be padded to a multiple of 64")
    if kv.ndim != 3 or tuple(kv.shape[1:]) != (1, 512):
        raise ValueError("GLM small-head MLA requires KV[N,1,512]")
    if (
        indices.ndim != 3
        or indices.shape[:2] != (q.shape[0], 1)
        or indices.dtype != torch.int32
        or indices.stride(-1) != 1
    ):
        raise ValueError(
            "GLM small-head MLA requires int32 indices[T,1,K] with contiguous K"
        )
    if not q.is_cuda or q.device != kv.device or q.device != indices.device:
        raise ValueError("Q, KV and indices must share a CUDA device")
    if not q.shape[0]:
        return torch.empty_like(q)
    kernel = sparse_attention_fwd_kernel_v1(
        8, 512, 0, indices.shape[-1], sm_scale=sm_scale
    )
    return kernel(
        q.contiguous().unsqueeze(0),
        kv.contiguous().unsqueeze(0),
        indices.contiguous().unsqueeze(0),
    ).squeeze(0)
