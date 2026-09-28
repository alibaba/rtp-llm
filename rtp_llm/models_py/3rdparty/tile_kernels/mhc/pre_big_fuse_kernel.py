import math

import tilelang
import torch
from tilelang import language as T


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,
        tilelang.PassConfigKey.TL_DISABLE_VECTORIZE_256: True,
    },
)
def _mhc_pre_big_fuse(
    hidden_size: int,
    rms_eps: float,
    mhc_pre_eps: float,
    mhc_sinkhorn_eps: float,
    mhc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 16,
    mhc_mult: int = 4,
    stabilize_mixes: bool = False,
    stabilize_comb: bool = False,
    fuse_norm: bool = False,
    norm_eps: float = 1e-6,
    parallel_reduction: bool = False,
):
    num_tokens = T.dynamic("num_tokens")
    mhc_mult3 = mhc_mult * (2 + mhc_mult)
    hidden_block = math.gcd(512, hidden_size)

    @T.prim_func
    def mhc_pre_big_fuse(
        gemm_out_mul: T.Tensor[(n_splits, num_tokens, mhc_mult3), T.float32],
        gemm_out_sqrsum: T.Tensor[(n_splits, num_tokens), T.float32],
        mhc_scale: T.Tensor[(3,), T.float32],
        mhc_base: T.Tensor[(mhc_mult3,), T.float32],
        residual: T.Tensor[(num_tokens, mhc_mult, hidden_size), T.bfloat16],
        # outputs
        post_mix: T.Tensor[(num_tokens, mhc_mult), T.float32],
        comb_mix: T.Tensor[(num_tokens, mhc_mult * mhc_mult), T.float32],
        layer_input: T.Tensor[(num_tokens, hidden_size), T.bfloat16],
        norm_weight: T.Tensor[(hidden_size if fuse_norm else 0,), T.bfloat16],
    ) -> None:
        with T.Kernel(num_tokens, threads=96) as pid:
            ##################################################################
            # _mhc_pre_norm_fn_fwd_norm
            mixes_shared = T.alloc_shared(mhc_mult3, T.float32)
            if T.get_thread_binding() < 32:
                rms = T.alloc_fragment(1, T.float32)
                mixes = T.alloc_fragment(mhc_mult3, T.float32)
                if parallel_reduction and n_splits > 1:
                    # Parallel reduction avoids a dependent global-memory load
                    # for each split. Keep the small HC projections in FP32.
                    partial_sqrsum = T.alloc_fragment(n_splits, T.float32)
                    partial_mixes = T.alloc_fragment((mhc_mult3, n_splits), T.float32)
                    for split in T.Parallel(n_splits):
                        partial_sqrsum[split] = gemm_out_sqrsum[split, pid]
                    for j, split in T.Parallel(mhc_mult3, n_splits):
                        partial_mixes[j, split] = gemm_out_mul[split, pid, j]
                    T.reduce_sum(partial_sqrsum, rms, dim=0)
                    T.reduce_sum(partial_mixes, mixes, dim=1)
                elif n_splits > 1:
                    # Preserve the CUDA consumer's ordered split reduction.
                    rms[0] = 0
                    for split in T.serial(n_splits):
                        rms[0] += gemm_out_sqrsum[split, pid]
                    for j in T.Parallel(mhc_mult3):
                        mixes[j] = 0
                        for split in T.serial(n_splits):
                            mixes[j] += gemm_out_mul[split, pid, j]
                else:
                    rms[0] = gemm_out_sqrsum[0, pid]
                    for j in T.Parallel(mhc_mult3):
                        mixes[j] = gemm_out_mul[0, pid, j]
                rms[0] = T.rsqrt(rms[0] / (mhc_mult * hidden_size) + rms_eps)
                for j in T.Parallel(mhc_mult3):
                    mixes[j] *= rms[0]
                    if stabilize_mixes:
                        # Preserve the explicit atomic backend's BF16 boundary.
                        mixes[j] = T.cast(T.cast(mixes[j], T.bfloat16), T.float32)
                T.copy(mixes, mixes_shared, disable_tma=True)

            # Warp 0 publishes the normalized projection to shared memory;
            # the other warps consume it for PRE mixing. Make that handoff
            # explicit so a faster producer backend (DeepGEMM) cannot expose
            # stale shared values during CUDA Graph replay.
            T.sync_threads()

            if T.get_thread_binding() < 32:
                ##################################################################
                # _mhc_pre_split_mixes_fwd (post & comb)
                cm = T.alloc_fragment((mhc_mult, mhc_mult), T.float32)
                for j in T.Parallel(mhc_mult):
                    post_mix[pid, j] = (
                        T.sigmoid(
                            mixes_shared[j + mhc_mult] * mhc_scale[1]
                            + mhc_base[j + mhc_mult]
                        )
                        * mhc_post_mult_value
                    )
                for j, k in T.Parallel(mhc_mult, mhc_mult):
                    cm[j, k] = (
                        mixes_shared[j * mhc_mult + k + mhc_mult * 2] * mhc_scale[2]
                        + mhc_base[j * mhc_mult + k + mhc_mult * 2]
                    )
                    if stabilize_comb:
                        # DeepGEMM split-K reductions vary by a few FP32 ULPs
                        # across graph replays. Sinkhorn can amplify those
                        # differences, so canonicalize only its 4x4 logits at
                        # the model's BF16 activation boundary.
                        cm[j, k] = T.cast(T.cast(cm[j, k], T.bfloat16), T.float32)

                ##################################################################
                # _mhc_sinkhorn_fwd
                row_sum = T.alloc_fragment(mhc_mult, T.float32)
                col_sum = T.alloc_fragment(mhc_mult, T.float32)

                # comb = comb.softmax(-1) + eps
                row_max = T.alloc_fragment(mhc_mult, T.float32)
                T.reduce_max(cm, row_max, dim=1)
                for j, k in T.Parallel(mhc_mult, mhc_mult):
                    cm[j, k] = T.exp(cm[j, k] - row_max[j])
                T.reduce_sum(cm, row_sum, dim=1)
                for j, k in T.Parallel(mhc_mult, mhc_mult):
                    cm[j, k] = cm[j, k] / row_sum[j] + mhc_sinkhorn_eps

                # comb = comb / (comb.sum(-2) + eps)
                T.reduce_sum(cm, col_sum, dim=0)
                for j, k in T.Parallel(mhc_mult, mhc_mult):
                    cm[j, k] = cm[j, k] / (col_sum[k] + mhc_sinkhorn_eps)

                for _ in T.serial(sinkhorn_repeat - 1):
                    # comb = comb / (comb.sum(-1) + eps)
                    T.reduce_sum(cm, row_sum, dim=1)
                    for j, k in T.Parallel(mhc_mult, mhc_mult):
                        cm[j, k] = cm[j, k] / (row_sum[j] + mhc_sinkhorn_eps)

                    # comb = comb / (comb.sum(-2) + eps)
                    T.reduce_sum(cm, col_sum, dim=0)
                    for j, k in T.Parallel(mhc_mult, mhc_mult):
                        cm[j, k] = cm[j, k] / (col_sum[k] + mhc_sinkhorn_eps)

                # save comb_mix to global memory
                for j, k in T.Parallel(mhc_mult, mhc_mult):
                    comb_mix[pid, j * mhc_mult + k] = cm[j, k]
            else:
                ##################################################################
                # _mhc_pre_split_mixes_fwd (pre)
                pre_mix_shared = T.alloc_shared(mhc_mult, T.float32)
                for j in T.Parallel(mhc_mult):
                    pre_mix_shared[j] = (
                        T.sigmoid(
                            mixes_shared[j] * mhc_scale[0] + mhc_base[j],
                        )
                        + mhc_pre_eps
                    )
                ###################################################################
                # _mhc_pre_apply_mix_fwd
                if fuse_norm:
                    output_shared = T.alloc_shared(hidden_size, T.bfloat16)
                    sumsq_per_pos = T.alloc_fragment(hidden_block, T.float32)
                    T.clear(sumsq_per_pos)
                for i0_h in T.Pipelined(hidden_size // hidden_block, num_stages=2):
                    xs = T.alloc_shared((mhc_mult, hidden_block), T.bfloat16)
                    xl = T.alloc_fragment((mhc_mult, hidden_block), T.float32)
                    T.copy(residual[pid, 0, i0_h * hidden_block], xs, disable_tma=True)
                    T.copy(xs, xl, disable_tma=True)

                    ol = T.alloc_fragment(hidden_block, T.float32)
                    T.clear(ol)

                    for i_mhc in T.serial(mhc_mult):
                        pre = pre_mix_shared[i_mhc]
                        for i1_h in T.Parallel(hidden_block):
                            ol[i1_h] += pre * xl[i_mhc, i1_h]

                    if fuse_norm:
                        # Match the SGLang fused boundary: the RMS denominator
                        # uses the FP32 readout, while the numerator retains the
                        # BF16 activation boundary before applying norm weights.
                        for i1_h in T.Parallel(hidden_block):
                            sumsq_per_pos[i1_h] += ol[i1_h] * ol[i1_h]
                            output_shared[i0_h * hidden_block + i1_h] = T.cast(
                                ol[i1_h], T.bfloat16
                            )
                    else:
                        T.copy(
                            ol, layer_input[pid, i0_h * hidden_block], disable_tma=True
                        )

                if fuse_norm:
                    sumsq = T.alloc_fragment(1, T.float32)
                    T.reduce_sum(sumsq_per_pos, sumsq, dim=0)
                    inv_rms = T.alloc_fragment(1, T.float32)
                    inv_rms[0] = T.rsqrt(sumsq[0] / hidden_size + norm_eps)
                    for i0_h in T.Pipelined(hidden_size // hidden_block, num_stages=2):
                        ws = T.alloc_shared(hidden_block, T.bfloat16)
                        wl = T.alloc_fragment(hidden_block, T.float32)
                        T.copy(norm_weight[i0_h * hidden_block], ws, disable_tma=True)
                        T.copy(ws, wl, disable_tma=True)
                        normalized = T.alloc_fragment(hidden_block, T.float32)
                        for i1_h in T.Parallel(hidden_block):
                            normalized[i1_h] = (
                                T.cast(
                                    output_shared[i0_h * hidden_block + i1_h], T.float32
                                )
                                * inv_rms[0]
                                * wl[i1_h]
                            )
                        T.copy(
                            normalized,
                            layer_input[pid, i0_h * hidden_block],
                            disable_tma=True,
                        )

    return mhc_pre_big_fuse
