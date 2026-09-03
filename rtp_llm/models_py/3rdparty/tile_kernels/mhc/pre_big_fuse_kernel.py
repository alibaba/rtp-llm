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
def _mhc_fused_post_pre_fwd(
    mhc: int,
    hidden: int,
    n_out: int,
    n_thr: int = 32,
    tile_n: int = 1,
    n_splits: int = 1,
) -> tilelang.JITKernel:
    """Fuse one mHC writeback with the next unit's pre-norm projection.

    This is the small-token decode kernel used by vLLM/SGLang's GLM-5 mHC
    path, adapted to the vendored TileLang API.  Each block owns one token,
    one output tile and one K split.  Only output tile zero materializes the
    BF16 residual and squared sum; every tile accumulates its slice of the
    next unit's 24 FP32 mix logits.  The vendored TileLang lowers indexed
    cross-warp shared stores incorrectly, so one warp per block is required
    here; split-K supplies enough blocks for decode occupancy.
    """

    num_tokens = T.dynamic("num_tokens")
    h_per_split = hidden // n_splits
    n_tiles = n_out // tile_n
    h_iters = h_per_split // n_thr
    num_warps = n_thr // 32

    assert mhc == 4
    assert n_out % tile_n == 0
    assert hidden % n_splits == 0
    assert h_per_split % n_thr == 0

    @T.prim_func
    def mhc_fused_post_pre_kernel(
        comb_mix: T.Tensor[(num_tokens, mhc, mhc), T.float32],
        residual_in: T.Tensor[(num_tokens, mhc, hidden), T.bfloat16],
        post_mix: T.Tensor[(num_tokens, mhc), T.float32],
        x_in: T.Tensor[(num_tokens, hidden), T.bfloat16],
        weight: T.Tensor[(n_out, mhc, hidden), T.float32],
        gemm_out: T.Tensor[(n_splits, num_tokens, n_out), T.float32],
        sqrsum_out: T.Tensor[(n_splits, num_tokens), T.float32],
        residual_out: T.Tensor[(num_tokens, mhc, hidden), T.bfloat16],
    ) -> None:
        with T.Kernel(num_tokens, n_tiles, n_splits, threads=n_thr) as (
            token_idx,
            out_tile,
            split_idx,
        ):
            tid = T.get_thread_binding()
            warp_id = tid // 32
            lane = tid % 32

            warp_sums = T.alloc_shared((num_warps, tile_n + 1), T.float32)
            post_shared = T.alloc_shared((mhc,), T.float32)
            comb_shared = T.alloc_shared((mhc, mhc), T.float32)

            post_local = T.alloc_fragment((mhc,), T.float32)
            comb_local = T.alloc_fragment((mhc, mhc), T.float32)
            accum = T.alloc_fragment((tile_n,), T.float32)
            sqrsum = T.alloc_fragment((1,), T.float32)
            new_residual = T.alloc_fragment((mhc,), T.float32)
            T.clear(accum)
            T.clear(sqrsum)

            T.copy(post_mix[token_idx, 0], post_shared, disable_tma=True)
            T.copy(comb_mix[token_idx, 0, 0], comb_shared, disable_tma=True)
            for out_hc in T.unroll(mhc):
                post_local[out_hc] = post_shared[out_hc]
            for in_hc in T.unroll(mhc):
                for out_hc in T.unroll(mhc):
                    comb_local[in_hc, out_hc] = comb_shared[in_hc, out_hc]

            split_start = split_idx * h_per_split
            for h_iter in T.serial(h_iters):
                h_idx = split_start + h_iter * n_thr + tid
                x_value = x_in[token_idx, h_idx]
                for out_hc in T.unroll(mhc):
                    new_residual[out_hc] = post_local[out_hc] * x_value
                    for in_hc in T.unroll(mhc):
                        new_residual[out_hc] += (
                            comb_local[in_hc, out_hc]
                            * residual_in[token_idx, in_hc, h_idx]
                        )
                    # The standalone post kernel materializes its FP32 FMA
                    # result through BF16 global memory before the following
                    # pre reads it.  Preserve that boundary in registers;
                    # otherwise the fused projection and RMS sum observe
                    # extra FP32 precision and subtly change model semantics.
                    new_residual[out_hc] = T.cast(
                        T.cast(new_residual[out_hc], T.bfloat16), T.float32
                    )

                if out_tile == 0:
                    for out_hc in T.unroll(mhc):
                        residual_out[token_idx, out_hc, h_idx] = new_residual[out_hc]
                        sqrsum[0] += new_residual[out_hc] * new_residual[out_hc]

                for out_lane in T.unroll(tile_n):
                    output_idx = out_tile * tile_n + out_lane
                    for out_hc in T.unroll(mhc):
                        accum[out_lane] += (
                            weight[output_idx, out_hc, h_idx] * new_residual[out_hc]
                        )

            for out_lane in T.unroll(tile_n):
                accum[out_lane] = T.warp_reduce_sum(accum[out_lane])
            if out_tile == 0:
                sqrsum[0] = T.warp_reduce_sum(sqrsum[0])

            if lane == 0:
                for out_lane in T.unroll(tile_n):
                    warp_sums[warp_id, out_lane] = accum[out_lane]
                if out_tile == 0:
                    warp_sums[warp_id, tile_n] = sqrsum[0]
            T.sync_threads()

            if warp_id == 0:
                if lane < tile_n:
                    value = T.alloc_var(T.float32, init=0.0)
                    for warp in T.unroll(num_warps):
                        value += warp_sums[warp, lane]
                    gemm_out[split_idx, token_idx, out_tile * tile_n + lane] = value
                if out_tile == 0 and lane == 0:
                    value = T.alloc_var(T.float32, init=0.0)
                    for warp in T.unroll(num_warps):
                        value += warp_sums[warp, tile_n]
                    sqrsum_out[split_idx, token_idx] = value

    return mhc_fused_post_pre_kernel


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
    ) -> None:
        with T.Kernel(num_tokens, threads=96) as pid:
            ##################################################################
            # _mhc_pre_norm_fn_fwd_norm
            mixes_shared = T.alloc_shared(mhc_mult3, T.float32)
            if T.get_thread_binding() < 32:
                rms = T.alloc_fragment(1, T.float32)
                mixes = T.alloc_fragment(mhc_mult3, T.float32)
                T.clear(mixes)
                rms[0] = 0
                for i_split in T.serial(n_splits):
                    rms[0] += gemm_out_sqrsum[i_split, pid]
                rms[0] = T.rsqrt(rms[0] / (mhc_mult * hidden_size) + rms_eps)
                for j in T.Parallel(mhc_mult3):
                    mixes[j] = 0
                    for i_split in T.serial(n_splits):
                        mixes[j] += gemm_out_mul[i_split, pid, j]
                    mixes[j] *= rms[0]
                T.copy(mixes, mixes_shared, disable_tma=True)

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

                    T.copy(ol, layer_input[pid, i0_h * hidden_block], disable_tma=True)

    return mhc_pre_big_fuse
