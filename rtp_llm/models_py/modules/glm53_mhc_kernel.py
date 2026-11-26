# Source: RTP-LLM e1106ec4b506f93f38429e499751ec522b58328e.

"""GLM53's source fixed-FP32 post/pre projection stage.

Keep this module lazy-imported by glm53_mhc so CPU/model imports do not load
TileLang. The decorator and kernel below are copied without mathematical edits.
"""

import tilelang
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
