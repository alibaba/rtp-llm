import math

import tilelang
from tilelang import language as T


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,
    },
)
def _mhc_head_fuse(
    hidden_size: int,
    rms_eps: float,
    mhc_pre_eps: float,
    mhc_mult: int = 4,
    n_thr: int = 128,
    h_blk: int = 1024,
):
    num_tokens = T.dynamic("num_tokens")
    mhc_hidden_size = mhc_mult * hidden_size
    hidden_block = math.gcd(h_blk, hidden_size)
    n_hidden_blocks = hidden_size // hidden_block

    @T.prim_func
    def mhc_head_fuse(
        residual: T.Tensor[(num_tokens, mhc_mult, hidden_size), T.bfloat16],
        fn: T.Tensor[(mhc_mult, mhc_hidden_size), T.float32],
        mhc_scale: T.Tensor[(1,), T.float32],
        mhc_base: T.Tensor[(mhc_mult,), T.float32],
        out: T.Tensor[(num_tokens, hidden_size), T.bfloat16],
    ) -> None:
        with T.Kernel(num_tokens, threads=n_thr) as pid:
            T.pdl_sync()

            sqrsum = T.alloc_reducer((1,), T.float32, replication="all")
            mixes = T.alloc_reducer((mhc_mult,), T.float32, replication="all")
            T.fill(sqrsum, 0.0)
            T.fill(mixes, 0.0)

            for i_mhc in T.serial(mhc_mult):
                for i_h in T.serial(n_hidden_blocks):
                    x_local = T.alloc_fragment(hidden_block, T.float32)
                    T.copy(residual[pid, i_mhc, i_h * hidden_block], x_local)

                    for k in T.Parallel(hidden_block):
                        sqrsum[0] += x_local[k] * x_local[k]

                    for mix_id in T.unroll(mhc_mult):
                        fn_local = T.alloc_fragment(hidden_block, T.float32)
                        T.copy(
                            fn[
                                mix_id,
                                i_mhc * hidden_size + i_h * hidden_block,
                            ],
                            fn_local,
                        )
                        for k in T.Parallel(hidden_block):
                            mixes[mix_id] += x_local[k] * fn_local[k]

            T.finalize_reducer(sqrsum)
            T.finalize_reducer(mixes)

            pre_mix = T.alloc_shared(mhc_mult, T.float32)
            rms = T.alloc_fragment(1, T.float32)
            rms[0] = T.rsqrt(sqrsum[0] / mhc_hidden_size + rms_eps)
            for mix_id in T.Parallel(mhc_mult):
                pre_mix[mix_id] = (
                    T.sigmoid(
                        mixes[mix_id] * rms[0] * mhc_scale[0] + mhc_base[mix_id]
                    )
                    + mhc_pre_eps
                )

            for i_h in T.Pipelined(n_hidden_blocks, num_stages=2):
                xs = T.alloc_shared((mhc_mult, hidden_block), T.bfloat16)
                xl = T.alloc_fragment((mhc_mult, hidden_block), T.float32)
                T.copy(residual[pid, 0, i_h * hidden_block], xs, disable_tma=True)
                T.copy(xs, xl)

                out_local = T.alloc_fragment(hidden_block, T.float32)
                T.clear(out_local)
                for i_mhc in T.serial(mhc_mult):
                    for k in T.Parallel(hidden_block):
                        out_local[k] += pre_mix[i_mhc] * xl[i_mhc, k]

                T.copy(out_local, out[pid, i_h * hidden_block], disable_tma=True)

            T.pdl_trigger()

    return mhc_head_fuse
