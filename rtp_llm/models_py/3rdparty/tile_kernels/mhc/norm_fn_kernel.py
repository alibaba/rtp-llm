import tilelang
import torch
from tilelang import language as T

_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_WGMMA: True,
}


@tilelang.jit
def _mhc_fn_normw_merge_fwd(
    m: int, n: int, dtype: T.dtype = T.float32
) -> tilelang.JITKernel:
    n_blk = 256

    @T.prim_func
    def _mhc_fn_normw_merge_fwd_(
        fn: T.Tensor[(m, n), dtype],
        normw: T.Tensor[n, dtype],
        out_fn: T.Tensor[(m, n), dtype],
    ) -> None:
        _ = dtype
        with T.Kernel(m, T.ceildiv(n, n_blk)) as (pid_m, pid_n):
            for i1_n in T.Parallel(n_blk):
                i_n = pid_n * n_blk + i1_n
                if i_n < n:
                    out_fn[pid_m, i_n] = fn[pid_m, i_n] * normw[i_n]

    return _mhc_fn_normw_merge_fwd_


@tilelang.jit
def _mhc_fn_normw_merge_bwd(
    m: int, n: int, dtype: T.dtype = T.float32
) -> tilelang.JITKernel:
    n_blk = 256

    @T.prim_func
    def _mhc_fn_normw_merge_bwd_(
        fn: T.Tensor[(m, n), dtype],
        normw: T.Tensor[n, dtype],
        out_fn_grad: T.Tensor[(m, n), dtype],
        fn_grad: T.Tensor[(m, n), dtype],
        normw_grad: T.Tensor[n, dtype],
    ) -> None:
        _ = dtype
        with T.Kernel(T.ceildiv(n, n_blk)) as pid_n:
            normw_frag = T.alloc_fragment(n_blk, dtype)
            T.copy(normw[pid_n * n_blk], normw_frag)

            normw_grad_frag = T.alloc_fragment(n_blk, dtype)
            T.clear(normw_grad_frag)

            for i_m in T.serial(m):
                for i1_n in T.Parallel(n_blk):
                    i_n = pid_n * n_blk + i1_n
                    if i_n < n:
                        fn_grad[i_m, i_n] += out_fn_grad[i_m, i_n] * normw_frag[i1_n]
                        normw_grad_frag[i1_n] += out_fn_grad[i_m, i_n] * fn[i_m, i_n]

            for i1_n in T.Parallel(n_blk):
                normw_grad[pid_n * n_blk + i1_n] += normw_grad_frag[i1_n]

    return _mhc_fn_normw_merge_bwd_


@tilelang.jit(pass_configs=_PASS_CONFIGS)
def _mhc_pre_norm_fn_fwd_mul(
    mhc_mult3: int,
    n_rms_group: int,
    rms_group_size: int,
    token_block: int = 32,
    hidden_block: int = 256,
) -> tilelang.JITKernel:
    assert mhc_mult3 <= 32
    num_tokens = T.dynamic("num_tokens")
    assert rms_group_size % hidden_block == 0

    @T.prim_func
    def _mhc_pre_norm_fn_fwd_mul_kernel(
        x: T.Tensor[(num_tokens, n_rms_group * rms_group_size), T.bfloat16],
        fn: T.Tensor[(mhc_mult3, n_rms_group * rms_group_size), T.float32],
        out: T.Tensor[(num_tokens, n_rms_group, mhc_mult3), T.float32],
        sqrsum: T.Tensor[(num_tokens, n_rms_group), T.float32],
    ) -> None:
        _ = mhc_mult3
        with T.Kernel(T.ceildiv(num_tokens, token_block), n_rms_group) as (
            pid_x,
            pid_y,
        ):
            out_frag = T.alloc_fragment((token_block, 32), T.float32)
            sqrsum_part = T.alloc_fragment((token_block, 4), T.float32)
            T.clear(out_frag)
            T.clear(sqrsum_part)
            for pz in T.Pipelined(rms_group_size // hidden_block, num_stages=2):
                x_smem_16 = T.alloc_shared((token_block, hidden_block), T.bfloat16)
                fn_smem = T.alloc_shared((32, hidden_block), T.float32)

                T.annotate_layout(
                    {x_smem_16: tilelang.layout.make_swizzled_layout(x_smem_16)}
                )

                T.copy(
                    x[pid_x * token_block, pid_y * rms_group_size + pz * hidden_block],
                    x_smem_16,
                )
                T.copy(fn[0, pid_y * rms_group_size + pz * hidden_block], fn_smem)

                x_frag_16 = T.alloc_fragment((token_block, hidden_block), T.bfloat16)
                T.copy(x_smem_16, x_frag_16)
                x_frag = T.alloc_fragment((token_block, hidden_block), T.float32)
                T.copy(x_frag_16, x_frag)

                for jj in T.serial(hidden_block // 4):
                    for i, j in T.Parallel(token_block, 4):
                        sqrsum_part[i, j] += (
                            x_frag[i, jj * 4 + j] * x_frag[i, jj * 4 + j]
                        )

                T.gemm(
                    x_frag,
                    fn_smem,
                    out_frag,
                    transpose_A=False,
                    transpose_B=True,
                    clear_accum=False,
                )
            sqrsum_l = T.alloc_fragment(token_block, T.float32)
            T.reduce_sum(sqrsum_part, sqrsum_l)
            for i in T.Parallel(token_block):
                sqrsum[pid_x * token_block + i, pid_y] = sqrsum_l[i]
            for i, j in T.Parallel(token_block, 32):
                if j < 24:
                    out[pid_x * token_block + i, pid_y, j] = out_frag[i, j]

    return _mhc_pre_norm_fn_fwd_mul_kernel


@tilelang.jit(pass_configs=_PASS_CONFIGS)
def _mhc_pre_norm_fn_fwd_norm(
    mhc_mult3: int,
    n_rms_group: int,
    rms_group_size: int,
    rms_eps: float,
    n_splits: int,
) -> tilelang.JITKernel:
    num_tokens = T.dynamic("num_tokens")
    n_thr = 32

    @T.prim_func
    def _mhc_pre_norm_fn_fwd_norm_kernel(
        out_mul_splitted: T.Tensor[
            (n_splits, num_tokens, n_rms_group, mhc_mult3), T.float32
        ],
        sqrsum_splitted: T.Tensor[(n_splits, num_tokens, n_rms_group), T.float32],
        out_mul: T.Tensor[(num_tokens, n_rms_group, mhc_mult3), T.float32],
        sqrsum: T.Tensor[(num_tokens, n_rms_group), T.float32],
        out: T.Tensor[(num_tokens, mhc_mult3), T.float32],
    ) -> None:
        with T.Kernel(num_tokens, threads=n_thr) as pid:
            rms = T.alloc_fragment(1, T.float32)
            out_l = T.alloc_fragment(mhc_mult3, T.float32)
            out_l0 = T.alloc_fragment(mhc_mult3, T.float32)
            T.clear(out_l)
            for k in T.serial(n_rms_group):
                rms[0] = 0
                for i_split in T.serial(n_splits):
                    rms[0] += sqrsum_splitted[i_split, pid, k]
                if T.get_thread_binding() == 0:
                    sqrsum[pid, k] = rms[0]
                rms[0] = T.rsqrt(rms[0] / rms_group_size + rms_eps)
                for j in T.Parallel(mhc_mult3):
                    out_l0[j] = 0
                    for i_split in T.serial(n_splits):
                        out_l0[j] += out_mul_splitted[i_split, pid, k, j]
                    out_l[j] += out_l0[j] * rms[0]
                T.copy(out_l0, out_mul[pid, k, :])
            T.copy(out_l[:], out[pid, :])

    return _mhc_pre_norm_fn_fwd_norm_kernel


@tilelang.jit(pass_configs=_PASS_CONFIGS)
def _mhc_pre_norm_fn_bwd_norm(
    mhc_mult3: int,
    n_rms_group: int,
    rms_group_size: int,
    rms_eps: float,
) -> tilelang.JITKernel:
    num_tokens = T.dynamic("num_tokens")
    n_thr = 32

    @T.prim_func
    def _mhc_pre_norm_fn_bwd_norm_kernel(
        # Gradient of output
        out_grad: T.Tensor[(num_tokens, mhc_mult3), T.float32],
        # Saved inputs
        out_mul: T.Tensor[(num_tokens, n_rms_group, mhc_mult3), T.float32],
        sqrsum: T.Tensor[(num_tokens, n_rms_group), T.float32],
        # Computed gradient of inputs
        out_mul_grad: T.Tensor[(num_tokens, n_rms_group, mhc_mult3), T.float32],
        sqrsum_grad: T.Tensor[(num_tokens, n_rms_group), T.float32],
    ) -> None:
        with T.Kernel(num_tokens, n_rms_group, threads=n_thr) as (pid_i, pid_k):
            sqrsum_frag = T.alloc_fragment(1, T.float32)
            sqrsum_frag[0] = sqrsum[pid_i, pid_k]
            rms_frag = T.alloc_fragment(1, T.float32)
            rms_frag[0] = T.rsqrt(sqrsum_frag[0] / rms_group_size + rms_eps)

            rms_grad_frag = T.alloc_reducer(1, T.float32, replication="all")
            T.clear(rms_grad_frag)
            for j in T.Parallel(mhc_mult3):
                out_mul_grad[pid_i, pid_k, j] = out_grad[pid_i, j] * rms_frag[0]
                rms_grad_frag[0] += out_grad[pid_i, j] * out_mul[pid_i, pid_k, j]
            T.finalize_reducer(rms_grad_frag)

            for kk in T.Parallel(1):
                sqrsum_grad[pid_i, pid_k + kk] = (
                    rms_grad_frag[kk]
                    * rms_frag[kk]
                    / (sqrsum_frag[kk] + rms_eps * rms_group_size)
                    / -2
                )

    return _mhc_pre_norm_fn_bwd_norm_kernel


@tilelang.jit(pass_configs=_PASS_CONFIGS)
def _mhc_pre_norm_fn_bwd_mul(
    mhc_mult3: int,
    n_rms_group: int,
    rms_group_size: int,
    token_block: int = 128,
    hidden_block: int = 128,
) -> tilelang.JITKernel:
    assert mhc_mult3 <= 32
    num_tokens = T.dynamic("num_tokens")
    assert rms_group_size % hidden_block == 0

    @T.prim_func
    def _mhc_pre_norm_fn_bwd_mul_kernel(
        # Gradient of output
        out_mul_grad: T.Tensor[(num_tokens, n_rms_group, mhc_mult3), T.float32],
        sqrsum_grad: T.Tensor[(num_tokens, n_rms_group), T.float32],
        # Saved inputs
        x: T.Tensor[(num_tokens, n_rms_group * rms_group_size), T.bfloat16],
        fn: T.Tensor[(mhc_mult3, n_rms_group * rms_group_size), T.float32],
        # Computed gradient of inputs
        x_grad: T.Tensor[(num_tokens, n_rms_group * rms_group_size), T.bfloat16],
        fn_grad: T.Tensor[(mhc_mult3, n_rms_group * rms_group_size), T.float32],
    ) -> None:
        with T.Kernel(n_rms_group, T.ceildiv(rms_group_size, hidden_block)) as (
            pid_y,
            pid_z,
        ):
            yz = pid_y * rms_group_size + pid_z * hidden_block

            fn_smem = T.alloc_shared((32, hidden_block), T.float32)
            for i, j in T.Parallel(32, hidden_block):
                if i < mhc_mult3:
                    fn_smem[i, j] = fn[i, yz + j]
                else:
                    fn_smem[i, j] = 0

            fn_grad_frag = T.alloc_fragment((32, hidden_block), T.float32)
            T.fill(fn_grad_frag, 0)

            for px in T.serial(T.ceildiv(num_tokens, token_block)):
                x_smem = T.alloc_shared((token_block, hidden_block), T.float32)
                T.copy(x[px * token_block, yz], x_smem)

                padded_grad = T.alloc_shared((token_block, 32), T.float32)
                for i, j in T.Parallel(token_block, 32):
                    if j < mhc_mult3:
                        padded_grad[i, j] = out_mul_grad[px * token_block + i, pid_y, j]
                    else:
                        padded_grad[i, j] = 0

                x_grad_frag = T.alloc_fragment((token_block, hidden_block), T.float32)
                T.copy(x_grad[px * token_block, yz], x_grad_frag)

                T.gemm(
                    padded_grad,
                    x_smem,
                    fn_grad_frag,
                    transpose_A=True,
                    transpose_B=False,
                    clear_accum=False,
                )
                T.gemm(
                    padded_grad,
                    fn_smem,
                    x_grad_frag,
                    transpose_A=False,
                    transpose_B=False,
                    clear_accum=False,
                )

                sqrsum_grad_frag = T.alloc_fragment((token_block, 1), T.float32)
                T.copy(sqrsum_grad[px * token_block, pid_y], sqrsum_grad_frag)
                for i, j in T.Parallel(token_block, hidden_block):
                    x_grad_frag[i, j] += 2 * x_smem[i, j] * sqrsum_grad_frag[i, 0]

                T.copy(x_grad_frag, x_grad[px * token_block, yz])

            T.copy(fn_grad_frag, fn_grad[0, yz])

    return _mhc_pre_norm_fn_bwd_mul_kernel


def round_to_tf32(x: torch.Tensor) -> torch.Tensor:
    return (x.view(torch.int32) + 0x1000).view(torch.float32)
