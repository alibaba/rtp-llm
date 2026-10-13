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
def _mhc_post_fwd(
    mhc: int, hidden: int, n_thr: int = 128, h_blk: int = 1024
) -> tilelang.JITKernel:
    n = T.dynamic("num_tokens")
    h = hidden

    h_blk = math.gcd(hidden, h_blk)

    @T.prim_func
    def _mhc_post_fwd_kernel(
        a: T.Tensor[(n, mhc, mhc), T.float32],
        b: T.Tensor[(n, mhc, h), T.bfloat16],
        c: T.Tensor[(n, mhc), T.float32],
        d: T.Tensor[(n, h), T.bfloat16],
        x: T.Tensor[(n, mhc, h), T.bfloat16],
    ) -> None:
        with T.Kernel(n, threads=n_thr) as pid_n:
            x_shared = T.alloc_shared((mhc, h_blk), T.bfloat16)
            b_shared = T.alloc_shared((mhc, h_blk), T.bfloat16)
            d_shared = T.alloc_shared(h_blk, T.bfloat16)

            x_local = T.alloc_fragment((mhc, h_blk), T.float32)
            b_local = T.alloc_fragment((mhc, h_blk), T.float32)
            d_local = T.alloc_fragment(h_blk, T.float32)

            a_local = T.alloc_fragment((mhc, mhc), T.float32)
            c_local = T.alloc_fragment(mhc, T.float32)
            T.copy(a[pid_n, 0, 0], a_local)
            T.copy(c[pid_n, 0], c_local)
            T.pdl_sync()

            for i0_h in T.Pipelined(T.ceildiv(h, h_blk), num_stages=2):
                T.copy(b[pid_n, 0, i0_h * h_blk], b_shared, disable_tma=True)
                T.copy(d[pid_n, i0_h * h_blk], d_shared, disable_tma=True)

                T.copy(b_shared, b_local)
                T.copy(d_shared, d_local)
                for i_mhco, i1_h in T.Parallel(mhc, h_blk):
                    x_local[i_mhco, i1_h] = c_local[i_mhco] * d_local[i1_h]
                    for i_mhci in T.serial(mhc):
                        x_local[i_mhco, i1_h] += (
                            a_local[i_mhci, i_mhco] * b_local[i_mhci, i1_h]
                        )
                T.copy(x_local, x_shared)

                T.copy(x_shared, x[pid_n, 0, i0_h * h_blk], disable_tma=True)

    return _mhc_post_fwd_kernel


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,
        tilelang.PassConfigKey.TL_DISABLE_VECTORIZE_256: True,
    },
    out_idx=[5, 6, 7, 8],
)
def _mhc_post_bwd(
    mhc: int, hidden: int, n_thr: int = 128, h_blk: int = 256
) -> tilelang.JITKernel:
    assert mhc == 4
    n = T.dynamic("num_tokens")
    h = hidden

    h_blk = math.gcd(hidden, h_blk)

    @T.prim_func
    def _mhc_post_bwd_kernel(
        dx: T.Tensor[(n, 4, h), T.bfloat16],
        a: T.Tensor[(n, 4, 4), T.float32],
        b: T.Tensor[(n, 4, h), T.bfloat16],
        c: T.Tensor[(n, 4), T.float32],
        d: T.Tensor[(n, h), T.bfloat16],
        da: T.Tensor[(n, 4, 4), T.float32],
        db: T.Tensor[(n, 4, h), T.bfloat16],
        dc: T.Tensor[(n, 4), T.float32],
        dd: T.Tensor[(n, h), T.bfloat16],
    ) -> None:
        with T.Kernel(n, threads=n_thr) as pid_n:
            dx_shared = T.alloc_shared((4, h_blk), T.bfloat16)
            b_shared = T.alloc_shared((4, h_blk), T.bfloat16)
            db_shared = T.alloc_shared((4, h_blk), T.bfloat16)
            d_shared = T.alloc_shared(h_blk, T.bfloat16)
            dd_shared = T.alloc_shared(h_blk, T.bfloat16)

            dx_local = T.alloc_fragment((4, h_blk), T.float32)
            b_local = T.alloc_fragment((4, h_blk), T.float32)
            db_local = T.alloc_fragment((4, h_blk), T.float32)
            d_local = T.alloc_fragment(h_blk, T.float32)
            dd_local = T.alloc_fragment(h_blk, T.float32)

            a_local = T.alloc_fragment((4, 4), T.float32)
            c_local = T.alloc_fragment(4, T.float32)
            T.copy(a[pid_n, 0, 0], a_local)
            T.copy(c[pid_n, 0], c_local)

            da_reducer = T.alloc_reducer((4, 4), T.float32, replication="all")
            dc_reducer = T.alloc_reducer(4, T.float32, replication="all")
            T.clear(da_reducer)
            T.clear(dc_reducer)

            for i0_h in T.Pipelined(T.ceildiv(h, h_blk), num_stages=3):
                T.copy(dx[pid_n, 0, i0_h * h_blk], dx_shared, disable_tma=True)
                T.copy(b[pid_n, 0, i0_h * h_blk], b_shared, disable_tma=True)
                T.copy(d[pid_n, i0_h * h_blk], d_shared, disable_tma=True)

                T.copy(dx_shared, dx_local)
                T.copy(b_shared, b_local)
                T.copy(d_shared, d_local)

                # da and db
                T.clear(db_local)
                for i_mhci in T.serial(4):
                    for i_mhco in T.serial(4):
                        for i1_h in T.Parallel(h_blk):
                            db_local[i_mhci, i1_h] += (
                                a_local[i_mhci, i_mhco] * dx_local[i_mhco, i1_h]
                            )
                            da_reducer[i_mhci, i_mhco] += (
                                b_local[i_mhci, i1_h] * dx_local[i_mhco, i1_h]
                            )

                # dc and dd
                T.clear(dd_local)
                for i_mhc in T.serial(4):
                    for i1_h in T.Parallel(h_blk):
                        dc_reducer[i_mhc] += d_local[i1_h] * dx_local[i_mhc, i1_h]
                        dd_local[i1_h] += c_local[i_mhc] * dx_local[i_mhc, i1_h]

                T.copy(db_local, db_shared)
                T.copy(dd_local, dd_shared)

                T.copy(db_shared, db[pid_n, 0, i0_h * h_blk], disable_tma=True)
                T.copy(dd_shared, dd[pid_n, i0_h * h_blk], disable_tma=True)

            T.finalize_reducer(da_reducer)
            T.finalize_reducer(dc_reducer)
            T.copy(da_reducer, da[pid_n, 0, 0])
            T.copy(dc_reducer, dc[pid_n, 0])

    return _mhc_post_bwd_kernel


def mhc_post_fwd(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    num_seqs, num_tokens, mhc, hidden = residual.shape

    assert x.dtype == torch.bfloat16, f"{x.dtype=}"
    assert residual.dtype == torch.bfloat16, f"{residual.dtype=}"
    assert post_layer_mix.dtype == torch.float32, f"{post_layer_mix.dtype=}"
    assert comb_res_mix.dtype == torch.float32, f"{comb_res_mix.dtype=}"
    assert x.shape == (num_seqs, num_tokens, hidden), f"{x.shape=}"
    assert post_layer_mix.shape == (
        num_seqs,
        num_tokens,
        mhc,
        1,
    ), f"{post_layer_mix.shape=}"
    assert comb_res_mix.shape == (
        num_seqs,
        num_tokens,
        mhc,
        mhc,
    ), f"{comb_res_mix.shape=}"

    residual = residual.contiguous()
    assert x.is_contiguous()
    assert post_layer_mix.is_contiguous()
    assert comb_res_mix.is_contiguous()

    if out is None:
        out = torch.empty_like(residual)
    kernel = _mhc_post_fwd(mhc, hidden)
    kernel(
        comb_res_mix.flatten(0, 1),
        residual.flatten(0, 1),
        post_layer_mix.flatten(0, 1).squeeze(-1),
        x.flatten(0, 1),
        out.flatten(0, 1),
    )
    return out


def mhc_post_bwd(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    d_o: torch.Tensor,
    fuse_grad_acc: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n = d_o.shape[0] * d_o.shape[1]
    mhc = d_o.shape[2]
    h = d_o.shape[3]

    bwd_kernel = _mhc_post_bwd(mhc, h)
    (
        d_comb_res_mix,
        d_residual,
        d_post_layer_mix,
        d_x,
    ) = bwd_kernel(
        d_o.contiguous().view(n, mhc, h),
        comb_res_mix.view(n, mhc, mhc),
        residual.view(n, mhc, h),
        post_layer_mix.view(n, mhc),
        x.view(n, h),
    )
    assert isinstance(d_x, torch.Tensor)
    assert isinstance(d_post_layer_mix, torch.Tensor)
    assert isinstance(d_comb_res_mix, torch.Tensor)
    assert isinstance(d_residual, torch.Tensor)
    if fuse_grad_acc:
        residual.untyped_storage().grad_from_mhc_post = d_residual

    return (
        d_x.view_as(x),
        d_residual.view_as(residual),
        d_post_layer_mix.view_as(post_layer_mix),
        d_comb_res_mix.view_as(comb_res_mix),
    )
