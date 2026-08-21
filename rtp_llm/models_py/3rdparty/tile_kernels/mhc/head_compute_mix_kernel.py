import tilelang
import torch
from tilelang import language as T

_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
}


@tilelang.jit(pass_configs=_PASS_CONFIGS)
def _mhc_head_compute_mix_fwd(
    mhc_mult: int,
    mhc_pre_eps: float,
    token_block_size: int,
) -> tilelang.JITKernel:
    num_tokens = T.dynamic("num_tokens")

    @T.prim_func
    def mhc_head_compute_mix_fwd_kernel(
        # Input
        input_mix: T.Tensor[(num_tokens, mhc_mult), T.float32],
        mhc_scale: T.Tensor[(1,), T.float32],
        mhc_base: T.Tensor[(mhc_mult,), T.float32],
        # Output
        output_mix: T.Tensor[(num_tokens, mhc_mult), T.float32],
    ) -> None:
        with T.Kernel(T.ceildiv(num_tokens, token_block_size)) as pid:
            for i1, j in T.Parallel(token_block_size, mhc_mult):
                i = pid * token_block_size + i1
                if i < num_tokens:
                    output_mix[i, j] = (
                        T.sigmoid(input_mix[i, j] * mhc_scale[0] + mhc_base[j])
                        + mhc_pre_eps
                    )

    return mhc_head_compute_mix_fwd_kernel


@tilelang.jit(pass_configs=_PASS_CONFIGS)
def _mhc_head_compute_mix_bwd(
    mhc_mult: int,
    token_block_size: int,
    num_sms: int,
) -> tilelang.JITKernel:
    num_tokens = T.dynamic("num_tokens")

    @T.prim_func
    def mhc_head_compute_mix_bwd_kernel(
        # Gradient of output
        output_mix_grad: T.Tensor[(num_tokens, mhc_mult), T.float32],
        # Cached activation
        input_mix: T.Tensor[(num_tokens, mhc_mult), T.float32],
        mhc_scale: T.Tensor[(1,), T.float32],
        mhc_base: T.Tensor[(mhc_mult,), T.float32],
        # Gradient of input
        input_mix_grad: T.Tensor[(num_tokens, mhc_mult), T.float32],
        mhc_scale_grad_partial: T.Tensor[(num_sms, 1), T.float32],
        mhc_base_grad_partial: T.Tensor[(num_sms, mhc_mult), T.float32],
    ) -> None:
        with T.Kernel(num_sms) as pid:
            mhc_scale_grad_reducer = T.alloc_reducer(1, T.float32, replication="all")
            mhc_base_grad_reducer = T.alloc_reducer(
                mhc_mult, T.float32, replication="all"
            )
            T.fill(mhc_scale_grad_reducer, 0)
            T.fill(mhc_base_grad_reducer, 0)
            for t in T.Persistent(
                [T.ceildiv(num_tokens, token_block_size)],
                num_sms,
                pid,
                group_size=1,
            ):
                grad_frag = T.alloc_fragment((token_block_size, mhc_mult), T.float32)
                input_recompute_frag = T.alloc_fragment(
                    (token_block_size, mhc_mult), T.float32
                )
                for i1, j in T.Parallel(token_block_size, mhc_mult):
                    i = t * token_block_size + i1
                    if i < num_tokens:
                        input_recompute_frag[i1, j] = T.sigmoid(
                            input_mix[i, j] * mhc_scale[0] + mhc_base[j],
                        )
                        grad_frag[i1, j] = (
                            input_recompute_frag[i1, j]
                            * (1 - input_recompute_frag[i1, j])
                            * output_mix_grad[i, j]
                        )
                        input_mix_grad[i, j] = grad_frag[i1, j] * mhc_scale[0]
                        mhc_scale_grad_reducer[0] += grad_frag[i1, j] * input_mix[i, j]
                        mhc_base_grad_reducer[j] += grad_frag[i1, j]
            T.finalize_reducer(mhc_scale_grad_reducer)
            T.finalize_reducer(mhc_base_grad_reducer)
            T.copy(mhc_scale_grad_reducer, mhc_scale_grad_partial[pid, :])
            T.copy(mhc_base_grad_reducer, mhc_base_grad_partial[pid, :])

    return mhc_head_compute_mix_bwd_kernel
