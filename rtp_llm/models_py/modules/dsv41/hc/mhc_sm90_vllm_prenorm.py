# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM delayed mHC pre-mix, Sinkhorn, readout and RMSNorm second kernel.

Ported from vLLM commit 0c81a0371f586cc50a0daad6d216fbac21cf1a2d
``vllm/model_executor/kernels/mhc/tilelang_kernels.py``. Candidate only.
"""

import math
import tilelang
from tilelang import language as T
from rtp_llm.models_py.modules.dsv4 import tilelang_kernels as _tilelang_kernels  # noqa: F401

ENABLE_PDL = False
_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,
}


def _is_mix_thread():
    return T.get_thread_binding() < 32


def _is_collapse_thread():
    return T.get_thread_binding() >= 32


@tilelang.jit(pass_configs=_PASS_CONFIGS)
def mhc_pre_big_fuse_with_norm_tilelang(
    gemm_out_mul,
    gemm_out_sqrsum,
    hc_scale,
    hc_base,
    residual,
    post_mix,
    comb_mix,
    layer_input,
    norm_weight,
    pre_mix_in,
    pre_mix_out,
    aux_out,
    hidden_size: int,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    norm_eps: float,
    n_splits: int = 16,
    hc_mult: int = 4,
    gemm_last_dim: int = -1,
    use_pre_mix_in: bool = False,
    save_pre_mix: bool = False,
    rms_numel: int = 0,
    split_mode: str = "fused",
    write_aux: bool = False,
):
    # Split modes require shifted mHC: input collapse uses a carried pre-mix.
    assert split_mode in ("fused", "stats", "input")
    assert split_mode == "fused" or save_pre_mix
    num_tokens = T.dynamic("num_tokens")
    hc_mult3 = hc_mult * (2 + hc_mult)
    if gemm_last_dim < 0:
        gemm_last_dim = hc_mult3
    if rms_numel == 0:
        rms_numel = hc_mult * hidden_size
    hidden_block = math.gcd(1024, hidden_size)

    gemm_out_mul: T.Tensor[[n_splits, num_tokens, gemm_last_dim], T.float32]  # type: ignore[no-redef, valid-type]
    gemm_out_sqrsum: T.Tensor[[n_splits, num_tokens], T.float32]  # type: ignore[no-redef, valid-type]
    hc_scale: T.Tensor[[3], T.float32]  # type: ignore[no-redef, valid-type]
    hc_base: T.Tensor[[hc_mult3], T.float32]  # type: ignore[no-redef, valid-type]
    residual: T.Tensor[[num_tokens, hc_mult, hidden_size], T.bfloat16]  # type: ignore[no-redef, valid-type]
    post_mix: T.Tensor[[num_tokens, hc_mult], T.float32]  # type: ignore[no-redef, valid-type]
    comb_mix: T.Tensor[[num_tokens, hc_mult * hc_mult], T.float32]  # type: ignore[no-redef, valid-type]
    layer_input: T.Tensor[[num_tokens, hidden_size], T.bfloat16]  # type: ignore[no-redef, valid-type]
    norm_weight: T.Tensor[[hidden_size], T.bfloat16]  # type: ignore[no-redef, valid-type]

    pre_mix_in: T.Tensor[[num_tokens, hc_mult], T.float32]  # type: ignore[no-redef, valid-type]
    pre_mix_out: T.Tensor[[num_tokens, hc_mult], T.float32]  # type: ignore[no-redef, valid-type]
    aux_out: T.Tensor[[num_tokens, hidden_size], T.bfloat16]  # type: ignore[no-redef, valid-type]

    with T.Kernel(num_tokens, threads=96) as i:
        if ENABLE_PDL:
            T.pdl_sync()
        mixes_shared = T.alloc_shared(hc_mult3, T.float32)
        if split_mode != "input":
            rms = T.alloc_fragment(1, T.float32)
            mixes = T.alloc_fragment(hc_mult3, T.float32)
            T.clear(mixes)
            rms[0] = 0

            for i_split in T.serial(n_splits):
                rms[0] += gemm_out_sqrsum[i_split, i]
            rms[0] = T.rsqrt(rms[0] / rms_numel + rms_eps)
            for j in T.Parallel(hc_mult3):
                mixes[j] = 0
                for i_split in T.serial(n_splits):
                    mixes[j] += gemm_out_mul[i_split, i, j]
                mixes[j] *= rms[0]
            T.copy(mixes, mixes_shared)

        if split_mode != "input" and _is_mix_thread():
            cm = T.alloc_fragment((hc_mult, hc_mult), T.float32)
            for j in T.Parallel(hc_mult):
                if save_pre_mix:
                    pre_mix_out[i, j] = (
                        T.sigmoid(mixes_shared[j] * hc_scale[0] + hc_base[j])
                        + hc_pre_eps
                    )
                post_mix[i, j] = (
                    T.sigmoid(
                        mixes_shared[j + hc_mult] * hc_scale[1] + hc_base[j + hc_mult]
                    )
                    * hc_post_mult_value
                )
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = (
                    mixes_shared[j * hc_mult + k + hc_mult * 2] * hc_scale[2]
                    + hc_base[j * hc_mult + k + hc_mult * 2]
                )

            row_sum = T.alloc_fragment(hc_mult, T.float32)
            col_sum = T.alloc_fragment(hc_mult, T.float32)

            row_max = T.alloc_fragment(hc_mult, T.float32)
            T.reduce_max(cm, row_max, dim=1)
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = T.exp(cm[j, k] - row_max[j])
            T.reduce_sum(cm, row_sum, dim=1)
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = cm[j, k] / row_sum[j] + hc_sinkhorn_eps

            T.reduce_sum(cm, col_sum, dim=0)
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = cm[j, k] / (col_sum[k] + hc_sinkhorn_eps)

            # A NaN input row (e.g. CUDA-graph padding) has a NaN RMS, so every
            # logit and the whole result is NaN already. Skip the remaining
            # iterations: IEEE division takes a slow path on NaN. rms is an
            # rsqrt, so it is >= 0 unless NaN (T.isfinite does not lower on HIP).
            if rms[0] >= 0.0:
                for _ in T.serial(sinkhorn_repeat - 1):
                    T.reduce_sum(cm, row_sum, dim=1)
                    for j, k in T.Parallel(hc_mult, hc_mult):
                        cm[j, k] = cm[j, k] / (row_sum[j] + hc_sinkhorn_eps)

                    T.reduce_sum(cm, col_sum, dim=0)
                    for j, k in T.Parallel(hc_mult, hc_mult):
                        cm[j, k] = cm[j, k] / (col_sum[k] + hc_sinkhorn_eps)

            for j, k in T.Parallel(hc_mult, hc_mult):
                comb_mix[i, j * hc_mult + k] = cm[j, k]
        elif split_mode != "stats" and _is_collapse_thread():
            pre_mix_shared = T.alloc_shared(hc_mult, T.float32)
            for j in T.Parallel(hc_mult):
                if use_pre_mix_in:
                    pre_mix_shared[j] = pre_mix_in[i, j]
                elif save_pre_mix:
                    pre_mix_shared[j] = T.if_then_else(j == 0, 1.0, 0.0)
                else:
                    pre_mix_shared[j] = (
                        T.sigmoid(mixes_shared[j] * hc_scale[0] + hc_base[j])
                        + hc_pre_eps
                    )

            # Pass 1: stash unnormalized weighted-sum output in shared memory
            # as bf16 (matches the rounding that RMSNorm would see) while
            # accumulating the per-position squared sum.
            output_shared = T.alloc_shared(hidden_size, T.bfloat16)
            sumsq_per_pos = T.alloc_fragment(hidden_block, T.float32)
            T.clear(sumsq_per_pos)

            for i0_h in T.Pipelined(hidden_size // hidden_block, num_stages=2):
                xs = T.alloc_shared((hc_mult, hidden_block), T.bfloat16)
                xl = T.alloc_fragment((hc_mult, hidden_block), T.float32)
                T.copy(residual[i, 0, i0_h * hidden_block], xs)
                T.copy(xs, xl)

                ol = T.alloc_fragment(hidden_block, T.float32)
                T.clear(ol)
                if write_aux:
                    # Aux consumers (draft models) take the plain stream mean
                    # of the same residual this collapse reads.
                    aux = T.alloc_fragment(hidden_block, T.float32)
                    T.clear(aux)

                for i_hc in T.serial(hc_mult):
                    pre = pre_mix_shared[i_hc]
                    for i1_h in T.Parallel(hidden_block):
                        ol[i1_h] += pre * xl[i_hc, i1_h]
                        if write_aux:
                            aux[i1_h] += xl[i_hc, i1_h]

                if write_aux:
                    for i1_h in T.Parallel(hidden_block):
                        aux_out[i, i0_h * hidden_block + i1_h] = T.bfloat16(
                            aux[i1_h] / hc_mult
                        )

                if save_pre_mix:
                    # Keep the BF16 boundary before the delayed input RMSNorm.
                    rounded = T.alloc_fragment(hidden_block, T.bfloat16)
                    T.copy(ol, rounded)
                    for i1_h in T.Parallel(hidden_block):
                        value = T.float32(rounded[i1_h])
                        sumsq_per_pos[i1_h] += value * value
                        output_shared[i0_h * hidden_block + i1_h] = rounded[i1_h]
                else:
                    for i1_h in T.Parallel(hidden_block):
                        sumsq_per_pos[i1_h] += ol[i1_h] * ol[i1_h]
                        output_shared[i0_h * hidden_block + i1_h] = T.bfloat16(ol[i1_h])

            sumsq = T.alloc_fragment(1, T.float32)
            T.reduce_sum(sumsq_per_pos, sumsq, dim=0)
            rsqrt_norm = T.alloc_fragment(1, T.float32)
            rsqrt_norm[0] = T.rsqrt(sumsq[0] / hidden_size + norm_eps)

            # Pass 2: scale by rsqrt * norm_weight and write the result to HBM.
            for i0_h in T.Pipelined(hidden_size // hidden_block, num_stages=2):
                w_shared = T.alloc_shared(hidden_block, T.bfloat16)
                w_local = T.alloc_fragment(hidden_block, T.float32)
                T.copy(norm_weight[i0_h * hidden_block], w_shared)
                T.copy(w_shared, w_local)

                ol = T.alloc_fragment(hidden_block, T.float32)
                for i1_h in T.Parallel(hidden_block):
                    ol[i1_h] = (
                        output_shared[i0_h * hidden_block + i1_h]
                        * rsqrt_norm[0]
                        * w_local[i1_h]
                    )

                T.copy(ol, layer_input[i, i0_h * hidden_block])

        if ENABLE_PDL:
            T.pdl_trigger()


__all__ = ["mhc_pre_big_fuse_with_norm_tilelang"]
