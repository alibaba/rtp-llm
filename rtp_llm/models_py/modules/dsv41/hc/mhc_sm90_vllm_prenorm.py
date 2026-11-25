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

_PASS_CONFIGS_FAST = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
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


@tilelang.jit(pass_configs=_PASS_CONFIGS_FAST)
def mhc_pre_collapsed_norm_tilelang(
    partials,
    square_partials,
    hc_scale,
    hc_base,
    collapsed,
    norm_weight,
    pre_mix_out,
    post_mix,
    comb_mix,
    layer_input,
    hidden_size: int,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    norm_eps: float,
    n_splits: int = 10,
    hc_mult: int = 4,
    threads: int = 256,
    rms_numel: int = 0,
):
    """Parallel SM90 epilogue consuming pre-collapsed BF16 input.

    In delayed mHC (``use_pre_mix_in=True``), the previous pre-mix is already
    known during the post/projection kernel, so that kernel writes the rounded
    BF16 collapsed input ``[M, H]`` while ``new_residual`` is still in
    registers.  This kernel eliminates the 4x5120 ``residual_out`` re-read and
    launches ``grid=(M, 2)`` (<=128 CTAs = 1 wave on 132 SMs even at M=64):

    - ``role == 1``: reduces ``partials``/``square_partials`` and runs the 4x4
      Sinkhorn iterations in registers using fast-math reciprocals.  For
      ``n_splits <= 20`` a single warp handles reduction and 4-thread
      row-parallel Sinkhorn with zero shared memory or block barriers.  For
      ``n_splits > 20`` all warps cooperatively reduce the splits first, then
      lane 0 runs a 4-way ILP unrolled Sinkhorn pass.
    - ``role == 0``: loads ``collapsed[tok, :]`` once into registers using
      128-bit ``bfloat16x8`` loads across 64 threads (matching FlashInfer's
      exact 64-thread x 8-elem reduction layout for H=5120), computes exact
      IEEE ``rsqrt``, scales by ``norm_weight``, and writes ``layer_input`` in
      parallel with Sinkhorn.
    """
    assert hc_mult == 4
    assert hidden_size == 5120
    norm_threads = 64
    norm_vec = 8
    norm_blocks = hidden_size // (norm_threads * norm_vec)  # 10 blocks of 512
    num_warps = threads // 32
    hc_mult3 = hc_mult * (2 + hc_mult)
    if rms_numel == 0:
        rms_numel = hc_mult * hidden_size
    inv_rms_numel = 1.0 / rms_numel
    coop_reduce = n_splits > 20

    num_tokens = T.dynamic("num_tokens")

    partials: T.Tensor[[n_splits, num_tokens, hc_mult3], T.float32]  # type: ignore[no-redef, valid-type]
    square_partials: T.Tensor[[n_splits, num_tokens], T.float32]  # type: ignore[no-redef, valid-type]
    hc_scale: T.Tensor[[3], T.float32]  # type: ignore[no-redef, valid-type]
    hc_base: T.Tensor[[hc_mult3], T.float32]  # type: ignore[no-redef, valid-type]
    collapsed: T.Tensor[[num_tokens, hidden_size], T.bfloat16]  # type: ignore[no-redef, valid-type]
    norm_weight: T.Tensor[[hidden_size], T.bfloat16]  # type: ignore[no-redef, valid-type]
    pre_mix_out: T.Tensor[[num_tokens, hc_mult], T.float32]  # type: ignore[no-redef, valid-type]
    post_mix: T.Tensor[[num_tokens, hc_mult], T.float32]  # type: ignore[no-redef, valid-type]
    comb_mix: T.Tensor[[num_tokens, hc_mult * hc_mult], T.float32]  # type: ignore[no-redef, valid-type]
    layer_input: T.Tensor[[num_tokens, hidden_size], T.bfloat16]  # type: ignore[no-redef, valid-type]

    with T.Kernel(num_tokens, 2, threads=threads) as (tok, role):
        tid = T.get_thread_binding()
        warp = tid // 32
        lane = tid % 32
        if role == 1:
            if coop_reduce:
                red_smem = T.alloc_shared((num_warps, hc_mult3 + 1), T.float32)
                mix_smem = T.alloc_shared((hc_mult3,), T.float32)

                acc = T.alloc_var(T.float32, init=0.0)
                if lane < hc_mult3:
                    for s_i in T.unroll((n_splits + num_warps - 1) // num_warps):
                        s = s_i * num_warps + warp
                        if s < n_splits:
                            acc += partials[s, tok, lane]
                    red_smem[warp, lane] = acc
                elif lane == hc_mult3:
                    for s_i in T.unroll((n_splits + num_warps - 1) // num_warps):
                        s = s_i * num_warps + warp
                        if s < n_splits:
                            acc += square_partials[s, tok]
                    red_smem[warp, hc_mult3] = acc
                T.sync_threads()

                if warp == 0:
                    sq_tot = T.alloc_var(T.float32, init=0.0)
                    for w_i in T.unroll(num_warps):
                        sq_tot += red_smem[w_i, hc_mult3]
                    rms_val = T.rsqrt(sq_tot * inv_rms_numel + rms_eps)

                    if lane < hc_mult3:
                        tot = T.alloc_var(T.float32, init=0.0)
                        for w_i in T.unroll(num_warps):
                            tot += red_smem[w_i, lane]
                        v_scaled = tot * rms_val
                        mix_smem[lane] = v_scaled
                        if lane < hc_mult:
                            pre_mix_out[tok, lane] = (
                                T.sigmoid(v_scaled * hc_scale[0] + hc_base[lane])
                                + hc_pre_eps
                            )
                        elif lane < 2 * hc_mult:
                            post_mix[tok, lane - hc_mult] = (
                                T.sigmoid(v_scaled * hc_scale[1] + hc_base[lane])
                                * hc_post_mult_value
                            )
                    T.sync_warp()

                    if lane == 0:
                        cm = T.alloc_local((hc_mult, hc_mult), T.float32)
                        invs = T.alloc_local((hc_mult,), T.float32)
                        s2 = hc_scale[2]
                        for r in T.unroll(hc_mult):
                            for c in T.unroll(hc_mult):
                                idx = 2 * hc_mult + r * hc_mult + c
                                cm[r, c] = mix_smem[idx] * s2 + hc_base[idx]
                        for r in T.unroll(hc_mult):
                            rm = T.max(
                                T.max(cm[r, 0], cm[r, 1]),
                                T.max(cm[r, 2], cm[r, 3]),
                            )
                            for c in T.unroll(hc_mult):
                                cm[r, c] = T.exp(cm[r, c] - rm)
                        for r in T.unroll(hc_mult):
                            invs[r] = 1.0 / (
                                (cm[r, 0] + cm[r, 1]) + (cm[r, 2] + cm[r, 3])
                            )
                        for r in T.unroll(hc_mult):
                            for c in T.unroll(hc_mult):
                                cm[r, c] = cm[r, c] * invs[r] + hc_sinkhorn_eps
                        for c in T.unroll(hc_mult):
                            invs[c] = 1.0 / (
                                (cm[0, c] + cm[1, c])
                                + (cm[2, c] + cm[3, c])
                                + hc_sinkhorn_eps
                            )
                        for c in T.unroll(hc_mult):
                            for r in T.unroll(hc_mult):
                                cm[r, c] = cm[r, c] * invs[c]
                        # NaN fast-exit: rms_val >= 0.0 is false when NaN.
                        if rms_val >= 0.0:
                            for _ in T.unroll(sinkhorn_repeat - 1):
                                for r in T.unroll(hc_mult):
                                    invs[r] = 1.0 / (
                                        (cm[r, 0] + cm[r, 1])
                                        + (cm[r, 2] + cm[r, 3])
                                        + hc_sinkhorn_eps
                                    )
                                for r in T.unroll(hc_mult):
                                    for c in T.unroll(hc_mult):
                                        cm[r, c] = cm[r, c] * invs[r]
                                for c in T.unroll(hc_mult):
                                    invs[c] = 1.0 / (
                                        (cm[0, c] + cm[1, c])
                                        + (cm[2, c] + cm[3, c])
                                        + hc_sinkhorn_eps
                                    )
                                for c in T.unroll(hc_mult):
                                    for r in T.unroll(hc_mult):
                                        cm[r, c] = cm[r, c] * invs[c]
                        for r in T.unroll(hc_mult):
                            for c in T.unroll(hc_mult):
                                comb_mix[tok, r * hc_mult + c] = cm[r, c]
            else:
                if tid < 32:
                    sq = T.alloc_var(T.float32, init=0.0)
                    for s in T.unroll(n_splits):
                        sq += square_partials[s, tok]
                    rms_val = T.rsqrt(sq * inv_rms_numel + rms_eps)

                    if tid < 2 * hc_mult:
                        acc = T.alloc_var(T.float32, init=0.0)
                        for s in T.unroll(n_splits):
                            acc += partials[s, tok, tid]
                        v_scaled = acc * rms_val
                        if tid < hc_mult:
                            pre_mix_out[tok, tid] = (
                                T.sigmoid(v_scaled * hc_scale[0] + hc_base[tid])
                                + hc_pre_eps
                            )
                        else:
                            post_mix[tok, tid - hc_mult] = (
                                T.sigmoid(v_scaled * hc_scale[1] + hc_base[tid])
                                * hc_post_mult_value
                            )

                    # All 32 lanes participate in warp shuffles unconditionally.
                    r_id = tid % hc_mult
                    row = T.alloc_local((hc_mult,), T.float32)
                    csum = T.alloc_local((hc_mult,), T.float32)
                    s2 = hc_scale[2]
                    for c in T.unroll(hc_mult):
                        idx = 2 * hc_mult + r_id * hc_mult + c
                        acc_c = T.alloc_var(T.float32, init=0.0)
                        for s in T.unroll(n_splits):
                            acc_c += partials[s, tok, idx]
                        row[c] = acc_c * rms_val * s2 + hc_base[idx]

                    rm = T.max(T.max(row[0], row[1]), T.max(row[2], row[3]))
                    for c in T.unroll(hc_mult):
                        row[c] = T.exp(row[c] - rm)
                    inv_r = T.alloc_var(
                        T.float32,
                        init=1.0 / ((row[0] + row[1]) + (row[2] + row[3])),
                    )
                    for c in T.unroll(hc_mult):
                        row[c] = row[c] * inv_r + hc_sinkhorn_eps

                    for c in T.unroll(hc_mult):
                        csum[c] = row[c] + T.shfl_xor(row[c], 1)
                    for c in T.unroll(hc_mult):
                        csum[c] = csum[c] + T.shfl_xor(csum[c], 2) + hc_sinkhorn_eps
                    for c in T.unroll(hc_mult):
                        row[c] = row[c] * (1.0 / csum[c])

                    if rms_val >= 0.0:
                        for _ in T.unroll(sinkhorn_repeat - 1):
                            inv_r = 1.0 / (
                                (row[0] + row[1])
                                + (row[2] + row[3])
                                + hc_sinkhorn_eps
                            )
                            for c in T.unroll(hc_mult):
                                row[c] = row[c] * inv_r
                            for c in T.unroll(hc_mult):
                                csum[c] = row[c] + T.shfl_xor(row[c], 1)
                            for c in T.unroll(hc_mult):
                                csum[c] = (
                                    csum[c] + T.shfl_xor(csum[c], 2) + hc_sinkhorn_eps
                                )
                            for c in T.unroll(hc_mult):
                                row[c] = row[c] * (1.0 / csum[c])

                    if tid < hc_mult:
                        for c in T.unroll(hc_mult):
                            comb_mix[tok, tid * hc_mult + c] = row[c]
        else:
            # Exact FlashInfer-matching 64-thread x 8-elem RMSNorm over collapsed[tok, :].
            warp_sq = T.alloc_shared((2,), T.float32)
            c_reg = T.alloc_local((norm_blocks, norm_vec), T.bfloat16)
            w_reg = T.alloc_local((norm_vec,), T.bfloat16)
            o_reg = T.alloc_local((norm_vec,), T.bfloat16)

            if tid < norm_threads:
                local_sq = T.alloc_var(T.float32, init=0.0)
                for b in T.unroll(norm_blocks):
                    h_base = b * (norm_threads * norm_vec) + tid * norm_vec
                    for v in T.vectorized(norm_vec):
                        c_reg[b, v] = collapsed[tok, h_base + v]
                    for v in T.unroll(norm_vec):
                        cv = T.float32(c_reg[b, v])
                        local_sq += cv * cv
                local_sq = T.warp_reduce_sum(local_sq)
                if lane == 0:
                    warp_sq[warp] = local_sq
            T.sync_threads()

            if tid < norm_threads:
                csq = warp_sq[0] + warp_sq[1]
                mean_sq = T.call_pure_extern(
                    "float32", "__fdiv_rn", csq, T.float32(hidden_size)
                )
                rsqrt_n = T.rsqrt(mean_sq + norm_eps)
                for b in T.unroll(norm_blocks):
                    h_base = b * (norm_threads * norm_vec) + tid * norm_vec
                    for v in T.vectorized(norm_vec):
                        w_reg[v] = norm_weight[h_base + v]
                    for v in T.unroll(norm_vec):
                        o_reg[v] = T.bfloat16(
                            T.float32(c_reg[b, v]) * rsqrt_n * T.float32(w_reg[v])
                        )
                    for v in T.vectorized(norm_vec):
                        layer_input[tok, h_base + v] = o_reg[v]


__all__ = [
    "mhc_pre_big_fuse_with_norm_tilelang",
    "mhc_pre_collapsed_norm_tilelang",
]
