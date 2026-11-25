# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hopper small-M (``1 <= M <= 64``) delayed-mHC boundary.

In delayed mHC (``use_pre_mix_in=True``), ``previous_pre`` is already known
before the post-mapping kernel runs.  Fusing the delayed readout
``collapsed = bf16(sum_hc previous_pre[hc] * new_residual[hc])`` directly into
the post/projection pass removes the 4x5120 ``residual_out`` re-read from the
epilogue kernel.

Three SM90 regimes cover ``1 <= M <= 64``:

- ``1 <= M < 16``: vectorized split-10 two-kernel SIMT path (``tile_n=4``,
  128 threads, 4-element vector loads) + zero-barrier 4-thread fast-math
  Sinkhorn / FlashInfer-matching 64-thread RMSNorm epilogue.
- ``16 <= M <= 24``: zero-barrier single-N-tile (``tile_n=24``) split-40
  one-warp SIMT kernel eliminating redundant residual reads + cooperative
  epilogue.
- ``24 < M <= 64``: coalesced post+collapse kernel + DeepGEMM native Hopper
  TF32 split-80 projection (reading the 1.97 MB weight once via TMA/WGMMA) +
  cooperative 8-warp epilogue (falling back to the split-10 SIMT kernel when
  DeepGEMM is unavailable or explicitly disabled).

Originally adapted from vLLM at 0c81a0371f586cc50a0daad6d216fbac21cf1a2d,
``vllm/model_executor/kernels/mhc/tilelang_kernels.py`` (Apache-2.0).
"""

import importlib
import math
import os

import tilelang
import torch
from tilelang import language as T

from rtp_llm.models_py.modules.dsv4 import (  # noqa: F401
    tilelang_kernels as _tilelang_kernels,
)

# ``3rdparty`` is not importable through a normal statement (leading digit).
prepare_tf32_rne_weight = importlib.import_module(
    "rtp_llm.models_py.3rdparty.tile_kernels.mhc.norm_fn_kernel"
).prepare_tf32_rne_weight

_mhc_pre_collapsed_norm = importlib.import_module(
    "rtp_llm.models_py.modules.dsv41.hc.mhc_sm90_vllm_prenorm"
).mhc_pre_collapsed_norm_tilelang

HC = 4
DIM = 5120
MIX = 24
SPLITS = 10
THREADS = 128
VEC = 4

# Single-N-tile (zero redundant residual reads, zero block barriers) regime.
_MID_MIN_TOKENS = 16
_MID_MAX_TOKENS = 24
_MID_SPLITS = 40
_MID_THREADS = 32
_MID_VEC = 4

# DeepGEMM TF32 split-K projection regime for 24 < M <= 64.
_DG_SPLITS = 80
_POST_COLLAPSED_SPLITS = 10
_POST_COLLAPSED_THREADS = 128
_POST_COLLAPSED_VEC = 4

# Epilogue launches grid=(M, 2) with 256 threads per CTA (<=128 CTAs at M=64).
_EPILOGUE_THREADS = 256

# Largest token count owned by the small-M SM90 backend (64 is also the largest
# decode CUDA-graph bucket; 64 < M <= 512 is owned by largebs_three_kernel).
MAX_TOKENS = 64

_EXPLICIT_TILELANG_BACKENDS = frozenset(
    ("tilelang_single", "tilelang", "single", "tilelang_splitk", "splitk")
)

_WARMED_STREAMS: set[tuple] = set()


def prepare_sm90_tf32_rne_weight(fn: torch.Tensor) -> torch.Tensor:
    """Materialize the native TMA TF32 RN-even values for the scalar projection.

    Standalone prenorm passes raw weights to DeepGEMM's TMA conversion. The
    fused projection uses FP32 arithmetic, so it must perform that conversion
    explicitly, including exact halfway values. The split-K TileLang projection
    consumes the same cached values, so both SM90 backends see one TF32 weight.
    """
    return prepare_tf32_rne_weight(fn)


def _has_deepgemm() -> bool:
    from rtp_llm.models_py.modules.dsv41.hc.v41_prenorm import _has_prenorm_gemm

    return _has_prenorm_gemm()


def _use_deepgemm_med(tokens: int) -> bool:
    if tokens <= _MID_MAX_TOKENS:
        return False
    requested = os.environ.get("DSV4_MHC_PRE_GEMM_BACKEND", "").strip().lower()
    if requested in _EXPLICIT_TILELANG_BACKENDS:
        return False
    return _has_deepgemm()


_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,
    tilelang.PassConfigKey.TL_DISABLE_VECTORIZE_256: True,
}


def _build_post_projection(
    tile_n: int,
    combine: bool,
    *,
    splits: int = SPLITS,
    threads: int = THREADS,
    vec: int = VEC,
):
    """Build the fused post-mapping + delayed-collapse + split-projection kernel.

    ``combine`` selects how the sublayer output element is obtained.  The plain
    variant loads the materialized BF16 tensor.  The deferred-combine variant
    rebuilds ``bf16(routed + shared)`` from the two BF16 operands: the FP32 sum
    is rounded once to BF16 and widened back, which is bit-for-bit the value
    ``_add_cast_kernel`` had stored, so the standalone combine kernel and its
    round trip through memory disappear.

    ``projection_tile == 0`` also writes ``residual_out``, the rounded BF16
    ``collapsed_out``, and the per-split ``square_partials`` while
    ``new_residual`` is still in registers.
    """
    num_tokens = T.dynamic("num_tokens")
    assert DIM % splits == 0
    hidden_per_split = DIM // splits
    step_h = threads * vec
    assert hidden_per_split % step_h == 0
    hidden_iters = hidden_per_split // step_h
    num_warps = threads // 32
    assert MIX % tile_n == 0
    num_projection_tiles = MIX // tile_n

    @T.prim_func
    def mhc_fused_post_projection(
        comb_mix: T.Tensor[(num_tokens, HC, HC), T.float32],
        residual_in: T.Tensor[(num_tokens, HC, DIM), T.bfloat16],
        post_mix: T.Tensor[(num_tokens, HC), T.float32],
        previous_pre: T.Tensor[(num_tokens, HC), T.float32],
        routed_in: T.Tensor[(num_tokens, DIM), T.bfloat16],
        shared_in: T.Tensor[(num_tokens, DIM), T.bfloat16],
        weight: T.Tensor[(MIX, HC, DIM), T.float32],
        partials: T.Tensor[(splits, num_tokens, MIX), T.float32],
        square_partials: T.Tensor[(splits, num_tokens), T.float32],
        residual_out: T.Tensor[(num_tokens, HC, DIM), T.bfloat16],
        collapsed_out: T.Tensor[(num_tokens, DIM), T.bfloat16],
    ) -> None:
        with T.Kernel(
            num_tokens, num_projection_tiles, splits, threads=threads
        ) as (token, projection_tile, split):
            tid = T.get_thread_binding()
            warp = tid // 32
            lane = tid % 32

            post_local = T.alloc_local((HC,), T.float32)
            pre_local = T.alloc_local((HC,), T.float32)
            comb_local = T.alloc_local((HC, HC), T.float32)
            for dst in T.unroll(HC):
                post_local[dst] = post_mix[token, dst]
                pre_local[dst] = previous_pre[token, dst]
            for src in T.unroll(HC):
                for dst in T.unroll(HC):
                    comb_local[src, dst] = comb_mix[token, src, dst]

            accum = T.alloc_local((tile_n,), T.float32)
            square = T.alloc_local((1,), T.float32)
            T.clear(accum)
            T.clear(square)

            hidden_start = split * hidden_per_split
            x_vec = T.alloc_local((vec,), T.bfloat16)
            s_vec = T.alloc_local((vec,), T.bfloat16)
            r_vec = T.alloc_local((HC, vec), T.bfloat16)
            nr_vec = T.alloc_local((HC, vec), T.bfloat16)
            c_vec = T.alloc_local((vec,), T.bfloat16)
            w_vec = T.alloc_local((vec,), T.float32)

            for iteration in T.serial(hidden_iters):
                h_base = hidden_start + iteration * step_h + tid * vec
                for v in T.vectorized(vec):
                    x_vec[v] = routed_in[token, h_base + v]
                if combine:
                    for v in T.vectorized(vec):
                        s_vec[v] = shared_in[token, h_base + v]
                for src in T.unroll(HC):
                    for v in T.vectorized(vec):
                        r_vec[src, v] = residual_in[token, src, h_base + v]

                for v in T.unroll(vec):
                    if combine:
                        # The retired add+cast kernel's exact BF16 value.
                        xv = T.float32(
                            T.bfloat16(T.float32(x_vec[v]) + T.float32(s_vec[v]))
                        )
                    else:
                        xv = T.float32(x_vec[v])
                    c_acc = T.alloc_var(T.float32, init=0.0)
                    for dst in T.unroll(HC):
                        nr = T.alloc_var(T.float32, init=post_local[dst] * xv)
                        for src in T.unroll(HC):
                            nr += comb_local[src, dst] * T.float32(r_vec[src, v])
                        # RTP's established boundary projects the materialized
                        # BF16 residual. Preserve that contract inline.
                        nr_bf16 = T.bfloat16(nr)
                        nr_vec[dst, v] = nr_bf16
                        if projection_tile == 0:
                            nr_f = T.float32(nr_bf16)
                            square[0] += nr_f * nr_f
                            c_acc += pre_local[dst] * nr_f
                    if projection_tile == 0:
                        # Keep the BF16 boundary before delayed input RMSNorm.
                        c_vec[v] = T.bfloat16(c_acc)

                if projection_tile == 0:
                    for dst in T.unroll(HC):
                        for v in T.vectorized(vec):
                            residual_out[token, dst, h_base + v] = nr_vec[dst, v]
                    for v in T.vectorized(vec):
                        collapsed_out[token, h_base + v] = c_vec[v]

                for lane_hc in T.unroll(HC):
                    for output in T.unroll(tile_n):
                        n_idx = projection_tile * tile_n + output
                        for v in T.vectorized(vec):
                            w_vec[v] = weight[n_idx, lane_hc, h_base + v]
                        for v in T.unroll(vec):
                            accum[output] += w_vec[v] * T.float32(nr_vec[lane_hc, v])

            if num_warps == 1:
                for output in T.unroll(tile_n):
                    val = T.warp_reduce_sum(accum[output])
                    if lane == 0:
                        partials[
                            split, token, projection_tile * tile_n + output
                        ] = val
                if projection_tile == 0:
                    sq_val = T.warp_reduce_sum(square[0])
                    if lane == 0:
                        square_partials[split, token] = sq_val
            else:
                warp_sums = T.alloc_shared((num_warps, tile_n + 1), T.float32)
                for output in T.unroll(tile_n):
                    accum[output] = T.warp_reduce_sum(accum[output])
                if projection_tile == 0:
                    square[0] = T.warp_reduce_sum(square[0])
                if lane == 0:
                    for output in T.unroll(tile_n):
                        warp_sums[warp, output] = accum[output]
                    if projection_tile == 0:
                        warp_sums[warp, tile_n] = square[0]
                T.sync_threads()

                if warp == 0:
                    if lane < tile_n:
                        value = T.alloc_var(T.float32, init=0.0)
                        for source_warp in T.unroll(num_warps):
                            value += warp_sums[source_warp, lane]
                        partials[
                            split, token, projection_tile * tile_n + lane
                        ] = value
                    if projection_tile == 0 and lane == 0:
                        sq_val = T.alloc_var(T.float32, init=0.0)
                        for source_warp in T.unroll(num_warps):
                            sq_val += warp_sums[source_warp, tile_n]
                        square_partials[split, token] = sq_val

    return mhc_fused_post_projection


def _build_post_collapsed(
    combine: bool,
    *,
    splits: int = _POST_COLLAPSED_SPLITS,
    threads: int = _POST_COLLAPSED_THREADS,
    vec: int = _POST_COLLAPSED_VEC,
):
    """Build the coalesced post-mapping + delayed-collapse kernel for DeepGEMM."""
    num_tokens = T.dynamic("num_tokens")
    assert DIM % splits == 0
    hidden_per_split = DIM // splits
    step_h = threads * vec
    assert hidden_per_split % step_h == 0
    hidden_iters = hidden_per_split // step_h

    @T.prim_func
    def mhc_post_collapsed(
        comb_mix: T.Tensor[(num_tokens, HC, HC), T.float32],
        residual_in: T.Tensor[(num_tokens, HC, DIM), T.bfloat16],
        post_mix: T.Tensor[(num_tokens, HC), T.float32],
        previous_pre: T.Tensor[(num_tokens, HC), T.float32],
        routed_in: T.Tensor[(num_tokens, DIM), T.bfloat16],
        shared_in: T.Tensor[(num_tokens, DIM), T.bfloat16],
        residual_out: T.Tensor[(num_tokens, HC, DIM), T.bfloat16],
        collapsed_out: T.Tensor[(num_tokens, DIM), T.bfloat16],
    ) -> None:
        with T.Kernel(num_tokens, splits, threads=threads) as (token, split):
            tid = T.get_thread_binding()

            post_local = T.alloc_local((HC,), T.float32)
            pre_local = T.alloc_local((HC,), T.float32)
            comb_local = T.alloc_local((HC, HC), T.float32)
            for dst in T.unroll(HC):
                post_local[dst] = post_mix[token, dst]
                pre_local[dst] = previous_pre[token, dst]
            for src in T.unroll(HC):
                for dst in T.unroll(HC):
                    comb_local[src, dst] = comb_mix[token, src, dst]

            x_vec = T.alloc_local((vec,), T.bfloat16)
            s_vec = T.alloc_local((vec,), T.bfloat16)
            r_vec = T.alloc_local((HC, vec), T.bfloat16)
            nr_vec = T.alloc_local((HC, vec), T.bfloat16)
            c_vec = T.alloc_local((vec,), T.bfloat16)

            hidden_start = split * hidden_per_split
            for iteration in T.serial(hidden_iters):
                h_base = hidden_start + iteration * step_h + tid * vec
                for v in T.vectorized(vec):
                    x_vec[v] = routed_in[token, h_base + v]
                if combine:
                    for v in T.vectorized(vec):
                        s_vec[v] = shared_in[token, h_base + v]
                for src in T.unroll(HC):
                    for v in T.vectorized(vec):
                        r_vec[src, v] = residual_in[token, src, h_base + v]

                for v in T.unroll(vec):
                    if combine:
                        xv = T.float32(
                            T.bfloat16(T.float32(x_vec[v]) + T.float32(s_vec[v]))
                        )
                    else:
                        xv = T.float32(x_vec[v])
                    c_acc = T.alloc_var(T.float32, init=0.0)
                    for dst in T.unroll(HC):
                        nr = T.alloc_var(T.float32, init=post_local[dst] * xv)
                        for src in T.unroll(HC):
                            nr += comb_local[src, dst] * T.float32(r_vec[src, v])
                        nr_bf16 = T.bfloat16(nr)
                        nr_vec[dst, v] = nr_bf16
                        c_acc += pre_local[dst] * T.float32(nr_bf16)
                    c_vec[v] = T.bfloat16(c_acc)

                for dst in T.unroll(HC):
                    for v in T.vectorized(vec):
                        residual_out[token, dst, h_base + v] = nr_vec[dst, v]
                for v in T.vectorized(vec):
                    collapsed_out[token, h_base + v] = c_vec[v]

    return mhc_post_collapsed


@tilelang.jit(pass_configs=_PASS_CONFIGS)
def _mhc_fused_post_projection(tile_n: int):
    """Fuse post mapping, delayed collapse, and split projection."""
    return _build_post_projection(tile_n, combine=False)


@tilelang.jit(pass_configs=_PASS_CONFIGS)
def _mhc_fused_post_projection_combine(tile_n: int):
    """Same kernel, consuming the deferred ``bf16(routed + shared)`` operands."""
    return _build_post_projection(tile_n, combine=True)


@tilelang.jit(pass_configs=_PASS_CONFIGS)
def _mhc_fused_post_projection_mid(combine: bool):
    """Single-N-tile (N=24) one-warp zero-barrier kernel for ``16 <= M <= 24``."""
    return _build_post_projection(
        MIX,
        combine=combine,
        splits=_MID_SPLITS,
        threads=_MID_THREADS,
        vec=_MID_VEC,
    )


@tilelang.jit(pass_configs=_PASS_CONFIGS)
def _mhc_post_collapsed(combine: bool):
    """Coalesced post + collapse kernel feeding DeepGEMM for ``24 < M <= 64``."""
    return _build_post_collapsed(combine=combine)


def _regime_key(tokens: int) -> int:
    if tokens < _MID_MIN_TOKENS:
        return 0
    if tokens <= _MID_MAX_TOKENS:
        return 1
    return 2 if _use_deepgemm_med(tokens) else 3


def run(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    weight: torch.Tensor,
    previous_pre: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    norm_weight: torch.Tensor,
    *,
    hc_norm_eps: float,
    pre_eps: float,
    sinkhorn_eps: float,
    sinkhorn_repeat: int,
    rmsnorm_eps: float,
    combine: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return residual, normalized input, next pre, post and combine mixes.

    ``combine`` carries the deferred ``(routed, shared)`` MoE pair.  When it is
    supplied the fused kernel rebuilds ``bf16(routed + shared)`` inline and
    ``x`` is used only for its token count and geometry.
    """
    tokens = x.shape[0]
    if not 0 < tokens <= MAX_TOKENS:
        raise ValueError(
            f"vLLM SM90 two-kernel mHC requires 1..{MAX_TOKENS} tokens, got {tokens}"
        )

    residual_out = torch.empty_like(residual)
    collapsed = torch.empty((tokens, DIM), dtype=torch.bfloat16, device=x.device)
    next_pre = torch.empty((tokens, HC), dtype=torch.float32, device=x.device)
    next_post = torch.empty_like(next_pre)
    next_comb = torch.empty((tokens, HC * HC), dtype=torch.float32, device=x.device)
    normalized = torch.empty((tokens, DIM), dtype=torch.bfloat16, device=x.device)

    routed_arg = x if combine is None else combine[0]
    shared_arg = x if combine is None else combine[1]
    is_combine = combine is not None

    if _MID_MIN_TOKENS <= tokens <= _MID_MAX_TOKENS:
        p_splits = _MID_SPLITS
        partials = torch.empty(
            (p_splits, tokens, MIX), dtype=torch.float32, device=x.device
        )
        square_partials = torch.empty(
            (p_splits, tokens), dtype=torch.float32, device=x.device
        )
        _mhc_fused_post_projection_mid(is_combine)(
            comb,
            residual,
            post,
            previous_pre,
            routed_arg,
            shared_arg,
            weight.view(MIX, HC, DIM),
            partials,
            square_partials,
            residual_out,
            collapsed,
        )
    elif _use_deepgemm_med(tokens):
        from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import tf32_hc_prenorm_gemm

        p_splits = _DG_SPLITS
        partials = torch.empty(
            (p_splits, tokens, MIX), dtype=torch.float32, device=x.device
        )
        square_partials = torch.empty(
            (p_splits, tokens), dtype=torch.float32, device=x.device
        )
        _mhc_post_collapsed(is_combine)(
            comb,
            residual,
            post,
            previous_pre,
            routed_arg,
            shared_arg,
            residual_out,
            collapsed,
        )
        tf32_hc_prenorm_gemm(
            residual_out.view(tokens, HC * DIM),
            weight.view(MIX, HC * DIM),
            partials,
            square_partials,
            p_splits,
        )
    else:
        p_splits = SPLITS
        tile_n = 4 if tokens < _MID_MIN_TOKENS else 12
        partials = torch.empty(
            (p_splits, tokens, MIX), dtype=torch.float32, device=x.device
        )
        square_partials = torch.empty(
            (p_splits, tokens), dtype=torch.float32, device=x.device
        )
        kernel = (
            _mhc_fused_post_projection_combine(tile_n)
            if is_combine
            else _mhc_fused_post_projection(tile_n)
        )
        kernel(
            comb,
            residual,
            post,
            previous_pre,
            routed_arg,
            shared_arg,
            weight.view(MIX, HC, DIM),
            partials,
            square_partials,
            residual_out,
            collapsed,
        )

    _mhc_pre_collapsed_norm(
        partials,
        square_partials,
        scale,
        base,
        collapsed,
        norm_weight,
        next_pre,
        next_post,
        next_comb,
        normalized,
        hidden_size=DIM,
        rms_eps=hc_norm_eps,
        hc_pre_eps=pre_eps,
        hc_sinkhorn_eps=sinkhorn_eps,
        hc_post_mult_value=2.0,
        sinkhorn_repeat=sinkhorn_repeat,
        norm_eps=rmsnorm_eps,
        n_splits=p_splits,
        hc_mult=HC,
        threads=_EPILOGUE_THREADS,
        rms_numel=HC * DIM,
    )
    return residual_out, normalized, next_pre, next_post, next_comb


def _stream_key(device: torch.device, tokens: int, combine: bool) -> tuple:
    stream = torch.cuda.current_stream(device)
    return stream.device.index, stream.cuda_stream, _regime_key(tokens), combine


def try_vllm_two_kernel_post_pre(
    attn_out: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    previous,
    next_hc,
    norm: torch.nn.Module,
    *,
    combine: tuple[torch.Tensor, torch.Tensor] | None = None,
):
    """Run the low-token SM90 path, returning ``None`` outside its contract.

    ``combine`` is the deferred ``(routed, shared)`` MoE pair.  ``attn_out`` is
    then only the geometry proxy for the combined sublayer output; the fused
    kernel rebuilds ``bf16(routed + shared)`` itself.
    """
    if (
        not isinstance(attn_out, torch.Tensor)
        or attn_out.device.type != "cuda"
        or attn_out.dtype != torch.bfloat16
        or attn_out.ndim != 3
        or attn_out.shape[-1] != DIM
        or not attn_out.is_contiguous()
        or not 0 < attn_out.numel() // DIM <= MAX_TOKENS
    ):
        return None
    device = attn_out.device
    if combine is not None:
        tokens = attn_out.numel() // DIM
        for operand in combine:
            if (
                not isinstance(operand, torch.Tensor)
                or operand.dtype != torch.bfloat16
                or operand.device != device
                or not operand.is_contiguous()
                or operand.numel() != tokens * DIM
            ):
                return None
    try:
        if tuple(torch.cuda.get_device_capability(device)) != (9, 0):
            return None
    except Exception:
        return None
    from rtp_llm.models_py.modules.base.cuda.norm import RMSNorm
    from rtp_llm.models_py.modules.dsv41.hc.delayed import DelayedHCUnit

    if (
        type(previous) is not DelayedHCUnit
        or type(next_hc) is not DelayedHCUnit
        or type(norm) is not RMSNorm
        or previous.dim != DIM
        or previous.hc_mult != HC
        or next_hc._previous_ref is None
        or next_hc._previous_ref() is not previous
        or next_hc.dim != DIM
        or next_hc.hc_mult != HC
        or next_hc.hc_sinkhorn_iters < 1
        or not all(
            isinstance(eps, (int, float)) and math.isfinite(eps) and eps >= 0
            for eps in (next_hc.norm_eps, next_hc.hc_eps, norm.variance_epsilon)
        )
    ):
        return None
    leading = tuple(attn_out.shape[:-1])
    tokens = attn_out.numel() // DIM
    contracts = (
        (residual, leading + (HC, DIM), torch.bfloat16),
        (post, leading + (HC, 1), torch.float32),
        (comb, leading + (HC, HC), torch.float32),
        (getattr(previous, "pre_mix_out", None), leading + (HC,), torch.float32),
        (getattr(next_hc, "fn", None), (MIX, HC * DIM), torch.float32),
        (getattr(next_hc, "scale", None), (3,), torch.float32),
        (getattr(next_hc, "base", None), (MIX,), torch.float32),
        (getattr(norm, "weight", None), (DIM,), torch.bfloat16),
    )
    if any(
        not isinstance(value, torch.Tensor)
        or tuple(value.shape) != shape
        or value.dtype != dtype
        or value.device != device
        or not value.is_contiguous()
        for value, shape, dtype in contracts
    ):
        return None
    if torch.is_grad_enabled() and (
        attn_out.requires_grad
        or any(value.requires_grad for value, _, _ in contracts)
    ):
        return None
    with torch.cuda.device(device):
        capturing = torch.cuda.is_current_stream_capturing()
        skey = _stream_key(device, tokens, combine is not None)
        if capturing and skey not in _WARMED_STREAMS:
            return None
        new_residual, normalized, new_pre, new_post, new_comb = run(
            attn_out.view(tokens, DIM),
            residual.view(tokens, HC, DIM),
            post.view(tokens, HC),
            comb.view(tokens, HC, HC),
            prepare_sm90_tf32_rne_weight(next_hc.fn),
            previous.pre_mix_out.view(tokens, HC),
            next_hc.scale,
            next_hc.base,
            norm.weight,
            hc_norm_eps=next_hc.norm_eps,
            pre_eps=next_hc.hc_eps,
            sinkhorn_eps=next_hc.hc_eps,
            sinkhorn_repeat=next_hc.hc_sinkhorn_iters,
            rmsnorm_eps=norm.variance_epsilon,
            combine=(
                None
                if combine is None
                else (
                    combine[0].view(tokens, DIM),
                    combine[1].view(tokens, DIM),
                )
            ),
        )
        if not capturing:
            _WARMED_STREAMS.add(skey)
    next_hc.pre_mix_out = new_pre.view(*leading, HC)
    return (
        new_residual.view(*leading, HC, DIM),
        normalized.view(*leading, DIM),
        new_post.view(*leading, HC, 1),
        new_comb.view(*leading, HC, HC),
    )


__all__ = ["run", "try_vllm_two_kernel_post_pre"]
