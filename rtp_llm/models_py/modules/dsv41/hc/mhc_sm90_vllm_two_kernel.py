# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hopper two-kernel delayed-mHC boundary adapted from vLLM.

The first TileLang kernel folds the delayed post mapping into the next
coefficient projection.  The existing RTP TileLang epilogue then reduces the
split partials, runs Sinkhorn, performs the delayed readout and writes the
BF16 layer input.  The caller applies the established RMSNorm implementation.

Adapted from vLLM at 0c81a0371f586cc50a0daad6d216fbac21cf1a2d,
``vllm/model_executor/kernels/mhc/tilelang_kernels.py`` (Apache-2.0).
"""

import importlib
import math

import tilelang
import torch
from tilelang import language as T

from rtp_llm.models_py.modules.dsv4 import (  # noqa: F401
    tilelang_kernels as _tilelang_kernels,
)
from rtp_llm.models_py.modules.dsv41.hc.v41_prenorm import prepare_tf32_weight
_mhc_pre_big_fuse_with_norm = importlib.import_module(
    "rtp_llm.models_py.modules.dsv41.hc.mhc_sm90_vllm_prenorm"
).mhc_pre_big_fuse_with_norm_tilelang

HC = 4
DIM = 5120
MIX = 24
SPLITS = 8
THREADS = 128
_WARMED_STREAMS: set[tuple[int, int]] = set()


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,
        tilelang.PassConfigKey.TL_DISABLE_VECTORIZE_256: True,
    },
)
def _mhc_fused_post_projection(tile_n: int):
    """Fuse post mapping and split projection for low-token decode."""
    num_tokens = T.dynamic("num_tokens")
    hidden_per_split = DIM // SPLITS
    hidden_iters = hidden_per_split // THREADS
    num_warps = THREADS // 32
    num_projection_tiles = MIX // tile_n

    @T.prim_func
    def kernel(
        comb_mix: T.Tensor[(num_tokens, HC, HC), T.float32],
        residual_in: T.Tensor[(num_tokens, HC, DIM), T.bfloat16],
        post_mix: T.Tensor[(num_tokens, HC), T.float32],
        x_in: T.Tensor[(num_tokens, DIM), T.bfloat16],
        weight: T.Tensor[(MIX, HC, DIM), T.float32],
        partials: T.Tensor[(SPLITS, num_tokens, MIX), T.float32],
        square_partials: T.Tensor[(SPLITS, num_tokens), T.float32],
        residual_out: T.Tensor[(num_tokens, HC, DIM), T.bfloat16],
    ) -> None:
        with T.Kernel(num_tokens, num_projection_tiles, SPLITS, threads=THREADS) as (
            token,
            projection_tile,
            split,
        ):
            tid = T.get_thread_binding()
            warp = tid // 32
            lane = tid % 32
            warp_sums = T.alloc_shared((num_warps, tile_n + 1), T.float32)
            post_shared = T.alloc_shared((HC,), T.float32)
            comb_shared = T.alloc_shared((HC, HC), T.float32)
            post_local = T.alloc_local((HC,), T.float32)
            comb_local = T.alloc_local((HC, HC), T.float32)
            accum = T.alloc_local((tile_n,), T.float32)
            square = T.alloc_local((1,), T.float32)
            new_residual = T.alloc_local((HC,), T.float32)

            T.clear(accum)
            T.clear(square)
            hidden_start = split * hidden_per_split
            T.copy(post_mix[token, 0], post_shared)
            T.copy(comb_mix[token, 0, 0], comb_shared)
            for dst in T.unroll(HC):
                post_local[dst] = post_shared[dst]
            for src in T.unroll(HC):
                for dst in T.unroll(HC):
                    comb_local[src, dst] = comb_shared[src, dst]

            for iteration in T.serial(hidden_iters):
                hidden = hidden_start + iteration * THREADS + tid
                for dst in T.unroll(HC):
                    new_residual[dst] = post_local[dst] * x_in[token, hidden]
                    for src in T.unroll(HC):
                        new_residual[dst] += (
                            comb_local[src, dst] * residual_in[token, src, hidden]
                        )
                    # RTP's established boundary projects the materialized BF16
                    # residual.  Preserve that contract inside the fused kernel.
                    new_residual[dst] = T.bfloat16(new_residual[dst])

                if projection_tile == 0:
                    for dst in T.unroll(HC):
                        residual_out[token, dst, hidden] = new_residual[dst]
                        square[0] += new_residual[dst] * new_residual[dst]

                for output in T.unroll(tile_n):
                    for lane_hc in T.unroll(HC):
                        accum[output] += (
                            weight[projection_tile * tile_n + output, lane_hc, hidden]
                            * new_residual[lane_hc]
                        )

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
                    value = T.alloc_var(T.float32, init=0.0)
                    for source_warp in T.unroll(num_warps):
                        value += warp_sums[source_warp, tile_n]
                    square_partials[split, token] = value

    return kernel


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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return residual, normalized input, next pre, post and combine mixes."""
    tokens = x.shape[0]
    if not 0 < tokens <= 32:
        raise ValueError(f"vLLM SM90 two-kernel mHC requires 1..32 tokens, got {tokens}")
    tile_n = 2 if tokens < 16 else 6
    residual_out = torch.empty_like(residual)
    partials = torch.empty(
        (SPLITS, tokens, MIX), dtype=torch.float32, device=x.device
    )
    square_partials = torch.empty(
        (SPLITS, tokens), dtype=torch.float32, device=x.device
    )
    next_pre = torch.empty((tokens, HC), dtype=torch.float32, device=x.device)
    next_post = torch.empty_like(next_pre)
    next_comb = torch.empty((tokens, HC * HC), dtype=torch.float32, device=x.device)
    normalized = torch.empty((tokens, DIM), dtype=torch.bfloat16, device=x.device)

    _mhc_fused_post_projection(tile_n)(
        comb,
        residual,
        post,
        x,
        weight.view(MIX, HC, DIM),
        partials,
        square_partials,
        residual_out,
    )
    _mhc_pre_big_fuse_with_norm(
        partials,
        square_partials,
        scale,
        base,
        residual_out,
        next_post,
        next_comb,
        normalized,
        norm_weight,
        previous_pre,
        next_pre,
        normalized,
        hidden_size=DIM,
        rms_eps=hc_norm_eps,
        hc_pre_eps=pre_eps,
        hc_sinkhorn_eps=sinkhorn_eps,
        hc_post_mult_value=2.0,
        sinkhorn_repeat=sinkhorn_repeat,
        norm_eps=rmsnorm_eps,
        n_splits=SPLITS,
        mhc_mult=HC,
        use_pre_mix_in=True,
        save_pre_mix=True,
        rms_numel=HC * DIM,
        split_mode="fused",
        write_aux=False,
    )
    return residual_out, normalized, next_pre, next_post, next_comb


def _stream_key(device: torch.device) -> tuple[int, int]:
    stream = torch.cuda.current_stream(device)
    return stream.device.index, stream.cuda_stream


def try_vllm_two_kernel_post_pre(
    attn_out: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    previous,
    next_hc,
    norm: torch.nn.Module,
):
    """Run the low-token SM90 path, returning ``None`` outside its contract."""
    if (
        not isinstance(attn_out, torch.Tensor)
        or attn_out.device.type != "cuda"
        or attn_out.dtype != torch.bfloat16
        or attn_out.ndim != 3
        or attn_out.shape[-1] != DIM
        or not attn_out.is_contiguous()
        or not 0 < attn_out.numel() // DIM <= 32
    ):
        return None
    device = attn_out.device
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
        if capturing and _stream_key(device) not in _WARMED_STREAMS:
            return None
        new_residual, normalized, new_pre, new_post, new_comb = run(
            attn_out.view(tokens, DIM),
            residual.view(tokens, HC, DIM),
            post.view(tokens, HC),
            comb.view(tokens, HC, HC),
            prepare_tf32_weight(next_hc.fn),
            previous.pre_mix_out.view(tokens, HC),
            next_hc.scale,
            next_hc.base,
            norm.weight,
            hc_norm_eps=next_hc.norm_eps,
            pre_eps=next_hc.hc_eps,
            sinkhorn_eps=next_hc.hc_eps,
            sinkhorn_repeat=next_hc.hc_sinkhorn_iters,
            rmsnorm_eps=norm.variance_epsilon,
        )
        if not capturing:
            _WARMED_STREAMS.add(_stream_key(device))
    next_hc.pre_mix_out = new_pre.view(*leading, HC)
    return (
        new_residual.view(*leading, HC, DIM),
        normalized.view(*leading, DIM),
        new_post.view(*leading, HC, 1),
        new_comb.view(*leading, HC, HC),
    )


__all__ = ["run", "try_vllm_two_kernel_post_pre"]
