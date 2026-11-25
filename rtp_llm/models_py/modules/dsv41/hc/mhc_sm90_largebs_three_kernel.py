# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM90 mid/large-batch delayed-mHC boundary in three kernels.

Phase-1 collapsed the seven-kernel SM90 ``64 < M <= 512`` chain into four
kernels (``post`` + split-K projection + fused pre epilogue + RMSNorm).  This
module removes the remaining standalone RMSNorm and the FFN ``add_cast``:

1. ``mhc_post`` (or ``mhc_post_combine`` when the deferred ``(routed, shared)``
   MoE pair is available) writes the new residual stream.
2. A split-K TF32 projection produces the ``[n_splits, M, 24]`` partials and
   the per-split square sums.  DeepGEMM's ``tf32_hc_prenorm_gemm`` is preferred;
   the TileLang ``_mhc_pre_norm_fn_fwd_mul`` kernel is the equivalent fallback.
3. ``mhc_pre_big_fuse_with_norm_tilelang`` reduces the partials, runs Sinkhorn,
   performs the delayed readout and applies the downstream layer RMSNorm in the
   same launch.

The combine variant rebuilds ``bf16(fp32(routed) + fp32(shared))`` inside the
post kernel -- exactly the value the retired standalone add+cast kernel stored
-- so the FFN seam pays three kernels instead of five and never round-trips the
combined ``[M, 5120]`` tensor through HBM.

DeepGEMM initializes persistent split barriers per CUDA stream.  As with the
low-token two-kernel backend, an unwarmed capture stream declines the fused
path instead of compiling or initializing barriers during capture.
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
_pre_big_fuse = importlib.import_module(
    "rtp_llm.models_py.3rdparty.tile_kernels.modeling.mhc.ops.pre_big_fuse"
)
_norm_fn = importlib.import_module(
    "rtp_llm.models_py.3rdparty.tile_kernels.modeling.mhc.ops.norm_fn"
)
_norm_fn_kernel = importlib.import_module(
    "rtp_llm.models_py.3rdparty.tile_kernels.mhc.norm_fn_kernel"
)
_mhc_pre_big_fuse_with_norm = importlib.import_module(
    "rtp_llm.models_py.modules.dsv41.hc.mhc_sm90_vllm_prenorm"
).mhc_pre_big_fuse_with_norm_tilelang
_functional = importlib.import_module(
    "rtp_llm.models_py.3rdparty.tile_kernels.modeling.mhc.functional"
)

HC = 4
DIM = 5120
MIX = 24
HC_HIDDEN = HC * DIM
BLOCK_K = 64
BLOCK_M = 64

# The two-kernel vLLM backend owns ``M <= 64``.  Phase-1 measured the fused
# pre chain as a win across the whole decode window; the lower bound is
# exclusive so the small-batch contract never changes.
MIN_TOKENS = 64
MAX_TOKENS = 512

_EXPLICIT_TILELANG_BACKENDS = frozenset(
    ("tilelang_single", "tilelang", "single", "tilelang_splitk", "splitk")
)

_WARMED_STREAMS: set[tuple[int, int]] = set()


def _stream_key(device: torch.device) -> tuple[int, int]:
    stream = torch.cuda.current_stream(device)
    return stream.device.index, stream.cuda_stream


def _has_deepgemm() -> bool:
    from rtp_llm.models_py.modules.dsv41.hc.v41_prenorm import _has_prenorm_gemm

    return _has_prenorm_gemm()


def _select_gemm_backend() -> str:
    """DeepGEMM when installed, unless TileLang is explicitly requested."""
    requested = os.environ.get("DSV4_MHC_PRE_GEMM_BACKEND", "").strip().lower()
    if requested in _EXPLICIT_TILELANG_BACKENDS:
        return "tilelang_splitk"
    return "deepgemm" if _has_deepgemm() else "tilelang_splitk"


def projection_splits(
    tokens: int, backend: str, device: torch.device | None = None
) -> int:
    """Runtime split count for one of the two projection backends."""
    if backend == "deepgemm":
        return _pre_big_fuse._compute_num_split(
            BLOCK_K, HC_HIDDEN, (tokens + BLOCK_M - 1) // BLOCK_M
        )
    index = (
        device.index
        if device is not None and device.index is not None
        else torch.cuda.current_device()
    )
    return _norm_fn._tilelang_split_plan(
        tokens, HC_HIDDEN, _norm_fn._tilelang_device_num_sms(index)
    )


_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,
    tilelang.PassConfigKey.TL_DISABLE_VECTORIZE_256: True,
}


@tilelang.jit(pass_configs=_PASS_CONFIGS)
def _mhc_post_combine_fwd(
    mhc: int, hidden: int, n_thr: int = 128, h_blk: int = 1024
) -> tilelang.JITKernel:
    """Post mixing fed by the deferred ``(routed, shared)`` BF16 pair.

    The combined sublayer output is rebuilt as
    ``bf16(fp32(routed) + fp32(shared))`` -- the exact value the standalone
    add+cast kernel stored -- before the post mapping consumes it.  The round
    trip through a BF16 shared tile is deliberate: it reproduces the retired
    kernel's store/load rounding, which a widened FP32 fold would not.
    """
    n = T.dynamic("num_tokens")
    h = hidden
    h_blk = math.gcd(hidden, h_blk)

    @T.prim_func
    def _mhc_post_combine_fwd_kernel(
        a: T.Tensor[(n, mhc, mhc), T.float32],
        b: T.Tensor[(n, mhc, h), T.bfloat16],
        c: T.Tensor[(n, mhc), T.float32],
        routed: T.Tensor[(n, h), T.bfloat16],
        shared: T.Tensor[(n, h), T.bfloat16],
        x: T.Tensor[(n, mhc, h), T.bfloat16],
    ) -> None:
        with T.Kernel(n, threads=n_thr) as pid_n:
            x_shared = T.alloc_shared((mhc, h_blk), T.bfloat16)
            b_shared = T.alloc_shared((mhc, h_blk), T.bfloat16)
            routed_shared = T.alloc_shared(h_blk, T.bfloat16)
            shared_shared = T.alloc_shared(h_blk, T.bfloat16)

            x_local = T.alloc_fragment((mhc, h_blk), T.float32)
            b_local = T.alloc_fragment((mhc, h_blk), T.float32)
            routed_local = T.alloc_fragment(h_blk, T.float32)
            shared_local = T.alloc_fragment(h_blk, T.float32)

            a_local = T.alloc_fragment((mhc, mhc), T.float32)
            c_local = T.alloc_fragment(mhc, T.float32)
            T.copy(a[pid_n, 0, 0], a_local)
            T.copy(c[pid_n, 0], c_local)
            T.pdl_sync()

            for i0_h in T.Pipelined(T.ceildiv(h, h_blk), num_stages=2):
                T.copy(b[pid_n, 0, i0_h * h_blk], b_shared, disable_tma=True)
                T.copy(routed[pid_n, i0_h * h_blk], routed_shared, disable_tma=True)
                T.copy(shared[pid_n, i0_h * h_blk], shared_shared, disable_tma=True)

                T.copy(b_shared, b_local)
                T.copy(routed_shared, routed_local)
                T.copy(shared_shared, shared_local)
                for i_mhco, i1_h in T.Parallel(mhc, h_blk):
                    # The retired add+cast kernel's exact BF16 value, widened
                    # back to FP32 by the multiplication.
                    x_local[i_mhco, i1_h] = c_local[i_mhco] * T.bfloat16(
                        routed_local[i1_h] + shared_local[i1_h]
                    )
                    for i_mhci in T.serial(mhc):
                        x_local[i_mhco, i1_h] += (
                            a_local[i_mhci, i_mhco] * b_local[i_mhci, i1_h]
                        )
                T.copy(x_local, x_shared)

                T.copy(x_shared, x[pid_n, 0, i0_h * h_blk], disable_tma=True)

    return _mhc_post_combine_fwd_kernel


def post_combine_fwd(
    routed: torch.Tensor,
    shared: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    *,
    out: torch.Tensor,
) -> torch.Tensor:
    """Run post mixing directly on the deferred ``(routed, shared)`` pair."""
    tokens, hc, hidden = residual.shape
    if hc != HC:
        raise ValueError(f"SM90 delayed-mHC post requires hc_mult={HC}")
    _mhc_post_combine_fwd(hc, hidden)(
        comb.view(tokens, hc, hc),
        residual.view(tokens, hc, hidden),
        post.view(tokens, hc),
        routed.view(tokens, hidden),
        shared.view(tokens, hidden),
        out.view(tokens, hc, hidden),
    )
    return out


def _run_projection(
    residual_out: torch.Tensor,
    fn: torch.Tensor,
    backend: str,
    tokens: int,
    partials: torch.Tensor,
    square_partials: torch.Tensor,
) -> int:
    """Write split-major projection partials; return the split count."""
    residual_flat = residual_out.view(tokens, HC_HIDDEN)
    if backend == "deepgemm":
        from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import tf32_hc_prenorm_gemm

        n_splits = projection_splits(tokens, backend, residual_out.device)
        tf32_hc_prenorm_gemm(residual_flat, fn, partials, square_partials, n_splits)
        return n_splits

    n_splits = projection_splits(tokens, backend, residual_out.device)
    kernel = _norm_fn_kernel._mhc_pre_norm_fn_fwd_mul(
        MIX, n_splits, HC_HIDDEN // n_splits
    )
    kernel(
        residual_flat,
        _norm_fn_kernel.prepare_tf32_rne_weight(fn),
        partials,
        square_partials,
    )
    return n_splits


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
    backend: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return residual, normalized input, next pre, post and combine mixes.

    ``combine`` carries the deferred ``(routed, shared)`` MoE pair.  When it is
    supplied the post kernel rebuilds ``bf16(routed + shared)`` inline and ``x``
    is used only for its token count and geometry.
    """
    tokens = x.shape[0]
    if not MIN_TOKENS < tokens <= MAX_TOKENS:
        raise ValueError(
            "SM90 large-batch three-kernel mHC requires "
            f"{MIN_TOKENS + 1}..{MAX_TOKENS} tokens, got {tokens}"
        )
    backend = backend or _select_gemm_backend()
    if backend not in ("deepgemm", "tilelang_splitk"):
        raise ValueError(f"Unsupported SM90 large-batch GEMM backend {backend!r}")

    residual_out = torch.empty_like(residual)
    if combine is None:
        # The shared TileLang post op keeps its established 4-D contract.
        _functional.mhc_post(
            x.view(1, tokens, DIM),
            residual.view(1, tokens, HC, DIM),
            post.view(1, tokens, HC, 1),
            comb.view(1, tokens, HC, HC),
            out=residual_out.view(1, tokens, HC, DIM),
        )
    else:
        routed, shared = combine
        post_combine_fwd(
            routed.view(tokens, DIM),
            shared.view(tokens, DIM),
            residual.view(tokens, HC, DIM),
            post.view(tokens, HC),
            comb.view(tokens, HC, HC),
            out=residual_out.view(tokens, HC, DIM),
        )

    # ``projection_splits`` is deterministic for a given token count, so the
    # scratch allocation and the reducer agree without a caller hint.
    n_splits = projection_splits(tokens, backend, x.device)
    partials = torch.empty(
        (n_splits, tokens, MIX), dtype=torch.float32, device=x.device
    )
    square_partials = torch.empty(
        (n_splits, tokens), dtype=torch.float32, device=x.device
    )
    next_pre = torch.empty((tokens, HC), dtype=torch.float32, device=x.device)
    next_post = torch.empty_like(next_pre)
    next_comb = torch.empty((tokens, HC * HC), dtype=torch.float32, device=x.device)
    normalized = torch.empty((tokens, DIM), dtype=torch.bfloat16, device=x.device)

    n_splits = _run_projection(
        residual_out, weight, backend, tokens, partials, square_partials
    )
    _mhc_pre_big_fuse_with_norm(
        partials,
        square_partials,
        scale,
        base,
        residual_out.view(tokens, HC, DIM),
        next_post,
        next_comb,
        normalized,
        norm_weight,
        previous_pre.view(tokens, HC),
        next_pre,
        normalized,
        hidden_size=DIM,
        rms_eps=hc_norm_eps,
        hc_pre_eps=pre_eps,
        hc_sinkhorn_eps=sinkhorn_eps,
        hc_post_mult_value=2.0,
        sinkhorn_repeat=sinkhorn_repeat,
        norm_eps=rmsnorm_eps,
        n_splits=n_splits,
        mhc_mult=HC,
        use_pre_mix_in=True,
        save_pre_mix=True,
        rms_numel=HC_HIDDEN,
        split_mode="fused",
        write_aux=False,
    )
    return residual_out, normalized, next_pre, next_post, next_comb


def try_sm90_largebs_post_pre(
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
    """Run the SM90 ``64 < M <= 512`` three-kernel path.

    Returns ``None`` outside the static contract.  ``combine`` is the deferred
    ``(routed, shared)`` MoE pair; ``attn_out`` is then only the geometry proxy
    for the combined sublayer output.
    """
    if (
        not isinstance(attn_out, torch.Tensor)
        or attn_out.device.type != "cuda"
        or attn_out.dtype != torch.bfloat16
        or attn_out.ndim != 3
        or attn_out.shape[-1] != DIM
        or not attn_out.is_contiguous()
        or not MIN_TOKENS < attn_out.numel() // DIM <= MAX_TOKENS
    ):
        return None
    device = attn_out.device
    tokens = attn_out.numel() // DIM
    if combine is not None:
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
    contracts = (
        (residual, leading + (HC, DIM), torch.bfloat16),
        (post, leading + (HC, 1), torch.float32),
        (comb, leading + (HC, HC), torch.float32),
        (getattr(previous, "pre_mix_out", None), leading + (HC,), torch.float32),
        (getattr(next_hc, "fn", None), (MIX, HC_HIDDEN), torch.float32),
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
        attn_out.requires_grad or any(value.requires_grad for value, _, _ in contracts)
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
            next_hc.fn,
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
            _WARMED_STREAMS.add(_stream_key(device))
    next_hc.pre_mix_out = new_pre.view(*leading, HC)
    return (
        new_residual.view(*leading, HC, DIM),
        normalized.view(*leading, DIM),
        new_post.view(*leading, HC, 1),
        new_comb.view(*leading, HC, HC),
    )


__all__ = [
    "MAX_TOKENS",
    "MIN_TOKENS",
    "post_combine_fwd",
    "projection_splits",
    "run",
    "try_sm90_largebs_post_pre",
]
