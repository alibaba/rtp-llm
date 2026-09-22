"""Bounded startup compilation of the delayed V4.1 HC inference paths.

The pinned DeepGEMM 2.8.0 mega_mhc implementation specializes on hidden,
split count, SM count and output modes; M is a runtime argument. The split
planner below mirrors csrc/apis/mega_mhc.hpp and layout/mega_mhc.cuh.
TileKernels norm_fn, split_mixes, sinkhorn, apply_mix and post likewise use
T.dynamic for M, so one launch covers all prefill lengths for a configuration.
The small-prenorm DG GEMM uses fixed split16, but RTP's Triton reduction has
TOKENS constexpr: every reachable M <= 64 needs its own reduction variant.

Only disposable copies of HC units/weights are executed. No live pre_mix_out,
predecessor reference, weight tensor cache attribute or KV cache is modified.
This is prefill eager warmup; CUDA graph capture streams still need their own
existing eager preparation for DeepGEMM's per-stream split barriers.
"""

from __future__ import annotations

import logging
import math
from functools import partial

import torch

from rtp_llm.models_py.modules.dsv4 import dsv4_kernel_jit_warmup as common
from rtp_llm.models_py.modules.dsv4.hc import v41_mega_mhc, v41_prenorm
from rtp_llm.models_py.modules.dsv4.hc.delayed import DelayedHCUnit, _tile_ops

_WARMED_KEYS: set[tuple] = set()


def _unit_key(unit):
    if type(unit) is not DelayedHCUnit:
        return None
    if (
        unit.hc_mult != 4
        or unit.dim <= 0
        or (unit.dim * unit.hc_mult) % 256
        or unit.hc_sinkhorn_iters < 1
        or any(
            not math.isfinite(eps) or eps < 0 for eps in (unit.norm_eps, unit.hc_eps)
        )
        or tuple(unit.fn.shape) != (24, unit.dim * 4)
        or tuple(unit.base.shape) != (24,)
        or tuple(unit.scale.shape) != (3,)
        or any(
            t.dtype != torch.float32 or not t.is_contiguous()
            for t in (unit.fn, unit.base, unit.scale)
        )
    ):
        return None
    return (unit.dim, unit.hc_mult, unit.hc_sinkhorn_iters, unit.norm_eps, unit.hc_eps)


def _collect_v41_hc_configs(v4):
    """Collect actual delayed units and in-block seams; never guess by class name."""
    units, pairs = {}, {}
    for module in v4.modules():
        key = _unit_key(module)
        if key is not None:
            units.setdefault(key, module)
        previous = getattr(module, "attn_hc", None)
        next_hc = getattr(module, "ffn_hc", None)
        norm = getattr(module, "ffn_norm", None)
        next_key = _unit_key(next_hc)
        if (
            _unit_key(previous) is None
            or next_key is None
            or previous.dim != 5120
            or next_hc.dim != 5120
            or next_hc._previous_ref is None
            or next_hc._previous_ref() is not previous
            or type(norm).__module__ != "rtp_llm.models_py.modules.base.cuda.norm"
            or type(norm).__name__ != "RMSNorm"
        ):
            continue
        from rtp_llm.models_py.modules.base.cuda.norm import RMSNorm

        if (
            type(norm) is not RMSNorm
            or tuple(norm.weight.shape) != (5120,)
            or norm.weight.dtype != torch.bfloat16
            or not norm.weight.is_contiguous()
            or not math.isfinite(norm.variance_epsilon)
            or norm.variance_epsilon < 0
        ):
            continue
        pairs.setdefault(next_key + (norm.variance_epsilon,), (previous, next_hc, norm))
    return units, pairs


def _mega_num_splits(tokens, hidden, num_sms):
    """Pinned DG nondeterministic heuristic (deterministic mode uses split16)."""
    m_blocks = (tokens + 63) // 64
    k_blocks = hidden // 64
    max_splits = min(64, max(16, num_sms // m_blocks))
    per_split = (k_blocks + max_splits - 1) // max_splits
    return (k_blocks + per_split - 1) // per_split


def _mega_representative_ms(max_m, num_sms, hidden=5120):
    """First M for every reachable split key, bounded without large allocations.

    After ceil(M/64) > num_sms/16 the default split16 branch is stable.
    If DG deterministic mode is already enabled all these launches hit its
    one split16 key. We do not change DG's process-wide runtime policy.
    """
    if max_m <= 0:
        return ()
    if num_sms <= 0 or hidden <= 0 or hidden % 64:
        raise ValueError("Invalid mega_mhc device/hidden geometry")
    max_m = min(int(max_m), 1 << 20)
    max_blocks = min((max_m + 63) // 64, num_sms // 16 + 1)
    seen, representatives = set(), []
    for block in range(1, max_blocks + 1):
        tokens = (block - 1) * 64 + 1
        splits = _mega_num_splits(tokens, hidden, num_sms)
        if splits not in seen:
            seen.add(splits)
            representatives.append(tokens)
    return tuple(representatives)


def _small_prenorm_ms(max_m):
    # DG's M is runtime, but _reduce_prenorm_kernel.TOKENS is constexpr.
    return tuple(range(1, min(max(int(max_m), 0), 64) + 1))


def _clone_unit(source, device):
    return DelayedHCUnit(
        source.fn.detach().to(device=device).clone(),
        source.base.detach().to(device=device).clone(),
        source.scale.detach().to(device=device).clone(),
        dim=source.dim,
        hc_mult=source.hc_mult,
        hc_sinkhorn_iters=source.hc_sinkhorn_iters,
        norm_eps=source.norm_eps,
        hc_eps=source.hc_eps,
    )


def _launch_small_prenorm(unit, tokens, device):
    residual = torch.ones((1, tokens, 4, unit.dim), dtype=torch.bfloat16, device=device)
    mixes = v41_prenorm.prenorm(residual, unit.fn, unit.norm_eps)
    if mixes is None:
        raise RuntimeError("V4.1 HC warmup small prenorm rejected a reachable shape")


def _launch_tilelang_chain(unit, tokens, device, *, include_norm):
    ops = _tile_ops()  # Initializes TileLang/libz3/TVM environment before imports.
    residual = torch.ones((1, tokens, 4, unit.dim), dtype=torch.bfloat16, device=device)
    mixes = (
        ops.mhc_pre_norm_fn(residual, unit.fn, None, unit.norm_eps, n_splits=1)
        if include_norm
        else torch.zeros((1, tokens, 24), dtype=torch.float32, device=device)
    )
    pre, post, comb = ops.mhc_pre_split_mixes(
        mixes, unit.scale, unit.base, 4, 2.0, unit.hc_eps
    )
    comb = ops.sinkhorn_normalize(comb, repeat=unit.hc_sinkhorn_iters, eps=unit.hc_eps)
    # The delayed collapse/final head uses this same apply_mix specialization.
    hidden = ops.mhc_pre_apply_mix(residual, pre)
    ops.mhc_post(hidden, residual, post, comb, out=residual)


def _clone_pair(pair, device):
    from rtp_llm.models_py.modules.base.cuda.norm import RMSNorm

    previous, next_hc, norm = pair
    previous, next_hc = _clone_unit(previous, device), _clone_unit(next_hc, device)
    next_hc.set_previous(previous)
    norm = RMSNorm(
        norm.weight.detach().to(device=device).clone(), norm.variance_epsilon
    )
    return previous, next_hc, norm


def _launch_mega(pair, tokens, device):
    previous, next_hc, norm = pair
    previous.pre_mix_out = torch.full(
        (tokens, 4), 0.25, dtype=torch.float32, device=device
    )
    residual = torch.ones((tokens, 4, 5120), dtype=torch.bfloat16, device=device)
    hidden = torch.zeros((tokens, 5120), dtype=torch.bfloat16, device=device)
    post = torch.full((tokens, 4, 1), 0.25, dtype=torch.float32, device=device)
    comb = (
        torch.eye(4, dtype=torch.float32, device=device)
        .expand(tokens, 4, 4)
        .contiguous()
    )
    if (
        v41_mega_mhc.try_fused_post_pre(
            hidden, residual, post, comb, previous, next_hc, norm
        )
        is None
    ):
        raise RuntimeError("V4.1 HC warmup mega_mhc rejected a reachable shape")


@torch.inference_mode()
def warmup_v41_hc_jit(v4, *, max_m, device):
    """Warm reachable HC variants before service readiness, outside graph capture."""
    if not common.model_warm_up_enabled():
        return
    device = torch.device(device)
    if not common._is_cuda_device(device) or int(max_m) <= 0:
        return
    if (
        torch.version.hip is not None
        or torch.cuda.get_device_capability(device)[0] != 10
    ):
        return
    common._assert_not_capturing()
    units, pairs = _collect_v41_hc_configs(v4)
    if not units:
        return
    small_enabled = v41_prenorm._has_prenorm_gemm()
    mega_enabled = bool(pairs) and v41_mega_mhc._get_mega_mhc() is not None
    num_sms = common._get_deep_gemm_num_sms(device) if mega_enabled else 0
    small_ms = _small_prenorm_ms(max_m) if small_enabled else ()
    mega_ms = _mega_representative_ms(int(max_m), num_sms) if mega_enabled else ()
    stream = torch.cuda.current_stream(device)
    key = (
        str(device),
        stream.cuda_stream,
        tuple(sorted(units)),
        tuple(sorted(pairs)),
        int(max_m),
        small_ms,
        mega_ms,
        num_sms,
    )
    if key in _WARMED_KEYS:
        return
    logging.info(
        "[V41 HC] JIT warmup: small M=%s mega M=%s configs=%d seams=%d",
        small_ms,
        mega_ms,
        len(units),
        len(pairs),
    )

    def launch():
        for config, source in units.items():
            unit = _clone_unit(source, device)
            use_small = small_enabled and unit.dim == 5120
            if use_small:
                for tokens in small_ms:
                    # The runtime wrapper contains both the DG GEMM and its
                    # Triton reduction. Only known transient compile errors
                    # receive either existing retry policy.
                    detail = f"config={config} m={tokens} split16"
                    common._run_triton_warmup_launch_with_retry(
                        "V41 HC prenorm",
                        detail,
                        partial(
                            common._run_deepgemm_warmup_launch_with_retry,
                            "V41 HC prenorm",
                            detail,
                            partial(_launch_small_prenorm, unit, tokens, device),
                            device=device,
                        ),
                        device=device,
                    )
            common._run_tilelang_warmup_launch_with_retry(
                "V41 HC TileLang",
                f"config={config}",
                partial(
                    _launch_tilelang_chain,
                    unit,
                    min(int(max_m), 65),
                    device,
                    include_norm=not use_small or int(max_m) > 64,
                ),
                device=device,
            )
            del unit
            common._sync_cuda(device)
            common._release_cuda_cache(device)
        if mega_enabled:
            for config, source in pairs.items():
                pair = _clone_pair(source, device)
                for tokens in mega_ms:
                    common._run_deepgemm_warmup_launch_with_retry(
                        "V41 mega_mhc",
                        f"config={config} m={tokens}",
                        partial(_launch_mega, pair, tokens, device),
                        device=device,
                    )
                del pair
                common._sync_cuda(device)
                common._release_cuda_cache(device)

    # Shared compilation cache, isolated temporary compiler files and retry
    # behavior match the established V4 startup warmups.
    with torch.cuda.device(device):
        common._run_deepgemm_warmup_launches_serialized("V41 HC", launch)
    _WARMED_KEYS.add(key)
    logging.info("[V41 HC] JIT warmup complete")


__all__ = ["warmup_v41_hc_jit"]
