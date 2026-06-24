"""Symm-mem buffer + scratch caches + capability gates for DeepGEMM
``fp8_fp4_mega_moe_fused``.

This is the fused-kernel sibling of ``mega_buf.py``.  The fused kernel
(``deep_gemm.fp8_fp4_mega_moe_fused``) folds the shared expert + the
``routed + shared`` add into the single Mega MoE kernel, so on top of the
symmetric-memory dispatch buffer it also needs two small per-rank scratch
tensors for the shared-expert intermediate tile (``mid_fp8`` / ``mid_sf``).

Like ``mega_buf.py`` everything here is single-layer staging shared across all
MoE layers via module-level caches — the previous layer's data is dead once the
next layer's MoE starts, so one buffer per process bounds symm/HBM memory.

The fused path is opt-in: it is only selected when ``DSV4_USE_MEGA_MOE_FUSED=1``
is set.  The default MoE path keeps using ``fp8_fp4_mega_moe`` (see
``mega_buf.py`` / ``strategies/mega.py``).
"""

import logging
import os

import torch

from .mega_buf import _mega_moe_unavailable_reason

# Module-level caches for the fused Mega MoE symm-mem dispatch buffer, the
# BF16 output staging buffer, and the shared-expert mid-tile scratch. Keyed by
# the shape parameters so different configs in one process don't collide; in
# practice there's only ever one entry per process.
_MEGA_FUSED_BUF_CACHE: dict = {}
_MEGA_FUSED_OUTPUT_CACHE: dict = {}
_MEGA_FUSED_MID_CACHE: dict = {}

_USE_MEGA_MOE_FUSED_ENV = "DSV4_USE_MEGA_MOE_FUSED"


def mega_moe_fused_requested() -> bool:
    """Whether the operator wants the fused kernel (opt-in via env).

    The default (env unset / ``0``) keeps the non-fused ``fp8_fp4_mega_moe``
    path. Only ``DSV4_USE_MEGA_MOE_FUSED=1`` switches to
    ``fp8_fp4_mega_moe_fused``.
    """
    return os.environ.get(_USE_MEGA_MOE_FUSED_ENV, "0") == "1"


def estimate_mega_moe_fused_symm_buffer_bytes(
    group_size: int,
    num_experts: int,
    num_max_tokens_per_rank: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
    use_fp8_dispatch: bool = True,
    activation: str = "swiglu",
) -> int | None:
    try:
        import deep_gemm

        return int(
            deep_gemm._C.get_symm_buffer_size_for_mega_moe_fused(
                group_size,
                num_experts,
                num_max_tokens_per_rank,
                num_topk,
                hidden,
                intermediate_hidden,
                use_fp8_dispatch,
                activation,
            )[0]
        )
    except Exception:
        return None


def _get_or_create_mega_fused_buf(
    group,
    num_experts,
    num_max_tokens_per_rank,
    num_topk,
    hidden,
    intermediate_hidden,
    use_fp8_dispatch,
    activation,
):
    import deep_gemm

    key = (
        id(group),
        num_experts,
        num_max_tokens_per_rank,
        num_topk,
        hidden,
        intermediate_hidden,
        bool(use_fp8_dispatch),
        activation,
    )
    buf = _MEGA_FUSED_BUF_CACHE.get(key)
    if buf is None:
        try:
            group_size = int(group.size())
        except Exception:
            group_size = 0
        estimated_bytes = (
            estimate_mega_moe_fused_symm_buffer_bytes(
                group_size=group_size,
                num_experts=num_experts,
                num_max_tokens_per_rank=num_max_tokens_per_rank,
                num_topk=num_topk,
                hidden=hidden,
                intermediate_hidden=intermediate_hidden,
                use_fp8_dispatch=use_fp8_dispatch,
                activation=activation,
            )
            if group_size > 0
            else None
        )
        buf = deep_gemm.get_symm_buffer_for_mega_moe_fused(
            group=group,
            num_experts=num_experts,
            num_max_tokens_per_rank=num_max_tokens_per_rank,
            num_topk=num_topk,
            hidden=hidden,
            intermediate_hidden=intermediate_hidden,
            use_fp8_dispatch=use_fp8_dispatch,
            activation=activation,
        )
        actual_bytes = None
        try:
            actual_bytes = int(buf.buffer.numel() * buf.buffer.element_size())
        except Exception:
            pass
        if actual_bytes is not None and estimated_bytes is not None:
            logging.info(
                "[DSV4 MegaMoEFused] allocated symm buffer: group_size=%d "
                "num_experts=%d max_tokens_per_rank=%d topk=%d hidden=%d "
                "intermediate=%d actual=%.3f GiB estimated=%.3f GiB",
                group_size,
                num_experts,
                num_max_tokens_per_rank,
                num_topk,
                hidden,
                intermediate_hidden,
                actual_bytes / (1024**3),
                estimated_bytes / (1024**3),
            )
        elif actual_bytes is not None:
            logging.info(
                "[DSV4 MegaMoEFused] allocated symm buffer: group_size=%d "
                "num_experts=%d max_tokens_per_rank=%d topk=%d hidden=%d "
                "intermediate=%d actual=%.3f GiB",
                group_size,
                num_experts,
                num_max_tokens_per_rank,
                num_topk,
                hidden,
                intermediate_hidden,
                actual_bytes / (1024**3),
            )
        _MEGA_FUSED_BUF_CACHE[key] = buf
    return buf


def _get_or_create_mega_fused_output(
    capacity,
    hidden,
    dtype,
    device,
):
    key = (device, hidden, dtype)
    cached = _MEGA_FUSED_OUTPUT_CACHE.get(key)
    if cached is not None and cached.size(0) >= capacity:
        return cached
    cached = torch.empty((max(capacity, 1), hidden), dtype=dtype, device=device)
    _MEGA_FUSED_OUTPUT_CACHE[key] = cached
    return cached


def _get_or_create_mega_fused_mid(
    capacity,
    intermediate_hidden,
    device,
):
    """Allocate (or reuse) the shared-expert mid-tile scratch tensors.

    Mirrors the test harness allocation
    (``DeepGEMM/opt_fused/test_mega_moe_fused.py``):

      mid_fp8: ``[capacity, intermediate_hidden]`` float8_e4m3fn, contiguous.
      mid_sf:  INT32 column-major (MN-major TMA-aligned), logical shape
               ``(_T_pad, intermediate_hidden // 128)`` via ``.T`` of a
               ``(intermediate_hidden // 128, _T_pad)`` tensor.

    ``_T_pad`` pads token count up so the SE L2 GEMM can consume the SF in
    UTCCP 4x32 token order; even a tiny decode batch needs one full 256-token
    SF tile. ``mid_sf.size(0) >= mid_fp8.size(0)`` is required by the C++.
    """
    import deep_gemm

    assert (
        intermediate_hidden % 128 == 0
    ), f"intermediate_hidden={intermediate_hidden} must be divisible by 128"
    capacity = max(int(capacity), 1)
    key = (device, intermediate_hidden)
    cached = _MEGA_FUSED_MID_CACHE.get(key)
    if cached is not None and int(cached["capacity"]) >= capacity:
        return cached["mid_fp8"], cached["mid_sf"]

    mid_fp8 = torch.empty(
        (capacity, intermediate_hidden),
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    t_pad = max(
        int(deep_gemm.get_tma_aligned_size(capacity, 4)),
        ((capacity + 255) // 256) * 256,
    )
    mid_sf = torch.empty(
        (intermediate_hidden // 128, t_pad),
        dtype=torch.int32,
        device=device,
    ).T
    _MEGA_FUSED_MID_CACHE[key] = {
        "capacity": capacity,
        "mid_fp8": mid_fp8,
        "mid_sf": mid_sf,
    }
    return mid_fp8, mid_sf


def _mega_moe_fused_unavailable_reason() -> str | None:
    """Return ``None`` when the fused Mega MoE kernel can run, else a reason.

    Builds on the non-fused availability check (dist init, world_size > 1,
    SM100, ...) and additionally requires the fused entrypoints from a recent
    DeepGEMM build.
    """
    base = _mega_moe_unavailable_reason()
    if base is not None:
        return base
    try:
        import deep_gemm

        for sym in (
            "fp8_fp4_mega_moe_fused",
            "get_symm_buffer_for_mega_moe_fused",
            "transform_weights_for_mega_moe_fused",
            "transform_shared_expert_weights_for_mega_moe_fused",
        ):
            if not hasattr(deep_gemm, sym):
                return f"deep_gemm.{sym} is missing (DeepGEMM too old for fused MoE)"
    except Exception as e:
        return f"failed to import deep_gemm: {e}"
    return None


def _mega_moe_fused_available() -> bool:
    return _mega_moe_fused_unavailable_reason() is None


def _mega_moe_fused_enabled() -> bool:
    """Fused path is opt-in: ``DSV4_USE_MEGA_MOE_FUSED=1`` AND available."""
    return mega_moe_fused_requested() and _mega_moe_fused_available()


def _mega_moe_fused_disabled_or_unavailable_reason() -> str:
    if not mega_moe_fused_requested():
        return f"{_USE_MEGA_MOE_FUSED_ENV} is not set to 1"
    return (
        _mega_moe_fused_unavailable_reason()
        or "unknown Mega MoE fused availability failure"
    )
