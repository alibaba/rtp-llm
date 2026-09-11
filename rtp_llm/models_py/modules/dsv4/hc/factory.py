"""Factory for DeepSeek-V4 Hyper-Connection implementations."""

from __future__ import annotations

import os

import torch
from rtp_llm.models_py.modules.dsv4.hc.base import HCHeadBase, HCMode, HCUnitBase
from rtp_llm.models_py.modules.dsv4.hc.utils import maybe_squeeze_hc_1d


def _mode_from_env() -> HCMode:
    raw = os.environ.get("DSV4_HC_IMPL", HCMode.TILELANG.value).lower()
    try:
        return HCMode(raw)
    except ValueError as exc:
        allowed = ", ".join(m.value for m in HCMode)
        raise ValueError(
            f"invalid DSV4_HC_IMPL={raw!r}; expected one of: {allowed}"
        ) from exc


def build_hc_unit(
    fn: torch.Tensor,
    base: torch.Tensor,
    scale: torch.Tensor,
    *,
    dim: int,
    hc_mult: int,
    hc_sinkhorn_iters: int,
    norm_eps: float,
    hc_eps: float,
    layer_id: int = -1,
    name: str = "",
    tp_size: int = 1,
    tp_rank: int = 0,
    platform_provider=None,
) -> HCUnitBase:
    builder = getattr(platform_provider, "build_hc_unit", None)
    if builder is not None:
        return builder(
            fn,
            base,
            maybe_squeeze_hc_1d(scale),
            dim=dim,
            hc_mult=hc_mult,
            hc_sinkhorn_iters=hc_sinkhorn_iters,
            norm_eps=norm_eps,
            hc_eps=hc_eps,
            layer_id=layer_id,
            name=name,
            tp_size=tp_size,
            tp_rank=tp_rank,
        )
    mode = _mode_from_env()
    scale = maybe_squeeze_hc_1d(scale)
    if mode is HCMode.TILELANG:
        from rtp_llm.models_py.modules.dsv4.hc.tilelang_impl import TileLangHCUnit

        unit = TileLangHCUnit(
            fn,
            base,
            scale,
            dim=dim,
            hc_mult=hc_mult,
            hc_sinkhorn_iters=hc_sinkhorn_iters,
            norm_eps=norm_eps,
            hc_eps=hc_eps,
            layer_id=layer_id,
            name=name,
        )
    elif mode is HCMode.HYBRID:
        from rtp_llm.models_py.modules.dsv4.hc.tilelang_impl import HybridHCUnit

        unit = HybridHCUnit(
            fn,
            base,
            scale,
            dim=dim,
            hc_mult=hc_mult,
            hc_sinkhorn_iters=hc_sinkhorn_iters,
            norm_eps=norm_eps,
            hc_eps=hc_eps,
            layer_id=layer_id,
            name=name,
        )
    else:
        from rtp_llm.models_py.modules.dsv4.hc.fallback_impl import FallbackHCUnit

        unit = FallbackHCUnit(
            fn,
            base,
            scale,
            dim=dim,
            hc_mult=hc_mult,
            hc_sinkhorn_iters=hc_sinkhorn_iters,
            norm_eps=norm_eps,
            hc_eps=hc_eps,
            layer_id=layer_id,
            name=name,
        )
    unit.tp_size = int(tp_size)
    unit.tp_rank = int(tp_rank)
    return unit


def build_hc_head(
    fn: torch.Tensor,
    base: torch.Tensor,
    scale: torch.Tensor,
    *,
    dim: int,
    hc_mult: int,
    norm_eps: float,
    hc_eps: float,
    tp_size: int = 1,
    tp_rank: int = 0,
    platform_provider=None,
) -> HCHeadBase:
    builder = getattr(platform_provider, "build_hc_head", None)
    if builder is not None:
        return builder(
            fn,
            base,
            maybe_squeeze_hc_1d(scale),
            dim=dim,
            hc_mult=hc_mult,
            norm_eps=norm_eps,
            hc_eps=hc_eps,
            tp_size=tp_size,
            tp_rank=tp_rank,
        )
    mode = _mode_from_env()
    scale = maybe_squeeze_hc_1d(scale)
    if mode is HCMode.TILELANG:
        from rtp_llm.models_py.modules.dsv4.hc.tilelang_impl import TileLangHCHead

        head = TileLangHCHead(
            fn,
            base,
            scale,
            dim=dim,
            hc_mult=hc_mult,
            norm_eps=norm_eps,
            hc_eps=hc_eps,
        )
    else:
        # Hybrid deliberately uses the validated FP32 fallback head because
        # only the PRE path has passed the production-shape PPU gate.
        from rtp_llm.models_py.modules.dsv4.hc.fallback_impl import FallbackHCHead

        head = FallbackHCHead(
            fn,
            base,
            scale,
            dim=dim,
            hc_mult=hc_mult,
            norm_eps=norm_eps,
            hc_eps=hc_eps,
        )
    head.tp_size = int(tp_size)
    head.tp_rank = int(tp_rank)
    return head
