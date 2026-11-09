"""Factory for DeepSeek-V4 Hyper-Connection implementations."""

from __future__ import annotations

import torch

from rtp_llm.models_py.modules.dsv41.hc.base import HCHeadBase, HCUnitBase
from rtp_llm.models_py.modules.dsv41.hc.utils import maybe_squeeze_hc_1d


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
) -> HCUnitBase:
    scale = maybe_squeeze_hc_1d(scale)
    from rtp_llm.models_py.modules.dsv41.hc.tilelang_impl import TileLangHCUnit

    return TileLangHCUnit(
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


def build_hc_head(
    fn: torch.Tensor,
    base: torch.Tensor,
    scale: torch.Tensor,
    *,
    dim: int,
    hc_mult: int,
    norm_eps: float,
    hc_eps: float,
) -> HCHeadBase:
    scale = maybe_squeeze_hc_1d(scale)
    from rtp_llm.models_py.modules.dsv41.hc.tilelang_impl import TileLangHCHead

    return TileLangHCHead(
        fn,
        base,
        scale,
        dim=dim,
        hc_mult=hc_mult,
        norm_eps=norm_eps,
        hc_eps=hc_eps,
    )
