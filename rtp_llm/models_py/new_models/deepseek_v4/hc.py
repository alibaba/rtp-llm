"""Checkpoint-owned FP32 parameters for main's original TileLang mHC modules."""

import torch
from torch import nn

from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.modules.dsv4.hc.tilelang_impl import (
    TileLangHCHead,
    TileLangHCUnit,
)


def _hc_parameters(dim: int, hc_mult: int, *, head: bool):
    if dim <= 0 or hc_mult <= 0:
        raise ValueError("V4 mHC geometry must be positive")
    n = hc_mult if head else hc_mult * (hc_mult + 2)
    return tuple(
        nn.Parameter(torch.empty(shape, dtype=torch.float32), requires_grad=False)
        for shape in ((n, hc_mult * dim), (n,), (1 if head else 3,))
    )


class OwnedHCUnit(TileLangHCUnit, RtpModule):
    def __init__(self, *, dim, hc_mult, **kwargs):
        super().__init__(
            *_hc_parameters(dim, hc_mult, head=False),
            dim=dim,
            hc_mult=hc_mult,
            **kwargs,
        )


class OwnedHCHead(TileLangHCHead, RtpModule):
    def __init__(self, *, dim, hc_mult, **kwargs):
        super().__init__(
            *_hc_parameters(dim, hc_mult, head=True),
            dim=dim,
            hc_mult=hc_mult,
            **kwargs,
        )
