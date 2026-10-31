"""PyTorch reference mixer for DeepSeek-V4 Hyper-Connections.

Production runs the TileLang path in ``hc/tilelang_impl.py`` (the factory
always builds it). This module keeps only the FP32 sinkhorn mix splitter
used by ``DelayedHCUnit``'s CPU/dev branch in ``hc/delayed.py``.
"""

from __future__ import annotations

from typing import Tuple

import torch


def _hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """PyTorch reference for the TileLang mHC pre mixer.

    Production uses the TileLang path in ``hc/tilelang_impl.py``; the factory
    always builds it. This helper backs ``DelayedHCUnit``'s CPU/dev branch in
    ``hc/delayed.py``.
    """
    hc = hc_mult
    *batch, mix_hc = mixes.size()

    pre_raw = mixes[..., :hc] * hc_scale[0] + hc_base[:hc]
    pre = pre_raw.sigmoid() + eps

    post_raw = mixes[..., hc : 2 * hc] * hc_scale[1] + hc_base[hc : 2 * hc]
    post = 2.0 * post_raw.sigmoid()

    comb_raw = mixes[..., 2 * hc :] * hc_scale[2] + hc_base[2 * hc :]
    comb = comb_raw.view(*batch, hc, hc)
    comb = comb.softmax(dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)

    return pre, post, comb
