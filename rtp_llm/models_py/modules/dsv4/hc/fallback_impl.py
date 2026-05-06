"""PyTorch reference implementation for DeepSeek-V4 Hyper-Connections."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

from rtp_llm.models_py.modules.dsv4.hc.base import HCHeadBase, HCUnitBase


def _hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """PyTorch reference for the TileLang mHC pre mixer.

    Production uses the TileLang path in ``hc/tilelang_impl.py``. This helper
    only backs ``DSV4_HC_IMPL=fallback`` for CPU/dev/reference runs.
    """
    hc = hc_mult
    *batch, mix_hc = mixes.size()
    assert mix_hc == (hc + 2) * hc, (
        f"mix_hc={mix_hc}, expected (hc+2)*hc={(hc + 2) * hc}"
    )

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


class FallbackHCUnit(HCUnitBase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fn_bf16 = self.fn.to(torch.bfloat16)

    def _fn_bf16(self) -> torch.Tensor:
        if (
            self.fn_bf16.shape != self.fn.shape
            or self.fn_bf16.device != self.fn.device
        ):
            self.fn_bf16 = self.fn.to(torch.bfloat16)
        return self.fn_bf16

    def _linear_mixes(self, x_flat: torch.Tensor, rsqrt: torch.Tensor) -> torch.Tensor:
        return (F.linear(x_flat, self._fn_bf16()) * rsqrt).float()

    def _pre_impl(self, x: torch.Tensor, dbg_tag=None):
        shape, dtype = x.size(), x.dtype
        x_flat = x.flatten(-2)
        x_flat_f32 = x_flat.float()
        rsqrt = torch.rsqrt(
            x_flat_f32.square().mean(-1, keepdim=True) + self.norm_eps
        ).to(dtype)
        mixes = self._linear_mixes(x_flat, rsqrt).contiguous()
        pre, post, comb = _hc_split_sinkhorn(
            mixes,
            self.scale,
            self.base,
            hc_mult=self.hc_mult,
            sinkhorn_iters=self.hc_sinkhorn_iters,
            eps=self.hc_eps,
        )
        y = torch.sum(pre.to(dtype).unsqueeze(-1) * x.view(*shape), dim=-2)
        return y.to(dtype), post.unsqueeze(-1), comb

    def _post_impl(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ) -> torch.Tensor:
        post_b = post.squeeze(-1) if post.dim() == residual.dim() else post
        if x.dtype == torch.bfloat16 and residual.dtype == torch.bfloat16:
            post_bf16 = post_b.to(x.dtype)
            first = post_bf16.unsqueeze(-1) * x.unsqueeze(-2)
            comb_bf16 = comb.to(x.dtype)
            second = torch.sum(comb_bf16.unsqueeze(-1) * residual.unsqueeze(-2), dim=-3)
            return (first + second).to(x.dtype)

        y = post_b.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(
            comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=-3
        )
        return y.type_as(x)


class FallbackHCHead(HCHeadBase):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fn_bf16 = self.fn.to(torch.bfloat16)

    def _head_impl(self, x: torch.Tensor) -> torch.Tensor:
        shape, dtype = x.size(), x.dtype
        x_flat = x.flatten(-2).float()
        rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x_flat, self.fn) * rsqrt
        pre = torch.sigmoid(mixes * self.scale + self.base) + self.hc_eps
        y = torch.sum(pre.unsqueeze(-1) * x_flat.view(shape), dim=-2)
        return y.to(dtype)
