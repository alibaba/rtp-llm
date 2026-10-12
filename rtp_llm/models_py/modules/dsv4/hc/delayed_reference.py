"""Standalone V4.1 reference with explicit per-forward coefficients.

Adapted from donor 88993fcb36838def206525a079145a823e879e8a and the
checkpoint's inference/model.py. The existing V4 Sinkhorn implementation is
shared; coefficient ownership follows V4.1. Upstream integration uses the
HCUnitBase API in delayed.py; this API remains for standalone component fixtures.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from rtp_llm.models_py.modules.dsv4.hc.fallback_impl import _hc_split_sinkhorn


def _residual(x: torch.Tensor) -> None:
    if (
        x.ndim not in (3, 4)
        or x.shape[-2] <= 0
        or x.shape[-1] <= 0
        or x.dtype not in (torch.bfloat16, torch.float16, torch.float32)
    ):
        raise ValueError(
            "Delayed mHC residual requires floating [T, hc, dim] or [B, S, hc, dim]"
        )


def _mix(tensor: torch.Tensor, shape: tuple[int, ...], device: torch.device) -> None:
    if (
        tuple(tensor.shape) != shape
        or tensor.dtype != torch.float32
        or tensor.device != device
    ):
        raise ValueError(
            "Delayed mHC coefficients must be FP32 with matching shape/device"
        )


def identity_pre_mix(residual: torch.Tensor) -> torch.Tensor:
    """Initial global-layer-zero readout selects the first residual lane."""
    _residual(residual)
    result = torch.zeros(
        residual.shape[:-1], dtype=torch.float32, device=residual.device
    )
    result[..., 0] = 1
    return result


def collapse_delayed(
    residual: torch.Tensor, previous_pre_mix: torch.Tensor
) -> torch.Tensor:
    """Use the predecessor's coefficients for attention/FFN or final readout."""
    _residual(residual)
    _mix(previous_pre_mix, tuple(residual.shape[:-1]), residual.device)
    return (
        (residual.float() * previous_pre_mix.unsqueeze(-1))
        .sum(dim=-2)
        .to(residual.dtype)
    )


@dataclass(frozen=True)
class DelayedHCMixes:
    next_pre_mix: torch.Tensor
    post: torch.Tensor
    residual_mix: torch.Tensor


class DelayedHCUnit(nn.Module):
    """Current post/residual mixing, with pre-mix passed to the next sublayer.

    ``pre`` requires the previous coefficients even for the first local layer.
    Stage zero explicitly constructs identity_pre_mix; stage one uses its PP
    frame. No predecessor pointers, cached pre_mix_out or learned head weights.
    """

    def __init__(
        self,
        fn: torch.Tensor,
        base: torch.Tensor,
        scale: torch.Tensor,
        *,
        dim: int,
        hc_mult: int = 4,
        hc_sinkhorn_iters: int = 20,
        norm_eps: float = 1e-6,
        hc_eps: float = 1e-6
    ):
        super().__init__()
        if any(type(n) is not int or n <= 0 for n in (dim, hc_mult, hc_sinkhorn_iters)):
            raise ValueError(
                "Delayed mHC dimensions and Sinkhorn iterations must be positive integers"
            )
        if any(
            not isinstance(e, (float, int))
            or isinstance(e, bool)
            or not math.isfinite(e)
            or e <= 0
            for e in (norm_eps, hc_eps)
        ):
            raise ValueError("Delayed mHC epsilons must be finite and positive")
        width = hc_mult * (hc_mult + 2)
        if (
            fn.shape != (width, hc_mult * dim)
            or base.numel() != width
            or scale.numel() != 3
        ):
            raise ValueError("Delayed mHC weights have incompatible dimensions")
        if any(
            t.dtype != torch.float32 or t.device != fn.device for t in (fn, base, scale)
        ):
            raise ValueError("Delayed mHC weights must be FP32 on one device")
        self.fn = fn.contiguous()
        self.base = base.reshape(width).contiguous()
        self.scale = scale.reshape(3).contiguous()
        self.dim = dim
        self.hc_mult = hc_mult
        self.hc_sinkhorn_iters = hc_sinkhorn_iters
        self.norm_eps = norm_eps
        self.hc_eps = hc_eps

    def _check(self, residual: torch.Tensor) -> None:
        _residual(residual)
        if (
            residual.shape[-2:] != (self.hc_mult, self.dim)
            or residual.device != self.fn.device
        ):
            raise ValueError("Delayed mHC residual disagrees with its weights")

    def pre(
        self, residual: torch.Tensor, previous_pre_mix: torch.Tensor
    ) -> tuple[torch.Tensor, DelayedHCMixes]:
        self._check(residual)
        collapsed = collapse_delayed(residual, previous_pre_mix)
        flat = residual.flatten(-2).float()
        projection = F.linear(flat, self.fn) * torch.rsqrt(
            flat.square().mean(dim=-1, keepdim=True) + self.norm_eps
        )
        pre, post, comb = _hc_split_sinkhorn(
            projection,
            self.scale,
            self.base,
            self.hc_mult,
            self.hc_sinkhorn_iters,
            self.hc_eps,
        )
        return collapsed, DelayedHCMixes(pre, post, comb)

    def post(
        self, output: torch.Tensor, residual: torch.Tensor, mixes: DelayedHCMixes
    ) -> torch.Tensor:
        self._check(residual)
        shape = tuple(residual.shape)
        if (
            output.shape != shape[:-2] + (self.dim,)
            or output.device != residual.device
            or output.dtype != residual.dtype
        ):
            raise ValueError("Delayed mHC sublayer output disagrees with its residual")
        if not isinstance(mixes, DelayedHCMixes):
            raise ValueError("Delayed mHC requires the current sublayer's mixes")
        _mix(mixes.next_pre_mix, shape[:-1], residual.device)
        _mix(mixes.post, shape[:-1], residual.device)
        _mix(
            mixes.residual_mix,
            shape[:-2] + (self.hc_mult, self.hc_mult),
            residual.device,
        )
        # comb[source, destination], matching the checkpoint's explicit reduction.
        combined = (
            mixes.residual_mix.unsqueeze(-1) * residual.float().unsqueeze(-2)
        ).sum(dim=-3)
        return (mixes.post.unsqueeze(-1) * output.float().unsqueeze(-2) + combined).to(
            residual.dtype
        )
