"""DeepSeek-V4 Hyper-Connections (mHC) residual layer.

Residual stream lives in shape `[B, T, hc_mult, d]` instead of the usual `[B, T, d]`.
Each transformer block applies hc_pre/F/hc_post twice (for attention and FFN):

  residual = x                         # [B, T, hc, d]
  x, post, comb = hc_pre(x, hc_fn_attn)
  x = norm(x); x = F_attn(x)           # F operates on [B, T, d]
  x = hc_post(x, residual, post, comb) # back to [B, T, hc, d]
  ...same for FFN

The pre/post/comb mixers are dynamically generated per token via:
  raw = RMSNorm(flatten(x)) @ hc_fn   # raw shape [B, T, mix_hc] where mix_hc = (hc+2)*hc
  pre  = sigmoid(raw[..., :hc] * scale[0] + bias[:hc]) + eps
  post = 2 * sigmoid(raw[..., hc:2hc] * scale[1] + bias[hc:2hc])
  comb_raw = raw[..., 2hc:] * scale[2] + bias[2hc:]   # reshape [B, T, hc, hc]
  comb = sinkhorn_knopp(softmax_row(comb_raw), iters=hc_sinkhorn_iters)

This is the PyTorch reference implementation. A fused TileLang kernel
(`hc_split_sinkhorn_kernel`) replaces this in production.

Reference: DeepSeek `inference/model.py` Block / Transformer.hc_head /
`inference/kernel.py` hc_split_sinkhorn_kernel.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """PyTorch reference of the TileLang hc_split_sinkhorn kernel.

    Args:
      mixes:    [..., (hc+2)*hc]
      hc_scale: [3]
      hc_base:  [(hc+2)*hc]
    Returns:
      pre:  [..., hc]    sigmoid + eps
      post: [..., hc]    2 * sigmoid
      comb: [..., hc, hc] doubly-stochastic (Sinkhorn-Knopp normalized)
    """
    hc = hc_mult
    *batch, mix_hc = mixes.size()
    assert mix_hc == (hc + 2) * hc, f"mix_hc={mix_hc}, expected (hc+2)*hc={(hc + 2) * hc}"

    pre_raw = mixes[..., :hc] * hc_scale[0] + hc_base[:hc]
    pre = pre_raw.sigmoid() + eps

    post_raw = mixes[..., hc:2 * hc] * hc_scale[1] + hc_base[hc:2 * hc]
    post = 2.0 * post_raw.sigmoid()

    comb_raw = mixes[..., 2 * hc:] * hc_scale[2] + hc_base[2 * hc:]
    comb = comb_raw.view(*batch, hc, hc)

    # Step A: row-softmax + eps
    comb = comb.softmax(dim=-1) + eps
    # Step B: column normalize
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    # Step C-...: alternating row/col normalization (sinkhorn_iters - 1 more times)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)

    return pre, post, comb


class HyperConnection(nn.Module):
    """Hyper-Connection residual operator for one (pre/post) pair.

    Holds the parameters that generate the (pre, post, comb) mixers:
      hc_fn:    [(hc+2)*hc, hc*d]   FP32
      hc_base:  [(hc+2)*hc]         FP32
      hc_scale: [3]                 FP32
    """

    def __init__(self, hc_mult: int, dim: int, hc_sinkhorn_iters: int = 20,
                 norm_eps: float = 1e-6, hc_eps: float = 1e-6):
        super().__init__()
        self.hc_mult = hc_mult
        self.dim = dim
        self.hc_sinkhorn_iters = hc_sinkhorn_iters
        self.norm_eps = norm_eps
        self.hc_eps = hc_eps
        mix_hc = (2 + hc_mult) * hc_mult
        # Bound externally by V4Transformer factory mode (Block.__init__) —
        # see QuantizedLinear note.
        self.hc_fn = None
        self.hc_base = None
        self.hc_scale = None

    def hc_pre(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Read out one mixed residual stream from the hc_mult copies.

        Args:
          x: [B, T, hc, d]
        Returns:
          y:    [B, T, d]    single-stream input for the layer body F
          post: [B, T, hc]   for hc_post
          comb: [B, T, hc, hc]
        """
        shape, dtype = x.size(), x.dtype
        x_flat = x.flatten(2).float()  # [B, T, hc*d]
        rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x_flat, self.hc_fn) * rsqrt
        pre, post, comb = hc_split_sinkhorn(
            mixes, self.hc_scale, self.hc_base,
            hc_mult=self.hc_mult,
            sinkhorn_iters=self.hc_sinkhorn_iters,
            eps=self.hc_eps,
        )
        # y[b, t, d] = sum_h pre[b, t, h] * x[b, t, h, d]
        y = torch.sum(pre.unsqueeze(-1) * x_flat.view(shape), dim=2)
        return y.to(dtype), post, comb

    def hc_post(self, x: torch.Tensor, residual: torch.Tensor,
                post: torch.Tensor, comb: torch.Tensor) -> torch.Tensor:
        """Write F's single-stream output back into the hc_mult residual copies.

        Args:
          x:        [B, T, d]      F's output
          residual: [B, T, hc, d]  the pre-pre x (input to hc_pre)
          post:     [B, T, hc]
          comb:     [B, T, hc, hc]
        Returns:
          y: [B, T, hc, d]
        """
        # post[b,t,i] * x[b,t,d] expanded over hc: post.unsqueeze(-1) * x.unsqueeze(-2)
        # comb[b,t,i,j] * residual[b,t,j,d] summed over j
        y = (post.unsqueeze(-1) * x.unsqueeze(-2)
             + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2))
        return y.type_as(x)


class HyperConnectionHead(nn.Module):
    """The hc_head reduction at the model output.

    Merges the hc_mult residual copies back to a single stream before the
    final norm + lm_head. Uses only `pre` (no post/comb), and pre is plain
    sigmoid (no factor-of-2) per the official `Transformer.hc_head` /
    `ParallelHead.hc_head`.
    """

    def __init__(self, hc_mult: int, dim: int, norm_eps: float = 1e-6, hc_eps: float = 1e-6):
        super().__init__()
        self.hc_mult = hc_mult
        self.dim = dim
        self.norm_eps = norm_eps
        self.hc_eps = hc_eps
        # Bound externally — see QuantizedLinear note.
        self.hc_head_fn = None
        self.hc_head_base = None
        self.hc_head_scale = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, hc, d]  ->  y: [B, T, d]"""
        shape, dtype = x.size(), x.dtype
        x_flat = x.flatten(2).float()
        rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x_flat, self.hc_head_fn) * rsqrt   # [B, T, hc]
        pre = torch.sigmoid(mixes * self.hc_head_scale + self.hc_head_base) + self.hc_eps
        y = torch.sum(pre.unsqueeze(-1) * x_flat.view(shape), dim=2)
        return y.to(dtype)


def expand_to_hc(x: torch.Tensor, hc_mult: int) -> torch.Tensor:
    """Embedding output [B, T, d] -> [B, T, hc, d] for the residual stream."""
    return x.unsqueeze(2).repeat(1, 1, hc_mult, 1)
