"""Manifold-Constrained Hyper-Connections (mHC) residual layer for DeepSeek-V4.

The hidden state in V4 is `[bsz, seqlen, hc_mult, dim]` — `hc_mult` (default 4)
copies of the residual stream maintained throughout the model. Each transformer
block alternates two passes:

    pre, post, comb = hc_split_sinkhorn(linear(rmsnorm(x), hc_fn) * rsqrt, hc_scale, hc_base)
    y      = sum(pre[..., None] * x, dim=hc_axis)               # reduce hc → 1
    y_attn = attn(attn_norm(y))
    x      = post[..., None] * y_attn[..., None, :] + sum(comb[..., None] * x[..., None, :], dim=hc_axis)
    # same pattern for ffn

`hc_fn` is `[(2 + hc_mult) * hc_mult, hc_mult * dim]`, `hc_scale` is `[3]`,
`hc_base` is `[(2 + hc_mult) * hc_mult]`. The Sinkhorn-Knopp inner loop projects
the comb matrix to a doubly-stochastic matrix on the Birkhoff polytope.

Reference: `inference/model.py:Block.hc_pre/hc_post` and
`inference/kernel.py:hc_split_sinkhorn_kernel` from the official DeepSeek-V4
release (`.deps/ds_v4_official/inference/`).
"""

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn


def hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int,
    sinkhorn_iters: int,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """PyTorch reference for the official `hc_split_sinkhorn` TileLang kernel.

    Args:
        mixes:    [N, (2 + hc_mult) * hc_mult] float32
        hc_scale: [3] float32 — scales for (pre, post, comb) groups
        hc_base:  [(2 + hc_mult) * hc_mult] float32 — biases for each mix slot
        hc_mult:  number of HC copies (4 in V4)
        sinkhorn_iters: 20 in V4
        eps:      1e-6 in V4

    Returns:
        pre:  [N, hc_mult]
        post: [N, hc_mult]
        comb: [N, hc_mult, hc_mult]  doubly-stochastic
    """
    n = mixes.shape[0]
    h = hc_mult
    # Slice layout matches the official kernel:
    # pre  ← mixes[:, 0:h]      * hc_scale[0] + hc_base[0:h]
    # post ← mixes[:, h:2h]     * hc_scale[1] + hc_base[h:2h]
    # comb ← mixes[:, 2h:2h+h*h]* hc_scale[2] + hc_base[2h:2h+h*h]   reshape → [N, h, h]
    pre = torch.sigmoid(mixes[:, :h] * hc_scale[0] + hc_base[:h]) + eps
    post = 2.0 * torch.sigmoid(mixes[:, h : 2 * h] * hc_scale[1] + hc_base[h : 2 * h])
    comb_logits = mixes[:, 2 * h :] * hc_scale[2] + hc_base[2 * h :]
    comb = comb_logits.view(n, h, h)

    # Iter 0 is special: row-softmax (numerically stable via subtract-max), then
    # one column-normalize. Subsequent iters are alternate row-/column-normalize.
    row_max, _ = comb.max(dim=-1, keepdim=True)
    comb = (comb - row_max).exp()
    comb = comb / (comb.sum(dim=-1, keepdim=True) + eps) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb


class MHCMixing(nn.Module):
    """Per-block mHC mixing parameters and pre/post fold operators.

    Owns one set of `(hc_fn, hc_scale, hc_base)`. Used twice per Block (once for
    attn, once for ffn), and once at the top-level head for the final 4→1 fold.
    """

    def __init__(self, dim: int, hc_mult: int, norm_eps: float):
        super().__init__()
        self.dim = dim
        self.hc_mult = hc_mult
        self.norm_eps = norm_eps
        mix_hc = (2 + hc_mult) * hc_mult
        hc_dim = hc_mult * dim
        # Stored fp32 — checkpoints use bf16 but compute is fp32 for stability.
        self.hc_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))

    def pre(
        self, x: torch.Tensor, sinkhorn_iters: int, eps: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reduce hc copies to 1 and emit (pre, post, comb) for this layer.

        x:      [b, s, hc, d]
        return: y [b, s, d], post [b, s, hc], comb [b, s, hc, hc]
        """
        b, s, hc, d = x.shape
        assert hc == self.hc_mult and d == self.dim, (
            f"MHCMixing.pre got x={x.shape}, expected hc={self.hc_mult} d={self.dim}"
        )
        x_flat = x.flatten(2).float()  # [b, s, hc*d]
        rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x_flat, self.hc_fn) * rsqrt  # [b, s, mix_hc]
        pre, post, comb = hc_split_sinkhorn(
            mixes.view(-1, mixes.shape[-1]),
            self.hc_scale,
            self.hc_base,
            self.hc_mult,
            sinkhorn_iters,
            eps,
        )
        pre = pre.view(b, s, hc)
        post = post.view(b, s, hc)
        comb = comb.view(b, s, hc, hc)
        # Reduce hc copies to 1 with the pre weights.
        y = (pre.unsqueeze(-1) * x.float()).sum(dim=2)  # [b, s, d]
        return y.to(x.dtype), post, comb

    @staticmethod
    def post(
        attn_out: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ) -> torch.Tensor:
        """Expand 1 → hc copies, mixing the new layer output back into the residual stream.

        attn_out: [b, s, d]
        residual: [b, s, hc, d]
        post:     [b, s, hc]
        comb:     [b, s, hc, hc]
        return:   [b, s, hc, d]
        """
        # post[..., None]: [b, s, hc, 1]; attn_out.unsqueeze(-2): [b, s, 1, d]
        # broadcast → [b, s, hc, d]
        new = post.unsqueeze(-1) * attn_out.unsqueeze(-2)
        # comb[..., None]: [b, s, hc, hc, 1]; residual.unsqueeze(-3): [b, s, 1, hc, d]?
        # Wait — official: torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
        # comb.unsqueeze(-1): [b, s, hc, hc, 1]; residual.unsqueeze(-2): [b, s, hc, 1, d]
        # → product [b, s, hc, hc, d], sum over dim=2 (the first hc axis = "from copy")
        # → [b, s, hc, d]
        mixed = (comb.unsqueeze(-1) * residual.unsqueeze(-2)).sum(dim=2)
        return (new + mixed).type_as(residual)


class MHCHead(nn.Module):
    """Top-level fold: hc copies → 1 just before final RMSNorm + lm_head.

    Simpler than MHCMixing — only the `pre` weights (sigmoid + eps), no post / comb,
    no Sinkhorn-Knopp, since we don't need to expand back.
    """

    def __init__(self, dim: int, hc_mult: int, norm_eps: float, hc_eps: float):
        super().__init__()
        self.dim = dim
        self.hc_mult = hc_mult
        self.norm_eps = norm_eps
        self.hc_eps = hc_eps
        hc_dim = hc_mult * dim
        self.hc_fn = nn.Parameter(torch.empty(hc_mult, hc_dim, dtype=torch.float32))
        self.hc_base = nn.Parameter(torch.empty(hc_mult, dtype=torch.float32))
        self.hc_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [b, s, hc, d] → [b, s, d]"""
        shape = x.shape
        x_flat = x.flatten(2).float()
        rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x_flat, self.hc_fn) * rsqrt  # [b, s, hc]
        pre = torch.sigmoid(mixes * self.hc_scale + self.hc_base) + self.hc_eps
        y = (pre.unsqueeze(-1) * x.float().view(shape)).sum(dim=2)
        return y.to(x.dtype)
