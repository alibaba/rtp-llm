"""PPU overrides; unchanged behavior is inherited from the shared DSV4 implementation."""

from __future__ import annotations
from rtp_llm.models_py.modules.dsv4.hc.tilelang_impl import (
    _require_contiguous,
    _raise_unavailable,
    TileLangHCHead,
)
from typing import NoReturn
import torch
from rtp_llm.models_py.modules.dsv4.hc.base import HCHeadBase, HCUnitBase
from rtp_llm.models_py.modules.dsv4.hc.mhc_tilelang import (
    tk_mhc_head,
    tk_mhc_head_fused,
    tk_mhc_head_fused_enabled,
    tk_mhc_post,
    tk_mhc_pre,
)
from rtp_llm.models_py.modules.dsv4.hc.utils import squeeze_hc_batch, wrap_hc_batch
from rtp_llm.models_py.modules.dsv4.hc.tilelang_impl import TileLangHCUnit as _BaseTileLangHCUnit

class TileLangHCUnit(_BaseTileLangHCUnit):
    def _pre_impl(self, x: torch.Tensor, dbg_tag=None):
        # Public HC pre input is [T, hc, dim] or [B, S, hc, dim].
        # TileLang requires [B, S, hc, dim], so flat prefill is wrapped as
        # [1, T, hc, dim] after rank/trailing-dim checks in HCUnitBase.
        tk_x, wrapped = wrap_hc_batch(x, 4, name="mhc_pre residual")
        # TileLang kernels assume dense row-major tensors. Flat prefill
        # unsqueeze is a metadata-only view that remains contiguous; any
        # non-contiguous input means an upstream layout change must be fixed
        # there instead of hiding an allocation in this hot HC path.
        tk_x = _require_contiguous(tk_x, name="mhc_pre residual")
        with torch.inference_mode():
            out = getattr(self, "_pre_operator", tk_mhc_pre)(
                tk_x,
                self.fn,
                self.scale,
                self.base,
                norm_eps=self.norm_eps,
                pre_eps=self.hc_eps,
                sinkhorn_eps=self.hc_eps,
                sinkhorn_iters=self.hc_sinkhorn_iters,
                hc_mult=self.hc_mult,
            )
        if out is None:
            _raise_unavailable("pre", x, self.hc_mult)
        y, post, comb = out
        return (
            squeeze_hc_batch(y, wrapped, name="mhc_pre y"),
            squeeze_hc_batch(post, wrapped, name="mhc_pre post"),
            squeeze_hc_batch(comb, wrapped, name="mhc_pre comb"),
        )

    def _post_impl(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ) -> torch.Tensor:
        # Public HC post inputs preserve the same leading token layout:
        # x [T, dim] / [B, S, dim], residual [T, hc, dim] /
        # [B, S, hc, dim], post [T, hc, 1] / [B, S, hc, 1], and comb
        # [T, hc, hc] / [B, S, hc, hc]. TileLang consumes the batched form.
        tk_x, wrapped = wrap_hc_batch(x, 3, name="mhc_post x")
        tk_residual, _ = wrap_hc_batch(residual, 4, name="mhc_post residual")
        tk_post, _ = wrap_hc_batch(post, 4, name="mhc_post post")
        tk_comb, _ = wrap_hc_batch(comb, 4, name="mhc_post comb")
        # Keep TileLang layout requirements explicit. In the normal path these
        # are already contiguous: ``x`` comes from attention/MoE outputs and
        # ``post``/``comb`` come directly from TileLang pre.
        tk_x = _require_contiguous(tk_x, name="mhc_post x")
        tk_residual = _require_contiguous(tk_residual, name="mhc_post residual")
        tk_post = _require_contiguous(tk_post, name="mhc_post post")
        tk_comb = _require_contiguous(tk_comb, name="mhc_post comb")
        with torch.inference_mode():
            # In-place reuse: the HC residual stream is overwritten by the next
            # sublayer immediately after this call, so the input buffer is dead
            # at return. Aliasing out=residual saves a per-call empty_like
            # (residual.numel() * 2 bytes, e.g. 7.5 GB at T=128K, hc=4,
            # dim=7168). Kernel-side safety: each block reads residual[pid_n]
            # into shared memory before writing out[pid_n] at the same slot.
            out = getattr(self, "_post_operator", tk_mhc_post)(
                tk_x,
                tk_residual,
                tk_post,
                tk_comb,
                hc_mult=self.hc_mult,
                out=tk_residual,
            )
        if out is None:
            _raise_unavailable("post", residual, self.hc_mult)
        return squeeze_hc_batch(out, wrapped, name="mhc_post output")
