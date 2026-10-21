"""TileLang implementation for DeepSeek-V4 Hyper-Connections."""

from __future__ import annotations

import torch

from rtp_llm.models_py.modules.dsv41.hc.base import HCHeadBase, HCUnitBase
from rtp_llm.models_py.modules.dsv41.hc.mhc_tilelang import (
    tk_mhc_head,
    tk_mhc_head_fused,
    tk_mhc_head_fused_enabled,
    tk_mhc_post,
    tk_mhc_pre,
)
from rtp_llm.models_py.modules.dsv41.hc.utils import squeeze_hc_batch, wrap_hc_batch


class TileLangHCUnit(HCUnitBase):
    def _pre_impl(self, x: torch.Tensor, dbg_tag=None):
        # Public HC pre input is [T, hc, dim] or [B, S, hc, dim].
        # TileLang requires [B, S, hc, dim], so flat prefill is wrapped as
        # [1, T, hc, dim].
        tk_x, wrapped = wrap_hc_batch(x, 4, name="mhc_pre residual")
        # TileLang kernels assume dense row-major tensors. Flat prefill
        # unsqueeze is a metadata-only view that remains contiguous; any
        # non-contiguous input means an upstream layout change must be fixed
        # there instead of hiding an allocation in this hot HC path.
        with torch.inference_mode():
            out = tk_mhc_pre(
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
        # These are already contiguous: ``x`` comes from attention/MoE outputs and
        # ``post``/``comb`` come directly from TileLang pre.
        with torch.inference_mode():
            # In-place reuse: the HC residual stream is overwritten by the next
            # sublayer immediately after this call, so the input buffer is dead
            # at return. Aliasing out=residual saves a per-call empty_like
            # (residual.numel() * 2 bytes, e.g. 7.5 GB at T=128K, hc=4,
            # dim=7168). Kernel-side safety: each block reads residual[pid_n]
            # into shared memory before writing out[pid_n] at the same slot.
            out = tk_mhc_post(
                tk_x,
                tk_residual,
                tk_post,
                tk_comb,
                hc_mult=self.hc_mult,
                out=tk_residual,
            )
        return squeeze_hc_batch(out, wrapped, name="mhc_post output")


class TileLangHCHead(HCHeadBase):
    def _head_impl(self, x: torch.Tensor) -> torch.Tensor:
        # Public HC head input is [T, hc, dim] or [B, S, hc, dim].
        # TileLang uses [B, S, hc, dim]; flat prefill is wrapped as [1, T, ...].
        tk_x, wrapped = wrap_hc_batch(x, 4, name="mhc_head residual")
        # Do not materialize non-contiguous views here; that would add a hidden
        # copy to every head reduce on mis-laid-out inputs.
        with torch.inference_mode():
            if tk_mhc_head_fused_enabled():
                out = tk_mhc_head_fused(
                    tk_x,
                    self.fn,
                    self.scale,
                    self.base,
                    norm_eps=self.norm_eps,
                    pre_eps=self.hc_eps,
                    hc_mult=self.hc_mult,
                )
            else:
                out = tk_mhc_head(
                    tk_x,
                    self.fn,
                    self.scale,
                    self.base,
                    norm_eps=self.norm_eps,
                    pre_eps=self.hc_eps,
                    hc_mult=self.hc_mult,
                )
        return squeeze_hc_batch(out, wrapped, name="mhc_head output")
