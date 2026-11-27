"""TileLang implementation for DeepSeek-V4 Hyper-Connections."""

from __future__ import annotations

from typing import NoReturn

import torch

from rtp_llm.models_py.modules.dsv4.hc.base import HCHeadBase, HCUnitBase
from rtp_llm.models_py.modules.dsv4.hc.mhc_tilelang import (
    tk_mhc_head_fused,
    tk_mhc_post,
    tk_mhc_pre,
)
from rtp_llm.models_py.modules.dsv4.hc.utils import squeeze_hc_batch, wrap_hc_batch

# Snapshot the default import so tests patching ``tilelang_impl.tk_mhc_post``
# still intercept ``TileLangHCUnit._post_impl``.
_DEFAULT_TK_MHC_POST = tk_mhc_post
_CUDA_MHC_POST_OP = None
_CUDA_MHC_POST_PROBE_DONE = False


def _get_cuda_mhc_post_op():
    global _CUDA_MHC_POST_OP, _CUDA_MHC_POST_PROBE_DONE
    if not _CUDA_MHC_POST_PROBE_DONE:
        if (
            getattr(torch.version, "cuda", None) is not None
            and getattr(torch.version, "hip", None) is None
        ):
            try:
                from rtp_llm.ops import rtp_llm_ops

                _CUDA_MHC_POST_OP = getattr(rtp_llm_ops, "dsv4_mhc_post_fwd", None)
            except Exception:
                _CUDA_MHC_POST_OP = None
        else:
            _CUDA_MHC_POST_OP = None
        _CUDA_MHC_POST_PROBE_DONE = True
    return _CUDA_MHC_POST_OP


def _can_use_cuda_mhc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    hc_mult: int,
) -> bool:
    if tk_mhc_post is not _DEFAULT_TK_MHC_POST:
        return False
    if _get_cuda_mhc_post_op() is None:
        return False
    if hc_mult != 4:
        return False
    if (
        x.requires_grad
        or residual.requires_grad
        or post.requires_grad
        or comb.requires_grad
    ):
        return False
    if not (residual.is_cuda and x.is_cuda and post.is_cuda and comb.is_cuda):
        return False
    dev = residual.device
    if x.device != dev or post.device != dev or comb.device != dev:
        return False
    if residual.dtype != torch.bfloat16 or x.dtype != torch.bfloat16:
        return False
    if post.dtype not in (torch.float32, torch.bfloat16) or comb.dtype not in (
        torch.float32,
        torch.bfloat16,
    ):
        return False
    if residual.numel() == 0 or x.numel() == 0:
        return False
    if (int(residual.shape[-1]) * int(residual.shape[-2])) % 256 != 0:
        return False
    return True


def _require_contiguous(t: torch.Tensor, *, name: str) -> torch.Tensor:
    if not t.is_contiguous():
        raise ValueError(
            f"{name} must be contiguous before TileLang mHC call; "
            f"got shape={tuple(t.shape)}, stride={tuple(t.stride())}"
        )
    return t


def _raise_unavailable(
    op: str,
    x: torch.Tensor,
    hc_mult: int,
    hint: str | None = None,
) -> NoReturn:
    hint = hint or (
        "Input must satisfy the TileLang mHC gates: bf16 contiguous CUDA "
        "residual, hc_mult=4, and (hc * dim) % 256 == 0."
    )
    raise RuntimeError(
        f"TileLang mHC {op} unavailable for shape={tuple(x.shape)}, "
        f"stride={tuple(x.stride())}, dtype={x.dtype}, device={x.device}, "
        f"hc_mult={hc_mult}. {hint}"
    )


class TileLangHCUnit(HCUnitBase):
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
            # into shared/register memory before writing out[pid_n] at the same slot.
            if _can_use_cuda_mhc_post(
                tk_x, tk_residual, tk_post, tk_comb, self.hc_mult
            ):
                cuda_op = _get_cuda_mhc_post_op()
                assert cuda_op is not None
                cuda_op(tk_x, tk_residual, tk_post, tk_comb, tk_residual)
                out = tk_residual
            else:
                out = tk_mhc_post(
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


class TileLangHCHead(HCHeadBase):
    def _head_impl(self, x: torch.Tensor) -> torch.Tensor:
        # Public HC head input is [T, hc, dim] or [B, S, hc, dim].
        # TileLang uses [B, S, hc, dim]; flat prefill is wrapped as [1, T, ...].
        tk_x, wrapped = wrap_hc_batch(x, 4, name="mhc_head residual")
        # Do not materialize non-contiguous views here; that would add a hidden
        # copy to every head reduce on mis-laid-out inputs.
        tk_x = _require_contiguous(tk_x, name="mhc_head residual")
        with torch.inference_mode():
            out = tk_mhc_head_fused(
                tk_x,
                self.fn,
                self.scale,
                self.base,
                norm_eps=self.norm_eps,
                pre_eps=self.hc_eps,
                hc_mult=self.hc_mult,
            )
            if out is None:
                _raise_unavailable(
                    "head_fused",
                    x,
                    self.hc_mult,
                    "Fused mHC head is unconditional; fused head must succeed.",
                )
        return squeeze_hc_batch(out, wrapped, name="mhc_head output")
