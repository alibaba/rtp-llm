"""TileLang-backed mHC entry points (pre / post / head).

Thin adapter over the vendored ``rtp_llm.models_py.3rdparty.tile_kernels``
(DeepSeek TileKernels). Each entry point caches a compile-success/failure
verdict — if a kernel fails to JIT once on the local tilelang revision, we
flip a sticky flag and the call site falls back to the PyTorch reference.

Why per-op flags rather than a single ``_TK_AVAILABLE``: empirically
``pre_big_fuse`` has hit ``decouple_type_cast`` ICEs while ``post`` /
``head_compute_mix`` compile fine — losing all three would be a
regression.

Activation gate (``_can_use_tk``):
  * ``DSV4_USE_TK_MHC=0`` env disables the path entirely (parity testing).
  * residual must be bf16 contiguous.
  * ``hc_mult == 4`` (only mult upstream guarantees).
  * ``torch.is_grad_enabled()`` must be False (we serve, not train).
  * residual.numel() > 0.
"""

from __future__ import annotations

import importlib
import logging
import os
from typing import Tuple

import torch

_log = logging.getLogger(__name__)

_TK_PRE_OK: bool | None = None  # None = untried, True/False = sticky verdict
_TK_POST_OK: bool | None = None
_TK_HEAD_OK: bool | None = None

# Lazy-imported callables (populated on first successful import).
_tk_mhc_pre = None
_tk_mhc_post = None
_tk_mhc_head = None


def _import_tk():
    """Import the vendored package, returning (pre, post, head) callables.

    Raises on import failure; caller pins the failure verdict.
    """
    global _tk_mhc_pre, _tk_mhc_post, _tk_mhc_head
    # Reuse the dsv4 tilelang env-prep (z3 path, TVM tmpdir).
    try:
        from rtp_llm.models_py.modules.dsv4 import tilelang_kernels as _dsv4_tl

        if hasattr(_dsv4_tl, "_ensure_tvm_tmpdir_writable"):
            _dsv4_tl._ensure_tvm_tmpdir_writable()
        if hasattr(_dsv4_tl, "_ensure_z3_loadable"):
            _dsv4_tl._ensure_z3_loadable()
    except Exception as e:  # pragma: no cover
        _log.debug("dsv4 tilelang env-prep skipped: %r", e)

    mod = importlib.import_module(
        "rtp_llm.models_py.3rdparty.tile_kernels.modeling.mhc.functional"
    )
    _tk_mhc_pre = mod.mhc_pre
    _tk_mhc_post = mod.mhc_post
    _tk_mhc_head = mod.mhc_head
    return _tk_mhc_pre, _tk_mhc_post, _tk_mhc_head


def _env_disabled() -> bool:
    return os.environ.get("DSV4_USE_TK_MHC", "1") == "0"


def _can_use_tk(residual: torch.Tensor, hc_mult: int) -> bool:
    if _env_disabled():
        return False
    if hc_mult != 4:
        return False
    if residual.dtype != torch.bfloat16:
        return False
    if torch.is_grad_enabled():
        return False
    if residual.numel() == 0:
        return False
    if not residual.is_contiguous():
        return False
    return True


def tk_mhc_pre(
    residual: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    *,
    norm_eps: float,
    pre_eps: float,
    sinkhorn_eps: float,
    sinkhorn_iters: int,
    hc_mult: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """TK ``mhc_pre`` wrapper.

    Returns (layer_input, post_mix [..., hc, 1], comb_mix [..., hc, hc])
    on success, or ``None`` to signal "fall back to REF".
    """
    global _TK_PRE_OK
    if not _can_use_tk(residual, hc_mult):
        return None
    if _TK_PRE_OK is False:
        return None
    try:
        if _tk_mhc_pre is None:
            _import_tk()
        out, (post_mix, comb_mix) = _tk_mhc_pre(
            residual,
            fn,
            scale,
            base,
            norm_weight=None,
            norm_eps=norm_eps,
            mhc_mult=hc_mult,
            post_mult_value=2.0,  # REF uses 2*sigmoid(...) for post
            pre_eps=pre_eps,
            sinkhorn_eps=sinkhorn_eps,
            sinkhorn_repeat=sinkhorn_iters,
        )
        _TK_PRE_OK = True
        return out, post_mix, comb_mix
    except Exception as e:
        if _TK_PRE_OK is None:
            _log.warning("tk mhc_pre disabled after JIT failure: %r", e)
        _TK_PRE_OK = False
        return None


def tk_mhc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_mix: torch.Tensor,
    comb_mix: torch.Tensor,
    hc_mult: int = 4,
) -> torch.Tensor | None:
    """TK ``mhc_post`` wrapper. Returns ``None`` on fallback."""
    global _TK_POST_OK
    if not _can_use_tk(residual, hc_mult):
        return None
    if x.dtype != torch.bfloat16:
        return None
    if _TK_POST_OK is False:
        return None
    try:
        if _tk_mhc_post is None:
            _import_tk()
        out = _tk_mhc_post(x, residual, post_mix, comb_mix)
        _TK_POST_OK = True
        return out
    except Exception as e:
        if _TK_POST_OK is None:
            _log.warning("tk mhc_post disabled after JIT failure: %r", e)
        _TK_POST_OK = False
        return None


def tk_mhc_head(
    residual: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    *,
    norm_eps: float,
    pre_eps: float,
    hc_mult: int = 4,
) -> torch.Tensor | None:
    """TK ``mhc_head`` wrapper. Returns ``None`` on fallback."""
    global _TK_HEAD_OK
    if not _can_use_tk(residual, hc_mult):
        return None
    if _TK_HEAD_OK is False:
        return None
    try:
        if _tk_mhc_head is None:
            _import_tk()
        out = _tk_mhc_head(
            residual,
            fn,
            scale,
            base,
            norm_weight=None,
            norm_eps=norm_eps,
            mhc_mult=hc_mult,
            pre_eps=pre_eps,
        )
        _TK_HEAD_OK = True
        return out
    except Exception as e:
        if _TK_HEAD_OK is None:
            _log.warning("tk mhc_head disabled after JIT failure: %r", e)
        _TK_HEAD_OK = False
        return None
