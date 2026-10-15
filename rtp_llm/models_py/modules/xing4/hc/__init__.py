"""Xing4.0 hyper-connection fused kernels and the unit-level factory (K5).

The per-kernel modules stay the math authorities for their segments:

* ``norm_triton``     (K1) row RMSNorm over the flattened hc*H row;
* ``mix_triton``      (K2) sigmoid gates + clamp/exp + sinkhorn comb;
* ``collapse_triton`` (K3) y = sum_i pre[..., i] * x[..., i, H];
* ``post_triton``     (K4) out = post(x) x + comb @ residual;
* ``fn_gemv_triton``  (K1+GEMM) fused row RMSNorm + hc_fn skinny GEMV
  producing the mixes directly from the stream (rows-gated by
  ``XING4_HC_FN_GEMV_MAX_TOKENS``; see fn_gemv_triton.py).

This package adds the unit-level factory that ``Xing4HCUnit``
(rtp_llm/models_py/model_desc/xing4_0_model.py) consults on every
``_pre_impl``/``_post_impl`` call:

* ``XING4_HC_IMPL`` environment variable (resolved per call, so a process
  can hot-switch backends and tests can drive both paths through one unit):

  - ``"triton"`` (default) — run the fused K1 + F.linear (GEMM, untouched) +
    K2 + K3 / K4 composition when the stream is a CUDA bf16 tensor and the
    kernel modules import; otherwise automatically fall back to the eager
    torch reference path, logging exactly one line per process.
  - ``"torch"`` — explicit eager reference path (no log, bit-stable fallback).
  - any other value raises ``ValueError`` (fail fast, same contract as
    ``XING4_HC_NORM_IMPL`` in norm_triton.py).

* The kernel modules import lazily, so importing this package (and the model
  file that imports it) never requires triton or CUDA; import failures
  surface through the auto-fallback above instead of an import error.

* Sub-kernel escape hatches are unaffected: the fused unit path calls the
  direct entries (``hc_row_rms_norm_triton`` and friends), so
  ``XING4_HC_NORM_IMPL`` only matters when norm_triton.py is used standalone.
"""

from __future__ import annotations

import logging
import os
from importlib import import_module
from typing import Any, Callable, NamedTuple, Optional

import torch

logger = logging.getLogger(__name__)

IMPL_ENV = "XING4_HC_IMPL"
TRITON_IMPL = "triton"
TORCH_IMPL = "torch"


class Xing4HcTritonEntries(NamedTuple):
    """The fused entry points consulted by ``Xing4HCUnit``."""

    row_rms_norm: Callable[..., Any]  # K1: hc_row_rms_norm_triton
    mix: Callable[..., Any]  # K2: hc_mix_triton
    collapse: Callable[..., Any]  # K3: hc_collapse
    post: Callable[..., Any]  # K4: xing4_hc_post
    # K1+GEMM fusion (round-2): one kernel produces the hc_fn mixes directly
    # from the stream, and the rows-gate resolver that dispatches it.
    norm_fn_gemv: Callable[..., Any]
    fn_gemv_max_tokens: Callable[[], int]


# Lazy re-exports kept for the package surface the K1 lane shipped (the
# kernel names resolve on first access so the import stays triton-free).
_LAZY_EXPORTS = {
    "hc_row_rms_norm": ("norm_triton", "hc_row_rms_norm"),
    "hc_row_rms_norm_torch": ("norm_triton", "hc_row_rms_norm_torch"),
    "hc_row_rms_norm_triton": ("norm_triton", "hc_row_rms_norm_triton"),
    "norm_triton": ("norm_triton", "norm_triton"),
    "hc_mix_triton": ("mix_triton", "hc_mix_triton"),
    "mix_triton": ("mix_triton", "mix_triton"),
    "hc_collapse": ("collapse_triton", "hc_collapse"),
    "collapse_triton": ("collapse_triton", "collapse_triton"),
    "hc_post_triton": ("post_triton", "hc_post_triton"),
    "xing4_hc_post": ("post_triton", "xing4_hc_post"),
    "post_triton": ("post_triton", "post_triton"),
    "hc_fn_gemv_triton": ("fn_gemv_triton", "hc_fn_gemv_triton"),
    "hc_fn_gemv_max_tokens": ("fn_gemv_triton", "hc_fn_gemv_max_tokens"),
    "fn_gemv_triton": ("fn_gemv_triton", "fn_gemv_triton"),
}

_triton_entries: Optional[Xing4HcTritonEntries] = None
_triton_import_error: Optional[BaseException] = None
_fallback_logged = False


def resolve_xing4_hc_impl() -> str:
    """Resolve ``XING4_HC_IMPL`` to ``"triton"`` (default) or ``"torch"``.

    Raises ``ValueError`` on any other value so a typo'd env var fails fast
    instead of silently selecting a backend.
    """
    raw = os.environ.get(IMPL_ENV)
    if raw is None or not raw.strip():
        return TRITON_IMPL
    impl = raw.strip().lower()
    if impl not in (TRITON_IMPL, TORCH_IMPL):
        raise ValueError(
            f"invalid {IMPL_ENV}={raw!r}; expected 'triton' or 'torch'"
        )
    return impl


def xing4_hc_triton_entries() -> Xing4HcTritonEntries:
    """Import and cache the fused kernel entry points (K1..K4 + GEMV).

    Importing is deferred to the first fused dispatch, so importing this
    package (and the model file that consults it) never requires triton or a
    CUDA driver. A failed import is cached and re-raised; the dispatcher
    converts it into the documented torch fallback.
    """
    global _triton_entries, _triton_import_error
    if _triton_import_error is not None:
        raise _triton_import_error
    if _triton_entries is None:
        try:
            from rtp_llm.models_py.modules.xing4.hc.collapse_triton import (
                hc_collapse,
            )
            from rtp_llm.models_py.modules.xing4.hc.fn_gemv_triton import (
                hc_fn_gemv_max_tokens,
                hc_fn_gemv_triton,
            )
            from rtp_llm.models_py.modules.xing4.hc.mix_triton import (
                hc_mix_triton,
            )
            from rtp_llm.models_py.modules.xing4.hc.norm_triton import (
                hc_row_rms_norm_triton,
            )
            from rtp_llm.models_py.modules.xing4.hc.post_triton import (
                xing4_hc_post,
            )

            _triton_entries = Xing4HcTritonEntries(
                row_rms_norm=hc_row_rms_norm_triton,
                mix=hc_mix_triton,
                collapse=hc_collapse,
                post=xing4_hc_post,
                norm_fn_gemv=hc_fn_gemv_triton,
                fn_gemv_max_tokens=hc_fn_gemv_max_tokens,
            )
        except BaseException as exc:  # noqa: BLE001 - cached and re-raised
            _triton_import_error = exc
            raise
    return _triton_entries


def xing4_hc_use_fused(x: torch.Tensor) -> bool:
    """Return whether the fused triton path should run for ``x``.

    The fused composition currently requires a CUDA bf16 stream (K4 is
    bf16-only) and an importable kernel set. When ``XING4_HC_IMPL`` selects
    triton (explicitly or by default) but the stream is not eligible, the
    caller automatically runs the eager torch reference path instead and
    exactly one warning line is logged per process; the explicit
    ``XING4_HC_IMPL=torch`` selection never logs. An illegal env value
    raises ``ValueError`` (see ``resolve_xing4_hc_impl``).
    """
    global _fallback_logged
    impl = resolve_xing4_hc_impl()
    if impl == TORCH_IMPL:
        return False
    reason: Optional[str] = None
    if not x.is_cuda:
        reason = "input tensor is not on CUDA"
    elif x.dtype != torch.bfloat16:
        reason = f"unsupported stream dtype {x.dtype} (fused path is bf16-only)"
    else:
        try:
            xing4_hc_triton_entries()
        except Exception as exc:
            reason = f"triton kernel import failed ({exc})"
    if reason is None:
        return True
    if not _fallback_logged:
        _fallback_logged = True
        logger.warning(
            "Xing4HCUnit fused path unavailable, falling back to the torch "
            "reference path: %s",
            reason,
        )
    return False


def _reset_fallback_log_for_test() -> None:
    """Re-arm the once-per-process auto-fallback log (test hook only)."""
    global _fallback_logged
    _fallback_logged = False


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        module_name, attr = _LAZY_EXPORTS[name]
        module = import_module(f".{module_name}", __name__)
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "IMPL_ENV",
    "TORCH_IMPL",
    "TRITON_IMPL",
    "Xing4HcTritonEntries",
    "resolve_xing4_hc_impl",
    "xing4_hc_triton_entries",
    "xing4_hc_use_fused",
]
