"""TileLang-backed mHC entry points (pre / post / head).

Thin adapter over the vendored ``rtp_llm.models_py.3rdparty.tile_kernels``
(DeepSeek TileKernels).

Two distinct failure modes, kept distinct on purpose:

  * **Gate miss** — input doesn't satisfy ``_can_use_tk`` (wrong dtype/device,
    non-contiguous, hc_mult != 4, ...). Wrapper returns ``None`` and the caller
    raises a shape-specific "unavailable" error. This is "not applicable",
    not "broken".
  * **Runtime failure** — import / JIT compile / kernel launch raised. Wrapper
    re-raises a ``RuntimeError`` chained (``from e``) to the original
    exception with shape context attached. Earlier versions swallowed these
    into ``None``, which lost the underlying cause (OOM, ICE, etc.).

Activation gate (``_can_use_tk``):
  * ``DSV4_USE_TK_MHC=0`` env disables the path entirely (parity testing).
  * residual must be bf16 contiguous.
  * residual must be on CUDA.
  * residual's flattened HC hidden size (``hc * dim``) must be divisible by
    256 — the ``pre_norm_fn`` GEMM kernel asserts ``hidden_block=256``
    divisibility. (The ``post`` / ``head_fuse`` / ``pre_apply_mix`` kernels
    handle arbitrary hidden via ``gcd``; this gate is keyed to the strictest.)
  * ``hc_mult == 4`` (only mult upstream guarantees).
  * ``torch.is_grad_enabled()`` must be False (we serve, not train).
  * residual.numel() > 0.
"""

from __future__ import annotations

import importlib
import os
from typing import Any, Dict, NoReturn, Tuple

import torch

# Lazy-imported callables (populated on first successful import).
_tk_mhc_pre = None
_tk_mhc_post = None
_tk_mhc_head_fused = None
_tk_mhc_head = None
_tk_mhc_pre_big_fuse_mod = None
_tk_mhc_post_ops_mod = None


def _import_tk():
    """Import the vendored package, populating the module-level callables.

    Raises on import failure — caller wraps with shape context.
    """
    global _tk_mhc_pre, _tk_mhc_post, _tk_mhc_head_fused, _tk_mhc_head, _tk_mhc_pre_big_fuse_mod, _tk_mhc_post_ops_mod
    # Importing tilelang_kernels triggers its module-level env-prep
    # (libz3 preload + TVM tmpdir setup), which must run before any tilelang
    # JIT import below.
    from rtp_llm.models_py.modules.dsv4 import (  # noqa: F401  # pyright: ignore[reportUnusedImport]
        tilelang_kernels,
    )

    mod = importlib.import_module(
        "rtp_llm.models_py.3rdparty.tile_kernels.modeling.mhc.functional"
    )
    _tk_mhc_pre = mod.mhc_pre
    _tk_mhc_post = mod.mhc_post
    _tk_mhc_head_fused = getattr(mod, "mhc_head_fuse", None)
    _tk_mhc_head = mod.mhc_head
    _tk_mhc_pre_big_fuse_mod = importlib.import_module(
        "rtp_llm.models_py.3rdparty.tile_kernels.modeling.mhc.ops.pre_big_fuse"
    )
    _tk_mhc_post_ops_mod = importlib.import_module(
        "rtp_llm.models_py.3rdparty.tile_kernels.modeling.mhc.ops.post"
    )
    return _tk_mhc_pre, _tk_mhc_post, _tk_mhc_head_fused, _tk_mhc_head


def _env_disabled() -> bool:
    return os.environ.get("DSV4_USE_TK_MHC", "1") == "0"


def _prefill_mhc_pre_out_unchecked_enabled() -> bool:
    return os.environ.get("DSV4_MHC_PRE_OUT_UNCHECKED", "0") == "1"


def tk_mhc_head_fused_enabled() -> bool:
    return os.environ.get("DSV4_MHC_HEAD_FUSED", "1") != "0"


def _head_fuse_disabled() -> bool:
    return not tk_mhc_head_fused_enabled()


def _shape_ctx(t: torch.Tensor, *, hc_mult: int, extra: str = "") -> str:
    base = (
        f"shape={tuple(t.shape)}, stride={tuple(t.stride())}, "
        f"dtype={t.dtype}, device={t.device}, hc_mult={hc_mult}"
    )
    return f"{base}, {extra}" if extra else base


def _raise_head_fused_unavailable(
    residual: torch.Tensor,
    hc_mult: int,
    reason: str,
    cause: BaseException | None = None,
) -> NoReturn:
    err = RuntimeError(
        f"TileLang fused mHC head unavailable: {reason}; "
        f"{_shape_ctx(residual, hc_mult=hc_mult)}. "
        "Set DSV4_MHC_HEAD_FUSED=0 to use the older TileLang head composition."
    )
    if cause is not None:
        raise err from cause
    raise err


def _can_use_tk(residual: torch.Tensor, hc_mult: int) -> bool:
    if _env_disabled():
        return False
    if hc_mult != 4:
        return False
    if residual.dtype != torch.bfloat16:
        return False
    if not residual.is_cuda:
        return False
    if torch.is_grad_enabled():
        return False
    if residual.numel() == 0:
        return False
    if not residual.is_contiguous():
        return False
    if (int(residual.shape[-1]) * int(residual.shape[-2])) % 256 != 0:
        return False
    return True


def tk_mhc_head_fused(
    residual: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    *,
    norm_eps: float,
    pre_eps: float,
    hc_mult: int = 4,
) -> torch.Tensor | None:
    """Fused TK ``mhc_head`` wrapper.

    Returns ``None`` only when the fused head is explicitly disabled. When the
    fused path is enabled, incompatibility or JIT/import failure is fatal so the
    caller cannot silently fall back to the older TileLang head composition.
    Runtime failures (import / JIT / kernel run) propagate as ``RuntimeError``
    chained via ``from e`` to the original exception.
    """
    if _head_fuse_disabled():
        return None
    if not _can_use_tk(residual, hc_mult):
        _raise_head_fused_unavailable(
            residual,
            hc_mult,
            "input does not satisfy TileLang fused mHC head gates",
        )
    if _tk_mhc_head_fused is None:
        try:
            _import_tk()
        except Exception as e:
            _raise_head_fused_unavailable(residual, hc_mult, "import failure", cause=e)
    if _tk_mhc_head_fused is None:
        _raise_head_fused_unavailable(
            residual, hc_mult, "vendored TileKernels has no mhc_head_fuse"
        )
    fused = _tk_mhc_head_fused
    assert fused is not None
    try:
        out = fused(
            residual,
            fn,
            scale.reshape(1),
            base,
            rms_eps=norm_eps,
            mhc_pre_eps=pre_eps,
        )
    except Exception as e:
        _raise_head_fused_unavailable(
            residual, hc_mult, "JIT / kernel failure", cause=e
        )
    return out


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
    on success. Returns ``None`` only when the input fails ``_can_use_tk``
    gates (caller raises a shape-specific "unavailable"). Runtime failures
    (import / JIT / kernel run) propagate as ``RuntimeError`` chained via
    ``from e`` to the original exception.
    """
    if not _can_use_tk(residual, hc_mult):
        return None
    try:
        if _tk_mhc_pre is None:
            _import_tk()
        pre_fn = _tk_mhc_pre
        assert pre_fn is not None
        out, (post_mix, comb_mix) = pre_fn(
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
        return out, post_mix, comb_mix
    except Exception as e:
        raise RuntimeError(
            "TileLang mhc_pre failed: " + _shape_ctx(residual, hc_mult=hc_mult)
        ) from e


def tk_mhc_pre_out(
    residual: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    *,
    norm_eps: float,
    pre_eps: float,
    sinkhorn_eps: float,
    sinkhorn_iters: int,
    workspace: Dict[str, Any],
    hc_mult: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Fast ``mhc_pre`` wrapper that reuses exact-shape output buffers."""
    if not _can_use_tk(residual, hc_mult):
        return None
    try:
        if _tk_mhc_pre_big_fuse_mod is None:
            _import_tk()
        mod = _tk_mhc_pre_big_fuse_mod
        if mod is None:
            raise RuntimeError("TileLang mHC pre_big_fuse module was not imported")

        mhc_mult = int(residual.shape[-2])
        hidden_size = int(residual.shape[-1])
        mhc_mult2 = mhc_mult * mhc_mult
        mhc_mult3 = mhc_mult * 2 + mhc_mult2
        mhc_hidden_size = mhc_mult * hidden_size
        residual_flat = residual.view(-1, mhc_mult, hidden_size)
        num_tokens = int(residual_flat.shape[0])

        backend = mod._requested_backend()
        use_unchecked = _prefill_mhc_pre_out_unchecked_enabled()
        n_splits = (
            mod._compute_num_split(64, mhc_hidden_size, mod._ceil_div(num_tokens, 64))
            if backend in ("deepgemm", "tilelang_splitk")
            else 1
        )
        key = (
            str(residual.device),
            tuple(residual.shape),
            backend,
            n_splits,
            hidden_size,
            mhc_mult,
            use_unchecked,
            float(norm_eps),
            float(pre_eps),
            float(sinkhorn_eps),
            int(sinkhorn_iters),
        )
        if workspace.get("key") != key:
            device = residual.device
            new_workspace: Dict[str, Any] = {"key": key}
            new_workspace["backend"] = backend
            new_workspace["n_splits"] = n_splits
            new_workspace["post_mix"] = torch.empty(
                num_tokens, mhc_mult, dtype=torch.float32, device=device
            )
            new_workspace["comb_mix"] = torch.empty(
                num_tokens, mhc_mult2, dtype=torch.float32, device=device
            )
            new_workspace["layer_input"] = torch.empty(
                num_tokens, hidden_size, dtype=torch.bfloat16, device=device
            )
            new_workspace["gemm_out_mul"] = torch.empty(
                n_splits, num_tokens, mhc_mult3, dtype=torch.float32, device=device
            )
            new_workspace["gemm_out_sqrsum"] = torch.empty(
                n_splits, num_tokens, dtype=torch.float32, device=device
            )
            if use_unchecked:
                new_workspace["pre_big_fuse_kernel"] = mod._mhc_pre_big_fuse(
                    hidden_size,
                    norm_eps,
                    pre_eps,
                    sinkhorn_eps,
                    2.0,
                    sinkhorn_iters,
                    n_splits=n_splits,
                    mhc_mult=mhc_mult,
                )
            workspace.clear()
            workspace.update(new_workspace)

        if use_unchecked:
            residual_flat = residual.view(-1, mhc_mult, hidden_size)
            if backend == "deepgemm":
                mod._run_deepgemm_splitk_gemm(
                    residual_flat.view(num_tokens, mhc_hidden_size),
                    fn,
                    workspace["gemm_out_mul"],
                    workspace["gemm_out_sqrsum"],
                    n_splits,
                )
            elif backend == "tilelang_single":
                actual_splits = mod._run_tilelang_single_gemm(
                    residual_flat,
                    fn,
                    workspace["gemm_out_mul"],
                    workspace["gemm_out_sqrsum"],
                    mhc_mult3,
                    mhc_hidden_size,
                )
                if actual_splits != n_splits:
                    raise RuntimeError(
                        "tilelang_single returned n_splits="
                        f"{actual_splits}, expected {n_splits}"
                    )
            elif backend == "tilelang_splitk":
                raise RuntimeError(
                    "DSV4_MHC_PRE_GEMM_BACKEND=tilelang_splitk is not wired in "
                    "this RTP TileKernels snapshot; use deepgemm or tilelang_single."
                )
            else:
                raise ValueError(
                    "Unsupported DSV4_MHC_PRE_GEMM_BACKEND="
                    f"{backend!r}; expected deepgemm, tilelang_splitk, or tilelang_single."
                )

            workspace["pre_big_fuse_kernel"](
                workspace["gemm_out_mul"],
                workspace["gemm_out_sqrsum"],
                scale,
                base,
                residual_flat,
                workspace["post_mix"],
                workspace["comb_mix"],
                workspace["layer_input"],
            )
            outer_shape = residual.shape[:-2]
            return (
                workspace["layer_input"].view(*outer_shape, hidden_size),
                workspace["post_mix"].view(*outer_shape, mhc_mult, 1),
                workspace["comb_mix"].view(*outer_shape, mhc_mult, mhc_mult),
            )

        post_mix, comb_mix, layer_input = mod.mhc_pre_big_fuse_out(
            residual,
            fn,
            scale,
            base,
            rms_eps=norm_eps,
            mhc_pre_eps=pre_eps,
            mhc_sinkhorn_eps=sinkhorn_eps,
            mhc_post_mult_value=2.0,
            sinkhorn_repeat=sinkhorn_iters,
            post_mix=workspace["post_mix"],
            comb_mix=workspace["comb_mix"],
            layer_input=workspace["layer_input"],
            gemm_out_mul=workspace["gemm_out_mul"],
            gemm_out_sqrsum=workspace["gemm_out_sqrsum"],
        )
        return layer_input, post_mix, comb_mix
    except Exception as e:
        raise RuntimeError(
            "TileLang mhc_pre_out failed: " + _shape_ctx(residual, hc_mult=hc_mult)
        ) from e


def tk_mhc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_mix: torch.Tensor,
    comb_mix: torch.Tensor,
    hc_mult: int = 4,
    out: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """TK ``mhc_post`` wrapper.

    Returns ``None`` only when ``residual`` fails ``_can_use_tk`` gates (caller
    raises a shape-specific "unavailable"). A non-bfloat16 sublayer output ``x``
    raises ``RuntimeError`` directly: that is an upstream dtype bug, not a
    TileLang availability issue, and must not be disguised as one. Runtime
    failures (import / JIT / kernel run) propagate as ``RuntimeError`` chained
    via ``from e`` to the original exception.

    Pass ``out=residual`` to write in place and skip the kernel's
    ``torch.empty_like(residual)`` allocation (7.5 GB at T=128K, hc=4, dim=7168).
    """
    if not _can_use_tk(residual, hc_mult):
        return None
    if x.dtype != torch.bfloat16:
        raise RuntimeError(
            "TileLang mhc_post requires a bfloat16 sublayer output x; got "
            f"x.dtype={x.dtype}, x.shape={tuple(x.shape)}. This is an upstream "
            "dtype bug — fix the producer rather than treating it as a "
            f"TileLang fallback. {_shape_ctx(residual, hc_mult=hc_mult)}"
        )
    try:
        if _tk_mhc_post is None:
            _import_tk()
        post_fn = _tk_mhc_post
        assert post_fn is not None
        return post_fn(x, residual, post_mix, comb_mix, out=out)
    except Exception as e:
        raise RuntimeError(
            "TileLang mhc_post failed: "
            + _shape_ctx(residual, hc_mult=hc_mult, extra=f"x.shape={tuple(x.shape)}")
        ) from e


def tk_mhc_post_fwd_out(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_mix: torch.Tensor,
    comb_mix: torch.Tensor,
    hc_mult: int = 4,
    out: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Inference-only ``mhc_post`` wrapper that bypasses autograd.Function."""
    if not _can_use_tk(residual, hc_mult):
        return None
    if x.dtype != torch.bfloat16:
        raise RuntimeError(
            "TileLang mhc_post_fwd_out requires a bfloat16 sublayer output x; got "
            f"x.dtype={x.dtype}, x.shape={tuple(x.shape)}. {_shape_ctx(residual, hc_mult=hc_mult)}"
        )
    try:
        if _tk_mhc_post_ops_mod is None:
            _import_tk()
        mod = _tk_mhc_post_ops_mod
        if mod is None:
            raise RuntimeError("TileLang mHC post ops module was not imported")
        return mod.mhc_post_fwd(x, residual, post_mix, comb_mix, out=out)
    except Exception as e:
        raise RuntimeError(
            "TileLang mhc_post_fwd_out failed: "
            + _shape_ctx(residual, hc_mult=hc_mult, extra=f"x.shape={tuple(x.shape)}")
        ) from e


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
    """TK ``mhc_head`` wrapper.

    Returns ``None`` only when the input fails ``_can_use_tk`` gates (caller
    raises a shape-specific "unavailable"). Runtime failures (import / JIT /
    kernel run) propagate as ``RuntimeError`` chained via ``from e`` to the
    original exception.
    """
    if not _can_use_tk(residual, hc_mult):
        return None
    try:
        if _tk_mhc_head is None:
            _import_tk()
        head_fn = _tk_mhc_head
        assert head_fn is not None
        return head_fn(
            residual,
            fn,
            scale,
            base,
            norm_weight=None,
            norm_eps=norm_eps,
            mhc_mult=hc_mult,
            pre_eps=pre_eps,
        )
    except Exception as e:
        raise RuntimeError(
            "TileLang mhc_head failed: " + _shape_ctx(residual, hc_mult=hc_mult)
        ) from e
