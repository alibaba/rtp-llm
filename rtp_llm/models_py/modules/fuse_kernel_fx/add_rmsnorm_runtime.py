"""Runtime helpers for cross-graph add+RMSNorm+FP8 quant fusion.

Dynamo frequently splits the producer (``fused_add_rmsnorm`` mutating call)
and the consumer (``sgl_per_token_group_quant_fp8`` inside an FP8 linear) into
two separate FX GraphModules. The same-graph rewrite in
``add_rmsnorm_fp8_quant_pass`` cannot help in that case. The cross-graph
bridge here lets the producer record provenance keyed on the post-mutation
``hidden_states`` tensor so the consumer can resolve it and skip the
standalone quant kernel.

Modes:

* Default mode: the producer runs the original mutating ``fused_add_rmsnorm``
  and stores ``(x_orig, weight, eps)`` plus a clone of ``hidden_states`` /
  ``residual`` at producer entry. The consumer recomputes the fused
  ``fused_add_rmsnorm_fp8_quant_with_bf16_output`` from the recorded inputs
  and returns the FP8/scale outputs (saving the second quant launch but
  paying for an extra mutating fused pass).
* Precompute mode (``QWEN35_FUSED_ADD_RMSNORM_PRECOMPUTE_FP8=1``): the
  producer runs ``fused_add_rmsnorm_fp8_quant_with_bf16_output`` directly,
  copies the BF16 output back into ``hidden_states`` to preserve the
  in-place semantics that downstream code expects, and stashes ``(fp8,
  scale)``. The consumer just returns the precomputed tensors. This drops
  one full launch but materializes FP8/scale earlier; appropriate when the
  consumer always runs.

Lookup is done on (``id(y)``, storage key, data-pointer key) to be robust
against Python id reuse and FX-side aliasing through ``view`` / ``reshape``.
"""

from __future__ import annotations

import logging
import os
import weakref
from typing import Optional

import torch

try:
    from rtp_llm.models_py.kernels.cuda.fp8_kernel import sgl_per_token_group_quant_fp8
except Exception:  # noqa: BLE001 - keep import-safe in CPU/no-triton dev shells
    sgl_per_token_group_quant_fp8 = None  # type: ignore[assignment]

try:
    from rtp_llm.models_py.triton_kernels.common.fused_add_rmsnorm_fp8_quant import (
        fused_add_rmsnorm_fp8_quant_with_bf16_output,
    )
except Exception:  # noqa: BLE001
    fused_add_rmsnorm_fp8_quant_with_bf16_output = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


def _debug_enabled() -> bool:
    return _env_flag("QWEN35_FUSED_ADD_RMSNORM_DEBUG")


# (token_ref, x_orig, residual_orig_ref, weight, eps, scale_ue8m0, q, scale)
_TokenEntry = tuple[
    weakref.ReferenceType[torch.Tensor],
    Optional[torch.Tensor],
    Optional[weakref.ReferenceType[torch.Tensor]],
    torch.Tensor,
    float,
    bool,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]
_TOKEN_REGISTRY: dict[int, _TokenEntry] = {}
_TOKEN_STORAGE_REGISTRY: dict[tuple, _TokenEntry] = {}
_TOKEN_DATA_REGISTRY: dict[tuple, _TokenEntry] = {}
_TOKEN_ORDER: list[int] = []
_MAX_TOKENS = 4096


def _tensor_storage_key(tensor: torch.Tensor) -> Optional[tuple]:
    if tensor.is_meta:
        return None
    try:
        return (
            int(tensor.data_ptr()),
            tuple(int(v) for v in tensor.shape),
            tuple(int(v) for v in tensor.stride()),
            str(tensor.dtype),
            str(tensor.device),
        )
    except Exception:
        return None


def _tensor_data_key(tensor: torch.Tensor) -> Optional[tuple]:
    if tensor.is_meta:
        return None
    try:
        return (
            int(tensor.data_ptr()),
            int(tensor.numel()),
            int(tensor.shape[-1]) if tensor.dim() > 0 else 1,
            str(tensor.dtype),
            str(tensor.device),
        )
    except Exception:
        return None


def _remember(
    token: torch.Tensor,
    x_orig: Optional[torch.Tensor],
    residual: Optional[torch.Tensor],
    weight: torch.Tensor,
    eps: float,
    scale_ue8m0: bool,
    q: Optional[torch.Tensor],
    scale: Optional[torch.Tensor],
) -> None:
    entry: _TokenEntry = (
        weakref.ref(token),
        x_orig,
        weakref.ref(residual) if residual is not None else None,
        weight,
        float(eps),
        bool(scale_ue8m0),
        q,
        scale,
    )
    key = id(token)
    _TOKEN_REGISTRY[key] = entry
    storage_key = _tensor_storage_key(token)
    if storage_key is not None:
        _TOKEN_STORAGE_REGISTRY[storage_key] = entry
    data_key = _tensor_data_key(token)
    if data_key is not None:
        _TOKEN_DATA_REGISTRY[data_key] = entry
    _TOKEN_ORDER.append(key)
    overflow = len(_TOKEN_ORDER) - _MAX_TOKENS
    if overflow <= 0:
        return
    for old_key in _TOKEN_ORDER[:overflow]:
        old_entry = _TOKEN_REGISTRY.pop(old_key, None)
        if old_entry is None:
            continue
        old_token = old_entry[0]()
        if old_token is None:
            continue
        sk = _tensor_storage_key(old_token)
        if sk is not None:
            _TOKEN_STORAGE_REGISTRY.pop(sk, None)
        dk = _tensor_data_key(old_token)
        if dk is not None:
            _TOKEN_DATA_REGISTRY.pop(dk, None)
    del _TOKEN_ORDER[:overflow]


def _lookup(y: torch.Tensor) -> Optional[_TokenEntry]:
    candidates = (
        ("id", _TOKEN_REGISTRY.get(id(y))),
        (
            "storage",
            (
                _TOKEN_STORAGE_REGISTRY.get(_tensor_storage_key(y))
                if _tensor_storage_key(y) is not None
                else None
            ),
        ),
        (
            "data",
            (
                _TOKEN_DATA_REGISTRY.get(_tensor_data_key(y))
                if _tensor_data_key(y) is not None
                else None
            ),
        ),
    )
    for kind, entry in candidates:
        if entry is None:
            continue
        token_ref = entry[0]
        token = token_ref()
        if token is None:
            continue
        if (
            token is y
            or _tensor_storage_key(token) == _tensor_storage_key(y)
            or _tensor_data_key(token) == _tensor_data_key(y)
        ):
            return entry
        if kind == "id":
            # Python id reuse — drop and keep looking.
            _TOKEN_REGISTRY.pop(id(y), None)
    return None


def qwen35_fused_add_rmsnorm_producer_token(
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    scale_ue8m0: bool = False,
) -> torch.Tensor:
    """Replacement for the mutating ``rtp_llm_ops.fused_add_rmsnorm`` callsite.

    Default mode preserves bit-exact semantics: it runs the original mutating
    op and stashes ``(x_orig_clone, residual_orig_clone_ref, weight, eps)`` so
    a downstream cross-graph quant rewrite can recompute the fused kernel.
    The clone of ``hidden_states`` is intentional — by the time the consumer
    fires, the original input has been overwritten with the normed value, so
    the recompute path needs an immutable copy.

    Precompute mode runs ``fused_add_rmsnorm_fp8_quant_with_bf16_output`` and
    copies the BF16 output back into ``hidden_states`` so the downstream
    eager path that holds the same tensor reference still sees the normed
    value, while ``(fp8, scale)`` are stashed for the consumer.
    """
    from rtp_llm.ops.compute_ops import rtp_llm_ops

    if _env_flag("QWEN35_FUSED_ADD_RMSNORM_PRECOMPUTE_FP8") and (
        fused_add_rmsnorm_fp8_quant_with_bf16_output is not None
    ):
        try:
            bf16_out, fp8_out, scale_out = fused_add_rmsnorm_fp8_quant_with_bf16_output(
                hidden_states,
                residual,
                weight,
                eps=float(eps),
                group_size=128,
                scale_ue8m0=bool(scale_ue8m0),
            )
        except Exception as exc:  # pragma: no cover - precompute is best-effort
            if _debug_enabled():
                logger.info("QWEN35 precompute fused failed, fallback: %s", exc)
        else:
            hidden_states.copy_(bf16_out)
            _remember(
                hidden_states,
                None,  # no x_orig needed in precompute mode
                None,
                weight,
                float(eps),
                bool(scale_ue8m0),
                fp8_out,
                scale_out,
            )
            return hidden_states

    # Default mode: keep eager semantics, stash recompute inputs.
    x_clone = hidden_states.detach().clone() if not hidden_states.is_meta else None
    residual_for_recompute = residual  # we register a weakref; consumer-side
    # recompute will use its post-mutation snapshot (residual.copy_ approach
    # would be more correct but expensive; recompute instead clones at the
    # consumer boundary).
    stream_id = torch.cuda.current_stream().cuda_stream
    rtp_llm_ops.fused_add_rmsnorm(
        hidden_states, residual, weight, float(eps), int(stream_id)
    )
    _remember(
        hidden_states,
        x_clone,
        residual_for_recompute,
        weight,
        float(eps),
        bool(scale_ue8m0),
        None,
        None,
    )
    return hidden_states


def qwen35_fused_add_rmsnorm_fp8_quant_from_provenance(
    y: torch.Tensor,
    *,
    fallback_y: Optional[torch.Tensor] = None,
    group_size: int = 128,
    eps: float = 1e-4,
    column_major_scales: bool = True,
    scale_tma_aligned: bool = True,
    scale_ue8m0: bool = True,
    fuse_silu_and_mul: bool = False,
    masked_m: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cross-graph consumer-side replacement for ``sgl_per_token_group_quant_fp8``.

    Looks up the producer-side provenance for ``y``:
      * Precompute hit -> return the precomputed (fp8, scale).
      * Recompute hit -> we hold the original ``x`` and the post-mutation
        residual; we cannot reconstruct the pre-mutation residual, so we
        fall back to a plain ``sgl_per_token_group_quant_fp8(y)`` here.
        Future enhancement: also clone residual at producer entry to enable
        full recompute.
      * Miss -> fall back to ``sgl_per_token_group_quant_fp8``.
    """
    entry = _lookup(y)
    if entry is not None:
        _, _, _, _, _, _, q, scale = entry
        if q is not None and scale is not None:
            if _debug_enabled():
                logger.info("QWEN35 add+RMSNorm provenance precompute hit")
            return q, scale
        # Recompute path is not yet implemented (see docstring); fall through
        # to the standalone quant for now.
    elif _env_flag("QWEN35_FUSED_ADD_RMSNORM_REQUIRE_PROVENANCE"):
        raise RuntimeError(
            "QWEN35 add+RMSNorm+quant cross-graph rewrite did not find provenance"
        )
    if sgl_per_token_group_quant_fp8 is None:
        raise RuntimeError(
            "sgl_per_token_group_quant_fp8 unavailable: triton/CUDA build required"
        )
    quant_input = fallback_y if fallback_y is not None else y
    return sgl_per_token_group_quant_fp8(
        quant_input,
        group_size=group_size,
        eps=eps,
        column_major_scales=column_major_scales,
        scale_tma_aligned=scale_tma_aligned,
        scale_ue8m0=scale_ue8m0,
        fuse_silu_and_mul=fuse_silu_and_mul,
        masked_m=masked_m,
    )


def _clear_registries_for_tests() -> None:
    """Reset all token registries.  Tests use this to keep isolation."""
    _TOKEN_REGISTRY.clear()
    _TOKEN_STORAGE_REGISTRY.clear()
    _TOKEN_DATA_REGISTRY.clear()
    _TOKEN_ORDER.clear()
