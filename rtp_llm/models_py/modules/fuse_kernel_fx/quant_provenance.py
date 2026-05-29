"""Shared provenance registry for cross-graph activation+FP8 quant fusion.

When Dynamo splits an activation op (sigmoid*mul, silu_and_mul, rmsnorm_gated)
and the subsequent FP8 quant (inside CudaFp8GEMMLinear) into different FX
subgraphs, the same-graph FX pass cannot match.  The producer token runs the
activation op AND the quant eagerly, stores (fp8, scale) keyed on the bf16
output tensor, and returns the bf16 for downstream code.  The consumer
(CudaFp8GEMMLinear.forward) looks up the registry before computing its own
quant, using the stored pair directly when found.

Lookup is done on (id, storage key, data-pointer key) to be robust against
Python id reuse and FX-side aliasing through view/reshape.
"""

from __future__ import annotations

import logging
import weakref
from typing import Optional

import torch

logger = logging.getLogger(__name__)

_QuantEntry = tuple[
    weakref.ReferenceType[torch.Tensor],  # weakref to key tensor
    torch.Tensor,  # fp8
    torch.Tensor,  # scale
]

_QUANT_REGISTRY: dict[int, _QuantEntry] = {}
_QUANT_STORAGE_REGISTRY: dict[tuple, _QuantEntry] = {}
_QUANT_DATA_REGISTRY: dict[tuple, _QuantEntry] = {}
_QUANT_ORDER: list[int] = []
_MAX_ENTRIES = 4096


def _tensor_storage_key(t: torch.Tensor) -> Optional[tuple]:
    if t.is_meta:
        return None
    try:
        return (
            int(t.data_ptr()),
            tuple(int(v) for v in t.shape),
            tuple(int(v) for v in t.stride()),
            str(t.dtype),
            str(t.device),
        )
    except Exception:
        return None


def _tensor_data_key(t: torch.Tensor) -> Optional[tuple]:
    if t.is_meta:
        return None
    try:
        return (
            int(t.data_ptr()),
            int(t.numel()),
            int(t.shape[-1]) if t.dim() > 0 else 1,
            str(t.dtype),
            str(t.device),
        )
    except Exception:
        return None


@torch.compiler.disable
def remember_quant(
    key_tensor: torch.Tensor,
    fp8: torch.Tensor,
    scale: torch.Tensor,
) -> None:
    """Store (fp8, scale) keyed on ``key_tensor`` for downstream consumer lookup."""
    entry: _QuantEntry = (weakref.ref(key_tensor), fp8, scale)
    key = id(key_tensor)
    _QUANT_REGISTRY[key] = entry
    sk = _tensor_storage_key(key_tensor)
    if sk is not None:
        _QUANT_STORAGE_REGISTRY[sk] = entry
    dk = _tensor_data_key(key_tensor)
    if dk is not None:
        _QUANT_DATA_REGISTRY[dk] = entry
    _QUANT_ORDER.append(key)
    overflow = len(_QUANT_ORDER) - _MAX_ENTRIES
    if overflow <= 0:
        return
    for old_key in _QUANT_ORDER[:overflow]:
        old_entry = _QUANT_REGISTRY.pop(old_key, None)
        if old_entry is None:
            continue
        old_ref = old_entry[0]()
        if old_ref is None:
            continue
        osk = _tensor_storage_key(old_ref)
        if osk is not None:
            _QUANT_STORAGE_REGISTRY.pop(osk, None)
        odk = _tensor_data_key(old_ref)
        if odk is not None:
            _QUANT_DATA_REGISTRY.pop(odk, None)
    del _QUANT_ORDER[:overflow]


@torch.compiler.disable
def lookup_quant(
    y: torch.Tensor,
) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    """Look up stored (fp8, scale) for tensor ``y``.  Returns None on miss."""
    candidates = (
        ("id", _QUANT_REGISTRY.get(id(y))),
        (
            "storage",
            (
                _QUANT_STORAGE_REGISTRY.get(_tensor_storage_key(y))
                if _tensor_storage_key(y) is not None
                else None
            ),
        ),
        (
            "data",
            (
                _QUANT_DATA_REGISTRY.get(_tensor_data_key(y))
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
            return entry[1], entry[2]
        if kind == "id":
            _QUANT_REGISTRY.pop(id(y), None)
    return None


def _clear_for_tests() -> None:
    _QUANT_REGISTRY.clear()
    _QUANT_STORAGE_REGISTRY.clear()
    _QUANT_DATA_REGISTRY.clear()
    _QUANT_ORDER.clear()
    _STRIDED_REGISTRY.clear()
    _STRIDED_STORAGE_REGISTRY.clear()
    _STRIDED_DATA_REGISTRY.clear()
    _STRIDED_ORDER.clear()
    _SLOT_CACHE.clear()


# ----------------------------------------------------------------------
# Strided-source provenance: lets a producer subgraph that calls
# ``contig = strided_view.contiguous()`` remember the original strided
# tensor, so a downstream rmsnorm-only subgraph can recover the strided
# input and replace the rmsnorm with ``fused_strided_rmsnorm``.
#
# Same triple-keyed lookup pattern as the quant registry to be robust
# against tensor identity reuse and aliasing.
# ----------------------------------------------------------------------

_StridedEntry = tuple[
    weakref.ReferenceType[torch.Tensor],  # weakref to the contig key tensor
    torch.Tensor,  # original strided source
]

_STRIDED_REGISTRY: dict[int, _StridedEntry] = {}
_STRIDED_STORAGE_REGISTRY: dict[tuple, _StridedEntry] = {}
_STRIDED_DATA_REGISTRY: dict[tuple, _StridedEntry] = {}
_STRIDED_ORDER: list[int] = []
_MAX_STRIDED_ENTRIES = 4096


@torch.compiler.disable
def remember_strided(contig: torch.Tensor, strided_source: torch.Tensor) -> None:
    """Store ``strided_source`` keyed on ``contig`` for downstream consumer lookup.

    Same triple-keyed scheme as ``remember_quant`` — robust against id() reuse.
    """
    entry: _StridedEntry = (weakref.ref(contig), strided_source)
    key = id(contig)
    _STRIDED_REGISTRY[key] = entry
    sk = _tensor_storage_key(contig)
    if sk is not None:
        _STRIDED_STORAGE_REGISTRY[sk] = entry
    dk = _tensor_data_key(contig)
    if dk is not None:
        _STRIDED_DATA_REGISTRY[dk] = entry
    _STRIDED_ORDER.append(key)
    overflow = len(_STRIDED_ORDER) - _MAX_STRIDED_ENTRIES
    if overflow <= 0:
        return
    for old_key in _STRIDED_ORDER[:overflow]:
        old_entry = _STRIDED_REGISTRY.pop(old_key, None)
        if old_entry is None:
            continue
        old_ref = old_entry[0]()
        if old_ref is None:
            continue
        osk = _tensor_storage_key(old_ref)
        if osk is not None:
            _STRIDED_STORAGE_REGISTRY.pop(osk, None)
        odk = _tensor_data_key(old_ref)
        if odk is not None:
            _STRIDED_DATA_REGISTRY.pop(odk, None)
    del _STRIDED_ORDER[:overflow]


@torch.compiler.disable
def lookup_strided(contig: torch.Tensor) -> Optional[torch.Tensor]:
    """Return the strided source tensor that produced ``contig``, or None."""
    candidates = (
        ("id", _STRIDED_REGISTRY.get(id(contig))),
        (
            "storage",
            (
                _STRIDED_STORAGE_REGISTRY.get(_tensor_storage_key(contig))
                if _tensor_storage_key(contig) is not None
                else None
            ),
        ),
        (
            "data",
            (
                _STRIDED_DATA_REGISTRY.get(_tensor_data_key(contig))
                if _tensor_data_key(contig) is not None
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
            token is contig
            or _tensor_storage_key(token) == _tensor_storage_key(contig)
            or _tensor_data_key(token) == _tensor_data_key(contig)
        ):
            return entry[1]
        if kind == "id":
            _STRIDED_REGISTRY.pop(id(contig), None)
    return None


# ----------------------------------------------------------------------
# Slot-based FIFO cache for cross-graph FP8 quant fusion.
#
# Dynamo allocates fresh tensor storage between compiled subgraphs, making
# identity/data_ptr-based lookups impossible. This FIFO bypasses that by
# using shape+dtype matching with execution-order pairing: producers push
# in execution order, consumers pop the first shape-matching entry.
# ----------------------------------------------------------------------

_SLOT_CACHE: list[tuple[tuple[int, ...], str, torch.Tensor, torch.Tensor]] = []
_SLOT_CACHE_MAX = 128


@torch.compiler.disable
def slot_push_quant(fp8: torch.Tensor, scale: torch.Tensor, *, tag: str = "") -> None:
    """Push (fp8, scale) to the FIFO slot cache for cross-graph consumers."""
    if len(_SLOT_CACHE) >= _SLOT_CACHE_MAX:
        _SLOT_CACHE.pop(0)
    _SLOT_CACHE.append((tuple(fp8.shape), tag, fp8, scale))


@torch.compiler.disable
def slot_pop_quant(
    shape: tuple[int, ...],
    scale_ue8m0: bool = True,
    *,
    tag: str = "",
) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    """Pop the first (fp8, scale) entry matching shape, scale dtype, and tag.

    Returns None if no matching entry is found.
    """
    expected_dtype = torch.int32 if scale_ue8m0 else torch.float32
    for i, (s, entry_tag, fp8, scale) in enumerate(_SLOT_CACHE):
        if s == shape and scale.dtype == expected_dtype and entry_tag == tag:
            _SLOT_CACHE.pop(i)
            return fp8, scale
    return None
