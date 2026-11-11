"""Pool-paged sparse-prefill value/key sources for pre-SM100 devices.

The frozen branch materializes, per kv_source layer and forward:

* ``g`` — the dequantized compressed global rows ``[span/ratio, 512]`` bf16
  (604MB at a 1M-token span), concatenated with the SWA rows by
  ``combine_kv`` into the dense value source for ``flash_mla_sparse_fwd``;
* ``k`` — the indexer keys ``[span/ratio, 128]`` fp32 (268MB at 1M), the
  scoring source for ``_select_indices``.

Both scale with the compressed span, not with the engine chunk, so long
contexts on tight-HBM devices die long before the full context (load52probe:
rank0 OOM at ~95K span; at 1M every rank would need ~870MB of dense buffers).

The same bytes already live in the GPU-resident csa/hca/indexer FP4 pools.
With ``DSV41_SPARSE_PAGED_VALUES=1`` (pre-SM100, single-request prefill,
no CP sharding, eager only) ``publish_paged_globals`` replaces the dense
publish with lazy markers:

* ``PagedGlobalValues`` gathers requested compressed rows through the same
  slot mapping and FP4 gather/dequant kernels the dense path uses, so the
  sparse-attention reference sees bitwise-identical values per selected row;
* ``LazyIndexerKeys`` precomputes only the slot table (2MB at 1M) and
  dequantizes key rows per scorer chunk;
* ``PagedSparseKVSource`` is the callable value source consumed by
  ``flash_mla._sparse_fwd_reference``: index < n_global reads the pool,
  index >= n_global reads the (window-bounded, dense) SWA rows, -1 rows are
  zero-filled and masked by ``topk_length`` exactly as before.

Peak transient becomes [ref_rows, K, 512] fp32 (~24MB) regardless of span.
Any ineligible shape falls back to the untouched dense path.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache

import torch

from ._v41_fp4_triton import (
    FP4_GLOBAL_HEAD_DIM,
    dequantize_indexer_k_fp4,
    dequantize_k_cache_bytes_fp4,
    gather_k_cache_bytes_fp4,
)

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def paged_sparse_values_enabled() -> bool:
    return os.environ.get("DSV41_SPARSE_PAGED_VALUES", "0") == "1"


def paged_sparse_device_eligible(device) -> bool:
    if not torch.cuda.is_available():
        return False
    index = device.index if device.index is not None else torch.cuda.current_device()
    major, _ = torch.cuda.get_device_capability(index)
    return major < 10


class PagedGlobalValues:
    """Lazy replacement for the dense dequantized globals ``g``."""

    __slots__ = ("attn", "region", "ratio", "count", "device")

    def __init__(self, attn, region, ratio: int, count: int, device):
        self.attn = attn
        self.region = region
        self.ratio = int(ratio)
        self.count = int(count)
        self.device = device

    def __len__(self) -> int:
        return self.count

    @property
    def shape(self):
        return (self.count, FP4_GLOBAL_HEAD_DIM)

    def gather(self, idx: torch.Tensor) -> torch.Tensor:
        """Dequantize the compressed rows named by ``idx`` ([N] long, >=0)."""
        pool = self.attn._source_pool(self.region)
        positions = (idx + 1) * self.ratio - 1
        req_ids = torch.zeros_like(positions)
        slots = self.attn._slots(self.region, positions, req_ids)
        raw = gather_k_cache_bytes_fp4(pool, slots)
        return dequantize_k_cache_bytes_fp4(raw)


class LazyIndexerKeys:
    """Lazy replacement for the dense indexer keys ``k`` ([count, 128] fp32).

    Only the slot table is materialized (int64 [count]); rows are gathered
    and dequantized per scorer key-chunk with the same kernel the dense
    producer used, so values are bitwise identical.
    """

    __slots__ = ("pool", "slots", "count")

    def __init__(self, pool, slots: torch.Tensor, count: int):
        self.pool = pool
        self.slots = slots
        self.count = int(count)

    def __len__(self) -> int:
        return self.count

    @property
    def shape(self):
        return (self.count, 128)

    def _gather(self, start: int, stop: int) -> torch.Tensor:
        return dequantize_indexer_k_fp4(self.pool, self.slots[start:stop])

    def float(self) -> torch.Tensor:
        # Full materialization; only reached for short spans (the scorer's
        # unchunked branch) — bounded by DSV41_PREFILL_INDEXER_KEY_CHUNK in
        # long-context configurations.
        return self._gather(0, self.count).float()

    def __getitem__(self, item):
        if not isinstance(item, slice):
            raise TypeError("LazyIndexerKeys supports only slice gathers")
        start = item.start or 0
        stop = self.count if item.stop is None else item.stop
        return self._gather(start, stop)


class PagedSparseKVSource:
    """Callable row source for ``_sparse_fwd_reference``.

    Index space mirrors ``combine_kv`` for a single request: rows
    ``[0, n_global)`` are compressed globals read lazily from the pool; rows
    ``[n_global, n_global + swa_rows)`` are the dense SWA workspace rows.
    ``-1`` indices gather zero-filled rows and are masked by ``topk_length``.
    """

    __slots__ = ("paged", "swa")

    def __init__(self, paged: PagedGlobalValues, swa):
        self.paged = paged
        self.swa = swa

    def __call__(self, idx: torch.Tensor) -> torch.Tensor:
        n_global = self.paged.count
        clamped = idx.clamp(min=0, max=max(n_global - 1, 0))
        vals = self.paged.gather(clamped)
        if self.swa is not None and self.swa.shape[0] > 0:
            sw_idx = (idx - n_global).clamp(min=0, max=self.swa.shape[0] - 1)
            is_sw = (idx >= n_global) & (idx >= 0)
            vals = torch.where(is_sw[:, None], self.swa[sw_idx], vals)
        return vals


def publish_paged_globals(attn, main_pool, index_pool, indexer_region, ends, ratio: int, device) -> bool:
    """Publish lazy (g, k) markers for a single-request prefill.

    Returns False (caller falls back to the dense producer) when any
    precondition fails; never raises into the forward path.
    """
    try:
        if ratio <= 0 or len(ends) != 1:
            return False
        count = int(ends[0]) // ratio
        if count <= 0:
            return False
        positions = (torch.arange(count, device=device, dtype=torch.long) + 1) * ratio - 1
        req_ids = torch.zeros(count, device=device, dtype=torch.long)
        slots = attn._slots(indexer_region, positions, req_ids)
        if slots is None:
            return False
        paged_g = PagedGlobalValues(attn, attn._global_region(), ratio, count, device)
        lazy_k = LazyIndexerKeys(index_pool, slots, count)
        attn._shared_attention["global"] = {attn.layer_id: [(paged_g, lazy_k)]}
        attn._shared_attention.pop("prefill_chunk_meta", None)
        attn._shared_attention.pop("prefill_kv_workspace", None)
        return True
    except Exception as exc:  # noqa: BLE001 — dense fallback keeps correctness
        logger.warning("[paged-prefill] falling back to dense globals: %r", exc)
        return False
