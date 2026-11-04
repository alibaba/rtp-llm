"""Map the PPU compute cache to framework-owned, tagged CacheStore pools.

Framework pools remain authoritative across forwards and across P/D. The
compute cache is a temporary logical view: paged global KV, a planar indexer
whose physical capacity is per RAW token, and fixed SWA/CSA rings are gathered
through their own block tables. No fixture or private network transport is used.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from rtp_llm.models_py.modules.dsv41.kv_cache_utils import as_attention_inputs_by_tag
from rtp_llm.models_py.modules.dsv41.ppu.cache import PPUKVCache
from rtp_llm.models_py.modules.dsv41.ppu.native_protocol import FRAMEWORK_POOL_TAGS

_FULL = {"global_kv_1", "global_kv_2", "indexer_kv"}
_WIDTH = {
    "global_kv_1": 288,
    "global_kv_2": 288,
    "indexer_kv": 68,
    "swa_kv": 528,
    "csa_state": 1024,
}


@dataclass
class _Pool:
    framework_tag: str
    handle: object
    tensor: torch.Tensor
    table: torch.Tensor
    entries: int
    span: int


class NativeCacheBridge:
    """Per-forward mapping; row ids are local batch rows, never DP owners."""

    def __init__(self, config, kv_cache, attention_inputs, *, device=None):
        self.config = config
        self.device = device
        self.native = kv_cache
        self.inputs = as_attention_inputs_by_tag(attention_inputs, kv_cache)
        self.logical = PPUKVCache(config, device=device)
        self.pools = {}
        if kv_cache is None:
            raise ValueError("native V4.1 forward requires framework KVCache resources")
        for key in self.logical.pool_keys():
            owner, tag = key
            framework_tag = FRAMEWORK_POOL_TAGS[tag]
            if framework_tag not in self.inputs:
                raise ValueError(
                    f"missing native attention inputs for pool {framework_tag} "
                    f"(PPU canonical {tag})"
                )
            handle = kv_cache.get_layer_cache(owner, framework_tag)
            tensor = handle.kv_cache_base
            if tensor.ndim < 2 or not tensor.is_contiguous():
                raise ValueError(
                    f"native pool {key} must be contiguous block-major storage"
                )
            tensor = tensor.view(tensor.shape[0], math.prod(tensor.shape[1:]))
            entries = int(kv_cache.get_entries_per_block(framework_tag))
            span = int(
                kv_cache.get_kernel_seq_size_per_block(framework_tag)
                if tag in _FULL
                else kv_cache.get_seq_size_per_block(framework_tag)
            )
            expected_dtype = torch.float32 if tag == "csa_state" else torch.uint8
            if entries <= 0 or span <= 0 or tensor.dtype != expected_dtype:
                raise ValueError(f"invalid native pool dtype/geometry for {key}")
            if tensor.shape[1] < entries * _WIDTH[tag]:
                raise ValueError(
                    f"native pool {key} stride is smaller than its logical payload"
                )
            if tag == "swa_kv" and entries < self.logical.geometry.window:
                raise ValueError(
                    "SWA native ring cannot hold the complete attention window"
                )
            if tag == "csa_state" and entries < 2:
                raise ValueError("CSA native ring cannot hold a ratio-2 boundary")
            inputs = self.inputs[framework_tag]
            field = (
                "kv_cache_kernel_block_id_device"
                if tag in _FULL
                else "kv_cache_block_id_device"
            )
            table = getattr(inputs, field, None)
            if table is None or not table.numel():
                table = getattr(inputs, field.removesuffix("_device"), None)
            if table is None or table.ndim != 2:
                raise ValueError(f"native pool {tag} requires its own block table")
            self.pools[key] = _Pool(
                framework_tag, handle, tensor, table.to(tensor.device), entries, span
            )

    def _coordinates(self, key, request_row, positions):
        pool = self.pools[key]
        positions = positions.to(device=pool.tensor.device, dtype=torch.int64)
        logical_blocks = positions // pool.span
        if positions.numel() and (
            int(positions.min()) < 0 or int(logical_blocks.max()) >= pool.table.shape[1]
        ):
            raise ValueError(f"native pool {key} position outside block table")
        blocks = pool.table[int(request_row), logical_blocks].long()
        if blocks.numel() and (
            int(blocks.min()) < 0 or int(blocks.max()) >= pool.tensor.shape[0]
        ):
            raise ValueError(
                f"native pool {key} refers to an unallocated physical block"
            )
        if key[1] in _FULL:
            ratio = self.logical.geometry.ratio(key[0])
            offsets = (positions % pool.span) // ratio
        else:
            offsets = positions % pool.entries
        if offsets.numel() and int(offsets.max()) >= pool.entries:
            raise ValueError(
                f"native pool {key} offset exceeds its logical entry count"
            )
        return blocks, offsets

    def _copy_rows(self, key, request_row, positions, rows=None):
        """Gather/scatter logical rows, preserving alignment bytes and planes."""
        pool = self.pools[key]
        blocks, offsets = self._coordinates(key, request_row, positions)
        width = _WIDTH[key[1]]
        result = (
            torch.empty(
                (positions.numel(), width),
                dtype=pool.tensor.dtype,
                device=pool.tensor.device,
            )
            if rows is None
            else None
        )
        # Bound advanced-index temporaries for long requests.
        for start in range(0, positions.numel(), 4096):
            stop = min(start + 4096, positions.numel())
            b, off = blocks[start:stop, None], offsets[start:stop, None]
            if key[1] == "indexer_kv":
                columns = torch.cat(
                    (
                        off * 64 + torch.arange(64, device=b.device),
                        pool.entries * 64 + off * 4 + torch.arange(4, device=b.device),
                    ),
                    dim=1,
                )
            else:
                columns = off * width + torch.arange(width, device=b.device)
            if rows is None:
                result[start:stop] = pool.tensor[b, columns]
            else:
                pool.tensor[b, columns] = rows[start:stop].to(pool.tensor.device)
        return result

    def _positions(self, key, length):
        tag = key[1]
        if tag in _FULL:
            ratio = self.logical.geometry.ratio(key[0])
            return torch.arange(length // ratio, dtype=torch.int64) * ratio
        if tag == "swa_kv":
            return torch.arange(max(0, length - self.logical.geometry.window), length)
        # Only the last unpaired input is consumed by the ratio-2 compressor.
        return (
            torch.arange(length - 1, length)
            if length % 2
            else torch.empty(0, dtype=torch.int64)
        )

    def import_requests(self, lengths):
        """Restore already-cached prefixes; cold prefill uses zero lengths."""
        for row, length in enumerate(lengths):
            self.logical.alloc(row, int(length))
            for key in self.logical.pool_keys():
                positions = self._positions(key, int(length))
                if not positions.numel():
                    continue
                rows = self._copy_rows(key, row, positions)
                if key[1] in _FULL:
                    count = positions.numel()
                    slots = self.logical._slots_for_blocks(key, row, count)
                else:
                    slots = self.logical.slots_at_positions(*key, row, positions)
                self.logical.insert_rows(*key, slots, rows)
        return self.logical

    def export_requests(self, cache, lengths, *, publish=False):
        """Commit valid logical rows, then invoke the native per-tag writers."""
        for row, length in enumerate(lengths):
            for key in cache.pool_keys():
                positions = self._positions(key, int(length))
                if not positions.numel():
                    continue
                if key[1] in _FULL:
                    slots = cache._slots_for_blocks(key, row, positions.numel())
                else:
                    slots = cache.slots_at_positions(*key, row, positions)
                packed = cache._gather(key, slots)
                self._copy_rows(key, row, positions, packed)
        if publish:
            if self.device is not None and torch.device(self.device).type == "cuda":
                torch.cuda.synchronize(self.device)
            for (owner, tag), pool in self.pools.items():
                inputs = self.inputs[pool.framework_tag]
                writer = getattr(inputs, "cache_store_writer", None)
                params = getattr(inputs, "cache_store_inputs", None)
                if writer is not None and params is not None:
                    writer.write(params, pool.handle)
