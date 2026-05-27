"""Phase 3 — DSv4 decode FMHA impl for CUDA graph capture/replay.

The C++ ``CudaGraphRunner`` (``cpp/cuda_graph/cuda_graph_runner.cc``)
expects:

* ``model.prepare_fmha_impl(inputs, is_cuda_graph=True)`` to return an
  object whose ``support_cuda_graph()`` is ``True`` (which it is iff
  ``prepare_cuda_graph`` is callable on the object).
* The captured ``forward`` to read its decode metadata from this impl
  rather than building fresh metadata each call. Replay re-fires the
  exact same kernel launches with frozen pointers — the only way to
  get different behavior across replays is to update the buffers
  ``forward`` reads from BEFORE replay. That update is
  ``prepare_cuda_graph(attn_inputs)``.

This impl is **DSv4-specific** and intentionally does not inherit
``MlaImplBase`` because DSv4's attention does not use the standard MLA
FMHA pipeline (custom CSA / HCA / SWA + TileLang sparse attention).
The impl only carries:

  * Pre-allocated ``DSv4DecodeAttnMetadataFP8`` (sized for the captured BS).
  * A ``prepare_cuda_graph(attn_inputs)`` method that updates the
    metadata in place from the framework attention inputs.

``DeepSeekV4Model._forward_decode`` reads the metadata off the impl
when one is supplied; otherwise it builds inline (Phase 2 eager path).
The eager path is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

from rtp_llm.models_py.modules.dsv4.fp8.decode.decode_attn_metadata import (
    DSv4DecodeAttnMetadataFP8,
    allocate_decode_metadata_fp8,
    update_decode_metadata_in_place_fp8,
)


@dataclass
class DSv4DecodeFmhaImplConfigFP8:
    """Geometry the impl needs at construction. Comes from V4Args via
    ``DeepSeekV4Model.prepare_fmha_impl``."""

    max_batch_size: int
    q_len: int
    window_size: int
    head_dim: int
    max_seq_len: int
    compress_ratios: List[int]
    index_topk: int

    # Phase 2 paged-decode wiring. When provided, the impl pre-allocates
    # per-attn_type block_table + slot_mapping buffers in the metadata
    # and ``prepare`` populates them from
    # ``attn_inputs.kv_cache_kernel_block_id_device_by_group``.
    #
    # ``paged_pool_specs[attn_type]`` is
    # ``(entries_per_block, tokens_per_block, max_blocks_per_req)``.
    paged_pool_specs: Dict[int, Tuple[int, int, int]] = field(default_factory=dict)

    # Snapshot of ``kv_cache.group_region_names`` (framework-owned group
    # ordering, one attn_type per entry). Position = group id. ``prepare``
    # iterates this to index ``attn_inputs.kv_cache_kernel_block_id_device_by_group``
    # without needing a live ``kv_cache`` (the CUDA-graph replay path
    # doesn't hand one in). Static for the allocator's lifetime, so
    # snapshot-at-construct is safe.
    group_region_names: List[int] = field(default_factory=list)


class DSv4DecodeFmhaImplFP8:
    """CUDA-graph-friendly DSv4 decode "FMHA impl".

    Owns persistent metadata buffers; ``prepare_cuda_graph`` updates them
    in place across replays so the captured graph reads fresh values
    from stable addresses.
    """

    def __init__(
        self,
        config: DSv4DecodeFmhaImplConfigFP8,
        device: torch.device,
        attn_inputs: Any,
    ) -> None:
        self.config = config
        self.device = device
        self.metadata: DSv4DecodeAttnMetadataFP8 = allocate_decode_metadata_fp8(
            max_batch_size=config.max_batch_size,
            q_len=config.q_len,
            window_size=config.window_size,
            head_dim=config.head_dim,
            max_seq_len=config.max_seq_len,
            compress_ratios=config.compress_ratios,
            index_topk=config.index_topk,
            device=device,
            paged_pool_specs=config.paged_pool_specs,
        )
        self._paged_entries_per_block: Dict[int, int] = {
            at: spec[0] for at, spec in config.paged_pool_specs.items()
        }
        self._paged_tokens_per_block: Dict[int, int] = {
            at: spec[1] for at, spec in config.paged_pool_specs.items()
        }
        self.prepare(attn_inputs, forbid_realloc=True)

    def support_cuda_graph(self) -> bool:
        # Always True for this impl class — the legacy ``callable(getattr(...))``
        # check was a copy of MlaImplBase's pattern that never evaluated to
        # False here (prepare_cuda_graph is hardcoded on the class).
        return True

    def _extract_paged_block_tables(
        self,
        attn_inputs: Any,
    ) -> Optional[Dict[int, torch.Tensor]]:
        if not self._paged_entries_per_block or not self.config.group_region_names:
            return None
        by_group = getattr(
            attn_inputs,
            "kv_cache_kernel_block_id_device_by_group",
            None,
        )
        if by_group is None or len(by_group) == 0:
            return None
        paged_block_tables: Dict[int, torch.Tensor] = {}
        for group_id, attn_type in enumerate(self.config.group_region_names):
            if group_id >= len(by_group):
                continue
            if attn_type not in self.config.paged_pool_specs:
                continue
            group_block_table = by_group[group_id]
            if group_block_table is None or group_block_table.numel() == 0:
                continue
            paged_block_tables[attn_type] = group_block_table
        return paged_block_tables or None

    def prepare(self, attn_inputs: Any, forbid_realloc: bool = False) -> None:
        """Update persistent metadata and paged block-table snapshots."""
        paged_block_tables = self._extract_paged_block_tables(attn_inputs)
        update_decode_metadata_in_place_fp8(
            self.metadata,
            attn_inputs,
            forbid_realloc=forbid_realloc,
            paged_block_tables=paged_block_tables,
            paged_pool_entries_per_block=self._paged_entries_per_block,
            paged_pool_tokens_per_block=self._paged_tokens_per_block,
        )

    def prepare_cuda_graph(self, attn_inputs: Any) -> None:
        """Called by ``CudaGraphRunner::prepareInputs`` between every
        replay."""
        paged_block_tables = self._extract_paged_block_tables(attn_inputs)
        if self._paged_entries_per_block and paged_block_tables is None:
            raise RuntimeError(
                "prepare_cuda_graph: paged_pool_specs configured "
                "but paged_block_tables is empty — "
                "framework did not provide block tables"
            )
        update_decode_metadata_in_place_fp8(
            self.metadata,
            attn_inputs,
            forbid_realloc=True,
            paged_block_tables=paged_block_tables,
            paged_pool_entries_per_block=self._paged_entries_per_block,
            paged_pool_tokens_per_block=self._paged_tokens_per_block,
        )
