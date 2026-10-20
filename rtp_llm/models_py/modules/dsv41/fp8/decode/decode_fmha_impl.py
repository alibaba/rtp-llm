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

from rtp_llm.models_py.modules.dsv41.fp8.decode.decode_attn_metadata import (
    DSv4DecodeAttnMetadataFP8,
    allocate_decode_metadata_fp8,
    update_decode_metadata_in_place_fp8,
)
from rtp_llm.models_py.modules.dsv41.kv_cache_utils import (
    as_attention_inputs_by_tag,
    as_tagged_attention_inputs,
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
    # ``attention_inputs[tag].kv_cache_kernel_block_id_device``.
    #
    # ``paged_pool_specs[attn_type]`` is
    # ``(entries_per_block, tokens_per_block, max_blocks_per_req)``.
    paged_pool_specs: Dict[int, Tuple[int, int, int]] = field(default_factory=dict)

    # Snapshot of ``kv_cache.group_tags`` (framework-owned group
    # ordering, one attn_type per entry). Position = group id. ``prepare``
    # iterates this to index ``attention_inputs[tag].kv_cache_kernel_block_id_device``
    # without needing a live ``kv_cache`` (the CUDA-graph replay path
    # doesn't hand one in). Static for the allocator's lifetime, so
    # snapshot-at-construct is safe.
    group_tags: List[int] = field(default_factory=list)


class DSv4DecodeFmhaImplFP8:
    """CUDA-graph-friendly DSv4 decode "FMHA impl".

    Owns persistent metadata buffers; ``prepare_cuda_graph`` updates them
    in place across replays so the captured graph reads fresh values
    from stable addresses.
    """

    # CudaGraphRunner can leave captured host mirrors untouched for this
    # backend. Preparation consumes only device lengths / block tables, and
    # replay reads the persistent device buffers built from them. The runner
    # preserves stream dependencies and padding of those device inputs.
    cuda_graph_uses_device_metadata = True

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
        # When the shared planner has no compressed-ratio (>1) pools, V4.1's
        # attention owns the compressed/index/state slot plans.  Its decode
        # metadata only has to refresh the always-written SWA pool (and the
        # decoder-SWA child), so the compressed/state/topk-buffer rebuild in
        # ``update_decode_metadata_in_place_fp8`` is dead work there.
        self._swa_only_metadata = not any(
            int(r) > 1 for r in config.compress_ratios
        )
        # The CUDA-graph runner refreshes its own stable per-group block-table
        # device buffers before every replay.  Alias those buffers here (at
        # construction, i.e. before capture) instead of copying them into a
        # second metadata-owned buffer on every prepare call.
        self._bound_block_tables: Optional[Dict[str, torch.Tensor]] = (
            self._bind_pool_block_tables(attn_inputs)
            if attn_inputs is not None
            else None
        )
        self.prepare(attn_inputs, forbid_realloc=True)

    def support_cuda_graph(self) -> bool:
        # Always True for this impl class — the legacy ``callable(getattr(...))``
        # check was a copy of MlaImplBase's pattern that never evaluated to
        # False here (prepare_cuda_graph is hardcoded on the class).
        return True

    def _extract_paged_block_tables(self, attn_inputs):
        from rtp_llm.models_py.modules.dsv41.kv_cache_utils import (
            build_block_tables_for_tags,
        )

        return build_block_tables_for_tags(
            None, attn_inputs, self.config.paged_pool_specs
        )

    def _bind_pool_block_tables(
        self, attn_inputs: Any
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Alias the graph runner's stable per-group block tables.

        ``CudaGraphRunner::prepareAttentionInputs`` already writes the current
        step's block ids into the captured per-group
        ``kv_cache_kernel_block_id_device`` buffers with one fused strided D2D
        copy.  Keeping a second metadata-owned copy therefore duplicates both
        the host loop and the device traffic.  Binding is only valid when the
        source is the capture-stable device tensor; otherwise fall back to the
        copying path.

        The Python planner sizes its tables from
        ``ceil(max_seq_len / kernel_seq_size_per_block) + 1`` while the C++
        capture buffer is sized by ``captureKernelBlockTableWidth`` (which
        includes speculative reserve and kernel-block expansion).  The C++
        width is the contract the captured kernels must read: keep the
        metadata tensor at that width so the alias is a contiguous view and the
        Triton row stride matches the source.
        """
        specs = self.config.paged_pool_specs
        if not specs:
            return None
        by_tag = as_attention_inputs_by_tag(attn_inputs)
        sources: Dict[str, torch.Tensor] = {}
        # Validate everything before mutating metadata, so a fallback never
        # leaves a partially resized table set behind.
        for attn_type in specs:
            key = str(attn_type)
            dst = self.metadata.pool_block_tables.get(key)
            src = getattr(
                by_tag.get(key), "kv_cache_kernel_block_id_device", None
            )
            if (
                dst is None
                or not isinstance(src, torch.Tensor)
                or not src.is_cuda
                or src.device.type != self.device.type
                or (
                    self.device.index is not None
                    and src.device.index != self.device.index
                )
                or src.dtype != torch.int32
                or src.dim() != 2
                or not src.is_contiguous()
                or src.shape[0] < dst.shape[0]
                or src.shape[1] < dst.shape[1]
            ):
                return None
            sources[key] = src
        if len(sources) != len(specs):
            return None

        bound: Dict[str, torch.Tensor] = {}
        for key, src in sources.items():
            dst = self.metadata.pool_block_tables[key]
            # The alias is already contiguous at the source width; replacing
            # the metadata tensor is the resize.
            bound[key] = src[: dst.shape[0]]
        self.metadata.pool_block_tables.update(bound)
        self._refresh_decoder_swa_block_tables(bound)
        return bound

    def _refresh_decoder_swa_block_tables(
        self, bound: Dict[str, torch.Tensor]
    ) -> None:
        """Keep the decoder-SWA child's table view on the freshly bound parent.

        ``allocate_decode_metadata_fp8`` creates ``decoder_swa_metadata``
        before this impl can bind the runner's tables.  The child holds its own
        ``pool_block_tables`` dict, so resizing/binding only the parent would
        leave decoder write-slot planning reading the old zero table.
        """
        child = getattr(self.metadata, "decoder_swa_metadata", None)
        if child is None:
            return
        from rtp_llm.models_py.modules.dsv41.attn_type import (
            DECODER_SWA_KV,
            SWA_KV,
        )

        for key, table in bound.items():
            if key != SWA_KV:
                child.pool_block_tables[key] = table
        region_table = self.metadata.pool_block_tables.get(DECODER_SWA_KV)
        if region_table is not None:
            child.pool_block_tables[SWA_KV] = region_table
            child.pool_block_tables[DECODER_SWA_KV] = region_table

    def prepare(self, attn_inputs: Any, forbid_realloc: bool = False) -> None:
        """Update persistent metadata and paged block-table snapshots.

        Called once at construction (before the capture harness runs its two
        warmup forwards + the captured forward). FlashMLA freezes its sparse
        tile schedule on the first warmup forward keyed on the ``topk_length``
        VALUES present now, and that schedule is reused for every replay. So we
        write FULL-WIDTH effective lengths here (``capture_full_width_lengths``)
        to max-provision the frozen schedule; the real per-request lengths are
        written before each replay by :meth:`prepare_cuda_graph`. See
        ``opt_flash_mla/design/01_cuda_graph_sched_meta_freeze.md``.
        """
        paged_block_tables = (
            None
            if self._bound_block_tables is not None
            else self._extract_paged_block_tables(attn_inputs)
        )
        update_decode_metadata_in_place_fp8(
            self.metadata,
            as_tagged_attention_inputs(attn_inputs),
            forbid_realloc=forbid_realloc,
            paged_block_tables=paged_block_tables,
            paged_pool_entries_per_block=self._paged_entries_per_block,
            paged_pool_tokens_per_block=self._paged_tokens_per_block,
            capture_full_width_lengths=True,
            bound_block_tables=self._bound_block_tables is not None,
            swa_only_metadata=self._swa_only_metadata,
        )

    def prepare_cuda_graph(self, attn_inputs: Any) -> None:
        """Called by ``CudaGraphRunner::prepareInputs`` between every
        replay.

        Writes the REAL per-request effective lengths
        (``capture_full_width_lengths=False``, the default): the captured
        FlashMLA kernel reads these for per-query masking so the sparse decode
        only scans ``min(window, seq_len)`` / ``seq_len // ratio`` instead of
        the full capture width, while the frozen (full-width) tile schedule from
        :meth:`prepare` stays a safe superset. See
        ``opt_flash_mla/design/01_cuda_graph_sched_meta_freeze.md``.
        """
        paged_block_tables = (
            None
            if self._bound_block_tables is not None
            else self._extract_paged_block_tables(attn_inputs)
        )
        update_decode_metadata_in_place_fp8(
            self.metadata,
            as_tagged_attention_inputs(attn_inputs),
            forbid_realloc=True,
            paged_block_tables=paged_block_tables,
            paged_pool_entries_per_block=self._paged_entries_per_block,
            paged_pool_tokens_per_block=self._paged_tokens_per_block,
            require_device_metadata=True,
            bound_block_tables=self._bound_block_tables is not None,
            swa_only_metadata=self._swa_only_metadata,
        )
