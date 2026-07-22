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

  * Pre-allocated ``DSv4DecodeAttnMetadata`` (sized for the captured BS).
  * A ``prepare_cuda_graph(attn_inputs)`` method that updates the
    metadata in place with the new ``start_pos`` (= ``attn_inputs.sequence_lengths``).

``DeepSeekV4Model._forward_decode`` reads the metadata off the impl
when one is supplied; otherwise it builds inline (Phase 2 eager path).
The eager path is unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from rtp_llm.models_py.modules.dsv4.attn_type import TAG_BY_ATTN_TYPE
from rtp_llm.models_py.modules.dsv4.decode.decode_attn_metadata import (
    DSv4DecodeAttnMetadata,
    allocate_decode_metadata,
    update_decode_metadata_in_place,
)

_ATTN_TYPE_BY_TAG = {tag: attn_type for attn_type, tag in TAG_BY_ATTN_TYPE.items()}


@dataclass
class DSv4DecodeFmhaImplConfig:
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
    # the selected group's ``attn_inputs.kv_cache_kernel_block_id_device``.
    #
    # ``paged_pool_specs[attn_type] =
    # (entries_per_block, tokens_per_block, max_blocks_per_req)``.
    # If empty, the legacy register_buffer-only path is used.
    paged_pool_specs: Dict[int, Tuple[int, int, int]] = field(default_factory=dict)


class DSv4DecodeFmhaImpl:
    """CUDA-graph-friendly DSv4 decode "FMHA impl".

    Owns persistent metadata buffers; ``prepare_cuda_graph`` updates them
    in place across replays so the captured graph reads fresh values
    from stable addresses.
    """

    def __init__(
        self,
        config: DSv4DecodeFmhaImplConfig,
        device: torch.device,
        attn_inputs=None,
    ) -> None:
        self.config = config
        self.device = device
        self.metadata: DSv4DecodeAttnMetadata = allocate_decode_metadata(
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
        # Cache entries_per_block lookup — passed unchanged to
        # update_decode_metadata_in_place every prepare call.
        self._paged_entries_per_block: Dict[int, int] = {
            at: spec[0] for at, spec in config.paged_pool_specs.items()
        }
        # Populate metadata so the initial dtype-check forward (called by
        # CudaGraphRunner::initCapture BEFORE any prepare_cuda_graph) reads
        # valid values rather than the zero/-1 sentinels from allocation.
        # Mirrors flashmla_sparse_impl.py:386 (create_params → prepare in __init__).
        # ``forbid_realloc=True`` here too — allocate_decode_metadata has
        # already created every destination buffer; update_decode_metadata_in_place
        # only ``.copy_`` into them, so any realloc on the first prepare is a
        # bug (and would silently bake the new ptr into the captured graph).
        if attn_inputs is not None:
            self.prepare(attn_inputs, forbid_realloc=True)

    def support_cuda_graph(self) -> bool:
        # Always True for this impl class — the legacy ``callable(getattr(...))``
        # check was a copy of MlaImplBase's pattern that never evaluated to
        # False here (prepare_cuda_graph is hardcoded on the class).
        return True

    def prepare(self, attn_inputs, forbid_realloc: bool = False) -> None:
        """Eager-path preparation: extract ``start_pos`` from
        ``attn_inputs.sequence_lengths`` (per
        ``NormalModelInputGatherer.cc:255`` this is exactly the absolute
        position of the new token's predecessor — i.e. our ``start_pos``).
        Then update the persistent metadata in place, including paged
        block_table snapshot when configured.
        """
        if isinstance(attn_inputs, Mapping):
            if not attn_inputs:
                raise RuntimeError("DSV4 attention input tag mapping must not be empty")
            attn_base = next(iter(attn_inputs.values()))
            tagged_inputs = attn_inputs.items()
        else:
            attn_base = attn_inputs
            tagged_inputs = ()

        seq_lens = attn_base.sequence_lengths
        if seq_lens.device != self.device:
            seq_lens = seq_lens.to(self.device)
        start_pos = seq_lens.to(torch.int32)
        # Clamp for warmup safety — the framework warmup probes at
        # max_seq_len then decode pushes start_pos past the freqs_cis
        # range. Same clamp used in DeepSeekV4Model._forward_decode.
        max_s = self.config.max_seq_len
        start_pos = torch.clamp(start_pos, min=0, max=max(0, max_s - 1))

        paged_block_tables: Optional[Dict[int, torch.Tensor]] = None
        if self._paged_entries_per_block:
            paged_block_tables = {}
            if not isinstance(attn_inputs, Mapping):
                if len(self.config.paged_pool_specs) != 1:
                    raise RuntimeError(
                        "plain DSV4 attention inputs require exactly one paged cache group"
                    )
                tagged_inputs = ((None, attn_inputs),)
            for tag, group_attn_inputs in tagged_inputs:
                attn_type = (
                    next(iter(self.config.paged_pool_specs))
                    if tag is None
                    else _ATTN_TYPE_BY_TAG.get(str(tag))
                )
                group_block_table = getattr(
                    group_attn_inputs, "kv_cache_kernel_block_id_device", None
                )
                if (
                    attn_type in self.config.paged_pool_specs
                    and group_block_table is not None
                    and group_block_table.numel() > 0
                ):
                    paged_block_tables[attn_type] = group_block_table
            paged_block_tables = paged_block_tables or None

        update_decode_metadata_in_place(
            self.metadata,
            start_pos,
            forbid_realloc=forbid_realloc,
            paged_block_tables=paged_block_tables,
            paged_pool_entries_per_block=self._paged_entries_per_block,
        )

    def prepare_cuda_graph(self, attn_inputs) -> None:
        """Called by ``CudaGraphRunner::prepareInputs`` between every
        replay. Re-runs ``prepare`` with ``forbid_realloc=True`` so any
        accidental buffer reallocation surfaces as an immediate error
        rather than a silent correctness bug (a captured graph still
        holds the old pointer and would compute on stale values)."""
        self.prepare(attn_inputs, forbid_realloc=True)
