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
  ``prepare_cuda_graph(attn_inputs, valid_request_count=...)``. This backend
  opts in to the runner's live request count; padded tensor shapes alone
  cannot distinguish live requests from CUDA Graph padding.

This impl is **DSv4-specific** and intentionally does not inherit
``MlaImplBase`` because DSv4's attention does not use the standard MLA
FMHA pipeline (custom CSA / HCA / SWA + TileLang sparse attention).
The impl only carries:

  * Pre-allocated ``DSv4DecodeAttnMetadataFP8`` (sized for the captured BS).
  * A ``prepare_cuda_graph`` method that updates metadata and selection
    validity in place, without changing the original MTP/KV-write lengths.

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
from rtp_llm.models_py.modules.dsv41.kv_cache_utils import as_tagged_attention_inputs


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
    cuda_graph_uses_valid_request_count = True

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
        # Capture-time C++ replay plan; None keeps the Python callback path.
        self.cuda_graph_replay_plan = self._build_cuda_graph_replay_plan(attn_inputs)

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
        paged_block_tables = self._extract_paged_block_tables(attn_inputs)
        update_decode_metadata_in_place_fp8(
            self.metadata,
            as_tagged_attention_inputs(attn_inputs),
            forbid_realloc=forbid_realloc,
            paged_block_tables=paged_block_tables,
            paged_pool_entries_per_block=self._paged_entries_per_block,
            paged_pool_tokens_per_block=self._paged_tokens_per_block,
            capture_full_width_lengths=True,
        )

    def prepare_cuda_graph(self, attn_inputs: Any, *, valid_request_count: int | None = None) -> None:
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
        if valid_request_count is None:
            # An older C++ runner cannot infer live rows from padded tensor
            # shapes. Fail rather than silently retain capture-time validity.
            raise ValueError("CUDA Graph replay requires the live request count from the matching C++ runner")
        paged_block_tables = self._extract_paged_block_tables(attn_inputs)
        update_decode_metadata_in_place_fp8(
            self.metadata,
            as_tagged_attention_inputs(attn_inputs),
            forbid_realloc=True,
            paged_block_tables=paged_block_tables,
            paged_pool_entries_per_block=self._paged_entries_per_block,
            paged_pool_tokens_per_block=self._paged_tokens_per_block,
            require_device_metadata=True,
            valid_request_count=valid_request_count,
        )

    def _build_cuda_graph_replay_plan(self, attn_inputs: Any) -> Any:
        """Build the capture-time C++ replay plan (one fused kernel launch).

        The plan pins raw device pointers to this impl's persistent metadata
        tensors and to the captured per-tag block-table mirrors that
        ``CudaGraphRunner`` refreshes with its fused strided D2D copy before
        each replay. Replay preparation then runs entirely in C++
        (``tryRunDsv41ReplayMetadataPrep``): no GIL, no Python interpreter,
        no per-replay aten/Triton launches.

        Returns ``None`` when the fused op is unavailable (non-CUDA builds,
        older wheels) so the runner keeps the Python callback path.
        """
        from rtp_llm.models_py.modules.dsv41.kv_cache_utils import (
            as_attention_inputs_by_tag,
        )
        from rtp_llm.models_py.modules.dsv41.attn_type import (
            CSA_KV,
            CSA_STATE,
            DECODER_SWA_KV,
            HCA_KV,
            HCA_STATE,
            INDEXER_KV,
            INDEXER_STATE,
            SWA_KV,
        )

        try:
            from rtp_llm.ops.compute_ops import rtp_llm_ops
            plan_factory = getattr(rtp_llm_ops, "dsv41_decode_replay_metadata_plan", None)
        except ImportError:
            return None
        if plan_factory is None:
            return None

        meta = self.metadata
        if not meta.pool_block_tables:
            return None

        by_tag = as_attention_inputs_by_tag(attn_inputs, None)
        block_tables = {}
        for tag, tagged in by_tag.items():
            table = getattr(tagged, "kv_cache_kernel_block_id_device", None)
            if table is None or table.numel() == 0:
                continue
            block_tables[tag] = table

        role_by_tag = {
            SWA_KV: "swa_kv",
            DECODER_SWA_KV: "decoder_swa_kv",
            CSA_KV: "csa_kv",
            HCA_KV: "hca_kv",
            INDEXER_KV: "indexer_kv",
            CSA_STATE: "state_ring",
            HCA_STATE: "state_ring",
            INDEXER_STATE: "state_ring",
        }

        pools = []           # (table, role, entries, tokens)
        write_slots = []     # aligned with the pools that own a slot buffer
        mirrors = []         # meta.pool_block_tables snapshots, same order
        swa_outputs = []     # (tensor, pool_index)
        swa_pool_index = {}
        for at, spec in self.config.paged_pool_specs.items():
            table = block_tables.get(str(at))
            mirror = meta.pool_block_tables.get(at)
            if table is None or mirror is None:
                continue
            role = role_by_tag.get(at)
            if role is None:
                continue
            entries, tokens, _max_blocks = (int(v) for v in spec)
            pool_index = len(pools)
            pools.append((table, role, entries, tokens))
            mirrors.append(mirror)
            slots = meta.pool_write_slot_mappings.get(at)
            if role == "decoder_swa_kv":
                # The decoder-SWA layers read the child metadata's own slot
                # buffer (update_decoder_swa_metadata), not the parent's
                # duplicate entry.
                child = meta.decoder_swa_metadata
                if child is not None:
                    slots = child.pool_write_slot_mappings.get(SWA_KV)
            write_slots.append(
                slots
                if slots is not None
                else torch.empty(0, dtype=torch.long, device=meta.start_pos.device)
            )
            if role in ("swa_kv", "decoder_swa_kv") and slots is not None:
                swa_pool_index[role] = pool_index

        if not pools:
            return None

        # SWA read-slot translation outputs: the parent uses the SWA_KV table;
        # the bounded-replay decoder child uses DECODER_SWA_KV.
        slot_outputs = []
        slot_pool_indices = []
        if meta.swa_global_slots is not None and "swa_kv" in swa_pool_index:
            slot_outputs.append(meta.swa_global_slots)
            slot_pool_indices.append(swa_pool_index["swa_kv"])
        child = meta.decoder_swa_metadata
        if (
            child is not None
            and child.swa_global_slots is not None
            and "decoder_swa_kv" in swa_pool_index
        ):
            slot_outputs.append(child.swa_global_slots)
            slot_pool_indices.append(swa_pool_index["decoder_swa_kv"])

        q_len = int(meta.q_len_per_req)
        batch = int(meta.start_pos.numel())
        max_start = max(0, int(meta.max_seq_len) - q_len)
        dev = meta.start_pos.device
        topk_buf = (
            meta.topk_buffer_compressed
            if meta.topk_buffer_compressed is not None
            else torch.empty(0, dtype=torch.int32, device=dev)
        )
        # The FP8 graph path always allocates these; placeholders keep the
        # factory's tensor contract on reduced configurations.
        empty_i64 = torch.empty(0, dtype=torch.long, device=dev)
        position_ids_long = meta.position_ids_long if meta.position_ids_long is not None else empty_i64
        seq_start = (
            meta.decode_seq_start_per_req
            if meta.decode_seq_start_per_req is not None
            else empty_i64
        )
        swa_topk_length = (
            meta.swa_topk_length
            if meta.swa_topk_length is not None
            else torch.zeros(batch, dtype=torch.int32, device=dev)
        )
        slot_mapping_swa = (
            meta.slot_mapping_swa
            if meta.slot_mapping_swa is not None
            else torch.full((batch * q_len,), -1, dtype=torch.int32, device=dev)
        )
        topk_window_idxs = (
            meta.topk_window_idxs
            if meta.topk_window_idxs is not None
            else torch.empty(0, dtype=torch.int32, device=dev)
        )
        swa_abs_idx = meta.swa_abs_idx if meta.swa_abs_idx is not None else topk_window_idxs

        return plan_factory(
            block_tables=[p[0] for p in pools],
            roles=[p[1] for p in pools],
            entries_per_block=[p[2] for p in pools],
            tokens_per_block=[p[3] for p in pools],
            valid_requests=meta.valid_requests,
            start_pos=meta.start_pos,
            position_ids=meta.position_ids,
            position_ids_long=position_ids_long,
            decode_seq_start_per_req=seq_start,
            slot_mapping_swa=slot_mapping_swa,
            topk_window_idxs=topk_window_idxs,
            swa_abs_idx=swa_abs_idx,
            topk_buffer_compressed=topk_buf,
            swa_topk_length=swa_topk_length,
            pool_write_slots=write_slots,
            pool_block_table_mirrors=mirrors,
            swa_slot_outputs=slot_outputs,
            swa_slot_pool_indices=slot_pool_indices,
            q_len=q_len,
            batch_size=batch,
            window=int(meta.window_size),
            max_start=max_start,
        )
