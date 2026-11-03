"""PPU overrides; unchanged behavior is inherited from the shared DSV4 implementation."""

from __future__ import annotations
from rtp_llm.models_py.modules.dsv4.fp8.decode.decode_attn_metadata import (
    _pool_compress_ratio,
    _parse_paged_pool_specs,
    _resolve_paged_pool_tokens_per_block,
    _compressed_domain_tokens_per_block,
    _compute_state_pool_slot_mapping,
    _update_compressor_state_slot_mappings,
    _build_swa_slot_mapping,
    _build_compressed_slot_mapping,
    _build_window_topk_idxs,
    _build_position_ids_2d,
    _build_start_pos_from_attention_inputs,
    _build_swa_slot_mapping_from_positions,
    _build_compressed_slot_mapping_from_positions,
    _build_window_topk_idxs_from_positions,
    _compressed_index_width_for_ratio,
    _update_topk_lengths_in_place,
    update_decode_metadata_in_place_fp8,
)
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import torch
from rtp_llm.models_py.modules.dsv4.fp8.decode.decode_attn_metadata import DSv4DecodeAttnMetadataFP8 as _BaseDSv4DecodeAttnMetadataFP8

RICH_SWA_BUCKET = None

RICH_CSA_BUCKET = "csa_kv"

RICH_HCA_BUCKET = "hca_kv"

RICH_SCHED_BUCKETS = (RICH_SWA_BUCKET, RICH_CSA_BUCKET, RICH_HCA_BUCKET)

RICH_SWA_PAGE = 128

RICH_CSA_PAGE = 64

RICH_HCA_PAGE = 2

RICH_SWA_INDEX_WIDTH = 128

RICH_CSA_INDEX_WIDTH = 512

RICH_HCA_INDEX_WIDTH = 64

@dataclass
class DSv4DecodeAttnMetadataFP8:
    """Metadata produced once per decode step, consumed by every layer.

    Fields are device-resident tensors unless noted. Built by
    :func:`build_decode_metadata_fp8` at the top of
    ``DeepSeekV4Model.forward`` (decode arm).

    Layer-type-aware fields are dicts keyed by ``compress_ratio`` so a
    layer just looks up its own slice — keeps the per-layer call site
    free of branching.
    """

    # Geometry (Python-scalar, used at op call time for shapes)
    batch_size: int
    q_len_per_req: int  # 1 in pure decode, 1+spec for MTP
    total_tokens: int  # batch_size * q_len_per_req
    window_size: int  # SWA size (V4: 128)
    head_dim: int  # V4: 512
    max_seq_len: int
    swa_buffer_t_dim: int  # = window_size
    compressed_buffer_t_dim_per_ratio: Dict[int, int]  # ratio -> max_seq_len // ratio

    # Per-request first-token absolute positions (device int32 tensor; B-shaped).
    # Normal decode has q_len=1. Target verify has q_len>1 and derives the full
    # request-major position stream as ``start_pos[:, None] + arange(q_len)``.
    start_pos: torch.Tensor  # [B] int32
    # Per-token absolute positions, flattened token-major over [B, q_len].
    # Generated internally and used as the single source of truth for RoPE,
    # slot mapping, topk windows, and cache lengths.
    position_ids: torch.Tensor  # [T_total] int32
    # Long/int64 view of ``position_ids`` for compressor/indexer metadata.
    # Built once per step so every decode layer can reuse the same tensor
    # instead of re-running ``position_ids.to(torch.long)``.
    position_ids_long: torch.Tensor  # [T_total] int64

    # Slot mappings — flat over [T_total] with per-request offset baked in.
    # SWA: applies to all layers.
    slot_mapping_swa: torch.Tensor  # [T_total] int32

    # Compressed-K slot mappings — keyed by ratio (4 for CSA, 128 for HCA).
    # ``slot_mapping_compressed[ratio][t] == -1`` if this token does NOT
    # land on a compression boundary for that ratio (so the write op
    # should skip it).
    slot_mapping_compressed: Dict[int, torch.Tensor]  # ratio -> [T_total] int32

    # Window topk indices (the "trivial" topk = last ``win`` tokens).
    # Already on the per-layer KV buffer's flat slot-space.
    # Shape [B, q_len, win]; -1 entries mark out-of-range (early decode).
    topk_window_idxs: torch.Tensor  # [B, S, win] int32

    # Pre-allocated indexer output buffer (only CSA / ratio=4 layers use it).
    # IndexerDecodeV4Op writes into this per-step. Shape [B, S, K_compressed].
    topk_buffer_compressed: torch.Tensor  # [B, S, K] int32

    # Number of valid compressed-K entries each request currently has
    # in the compressed pool, per ratio. Used to mask topk after gather.
    # ``compressed_lens[ratio][r] = (start_pos[r] + 1) // ratio``  (after
    # this step's writes are applied).
    compressed_lens: Dict[int, torch.Tensor]  # ratio -> [B] int32
    # Per-token compressed lengths for target-verify/speculative decode.
    # ``compressed_lens_per_token[ratio][b, s] = (position_ids[b, s] + 1) // ratio``.
    # CSA indexer layers use this directly instead of recomputing it per layer.
    compressed_lens_per_token: Dict[int, torch.Tensor]  # ratio -> [B, S] int32

    # Concatenated topk indices per layer-type. For SWA-only layers,
    # equals ``topk_window_idxs``. For compressed layers (CSA/HCA),
    # equals ``cat([topk_window_idxs, topk_compressed_for_ratio], dim=-1)``.
    # IndexerDecodeV4Op fills the compressed half in-place.
    topk_total_by_ratio: Dict[
        int, torch.Tensor
    ]  # ratio -> [B, S, win+K_by_ratio] int32

    # Per-request running compressed-K offset (= win, since the
    # compressed pool starts right after the SWA ring buffer). Same for
    # every request in current scheme.
    compressed_offset: int = 128  # = window_size by default

    # ---- Cuda graph: reserved for Phase 3 (forbid_realloc, fixed addr) ----
    is_cuda_graph: bool = False

    # ------------------------------------------------------------------
    # Paged-decode metadata (paged BlockPool read/write).
    # Populated by the model + impl once the framework block tables are wired.
    # ------------------------------------------------------------------

    # Per-tag framework block_table: [max_B, max_blocks_per_req] int32.
    # Source: ``attention_inputs[tag].kv_cache_kernel_block_id_device``.
    # Keys are cache tags ("csa_kv" ... "swa_kv") from
    # :mod:`rtp_llm.models_py.modules.dsv4.kv_cache_utils`; only pools that the
    # model actually uses are present.
    pool_block_tables: Dict[str, torch.Tensor] = field(default_factory=dict)

    # Optional [B*Q] device bool in request-major order. A backend may use
    # this to suppress routed MoE work for graph padding / fake requests.
    active_token_mask: Optional[torch.Tensor] = None

    # Per-tag raw-token coverage for one block_table row. For the
    # seq=16384/kernel=128 layout, FULL paged pools carry 128 here while
    # SWA_KV carries 16384. Compressed writers convert this to their
    # compressed-index domain by dividing by the compression ratio.
    paged_pool_tokens_per_block: Dict[str, int] = field(default_factory=dict)

    # Per-tag new-token write slot mapping: [max_T_total] int64.
    # ``slot[t] = block_id * entries_per_block + in_block``. Block-table row
    # tokens and in-block ring entries may differ for SWA_KV; compressed
    # writers map in the compressed-index domain. ``-1`` = skip.
    pool_write_slot_mappings: Dict[str, torch.Tensor] = field(default_factory=dict)

    # Per-tag compressor state-pool slot mapping: [max_T_total] int64.
    # These are pure metadata for CompressorFP8.launch, identical across
    # same-type layers in a decode step. Building them here lets decode pass
    # a ready CompressorMeta down to every CSA/HCA/indexer compressor instead
    # of rebuilding state_slots / kv_slots / token_to_req inside each layer.
    compressor_state_slot_mappings: Dict[str, torch.Tensor] = field(
        default_factory=dict
    )

    # Per-step absolute positions for the SWA window — left-aligned,
    # ``-1`` padded for pre-sequence-start entries. Shape ``[B, q_len, win]``
    # int32. UNLIKE ``topk_window_idxs`` (which is request-local ring
    # slots in [0, win)), this carries the actual absolute token
    # positions in [0, max_seq_len) so it can be turned into global pool
    # slot ids via ``translate_local_to_global_slots``. Optional —
    # populated only when paged-decode read is enabled.
    swa_abs_idx: torch.Tensor = field(default=None)  # type: ignore[assignment]

    # Phase F: ``layer_pool_descs`` deleted — Attention resolves pool
    # views via ``self._kv_cache.get_layer_cache(layer_id, tag)``
    # at call time, no per-layer descriptor cache needed.

    # Iter3.3: shared-across-layers request-id mapping ``[T]`` int32 = the
    # ``arange(bsz)`` (for q_len=1) passed to every ``translate_local_to_global_slots``
    # call. 43 layers × same arange = one cached tensor.
    req_id_per_token: Optional[torch.Tensor] = None
    # Int64 companion consumed by ``CompressorFP8.prepare_metadata``.
    req_id_per_token_long: Optional[torch.Tensor] = None
    # Decode per-request first position and compact cu-seq offsets used by
    # compressor raw-path metadata for q_len > 1. These are layer-invariant.
    decode_seq_start_per_req: Optional[torch.Tensor] = None  # [B] int64
    decode_cu_seq_per_req: Optional[torch.Tensor] = None  # [B + 1] int64

    # Iter3.3: cached ``swa_global_slots`` = translate_local_to_global_slots(
    # req_id, swa_pool_bt, swa_abs_idx, swa_eb). Shape [T, win] int32.
    # Shared across all 43 layers (SWA block table + swa_abs_idx are
    # identical). When non-None the per-layer body skips the triton
    # translator call and reads this directly.
    swa_global_slots: Optional[torch.Tensor] = None

    # Iter3.3: cached ``hca_cmp_global_slots`` = translate_local_to_global_slots(
    # req_id, hca_pool_bt, hca_cmp_local, hca_eb). Shape [T, K_h] int32.
    # Shared across all HCA layers in a step (HCA block table + dense cmp
    # idx are identical across HCA layers). Only populated when an HCA
    # pool is bound.
    hca_cmp_global_slots: Optional[torch.Tensor] = None

    # FlashMLA ``sched_meta`` cache — per-(batch_size, q_len, extra type, width).
    # Mirrors vLLM's ``swa_metadata.tile_sched_{swaonly,c4a,c128a}`` pattern:
    # the planner is called once per (mode, B) combination and the returned
    # ``sched_meta`` is reused across all layers of the same type in a decode
    # step (43× in DSv4). Lives on the metadata so:
    #   * Eager path: rebuilt each step with the metadata instance, freed
    #     when metadata GCs — no cross-step accumulation.
    #   * CUDA graph path: populated lazily during capture (tensors alloc'd
    #     via the graph-aware allocator), stays alive for the impl lifetime
    #     so replay at the same captured B reuses the same Python tensors.
    # CSA and HCA layers MUST NOT share a sched_meta: the first
    # ``flash_mla_with_kvcache`` call bakes ``extra_k_cache.shape[1]``
    # (page_block_size of the compressed pool) into the sched_meta, and the
    # wheel enforces ``sched_meta.config.extra_page_block_size ==
    # extra_k_cache.shape[1]`` on every subsequent call. CSA_KV and HCA_KV
    # pools have different page_block_size → separate cache keys.
    # ``extra_at`` is the cache tag of the compressed pool
    # (``CSA_KV`` / ``HCA_KV``) or ``None`` for single-pool SWA-only.
    sched_meta_cache: Dict[Tuple[int, int, int, Optional[str], int], Any] = field(
        default_factory=dict
    )

    # opt_flash_mla: graph-stable per-request effective lengths fed to FlashMLA
    # sparse decode as ``topk_length`` / ``extra_topk_length``. Derived from the
    # same ``start_pos`` that produces ``compressed_lens``, so the kernel only
    # scans the real valid width instead of the CUDA-graph capture width (e.g.
    # HCA 8192 -> the true ``(start_pos + q_len) // 128``).
    # FlashMLA contract (flash_mla_interface.py:90): both are ``[B] int32`` and
    # bound the leftmost indices processed; ``-1`` entries inside the bound are
    # skipped (interface line 191), so the MTP ``q_len > 1`` step may safely use
    # the last token's (largest) length for the whole request.
    #   * swa_topk_length[b]                    = min(window_size, start_pos[b] + q_len)
    #   * compressed_topk_length_by_ratio[r][b] = min(extra_index_width_r,
    #                                                  (start_pos[b] + q_len) // r)
    #       (r=4 width == index_topk  -> CSA indexer top-k cap;
    #        r=128 width == HCA dense width -> no-op clamp, defensive)
    # Left at ``None`` / empty when the paged read is not wired (warmup /
    # non-paged); callers then pass ``None`` and FlashMLA keeps the legacy
    # full-capture-width behavior. See
    # ``opt_flash_mla/design/00_flash_mla_length_semantics.md``.
    swa_topk_length: Optional[torch.Tensor] = None  # [B] int32
    compressed_topk_length_by_ratio: Dict[int, torch.Tensor] = field(
        default_factory=dict
    )  # ratio -> [B] int32
    # Optional per-step RoPE rows, keyed by the source table's object identity.
    # The metadata implementation retains the source tables and output storage.
    rope_freqs_by_source: Dict[int, torch.Tensor] = field(default_factory=dict)

def get_or_build_sched_meta(
    metadata: "DSv4DecodeAttnMetadataFP8",
    *,
    batch_size: int,
    q_len: int,
    num_heads: int,
    topk: int,
    extra_attn_type: Optional[str] = None,
    extra_index_width: Optional[int] = None,
    platform_provider=None,
) -> Any:
    """Lazy-build + cache FlashMLA ``sched_meta`` on the metadata object.

    Replaces the ``SparseAttnV4DecodeFp8Op._sched_meta_cache`` anti-pattern
    (per-op instance cache accumulating across steps). Matches vLLM's
    ``DeepseekSparseSWAMetadataBuilder.build_tile_scheduler`` design —
    sched_meta lives on the per-step metadata object.

    Cache key is ``(batch_size, q_len, topk, extra_attn_type, extra_index_width)``:
      * The FlashMLA wheel bakes ``config.b`` (batch size) and
        ``config.extra_page_block_size`` (extra_k_cache page size) into the
        sched_meta on the first ``flash_mla_with_kvcache`` call and asserts
        equality on every subsequent call.
      * SWA-only (single pool) → ``extra_attn_type=None``.
      * CSA layers (dual pool, CSA_KV as extra) → ``extra_attn_type=CSA_KV``.
      * HCA layers (dual pool, HCA_KV as extra) → ``extra_attn_type=HCA_KV``.
      * CSA_KV and HCA_KV pools have different ``page_block_size``, so
        they MUST use separate sched_meta instances.

    Only the three rich-ABI buckets (SWA-only, CSA, HCA) are accepted.  A
    caller using a new geometry must add an explicit bucket instead of
    silently reusing a scheduler with incompatible tile geometry.

    CUDA-graph + opt_flash_mla (effective ``topk_length``): FlashMLA builds its
    tile schedule (``tile_scheduler_metadata`` / ``num_splits``) lazily on the
    FIRST ``flash_mla_with_kvcache`` call for a sched_meta, keyed on the
    ``topk_length`` / ``extra_topk_length`` VALUES, then freezes + reuses it
    (``csrc/api/sparse_decode.h:420``; interface docstring line 80 — reuse is
    only valid while those values stay the same). The capture harness runs two
    eager WARMUP forwards before the captured forward
    (``cuda_graph_runner.cc:880``); if the schedule is built there it is NOT
    inside the graph, so every replay reuses a schedule frozen at the warmup
    lengths -> stale -> IMA once per-request lengths vary. To keep the schedule
    in sync with the replay lengths we DROP the warmup-built sched_meta the
    moment the stream starts capturing, forcing the build kernel to be captured
    inside the graph so it re-runs (with fresh ``topk_length`` values) on every
    replay. See ``opt_flash_mla/design/01_cuda_graph_sched_meta_freeze.md``.
    """
    from rtp_llm.models_py.utils.flash_mla_legacy import (
        get_mla_metadata,  # type: ignore[import-not-found]
    )

    capturing = False
    try:
        capturing = (
            torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
        )
    except Exception:
        capturing = False
    # On the non-capturing -> capturing transition (entry into the captured
    # forward, after the warmup forwards), discard every warmup-built sched_meta
    # so the schedule build is (re)captured inside the graph.
    if capturing and not getattr(metadata, "_sched_meta_capturing", False):
        metadata.sched_meta_cache.clear()
    if getattr(metadata, "_sched_meta_capturing", False) != capturing:
        metadata._sched_meta_capturing = capturing

    key = (batch_size, extra_attn_type)
    key_factory = getattr(platform_provider, "scheduler_key", None)
    if key_factory is not None:
        key = key_factory(batch_size, q_len, topk, extra_attn_type, extra_index_width)
    sched_meta = metadata.sched_meta_cache.get(key)
    if sched_meta is None:
        sched_meta, _ = get_mla_metadata(
            cache_seqlens=None,
            num_q_tokens_per_head_k=batch_size * q_len * num_heads,
            topk=topk,
            num_heads_q=num_heads,
            num_heads_k=1,
            is_fp8_kvcache=True,
        )
        metadata.sched_meta_cache[key] = sched_meta
    return sched_meta

def allocate_decode_metadata_fp8(
    max_batch_size: int,
    q_len: int,
    window_size: int,
    head_dim: int,
    max_seq_len: int,
    compress_ratios: List[int],
    index_topk: int,
    device: torch.device,
    paged_pool_specs: Optional[Dict[str, Tuple[int, int, int]]] = None,
) -> "DSv4DecodeAttnMetadataFP8":
    """Pre-allocate a ``DSv4DecodeAttnMetadataFP8`` sized for ``max_batch_size``.

    Used by Phase 3 CUDA-graph capture: the impl owns a single metadata
    instance whose tensors live at fixed addresses; ``update_in_place``
    rewrites the contents per step. The captured graph then reads from
    these stable addresses on every replay.

    Tensor sizes are MAX, so ``update_in_place(start_pos)`` with
    ``start_pos.shape[0] == bs`` only writes the ``[:bs, ...]`` prefix.
    The captured graph at fixed BS slices these tensors at construction
    time (via ``DSv4DecodeFmhaImplFP8``); since each captured graph is
    per-BS, the slice offset is constant and known at capture.
    """
    B = max_batch_size
    T_total = B * q_len
    swa_buffer_stride = window_size

    start_pos = torch.zeros(B, dtype=torch.int32, device=device)
    position_ids = torch.zeros(T_total, dtype=torch.int32, device=device)
    position_ids_long = torch.zeros(T_total, dtype=torch.long, device=device)
    slot_swa = torch.full((T_total,), -1, dtype=torch.int32, device=device)
    topk_window = torch.full(
        (B, q_len, window_size),
        -1,
        dtype=torch.int32,
        device=device,
    )
    topk_buffer_compressed = torch.full(
        (B, q_len, index_topk),
        -1,
        dtype=torch.int32,
        device=device,
    )

    unique_ratios: List[int] = sorted({r for r in compress_ratios if r > 1})
    slot_compressed: Dict[int, torch.Tensor] = {}
    compressed_lens: Dict[int, torch.Tensor] = {}
    compressed_lens_per_token: Dict[int, torch.Tensor] = {}
    compressed_buffer_stride_per_ratio: Dict[int, int] = {}
    topk_total_by_ratio: Dict[int, torch.Tensor] = {}
    # opt_flash_mla: per-ratio FlashMLA ``extra_topk_length`` buffer (graph-stable).
    compressed_topk_length_by_ratio: Dict[int, torch.Tensor] = {}
    for r in unique_ratios:
        stride = max_seq_len // r
        compressed_buffer_stride_per_ratio[r] = stride
        slot_compressed[r] = torch.full(
            (T_total,),
            -1,
            dtype=torch.int32,
            device=device,
        )
        compressed_lens[r] = torch.zeros(B, dtype=torch.int32, device=device)
        compressed_lens_per_token[r] = torch.zeros(
            (B, q_len), dtype=torch.int32, device=device
        )
        compressed_index_width = _compressed_index_width_for_ratio(
            r, stride, index_topk
        )
        topk_total_by_ratio[r] = torch.full(
            (B, q_len, window_size + compressed_index_width),
            -1,
            dtype=torch.int32,
            device=device,
        )
        compressed_topk_length_by_ratio[r] = torch.zeros(
            B, dtype=torch.int32, device=device
        )

    # opt_flash_mla: SWA ``topk_length`` buffer (graph-stable). Filled in
    # ``update_decode_metadata_in_place_fp8`` from ``start_pos`` each step.
    swa_topk_length = torch.zeros(B, dtype=torch.int32, device=device)

    # Phase 2: paged pool block_table + write slot mapping buffers.
    # paged_pool_specs maps:
    #   cache tag → (entries_per_block, tokens_per_block, max_blocks_per_req)
    # We never reallocate after construction, so the captured graph reads
    # from these stable addresses. When a pool is absent on a layer, the
    # corresponding entries are simply unused.
    pool_block_tables: Dict[str, torch.Tensor] = {}
    pool_write_slot_mappings: Dict[str, torch.Tensor] = {}
    compressor_state_slot_mappings: Dict[str, torch.Tensor] = {}
    paged_entries_from_specs, paged_tokens_from_specs, paged_max_blocks = (
        _parse_paged_pool_specs(paged_pool_specs)
    )
    if paged_pool_specs:
        from rtp_llm.models_py.modules.dsv4.kv_cache_utils import (
            CSA_STATE,
            HCA_STATE,
            INDEXER_STATE,
            SWA_KV,
        )

        for attn_type, max_blocks in paged_max_blocks.items():
            pool_block_tables[attn_type] = torch.zeros(
                B, max_blocks, dtype=torch.int32, device=device
            )
            # SWA_KV is the always-write path (mask_negative=False downstream),
            # so it must hold a *valid* slot at all times — including during
            # framework warmup where update_decode_metadata_in_place_fp8's paged
            # branch is skipped (no block_tables yet). Slot 0 is always a
            # valid pool index; warmup writes overlap there harmlessly.
            # Compressed pools (CSA/HCA/INDEXER) use mask_negative=True and
            # require -1 sentinel for non-boundary tokens.
            sentinel = 0 if attn_type == SWA_KV else -1
            pool_write_slot_mappings[attn_type] = torch.full(
                (T_total,), sentinel, dtype=torch.int64, device=device
            )
        for attn_type in (CSA_STATE, HCA_STATE, INDEXER_STATE):
            if attn_type in paged_max_blocks:
                compressor_state_slot_mappings[attn_type] = torch.full(
                    (T_total,),
                    -1,
                    dtype=torch.int64,
                    device=device,
                )

    # Pre-allocate swa_abs_idx[B, q_len, win] int32 (always — Phase 2B-2a
    # paged read may consume it; cost is trivial compared to the rest).
    swa_abs_idx = torch.full(
        (B, q_len, window_size),
        -1,
        dtype=torch.int32,
        device=device,
    )
    # Iter3.3: pre-allocated buffers for the shared-across-layers translate
    # outputs. Under CUDA graph capture the graph bakes in the buffer
    # address, so refill MUST .copy_() into the existing storage instead of
    # reassigning ``meta.swa_global_slots = translate(...)`` — that would
    # leave captured reads pointing at stale memory. ``req_id_per_token``
    # is deterministic (``arange(bs)`` for q_len=1) so we fill it once here.
    req_id_per_token_alloc = torch.arange(B, dtype=torch.int32, device=device)
    if q_len != 1:
        req_id_per_token_alloc = req_id_per_token_alloc.repeat_interleave(
            q_len
        ).contiguous()
    req_id_per_token_long_alloc = req_id_per_token_alloc.to(torch.long)
    decode_seq_start_per_req_alloc = torch.zeros(B, dtype=torch.int64, device=device)
    decode_cu_seq_per_req_alloc = torch.arange(
        0,
        (B + 1) * q_len,
        q_len,
        dtype=torch.int64,
        device=device,
    )
    swa_global_slots_alloc = torch.full(
        (B * q_len, window_size), -1, dtype=torch.int32, device=device
    )
    hca_stride = compressed_buffer_stride_per_ratio.get(128, index_topk)
    hca_dense_width = _compressed_index_width_for_ratio(128, hca_stride, index_topk)
    hca_cmp_global_slots_alloc = torch.full(
        (B * q_len, hca_dense_width), -1, dtype=torch.int32, device=device
    )

    return DSv4DecodeAttnMetadataFP8(
        batch_size=B,
        q_len_per_req=q_len,
        total_tokens=T_total,
        window_size=window_size,
        head_dim=head_dim,
        max_seq_len=max_seq_len,
        swa_buffer_t_dim=window_size,
        compressed_buffer_t_dim_per_ratio=compressed_buffer_stride_per_ratio,
        start_pos=start_pos,
        position_ids=position_ids,
        position_ids_long=position_ids_long,
        slot_mapping_swa=slot_swa,
        slot_mapping_compressed=slot_compressed,
        topk_window_idxs=topk_window,
        topk_buffer_compressed=topk_buffer_compressed,
        compressed_lens=compressed_lens,
        compressed_lens_per_token=compressed_lens_per_token,
        topk_total_by_ratio=topk_total_by_ratio,
        compressed_offset=window_size,
        is_cuda_graph=True,
        pool_block_tables=pool_block_tables,
        paged_pool_tokens_per_block=paged_tokens_from_specs,
        pool_write_slot_mappings=pool_write_slot_mappings,
        compressor_state_slot_mappings=compressor_state_slot_mappings,
        swa_abs_idx=swa_abs_idx,
        req_id_per_token=req_id_per_token_alloc,
        req_id_per_token_long=req_id_per_token_long_alloc,
        decode_seq_start_per_req=decode_seq_start_per_req_alloc,
        decode_cu_seq_per_req=decode_cu_seq_per_req_alloc,
        swa_global_slots=swa_global_slots_alloc,
        hca_cmp_global_slots=hca_cmp_global_slots_alloc,
        swa_topk_length=swa_topk_length,
        compressed_topk_length_by_ratio=compressed_topk_length_by_ratio,
    )

def build_decode_metadata_fp8(
    attention_inputs: Any,
    q_len: int,
    window_size: int,
    head_dim: int,
    max_seq_len: int,
    compress_ratios: List[int],
    index_topk: int,
    device: torch.device,
    *,
    dtype: torch.dtype = torch.bfloat16,  # noqa: ARG001 — reserved for future
    paged_block_tables: Optional[Dict[str, torch.Tensor]] = None,
    paged_pool_entries_per_block: Optional[Dict[str, int]] = None,
    paged_pool_tokens_per_block: Optional[Dict[str, int]] = None,
) -> DSv4DecodeAttnMetadataFP8:
    """Top-level builder. Call ONCE per decode forward step.

    Args:
        attention_inputs: framework attention inputs (PyAttentionInputs) — the
            first-token position is derived from ``sequence_lengths`` for normal
            decode and from ``prefix_lengths`` for target-verify / MTP draft
            prefill. A bare ``[B] int32`` tensor is also accepted for focused
            metadata tests.
        q_len: tokens per request this step (1 for pure decode).
        window_size: SWA size (V4: 128).
        head_dim: 512 for V4 attn KV; passed through for downstream
            shape-checking.
        max_seq_len: configured max — sets the compressed-K buffer
            stride for slot computation.
        compress_ratios: per-layer ratios; we deduplicate to figure out
            which compressed pools we need slot mappings for.
        index_topk: K (V4: 512) — for pre-allocating the indexer
            output buffer.
        device: cuda device.

    Returns:
        Fully-populated DSv4DecodeAttnMetadataFP8.
    """
    start_pos = _build_start_pos_from_attention_inputs(
        attention_inputs,
        device,
        max_seq_len,
        q_len,
    )

    B = start_pos.shape[0]
    T_total = B * q_len
    position_ids_2d = _build_position_ids_2d(start_pos, q_len, device)
    position_ids_flat = position_ids_2d.reshape(-1).contiguous()
    position_ids_long = position_ids_flat.to(torch.long)

    # SWA slot mapping (always needed).
    swa_buffer_stride = window_size
    slot_swa = _build_swa_slot_mapping_from_positions(
        position_ids_2d, window_size, swa_buffer_stride
    )

    # Window topk — request-local ring positions.
    topk_window = _build_window_topk_idxs_from_positions(position_ids_2d, window_size)

    # Per-ratio compressed slot mappings.
    unique_ratios: List[int] = sorted({r for r in compress_ratios if r > 1})
    slot_compressed: Dict[int, torch.Tensor] = {}
    compressed_lens: Dict[int, torch.Tensor] = {}
    compressed_lens_per_token: Dict[int, torch.Tensor] = {}
    compressed_buffer_stride_per_ratio: Dict[int, int] = {}
    for r in unique_ratios:
        stride = max_seq_len // r
        compressed_buffer_stride_per_ratio[r] = stride
        slot_compressed[r] = _build_compressed_slot_mapping_from_positions(
            position_ids_2d, r, stride
        )
        # After-this-step length: floor((start_pos + q_len) / r)
        # (each request now has this many compressed entries).
        compressed_lens_per_token[r] = ((position_ids_2d + 1) // r).to(torch.int32)
        compressed_lens[r] = compressed_lens_per_token[r][:, -1].contiguous()

    # opt_flash_mla: FlashMLA per-request effective lengths (eager path).
    # Same formula / clamp width as ``_update_topk_lengths_in_place`` so eager
    # and CUDA-graph paths produce numerically identical lengths.
    #   swa_topk_length[b]            = min(window, last_token_pos + 1) = min(window, start_pos + q_len)
    #   compressed_topk_length[r][b]  = min(extra_index_width_r, compressed_lens[r][b])
    swa_topk_length = (
        (position_ids_2d[:, -1] + 1).clamp(min=0, max=window_size).to(torch.int32)
    )
    compressed_topk_length_by_ratio: Dict[int, torch.Tensor] = {}
    for r in unique_ratios:
        width_r = _compressed_index_width_for_ratio(
            r, compressed_buffer_stride_per_ratio[r], index_topk
        )
        compressed_topk_length_by_ratio[r] = (
            compressed_lens[r].clamp(max=int(width_r)).to(torch.int32)
        )

    # Indexer output buffer — pre-allocated; IndexerDecodeV4Op fills it.
    # Indexer is only present in compress_ratio==4 layers, so K=index_topk.
    topk_buffer_compressed = torch.full(
        (B, q_len, index_topk),
        -1,
        dtype=torch.int32,
        device=device,
    )

    # Concatenated topk per ratio.
    # The compressed-half indices live in the compressed pool, which the
    # caller stitches behind the SWA pool: compressed_offset = window_size.
    # Per-layer call site uses these indices directly against the per-request
    # KV view ``buffer[r, : ]``.  IndexerDecodeV4Op writes into the slice
    # ``topk_total_by_ratio[ratio][:, :, window_size:]`` for ratio==4.
    topk_total_by_ratio: Dict[int, torch.Tensor] = {}
    for r in unique_ratios:
        compressed_index_width = _compressed_index_width_for_ratio(
            r, compressed_buffer_stride_per_ratio[r], index_topk
        )
        total = torch.full(
            (B, q_len, window_size + compressed_index_width),
            -1,
            dtype=torch.int32,
            device=device,
        )
        # Window half: pre-fill from window topk (request-local ring slots).
        total[:, :, :window_size] = topk_window
        # The compressed half is filled per-call by indexer (CSA r=4) or by
        # _build_dense_compressed_idxs (HCA r=128). For HCA r=128 we fill
        # here directly since it's deterministic (read all valid compressed entries).
        if r != 4:
            # Dense compressed read: indices [0..compressed_lens[r]) per request,
            # right-padded with -1. HCA must cover the full compressed pool
            # width (max_seq_len / 128); using index_topk would truncate long
            # contexts such as 1M tokens to only 512 compressed entries.
            K_dense = compressed_index_width
            dense_idxs = (
                torch.arange(K_dense, device=device, dtype=torch.int32)
                .view(
                    1,
                    1,
                    K_dense,
                )
                .expand(B, q_len, K_dense)
            )
            cmp_lens_per_token = compressed_lens_per_token[r].view(B, q_len, 1)
            valid = dense_idxs < cmp_lens_per_token
            # Indices are *request-local*: position within the compressed pool
            # (i.e. 0..max_seq_len/ratio).
            total[:, :, window_size:] = torch.where(
                valid,
                dense_idxs,
                torch.full_like(dense_idxs, -1),
            )
        # else (r==4): compressed half stays -1 sentinel until IndexerDecodeV4Op fills it.
        topk_total_by_ratio[r] = total

    # Phase 2 paged eager-path metadata (allocated fresh per step — eager
    # path doesn't share buffers across steps, unlike DSv4DecodeFmhaImplFP8).
    pool_block_tables: Dict[str, torch.Tensor] = {}
    pool_write_slot_mappings: Dict[str, torch.Tensor] = {}
    compressor_state_slot_mappings: Dict[str, torch.Tensor] = {}
    paged_pool_tokens_per_block_resolved: Dict[str, int] = {}
    req_id_per_token: Optional[torch.Tensor] = None
    req_id_per_token_long: Optional[torch.Tensor] = None
    decode_seq_start_per_req: Optional[torch.Tensor] = None
    decode_cu_seq_per_req: Optional[torch.Tensor] = None
    swa_global_slots: Optional[torch.Tensor] = None
    hca_cmp_global_slots: Optional[torch.Tensor] = None
    if paged_block_tables and paged_pool_entries_per_block:
        from rtp_llm.models_py.modules.dsv4.fp8.decode.pool_slot_mapping import (
            compute_kv_pool_slot_mapping,
        )
        from rtp_llm.models_py.modules.dsv4.kv_cache_utils import (
            CSA_KV,
            HCA_KV,
            INDEXER_KV,
            SWA_KV,
        )

        # Snapshot block tables (clone so downstream writes can't surprise
        # the framework's tensor).
        for at, bt in paged_block_tables.items():
            pool_block_tables[at] = bt[:B].contiguous().clone()
        paged_pool_tokens_per_block_resolved = _resolve_paged_pool_tokens_per_block(
            paged_pool_entries_per_block,
            paged_pool_tokens_per_block,
        )

        if SWA_KV in pool_block_tables:
            E = paged_pool_entries_per_block[SWA_KV]
            swa_tokens_per_block = paged_pool_tokens_per_block_resolved[SWA_KV]
            pool_write_slot_mappings[SWA_KV] = compute_kv_pool_slot_mapping(
                pool_block_tables[SWA_KV],
                position_ids_flat,
                pool_entries_per_block=E,
                pool_tokens_per_block=swa_tokens_per_block,
                ring_entries=E,
            )

        for ratio_key, attn_type_writers in (
            (4, [CSA_KV, INDEXER_KV]),
            (128, [HCA_KV]),
        ):
            if ratio_key not in compressed_lens:
                continue
            abs_pos_plus_1 = position_ids_2d + 1
            on_boundary = (abs_pos_plus_1 % ratio_key) == 0
            cmp_idx = abs_pos_plus_1 // ratio_key - 1
            cmp_idx_with_skip = torch.where(
                on_boundary,
                cmp_idx,
                torch.full_like(cmp_idx, -1),
            ).reshape(-1)
            for at in attn_type_writers:
                if at not in pool_block_tables:
                    continue
                E = paged_pool_entries_per_block[at]
                tokens_per_block = _compressed_domain_tokens_per_block(
                    at, paged_pool_tokens_per_block_resolved[at]
                )
                pool_write_slot_mappings[at] = compute_kv_pool_slot_mapping(
                    pool_block_tables[at],
                    cmp_idx_with_skip,
                    pool_entries_per_block=E,
                    pool_tokens_per_block=tokens_per_block,
                    ring_entries=E,
                )

    # Phase 2B-2a: SWA absolute-position window (paged read).
    abs_pos_eager = position_ids_2d
    win_range = torch.arange(window_size, device=device, dtype=torch.int32).view(
        1,
        1,
        window_size,
    )
    win_start = (abs_pos_eager.unsqueeze(-1) - window_size + 1).clamp(min=0)
    candidate = win_start + win_range
    swa_abs_idx = torch.where(
        candidate <= abs_pos_eager.unsqueeze(-1),
        candidate,
        torch.full_like(candidate, -1),
    )
    if paged_block_tables and paged_pool_entries_per_block:
        from rtp_llm.models_py.modules.dsv4.fp8.decode.paged_topk_translator import (
            build_req_id_per_token,
            translate_local_to_global_slots,
        )
        from rtp_llm.models_py.modules.dsv4.kv_cache_utils import HCA_KV, SWA_KV

        req_id_per_token = build_req_id_per_token(int(B), q_len, device)
        req_id_per_token_long = req_id_per_token.to(torch.long)
        decode_seq_start_per_req = start_pos.to(torch.int64).contiguous()
        decode_cu_seq_per_req = torch.arange(
            0,
            (int(B) + 1) * q_len,
            q_len,
            dtype=torch.int64,
            device=device,
        )

        from rtp_llm.models_py.modules.dsv4.kv_cache_utils import (
            CSA_STATE,
            HCA_STATE,
            INDEXER_STATE,
        )

        for at in (CSA_STATE, HCA_STATE, INDEXER_STATE):
            if at not in pool_block_tables:
                continue
            entries_per_block = paged_pool_entries_per_block[at]
            compressor_state_slot_mappings[at] = _compute_state_pool_slot_mapping(
                pool_block_tables[at],
                position_ids_long,
                req_id_per_token_long,
                entries_per_block,
                paged_pool_tokens_per_block_resolved[at],
            )

        if SWA_KV in pool_block_tables:
            swa_eb = paged_pool_entries_per_block[SWA_KV]
            swa_tokens_per_block = paged_pool_tokens_per_block_resolved[SWA_KV]
            swa_global_slots = translate_local_to_global_slots(
                req_id_per_token,
                pool_block_tables[SWA_KV],
                swa_abs_idx.reshape(T_total, window_size),
                entries_per_block=swa_eb,
                tokens_per_block_for_block_table=swa_tokens_per_block,
            )

        if HCA_KV in pool_block_tables and 128 in topk_total_by_ratio:
            hca_eb = paged_pool_entries_per_block[HCA_KV]
            hca_tokens_per_block = _compressed_domain_tokens_per_block(
                HCA_KV, paged_pool_tokens_per_block_resolved[HCA_KV]
            )
            hca_tt = topk_total_by_ratio[128]
            K_h = hca_tt.shape[-1] - window_size
            hca_cmp_global_slots = translate_local_to_global_slots(
                req_id_per_token,
                pool_block_tables[HCA_KV],
                hca_tt[:, :, window_size:].reshape(T_total, K_h).contiguous(),
                entries_per_block=hca_eb,
                tokens_per_block_for_block_table=hca_tokens_per_block,
            )

    return DSv4DecodeAttnMetadataFP8(
        batch_size=B,
        q_len_per_req=q_len,
        total_tokens=T_total,
        window_size=window_size,
        head_dim=head_dim,
        max_seq_len=max_seq_len,
        swa_buffer_t_dim=window_size,
        compressed_buffer_t_dim_per_ratio=compressed_buffer_stride_per_ratio,
        start_pos=start_pos,
        position_ids=position_ids_flat,
        position_ids_long=position_ids_long,
        slot_mapping_swa=slot_swa,
        slot_mapping_compressed=slot_compressed,
        topk_window_idxs=topk_window,
        topk_buffer_compressed=topk_buffer_compressed,
        compressed_lens=compressed_lens,
        compressed_lens_per_token=compressed_lens_per_token,
        topk_total_by_ratio=topk_total_by_ratio,
        compressed_offset=window_size,
        is_cuda_graph=False,
        pool_block_tables=pool_block_tables,
        paged_pool_tokens_per_block=paged_pool_tokens_per_block_resolved,
        pool_write_slot_mappings=pool_write_slot_mappings,
        compressor_state_slot_mappings=compressor_state_slot_mappings,
        swa_abs_idx=swa_abs_idx,
        req_id_per_token=req_id_per_token,
        req_id_per_token_long=req_id_per_token_long,
        decode_seq_start_per_req=decode_seq_start_per_req,
        decode_cu_seq_per_req=decode_cu_seq_per_req,
        swa_global_slots=swa_global_slots,
        hca_cmp_global_slots=hca_cmp_global_slots,
        swa_topk_length=swa_topk_length,
        compressed_topk_length_by_ratio=compressed_topk_length_by_ratio,
    )
