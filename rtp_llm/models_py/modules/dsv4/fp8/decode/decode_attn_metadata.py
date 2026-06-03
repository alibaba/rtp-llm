"""Per-step decode attention metadata for DeepSeek-V4.

For batched decode (B requests, q_len=1 each, may be 1+spec later), we
compute once per forward step:

  * ``slot_mapping_swa[T_total]`` int32 — for each generated token,
    the absolute SWA ring-buffer slot to write its K into. Slot is
    flat across batch (i.e. per-request offset baked in: see below).
  * ``slot_mapping_compressed[T_total]`` int32 — same idea for the
    compressed-K buffer (only used by layers with ``compress_ratio>0``;
    SWA-only layers ignore it).
  * ``topk_window_idxs[B, S, win]`` int32 — sliding-window topk
    (the "trivial" topk: read the last ``win`` tokens). Per-request
    cyclic ring offsets baked in.
  * ``topk_buffer_compressed[B, S, K_compressed]`` int32 — pre-allocated
    output buffer for the indexer. Filled in by IndexerDecodeV4Op
    layer-by-layer (each compress=4 layer overwrites; layer compute is
    sequential in Python, so a single buffer is fine).
  * ``compressed_lens[B]`` int32 — number of compressed-K entries each
    request has accumulated (used to mask invalid topk slots in CSA
    layers).

This builder is stateless and graph-friendly: every output tensor lives
on cuda, computed via vectorized torch ops, no Python-per-token loops.

KV layout (during Phase 1-3):
  Each request ``r`` owns a slot range in the per-layer register_buffer:
    SWA          : ``buffer[r, 0 : win]``         — ring of last ``win`` K
    Compressed-K : ``buffer[r, win : win + max_seq_len // ratio]``
  Per-step we write 1 token to:
    SWA          : ``buffer[r, start_pos[r] % win]``
    Compressed-K : iff ``(start_pos[r]+1) % ratio == 0``,
                   ``buffer[r, win + (start_pos[r]+1) // ratio - 1]``

The 2-D buffer is flattened to a 1-D slot-major view at op call time —
the slot indices we produce here are ``r * stride + offset_in_request``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch


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

    # Per-attn_type framework block_table: [max_B, max_blocks_per_req] int32.
    # Source: ``attn_inputs.kv_cache_kernel_block_id_device_by_group[gid]``.
    # Keys are attn_type ids (1=CSA_KV..7=SWA_KV) from
    # :mod:`rtp_llm.models_py.modules.dsv4.attn_type`; only pools that the
    # model actually uses are present.
    pool_block_tables: Dict[int, torch.Tensor] = field(default_factory=dict)

    # Per-attn_type raw-token coverage for one block_table row. For the
    # seq=16384/kernel=128 layout, FULL paged pools carry 128 here while
    # SWA_KV carries 16384. Compressed writers convert this to their
    # compressed-index domain by dividing by the compression ratio.
    paged_pool_tokens_per_block: Dict[int, int] = field(default_factory=dict)

    # Per-attn_type new-token write slot mapping: [max_T_total] int64.
    # ``slot[t] = block_id * entries_per_block + in_block``. Block-table row
    # tokens and in-block ring entries may differ for SWA_KV; compressed
    # writers map in the compressed-index domain. ``-1`` = skip.
    pool_write_slot_mappings: Dict[int, torch.Tensor] = field(default_factory=dict)

    # Per-attn_type compressor state-pool slot mapping: [max_T_total] int64.
    # These are pure metadata for CompressorFP8.launch, identical across
    # same-type layers in a decode step. Building them here lets decode pass
    # a ready CompressorMeta down to every CSA/HCA/indexer compressor instead
    # of rebuilding state_slots / kv_slots / token_to_req inside each layer.
    compressor_state_slot_mappings: Dict[int, torch.Tensor] = field(
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
    # views via ``self._kv_cache.get_layer_cache(layer_id, attn_type)``
    # at call time, no per-layer descriptor cache needed.

    # Iter3.3: shared-across-layers request-id mapping ``[T]`` int32 = the
    # ``arange(bsz)`` (for q_len=1) passed to every ``translate_local_to_global_slots``
    # call. 43 layers × same arange = one cached tensor.
    req_id_per_token: Optional[torch.Tensor] = None
    # Int64 companion consumed by ``CompressorFP8.prepare_metadata``.
    req_id_per_token_long: Optional[torch.Tensor] = None
    # Decode per-request first position and compact cu-seq offsets used by
    # compressor raw-path metadata for q_len > 1. These are layer-invariant.
    decode_seq_start_per_req: Optional[torch.Tensor] = None  # [B] int32
    decode_cu_seq_per_req: Optional[torch.Tensor] = None  # [B + 1] int32

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

    # FlashMLA ``sched_meta`` cache — per-(batch_size, extra_attn_type).
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
    # ``extra_at`` is the ``attn_type`` int for the compressed pool
    # (``CSA_KV`` / ``HCA_KV``) or ``None`` for single-pool SWA-only.
    sched_meta_cache: Dict[Tuple[int, Optional[int]], Any] = field(default_factory=dict)

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


def get_or_build_sched_meta(
    metadata: "DSv4DecodeAttnMetadataFP8",
    *,
    batch_size: int,
    q_len: int,
    num_heads: int,
    topk: int,
    extra_attn_type: Optional[int] = None,
) -> Any:
    """Lazy-build + cache FlashMLA ``sched_meta`` on the metadata object.

    Replaces the ``SparseAttnV4DecodeFp8Op._sched_meta_cache`` anti-pattern
    (per-op instance cache accumulating across steps). Matches vLLM's
    ``DeepseekSparseSWAMetadataBuilder.build_tile_scheduler`` design —
    sched_meta lives on the per-step metadata object.

    Cache key is ``(batch_size, extra_attn_type)``:
      * The FlashMLA wheel bakes ``config.b`` (batch size) and
        ``config.extra_page_block_size`` (extra_k_cache page size) into the
        sched_meta on the first ``flash_mla_with_kvcache`` call and asserts
        equality on every subsequent call.
      * SWA-only (single pool) → ``extra_attn_type=None``.
      * CSA layers (dual pool, CSA_KV as extra) → ``extra_attn_type=CSA_KV``.
      * HCA layers (dual pool, HCA_KV as extra) → ``extra_attn_type=HCA_KV``.
      * CSA_KV and HCA_KV pools have different ``page_block_size``, so
        they MUST use separate sched_meta instances.

    Within one (B, extra_attn_type) bucket ``q_len / num_heads / topk`` are
    process-constants.

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
    from flash_mla import get_mla_metadata  # type: ignore[import-not-found]

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


def _pool_compress_ratio(attn_type: int) -> int:
    from rtp_llm.models_py.modules.dsv4.attn_type import CSA_KV, HCA_KV, INDEXER_KV

    if int(attn_type) in (int(CSA_KV), int(INDEXER_KV)):
        return 4
    if int(attn_type) == int(HCA_KV):
        return 128
    return 1


def _parse_paged_pool_specs(
    paged_pool_specs: Optional[Dict[int, Tuple[int, int, int]]],
) -> Tuple[Dict[int, int], Dict[int, int], Dict[int, int]]:
    """Parse ``attn_type -> (entries, raw_tokens_per_block, max_blocks)``."""
    entries_by_pool: Dict[int, int] = {}
    tokens_by_pool: Dict[int, int] = {}
    max_blocks_by_pool: Dict[int, int] = {}
    if not paged_pool_specs:
        return entries_by_pool, tokens_by_pool, max_blocks_by_pool

    for attn_type, spec in paged_pool_specs.items():
        values = tuple(int(v) for v in spec)
        if len(values) != 3:
            raise ValueError(
                "paged_pool_specs values must be "
                "(entries_per_block, tokens_per_block, max_blocks_per_req), "
                f"got attn_type={attn_type}, spec={spec!r}"
            )
        entries_per_block, tokens_per_block, max_blocks = values
        if entries_per_block <= 0 or tokens_per_block <= 0 or max_blocks <= 0:
            raise ValueError(
                "paged_pool_specs values must be positive, "
                f"got attn_type={attn_type}, spec={spec!r}"
            )
        entries_by_pool[int(attn_type)] = entries_per_block
        tokens_by_pool[int(attn_type)] = tokens_per_block
        max_blocks_by_pool[int(attn_type)] = max_blocks
    return entries_by_pool, tokens_by_pool, max_blocks_by_pool


def _resolve_paged_pool_tokens_per_block(
    entries_by_pool: Dict[int, int],
    tokens_by_pool: Optional[Dict[int, int]],
) -> Dict[int, int]:
    resolved: Dict[int, int] = {}
    if tokens_by_pool is None:
        raise ValueError("paged_pool_tokens_per_block is required for paged pools")
    for attn_type in entries_by_pool:
        if attn_type not in tokens_by_pool:
            raise ValueError(
                "paged_pool_tokens_per_block missing attn_type=%s" % (attn_type,)
            )
        tokens_per_block = int(tokens_by_pool[attn_type])
        if tokens_per_block <= 0:
            raise ValueError(
                "paged pool tokens_per_block must be positive, "
                f"got attn_type={attn_type}, tokens_per_block={tokens_per_block}"
            )
        resolved[int(attn_type)] = tokens_per_block
    return resolved


def _compressed_domain_tokens_per_block(
    attn_type: int,
    raw_tokens_per_block: int,
) -> int:
    ratio = _pool_compress_ratio(attn_type)
    if ratio <= 1:
        return int(raw_tokens_per_block)
    if int(raw_tokens_per_block) % ratio != 0:
        raise ValueError(
            "compressed pool raw tokens_per_block must be divisible by "
            f"compress ratio, got attn_type={attn_type}, "
            f"tokens_per_block={raw_tokens_per_block}, ratio={ratio}"
        )
    return int(raw_tokens_per_block) // ratio


def _compute_state_pool_slot_mapping(
    block_table: torch.Tensor,
    positions: torch.Tensor,
    req_idx: torch.Tensor,
    entries_per_block: int,
    tokens_per_block: int,
) -> torch.Tensor:
    """Match CompressorFP8._compute_state_slot_mapping.

    State pools are cyclic per request: block index is
    ``(pos // tokens_per_block) % max_blocks``.  Block id <= 0 is the
    unallocated sentinel and maps to -1.

    ``tokens_per_block``: physical block size in tokens for block_table
    indexing. The ring in-block offset always uses
    ``entries_per_block`` (= R).
    """
    if positions.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=positions.device)
    tpb = tokens_per_block
    pos_i64 = positions.to(torch.long)
    req_i64 = req_idx.to(torch.long)
    bt_long = block_table.to(torch.long)
    max_blocks = int(bt_long.shape[1])
    block_in_seq = (pos_i64 // tpb) % max_blocks
    in_block = pos_i64 % entries_per_block
    block_id = bt_long[req_i64, block_in_seq]
    slot = block_id * entries_per_block + in_block
    return torch.where(block_id > 0, slot, torch.full_like(slot, -1))


def _update_compressor_state_slot_mappings(
    meta: "DSv4DecodeAttnMetadataFP8",
    bs: int,
    paged_pool_entries_per_block: Dict[int, int],
) -> None:
    if not meta.compressor_state_slot_mappings:
        return
    if meta.position_ids_long is None or meta.req_id_per_token_long is None:
        return

    T = bs * meta.q_len_per_req
    positions = meta.position_ids_long[:T]
    req_idx = meta.req_id_per_token_long[:T]
    for at, out in meta.compressor_state_slot_mappings.items():
        if at not in meta.pool_block_tables:
            continue
        entries_per_block = paged_pool_entries_per_block[at]
        tokens_per_block = meta.paged_pool_tokens_per_block[at]
        mapped = _compute_state_pool_slot_mapping(
            meta.pool_block_tables[at][:bs],
            positions,
            req_idx,
            entries_per_block,
            tokens_per_block=tokens_per_block,
        )
        out[:T].copy_(mapped)


def _build_swa_slot_mapping(
    start_pos: torch.Tensor, q_len: int, window_size: int, swa_buffer_stride: int
) -> torch.Tensor:
    """For each (req r, q-index s) compute the flat slot in the SWA
    register_buffer. SWA buffer is per-request ``[window_size, head_dim]``
    flattened: slot = r * swa_buffer_stride + ((start_pos[r] + s) % window_size).

    Args:
        start_pos: [B] int32, host or device.
        q_len: scalar (1 for pure decode, 1+spec for MTP).
        window_size: SWA window (128).
        swa_buffer_stride: row stride of flattened buffer = window_size.

    Returns:
        [B * q_len] int32 device tensor.
    """
    B = start_pos.shape[0]
    device = start_pos.device
    if start_pos.dtype != torch.int32:
        start_pos = start_pos.to(torch.int32)
    # offsets [B, q_len] = start_pos[:, None] + arange(q_len)[None, :]
    s_offsets = start_pos.view(B, 1) + torch.arange(
        q_len, device=device, dtype=torch.int32
    ).view(1, q_len)
    # ring offset within window
    in_ring = s_offsets % window_size
    # flat slot
    req_base = (
        torch.arange(B, device=device, dtype=torch.int32).view(B, 1) * swa_buffer_stride
    )
    return (req_base + in_ring).reshape(-1)


def _build_compressed_slot_mapping(
    start_pos: torch.Tensor, q_len: int, ratio: int, compressed_buffer_stride: int
) -> torch.Tensor:
    """For each (req r, q-index s) where the absolute position
    ``start_pos[r] + s`` is on a compression boundary
    (i.e. ``(start_pos[r] + s + 1) % ratio == 0``), compute the slot
    in the compressed-K buffer; else -1.

    Compressed slot index within request: ``(start_pos[r] + s + 1) // ratio - 1``.

    Args:
        start_pos: [B] int32.
        q_len: scalar.
        ratio: 4 (CSA) or 128 (HCA).
        compressed_buffer_stride: row stride of flattened buffer
            = ``max_seq_len // ratio``.

    Returns:
        [B * q_len] int32 device tensor (-1 means skip).
    """
    B = start_pos.shape[0]
    device = start_pos.device
    if start_pos.dtype != torch.int32:
        start_pos = start_pos.to(torch.int32)
    abs_pos_plus_1 = (
        start_pos.view(B, 1)
        + torch.arange(q_len, device=device, dtype=torch.int32).view(1, q_len)
        + 1
    )  # [B, q_len]
    on_boundary = (abs_pos_plus_1 % ratio) == 0  # [B, q_len] bool
    in_req = abs_pos_plus_1 // ratio - 1  # [B, q_len] int32
    req_base = (
        torch.arange(B, device=device, dtype=torch.int32).view(B, 1)
        * compressed_buffer_stride
    )
    flat = (req_base + in_req).reshape(-1)
    mask = on_boundary.reshape(-1)
    return torch.where(mask, flat, torch.full_like(flat, -1))


def _build_window_topk_idxs(
    start_pos: torch.Tensor, q_len: int, window_size: int
) -> torch.Tensor:
    """For each (req, q-index) build the int32 [win] topk pointing at
    the request-local ring positions to read.

    For abs_pos = start_pos[r] + s:
      * If abs_pos == 0: read [0, -1, -1, ...] (only token 0 valid).
      * If abs_pos < window_size: ascending from 0 to abs_pos, padding
        the rest with -1.
      * Else: cyclic — start at (abs_pos+1) % win, wrap once around.

    Returns the indices in the SAME flat slot-space the kernel reads,
    so we use *request-local* ring positions [0, win) — the kernel
    indexes into the per-request kv slice.

    Args:
        start_pos: [B] int32.
        q_len: scalar.
        window_size: SWA window.

    Returns:
        [B, q_len, window_size] int32.
    """
    B = start_pos.shape[0]
    device = start_pos.device
    if start_pos.dtype != torch.int32:
        start_pos = start_pos.to(torch.int32)
    # abs_pos[r, s] = start_pos[r] + s
    abs_pos = start_pos.view(B, 1) + torch.arange(
        q_len, device=device, dtype=torch.int32
    ).view(
        1, q_len
    )  # [B, q_len]
    # LEFT-aligned: valid entries first, -1 padding at the tail.
    # Full ring (abs_pos >= win-1): cyclic read oldest→newest starting at (sp+1)%win.
    # Partial ring (abs_pos < win-1): ascending [0..abs_pos], then -1 padding.
    k_range = torch.arange(window_size, device=device, dtype=torch.int32).view(
        1, 1, window_size
    )
    abs_pos_b = abs_pos.unsqueeze(-1)  # [B, q_len, 1]
    sp = (abs_pos % window_size).unsqueeze(-1)  # [B, q_len, 1] ring write position
    # Full ring: k-th entry = (sp + 1 + k) % win  (oldest slot first)
    ring_full_idx = (sp + 1 + k_range) % window_size
    # Partial ring: k-th entry = k if k <= abs_pos, else -1
    partial_idx = torch.where(
        k_range <= abs_pos_b, k_range, torch.full_like(k_range, -1)
    )
    is_full = abs_pos_b >= (window_size - 1)
    return torch.where(is_full, ring_full_idx, partial_idx)


def _build_position_ids_2d(
    start_pos: torch.Tensor,
    q_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Return absolute decode positions as ``[B, q_len]`` int32.

    The public metadata contract stores positions flat ``[T]``. Internally
    the slot/topk math needs a temporary request-major view. The framework
    position_ids field is intentionally not consumed here; DSv4 decode builds
    its own contiguous per-request position stream from the first token
    position and q_len.

    Caller is expected to have already moved ``start_pos`` to ``device`` /
    ``int32`` (see :func:`_build_start_pos_from_attention_inputs`); the
    ``.to`` here is a no-op in that case.
    """
    start_pos = start_pos.to(device=device, dtype=torch.int32)
    B = int(start_pos.shape[0])
    return start_pos.view(B, 1) + torch.arange(
        q_len, device=device, dtype=torch.int32
    ).view(1, q_len)


def _build_start_pos_from_attention_inputs(
    attention_inputs: Any,
    device: torch.device,
    max_seq_len: int,
    q_len: int,
) -> torch.Tensor:
    """Derive DSv4 decode first-token positions from framework attention inputs."""
    if isinstance(attention_inputs, torch.Tensor):
        start_pos = attention_inputs
    else:
        is_target_verify = bool(getattr(attention_inputs, "is_target_verify", False))
        is_prefill = bool(getattr(attention_inputs, "is_prefill", False))
        # Target verify and MTP draft-prefill CUDA graph are both multi-token
        # decode-shaped batches.  CudaGraphRunner copies prefix_lengths for
        # prefill graph replay but leaves sequence_lengths at capture-time
        # sentinel values, so prefix_lengths is the only valid first-token
        # position source for those q_len > 1 paths.
        if is_target_verify or (is_prefill and q_len > 1):
            start_pos = attention_inputs.prefix_lengths
        else:
            start_pos = attention_inputs.sequence_lengths

    start_pos = start_pos.to(device=device, dtype=torch.int32)
    # Cuda-graph capture/warmup can hand us sentinel prefix lengths near
    # max_seq_len. For multi-token speculative batches, clamping only the first
    # token to max_seq_len - 1 still lets later tokens exceed the sized KV and
    # compressed-pool capacity, e.g. q_len=4 with a ratio-4 boundary in the
    # tail. Clamp the first token so the whole [B, q_len] position window fits.
    max_start = max(0, int(max_seq_len) - int(q_len))
    return torch.clamp(start_pos, min=0, max=max_start)


def _build_swa_slot_mapping_from_positions(
    position_ids_2d: torch.Tensor, window_size: int, swa_buffer_stride: int
) -> torch.Tensor:
    B = int(position_ids_2d.shape[0])
    device = position_ids_2d.device
    in_ring = position_ids_2d % window_size
    req_base = (
        torch.arange(B, device=device, dtype=torch.int32).view(B, 1) * swa_buffer_stride
    )
    return (req_base + in_ring).reshape(-1)


def _build_compressed_slot_mapping_from_positions(
    position_ids_2d: torch.Tensor, ratio: int, compressed_buffer_stride: int
) -> torch.Tensor:
    B = int(position_ids_2d.shape[0])
    device = position_ids_2d.device
    abs_pos_plus_1 = position_ids_2d + 1
    on_boundary = (abs_pos_plus_1 % ratio) == 0
    in_req = abs_pos_plus_1 // ratio - 1
    req_base = (
        torch.arange(B, device=device, dtype=torch.int32).view(B, 1)
        * compressed_buffer_stride
    )
    flat = (req_base + in_req).reshape(-1)
    mask = on_boundary.reshape(-1)
    return torch.where(mask, flat, torch.full_like(flat, -1))


def _build_window_topk_idxs_from_positions(
    position_ids_2d: torch.Tensor, window_size: int
) -> torch.Tensor:
    device = position_ids_2d.device
    k_range = torch.arange(window_size, device=device, dtype=torch.int32).view(
        1, 1, window_size
    )
    abs_pos_b = position_ids_2d.unsqueeze(-1)
    sp = (position_ids_2d % window_size).unsqueeze(-1)
    ring_full_idx = (sp + 1 + k_range) % window_size
    partial_idx = torch.where(
        k_range <= abs_pos_b, k_range, torch.full_like(k_range, -1)
    )
    is_full = abs_pos_b >= (window_size - 1)
    return torch.where(is_full, ring_full_idx, partial_idx)


def _compressed_index_width_for_ratio(
    ratio: int, compressed_stride: int, index_topk: int
) -> int:
    """Compressed-index tensor width used by decode attention for one ratio."""
    if int(ratio) == 4:
        return int(index_topk)
    # FlashMLA sparse decode requires topk/extra_topk widths to be multiples
    # of 64. HCA still reads dense compressed indices; the extra slots are
    # right-padded with -1 and do not represent selected top-k tokens.
    return ((int(compressed_stride) + 63) // 64) * 64


def _update_topk_lengths_in_place(
    meta: "DSv4DecodeAttnMetadataFP8",
    start_pos: torch.Tensor,
    bs: int,
    full_width: bool = False,
) -> None:
    """opt_flash_mla: fill the FlashMLA effective-length buffers in place.

    Reuses the already-computed ``compressed_lens`` (written by
    ``fused_update_decode_meta_pure`` just above the call site) so there is a
    single source of truth shared with the eager builder
    (:func:`build_decode_metadata_fp8`).

    Graph-safe: every write is an in-place ``.copy_()`` into a pre-allocated
    buffer (never rebinds the attribute), so a captured CUDA graph keeps reading
    from the same address across replays.

      * ``swa_topk_length[b]            = min(window_size, start_pos[b] + q_len)``
      * ``compressed_topk_length_by_ratio[r][b] =
            min(extra_index_width_r, compressed_lens[r][b])``
        where ``extra_index_width_r = topk_total_by_ratio[r].shape[-1] - window``
        (== ``index_topk`` for r=4, == HCA dense width for r=128).

    ``full_width`` (CUDA-graph capture/warmup only): write the **maximum**
    length (``window_size`` for SWA, ``extra_index_width_r`` for compressed)
    instead of the seq-len-derived value. This is REQUIRED for graph capture.

    FlashMLA builds its ``tile_scheduler_metadata`` / ``num_splits`` lazily on
    the FIRST ``flash_mla_with_kvcache`` call (sparse_decode.h:420) keyed on the
    ``topk_length`` VALUES, then freezes + reuses it. The capture harness
    (cuda_graph_runner.cc:880-881) runs two WARMUP forwards before the captured
    forward, so the schedule is built during warmup and is NOT re-captured into
    the graph; every replay reuses that frozen schedule. If the schedule is
    built for short warmup lengths, a longer real replay length over-runs it
    (IMA). Writing the full width here makes the frozen schedule identical to the
    ``topk_length is None`` (baseline) schedule — a valid superset for any real
    replay length. The captured kernel still reads the per-replay REAL lengths
    (written by ``prepare_cuda_graph`` -> ``full_width=False``) for per-query
    masking, so the compute saving is preserved. See
    ``opt_flash_mla/design/01_cuda_graph_sched_meta_freeze.md``.
    """
    q_len = int(meta.q_len_per_req)
    window_size = int(meta.window_size)
    if meta.swa_topk_length is not None:
        if full_width:
            meta.swa_topk_length[:bs].fill_(window_size)
        else:
            swa_len = (start_pos.to(torch.int32) + q_len).clamp_(min=0, max=window_size)
            meta.swa_topk_length[:bs].copy_(swa_len)
    for r, out in meta.compressed_topk_length_by_ratio.items():
        tt = meta.topk_total_by_ratio.get(r)
        width_r = int(tt.shape[-1]) - window_size if tt is not None else None
        if full_width:
            if width_r is not None:
                out[:bs].fill_(width_r)
            continue
        cmp_len = meta.compressed_lens.get(r)
        if cmp_len is None:
            continue
        eff = cmp_len[:bs]
        if width_r is not None:
            eff = eff.clamp(max=width_r)
        out[:bs].copy_(eff)


def allocate_decode_metadata_fp8(
    max_batch_size: int,
    q_len: int,
    window_size: int,
    head_dim: int,
    max_seq_len: int,
    compress_ratios: List[int],
    index_topk: int,
    device: torch.device,
    paged_pool_specs: Optional[Dict[int, Tuple[int, int, int]]] = None,
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
    #   attn_type → (entries_per_block, tokens_per_block, max_blocks_per_req)
    # We never reallocate after construction, so the captured graph reads
    # from these stable addresses. When a pool is absent on a layer, the
    # corresponding entries are simply unused.
    pool_block_tables: Dict[int, torch.Tensor] = {}
    pool_write_slot_mappings: Dict[int, torch.Tensor] = {}
    compressor_state_slot_mappings: Dict[int, torch.Tensor] = {}
    paged_entries_from_specs, paged_tokens_from_specs, paged_max_blocks = (
        _parse_paged_pool_specs(paged_pool_specs)
    )
    if paged_pool_specs:
        from rtp_llm.models_py.modules.dsv4.attn_type import (
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
    decode_seq_start_per_req_alloc = torch.zeros(B, dtype=torch.int32, device=device)
    decode_cu_seq_per_req_alloc = torch.arange(
        0,
        (B + 1) * q_len,
        q_len,
        dtype=torch.int32,
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


def update_decode_metadata_in_place_fp8(
    meta: "DSv4DecodeAttnMetadataFP8",
    attention_inputs: Any,
    *,
    forbid_realloc: bool = False,
    paged_block_tables: Optional[Dict[int, torch.Tensor]] = None,
    paged_pool_entries_per_block: Optional[Dict[int, int]] = None,
    paged_pool_tokens_per_block: Optional[Dict[int, int]] = None,
    capture_full_width_lengths: bool = False,
) -> None:
    """Recompute every metadata buffer IN PLACE for new attention inputs.

    Contract:
      * Every output tensor reuses its prior storage (``data_ptr()``
        unchanged); ``forbid_realloc=True`` makes any accidental realloc
        an immediate error rather than silent-correctness-bug.
      * Writes the prefix ``[:bs]`` (or ``[:bs * q_len]`` for slot
        mappings); leaves the tail at sentinel (-1) from
        ``allocate_decode_metadata_fp8``. The captured CUDA graph at that
        BS only reads the prefix, so the tail never matters.

    Mirrors :func:`build_decode_metadata_fp8` arithmetic. Kept as a separate
    function (not method) so it's callable from ``DSv4DecodeFmhaImplFP8.prepare``
    and from unit tests without instantiating the impl.

    Args:
        meta: Pre-allocated metadata (from ``allocate_decode_metadata_fp8``).
        attention_inputs: Framework attention inputs. Normal decode derives
            the first-token position from ``sequence_lengths``; target verify
            derives it from ``prefix_lengths`` because C++ clears
            ``sequence_lengths`` for that path. A tensor is accepted for
            focused metadata tests.
        forbid_realloc: If True, asserts every write reuses the existing
            tensor storage (sanity check for the captured-graph path).
    """
    q_len = meta.q_len_per_req
    window_size = meta.window_size
    device = meta.start_pos.device

    start_pos = _build_start_pos_from_attention_inputs(
        attention_inputs,
        device,
        meta.max_seq_len,
        q_len,
    )
    bs = int(start_pos.shape[0])
    position_ids_2d = _build_position_ids_2d(start_pos, q_len, device)
    position_ids_flat = position_ids_2d.reshape(-1).contiguous()
    if paged_pool_entries_per_block:
        if paged_pool_tokens_per_block is None:
            raise ValueError("paged_pool_tokens_per_block is required for paged pools")

    # snapshot pointers for the realloc-forbidden mode
    if forbid_realloc:
        ptr_snap = {
            "start_pos": meta.start_pos.data_ptr(),
            "position_ids": meta.position_ids.data_ptr(),
            "slot_swa": meta.slot_mapping_swa.data_ptr(),
            "topk_window": meta.topk_window_idxs.data_ptr(),
            "topk_buffer_compressed": meta.topk_buffer_compressed.data_ptr(),
        }
        if meta.position_ids_long is not None:
            ptr_snap["position_ids_long"] = meta.position_ids_long.data_ptr()
        if meta.decode_seq_start_per_req is not None:
            ptr_snap["decode_seq_start_per_req"] = (
                meta.decode_seq_start_per_req.data_ptr()
            )
        for r, t in meta.slot_mapping_compressed.items():
            ptr_snap[f"slot_compressed[{r}]"] = t.data_ptr()
        for r, t in meta.compressed_lens.items():
            ptr_snap[f"compressed_lens[{r}]"] = t.data_ptr()
        for r, t in meta.compressed_lens_per_token.items():
            ptr_snap[f"compressed_lens_per_token[{r}]"] = t.data_ptr()
        for r, t in meta.topk_total_by_ratio.items():
            ptr_snap[f"topk_total_by_ratio[{r}]"] = t.data_ptr()
        for at, t in meta.compressor_state_slot_mappings.items():
            ptr_snap[f"compressor_state_slot_mappings[{at}]"] = t.data_ptr()
        if meta.swa_global_slots is not None:
            ptr_snap["swa_global_slots"] = meta.swa_global_slots.data_ptr()
        if meta.hca_cmp_global_slots is not None:
            ptr_snap["hca_cmp_global_slots"] = meta.hca_cmp_global_slots.data_ptr()
        if meta.swa_topk_length is not None:
            ptr_snap["swa_topk_length"] = meta.swa_topk_length.data_ptr()
        for r, t in meta.compressed_topk_length_by_ratio.items():
            ptr_snap[f"compressed_topk_length_by_ratio[{r}]"] = t.data_ptr()

    meta.position_ids[: bs * q_len].copy_(position_ids_flat)
    if meta.position_ids_long is not None:
        meta.position_ids_long[: bs * q_len].copy_(position_ids_flat.to(torch.long))
    if meta.decode_seq_start_per_req is not None:
        meta.decode_seq_start_per_req[:bs].copy_(start_pos)
    meta.topk_buffer_compressed[:bs].fill_(-1)

    from rtp_llm.models_py.modules.dsv4.fp8.decode._fused_prepare_meta_triton import (
        fused_update_decode_meta_pure,
    )

    fused_update_decode_meta_pure(meta, start_pos, meta.max_seq_len)

    # opt_flash_mla: derive FlashMLA topk_length / extra_topk_length from the
    # just-filled compressed_lens + start_pos, in place (graph-stable buffers).
    # During CUDA-graph capture/warmup (capture_full_width_lengths=True) write
    # the FULL width so FlashMLA's frozen tile schedule is max-provisioned and
    # safe for any real replay length; real per-request lengths are written
    # before each replay by prepare_cuda_graph(). See _update_topk_lengths_in_place.
    _update_topk_lengths_in_place(
        meta, start_pos, bs, full_width=capture_full_width_lengths
    )

    # ------------------------------------------------------------------
    # Phase 2: paged write slot mappings (per attn_type).
    # SWA-K writes EVERY decode token; CSA-K / HCA-K / INDEXER-K write
    # ONLY on their respective compression boundaries (sentinel ``-1``
    # otherwise — the write op honors that via ``mask_negative=True``).
    # ------------------------------------------------------------------
    if paged_block_tables and paged_pool_entries_per_block:
        paged_pool_tokens_per_block = _resolve_paged_pool_tokens_per_block(
            paged_pool_entries_per_block,
            paged_pool_tokens_per_block or meta.paged_pool_tokens_per_block,
        )
        meta.paged_pool_tokens_per_block = dict(paged_pool_tokens_per_block)
        # Snapshot block_table content into the metadata's stable buffer
        # (forbid_realloc-friendly: just `.copy_` the prefix). Skip pools
        # without a metadata buffer (e.g. SWA-only layers don't carry CSA).
        for at, src_bt in paged_block_tables.items():
            dst_bt = meta.pool_block_tables.get(at)
            if dst_bt is None:
                continue
            n_rows = min(src_bt.shape[0], dst_bt.shape[0])
            n_cols = min(src_bt.shape[1], dst_bt.shape[1])
            # Zero stale rows beyond current bs to avoid carrying old block
            # ids (defensive — graph reads only [:bs] anyway).
            dst_bt[bs:].zero_()
            dst_bt[:n_rows, :n_cols].copy_(src_bt[:n_rows, :n_cols])
            if n_cols < dst_bt.shape[1]:
                dst_bt[:n_rows, n_cols:].zero_()

        from rtp_llm.models_py.modules.dsv4.fp8.decode._fused_prepare_meta_triton import (
            fused_phase2b_pool_slot_mapping,
        )

        fused_phase2b_pool_slot_mapping(
            meta,
            start_pos,
            bs,
            paged_pool_entries_per_block,
            paged_pool_tokens_per_block,
        )

        _update_compressor_state_slot_mappings(
            meta,
            bs,
            paged_pool_entries_per_block,
        )

    # Iter3.3: precompute translate_swa / translate_hca once per step.
    # Shared across all 43 attention layers. Graph-safe: ``.copy_()`` into
    # pre-allocated buffers on meta (allocated at max_bs); never reassigns
    # the attribute, so graph replays read from fixed addresses.
    # ``req_id_per_token`` is deterministic (``arange(B)``) so it was filled
    # at allocate time and stays stable.
    if paged_block_tables and paged_pool_entries_per_block:
        from rtp_llm.models_py.modules.dsv4.attn_type import HCA_KV, SWA_KV
        from rtp_llm.models_py.modules.dsv4.fp8.decode.paged_topk_translator import (
            translate_local_to_global_slots,
        )

        T = bs * q_len
        req_id_bs = (
            meta.req_id_per_token[:T] if meta.req_id_per_token is not None else None
        )

        if (
            req_id_bs is not None
            and meta.swa_global_slots is not None
            and SWA_KV in meta.pool_block_tables
            and meta.swa_abs_idx is not None
        ):
            swa_eb = paged_pool_entries_per_block[SWA_KV]
            swa_tokens_per_block = paged_pool_tokens_per_block[SWA_KV]
            swa_local = meta.swa_abs_idx[:bs].reshape(T, window_size)
            swa_global_new = translate_local_to_global_slots(
                req_id_bs,
                meta.pool_block_tables[SWA_KV][:bs],
                swa_local,
                entries_per_block=swa_eb,
                tokens_per_block_for_block_table=swa_tokens_per_block,
            )
            meta.swa_global_slots[:T].copy_(swa_global_new)

        # HCA layers all share the dense-idx-masked cmp_local_raw (already
        # materialised as topk_total[r=128][:, :, win:]); translate it once.
        if (
            req_id_bs is not None
            and meta.hca_cmp_global_slots is not None
            and HCA_KV in meta.pool_block_tables
            and 128 in meta.topk_total_by_ratio
        ):
            hca_eb = paged_pool_entries_per_block[HCA_KV]
            hca_tokens_per_block = _compressed_domain_tokens_per_block(
                HCA_KV, paged_pool_tokens_per_block[HCA_KV]
            )
            hca_tt = meta.topk_total_by_ratio[128]
            K_h = hca_tt.shape[-1] - window_size
            hca_cmp_local = hca_tt[:bs, :, window_size:].reshape(T, K_h).contiguous()
            hca_global_new = translate_local_to_global_slots(
                req_id_bs,
                meta.pool_block_tables[HCA_KV][:bs],
                hca_cmp_local,
                entries_per_block=hca_eb,
                tokens_per_block_for_block_table=hca_tokens_per_block,
            )
            meta.hca_cmp_global_slots[:T].copy_(hca_global_new)

    # Update Python-scalar geometry (cheap — these are not captured into the graph)
    meta.batch_size = bs
    meta.total_tokens = bs * q_len

    if forbid_realloc:
        # Verify every storage pointer is unchanged.
        cur = {
            "start_pos": meta.start_pos.data_ptr(),
            "position_ids": meta.position_ids.data_ptr(),
            "slot_swa": meta.slot_mapping_swa.data_ptr(),
            "topk_window": meta.topk_window_idxs.data_ptr(),
            "topk_buffer_compressed": meta.topk_buffer_compressed.data_ptr(),
        }
        if meta.position_ids_long is not None:
            cur["position_ids_long"] = meta.position_ids_long.data_ptr()
        if meta.decode_seq_start_per_req is not None:
            cur["decode_seq_start_per_req"] = meta.decode_seq_start_per_req.data_ptr()
        for r, t in meta.slot_mapping_compressed.items():
            cur[f"slot_compressed[{r}]"] = t.data_ptr()
        for r, t in meta.compressed_lens.items():
            cur[f"compressed_lens[{r}]"] = t.data_ptr()
        for r, t in meta.compressed_lens_per_token.items():
            cur[f"compressed_lens_per_token[{r}]"] = t.data_ptr()
        for r, t in meta.topk_total_by_ratio.items():
            cur[f"topk_total_by_ratio[{r}]"] = t.data_ptr()
        for at, t in meta.compressor_state_slot_mappings.items():
            cur[f"compressor_state_slot_mappings[{at}]"] = t.data_ptr()
        if meta.swa_global_slots is not None:
            cur["swa_global_slots"] = meta.swa_global_slots.data_ptr()
        if meta.hca_cmp_global_slots is not None:
            cur["hca_cmp_global_slots"] = meta.hca_cmp_global_slots.data_ptr()
        if meta.swa_topk_length is not None:
            cur["swa_topk_length"] = meta.swa_topk_length.data_ptr()
        for r, t in meta.compressed_topk_length_by_ratio.items():
            cur[f"compressed_topk_length_by_ratio[{r}]"] = t.data_ptr()
        for k, p_before in ptr_snap.items():
            assert cur[k] == p_before, (
                f"update_decode_metadata_in_place_fp8(forbid_realloc=True) "
                f"reallocated buffer '{k}': before={hex(p_before)} after={hex(cur[k])}"
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
    paged_block_tables: Optional[Dict[int, torch.Tensor]] = None,
    paged_pool_entries_per_block: Optional[Dict[int, int]] = None,
    paged_pool_tokens_per_block: Optional[Dict[int, int]] = None,
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
    pool_block_tables: Dict[int, torch.Tensor] = {}
    pool_write_slot_mappings: Dict[int, torch.Tensor] = {}
    compressor_state_slot_mappings: Dict[int, torch.Tensor] = {}
    paged_pool_tokens_per_block_resolved: Dict[int, int] = {}
    req_id_per_token: Optional[torch.Tensor] = None
    req_id_per_token_long: Optional[torch.Tensor] = None
    decode_seq_start_per_req: Optional[torch.Tensor] = None
    decode_cu_seq_per_req: Optional[torch.Tensor] = None
    swa_global_slots: Optional[torch.Tensor] = None
    hca_cmp_global_slots: Optional[torch.Tensor] = None
    if paged_block_tables and paged_pool_entries_per_block:
        from rtp_llm.models_py.modules.dsv4.attn_type import (
            CSA_KV,
            HCA_KV,
            INDEXER_KV,
            SWA_KV,
        )
        from rtp_llm.models_py.modules.dsv4.fp8.decode.pool_slot_mapping import (
            compute_kv_pool_slot_mapping,
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
        from rtp_llm.models_py.modules.dsv4.attn_type import HCA_KV, SWA_KV
        from rtp_llm.models_py.modules.dsv4.fp8.decode.paged_topk_translator import (
            build_req_id_per_token,
            translate_local_to_global_slots,
        )

        req_id_per_token = build_req_id_per_token(int(B), q_len, device)
        req_id_per_token_long = req_id_per_token.to(torch.long)
        decode_seq_start_per_req = start_pos.to(torch.int32).contiguous()
        decode_cu_seq_per_req = torch.arange(
            0,
            (int(B) + 1) * q_len,
            q_len,
            dtype=torch.int32,
            device=device,
        )

        from rtp_llm.models_py.modules.dsv4.attn_type import (
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
