"""PPU overrides; unchanged behavior is inherited from the shared DSV4 implementation."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import torch
from rtp_llm.models_py.modules.dsv4.fp8.decode import decode_attn_metadata as _base

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
class DSv4DecodeAttnMetadataFP8(_base.DSv4DecodeAttnMetadataFP8):
    active_token_mask: Optional[torch.Tensor] = None
    sched_meta_cache: Dict[Tuple[int, int, int, Optional[str], int], Any] = field(default_factory=dict)
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
    metadata = _base.allocate_decode_metadata_fp8(
        max_batch_size=max_batch_size,
        q_len=q_len,
        window_size=window_size,
        head_dim=head_dim,
        max_seq_len=max_seq_len,
        compress_ratios=compress_ratios,
        index_topk=index_topk,
        device=device,
        paged_pool_specs=paged_pool_specs,
    )
    return DSv4DecodeAttnMetadataFP8(**vars(metadata))

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
    metadata = _base.build_decode_metadata_fp8(
        attention_inputs=attention_inputs,
        q_len=q_len,
        window_size=window_size,
        head_dim=head_dim,
        max_seq_len=max_seq_len,
        compress_ratios=compress_ratios,
        index_topk=index_topk,
        device=device,
        dtype=dtype,
        paged_block_tables=paged_block_tables,
        paged_pool_entries_per_block=paged_pool_entries_per_block,
        paged_pool_tokens_per_block=paged_pool_tokens_per_block,
    )
    return DSv4DecodeAttnMetadataFP8(**vars(metadata))
