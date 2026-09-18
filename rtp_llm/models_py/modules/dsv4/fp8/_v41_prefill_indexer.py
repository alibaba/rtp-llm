"""V4.1 prefill index scoring using the existing row-scale FP8 kernels.

Keep the cache's original FP8 bytes/scales: dequantizing and quantizing again
would change the continuous-scale quantization boundary. CP exchanges raw bytes,
then restores logical key order before the dense, causally bounded scorer.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

import torch


@dataclass(frozen=True)
class PrefillIndexerKeys:
    quant: torch.Tensor
    scale: torch.Tensor

    def __len__(self) -> int:
        return self.quant.shape[0]


def is_supported(device: torch.device, num_heads: int, head_dim: int) -> bool:
    """Gate unsupported devices/dependencies, never catch execution failures."""
    if (
        os.environ.get("DSV41_FUSED_PREFILL_INDEXER", "1") == "0"
        or torch.device(device).type != "cuda"
        or num_heads not in (32, 64)
        or head_dim != 128
    ):
        return False
    try:
        from ._indexer_score import has_fp8_mqa_logits
    except ImportError:
        return False
    return has_fp8_mqa_logits() and torch.cuda.get_device_capability(device)[0] in (
        9,
        10,
    )


def logits_chunk_rows(num_keys: int) -> int:
    """Bound logical FP32 logits to 256 MiB, with at most 512 queries.

    DeepGEMM additionally pads its row stride by up to 511 keys and aligns
    query rows to its small Q tile. This is independent of the head dimension;
    no [queries, heads, keys] temporary is materialized.
    """
    if num_keys < 0:
        raise ValueError("num_keys must be nonnegative")
    if num_keys == 0:
        return 512
    return max(1, min(512, (256 * 1024 * 1024) // (4 * num_keys)))


def quantize_indexer_q(q: torch.Tensor, weights: torch.Tensor):
    """Preserve absmax/448 Q scaling and fold it into the existing weights."""
    from ._indexer_q_quant_triton import indexer_q_fp8_quant_fold

    quant, folded = indexer_q_fp8_quant_fold(
        q.contiguous().unsqueeze(0), weights.contiguous().unsqueeze(0)
    )
    return quant.squeeze(0), folded.squeeze(0)


def quantize_indexer_k_reference(k: torch.Tensor):
    """Pool-free reference/warmup boundary, identical to fp8_roundtrip."""
    value = k.float()
    scale = (value.abs().amax(-1) / 448.0).clamp_min(1e-12)
    return (value / scale[:, None]).to(torch.float8_e4m3fn), scale


def gather_indexer_keys(
    pool: torch.Tensor,
    slot_mapping_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    count: int,
    reqid: int,
    ratio: int,
    cp_ctx,
    owner_tokens_per_block: int,
) -> PrefillIndexerKeys:
    """Read exact cache bytes through V4.1's established slot mapping.

    Physical pool page size and CP ownership block size are independent. The
    latter is expressed in compressed entries; the callback maps original
    token positions through the typed pool tables for both ratios 1 and 2.
    """
    from ._indexer_cp_gather_triton import gather_indexer_k_for_prefill

    device = pool.device
    if count == 0:
        return PrefillIndexerKeys(
            torch.empty(0, 128, dtype=torch.float8_e4m3fn, device=device),
            torch.empty(0, dtype=torch.float32, device=device),
        )
    sharded = cp_ctx is not None and cp_ctx.cp_size > 1 and cp_ctx.kv_cache_sharded
    if not sharded:
        ids = torch.arange(count, dtype=torch.long, device=device)
        slots = slot_mapping_fn((ids + 1) * ratio - 1, torch.full_like(ids, reqid))
        return PrefillIndexerKeys(*gather_indexer_k_for_prefill(pool, slots))

    from ._indexer_cp_assembler import assemble_indexer_k, build_indexer_cp_chunk_plan

    if ratio not in (1, 2) or owner_tokens_per_block % ratio:
        raise ValueError("V4.1 indexer CP ownership must align to its ratio")
    owner_entries = owner_tokens_per_block // ratio
    plan = build_indexer_cp_chunk_plan(
        cp_ctx,
        torch.tensor([count], dtype=torch.long, device=device),
        block_size=pool.shape[1],
        owner_block_size=owner_entries,
        device=device,
        total_kv_len=count,
    )
    local = torch.arange(plan.total_local_T, dtype=torch.long, device=device)
    logical = (
        local // owner_entries * cp_ctx.cp_size + cp_ctx.cp_rank
    ) * owner_entries + local.remainder(owner_entries)
    # Padding must not read an unallocated virtual page. Map a valid owned row
    # first, then explicitly replace padding slots with the gather skip marker.
    safe_logical = logical.clamp_max(count - 1)
    slots = slot_mapping_fn(
        (safe_logical + 1) * ratio - 1, torch.full_like(local, reqid)
    )
    slots = torch.where(logical < count, slots, -1)
    local_quant, local_scale = gather_indexer_k_for_prefill(pool, slots)
    quant = torch.empty(count, 128, dtype=torch.float8_e4m3fn, device=device)
    scale = torch.empty(count, dtype=torch.float32, device=device)
    assemble_indexer_k(
        plan=plan,
        local_k_quant=local_quant.view(torch.uint8),
        local_k_scale=local_scale.view(torch.uint8).view(-1, 4),
        out_k_quant=quant.view(torch.uint8),
        out_k_scale=scale.view(torch.uint8).view(-1, 4),
    )
    return PrefillIndexerKeys(quant, scale)


def score_indexer_chunk(q_fp8, w_fold, k_fp8, k_scale, visible):
    """Fused dot/ReLU/head reduction; keep causal padding at negative infinity."""
    from ._indexer_score import fp8_mqa_indexer_score

    count = k_fp8.shape[0]
    if q_fp8.shape[0] == 0 or count == 0:
        return torch.empty(
            q_fp8.shape[0], count, dtype=torch.float32, device=q_fp8.device
        )
    ends = visible.to(torch.int32).clamp(0, count).contiguous()
    starts = torch.zeros_like(ends)
    # The scheduler uses per-query bounds to skip future K tiles. Explicitly
    # mask below as well: topk and candidate selection must never observe the
    # scheduler's padding, including a row with no completed ratio-2 pair.
    logits = fp8_mqa_indexer_score(
        q_fp8,
        w_fold,
        k_fp8,
        k_scale,
        starts,
        ends,
        clean_logits=True,
        # Positive max_seqlen_k selects DG's compressed-logit layout, which
        # is incompatible with clean_logits. This caller uses ordinary [M,K].
        max_seqlen_k=0,
    )
    logits.masked_fill_(
        torch.arange(count, device=logits.device)[None] >= ends[:, None],
        -torch.inf,
    )
    return logits
