"""Mean-pool compression for the qwen4 QSA indexer's KV + state pools.

The DSv4 path uses ``CompressorFP8`` (1448 lines, weight-based: ``{ape,wkv,wgate,norm}``).
This module is the parameter-free replacement the qwen4 checkpoint ships: raw keys are
averaged in fp32 over each ``compress_ratio``-sized block, then normed, partial-roped,
cast to bf16, and written to the ``indexer_kv`` pool as a 256-byte entry for the
released 128-wide indexer head. This is the same format consumed by
``qsa_paged_indexer_score``; there is no inline scale in the production ABI.

The ``indexer_state`` pool retains the raw keys of the incomplete block so the next
block's mean can be computed (the checkpoint does not keep a running sum; we keep fp32
raw keys via the state ring to reproduce the reference's bit-exact summation order).

The functions here are **production-shaped but pool-agnostic**: they accept a flat buffer
(view of the ``indexer_kv`` / ``indexer_state`` pool tensors) and return metadata about
where entries were written, so the caller can set up the actual pool views through
``PoolBackedModule.set_pool_context`` and call these functions with the real tensors.
"""

from dataclasses import dataclass
from typing import Literal, Optional

import torch

from rtp_llm.models_py.modules.qwen4_exp.indexer import apply_partial_rope
from rtp_llm.models_py.modules.qwen4_exp.norm import exact_head_rms_norm

InvalidBlockPolicy = Literal["raise", "skip"]


@dataclass(frozen=True)
class IndexerCacheUndo:
    """Original values for the unique side-cache slots touched by one write."""

    state_pool: torch.Tensor
    state_slots: torch.Tensor
    original_state: torch.Tensor
    kv_pool: torch.Tensor
    kv_slots: torch.Tensor
    original_kv: torch.Tensor


def _capture_pool_slots(
    pool: torch.Tensor, destinations: list[tuple[int, int, torch.Tensor]]
) -> tuple[torch.Tensor, torch.Tensor]:
    entries_per_block = int(pool.shape[1])
    # A long prefill can wrap the state ring and touch one destination more than
    # once. Preserve the value that existed before the first write only.
    unique_slots = list(
        dict.fromkeys(
            block_id * entries_per_block + offset
            for block_id, offset, _ in destinations
        )
    )
    slots = torch.tensor(unique_slots, dtype=torch.long, device=pool.device)
    original = pool.flatten(0, 1).index_select(0, slots).clone()
    return slots, original


def restore_indexer_cache(undo: IndexerCacheUndo) -> None:
    """Restore a captured side-cache write on the caller's current stream."""
    undo.state_pool.flatten(0, 1).index_copy_(0, undo.state_slots, undo.original_state)
    undo.kv_pool.flatten(0, 1).index_copy_(0, undo.kv_slots, undo.original_kv)


def _physical_block(
    block_table: torch.Tensor,
    request_idx: int,
    logical_block: int,
    *,
    num_physical_blocks: int,
    tag: str,
    invalid_block_policy: InvalidBlockPolicy,
) -> Optional[int]:
    """Resolve one request-local block-table entry without flattening batches."""
    table_width = int(block_table.shape[1])
    if logical_block < 0 or logical_block >= table_width:
        if invalid_block_policy == "skip":
            return None
        raise ValueError(
            f"{tag} logical block {logical_block} is outside block_table width "
            f"{table_width} for request {request_idx}"
        )
    block_id = int(block_table[request_idx, logical_block].item())
    if block_id <= 0:
        if invalid_block_policy == "skip":
            return None
        raise ValueError(
            f"{tag} block_id must be > 0 for request {request_idx}, logical "
            f"block {logical_block}; got {block_id}"
        )
    if block_id >= num_physical_blocks:
        raise IndexError(
            f"{tag} physical block {block_id} is outside pool size "
            f"{num_physical_blocks}"
        )
    return block_id


def write_indexer_cache(
    raw_keys: torch.Tensor,
    cu_seqlens: torch.Tensor,
    start_positions: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    k_norm_gamma: torch.Tensor,
    kv_pool: torch.Tensor,
    kv_block_table: torch.Tensor,
    state_pool: torch.Tensor,
    state_block_table: torch.Tensor,
    *,
    ratio: int,
    kv_tokens_per_block: int,
    state_tokens_per_block: int,
    norm_eps: float = 1e-6,
    invalid_block_policy: InvalidBlockPolicy = "raise",
    capture_undo: bool = False,
) -> dict:
    """Write ragged raw indexer keys to the state ring and completed means to KV.

    ``raw_keys`` is the packed projection output for this invocation. Request
    ``b`` occupies ``raw_keys[cu_seqlens[b]:cu_seqlens[b+1]]`` and starts at
    absolute token position ``start_positions[b]``.  This explicit metadata is
    sufficient for both ragged prefill and one-token decode, including a block
    completed using raw keys retained by an earlier invocation.

    Pool views are the typed production layouts:

    * ``kv_pool[num_blocks, kv_tokens_per_block / ratio, head_dim]`` bf16
    * ``state_pool[num_blocks, ring_entries, head_dim]`` fp32

    Each pool uses its *own* ``[B, max_blocks]`` block table. Block id zero is
    reserved by the framework. ``invalid_block_policy='raise'`` fails before
    mutating either pool; ``'skip'`` records ``-1`` for that write. Physical ids
    beyond the pool extent always raise because skipping them could hide memory
    corruption. Both tables use absolute logical-block columns. Only the payload
    offset inside an ``indexer_state`` block is a ring.

    ``rope_cos`` / ``rope_sin`` must cover absolute positions and may be shared
    ``[max_position, rotary_dim]`` or request-specific
    ``[B, max_position, rotary_dim]``.
    """
    if invalid_block_policy not in ("raise", "skip"):
        raise ValueError(
            "invalid_block_policy must be 'raise' or 'skip', got "
            f"{invalid_block_policy!r}"
        )
    if raw_keys.dim() != 2:
        raise ValueError(f"raw_keys must be [N, head_dim], got {raw_keys.shape}")
    if raw_keys.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"raw_keys must be floating point, got {raw_keys.dtype}")
    if ratio <= 0:
        raise ValueError(f"ratio must be positive, got {ratio}")
    if kv_tokens_per_block <= 0 or kv_tokens_per_block % ratio:
        raise ValueError(
            "kv_tokens_per_block must be positive and divisible by ratio, got "
            f"{kv_tokens_per_block} and {ratio}"
        )
    if state_tokens_per_block <= 0:
        raise ValueError(
            f"state_tokens_per_block must be positive, got {state_tokens_per_block}"
        )

    head_dim = int(raw_keys.shape[1])
    if k_norm_gamma.shape != (head_dim,):
        raise ValueError(f"k_norm_gamma must be [{head_dim}], got {k_norm_gamma.shape}")
    kv_entries_per_block = kv_tokens_per_block // ratio
    if (
        kv_pool.dim() != 3
        or kv_pool.dtype != torch.bfloat16
        or tuple(kv_pool.shape[1:]) != (kv_entries_per_block, head_dim)
    ):
        raise ValueError(
            "kv_pool must be bf16 [num_blocks, "
            f"{kv_entries_per_block}, {head_dim}], got {kv_pool.shape} "
            f"{kv_pool.dtype}"
        )
    if (
        state_pool.dim() != 3
        or state_pool.dtype != torch.float32
        or int(state_pool.shape[2]) != head_dim
    ):
        raise ValueError(
            "state_pool must be fp32 [num_blocks, ring_entries, "
            f"{head_dim}], got {state_pool.shape} {state_pool.dtype}"
        )
    state_ring_entries = int(state_pool.shape[1])
    if state_ring_entries < ratio:
        raise ValueError(
            f"state ring needs at least ratio={ratio} entries, got "
            f"{state_ring_entries}"
        )

    if cu_seqlens.dim() != 1 or cu_seqlens.numel() < 2:
        raise ValueError("cu_seqlens must be a [B + 1] vector")
    batch_size = int(cu_seqlens.numel()) - 1
    if start_positions.shape != (batch_size,):
        raise ValueError(
            f"start_positions must be [{batch_size}], got {start_positions.shape}"
        )
    for name, table in (
        ("kv_block_table", kv_block_table),
        ("state_block_table", state_block_table),
    ):
        if table.dim() != 2 or int(table.shape[0]) != batch_size:
            raise ValueError(f"{name} must be [B, max_blocks], got {table.shape}")
        if table.dtype != torch.int32:
            raise ValueError(f"{name} must be int32, got {table.dtype}")
        if int(table.shape[1]) == 0:
            raise ValueError(f"{name} must contain at least one block column")

    offsets = [int(v) for v in cu_seqlens.tolist()]
    if offsets[0] != 0 or offsets[-1] != int(raw_keys.shape[0]):
        raise ValueError(
            f"cu_seqlens must start at 0 and end at N={raw_keys.shape[0]}, got "
            f"{offsets[0]} and {offsets[-1]}"
        )
    if any(lo > hi for lo, hi in zip(offsets, offsets[1:])):
        raise ValueError("cu_seqlens must be non-decreasing")
    starts = [int(v) for v in start_positions.tolist()]
    if any(pos < 0 for pos in starts):
        raise ValueError(f"start_positions must be non-negative, got {starts}")

    if rope_cos.shape != rope_sin.shape or rope_cos.dim() not in (2, 3):
        raise ValueError(
            "rope_cos/rope_sin must have equal [max_pos, rotary_dim] or "
            f"[B, max_pos, rotary_dim] shapes, got {rope_cos.shape} and "
            f"{rope_sin.shape}"
        )
    if rope_cos.dim() == 3 and int(rope_cos.shape[0]) != batch_size:
        raise ValueError(
            f"request-specific RoPE table must have B={batch_size}, got "
            f"{rope_cos.shape[0]}"
        )
    rotary_dim = int(rope_cos.shape[-1])
    if rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2:
        raise ValueError(
            f"rotary_dim must be positive, even, and <= {head_dim}; got "
            f"{rotary_dim}"
        )
    rope_positions = int(rope_cos.shape[-2])
    max_position = max(
        (starts[b] + offsets[b + 1] - offsets[b] for b in range(batch_size)),
        default=0,
    )
    if max_position > rope_positions:
        raise ValueError(
            f"RoPE tables cover {rope_positions} positions, need {max_position}"
        )

    # Stage every value and destination first. Strict-mode validation therefore
    # cannot leave one pool half-written if a later request has bad metadata.
    state_writes = []
    kv_writes = []
    state_slots = torch.full(
        (raw_keys.shape[0],), -1, dtype=torch.long, device=raw_keys.device
    )
    kv_slots = torch.full_like(state_slots, -1)
    completed = torch.zeros_like(state_slots, dtype=torch.bool)

    for request_idx in range(batch_size):
        flat_lo, flat_hi = offsets[request_idx], offsets[request_idx + 1]
        seq_start = starts[request_idx]
        seq_len = flat_hi - flat_lo

        for local_idx in range(seq_len):
            flat_idx = flat_lo + local_idx
            position = seq_start + local_idx
            state_logical_block = position // state_tokens_per_block
            state_block = _physical_block(
                state_block_table,
                request_idx,
                state_logical_block,
                num_physical_blocks=int(state_pool.shape[0]),
                tag="indexer_state",
                invalid_block_policy=invalid_block_policy,
            )
            if state_block is not None:
                state_offset = position % state_ring_entries
                state_slots[flat_idx] = state_block * state_ring_entries + state_offset
                state_writes.append(
                    (state_block, state_offset, raw_keys[flat_idx].float())
                )

            if (position + 1) % ratio:
                continue
            completed[flat_idx] = True
            block_start = position - ratio + 1
            gathered = []
            source_missing = False
            for source_position in range(block_start, position + 1):
                source_local = source_position - seq_start
                if 0 <= source_local < seq_len:
                    gathered.append(raw_keys[flat_lo + source_local].float())
                    continue
                source_logical_block = source_position // state_tokens_per_block
                source_block = _physical_block(
                    state_block_table,
                    request_idx,
                    source_logical_block,
                    num_physical_blocks=int(state_pool.shape[0]),
                    tag="indexer_state source",
                    invalid_block_policy=invalid_block_policy,
                )
                if source_block is None:
                    source_missing = True
                    break
                gathered.append(
                    state_pool[source_block, source_position % state_ring_entries]
                )
            if source_missing:
                continue

            kv_logical_block = block_start // kv_tokens_per_block
            kv_block = _physical_block(
                kv_block_table,
                request_idx,
                kv_logical_block,
                num_physical_blocks=int(kv_pool.shape[0]),
                tag="indexer_kv",
                invalid_block_policy=invalid_block_policy,
            )
            if kv_block is None:
                continue
            compressed_idx = block_start // ratio
            kv_offset = compressed_idx % kv_entries_per_block
            pooled = torch.stack(gathered).float().mean(dim=0).to(raw_keys.dtype)
            pooled = exact_head_rms_norm(pooled, k_norm_gamma, norm_eps)
            cos = (
                rope_cos[block_start]
                if rope_cos.dim() == 2
                else rope_cos[request_idx, block_start]
            )
            sin = (
                rope_sin[block_start]
                if rope_sin.dim() == 2
                else rope_sin[request_idx, block_start]
            )
            pooled = apply_partial_rope(pooled, cos, sin).to(torch.bfloat16)
            kv_slots[flat_idx] = kv_block * kv_entries_per_block + kv_offset
            kv_writes.append((kv_block, kv_offset, pooled))

    undo = None
    if capture_undo:
        # Capture only after every value and destination has been validated and
        # staged, immediately before the first persistent copy.
        state_undo_slots, original_state = _capture_pool_slots(state_pool, state_writes)
        kv_undo_slots, original_kv = _capture_pool_slots(kv_pool, kv_writes)
        undo = IndexerCacheUndo(
            state_pool=state_pool,
            state_slots=state_undo_slots,
            original_state=original_state,
            kv_pool=kv_pool,
            kv_slots=kv_undo_slots,
            original_kv=original_kv,
        )

    try:
        for block_id, offset, value in state_writes:
            state_pool[block_id, offset].copy_(value)
        for block_id, offset, value in kv_writes:
            kv_pool[block_id, offset].copy_(value)
    except BaseException:
        if undo is not None:
            restore_indexer_cache(undo)
            if state_pool.is_cuda:
                torch.cuda.current_stream(state_pool.device).synchronize()
        raise

    result = {
        "state_slots": state_slots,
        "kv_slots": kv_slots,
        "completed": completed,
        "num_state_writes": len(state_writes),
        "num_kv_writes": len(kv_writes),
    }
    if capture_undo:
        result["undo"] = undo
    return result


def compress_prefill(
    hidden_states: torch.Tensor,  # [B, T, D]
    qk_proj: torch.Tensor,  # [(n_heads+1)*head_dim, D]
    k_norm_gamma: torch.Tensor,  # [head_dim]
    cos: torch.Tensor,  # [B, T, rotary_dim] (full-range)
    sin: torch.Tensor,  # [B, T, rotary_dim]
    *,
    ratio: int,
    head_dim: int,
    norm_eps: float = 1e-6,
) -> dict:
    """Full prefill compression returning the compressed pool entries + raw keys.

    This is the **flat-buffer form**: instead of writing to framework-managed paged
    pools, it returns tensors that let the caller (or a test) verify the compressed
    content by placing it in a pool buffer.

    Returns a dict::

        raw_keys      [B, T, D] fp32
        pooled_k      [B, nb, D] bf16          (one per **complete** block)
        block_starts  [B, nb] int64            first token index of each block
        num_blocks    [B] int64                how many complete blocks per batch

    The caller can copy ``pooled_k[i, b]`` directly into the bf16 view of the
    ``indexer_kv`` pool at slot ``i * num_blocks + b``.
    """
    if hidden_states.dim() != 3:
        raise ValueError(
            f"hidden_states must be [B, T, hidden], got {tuple(hidden_states.shape)}"
        )
    if qk_proj.dim() != 2 or qk_proj.shape[1] != hidden_states.shape[-1]:
        raise ValueError(
            "qk_proj must be [(n_heads + 1) * head_dim, hidden], got "
            f"{tuple(qk_proj.shape)} for hidden={hidden_states.shape[-1]}"
        )
    if ratio <= 0:
        raise ValueError(f"ratio must be positive, got {ratio}")
    if head_dim <= 0 or qk_proj.shape[0] % head_dim:
        raise ValueError(
            f"qk_proj output {qk_proj.shape[0]} is not divisible by head_dim={head_dim}"
        )
    n_heads = (qk_proj.shape[0] // head_dim) - 1
    if n_heads <= 0:
        raise ValueError("qk_proj must contain at least one Q head and one K head")
    if k_norm_gamma.shape != (head_dim,):
        raise ValueError(
            f"k_norm_gamma must be [{head_dim}], got {tuple(k_norm_gamma.shape)}"
        )
    B, T, _ = hidden_states.shape
    device = hidden_states.device
    if cos.shape[:2] != (B, T) or sin.shape != cos.shape:
        raise ValueError(
            f"cos/sin must have matching [B, T, rotary_dim] shapes, got "
            f"{tuple(cos.shape)} and {tuple(sin.shape)}"
        )
    rotary_dim = cos.shape[-1]
    if rotary_dim <= 0 or rotary_dim > head_dim or rotary_dim % 2:
        raise ValueError(
            f"rotary_dim must be positive, even, and <= head_dim; got {rotary_dim}"
        )

    qk = hidden_states @ qk_proj.T
    _, raw_k = qk.split([n_heads * head_dim, head_dim], dim=-1)
    raw_keys = raw_k.float()  # [B, T, D] fp32

    num_blocks_full = T // ratio
    # Truncate raw_keys to the token positions that form complete blocks.
    grouped = raw_keys[:, : num_blocks_full * ratio]  # [B, nb*ratio, D]
    grouped = grouped.unflatten(1, (num_blocks_full, ratio))  # [B, nb, ratio, D]
    # Mean in fp32 (upstream orders the summation the same way).
    pooled = grouped.mean(dim=2).to(raw_k.dtype)  # [B, nb, D]

    # Apply k_layernorm (with +1) and partial RoPE at each block's first position.
    pooled = exact_head_rms_norm(pooled, k_norm_gamma[None, None, :], norm_eps)
    block_starts = torch.arange(num_blocks_full, device=device) * ratio  # [nb]
    pooled_rotated = apply_partial_rope(
        pooled,
        cos[:, block_starts],
        sin[:, block_starts],
    )
    pooled_k = pooled_rotated.to(torch.bfloat16).contiguous()

    return {
        "raw_keys": raw_keys,
        "pooled_k": pooled_k,
        "block_starts": block_starts.expand(B, -1),
        "num_blocks": torch.full(
            (B,), num_blocks_full, dtype=torch.long, device=device
        ),
    }
