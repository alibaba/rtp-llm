"""CP-sharded compressed-attention planning helpers.

These helpers are pure tensor arithmetic used by the cache-hit attention
optimization. They intentionally do not launch collectives or kernels, so the
communication threshold and CSA topk ownership mapping can be unit-tested
without distributed runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from rtp_llm.models_py.modules.dsv4.cp import cp_padded_local_kv_lens


@dataclass(frozen=True)
class CPAttentionCommBytes:
    """Per-rank communication estimate, excluding the common indexer gather."""

    packed_kv_gather: int
    raw_q_gather: int
    topk_gather: int
    output_gather: int
    lse_gather: int

    @property
    def raw_q_merge_total(self) -> int:
        return (
            self.raw_q_gather + self.topk_gather + self.output_gather + self.lse_gather
        )


def cp_attention_comm_bytes(
    *,
    prefix_len: int,
    input_len: int,
    cp_size: int,
    compress_ratio: int,
    include_topk_gather: bool,
    packed_kv_slot_bytes: int = 584,
    num_heads: int = 64,
    head_dim: int = 512,
    topk: int = 512,
    element_bytes: int = 2,
) -> CPAttentionCommBytes:
    """Estimate per-rank communication for two CP-sharded attention choices.

    ``packed_kv_gather`` is the current path: gather compressed KV slots.
    ``raw_q_merge_total`` is the first optimized path from the plan: gather
    raw Q (and CSA topk when needed), run local attention over a disjoint KV
    shard, then gather O/LSE for logsumexp merge before the existing output
    projection.
    """
    if cp_size <= 0:
        raise ValueError(f"cp_size must be positive, got {cp_size}")
    if compress_ratio <= 0:
        raise ValueError(f"compress_ratio must be positive, got {compress_ratio}")
    if prefix_len < 0 or input_len < 0:
        raise ValueError(
            f"prefix_len and input_len must be non-negative, got {prefix_len}, {input_len}"
        )

    alpha_num = cp_size - 1
    total_compressed = (prefix_len + input_len) // compress_ratio
    packed = alpha_num * total_compressed * packed_kv_slot_bytes // cp_size
    raw_q = alpha_num * input_len * num_heads * head_dim * element_bytes // cp_size
    topk_bytes = (
        alpha_num * input_len * topk * 4 // cp_size if include_topk_gather else 0
    )
    output = alpha_num * input_len * num_heads * head_dim * element_bytes // cp_size
    lse = alpha_num * input_len * num_heads * 4 // cp_size
    return CPAttentionCommBytes(
        packed_kv_gather=packed,
        raw_q_gather=raw_q,
        topk_gather=topk_bytes,
        output_gather=output,
        lse_gather=lse,
    )


def prefer_raw_q_merge_attention(
    *,
    prefix_len: int,
    input_len: int,
    cp_size: int,
    compress_ratio: int,
    include_topk_gather: bool,
) -> bool:
    """Return True when raw-Q/O/LSE merge communicates less than KV gather."""
    if input_len <= 0:
        return False
    estimate = cp_attention_comm_bytes(
        prefix_len=prefix_len,
        input_len=input_len,
        cp_size=cp_size,
        compress_ratio=compress_ratio,
        include_topk_gather=include_topk_gather,
    )
    return estimate.raw_q_merge_total < estimate.packed_kv_gather


def prefer_raw_q_merge_attention_conservative(
    *,
    prefix_len: int,
    input_len: int,
    compress_ratio: int,
    include_topk_gather: bool,
) -> bool:
    """Plan-level conservative runtime gate.

    The raw-byte break-even is roughly ``P/T > 912`` for CSA (r=4) and
    ``P/T > 28786`` for HCA (r=128). Runtime uses round numbers to leave room
    for extra kernels, topk remap, workspace construction and NCCL latency.
    """
    if input_len <= 0:
        return False
    ratio = prefix_len / input_len
    if compress_ratio == 4 and include_topk_gather:
        return ratio >= 1024
    if compress_ratio == 128 and not include_topk_gather:
        return ratio >= 32768
    return prefer_raw_q_merge_attention(
        prefix_len=prefix_len,
        input_len=input_len,
        cp_size=2,
        compress_ratio=compress_ratio,
        include_topk_gather=include_topk_gather,
    )


def remap_topk_to_cp_local(
    topk_indices: torch.Tensor,
    *,
    per_req_total_kv_lens: torch.Tensor,
    cp_size: int,
    cp_rank: int,
    block_size: int,
    req_id_per_token: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Filter global/request-local topk to the current CP rank's local KV ids.

    Args:
        topk_indices: ``[T, K]`` int tensor. Entries are request-local
            compressed KV indices; ``-1`` means invalid.
        per_req_total_kv_lens: ``[B]`` compressed KV lengths for each request.
        cp_size/cp_rank/block_size: RR ownership rule
            ``owner = (logical_block_idx % cp_size)``.
        req_id_per_token: optional ``[T]`` request id. If omitted, B must be 1.

    Returns:
        ``[T, K]`` int32 topk where non-owned entries are ``-1`` and owned
        entries are remapped to this rank's compact local-K workspace ids.
    """
    if topk_indices.dim() != 2:
        raise ValueError(f"topk_indices must be [T,K], got {tuple(topk_indices.shape)}")
    if cp_size <= 0:
        raise ValueError(f"cp_size must be positive, got {cp_size}")
    if not (0 <= cp_rank < cp_size):
        raise ValueError(f"cp_rank({cp_rank}) out of range [0,{cp_size})")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if per_req_total_kv_lens.dim() != 1:
        raise ValueError(
            "per_req_total_kv_lens must be 1D, got "
            f"{tuple(per_req_total_kv_lens.shape)}"
        )

    device = topk_indices.device
    per_req = per_req_total_kv_lens.to(device=device, dtype=torch.int64).contiguous()
    B = int(per_req.numel())
    T = int(topk_indices.shape[0])
    if req_id_per_token is None:
        if B != 1:
            raise ValueError("req_id_per_token is required when B > 1")
        req = torch.zeros((T,), dtype=torch.int64, device=device)
    else:
        req = req_id_per_token.to(device=device, dtype=torch.int64).reshape(-1)
        if int(req.numel()) != T:
            raise ValueError(
                f"req_id_per_token length {int(req.numel())} != topk rows {T}"
            )

    local_lens = cp_padded_local_kv_lens(per_req, cp_size, block_size)
    cu_local = torch.zeros(B + 1, dtype=torch.int64, device=device)
    cu_local[1:] = torch.cumsum(local_lens, dim=0)

    idx = topk_indices.to(torch.int64)
    valid = idx >= 0
    # Keep invalid positions numerically harmless for the arithmetic below.
    safe_idx = torch.where(valid, idx, torch.zeros_like(idx))
    block = safe_idx // block_size
    token_in_block = safe_idx % block_size
    owner = block % cp_size
    local_block = block // cp_size
    local_pos = local_block * block_size + token_in_block
    req_base = cu_local.index_select(0, req).unsqueeze(1)
    req_len = per_req.index_select(0, req).unsqueeze(1)
    owned = valid & (owner == cp_rank) & (safe_idx < req_len)
    local_idx = req_base + local_pos
    return torch.where(owned, local_idx, torch.full_like(local_idx, -1)).to(torch.int32)


def build_swa_cp_local_indices(
    global_positions: torch.Tensor,
    *,
    prefix_lengths: torch.Tensor,
    cp_size: int,
    cp_rank: int,
    window_size: int,
    M: int,
    N: int,
    req_id_per_token: Optional[torch.Tensor] = None,
    owner_chunk_size: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build rank-local SWA workspace indices for logical SWA partitioning.

    SWA_KV is physically replicated on every CP rank. The raw-Q merge path
    must still partition SWA keys logically; otherwise all ranks would attend
    the same SWA window and LSE/O merge would double-count those keys. This
    helper assigns each visible SWA token to exactly one rank by
    ``owner = (global_position // owner_chunk_size) % cp_size`` and returns a
    compact per-query index list into the existing workspace layout.

    Args:
        global_positions: ``[T]`` absolute query positions.
        prefix_lengths: ``[B]`` per-request prefix/start positions.
        cp_size/cp_rank: CP partition.
        window_size: SWA attention window.
        M/N: workspace stride and compressed-region length. SWA rows start at
            ``req_id * M + N``.
        req_id_per_token: optional ``[T]`` request id. Required when B > 1.
        owner_chunk_size: owner granularity. ``1`` gives token-level balance.

    Returns:
        ``(indices, lens)`` where ``indices`` is ``[T, window_size]`` int32
        filled with compact rank-local SWA workspace indices or ``-1`` and
        ``lens`` is the valid prefix length per row.
    """
    if global_positions.dim() != 1:
        raise ValueError(
            f"global_positions must be 1D, got {tuple(global_positions.shape)}"
        )
    if prefix_lengths.dim() != 1:
        raise ValueError(
            f"prefix_lengths must be 1D, got {tuple(prefix_lengths.shape)}"
        )
    if cp_size <= 0:
        raise ValueError(f"cp_size must be positive, got {cp_size}")
    if not (0 <= cp_rank < cp_size):
        raise ValueError(f"cp_rank({cp_rank}) out of range [0,{cp_size})")
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")
    if M <= 0 or N < 0:
        raise ValueError(f"M must be positive and N non-negative, got M={M}, N={N}")
    if owner_chunk_size <= 0:
        raise ValueError(f"owner_chunk_size must be positive, got {owner_chunk_size}")

    device = global_positions.device
    T = int(global_positions.numel())
    B = int(prefix_lengths.numel())
    if req_id_per_token is None:
        if B != 1:
            raise ValueError("req_id_per_token is required when B > 1")
        req = torch.zeros((T,), dtype=torch.int64, device=device)
    else:
        req = req_id_per_token.to(device=device, dtype=torch.int64).reshape(-1)
        if int(req.numel()) != T:
            raise ValueError(
                f"req_id_per_token length {int(req.numel())} != positions {T}"
            )

    gp = global_positions.to(device=device, dtype=torch.int64).reshape(-1)
    prefix = prefix_lengths.to(device=device, dtype=torch.int64).reshape(-1)
    indices = torch.full((T, window_size), -1, dtype=torch.int32, device=device)
    lens = torch.zeros((T,), dtype=torch.int32, device=device)

    # This helper is intentionally pure/planning code. The production hot path
    # should use a Triton combine kernel with the same owner rule.
    for row in range(T):
        req_id = int(req[row].item())
        prefix_i = int(prefix[req_id].item())
        gp_i = int(gp[row].item())
        p = min(prefix_i, window_size - 1)
        gather_start = prefix_i - p
        swa_len = min(gp_i + 1, window_size)
        key_start = gp_i - swa_len + 1
        out_col = 0
        for key_pos in range(key_start, gp_i + 1):
            owner = (key_pos // owner_chunk_size) % cp_size
            if owner != cp_rank:
                continue
            indices[row, out_col] = req_id * M + N + (key_pos - gather_start)
            out_col += 1
        lens[row] = out_col
    return indices, lens
