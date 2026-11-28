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

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except Exception:  # pragma: no cover
    triton = None
    tl = None
    _TRITON_AVAILABLE = False

from rtp_llm.models_py.modules.dsv4.cp import cp_padded_local_kv_lens


if _TRITON_AVAILABLE:

    @triton.jit(
        do_not_specialize=[
            "T",
            "K",
            "in_stride0",
            "in_stride1",
            "out_stride0",
            "out_stride1",
            "req_stride",
            "per_req_stride",
            "cu_cmp_stride",
            "cp_size",
            "cp_rank",
            "block_size",
        ]
    )
    def _remap_topk_to_cp_local_kernel(
        topk_ptr,
        per_req_ptr,
        req_ptr,
        cu_cmp_ptr,
        out_ptr,
        T,
        K,
        in_stride0,
        in_stride1,
        out_stride0,
        out_stride1,
        req_stride,
        per_req_stride,
        cu_cmp_stride,
        cp_size,
        cp_rank,
        block_size,
        BLOCK_M: tl.constexpr,
        BLOCK_K: tl.constexpr,
        HAS_REQ_IDS: tl.constexpr,
        HAS_CU_CMP: tl.constexpr,
    ):
        """Fused CP top-k ownership filtering and local workspace index remap."""
        pid_m = tl.program_id(0).to(tl.int64)
        pid_k = tl.program_id(1).to(tl.int64)

        row_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        m_mask = row_m < T
        if HAS_REQ_IDS:
            req = tl.load(req_ptr + row_m * req_stride, mask=m_mask, other=0).to(
                tl.int64
            )
        else:
            req = tl.zeros((BLOCK_M,), dtype=tl.int64)
        req_len = tl.load(per_req_ptr + req * per_req_stride, mask=m_mask, other=0).to(
            tl.int64
        )
        if HAS_CU_CMP:
            req_base = tl.load(
                cu_cmp_ptr + req * cu_cmp_stride, mask=m_mask, other=0
            ).to(tl.int64)
        else:
            req_base = tl.zeros((BLOCK_M,), dtype=tl.int64)

        col_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = col_k < K
        tile_mask = m_mask[:, None] & k_mask[None, :]
        idx = tl.load(
            topk_ptr
            + row_m[:, None] * in_stride0.to(tl.int64)
            + col_k[None, :] * in_stride1.to(tl.int64),
            mask=tile_mask,
            other=-1,
        ).to(tl.int64)
        valid = idx >= 0
        safe_idx = tl.where(valid, idx, 0)
        block = safe_idx // block_size
        token_in_block = safe_idx % block_size
        owner = block % cp_size
        local_block = block // cp_size
        local_pos = local_block * block_size + token_in_block
        owned = valid & (owner == cp_rank) & (safe_idx < req_len[:, None])
        out_val = tl.where(owned, req_base[:, None] + local_pos, -1).to(tl.int32)
        tl.store(
            out_ptr
            + row_m[:, None] * out_stride0.to(tl.int64)
            + col_k[None, :] * out_stride1.to(tl.int64),
            out_val,
            mask=tile_mask,
        )

    @triton.jit(
        do_not_specialize=[
            "T",
            "window",
            "M",
            "N",
            "gp_stride",
            "req_stride",
            "prefix_stride",
            "cu_swa_stride",
            "out_stride0",
            "cp_size",
            "cp_rank",
            "owner_chunk_size",
        ]
    )
    def _build_swa_cp_local_indices_kernel(
        gp_ptr,
        req_ptr,
        prefix_ptr,
        cu_swa_ptr,
        indices_ptr,
        lens_ptr,
        T,
        window,
        M,
        N,
        gp_stride,
        req_stride,
        prefix_stride,
        cu_swa_stride,
        out_stride0,
        cp_size,
        cp_rank,
        owner_chunk_size,
        BLOCK_W: tl.constexpr,
        HAS_REQ_IDS: tl.constexpr,
        HAS_CU_SWA: tl.constexpr,
    ):
        """Single-pass compact CP-owned SWA workspace indices without D2H sync."""
        t = tl.program_id(0).to(tl.int64)
        if t >= T:
            return
        if HAS_REQ_IDS:
            req = tl.load(req_ptr + t * req_stride).to(tl.int64)
        else:
            req = tl.zeros((), dtype=tl.int64)
        gp = tl.load(gp_ptr + t * gp_stride).to(tl.int64)
        prefix = tl.load(prefix_ptr + req * prefix_stride).to(tl.int64)
        p_row = tl.minimum(prefix, window - 1)
        gather_start = prefix - p_row
        swa_len = tl.minimum(gp + 1, window)
        key_start = gp - swa_len + 1
        if HAS_CU_SWA:
            ws_base = tl.load(cu_swa_ptr + req * cu_swa_stride).to(tl.int64)
        else:
            ws_base = req * M.to(tl.int64) + N.to(tl.int64)

        out_row_ptr = indices_ptr + t * out_stride0.to(tl.int64)
        write_col = tl.zeros((), dtype=tl.int32)
        for w_start in range(0, window, BLOCK_W):
            col_offset = w_start + tl.arange(0, BLOCK_W)
            valid = col_offset < swa_len
            key_pos = key_start + col_offset
            owned = (
                valid
                & (key_pos >= 0)
                & (((key_pos // owner_chunk_size) % cp_size) == cp_rank)
            )
            owned_i32 = owned.to(tl.int32)
            local_ranks = tl.cumsum(owned_i32, axis=0) - 1
            dst_col = write_col + local_ranks
            target_val = (ws_base + (key_pos - gather_start)).to(tl.int32)
            tl.store(out_row_ptr + dst_col, target_val, mask=owned)
            write_col += tl.sum(owned_i32, axis=0)

        for w_start in range(0, window, BLOCK_W):
            col_offset = w_start + tl.arange(0, BLOCK_W)
            tail_mask = (col_offset >= write_col) & (col_offset < window)
            tl.store(out_row_ptr + col_offset, -1, mask=tail_mask)

        tl.store(lens_ptr + t, write_col)


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
    # Path A — packed KV gather: total compressed K spans prefix + input,
    # each rank receives (C-1)/C of the global pool (its non-owned share).
    packed = alpha_num * total_compressed * packed_kv_slot_bytes // cp_size
    # Path B — raw Q / topk / O / LSE gathers. Q (and CSA topk) are sharded
    # token-wise across cp ranks (T/C per rank); after AG each rank receives
    # (C-1)/C × T worth of bytes. O and LSE are NOT sharded: each rank
    # produces a partial O and partial LSE for the FULL T queries against
    # its local KV shard (used by the subsequent logsumexp merge), so each
    # rank sends T × H × D (resp. T × H × 4) and receives (C-1) × that.
    # The strip-back to own T/C rows happens only AFTER merge — it saves
    # GPU memory, not network bytes.
    raw_q = alpha_num * input_len * num_heads * head_dim * element_bytes // cp_size
    topk_bytes = (
        alpha_num * input_len * topk * 4 // cp_size if include_topk_gather else 0
    )
    output = alpha_num * input_len * num_heads * head_dim * element_bytes
    lse = alpha_num * input_len * num_heads * 4
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
    """Plan-level conservative runtime gate (cp_size=2 default geometry).

    With output_gather and lse_gather corrected to NOT divide by cp_size
    (each rank's partial O/LSE covers the FULL T queries), the raw-byte
    break-even tightens to ``P/T >= 1363`` for CSA (r=4) and ``P/T >= 43227``
    for HCA (r=128) at cp_size=2 with DSV4-Flash/Pro dims
    (H=64, head_dim=512, BF16). Runtime uses round numbers with ~10% safety
    margin to account for NCCL launch latency and slight workload variance.
    """
    if input_len <= 0:
        return False
    ratio = prefix_len / input_len
    if compress_ratio == 4 and include_topk_gather:
        return ratio >= 1500
    if compress_ratio == 128 and not include_topk_gather:
        return ratio >= 48000
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
    cu_cmp_offsets: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Filter global/request-local topk to the current CP rank's local KV ids.

    Args:
        topk_indices: ``[T, K]`` int tensor. Entries are request-local
            compressed KV indices; ``-1`` means invalid.
        per_req_total_kv_lens: ``[B]`` compressed KV lengths for each request.
        cp_size/cp_rank/block_size: RR ownership rule
            ``owner = (logical_block_idx % cp_size)``.
        req_id_per_token: optional ``[T]`` request id. If omitted, B must be 1.
        cu_cmp_offsets: optional ``[>= B]`` ragged destination base offset per
            request in a flat 2D local workspace. When omitted, defaults to
            packed ``cu_local`` offsets.

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
    B = int(per_req_total_kv_lens.numel())
    T = int(topk_indices.shape[0])
    K = int(topk_indices.shape[1])

    if topk_indices.is_cuda and _TRITON_AVAILABLE:
        if req_id_per_token is None:
            if B != 1:
                raise ValueError("req_id_per_token is required when B > 1")
            has_req_ids = False
            req_t = topk_indices
        else:
            req_t = req_id_per_token.reshape(-1)
            if int(req_t.numel()) != T:
                raise ValueError(
                    f"req_id_per_token length {int(req_t.numel())} != topk rows {T}"
                )
            if req_t.device != device or req_t.dtype not in (torch.int32, torch.int64):
                req_t = req_t.to(device=device, dtype=torch.int64)
            has_req_ids = True

        per_req_t = per_req_total_kv_lens
        if per_req_t.device != device or per_req_t.dtype not in (
            torch.int32,
            torch.int64,
        ):
            per_req_t = per_req_t.to(device=device, dtype=torch.int64)

        if cu_cmp_offsets is not None:
            cu_cmp_t = cu_cmp_offsets.reshape(-1)
            if int(cu_cmp_t.numel()) < B:
                raise ValueError(
                    f"cu_cmp_offsets length {int(cu_cmp_t.numel())} < batch size {B}"
                )
            if cu_cmp_t.device != device or cu_cmp_t.dtype not in (
                torch.int32,
                torch.int64,
            ):
                cu_cmp_t = cu_cmp_t.to(device=device, dtype=torch.int64)
            has_cu_cmp = True
        elif B == 1:
            cu_cmp_t = per_req_t
            has_cu_cmp = False
        else:
            local_lens = cp_padded_local_kv_lens(per_req_t, cp_size, block_size)
            cu_cmp_t = torch.empty(B + 1, dtype=torch.int64, device=device)
            cu_cmp_t[0] = 0
            cu_cmp_t[1:] = torch.cumsum(local_lens, dim=0)
            has_cu_cmp = True

        if T == 0 or K == 0:
            return torch.empty((T, K), dtype=torch.int32, device=device)

        out = torch.empty((T, K), dtype=torch.int32, device=device)
        block_m = 4
        block_k = 256
        _remap_topk_to_cp_local_kernel[
            (triton.cdiv(T, block_m), triton.cdiv(K, block_k))
        ](
            topk_indices,
            per_req_t,
            req_t if has_req_ids else per_req_t,
            cu_cmp_t,
            out,
            T=T,
            K=K,
            in_stride0=int(topk_indices.stride(0)),
            in_stride1=int(topk_indices.stride(1)),
            out_stride0=int(out.stride(0)),
            out_stride1=int(out.stride(1)),
            req_stride=int(req_t.stride(0)) if has_req_ids else 1,
            per_req_stride=int(per_req_t.stride(0)),
            cu_cmp_stride=int(cu_cmp_t.stride(0)),
            cp_size=int(cp_size),
            cp_rank=int(cp_rank),
            block_size=int(block_size),
            BLOCK_M=block_m,
            BLOCK_K=block_k,
            HAS_REQ_IDS=has_req_ids,
            HAS_CU_CMP=has_cu_cmp,
            num_warps=4,
        )
        return out

    per_req = per_req_total_kv_lens.to(device=device, dtype=torch.int64).contiguous()
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

    if cu_cmp_offsets is not None:
        cu_cmp = cu_cmp_offsets.to(device=device, dtype=torch.int64).reshape(-1)
        if int(cu_cmp.numel()) < B:
            raise ValueError(
                f"cu_cmp_offsets length {int(cu_cmp.numel())} < batch size {B}"
            )
        req_base = cu_cmp.index_select(0, req).unsqueeze(1)
    else:
        local_lens = cp_padded_local_kv_lens(per_req, cp_size, block_size)
        cu_local = torch.zeros(B + 1, dtype=torch.int64, device=device)
        cu_local[1:] = torch.cumsum(local_lens, dim=0)
        req_base = cu_local.index_select(0, req).unsqueeze(1)

    idx = topk_indices.to(torch.int64)
    valid = idx >= 0
    # Keep invalid positions numerically harmless for the arithmetic below.
    safe_idx = torch.where(valid, idx, torch.zeros_like(idx))
    block = safe_idx // block_size
    token_in_block = safe_idx % block_size
    owner = block % cp_size
    local_block = block // cp_size
    local_pos = local_block * block_size + token_in_block
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
    M: int = 0,
    N: int = 0,
    req_id_per_token: Optional[torch.Tensor] = None,
    owner_chunk_size: int = 1,
    cu_swa_offsets: Optional[torch.Tensor] = None,
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
        M/N: workspace stride and compressed-region length when
            ``cu_swa_offsets is None`` (SWA rows start at ``req_id * M + N``).
        req_id_per_token: optional ``[T]`` request id. Required when B > 1.
        owner_chunk_size: owner granularity. ``1`` gives token-level balance.
        cu_swa_offsets: optional ``[>= B]`` ragged SWA start row per request in
            a flat 2D workspace.

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
    if cu_swa_offsets is None and (M <= 0 or N < 0):
        raise ValueError(f"M must be positive and N non-negative, got M={M}, N={N}")
    if owner_chunk_size <= 0:
        raise ValueError(f"owner_chunk_size must be positive, got {owner_chunk_size}")

    device = global_positions.device
    T = int(global_positions.numel())
    B = int(prefix_lengths.numel())
    window = int(window_size)
    if cu_swa_offsets is not None:
        cu_swa = cu_swa_offsets.reshape(-1)
        if int(cu_swa.numel()) < B:
            raise ValueError(
                f"cu_swa_offsets length {int(cu_swa.numel())} < batch size {B}"
            )
    else:
        cu_swa = None

    if global_positions.is_cuda and _TRITON_AVAILABLE:
        if req_id_per_token is None:
            if B != 1:
                raise ValueError("req_id_per_token is required when B > 1")
            has_req_ids = False
            req_t = global_positions
        else:
            req_t = req_id_per_token.reshape(-1)
            if int(req_t.numel()) != T:
                raise ValueError(
                    f"req_id_per_token length {int(req_t.numel())} != positions {T}"
                )
            if req_t.device != device or req_t.dtype not in (torch.int32, torch.int64):
                req_t = req_t.to(device=device, dtype=torch.int64)
            has_req_ids = True

        if T == 0:
            return (
                torch.empty((0, window), dtype=torch.int32, device=device),
                torch.empty((0,), dtype=torch.int32, device=device),
            )

        gp_t = global_positions.reshape(-1)
        if gp_t.device != device or gp_t.dtype not in (torch.int32, torch.int64):
            gp_t = gp_t.to(device=device, dtype=torch.int64)
        prefix_t = prefix_lengths.reshape(-1)
        if prefix_t.device != device or prefix_t.dtype not in (
            torch.int32,
            torch.int64,
        ):
            prefix_t = prefix_t.to(device=device, dtype=torch.int64)
        if cu_swa is not None:
            if cu_swa.device != device or cu_swa.dtype not in (
                torch.int32,
                torch.int64,
            ):
                cu_swa = cu_swa.to(device=device, dtype=torch.int64)
            has_cu_swa = True
        else:
            cu_swa = gp_t
            has_cu_swa = False

        indices = torch.empty((T, window), dtype=torch.int32, device=device)
        lens = torch.empty((T,), dtype=torch.int32, device=device)
        block_w = 512 if window > 256 else 256
        _build_swa_cp_local_indices_kernel[(T,)](
            gp_t,
            req_t if has_req_ids else gp_t,
            prefix_t,
            cu_swa,
            indices,
            lens,
            T=T,
            window=window,
            M=int(M),
            N=int(N),
            gp_stride=int(gp_t.stride(0)),
            req_stride=int(req_t.stride(0)) if has_req_ids else 1,
            prefix_stride=int(prefix_t.stride(0)),
            cu_swa_stride=int(cu_swa.stride(0)),
            out_stride0=int(indices.stride(0)),
            cp_size=int(cp_size),
            cp_rank=int(cp_rank),
            owner_chunk_size=int(owner_chunk_size),
            BLOCK_W=block_w,
            HAS_REQ_IDS=has_req_ids,
            HAS_CU_SWA=has_cu_swa,
            num_warps=4,
        )
        return indices, lens

    if cu_swa is not None:
        cu_swa = cu_swa.to(device=device, dtype=torch.int64)
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

    prefix_per_row = prefix.index_select(0, req)  # [T]
    p_per_row = torch.clamp(prefix_per_row, max=window - 1)  # [T]
    gather_start_per_row = prefix_per_row - p_per_row  # [T]
    swa_len_per_row = torch.clamp(gp + 1, max=window)  # [T]
    key_start_per_row = gp - swa_len_per_row + 1  # [T]

    col_offset = torch.arange(window, device=device, dtype=torch.int64).unsqueeze(
        0
    )  # [1, W]
    key_pos = key_start_per_row.unsqueeze(1) + col_offset  # [T, W]
    valid_in_window = col_offset < swa_len_per_row.unsqueeze(1)  # [T, W] bool
    owned_mask = (
        ((key_pos // owner_chunk_size) % cp_size) == cp_rank
    ) & valid_in_window  # [T, W] bool

    # Target workspace index for each (row, key_pos) cell — only meaningful
    # where owned_mask is True; the rest get filtered out below.
    if cu_swa is not None:
        swa_ws_base = cu_swa.index_select(0, req).unsqueeze(1)
    else:
        swa_ws_base = req.unsqueeze(1) * M + N
    target_value = swa_ws_base + (key_pos - gather_start_per_row.unsqueeze(1))  # [T, W]

    # Per-row rank of each owned cell (0, 1, 2, ... in row-scan order).
    # cumsum on bool produces the 1-indexed rank at each owned cell.
    rank_in_row = owned_mask.cumsum(dim=1) - 1  # [T, W] int64

    # Flatten owned cells and scatter into [T, W] output at (row, rank_in_row).
    row_idx_grid = (
        torch.arange(T, device=device, dtype=torch.int64)
        .unsqueeze(1)
        .expand(-1, window)
    )  # [T, W]
    flat_mask = owned_mask.reshape(-1)
    flat_row = row_idx_grid.reshape(-1).masked_select(flat_mask)
    flat_col = rank_in_row.reshape(-1).masked_select(flat_mask)
    flat_val = target_value.reshape(-1).masked_select(flat_mask).to(torch.int32)

    indices = torch.full((T, window), -1, dtype=torch.int32, device=device)
    if flat_row.numel() > 0:
        indices.index_put_((flat_row, flat_col), flat_val, accumulate=False)
    lens = owned_mask.sum(dim=1).to(torch.int32)
    return indices, lens
