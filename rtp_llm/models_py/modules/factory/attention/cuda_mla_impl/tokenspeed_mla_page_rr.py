"""Partial MLA using prepared query tables and the existing TokenSpeed API."""

import torch
import triton
from tokenspeed_mla import mla_decode as backend

from rtp_llm.models_py.triton_kernels.mla.lse_merge import _set_empty_partial_identity


def tokenspeed_mla_page_rr_decode(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    workspace_buffer: torch.Tensor,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    query_block_tables: torch.Tensor,
    local_causal_lens: torch.Tensor,
    max_local_seq_len: int,
    softmax_scale: float,
    output_scale: float = 1.0,
    *,
    normalize_empty: bool = True,
    local_seq_lens: torch.Tensor | None = None,
    global_causal_lens: torch.Tensor | None = None,
    cp_world: int = 1,
    cp_rank: int = 0,
    cp_interleave_size: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return partial O[B,Q,H,L] and base-2 LSE[B,Q,H].

    query_block_tables is compact int32 [B*Q,M], or [B,M] when compact
    local_seq_lens[B] supplies the per-request KV scan length. In that case
    keep the query dimension so TokenSpeed 0.2.11 can pack adjacent query/head
    rows. FP8 compact queries also require global_causal_lens[B], the final
    global exclusive bound, and the Page-RR ownership topology. BF16 and
    single-query/ragged calls use the per-query local-length path.
    local_causal_lens is compact int32 [B,Q]. ``normalize_empty``
    supplies zero output and negative-infinity LSE for empty shards when the
    immediate consumer does not already mask them.
    """
    batch, query_count, heads, dim = query.shape
    rows = batch * query_count
    multi_query = local_seq_lens is not None
    if (
        query.dtype not in (torch.bfloat16, torch.float8_e4m3fn)
        or kv_cache.dtype != query.dtype
    ):
        raise ValueError("Page-RR MLA requires matching BF16 or E4M3 query/cache")
    if dim != kv_lora_rank + qk_rope_head_dim or max_local_seq_len <= 0:
        raise ValueError("invalid Page-RR MLA dimensions or maximum local length")
    if (
        query_block_tables.ndim != 2
        or query_block_tables.shape[0] != (batch if multi_query else rows)
        or not query_block_tables.is_contiguous()
    ):
        raise ValueError("Page-RR page tables must be contiguous with one row per attention sequence")
    if (
        local_causal_lens.shape != (batch, query_count)
        or not local_causal_lens.is_contiguous()
    ):
        raise ValueError("Page-RR local causal lengths must be contiguous [B,Q]")
    if (
        query_block_tables.dtype != torch.int32
        or local_causal_lens.dtype != torch.int32
    ):
        raise ValueError("Page-RR page tables and local lengths must be int32")
    if not (
        query.device
        == kv_cache.device
        == query_block_tables.device
        == local_causal_lens.device
        == workspace_buffer.device
    ):
        raise ValueError("Page-RR MLA inputs must share one CUDA device")
    if workspace_buffer.dtype != torch.int8 or not workspace_buffer.is_contiguous():
        raise ValueError("Page-RR MLA requires a contiguous int8 workspace")
    if multi_query and (
        local_seq_lens.shape != (batch,)
        or local_seq_lens.dtype != torch.int32
        or local_seq_lens.device != query.device
        or not local_seq_lens.is_contiguous()
    ):
        raise ValueError("Page-RR KV scan lengths must be contiguous CUDA int32 [B]")
    if multi_query:
        if query.dtype != torch.float8_e4m3fn or query_count <= 1:
            raise ValueError(
                "TokenSpeed 0.2.11 compact Page-RR queries require multi-query FP8"
            )
        if (
            global_causal_lens is None
            or global_causal_lens.shape != (batch,)
            or global_causal_lens.dtype != torch.int32
            or global_causal_lens.device != query.device
            or not global_causal_lens.is_contiguous()
            or cp_world <= 1
            or not 0 <= cp_rank < cp_world
            or cp_interleave_size <= 0
        ):
            raise ValueError(
                "Compact Page-RR queries require global bounds and valid CP ownership"
            )
    elif global_causal_lens is not None or (cp_world, cp_rank, cp_interleave_size) != (
        1,
        0,
        1,
    ):
        raise ValueError("Flat Page-RR queries use local lengths without CP metadata")
    shape = (batch, query_count, heads, kv_lora_rank)
    # Both compute modes expose the same BF16 partial-output contract.
    out = torch.empty(shape, device=query.device, dtype=torch.bfloat16)
    _, lse = backend.tokenspeed_mla_decode(
        query=query if multi_query else query.view(rows, 1, heads, dim),
        kv_cache=kv_cache,
        workspace_buffer=workspace_buffer,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        block_tables=query_block_tables,
        seq_lens=local_seq_lens if multi_query else local_causal_lens.view(-1),
        max_seq_len=max_local_seq_len,
        softmax_scale=softmax_scale,
        output_scale=output_scale,
        out=out if multi_query else out.view(rows, 1, heads, kv_lora_rank),
        is_var_seq=True,
        causal_mask=True,
        enable_pdl=False,
        return_lse=True,
        enable_packed_q=multi_query,
        **(
            {
                "causal_seqs": global_causal_lens,
                "cp_world": cp_world,
                "cp_rank": cp_rank,
                "cp_interleave_size": cp_interleave_size,
            }
            if multi_query
            else {}
        ),
    )
    lse = lse.view(batch, query_count, heads)
    if normalize_empty:
        # Standalone callers observe the exact merge identity for empty shards.
        _set_empty_partial_identity[(rows,)](
            out,
            lse,
            local_causal_lens,
            elements=heads * kv_lora_rank,
            heads=heads,
            block_size=256,
            lse_block_size=triton.next_power_of_2(heads),
        )
    return out, lse
