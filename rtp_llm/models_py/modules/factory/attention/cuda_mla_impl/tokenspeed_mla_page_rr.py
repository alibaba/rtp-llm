"""Partial MLA using prepared query tables and the existing TokenSpeed API."""

from typing import Optional

import torch
import triton
import triton.language as tl
from tokenspeed_mla import mla_decode as backend


@triton.jit
def _set_empty_partial_identity(
    output,
    lse,
    lengths,
    elements: tl.constexpr,
    heads: tl.constexpr,
    block_size: tl.constexpr,
    lse_block_size: tl.constexpr,
):
    row = tl.program_id(0)
    if tl.load(lengths + row) == 0:
        for start in tl.range(0, elements, block_size):
            offsets = start + tl.arange(0, block_size)
            tl.store(output + row * elements + offsets, 0.0, offsets < elements)
        offsets = tl.arange(0, lse_block_size)
        tl.store(lse + row * heads + offsets, -float("inf"), offsets < heads)


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
    out: Optional[torch.Tensor] = None,
    enable_pdl: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return partial O[B,Q,H,L] and base-2 LSE[B,Q,H].

    query_block_tables is compact int32 [B*Q,M], prepared once before the
    layer loop. local_causal_lens is compact int32 [B,Q]. Empty shards return
    zero output and negative-infinity LSE.
    """
    batch, query_count, heads, dim = query.shape
    rows = batch * query_count
    if (
        query.dtype not in (torch.bfloat16, torch.float16)
        or kv_cache.dtype != query.dtype
    ):
        raise ValueError("Page-RR MLA requires matching BF16 or FP16 query/cache")
    if dim != kv_lora_rank + qk_rope_head_dim or max_local_seq_len <= 0:
        raise ValueError("invalid Page-RR MLA dimensions or maximum local length")
    if (
        query_block_tables.ndim != 2
        or query_block_tables.shape[0] != rows
        or not query_block_tables.is_contiguous()
    ):
        raise ValueError("Page-RR query page tables must be contiguous [B*Q,M]")
    if (
        local_causal_lens.shape != (batch, query_count)
        or not local_causal_lens.is_contiguous()
    ):
        raise ValueError("Page-RR local causal lengths must be contiguous [B,Q]")
    if query_block_tables.dtype != torch.int32 or local_causal_lens.dtype != torch.int32:
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
    shape = (batch, query_count, heads, kv_lora_rank)
    if out is None:
        out = torch.empty(shape, device=query.device, dtype=query.dtype)
    elif (
        out.shape != shape
        or out.dtype != query.dtype
        or out.device != query.device
        or not out.is_contiguous()
    ):
        raise ValueError(
            "Page-RR output must be contiguous [B,Q,H,L] with query dtype/device"
        )
    _, lse = backend.tokenspeed_mla_decode(
        query=query.view(rows, 1, heads, dim),
        kv_cache=kv_cache,
        workspace_buffer=workspace_buffer,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        block_tables=query_block_tables,
        seq_lens=local_causal_lens.view(-1),
        max_seq_len=max_local_seq_len,
        softmax_scale=softmax_scale,
        output_scale=output_scale,
        out=out.view(rows, 1, heads, kv_lora_rank),
        is_var_seq=True,
        causal_mask=True,
        enable_pdl=enable_pdl,
        return_lse=True,
    )
    lse = lse.view(batch, query_count, heads)
    # Empty outputs may be unwritten on replay. Supply the exact merge identity.
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
