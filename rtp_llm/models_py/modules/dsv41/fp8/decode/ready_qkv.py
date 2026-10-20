"""Forward-local REUSE Q/KV/physical-index producer and FlashMLA PDL edge.

FULL/REINDEX must score the current Index Q; they cannot use ready indices.
No device values or cross-forward pointer memoization participate in admission.
"""

from dataclasses import dataclass
import os

import torch

from .. import _v41_qb_kv as producer


@dataclass
class ReadyIndexPlan:
    req: torch.Tensor
    table: torch.Tensor
    local: torch.Tensor
    logical_entries: int
    entries: int
    pdl: bool = False
    positions: torch.Tensor | None = None

    def supports(self, device, rows):
        return (
            0 < self.logical_entries <= 2**31 - 1
            and 0 < self.entries <= 2**31 - 1
            and self.req.shape == (rows,)
            and self.req.is_contiguous()
            and self.table.ndim == 2
            and min(self.table.shape) > 0
            and self.table.stride(1) == 1
            and self.table.stride(0) >= self.table.shape[1]
            and self.local.ndim == 2
            and self.local.shape[0] == rows
            and self.local.shape[1] > 0
            and self.local.shape[1] % 128 == 0
            and self.local.stride(1) == 1
            and self.local.stride(0) >= self.local.shape[1]
            and all(
                t.device == device and t.dtype == torch.int32 and not t.requires_grad
                for t in (self.req, self.table, self.local)
            )
        )


def make_ready_plan(attn, mega, metadata, rows, swa_pool, swa_indices):
    if (
        os.environ.get("DSV41_FUSED_READY_QKV", "0") != "1"
        or mega is None
        or not attn.compress_ratio
        or attn.is_kv_source
        or attn.is_index_source
        or (mega.heads, mega.padded_heads, mega.groups) != (64, 64, 8)
        or mega.merged_qb is not None
        or rows not in producer._READY_CONFIGS
    ):
        return None
    cp = attn._cp_ctx
    if cp is not None and cp.cp_size > 1 and cp.kv_cache_sharded:
        return None
    selected = attn._shared_attention.get("topk", {}).get(attn.index_source_layer_id)
    if selected is None:
        return None
    from .._kv_cache_utils import require_pool_tokens_per_block

    region = attn._global_region()
    pool = attn._source_pool(region)
    raw_tpb = require_pool_tokens_per_block(attn._kv_cache, region=region)
    if raw_tpb % attn.compress_ratio or raw_tpb // attn.compress_ratio != pool.shape[1]:
        return None
    plan = ReadyIndexPlan(
        metadata.req_id_per_token[:rows],
        attn._block_tables_by_type[region],
        selected,
        raw_tpb // attn.compress_ratio,
        pool.shape[1],
    )
    device = mega.query_weight.device
    if (
        not plan.supports(device, rows)
        or not producer._device_supported(device)
        or not mega.supports_decode(swa_pool, swa_indices, pool, selected)
    ):
        return None
    ops = producer._ops()
    if not getattr(ops, "dsv41_qb_kv_available", lambda: False)() or not callable(
        getattr(ops, "dsv41_qb_kv_index_out", None)
    ):
        return None
    # Conversion, if needed, must precede the producer. Inserting a cast after
    # it would create an ordinary dependency and defeat the direct PDL edge.
    plan.positions = attn._mega_position_chunks(metadata.position_ids, rows, rows)[0]
    plan.pdl = (
        os.environ.get("DSV41_READY_FLASHMLA_PDL", "1") == "1"
        and getattr(ops, "dsv41_flashmla_pdl_available", lambda: False)()
        and callable(getattr(ops, "dsv41_flashmla_decode_out", None))
        and _consumer_supported(mega, plan.positions, rows)
    )
    return plan


def _consumer_supported(mega, positions, rows):
    return (
        (mega.heads, mega.padded_heads, mega.groups) == (64, 64, 8)
        and positions.shape == (rows,)
        and positions.dtype == torch.int32
        and positions.is_contiguous()
        and positions.device == mega.query_weight.device
        and mega.cos_sin.dtype == torch.float32
        and mega.cos_sin.ndim == 2
        and mega.cos_sin.shape[1] == 64
        and mega.cos_sin.is_contiguous()
        and mega.sink.dtype == torch.float32
        and mega.sink.shape == (64,)
        and mega.sink.is_contiguous()
        and mega.cos_sin.device == mega.sink.device == positions.device
    )


def flashmla_decode(
    mega,
    q,
    pool,
    indices,
    positions,
    extra,
    extra_indices,
    *,
    swa_lengths=None,
    extra_lengths=None
):
    """Use the source-pinned native consumer; no JIT or global PDL switch."""
    if swa_lengths is not None or extra_lengths is not None:
        raise ValueError(
            "REUSE PDL consumer uses masked physical indices, without length overrides"
        )
    rows = q.shape[0]
    output = torch.empty((rows, 8, 4096), device=q.device, dtype=torch.float8_e4m3fn)
    scales = torch.empty(
        (8, 32, (rows + 3) // 4 * 4), device=q.device, dtype=torch.int32
    ).permute(2, 0, 1)[:rows]
    lse = torch.empty((rows, 64), device=q.device, dtype=torch.float32)
    producer._ops().dsv41_flashmla_decode_out(
        q,
        pool,
        indices,
        extra,
        extra_indices,
        positions,
        mega.cos_sin,
        mega.sink,
        output,
        scales,
        lse,
        mega.softmax_scale,
        True,
    )
    return output, scales
