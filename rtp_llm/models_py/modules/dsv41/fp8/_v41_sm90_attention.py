"""Hopper adapter: packed V4.1 storage, selected-row BF16 FlashMLA compute.

The persistent SWA/GLOBAL cache ABI stays 528/288 bytes. Only the selected
rows are decoded; staging is bounded independently of cache capacity. This
is explicitly a BF16 compute backend, not native FP4 or FP8 attention.
"""

import torch

from rtp_llm.models_py.modules.dsv41.flash_mla import flash_mla_sparse_fwd
from rtp_llm.models_py.modules.dsv41.fp8._v41_fp4_triton import (
    dequantize_k_cache_slots_fp4,
)
from rtp_llm.models_py.modules.dsv41.fp8._v41_swa_triton import (
    dequantize_swa_k_cache,
)


def _slots_with_lengths(slots, lengths, batch, query_len):
    slots = slots.reshape(batch * query_len, -1)
    if lengths is None:
        return slots
    if lengths.device != slots.device or lengths.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("Selection lengths must be device-side int32/int64")
    if lengths.numel() == batch:
        lengths = lengths.reshape(batch, 1).expand(batch, query_len).reshape(-1)
    elif lengths.numel() == batch * query_len:
        lengths = lengths.reshape(-1)
    else:
        raise ValueError("Selection lengths must have B or B*S elements")
    columns = torch.arange(slots.shape[1], device=slots.device)
    return torch.where(columns[None] < lengths[:, None], slots, -1)


def selected_rows_attention(
    q,
    swa_pool,
    swa_slots,
    sink,
    softmax_scale,
    *,
    global_pool=None,
    global_slots=None,
    swa_lengths=None,
    global_lengths=None,
    return_lse=False,
):
    """Same output ABI as the native paged op, including its unsunk LSE.

    A single sparse softmax consumes both pools and the sink. Chunking is
    static in query count, so graph replay may update slots and lengths.
    The adapter never re-quantizes cache values. A future FP8 consumer can
    implement this boundary separately with its own accuracy qualification.
    """
    if q.dtype != torch.bfloat16 or q.ndim != 4 or q.shape[-1] != 512:
        raise ValueError("SM90 V4.1 attention requires BF16 [B,S,H,512] Q")
    if swa_pool.dtype != torch.uint8 or swa_pool.shape[-1] != 528:
        raise ValueError("SM90 V4.1 attention requires the 528B SWA pool")
    if (global_pool is None) != (global_slots is None):
        raise ValueError("GLOBAL pool and slots must be provided together")
    if global_pool is not None and (
        global_pool.dtype != torch.uint8 or global_pool.shape[-1] != 288
    ):
        raise ValueError("SM90 V4.1 attention requires the 288B GLOBAL pool")
    batch, query_len, heads, dim = q.shape
    if not 0 < heads <= 128:
        raise ValueError("Hopper sparse FlashMLA supports at most 128 query heads")
    rows = batch * query_len
    output = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    lse = torch.empty((rows, heads), dtype=torch.float32, device=q.device)
    if rows == 0:
        return (
            (output, lse.view(batch, query_len, heads).transpose(1, 2))
            if return_lse
            else output
        )
    swa = _slots_with_lengths(swa_slots, swa_lengths, batch, query_len)
    glob = (
        _slots_with_lengths(global_slots, global_lengths, batch, query_len)
        if global_slots is not None
        else None
    )
    width = swa.shape[1] + (0 if glob is None else glob.shape[1])
    if width == 0:
        raise ValueError("SM90 V4.1 attention needs a nonempty selection width")
    # Each of the gathered input and concatenated staging tensors is <=16MiB
    # for ordinary model widths. At least one complete query must fit.
    chunk_rows = max(1, (16 * 1024 * 1024) // (width * dim * 2))
    flat_q = q.reshape(rows, heads, dim)
    flat_out = output.view(rows, heads, dim)
    for start in range(0, rows, chunk_rows):
        end = min(rows, start + chunk_rows)
        swa_part = swa[start:end]
        values = dequantize_swa_k_cache(swa_pool, swa_part).view(
            end - start, swa.shape[1], dim
        )
        selected = swa_part
        if glob is not None:
            global_part = glob[start:end]
            global_values = dequantize_k_cache_slots_fp4(global_pool, global_part).view(
                end - start, glob.shape[1], dim
            )
            values = torch.cat((values, global_values), dim=1)
            selected = torch.cat((selected, global_part), dim=1)
        indices = torch.arange(
            (end - start) * width, dtype=torch.int32, device=q.device
        ).view(end - start, width)
        indices = torch.where(selected >= 0, indices, -1).unsqueeze(1)
        actual, _, actual_lse = flash_mla_sparse_fwd(
            flat_q[start:end].contiguous(),
            values.reshape(-1, 1, dim),
            indices,
            softmax_scale,
            attn_sink=sink,
        )
        # The paged decode ABI reports +inf for an empty row. A sparse-prefill
        # provider may instead emit -inf/NaN; empty selections contribute zero.
        nonempty = (selected >= 0).any(dim=-1)
        flat_out[start:end].copy_(torch.where(nonempty[:, None, None], actual, 0.0))
        lse[start:end].copy_(torch.where(nonempty[:, None], actual_lse, float("inf")))
    if return_lse:
        return output, lse.view(batch, query_len, heads).transpose(1, 2).contiguous()
    return output
