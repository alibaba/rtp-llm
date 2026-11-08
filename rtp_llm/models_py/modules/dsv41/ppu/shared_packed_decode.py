"""Ordinary eager decode over the framework's original packed KV pools.

The shared coordinator owns support checks and physical-slot construction.
This adapter consumes its zero-copy 528B SWA / 288B global views directly;
only Q may be padded or made contiguous. BF16 Q does not select the separate
Q8/KV8/P8 quantization contract.

Native ABI reference: FlashMLA-for-sail 4305d1bdaeb8451eb5d78c67a5122ecf38f5014d,
flash_mla/flash_mla_interface.py, SHA256
8861f77da7d7900ffeb60cb2f652176d4342c947d516fbcc0928eba81e952d43.
"""

import torch

from rtp_llm.models_py.modules.dsv41.ppu.attention import _pad_heads


def packed_decode(q, plan, *, softmax_scale, attn_sink):
    """Return BF16 ``[rows, real_heads, 512]`` from a prequalified plan.

    ``plan`` retains every cache view, physical index tensor and actual live
    prefix length until this current-stream launch has been submitted. Each
    row is one independent ordinary decode query. Graph, verify, DSpark,
    idle and nonresident SWA histories are rejected before writes by the
    caller; there is deliberately no exception-triggered retry here.
    """
    if q.ndim != 3 or q.shape[-1] != 512 or q.dtype != torch.bfloat16:
        raise ValueError("packed decode requires BF16 [rows, heads, 512] Q")
    if q.shape[0] == 0:
        return q.new_empty(q.shape)

    query, sink, heads = _pad_heads(q, attn_sink)
    # Schedulers are not reused across lengths/requests: the pinned ABI
    # permits reuse only when values as well as shapes remain unchanged.
    metadata, num_splits = plan.metadata_factory()
    output, _ = plan.native(
        q=query.unsqueeze(1).contiguous(),
        k_cache=plan.swa_cache,
        block_table=None,
        cache_seqlens=None,
        head_dim_v=512,
        tile_scheduler_metadata=metadata,
        num_splits=num_splits,
        softmax_scale=softmax_scale,
        causal=False,
        is_fp8_kvcache=True,
        indices=plan.swa_indices,
        attn_sink=sink,
        extra_k_cache=plan.global_cache,
        extra_indices_in_kvcache=plan.global_indices,
        topk_length=plan.swa_lengths,
        extra_topk_length=plan.global_lengths,
    )
    output = output[:, 0, :heads]
    return torch.where(
        plan.valid_mask[:, None, None], output, torch.zeros_like(output)
    )
