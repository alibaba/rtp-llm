"""FlashKDA prefill with RTP's ordinary per-block recurrent checkpoints.

FlashKDA stores [value, key]; RTP stores [key, value]. This adapter converts
at the cache boundary and keeps every intermediate recurrent state in FP32.
Splitting an operator call at cache boundaries does not reschedule the model
or enable scheduler chunked prefill.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class StateSegment:
    start: int
    end: int
    cache_block: int


@dataclass(frozen=True)
class StateSequence:
    initial_block: int | None
    segments: tuple[StateSegment, ...]


def plan_state_sequences(cu_seqlens, prefix_lengths, block_table, block_size):
    """Consume CPU metadata; zero block IDs designate virtual requests."""
    if block_size <= 0 or len(cu_seqlens) != len(prefix_lengths) + 1:
        raise ValueError("Invalid KDA sequence geometry")
    if len(block_table) != len(prefix_lengths) or cu_seqlens[0] != 0:
        raise ValueError("Invalid KDA block table geometry")
    result = []
    for request, prefix in enumerate(prefix_lengths):
        start, end = cu_seqlens[request : request + 2]
        if prefix < 0 or start < 0 or end < start:
            raise ValueError("Negative KDA prefix or sequence length")
        table = block_table[request]
        if start == end:
            result.append(StateSequence(None, ()))
            continue
        first, last = prefix // block_size, (prefix + end - start - 1) // block_size
        if last >= len(table):
            raise ValueError("KDA block table does not cover request")
        ids = table[first : last + 1]
        if all(block == 0 for block in ids):
            if prefix != 0:
                raise ValueError("Virtual KDA request cannot have cached history")
            result.append(StateSequence(None, ()))
            continue
        if any(block <= 0 for block in ids):
            raise ValueError("Real KDA request contains a null or negative cache block")
        initial = table[(prefix - 1) // block_size] if prefix else None
        if initial is not None and initial <= 0:
            raise ValueError("Missing KDA prefix state")
        segments = []
        position = prefix
        while start < end:
            count = min(end - start, block_size - position % block_size)
            segments.append(
                StateSegment(start, start + count, table[position // block_size])
            )
            start += count
            position += count
        result.append(StateSequence(initial, tuple(segments)))
    return tuple(result)


def flash_kda_paged_prefill(
    q,
    k,
    v,
    g,
    beta,
    a_log,
    dt_bias,
    lower_bound,
    cache_states,
    cu_seqlens,
    prefix_lengths,
    block_table,
    block_size,
):
    """Run BF16 projections and publish only real request states to RTP cache.

    q/k/v/g: [tokens, heads, 128], beta: pre-sigmoid [tokens, heads].
    cache_states: FP32 [blocks, heads, key, value], possibly row-strided.
    Metadata arguments are CPU lists prepared outside CUDA Graph capture.
    """
    try:
        import flash_kda
    except ImportError as exc:
        raise RuntimeError(
            "K3 native prefill requires the pinned FlashKDA backend"
        ) from exc
    capability = getattr(torch.ops.flash_kda, "supports_fp32_recurrence", None)
    if capability is None or not capability():
        raise RuntimeError(
            "K3 native KDA requires a backend with FP32 recurrent accumulation"
        )
    if q.dtype != torch.bfloat16 or any(x.dtype != q.dtype for x in (k, v, g, beta)):
        raise ValueError("FlashKDA requires BF16 Q/K/V/G and raw beta logits")
    if q.ndim != 3 or q.shape[-1] != 128 or any(x.shape != q.shape for x in (k, v, g)):
        raise ValueError("FlashKDA requires matching [tokens, heads, 128] projections")
    if beta.shape != q.shape[:2] or cache_states.dtype != torch.float32:
        raise ValueError("Invalid KDA beta or recurrent state precision")
    if cache_states.shape[1:] != (q.shape[1], 128, 128):
        raise ValueError("Invalid RTP KDA state layout")
    if a_log.dtype != torch.float32 or dt_bias.dtype != torch.float32:
        raise ValueError("KDA gate parameters must be FP32")
    if not q.is_cuda or any(
        x.device != q.device for x in (k, v, g, beta, a_log, dt_bias, cache_states)
    ):
        raise ValueError("FlashKDA tensors must share one CUDA device")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "Paged KDA prefill planning must run outside CUDA Graph capture"
        )
    if cu_seqlens[-1] != q.shape[0]:
        raise ValueError("KDA metadata must cover the physical token shape")
    sequences = plan_state_sequences(
        cu_seqlens, prefix_lengths, block_table, block_size
    )
    used_blocks = [s.cache_block for seq in sequences for s in seq.segments]
    used_blocks += [
        seq.initial_block for seq in sequences if seq.initial_block is not None
    ]
    if used_blocks and max(used_blocks) >= cache_states.shape[0]:
        raise ValueError("KDA state block exceeds cache capacity")
    out = torch.zeros_like(v)
    max_tokens = max(
        (s.end - s.start for seq in sequences for s in seq.segments), default=0
    )
    if not max_tokens:
        return out
    workspace = torch.empty(
        flash_kda.get_workspace_size(max_tokens, q.shape[1], 1),
        dtype=torch.uint8,
        device=q.device,
    )
    a_log, dt_bias = a_log.contiguous(), dt_bias.reshape(q.shape[1], 128).contiguous()
    for seq in sequences:
        if not seq.segments:
            continue
        state = (
            torch.zeros((1, q.shape[1], 128, 128), dtype=torch.float32, device=q.device)
            if seq.initial_block is None
            else cache_states[seq.initial_block]
            .transpose(-1, -2)
            .unsqueeze(0)
            .contiguous()
        )
        for segment in seq.segments:
            start, end = segment.start, segment.end
            xs = [x[start:end].unsqueeze(0).contiguous() for x in (q, k, v, g)]
            final = torch.empty_like(state)
            torch.ops.flash_kda.fwd(
                *xs,
                beta[start:end].unsqueeze(0).contiguous(),
                128**-0.5,
                out[start:end].unsqueeze(0),
                workspace,
                a_log,
                dt_bias,
                float(lower_bound),
                state,
                final
            )
            cache_states[segment.cache_block].copy_(final[0].transpose(-1, -2))
            state = final
    return out
