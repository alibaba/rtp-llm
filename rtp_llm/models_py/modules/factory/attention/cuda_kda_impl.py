"""cuLA KDA prefill with RTP's paged FP32 recurrent checkpoints.

The cache and cuLA checkpoint tensors use [key, value] state layout. Operator
calls may split at cache boundaries without changing scheduler prefill batches.
"""

from dataclasses import dataclass

import torch


_MAX_PACKED_CULA_CHECKPOINT_BYTES = 32 << 20


def plan_cula_checkpoint_groups(segments, block_size, max_pages=4):
    """Bound cuLA scratch while keeping each call aligned to cache pages."""
    if block_size <= 0 or max_pages <= 0:
        raise ValueError("cuLA checkpoint group geometry must be positive")
    if not segments:
        return ()
    groups = []
    first_partial = segments[0].end - segments[0].start < block_size
    if first_partial:
        groups.append(segments[:1])
    for index in range(int(first_partial), len(segments), max_pages):
        groups.append(segments[index : index + max_pages])
    return tuple(groups)


def _cula_paged_prefill(
    chunk_kda, q, k, v, g, beta, a_log, dt_bias, lower_bound,
    cache_states, sequences, block_size,
):
    """Publish cuLA's FP32 checkpoints into RTP's paged recurrent cache."""
    if block_size % 64:
        raise ValueError("cuLA KDA checkpoint span must be a multiple of 64")
    heads = q.shape[1]
    output = None
    for sequence in sequences:
        segments = sequence.segments
        if not segments:
            continue
        state = (
            torch.zeros((1, heads, 128, 128), dtype=torch.float32, device=q.device)
            if sequence.initial_block is None
            else cache_states[sequence.initial_block].unsqueeze(0).float().contiguous()
        )
        # One aligned sequence can publish all page checkpoints in one cuLA
        # call. Keep a scratch bound and preserve the segmented path for an
        # in-page reused prefix or a multi-request batch.
        packed = (
            len(sequences) == 1
            and len(segments) > 4
            and len(segments) * heads * 128 * 128 * 4
            <= _MAX_PACKED_CULA_CHECKPOINT_BYTES
            and segments[-1].end == q.shape[0]
            and all(
                segment.start == index * block_size
                and segment.end == min((index + 1) * block_size, q.shape[0])
                for index, segment in enumerate(segments)
            )
        )
        groups = (
            (segments,)
            if packed
            else plan_cula_checkpoint_groups(segments, block_size)
        )
        for group in groups:
            if not group:
                continue
            start, end = group[0].start, group[-1].end
            inputs = [x[start:end].unsqueeze(0).contiguous() for x in (q, k, v, g)]
            raw_beta = beta[start:end].unsqueeze(0).contiguous()
            if raw_beta.data_ptr() % 16:
                raw_beta = raw_beta.clone()
            checkpoints = torch.empty(
                (1, len(group), heads, 128, 128),
                dtype=torch.float32, device=q.device,
            )
            with torch.inference_mode():
                values, final, published = chunk_kda(
                    *inputs, raw_beta, scale=128**-0.5,
                    initial_state=state, output_final_state=False,
                    use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
                    use_beta_sigmoid_in_kernel=True,
                    cu_seqlens=None, cu_seqlens_cpu=None,
                    safe_gate=lower_bound is not None,
                    lower_bound=None if lower_bound is None else float(lower_bound),
                    disable_recompute=False,
                    use_intracard_cp=False, A_log=a_log, dt_bias=dt_bias,
                    checkpoint_interval=block_size, checkpoint_states=checkpoints,
                    checkpoint_offsets=None,
                )
            if final is not None or published is None or published.data_ptr() != checkpoints.data_ptr():
                raise RuntimeError("cuLA did not publish the requested FP32 KDA checkpoints")
            if packed:
                # The sole group covers every token, so cuLA's result is the
                # final output; no zero-fill or second full-tensor copy is needed.
                output = values[0].to(q.dtype)
            else:
                if output is None:
                    output = torch.zeros_like(v)
                output[start:end].copy_(values[0].to(q.dtype))
            for index, segment in enumerate(group):
                if segment.cache_block > 0:
                    cache_states[segment.cache_block].copy_(checkpoints[0, index])
            state = checkpoints[:, -1].contiguous()
    return torch.zeros_like(v) if output is None else output


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
    """Zero IDs designate virtual requests; -1 skips a state checkpoint store."""
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
        if any(block == 0 or block < -1 for block in ids):
            raise ValueError("Real KDA request contains a null or invalid cache block")
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


def cula_kda_paged_prefill(
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
    cache_states: [blocks, heads, key, value], possibly row-strided.
    cuLA computes FP32 checkpoints; writes retain the cache storage dtype.
    Metadata arguments are CPU lists prepared outside CUDA Graph capture.
    """
    from cula.kda import chunk_kda

    if q.dtype != torch.bfloat16 or any(x.dtype != q.dtype for x in (k, v, g, beta)):
        raise ValueError("cuLA KDA requires BF16 Q/K/V/G and raw beta logits")
    if q.ndim != 3 or q.shape[-1] != 128 or any(x.shape != q.shape for x in (k, v, g)):
        raise ValueError("cuLA KDA requires matching [tokens, heads, 128] projections")
    if beta.shape != q.shape[:2] or cache_states.dtype not in (
        torch.float32, torch.bfloat16, torch.float16
    ):
        raise ValueError("Invalid KDA beta or recurrent state precision")
    if cache_states.shape[1:] != (q.shape[1], 128, 128):
        raise ValueError("Invalid RTP KDA state layout")
    if a_log.dtype != torch.float32 or dt_bias.dtype != torch.float32:
        raise ValueError("KDA gate parameters must be FP32")
    if not q.is_cuda or any(
        x.device != q.device for x in (k, v, g, beta, a_log, dt_bias, cache_states)
    ):
        raise ValueError("cuLA KDA tensors must share one CUDA device")
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
    a_log, dt_bias = a_log.contiguous(), dt_bias.reshape(q.shape[1], 128).contiguous()
    return _cula_paged_prefill(
        chunk_kda, q, k, v, g, beta, a_log, dt_bias, lower_bound,
        cache_states, sequences, block_size,
    )
