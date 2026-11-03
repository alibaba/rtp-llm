"""cuLA KDA prefill with RTP's paged FP32 recurrent checkpoints.

The cache and cuLA checkpoint tensors use [key, value] state layout. Aligned
batches use one call; in-page prefixes retain page-boundary segmented execution.
"""

from dataclasses import dataclass

import torch


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
    output = torch.zeros_like(v)
    for sequence in sequences:
        segments = sequence.segments
        if not segments:
            continue
        state = (
            torch.zeros((1, heads, 128, 128), dtype=torch.float32, device=q.device)
            if sequence.initial_block is None
            else cache_states[sequence.initial_block].unsqueeze(0).float().contiguous()
        )
        groups = plan_cula_checkpoint_groups(segments, block_size)
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
            output[start:end].copy_(values[0].to(q.dtype))
            for index, segment in enumerate(group):
                if segment.cache_block > 0:
                    cache_states[segment.cache_block].copy_(checkpoints[0, index])
            state = checkpoints[:, -1].contiguous()
    return output


@dataclass(frozen=True)
class StateSegment:
    start: int
    end: int
    cache_block: int


@dataclass(frozen=True)
class StateSequence:
    initial_block: int | None
    segments: tuple[StateSegment, ...]


@dataclass(frozen=True)
class PackedCulaKdaPrefill:
    cu_seqlens: torch.Tensor | None
    cu_seqlens_cpu: torch.Tensor | None
    checkpoint_offsets: torch.Tensor | None
    initial_blocks: torch.Tensor
    checkpoint_blocks: torch.Tensor
    initial_states: torch.Tensor
    checkpoints: torch.Tensor
    padding_ranges: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class CulaKdaPrefillMetadata:
    block_size: int
    sequences: tuple[StateSequence, ...]
    packed: PackedCulaKdaPrefill | None


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


def prepare_cula_kda_prefill_metadata(
    cu_seqlens,
    prefix_lengths,
    block_table,
    block_size,
    cache_states,
):
    """Prepare one round's immutable page mapping and layer-shared workspace."""
    if block_size <= 0 or block_size % 64:
        raise ValueError("cuLA KDA checkpoint span must be a positive multiple of 64")
    sequences = plan_state_sequences(
        cu_seqlens, prefix_lengths, block_table, block_size
    )
    used_blocks = [s.cache_block for seq in sequences for s in seq.segments]
    used_blocks += [
        seq.initial_block for seq in sequences if seq.initial_block is not None
    ]
    if used_blocks and max(used_blocks) >= cache_states.shape[0]:
        raise ValueError("KDA state block exceeds cache capacity")
    published = set()
    for seq in sequences:
        if seq.initial_block in published:
            # Preserve sequential read-after-write for aliased request state.
            return CulaKdaPrefillMetadata(block_size, sequences, None)
        published.update(s.cache_block for s in seq.segments if s.cache_block > 0)
    if not any(seq.segments for seq in sequences) or any(
        seq.segments and prefix % block_size
        for seq, prefix in zip(sequences, prefix_lengths)
    ):
        return CulaKdaPrefillMetadata(block_size, sequences, None)

    cu_values, offsets = [0], [0]
    initial_blocks, checkpoint_blocks, padding = [], [], []
    for seq, start, end in zip(sequences, cu_seqlens, cu_seqlens[1:]):
        if start == end:
            continue
        cu_values.append(end)
        initial_blocks.append(
            seq.initial_block if seq.initial_block is not None else -1
        )
        if seq.segments:
            checkpoint_blocks.extend(segment.cache_block for segment in seq.segments)
        else:
            # Keep virtual rows in the token layout, with no state publication.
            checkpoint_blocks.extend(
                [-1] * ((end - start + block_size - 1) // block_size)
            )
            padding.append((start, end))
        offsets.append(len(checkpoint_blocks))
    if not initial_blocks:
        return CulaKdaPrefillMetadata(block_size, sequences, None)

    # The segmented path publishes in order. Preserve its last-write behavior
    # if a rolling page table maps several checkpoints to the same slot.
    seen = set()
    for index in reversed(range(len(checkpoint_blocks))):
        block = checkpoint_blocks[index]
        if block > 0:
            if block in seen:
                checkpoint_blocks[index] = -1
            seen.add(block)

    device = cache_states.device
    heads = cache_states.shape[1]
    single_sequence = len(initial_blocks) == 1
    cpu_cu = (
        None
        if single_sequence
        else torch.tensor(cu_values, dtype=torch.int32, pin_memory=True)
    )
    packed = PackedCulaKdaPrefill(
        cu_seqlens=None if single_sequence else cpu_cu.to(device, non_blocking=True),
        cu_seqlens_cpu=cpu_cu,
        checkpoint_offsets=(
            None
            if single_sequence
            else torch.tensor(offsets, dtype=torch.int32, device=device)
        ),
        initial_blocks=torch.tensor(initial_blocks, dtype=torch.int64, device=device),
        checkpoint_blocks=torch.tensor(
            checkpoint_blocks, dtype=torch.int64, device=device
        ),
        initial_states=torch.empty(
            (len(initial_blocks), heads, 128, 128), dtype=torch.float32, device=device
        ),
        checkpoints=torch.empty(
            (1, len(checkpoint_blocks), heads, 128, 128),
            dtype=torch.float32,
            device=device,
        ),
        padding_ranges=tuple(padding),
    )
    return CulaKdaPrefillMetadata(block_size, sequences, packed)


def _cula_packed_prefill(
    chunk_kda,
    q,
    k,
    v,
    g,
    beta,
    a_log,
    dt_bias,
    lower_bound,
    cache_states,
    metadata,
    block_size,
):
    from rtp_llm.models_py.triton_kernels.kda.prefill_cache import (
        load_kda_prefill_states,
        store_kda_prefill_states,
    )

    load_kda_prefill_states(
        cache_states, metadata.initial_blocks, metadata.initial_states
    )
    raw_beta = beta.unsqueeze(0).contiguous()
    if raw_beta.data_ptr() % 16:
        raw_beta = raw_beta.clone()
    with torch.inference_mode():
        output, final, published = chunk_kda(
            *(x.unsqueeze(0).contiguous() for x in (q, k, v, g)),
            raw_beta,
            scale=128**-0.5,
            initial_state=metadata.initial_states,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            cu_seqlens=metadata.cu_seqlens,
            cu_seqlens_cpu=metadata.cu_seqlens_cpu,
            safe_gate=lower_bound is not None,
            lower_bound=None if lower_bound is None else float(lower_bound),
            disable_recompute=False,
            use_intracard_cp=False,
            A_log=a_log,
            dt_bias=dt_bias,
            checkpoint_interval=block_size,
            checkpoint_states=metadata.checkpoints,
            checkpoint_offsets=metadata.checkpoint_offsets,
        )
    if (
        final is not None
        or published is None
        or published.data_ptr() != metadata.checkpoints.data_ptr()
    ):
        raise RuntimeError("cuLA did not publish the requested FP32 KDA checkpoints")
    store_kda_prefill_states(
        metadata.checkpoints[0], metadata.checkpoint_blocks, cache_states
    )
    with torch.inference_mode():
        output = output[0].to(q.dtype)
        for start, end in metadata.padding_ranges:
            output[start:end].zero_()
    return output


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
    *,
    metadata=None,
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
    if block_size <= 0 or block_size % 64:
        raise ValueError("cuLA KDA checkpoint span must be a positive multiple of 64")
    if metadata is None:
        metadata = prepare_cula_kda_prefill_metadata(
            cu_seqlens, prefix_lengths, block_table, block_size, cache_states
        )
    a_log, dt_bias = a_log.contiguous(), dt_bias.reshape(q.shape[1], 128).contiguous()
    if metadata.packed is not None:
        return _cula_packed_prefill(
            chunk_kda,
            q,
            k,
            v,
            g,
            beta,
            a_log,
            dt_bias,
            lower_bound,
            cache_states,
            metadata.packed,
            block_size,
        )
    return _cula_paged_prefill(
        chunk_kda,
        q,
        k,
        v,
        g,
        beta,
        a_log,
        dt_bias,
        lower_bound,
        cache_states,
        metadata.sequences,
        block_size,
    )
