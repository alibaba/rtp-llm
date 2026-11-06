"""Bound GDN intermediates while retaining FP32 recurrence across tiles."""

import torch

from rtp_llm.models_py.triton_kernels.fla.block import store_ssm_state_to_block_map
from rtp_llm.models_py.triton_kernels.fla.chunk import chunk_gated_delta_rule
from rtp_llm.models_py.triton_kernels.fla.index import prepare_chunk_indices
from rtp_llm.models_py.triton_kernels.fla.l2norm import l2norm_fwd


def chunk_gated_delta_rule_with_cache_store(
    q,
    k,
    v,
    g,
    beta,
    initial_state,
    cu_seqlens,
    prefix_lengths,
    block_map,
    ssm_states,
    seq_size_per_block,
    max_tokens=65536,
    cu_seqlens_cpu=None,
):
    """Run existing kernels in aligned tiles and store the same cache bytes.

    This only tiles the recurrent operator; attention and engine scheduling
    still see the full input. Carry the FP32 final state directly rather than
    reloading the narrower prefix cache between tiles.
    """
    if (
        seq_size_per_block <= 0
        or max_tokens <= 0
        or max_tokens % 64
        or max_tokens % seq_size_per_block
    ):
        raise ValueError("GDN tile size must align with chunks and cache blocks")
    output = torch.empty_like(v)
    boundaries = (
        cu_seqlens_cpu.tolist()
        if cu_seqlens_cpu is not None
        else cu_seqlens.cpu().tolist()
    )
    # Reuse metadata for full tiles within this invocation. Keeping this local
    # avoids retaining GPU tensors across requests or sharing them across devices.
    tile_metadata = {}
    short_batches = []
    short_batch = {False: [], True: []}
    short_batch_tokens = {False: 0, True: 0}
    for batch, (begin, end) in enumerate(zip(boundaries, boundaries[1:])):
        if end == begin:
            continue
        if end - begin <= max_tokens:
            # Bound h by the number of chunks, including each sequence's tail.
            # Many one-token sequences must not allocate max_tokens full states.
            padded_tokens = ((end - begin + 63) // 64) * 64
            # Preserve the original per-sequence normalization kernel. Its two
            # arithmetic paths are not bitwise interchangeable in BF16.
            blocked_norm = q.shape[-1] <= 512 and (end - begin) * q.shape[-2] <= 128
            if short_batch_tokens[blocked_norm] + padded_tokens > max_tokens:
                short_batches.append((blocked_norm, short_batch[blocked_norm]))
                short_batch[blocked_norm] = []
                short_batch_tokens[blocked_norm] = 0
            short_batch[blocked_norm].append(batch)
            short_batch_tokens[blocked_norm] += padded_tokens
            continue
        state = initial_state[batch : batch + 1]
        for start in range(begin, end, max_tokens):
            stop = min(start + max_tokens, end)
            tile_length = stop - start
            if tile_length not in tile_metadata:
                lengths = cu_seqlens.new_tensor([0, tile_length])
                chunks = torch.arange(
                    (tile_length + 63) // 64, dtype=cu_seqlens.dtype, device="cpu"
                )
                indices = torch.stack((torch.zeros_like(chunks), chunks), dim=1).to(
                    device=cu_seqlens.device
                )
                tile_metadata[tile_length] = lengths, indices
            lengths, indices = tile_metadata[tile_length]
            prefix = prefix_lengths[batch : batch + 1] + (start - begin)
            tile, h, state = chunk_gated_delta_rule(
                q[:, start:stop],
                k[:, start:stop],
                v[:, start:stop],
                g[:, start:stop],
                beta[:, start:stop],
                initial_state=state,
                output_final_state=True,
                # Each tile contains one sequence. The fixed-length path has
                # the same h/final-state layout and needs no GPU length readback.
                cu_seqlens=None,
                use_qk_l2norm_in_kernel=True,
            )
            store_ssm_state_to_block_map(
                h,
                state,
                prefix,
                lengths,
                block_map[batch : batch + 1],
                ssm_states,
                seq_size_per_block,
                chunk_size=64,
                chunk_indices=indices,
            )
            output[:, start:stop].copy_(tile)
            # Do not keep the preceding tile's h/output alive while allocating
            # the next one. State remains FP32 until the entire sequence ends.
            del tile, h

    for blocked_norm, batches in short_batch.items():
        if batches:
            short_batches.append((blocked_norm, batches))
    for blocked_norm, batches in short_batches:
        ranges = [(boundaries[batch], boundaries[batch + 1]) for batch in batches]
        contiguous = all(left[1] == right[0] for left, right in zip(ranges, ranges[1:]))
        token_indices = None
        if contiguous:
            begin, end = ranges[0][0], ranges[-1][1]
            packed = [tensor[:, begin:end] for tensor in (q, k, v, g, beta)]
        else:
            token_indices = torch.cat(
                [torch.arange(begin, end, device="cpu") for begin, end in ranges]
            ).to(device=q.device)
            packed = [
                tensor.index_select(1, token_indices) for tensor in (q, k, v, g, beta)
            ]
        batch_indices = torch.tensor(batches, dtype=torch.long, device=q.device)
        offsets = [0]
        for begin, end in ranges:
            offsets.append(offsets[-1] + end - begin)
        lengths_cpu = torch.tensor(offsets, dtype=cu_seqlens.dtype, device="cpu")
        lengths = lengths_cpu.to(device=cu_seqlens.device)
        indices = prepare_chunk_indices(lengths, 64, lengths_cpu)
        packed[0] = l2norm_fwd(packed[0], use_blocked_kernel=blocked_norm)
        packed[1] = l2norm_fwd(packed[1], use_blocked_kernel=blocked_norm)
        tile, h, final_state = chunk_gated_delta_rule(
            *packed,
            initial_state=initial_state.index_select(0, batch_indices),
            output_final_state=True,
            cu_seqlens=lengths if len(batches) > 1 else None,
            cu_seqlens_cpu=lengths_cpu if len(batches) > 1 else None,
            use_qk_l2norm_in_kernel=False,
        )
        store_ssm_state_to_block_map(
            h,
            final_state,
            prefix_lengths.index_select(0, batch_indices),
            lengths,
            block_map.index_select(0, batch_indices),
            ssm_states,
            seq_size_per_block,
            chunk_size=64,
            chunk_indices=indices,
        )
        if token_indices is None:
            output[:, ranges[0][0] : ranges[-1][1]].copy_(tile)
        else:
            output.index_copy_(1, token_indices, tile)
        del tile, h, final_state, packed
    return output
