"""Bound GDN intermediates while retaining FP32 recurrence across tiles."""

import torch

from rtp_llm.models_py.triton_kernels.fla.block import store_ssm_state_to_block_map
from rtp_llm.models_py.triton_kernels.fla.chunk import chunk_gated_delta_rule


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
    boundaries = cu_seqlens.cpu().tolist()
    for batch, (begin, end) in enumerate(zip(boundaries, boundaries[1:])):
        state = initial_state[batch : batch + 1]
        for start in range(begin, end, max_tokens):
            stop = min(start + max_tokens, end)
            lengths = cu_seqlens.new_tensor([0, stop - start])
            prefix = prefix_lengths[batch : batch + 1] + (start - begin)
            tile, h, state = chunk_gated_delta_rule(
                q[:, start:stop],
                k[:, start:stop],
                v[:, start:stop],
                g[:, start:stop],
                beta[:, start:stop],
                initial_state=state,
                output_final_state=True,
                cu_seqlens=lengths,
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
            )
            output[:, start:stop].copy_(tile)
            # Do not keep the preceding tile's h/output alive while allocating
            # the next one. State remains FP32 until the entire sequence ends.
            del tile, h
    return output
