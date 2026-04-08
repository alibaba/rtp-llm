# -*- coding: utf-8 -*-
"""
FlashInfer GDN kernel wrappers for RTP-LLM.

Bridges RTP-LLM's block_map-based cache indexing to FlashInfer's pool-based
state management. FlashInfer GDN kernels require SM90+ (Hopper/Blackwell).

State layout compatibility:
  RTP-LLM ssm_states: [num_blocks, HV, V, K]  (K-last, via LinearCacheConverter)
  FlashInfer pool:     [pool_size, HV, V, K]   (K-last, K-contiguous)
  → Layouts match exactly. No transpose needed.

Gate parameter handling:
  FlashInfer decode kernels accept raw A_log, a, dt_bias, b and apply
  softplus/sigmoid internally. This eliminates the separate fused_gdn_gating call.
"""

from typing import Optional, Tuple

import torch


def flashinfer_gdn_decode(
    q: torch.Tensor,  # [B, 1, HK, K]
    k: torch.Tensor,  # [B, 1, HK, K]
    v: torch.Tensor,  # [B, 1, HV, V]
    A_log: torch.Tensor,  # [HV]
    a: torch.Tensor,  # [B, HV] or [B, 1, HV]
    dt_bias: torch.Tensor,  # [HV]
    b: torch.Tensor,  # [B, HV] or [B, 1, HV]
    ssm_states: torch.Tensor,  # [num_blocks, HV, V, K] — the paged state pool
    block_map: torch.Tensor,  # [B, max_blocks_per_seq]
    sequence_lengths: torch.Tensor,  # [B] int32 — current seq lengths (inclusive of new token)
    seq_size_per_block: int,
    use_qk_l2norm: bool = True,
) -> torch.Tensor:
    """
    FlashInfer GDN decode using paged state pool.

    Replaces: fused_gdn_gating + fused_recurrent_gated_delta_rule (2 kernel launches → 1).

    Returns:
        attn_out: [B, 1, HV, V]
    """
    from flashinfer.gdn_decode import gated_delta_rule_decode_pretranspose

    B = q.shape[0]

    # Ensure a, b have shape [B, 1, HV] (FlashInfer expects 3D)
    if a.dim() == 2:
        a = a.unsqueeze(1)
    if b.dim() == 2:
        b = b.unsqueeze(1)

    # Compute read block indices: the block containing the last token
    # sequence_lengths is the length AFTER adding the new token
    # read from block at (seq_len - 1) // spb
    read_block_offsets = (sequence_lengths - 1) // seq_size_per_block
    initial_state_indices = block_map[
        torch.arange(B, device=block_map.device), read_block_offsets
    ].to(torch.int64)

    output, _ = gated_delta_rule_decode_pretranspose(
        q=q,
        k=k,
        v=v,
        state=None,  # use pool path
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        initial_state=ssm_states,
        initial_state_indices=initial_state_indices,
        use_qk_l2norm=use_qk_l2norm,
    )

    # Handle cross-block-boundary state copy:
    # FlashInfer updates state in-place at initial_state_indices (read block).
    # If the new token crosses a block boundary (seq_len % spb == 0),
    # we need to copy state from the read block to the write block.
    write_block_offsets = sequence_lengths // seq_size_per_block
    cross_boundary_mask = (sequence_lengths % seq_size_per_block) == 0
    if cross_boundary_mask.any():
        cross_indices = cross_boundary_mask.nonzero(as_tuple=True)[0]
        src_blocks = initial_state_indices[cross_indices]
        dst_block_offsets = write_block_offsets[cross_indices]
        dst_blocks = block_map[cross_indices, dst_block_offsets].to(torch.int64)
        # Copy state: src → dst
        ssm_states[dst_blocks] = ssm_states[src_blocks]

    return output


def flashinfer_gdn_prefill(
    q: torch.Tensor,  # [total_tokens, HK, K]
    k: torch.Tensor,  # [total_tokens, HK, K]
    v: torch.Tensor,  # [total_tokens, HV, V]
    g: torch.Tensor,  # [total_tokens, HV] float32
    beta: torch.Tensor,  # [total_tokens, HV] float32
    initial_state: Optional[torch.Tensor],  # [N, HV, V, K] float32
    cu_seqlens: torch.Tensor,  # [N+1] int64
    use_qk_l2norm: bool = True,
    output_final_state: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    FlashInfer GDN prefill using CUTLASS TMA kernel (SM90+).

    Replaces FLA Triton chunk_gated_delta_rule.

    Note: FlashInfer prefill does NOT return intermediate chunk states (h).
    Only final_state per sequence is returned. This reduces prefix caching
    granularity from per-chunk to per-sequence, which is acceptable.

    Returns:
        (output, final_state) where:
        - output: [total_tokens, HV, V]
        - final_state: [N, HV, V, K] float32 (if output_final_state=True)
    """
    from flashinfer.gdn_prefill import chunk_gated_delta_rule

    result = chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=use_qk_l2norm,
    )

    if output_final_state:
        return result  # (output, final_state)
    else:
        return result, None


def flashinfer_gdn_mtp_decode(
    q: torch.Tensor,  # [B, T, HK, K]
    k: torch.Tensor,  # [B, T, HK, K]
    v: torch.Tensor,  # [B, T, HV, V]
    A_log: torch.Tensor,  # [HV]
    a: torch.Tensor,  # [B, T, HV]
    dt_bias: torch.Tensor,  # [HV]
    b: torch.Tensor,  # [B, T, HV]
    ssm_states: torch.Tensor,  # [num_blocks, HV, V, K]
    block_map: torch.Tensor,  # [B, max_blocks_per_seq]
    sequence_lengths: torch.Tensor,  # [B] int32
    seq_size_per_block: int,
    use_qk_l2norm: bool = True,
    intermediate_states_buffer: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    FlashInfer MTP decode for speculative decoding (T > 1 tokens per request).

    Returns:
        attn_out: [B, T, HV, V]
    """
    from flashinfer.gdn_decode import gated_delta_rule_mtp

    B = q.shape[0]

    read_block_offsets = (sequence_lengths - 1) // seq_size_per_block
    initial_state_indices = block_map[
        torch.arange(B, device=block_map.device), read_block_offsets
    ].to(torch.int64)

    output, _ = gated_delta_rule_mtp(
        q=q,
        k=k,
        v=v,
        initial_state=ssm_states,
        initial_state_indices=initial_state_indices,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        intermediate_states_buffer=intermediate_states_buffer,
        disable_state_update=False,
        use_qk_l2norm=use_qk_l2norm,
    )

    return output
