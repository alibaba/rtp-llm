# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors
#
# Adapted for rtp-llm: forward-only, supports USE_EXP2 for log2-space gates.
#
# Only the deterministic CUBLAS chunk-state recurrence remains here. The old
# Triton block-dim-64 bring-up comparator still exists in the upstream FLA
# copy under triton_kernels/fla/.

import torch


def chunk_gated_delta_rule_fwd_h_cublas(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    save_new_value: bool = True,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    use_exp2: bool = True,
    transpose_state_layout: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Deterministic K3 chunk-state recurrence using CUDA BLAS GEMMs.

    The original FLA Triton kernel accumulates a BF16 tensor-core product into
    a non-zero FP32 recurrent state.  Triton 3.6 and 3.7 lower that operation
    with different reduction groupings, which changes a few BF16 KDA outputs
    and can cascade into different MoE expert IDs.  CUDA BLAS' full-K
    ``baddbmm(..., out_dtype=float32)`` reproduces the sealed K3 Dummy state
    and output bit-for-bit on the supported CUDA13 stack.

    This is the correctness backend.  The Triton implementation remains
    available for profiling and can replace it once a compiler-version-stable
    kernel has component parity.  The Python chunk loop is deliberately
    simple; performance optimization is out of scope for the accuracy phase.
    """

    del chunk_indices
    if g is not None or gk is None:
        raise ValueError(
            "the K3 CUBLAS state backend requires gk and does not support g"
        )
    if not use_exp2:
        raise ValueError("the K3 CUBLAS state backend requires exp2 gates")
    if not k.is_cuda or not w.is_cuda or not u.is_cuda or not gk.is_cuda:
        raise ValueError("the K3 CUBLAS state backend requires CUDA tensors")
    if k.dtype != w.dtype or k.dtype != u.dtype:
        raise ValueError("k, w and u must have the same dtype")

    batch, token_count, key_heads, key_dim = k.shape
    value_heads, value_dim = u.shape[2], u.shape[3]
    if value_heads % key_heads != 0:
        raise ValueError(
            f"value heads {value_heads} must be divisible by key heads {key_heads}"
        )
    if w.shape != (batch, token_count, value_heads, key_dim):
        raise ValueError(
            f"unexpected w shape {tuple(w.shape)} for k/u shapes "
            f"{tuple(k.shape)}/{tuple(u.shape)}"
        )
    if gk.shape != (batch, token_count, value_heads, key_dim):
        raise ValueError(
            f"unexpected gk shape {tuple(gk.shape)} for value heads {value_heads}"
        )
    if initial_state is not None and initial_state.dtype != torch.float32:
        raise ValueError("initial_state must be FP32")

    if cu_seqlens is None:
        sequence_ranges = [
            (batch_index, 0, token_count) for batch_index in range(batch)
        ]
        chunks_per_sequence = (token_count + chunk_size - 1) // chunk_size
        total_chunks = chunks_per_sequence
        h = k.new_empty(
            batch,
            total_chunks,
            value_heads,
            value_dim if transpose_state_layout else key_dim,
            key_dim if transpose_state_layout else value_dim,
        )
    else:
        if batch != 1:
            raise ValueError("cu_seqlens requires flattened inputs with batch size 1")
        boundaries = [int(value) for value in cu_seqlens.detach().cpu().tolist()]
        if not boundaries or boundaries[0] != 0 or boundaries[-1] != token_count:
            raise ValueError(
                f"invalid cu_seqlens boundaries {boundaries} for T={token_count}"
            )
        sequence_ranges = [
            (0, boundaries[index], boundaries[index + 1])
            for index in range(len(boundaries) - 1)
        ]
        total_chunks = sum(
            (stop - start + chunk_size - 1) // chunk_size
            for _, start, stop in sequence_ranges
        )
        h = k.new_empty(
            1,
            total_chunks,
            value_heads,
            value_dim if transpose_state_layout else key_dim,
            key_dim if transpose_state_layout else value_dim,
        )

    sequence_count = len(sequence_ranges)
    final_state = (
        torch.empty(
            sequence_count,
            value_heads,
            value_dim if transpose_state_layout else key_dim,
            key_dim if transpose_state_layout else value_dim,
            dtype=torch.float32,
            device=k.device,
        )
        if output_final_state
        else None
    )
    v_new = torch.empty_like(u) if save_new_value else None
    key_head_repeat = value_heads // key_heads
    global_chunk_index = 0

    for sequence_index, (batch_index, seq_start, seq_stop) in enumerate(
        sequence_ranges
    ):
        if initial_state is None:
            state = torch.zeros(
                value_heads,
                key_dim,
                value_dim,
                dtype=torch.float32,
                device=k.device,
            )
        else:
            state = initial_state[sequence_index]
            if transpose_state_layout:
                state = state.transpose(-1, -2)
            state = state.contiguous()

        local_chunk_index = 0
        for start in range(seq_start, seq_stop, chunk_size):
            stop = min(start + chunk_size, seq_stop)
            length = stop - start
            stored_state = state.to(k.dtype)
            target_chunk = (
                global_chunk_index if cu_seqlens is not None else local_chunk_index
            )
            if transpose_state_layout:
                h[batch_index, target_chunk] = stored_state.transpose(-1, -2)
            else:
                h[batch_index, target_chunk] = stored_state

            # baddbmm accepts the head-interleaved strided views directly.
            # Materializing both chunks here adds two D2D copies per Python
            # iteration (2,048 copies per layer at a 65K context) without
            # changing the CUDA BLAS result.
            w_chunk = w[batch_index, start:stop].permute(1, 0, 2)
            u_chunk = u[batch_index, start:stop].permute(1, 0, 2)
            new_value = torch.baddbmm(
                u_chunk.float(),
                w_chunk,
                stored_state,
                beta=1,
                alpha=-1,
                out_dtype=torch.float32,
            )
            new_value_bf16 = new_value.to(k.dtype)
            if v_new is not None:
                v_new[batch_index, start:stop] = new_value_bf16.permute(1, 0, 2)

            decay = torch.exp2(gk[batch_index, stop - 1].float()).unsqueeze(-1)
            decayed_state = state * decay
            k_chunk = k[batch_index, start:stop]
            if key_head_repeat != 1:
                k_chunk = k_chunk.repeat_interleave(key_head_repeat, dim=1)
            k_chunk = k_chunk.permute(1, 2, 0).contiguous()
            state = decayed_state + torch.bmm(
                k_chunk,
                new_value_bf16,
                out_dtype=torch.float32,
            )

            local_chunk_index += 1
            if cu_seqlens is not None:
                global_chunk_index += 1

        if final_state is not None:
            if transpose_state_layout:
                final_state[sequence_index] = state.transpose(-1, -2)
            else:
                final_state[sequence_index] = state

    return h, v_new, final_state
