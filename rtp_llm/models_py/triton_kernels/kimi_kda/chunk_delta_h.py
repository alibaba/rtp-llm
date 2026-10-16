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

import os

import torch
from rtp_llm.models_py.triton_kernels.kimi_kda import chunk_copies, state_update

_REUSE_BIAS_OUTPUT = os.environ.get("GLM53_KDA_REUSE_BIAS_OUTPUT", "0") == "1"


def _sequence_boundaries(cu_seqlens, token_count):
    from rtp_llm.models_py.modules.dsv4.forward_metadata import (
        metadata_cache,
        tensor_key,
    )

    cache = (
        metadata_cache()
        if os.environ.get("RTP_KDA_SHARED_HOST_METADATA", "1") != "0"
        else None
    )
    key = ("kda_sequence_boundaries", tensor_key(cu_seqlens), token_count)
    if cache is not None and key in cache:
        return cache[key]
    with torch.profiler.record_function("kda.state.sequence_boundaries_d2h"):
        boundaries = [int(value) for value in cu_seqlens.detach().cpu().tolist()]
    if not boundaries or boundaries[0] != 0 or boundaries[-1] != token_count:
        raise ValueError(
            f"invalid cu_seqlens boundaries {boundaries} for T={token_count}"
        )
    if cache is not None:
        # The context is discarded after every forward, including exceptions.
        # GPU metadata may be refreshed in-place between requests/graph steps.
        cache[key] = boundaries
    return boundaries


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
    intermediate_state_dtype: torch.dtype | None = None,
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
            dtype=intermediate_state_dtype or k.dtype,
        )
    else:
        if batch != 1:
            raise ValueError("cu_seqlens requires flattened inputs with batch size 1")
        boundaries = _sequence_boundaries(cu_seqlens, token_count)
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
            dtype=intermediate_state_dtype or k.dtype,
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
    reuse_bias_output = (
        _REUSE_BIAS_OUTPUT
        and k.dtype == torch.bfloat16
        and (
            not torch.is_grad_enabled()
            or not any(
                tensor is not None and tensor.requires_grad
                for tensor in (k, w, u, gk, initial_state)
            )
        )
    )
    global_chunk_index = 0
    write_checkpoints = (
        state_update._ENABLED
        and state_update._WRITE_CHECKPOINTS
        and h.dtype == torch.float32
        and not transpose_state_layout
    )

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
        checkpoint_written = False
        prepared_next = None
        for start in range(seq_start, seq_stop, chunk_size):
            stop = min(start + chunk_size, seq_stop)
            length = stop - start
            # Fusion preserves the original contiguous BLAS operands; unsupported
            # shapes/layouts and differentiable calls retain the original copies.
            prepared = prepared_next
            prepared_next = None
            if prepared is None:
                prepared = (
                    chunk_copies.prepare_chunk_copies(
                        state, u[batch_index, start:stop], k[batch_index, start:stop]
                    )
                    if reuse_bias_output and chunk_copies._ENABLED
                    else None
                )
            stored_state = prepared[0] if prepared is not None else state.to(k.dtype)
            target_chunk = (
                global_chunk_index if cu_seqlens is not None else local_chunk_index
            )
            if not checkpoint_written:
                if transpose_state_layout:
                    h[batch_index, target_chunk] = state.transpose(-1, -2)
                else:
                    h[batch_index, target_chunk] = state

            # baddbmm accepts the head-interleaved strided views directly.
            # Materializing both chunks here adds two D2D copies per Python
            # iteration (2,048 copies per layer at a 65K context) without
            # changing the CUDA BLAS result.
            w_chunk = w[batch_index, start:stop].permute(1, 0, 2)
            u_chunk = u[batch_index, start:stop].permute(1, 0, 2)
            if reuse_bias_output:
                # BF16 -> FP32 owns new storage. Match the original GEMM's
                # contiguous output layout and use its bias as the destination,
                # avoiding the separate bias-to-output copy. Inputs stay intact.
                new_value = (
                    prepared[1]
                    if prepared is not None
                    else u_chunk.to(
                        dtype=torch.float32, memory_format=torch.contiguous_format
                    )
                )
                torch.baddbmm(
                    new_value,
                    w_chunk,
                    stored_state,
                    beta=1,
                    alpha=-1,
                    out_dtype=torch.float32,
                    out=new_value,
                )
            else:
                new_value = torch.baddbmm(
                    u_chunk.float(),
                    w_chunk,
                    stored_state,
                    beta=1,
                    alpha=-1,
                    out_dtype=torch.float32,
                )
            new_value_bf16 = (
                chunk_copies.finish_chunk_copies(
                    new_value, v_new[batch_index, start:stop]
                )
                if prepared is not None and v_new is not None
                else None
            )
            if new_value_bf16 is None:
                new_value_bf16 = new_value.to(k.dtype)
                if v_new is not None:
                    v_new[batch_index, start:stop] = new_value_bf16.permute(1, 0, 2)

            log_decay = gk[batch_index, stop - 1]
            if not state_update._ENABLED:
                decay = torch.exp2(log_decay.float()).unsqueeze(-1)
                decayed_state = state * decay
            if prepared is not None:
                k_chunk = prepared[2]
            else:
                k_chunk = k[batch_index, start:stop]
                if key_head_repeat != 1:
                    k_chunk = k_chunk.repeat_interleave(key_head_repeat, dim=1)
                k_chunk = k_chunk.permute(1, 2, 0).contiguous()
            delta = torch.bmm(
                k_chunk,
                new_value_bf16,
                out_dtype=torch.float32,
            )
            if state_update._ENABLED:
                state_output = None
                if write_checkpoints:
                    if stop < seq_stop:
                        state_output = h[batch_index, target_chunk + 1]
                    elif final_state is not None:
                        state_output = final_state[sequence_index]
                prefetched = None
                if (
                    prepared is not None
                    and chunk_copies._PREFETCH_STATE
                    and stop < seq_stop
                ):
                    next_stop = min(stop + chunk_size, seq_stop)
                    prefetched = chunk_copies.update_prepare_chunk(
                        state,
                        delta,
                        log_decay,
                        u[batch_index, stop:next_stop],
                        k[batch_index, stop:next_stop],
                        state_output,
                    )
                if prefetched is not None:
                    updated, prepared_next = prefetched
                else:
                    updated = state_update.kda_state_update(
                        state, delta, log_decay, output=state_output
                    )
                checkpoint_written = updated is not None and state_output is not None
                state = (
                    updated
                    if updated is not None
                    else state * torch.exp2(log_decay.float()).unsqueeze(-1) + delta
                )
            else:
                state = decayed_state + delta

            local_chunk_index += 1
            if cu_seqlens is not None:
                global_chunk_index += 1

        if final_state is not None and not checkpoint_written:
            if transpose_state_layout:
                final_state[sequence_index] = state.transpose(-1, -2)
            else:
                final_state[sequence_index] = state

    return h, v_new, final_state
