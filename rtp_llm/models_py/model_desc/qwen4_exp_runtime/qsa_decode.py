"""QSA decode planning and writes; the context owns undo."""

from __future__ import annotations

import math
from typing import Any

import torch
from rtp_llm.models.qwen4_exp.qwen4_exp_kv_cache import (
    INDEXER_KV_TAG,
    INDEXER_STATE_TAG,
)
from rtp_llm.models_py.model_desc.qwen4_exp_runtime.qsa_cache_views import (
    _block_table,
    _typed_2d_pool,
    _validate_required_blocks,
)
from rtp_llm.models_py.model_desc.qwen4_exp_runtime.qsa_selection import (
    select_qsa_paged_tokens,
)
from rtp_llm.models_py.modules.qwen4_exp.qsa.indexer import (
    apply_partial_rope,
    build_base_rope,
    build_qsa_rope,
    build_qsa_rope_from_logical_positions,
    is_qsa_rope_style,
)


def select_decode_tokens(
    self,
    q: torch.Tensor,
    raw_keys: torch.Tensor,
    *,
    indexer: Any,
    rope_config: Any,
) -> torch.Tensor:
    """Write/score one ordinary text decode token per request.

    ``q`` and ``raw_keys`` are the two outputs of one indexer projection.
    The raw key is first committed to the state ring (and, when it closes a
    compression group, to the compressed KV pool).  The same projected q is
    then rotated and scored against every visible compressed entry.
    """
    graph_decode = bool(self.main_inputs.is_cuda_graph)
    input_lengths = self._validate_mode_and_metadata(
        expect_prefill=False, allow_graph_decode=graph_decode
    )
    sequence_lengths = self.main_inputs.sequence_lengths
    batch_size = int(sequence_lengths.numel())
    head_dim = int(indexer.head_dim)
    head_num = int(indexer.n_heads)
    ratio = int(indexer.compress_ratio)
    token_budget = int(indexer.token_budget)
    if batch_size <= 0 or len(input_lengths) != batch_size:
        raise RuntimeError(
            "qwen4_exp QSA ordinary decode requires non-empty, batch-aligned "
            "sequence_lengths"
        )
    if q.dim() != 3 or tuple(q.shape) != (batch_size, head_num, head_dim):
        raise RuntimeError(
            "qwen4_exp QSA decode q must be "
            f"[{batch_size}, {head_num}, {head_dim}], got {tuple(q.shape)}"
        )
    if raw_keys.dim() != 2 or tuple(raw_keys.shape) != (batch_size, head_dim):
        raise RuntimeError(
            "qwen4_exp QSA decode raw keys must be "
            f"[{batch_size}, {head_dim}], got {tuple(raw_keys.shape)}"
        )
    if q.dtype != torch.bfloat16 or raw_keys.dtype != torch.bfloat16:
        raise RuntimeError(
            "qwen4_exp QSA ordinary decode currently requires BF16 q/raw keys"
        )
    if q.device != raw_keys.device:
        raise RuntimeError("qwen4_exp QSA decode q/raw keys must share a device")
    # The draft projection yields a strided raw-key view.  The fixed-shape
    # writer accepts contiguous rows, and this copy is captured in Graph.
    raw_keys = raw_keys.contiguous()

    # The engine keeps decode lengths on pinned CPU memory. Derive the
    # scorer's output width there, before the H2D copy, rather than
    # synchronizing a GPU max reduction after every sparse layer.
    host_max_compressed = (
        (int(sequence_lengths.max().item()) + 1) // ratio
        if sequence_lengths.device.type == "cpu"
        else None
    )
    # Replay refreshes this device tensor before launching the captured graph.
    # Reading the pinned mirror inside capture would freeze an H2D dependency.
    sequence_lengths = (
        self.main_inputs.sequence_lengths_plus_1_device - 1
        if graph_decode
        else sequence_lengths.to(device=q.device, dtype=torch.int32, non_blocking=True)
    ).contiguous()
    invalid_lengths = (
        bool(torch.any(sequence_lengths < 0).item())
        if host_max_compressed is None
        else bool(torch.any(self.main_inputs.sequence_lengths < 0).item())
    )
    if invalid_lengths:
        raise RuntimeError("qwen4_exp QSA sequence_lengths must be non-negative")
    position_ids = self.main_inputs.combo_position_ids
    index_factor = int(rope_config.index_factor)
    if position_ids is None or int(position_ids.numel()) != batch_size * index_factor:
        actual = 0 if position_ids is None else int(position_ids.numel())
        raise RuntimeError(
            "qwen4_exp QSA ordinary decode position ids do not match the "
            f"configured index_factor={index_factor}: got {actual} values "
            f"for batch={batch_size}"
        )
    # The MTP draft transports the engine's absolute positions, whose
    # bookkeeping intentionally differs from the logical cache rows, so
    # only non-draft decodes can be validated against sequence_lengths.
    decode_logical_positions = None if self.is_mtp_draft else sequence_lengths
    get_frequency = getattr(indexer, "rope_inv_freq", None)
    frequency = get_frequency(rope_config) if callable(get_frequency) else None
    prepared = None
    if graph_decode and self.is_mtp_draft:
        from rtp_llm.models_py.modules.qwen4_exp.qsa.draft_decode_positions_triton import (
            try_prepare_decode_positions,
        )

        prepared = try_prepare_decode_positions(
            sequence_lengths, position_ids, frequency, rope_config
        )
    try:
        if prepared is not None:
            (
                current_cos,
                current_sin,
                rope_cos,
                rope_sin,
                visible_token_lengths,
                compressed_lengths,
            ) = prepared
        elif graph_decode and self.is_mtp_draft:
            if not is_qsa_rope_style(rope_config, "Base"):
                raise RuntimeError("qwen4_exp draft Graph requires text-only Base RoPE")
            # MTP's text position generator repeats one scalar across its
            # axes. Eager checks equality; Graph capture cannot synchronize.
            current_cos, current_sin = build_base_rope(
                position_ids,
                rope_config,
                token_count=batch_size,
                dtype=raw_keys.dtype,
                device=raw_keys.device,
                validate_position_axes=False,
                inv_freq=frequency,
            )
        elif graph_decode:
            # Graph capture uses the engine's canonical text
            # positions. The C++ graph owner refreshes lengths before replay.
            current_cos, current_sin = build_qsa_rope_from_logical_positions(
                sequence_lengths,
                rope_config,
                token_count=batch_size,
                dtype=raw_keys.dtype,
                device=raw_keys.device,
                inv_freq=frequency,
            )
        else:
            current_cos, current_sin = build_qsa_rope(
                position_ids,
                rope_config,
                token_count=batch_size,
                dtype=raw_keys.dtype,
                device=raw_keys.device,
                logical_positions=decode_logical_positions,
            )
    except ValueError as error:
        raise RuntimeError(f"qwen4_exp QSA ordinary decode {error}") from error

    kv_tokens_per_block = int(self.indexer_kv_cache.seq_size_per_block)
    if kv_tokens_per_block <= 0 or kv_tokens_per_block % ratio:
        raise RuntimeError(
            "qwen4_exp QSA indexer KV page size must be positive and divisible "
            f"by ratio={ratio}, got {kv_tokens_per_block}"
        )
    kv_entries_per_block = kv_tokens_per_block // ratio
    kv_pool = _typed_2d_pool(
        self.indexer_kv_cache,
        tag=INDEXER_KV_TAG,
        storage_dtype=torch.uint8,
        payload_dtype=torch.bfloat16,
        entries=kv_entries_per_block,
        width=head_dim,
    )
    state_pool = _typed_2d_pool(
        self.indexer_state_cache,
        tag=INDEXER_STATE_TAG,
        storage_dtype=torch.float32,
        payload_dtype=torch.float32,
        entries=2 * ratio,
        width=head_dim,
    )
    if kv_pool.device != q.device or state_pool.device != q.device:
        raise RuntimeError(
            "qwen4_exp QSA side pools and decode projection must share a device"
        )
    kv_table = _block_table(
        self.indexer_kv_inputs,
        "kv_cache_kernel_block_id_device",
        tag=INDEXER_KV_TAG,
        batch_size=batch_size,
        device=q.device,
    )
    state_table = _block_table(
        self.indexer_state_inputs,
        "kv_cache_block_id_device",
        tag=INDEXER_STATE_TAG,
        batch_size=batch_size,
        device=q.device,
    )

    active = state_table.ne(0).any(dim=1) if self.main_inputs.is_s_padded else None

    if prepared is None:
        visible_token_lengths = sequence_lengths + 1
        compressed_lengths = visible_token_lengths // ratio
    if active is not None:
        visible_token_lengths = torch.where(active, visible_token_lengths, 0)
        compressed_lengths = torch.where(active, compressed_lengths, 0)
    compressed_capacity = int(kv_table.shape[1]) * kv_entries_per_block
    exceeds_capacity = (
        host_max_compressed > compressed_capacity
        if host_max_compressed is not None
        else bool(torch.any(compressed_lengths > compressed_capacity).item())
    )
    if exceeds_capacity:
        raise RuntimeError(
            "qwen4_exp QSA compressed decode length exceeds its tag-local "
            "indexer KV block table"
        )
    if not graph_decode:
        required_kv_columns = (
            compressed_lengths + kv_entries_per_block - 1
        ) // kv_entries_per_block
        _validate_required_blocks(
            kv_table,
            required_kv_columns,
            pool_blocks=int(kv_pool.shape[0]),
            tag=INDEXER_KV_TAG,
        )

    # Only rows that close a compression group consume these values. Build
    # one block-start RoPE row per request instead of [0, context_len).
    if prepared is None:
        block_starts = sequence_lengths.to(torch.int64)
        block_starts = block_starts - block_starts.remainder(ratio)
        rope_cos, rope_sin = build_qsa_rope_from_logical_positions(
            block_starts,
            rope_config,
            token_count=batch_size,
            dtype=raw_keys.dtype,
            device=raw_keys.device,
            inv_freq=frequency,
        )
    fused_writer = True
    if fused_writer:
        from rtp_llm.models_py.modules.qwen4_exp.qsa.indexer_decode_triton import (
            is_supported,
            write_decode_key_with_undo_,
        )

        fused_writer = is_supported(
            raw_keys,
            sequence_lengths,
            rope_cos,
            rope_sin,
            indexer.k_norm_gamma,
            kv_pool,
            kv_table,
            state_pool,
            state_table,
            kv_tokens_per_block=kv_tokens_per_block,
            state_tokens_per_block=int(self.indexer_state_cache.seq_size_per_block),
            ratio=ratio,
        )
    if graph_decode and not fused_writer:
        raise RuntimeError(
            "qwen4_exp QSA Graph decode requires the fixed-shape Triton writer"
        )
    if fused_writer:
        if self._side_cache_undo is not None:
            raise RuntimeError("qwen4_exp QSA side-cache transaction is already active")
        undo = write_decode_key_with_undo_(
            raw_keys,
            sequence_lengths,
            rope_cos,
            rope_sin,
            indexer.k_norm_gamma,
            kv_pool,
            kv_table,
            state_pool,
            state_table,
            norm_eps=float(indexer.norm_eps),
        )
        object.__setattr__(self, "_side_cache_undo", undo)
    else:
        cu_seqlens = torch.arange(batch_size + 1, dtype=torch.int32, device=q.device)
        self._write_indexer_cache_transactional(
            raw_keys,
            cu_seqlens,
            sequence_lengths.to(torch.int64),
            rope_cos,
            rope_sin,
            indexer.k_norm_gamma,
            kv_pool,
            kv_table,
            state_pool,
            state_table,
            ratio=ratio,
            kv_tokens_per_block=kv_tokens_per_block,
            state_tokens_per_block=int(self.indexer_state_cache.seq_size_per_block),
            norm_eps=float(indexer.norm_eps),
            rope_is_token_aligned=True,
        )

    rotated_q = apply_partial_rope(
        q, current_cos.unsqueeze(1), current_sin.unsqueeze(1)
    )
    get_weights = getattr(indexer, "score_weights", None)
    score_weights = get_weights(batch_size, q.device) if callable(get_weights) else None
    if score_weights is None:
        score_weights = torch.full(
            (batch_size, head_num),
            1.0 / math.sqrt(head_dim),
            dtype=torch.float32,
            device=q.device,
        )
    from rtp_llm.models_py.modules.qwen4_exp.qsa.indexer_paged_score import (
        qsa_paged_indexer_score,
    )

    block_logits = qsa_paged_indexer_score(
        rotated_q.unsqueeze(1).contiguous(),
        score_weights,
        kv_pool.flatten(0, 1),
        kv_table,
        compressed_lengths.unsqueeze(1),
        block_size=kv_entries_per_block,
        # _validate_required_blocks checked every visible physical ID;
        # the score kernel also masks invalid IDs before reading the pool.
        validate_block_table=False,
        max_ctx_len=(
            host_max_compressed
            if host_max_compressed is not None
            else int(compressed_lengths.max().item())
        ),
    )
    return select_qsa_paged_tokens(
        block_logits,
        visible_token_lengths,
        compress_ratio=ratio,
        token_budget=token_budget,
        # Sequence lengths and required block-table capacity were checked
        # above; repeating the device-to-host check per sparse layer adds
        # a synchronization to every decode step.
        validate_lengths=False,
        scores_are_masked=True,
    )
