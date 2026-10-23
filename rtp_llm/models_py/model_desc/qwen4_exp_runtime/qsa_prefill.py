"""QSA prefill planning and writes; the context owns undo."""

from __future__ import annotations

import math
from typing import Any

import torch

from rtp_llm.models.qwen4_exp.qwen4_exp_kv_cache import (
    INDEXER_KV_TAG,
    INDEXER_STATE_TAG,
)
from rtp_llm.models_py.modules.qwen4_exp.qsa.indexer import (
    apply_partial_rope,
    build_qsa_rope,
    build_qsa_rope_from_logical_positions,
    is_qsa_rope_style,
)

from .qsa_cache_views import _block_table, _same_tensor_metadata, _typed_2d_pool
from .qsa_selection import select_qsa_paged_tokens


def write_prefill_indexer_cache(
    self,
    raw_keys: torch.Tensor,
    *,
    indexer: Any,
    rope_config: Any,
) -> tuple[list[int], torch.Tensor, torch.Tensor, dict]:
    """Write one ragged prefill projection to both side pools.

    This path requires zero prefixes. Page-aligned prefix reuse uses
    select_prefix_reuse_prefill_tokens and its paged writer instead.
    """
    lengths = self._validate_mode_and_metadata(expect_prefill=True)
    token_count = int(raw_keys.shape[0]) if raw_keys.dim() == 2 else -1
    if token_count < 0 or sum(lengths) != token_count:
        raise RuntimeError(
            f"qwen4_exp QSA input_lengths={lengths} do not partition "
            f"the {token_count} packed raw keys"
        )
    head_dim = int(indexer.head_dim)
    ratio = int(indexer.compress_ratio)
    if int(raw_keys.shape[1]) != head_dim:
        raise RuntimeError(
            f"qwen4_exp QSA raw key width is {raw_keys.shape[1]}, expected {head_dim}"
        )

    kv_tokens_per_block = int(self.indexer_kv_cache.seq_size_per_block)
    if kv_tokens_per_block <= 0 or kv_tokens_per_block % ratio:
        raise RuntimeError(
            "qwen4_exp QSA indexer KV page size must be positive and divisible "
            f"by ratio={ratio}, got {kv_tokens_per_block}"
        )
    prefix_tensor = self.main_inputs.prefix_lengths
    if (
        prefix_tensor is None
        or prefix_tensor.dim() != 1
        or int(prefix_tensor.numel()) != len(lengths)
        or prefix_tensor.dtype != torch.int32
    ):
        raise RuntimeError("qwen4_exp QSA prefill requires int32 prefix_lengths [B]")
    start_positions = prefix_tensor.to(
        device=raw_keys.device, dtype=torch.int64, non_blocking=True
    ).contiguous()
    kv_pool = _typed_2d_pool(
        self.indexer_kv_cache,
        tag=INDEXER_KV_TAG,
        storage_dtype=torch.uint8,
        payload_dtype=torch.bfloat16,
        entries=kv_tokens_per_block // ratio,
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
    if kv_pool.device != raw_keys.device or state_pool.device != raw_keys.device:
        raise RuntimeError("qwen4_exp QSA side pools and raw keys must share a device")

    batch_size = len(lengths)
    kv_table = _block_table(
        self.indexer_kv_inputs,
        "kv_cache_kernel_block_id_device",
        tag=INDEXER_KV_TAG,
        batch_size=batch_size,
        device=raw_keys.device,
    )
    state_table = _block_table(
        self.indexer_state_inputs,
        "kv_cache_block_id_device",
        tag=INDEXER_STATE_TAG,
        batch_size=batch_size,
        device=raw_keys.device,
    )
    cu_seqlens = self.main_inputs.cu_seqlens_device
    if (
        cu_seqlens.dim() != 1
        or cu_seqlens.dtype != torch.int32
        or int(cu_seqlens.numel()) != batch_size + 1
        or cu_seqlens.device != raw_keys.device
    ):
        raise RuntimeError("qwen4_exp QSA requires device int32 cu_seqlens [B + 1]")

    def check_cu_partition():
        expected_cu = torch.tensor(
            [0] + list(torch.tensor(lengths).cumsum(0).tolist()),
            dtype=torch.int32,
            device=raw_keys.device,
        )
        if not bool(torch.equal(cu_seqlens, expected_cu)):
            raise RuntimeError("qwen4_exp QSA cu_seqlens do not match input_lengths")

    defer_cu = (
        ratio == 4
        and int(self.indexer_state_cache.seq_size_per_block) == kv_tokens_per_block
        and raw_keys.is_cuda
    )
    if not defer_cu:
        check_cu_partition()

    position_ids = self.main_inputs.combo_position_ids
    index_factor = int(rope_config.index_factor)
    if position_ids is None or int(position_ids.numel()) != token_count * index_factor:
        actual = 0 if position_ids is None else int(position_ids.numel())
        raise RuntimeError(
            "qwen4_exp QSA position ids do not match the configured "
            f"index_factor={index_factor}: got {actual} values for "
            f"{token_count} tokens"
        )
    positions = position_ids.reshape(token_count, index_factor)
    max_len = max(lengths)
    rotary_dim = int(rope_config.dim)
    rope_cos = torch.zeros(
        (batch_size, max_len, rotary_dim),
        dtype=raw_keys.dtype,
        device=raw_keys.device,
    )
    rope_sin = torch.zeros_like(rope_cos)
    offset = 0
    for request_idx, seq_len in enumerate(lengths):
        end = offset + seq_len
        # The MTP draft prefill carries the engine's shifted positions
        # (prompt positions moved one row left plus a tail anchor), so its
        # RoPE positions intentionally differ from the logical cache rows.
        logical_positions = None
        if is_qsa_rope_style(rope_config, "Base") and not self.is_mtp_draft:
            prefix_len = 0
            logical_positions = prefix_len + torch.arange(
                seq_len, dtype=torch.int64, device=raw_keys.device
            )
        try:
            cos, sin = build_qsa_rope(
                positions[offset:end].reshape(-1),
                rope_config,
                token_count=seq_len,
                dtype=raw_keys.dtype,
                device=raw_keys.device,
                logical_positions=logical_positions,
            )
        except ValueError as error:
            prefixes = self.main_inputs.prefix_lengths
            prefix_head = prefixes[:6].tolist() if prefixes is not None else None
            raise RuntimeError(
                f"qwen4_exp QSA prefill {error} "
                f"[request={request_idx} seq_len={seq_len} "
                f"lengths[:6]={lengths[:6]} prefix_lengths[:6]={prefix_head} "
                f"is_mtp_draft={self.is_mtp_draft}]"
            ) from error
        rope_cos[request_idx, :seq_len].copy_(cos)
        rope_sin[request_idx, :seq_len].copy_(sin)
        offset = end

    if (
        ratio == 4
        and int(self.indexer_state_cache.seq_size_per_block) == kv_tokens_per_block
        and raw_keys.is_cuda
    ):
        from rtp_llm.models_py.modules.qwen4_exp.qsa.indexer_prefill_triton import (
            write_zero_prefix_prefill,
        )

        if self._side_cache_undo is not None:
            raise RuntimeError("qwen4_exp QSA side-cache transaction is already active")
        try:
            fused_result = write_zero_prefix_prefill(
                raw_keys,
                cu_seqlens,
                lengths,
                rope_cos,
                rope_sin,
                indexer.k_norm_gamma,
                kv_pool,
                kv_table,
                state_pool,
                state_table,
                page_size=kv_tokens_per_block,
                norm_eps=float(indexer.norm_eps),
                emit_dense_keys=True,
                host_cu_seqlens=(
                    getattr(self.main_inputs, "cu_seqlens", None)
                    if not self.main_inputs.is_cuda_graph
                    else None
                ),
                host_kv_table=(
                    getattr(self.indexer_kv_inputs, "kv_cache_kernel_block_id", None)
                    if not self.main_inputs.is_cuda_graph
                    else None
                ),
                host_state_table=(
                    getattr(self.indexer_state_inputs, "kv_cache_block_id", None)
                    if not self.main_inputs.is_cuda_graph
                    else None
                ),
            )
        except ValueError:
            # Preserve the runtime's CU diagnostic; error-only checking
            # happens before either side pool was mutated by the writer.
            if defer_cu:
                check_cu_partition()
            raise
        if fused_result is not None:
            object.__setattr__(self, "_side_cache_undo", fused_result.pop("undo"))
            return lengths, rope_cos, rope_sin, fused_result

    # Unsupported geometry or aliased pages returned without a mutation.
    # The fallback writer still needs the ordinary partition preflight.
    if defer_cu:
        check_cu_partition()
    result = self._write_indexer_cache_transactional(
        raw_keys,
        cu_seqlens,
        start_positions,
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
    )
    return lengths, rope_cos, rope_sin, result


def select_draft_graph_tokens(
    self,
    q: torch.Tensor,
    raw_keys: torch.Tensor,
    *,
    indexer: Any,
    rope_config: Any,
    raw_query: bool = False,
) -> torch.Tensor:
    """Capture a fixed-capacity draft plan; replay validates host mirrors."""
    from rtp_llm.models_py.model_desc.qwen4_exp_runtime.draft_prefill_graph import (
        draft_graph_lanes,
        draft_graph_rows,
    )
    from rtp_llm.models_py.modules.qwen4_exp.qsa.indexer_decode_triton import (
        write_draft_window_with_undo_,
    )
    from rtp_llm.models_py.modules.qwen4_exp.qsa.indexer_paged_score import (
        qsa_paged_indexer_score,
    )

    inputs = self.main_inputs
    if (
        not self.is_mtp_draft
        or not inputs.is_cuda_graph
        or not inputs.is_prefill
        or inputs.is_target_verify
    ):
        raise RuntimeError(
            "QSA draft graph requires an explicit MTP draft prefill context"
        )
    if int(indexer.compress_ratio) != 4 or int(indexer.head_dim) != 128:
        raise RuntimeError("QSA draft graph requires ratio 4 and head dimension 128")
    if (
        not q.is_cuda
        or q.dtype != torch.bfloat16
        or raw_keys.dtype != q.dtype
        or raw_keys.device != q.device
    ):
        raise RuntimeError("QSA draft graph requires device-local BF16 projections")
    tokens, heads, dim = q.shape
    batch = inputs.input_lengths_device.numel()
    lanes = draft_graph_lanes(inputs, tokens)
    if tuple(raw_keys.shape) != (tokens, dim):
        raise RuntimeError("QSA draft graph raw-key projection capacity changed")
    for tagged in (
        self.main_inputs,
        self.indexer_kv_inputs,
        self.indexer_state_inputs,
    ):
        if (
            not tagged.is_cuda_graph
            or not tagged.is_prefill
            or tagged.is_target_verify
            or tagged.context_parallel_info is not None
            or tagged.cache_store_inputs is not None
        ):
            raise RuntimeError("QSA draft graph cache-region modes disagree")
        for name in (
            "input_lengths_device",
            "prefix_lengths_device",
            "cu_seqlens_device",
        ):
            if not _same_tensor_metadata(getattr(inputs, name), getattr(tagged, name)):
                raise RuntimeError("QSA draft graph cache-region metadata disagree")
    page_size = int(self.indexer_kv_cache.seq_size_per_block)
    if (
        page_size not in (64, 128)
        or self.indexer_state_cache.seq_size_per_block != page_size
    ):
        raise RuntimeError("QSA draft graph requires matching 64/128-token side pages")
    kv_pool = _typed_2d_pool(
        self.indexer_kv_cache,
        tag=INDEXER_KV_TAG,
        storage_dtype=torch.uint8,
        payload_dtype=torch.bfloat16,
        entries=page_size // 4,
        width=128,
    )
    state_pool = _typed_2d_pool(
        self.indexer_state_cache,
        tag=INDEXER_STATE_TAG,
        storage_dtype=torch.float32,
        payload_dtype=torch.float32,
        entries=8,
        width=128,
    )
    kv_table = _block_table(
        self.indexer_kv_inputs,
        "kv_cache_kernel_block_id_device",
        tag=INDEXER_KV_TAG,
        batch_size=batch,
        device=q.device,
    )
    state_table = _block_table(
        self.indexer_state_inputs,
        "kv_cache_block_id_device",
        tag=INDEXER_STATE_TAG,
        batch_size=batch,
        device=q.device,
    )
    from rtp_llm.models_py.modules.qwen4_exp.qsa.draft_prepare_triton import (
        try_prepare,
        try_prepare_metadata,
    )

    get_frequency = getattr(indexer, "rope_inv_freq", None)
    frequency = get_frequency(rope_config) if callable(get_frequency) else None
    inverse = None
    metadata = try_prepare_metadata(
        q,
        inputs.input_lengths_device,
        inputs.prefix_lengths_device,
        inputs.cu_seqlens_device,
        frequency,
        rope_config=rope_config,
        q_gamma=indexer.q_norm_gamma if raw_query else None,
        norm_eps=indexer.norm_eps,
    )
    if metadata is not None:
        padded, block_cos, block_sin, weights, compressed_lengths, visible = metadata
        prepared = metadata
    else:
        if raw_query:
            from rtp_llm.models_py.modules.base.common.zero_centered_rms_norm import (
                exact_head_rms_norm,
            )

            q = exact_head_rms_norm(q, indexer.q_norm_gamma, indexer.norm_eps)
        sources, inverse, valid, positions, visible = draft_graph_rows(inputs, tokens)
        prepared = try_prepare(
            q, positions, sources, visible, frequency, rope_config=rope_config
        )
        if prepared is not None:
            padded, block_cos, block_sin, weights, compressed_lengths = prepared
    if prepared is None:
        cos, sin = build_qsa_rope_from_logical_positions(
            positions,
            rope_config,
            token_count=tokens,
            dtype=q.dtype,
            device=q.device,
            inv_freq=frequency,
        )
        block_starts = positions - positions.remainder(4)
        block_cos, block_sin = build_qsa_rope_from_logical_positions(
            block_starts,
            rope_config,
            token_count=tokens,
            dtype=q.dtype,
            device=q.device,
            inv_freq=frequency,
        )
    if self._side_cache_undo is not None:
        raise RuntimeError("QSA draft graph side-cache transaction is already active")
    undo = write_draft_window_with_undo_(
        raw_keys,
        inputs.cu_seqlens_device,
        inputs.prefix_lengths_device,
        inputs.input_lengths_device,
        block_cos,
        block_sin,
        indexer.k_norm_gamma,
        kv_pool,
        kv_table,
        state_pool,
        state_table,
        norm_eps=float(indexer.norm_eps),
    )
    object.__setattr__(self, "_side_cache_undo", undo)
    if prepared is None:
        rotated = apply_partial_rope(q, cos.unsqueeze(1), sin.unsqueeze(1))
        padded = rotated.index_select(0, sources).reshape(batch, lanes, heads, dim)
        weights = torch.full(
            (batch * lanes, heads),
            1.0 / math.sqrt(dim),
            device=q.device,
            dtype=torch.float32,
        )
        compressed_lengths = visible // 4
    logits = qsa_paged_indexer_score(
        padded.contiguous(),
        weights,
        kv_pool.flatten(0, 1),
        kv_table,
        compressed_lengths,
        block_size=page_size // 4,
        max_ctx_len=int(kv_table.shape[1]) * (page_size // 4),
        validate_block_table=False,
    )
    from rtp_llm.models_py.modules.qwen4_exp.qsa.cub_topk import (
        try_cub_draft_qsa_tokens,
    )

    if int(indexer.token_budget) == 2048:
        packed = try_cub_draft_qsa_tokens(
            logits,
            visible.reshape(-1),
            inputs.input_lengths_device,
            inputs.cu_seqlens_device,
        )
        if packed is not None:
            return packed
    selected = select_qsa_paged_tokens(
        logits,
        visible.reshape(-1),
        compress_ratio=4,
        token_budget=int(indexer.token_budget),
        validate_lengths=False,
        scores_are_masked=True,
    )
    from rtp_llm.models_py.modules.qwen4_exp.qsa.draft_layout_triton import (
        try_restore_selection,
    )

    if inverse is None:
        _, inverse, valid, _, _ = draft_graph_rows(inputs, tokens)
    restored = try_restore_selection(selected, inverse, valid)
    if restored is not None:
        return restored
    packed = selected.index_select(0, inverse)
    return torch.where(valid.unsqueeze(1), packed, -1)


def select_draft_incremental_prefill_tokens(
    self,
    q: torch.Tensor,
    raw_keys: torch.Tensor,
    *,
    indexer: Any,
    rope_config: Any,
) -> torch.Tensor:
    return self._incremental_prefill_selection(
        q, raw_keys, indexer=indexer, rope_config=rope_config, draft=True
    )


def select_prefix_reuse_prefill_tokens(
    self,
    q: torch.Tensor,
    raw_keys: torch.Tensor,
    *,
    indexer: Any,
    rope_config: Any,
) -> torch.Tensor:
    """Selection for a page-aligned prefix-reuse target prefill."""
    return self._incremental_prefill_selection(
        q, raw_keys, indexer=indexer, rope_config=rope_config, draft=False
    )


def select_draft_prefix_reuse_prefill_tokens(
    self,
    q: torch.Tensor,
    raw_keys: torch.Tensor,
    *,
    indexer: Any,
    rope_config: Any,
) -> torch.Tensor:
    """Serve a draft-model prefix hit before bounded MTP continuation.

    MTP hands this prefill to the draft model with local cu_kv coordinates,
    while prefix_lengths and the paged cache tables still describe the
    absolute target cache rows. QSA must score/write by the latter, then
    subsequent accepted-token commits use bounded draft incremental mode.
    """
    return self._incremental_prefill_selection(
        q,
        raw_keys,
        indexer=indexer,
        rope_config=rope_config,
        draft=False,
        draft_prefix_reuse=True,
    )


def _incremental_prefill_selection(
    self,
    q: torch.Tensor,
    raw_keys: torch.Tensor,
    *,
    indexer: Any,
    rope_config: Any,
    draft: bool,
    draft_prefix_reuse: bool = False,
) -> torch.Tensor:
    """Shared paged selection for incremental prefill builds.

    ``draft=True`` serves the post-rejection MTP draft commit (shifted
    positions, at most ``ratio`` new tokens per row).  ``draft=False``
    serves a page-aligned prefix-reuse target prefill: the writer seals the
    request's completed blocks into both side pools first, so one paged
    score covers the reused prefix blocks and the new blocks uniformly and
    the emitted token ids are absolute KV positions.
    """
    label = "draft incremental prefill" if draft else "prefix-reuse prefill"

    token_count = int(raw_keys.shape[0]) if raw_keys.dim() == 2 else -1
    plan = self._draft_incremental_prefill_geometry(
        indexer=indexer,
        token_count=token_count,
        device=raw_keys.device,
        rope_config=rope_config,
        draft=draft,
        draft_prefix_reuse=draft_prefix_reuse,
    )
    head_num = int(plan.head_num)
    head_dim = int(plan.head_dim)
    if q.dim() != 3 or tuple(q.shape) != (token_count, head_num, head_dim):
        raise RuntimeError(
            f"qwen4_exp QSA {label} q must be packed "
            f"[{token_count}, {head_num}, {head_dim}], got {tuple(q.shape)}"
        )
    if raw_keys.dim() != 2 or tuple(raw_keys.shape) != (token_count, head_dim):
        raise RuntimeError(
            f"qwen4_exp QSA {label} raw keys must be packed "
            f"[{token_count}, {head_dim}], got {tuple(raw_keys.shape)}"
        )
    if q.dtype != torch.bfloat16 or raw_keys.dtype != torch.bfloat16:
        raise RuntimeError(
            f"qwen4_exp QSA {label} currently requires " "BF16 q/raw keys"
        )
    if q.device != raw_keys.device:
        raise RuntimeError(f"qwen4_exp QSA {label} q/raw keys must share " "a device")
    if not q.is_cuda:
        raise RuntimeError(
            f"qwen4_exp QSA {label} paged scoring requires " "CUDA tensors"
        )
    triton_draft_window = (
        draft
        and max(plan.lengths) <= 4
        and int(plan.ratio) == 4
        and int(plan.head_dim) == 128
        and int(plan.kv_tokens_per_block) in (64, 128)
        and int(plan.state_tokens_per_block) == int(plan.kv_tokens_per_block)
    )
    if triton_draft_window:
        from rtp_llm.models_py.modules.qwen4_exp.qsa.indexer_decode_triton import (
            write_draft_window_with_undo_,
        )

        if self._side_cache_undo is not None:
            raise RuntimeError("qwen4_exp QSA side-cache transaction is already active")
        undo = write_draft_window_with_undo_(
            raw_keys,
            plan.cu_seqlens,
            plan.prefixes.to(torch.int32).contiguous(),
            self.main_inputs.input_lengths.to(
                device=raw_keys.device, dtype=torch.int32
            ).contiguous(),
            plan.rope_cos,
            plan.rope_sin,
            indexer.k_norm_gamma,
            plan.kv_pool,
            plan.kv_table,
            plan.state_pool,
            plan.state_table,
            norm_eps=float(indexer.norm_eps),
        )
        object.__setattr__(self, "_side_cache_undo", undo)
    else:
        self._write_indexer_cache_transactional(
            raw_keys,
            plan.cu_seqlens,
            plan.prefixes,
            plan.rope_cos,
            plan.rope_sin,
            indexer.k_norm_gamma,
            plan.kv_pool,
            plan.kv_table,
            plan.state_pool,
            plan.state_table,
            ratio=int(plan.ratio),
            kv_tokens_per_block=int(plan.kv_tokens_per_block),
            state_tokens_per_block=int(plan.state_tokens_per_block),
            norm_eps=float(indexer.norm_eps),
            rope_is_token_aligned=True,
        )

    rotated_q = apply_partial_rope(
        q,
        plan.current_cos.unsqueeze(1),
        plan.current_sin.unsqueeze(1),
    )
    score_weights = torch.full(
        (token_count, head_num),
        1.0 / math.sqrt(head_dim),
        dtype=torch.float32,
        device=q.device,
    )
    from rtp_llm.models_py.modules.qwen4_exp.qsa.indexer_paged_score import (
        qsa_paged_indexer_score,
    )

    # The draft has at most one compression group's worth of rows per
    # request. Pad only that short dimension so all requests share one
    # paged-score launch and one top-k selection. Invalid rows have zero
    # visible context and are removed before returning packed token IDs.
    lengths = plan.lengths
    batch_size = len(lengths)
    max_rows = max(lengths)
    offsets = []
    valid_rows = []
    offset = 0
    for request_idx, length in enumerate(lengths):
        offsets.extend(offset + min(row, length - 1) for row in range(max_rows))
        valid_rows.extend(request_idx * max_rows + row for row in range(length))
        offset += length
    row_map = torch.tensor(offsets, dtype=torch.int64, device=q.device)
    padded_q = rotated_q.index_select(0, row_map).reshape(
        batch_size, max_rows, head_num, head_dim
    )
    padded_weights = score_weights.index_select(0, row_map)
    compressed_lengths = plan.compressed_lengths
    block_logits = qsa_paged_indexer_score(
        padded_q,
        padded_weights,
        plan.kv_pool.flatten(0, 1),
        plan.kv_table,
        compressed_lengths,
        block_size=int(plan.kv_entries_per_block),
        max_ctx_len=int(plan.max_ctx_len),
        validate_block_table=False,
    )
    selected = select_qsa_paged_tokens(
        block_logits,
        plan.visible_lengths.reshape(-1).to(torch.int32),
        compress_ratio=int(plan.ratio),
        token_budget=int(plan.token_budget),
        validate_lengths=False,
    )
    return selected.index_select(
        0, torch.tensor(valid_rows, dtype=torch.int64, device=q.device)
    )
