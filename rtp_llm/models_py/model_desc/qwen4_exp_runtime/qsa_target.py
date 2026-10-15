"""QSA target planning and writes; the context owns undo."""

from __future__ import annotations

import math
from typing import Any

import torch

from rtp_llm.models_py.modules.qwen4_exp.qsa.indexer import (
    apply_partial_rope,
    build_qsa_rope_from_logical_positions,
)

from .qsa_selection import select_qsa_paged_tokens


def select_target_verify_tokens(
    self,
    q: torch.Tensor,
    raw_keys: torch.Tensor,
    *,
    indexer: Any,
    rope_config: Any,
) -> torch.Tensor:
    """Select all ``gamma + 1`` target rows using a bounded tail overwrite."""
    token_count = int(raw_keys.shape[0]) if raw_keys.dim() == 2 else -1
    plan = self._target_verify_plan
    object.__setattr__(self, "_target_verify_plan", None)
    if plan is None:
        plan = self._target_verify_geometry(
            indexer=indexer, token_count=token_count, device=raw_keys.device
        )
    elif (
        int(plan.batch_size) * int(plan.query_len) != token_count
        or plan.kv_pool.device != raw_keys.device
        or plan.indexer_id != id(indexer)
    ):
        raise RuntimeError("qwen4_exp QSA target verification plan changed")
    batch_size = int(plan.batch_size)
    query_len = int(plan.query_len)
    head_num = int(indexer.n_heads)
    head_dim = int(indexer.head_dim)
    ratio = int(indexer.compress_ratio)
    token_budget = int(indexer.token_budget)
    if q.dim() != 3 or tuple(q.shape) != (token_count, head_num, head_dim):
        raise RuntimeError(
            "qwen4_exp QSA target-verify q must be packed "
            f"[{token_count}, {head_num}, {head_dim}], got {tuple(q.shape)}"
        )
    if raw_keys.dim() != 2 or tuple(raw_keys.shape) != (token_count, head_dim):
        raise RuntimeError(
            "qwen4_exp QSA target-verify raw keys must be packed "
            f"[{token_count}, {head_dim}], got {tuple(raw_keys.shape)}"
        )
    if q.dtype != torch.bfloat16 or raw_keys.dtype != torch.bfloat16:
        raise RuntimeError(
            "qwen4_exp QSA target-verify currently requires BF16 q/raw keys"
        )
    if q.device != raw_keys.device:
        raise RuntimeError("qwen4_exp QSA target-verify q/raw keys must share a device")
    if not q.is_cuda:
        raise RuntimeError(
            "qwen4_exp QSA target-verify paged scoring requires CUDA tensors"
        )

    flat_query_positions = plan.query_positions
    block_starts = plan.block_starts
    if flat_query_positions is None or block_starts is None:
        query_positions = plan.visible_lengths - 1
        flat_query_positions = query_positions.reshape(-1).to(torch.int64)
        block_starts = flat_query_positions - flat_query_positions.remainder(ratio)
    get_frequency = getattr(indexer, "rope_inv_freq", None)
    frequency = get_frequency(rope_config) if callable(get_frequency) else None
    rope_cos, rope_sin = build_qsa_rope_from_logical_positions(
        block_starts,
        rope_config,
        token_count=token_count,
        dtype=raw_keys.dtype,
        device=raw_keys.device,
        inv_freq=frequency,
    )
    current_cos, current_sin = build_qsa_rope_from_logical_positions(
        flat_query_positions,
        rope_config,
        token_count=token_count,
        dtype=raw_keys.dtype,
        device=raw_keys.device,
        inv_freq=frequency,
    )
    target_writer = True
    target_writer = target_writer and (
        1 <= query_len <= ratio == 4
        and head_dim == 128
        and int(plan.kv_tokens_per_block) == 128
        and int(plan.state_tokens_per_block) == 128
    )
    if bool(self.main_inputs.is_cuda_graph) and not target_writer:
        raise RuntimeError("qwen4_exp QSA target Graph requires Triton writer")
    if target_writer:
        from rtp_llm.models_py.modules.qwen4_exp.qsa.indexer_decode_triton import (
            write_target_window_with_undo_,
        )

        if self._side_cache_undo is not None:
            raise RuntimeError("qwen4_exp QSA side-cache transaction is already active")
        undo = write_target_window_with_undo_(
            raw_keys,
            plan.prefixes,
            rope_cos,
            rope_sin,
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
            plan.prefixes.to(torch.int64),
            rope_cos,
            rope_sin,
            indexer.k_norm_gamma,
            plan.kv_pool,
            plan.kv_table,
            plan.state_pool,
            plan.state_table,
            ratio=ratio,
            kv_tokens_per_block=int(plan.kv_tokens_per_block),
            state_tokens_per_block=int(plan.state_tokens_per_block),
            norm_eps=float(indexer.norm_eps),
            rope_is_token_aligned=True,
        )

    q = q.view(batch_size, query_len, head_num, head_dim)
    rotated_q = apply_partial_rope(
        q,
        current_cos.view(batch_size, query_len, 1, -1),
        current_sin.view(batch_size, query_len, 1, -1),
    )
    get_weights = getattr(indexer, "score_weights", None)
    score_weights = (
        get_weights(token_count, q.device) if callable(get_weights) else None
    )
    if score_weights is None:
        score_weights = torch.full(
            (token_count, head_num),
            1.0 / math.sqrt(head_dim),
            dtype=torch.float32,
            device=q.device,
        )
    from rtp_llm.models_py.modules.qwen4_exp.qsa.indexer_paged_score import (
        qsa_paged_indexer_score,
    )

    compressed_lengths = plan.compressed_lengths
    block_logits = qsa_paged_indexer_score(
        rotated_q.contiguous(),
        score_weights,
        plan.kv_pool.flatten(0, 1),
        plan.kv_table,
        compressed_lengths,
        block_size=int(plan.kv_entries_per_block),
        validate_block_table=False,
        max_ctx_len=int(plan.max_ctx_len),
    )
    return select_qsa_paged_tokens(
        block_logits,
        plan.visible_lengths.reshape(-1),
        compress_ratio=ratio,
        token_budget=token_budget,
        validate_lengths=False,
    )
