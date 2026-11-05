"""Pure validation and layout planning before projection or pool writes."""

from __future__ import annotations

from typing import Any

import torch

from rtp_llm.models.qwen4_exp.qwen4_exp_kv_cache import (
    INDEXER_KV_TAG,
    INDEXER_STATE_TAG,
)
from rtp_llm.models_py.modules.qwen4_exp.qsa.indexer import (
    build_qsa_rope,
    build_qsa_rope_from_logical_positions,
    is_qsa_rope_style,
)

from .qsa_cache_views import _block_table, _typed_2d_pool, _validate_required_blocks
from .qsa_plans import IncrementalPrefillPlan, TargetVerifyPlan


def _draft_incremental_prefill_geometry(
    self,
    *,
    indexer: Any,
    token_count: int,
    device: torch.device,
    rope_config: Any,
    draft: bool = True,
    draft_prefix_reuse: bool = False,
) -> IncrementalPrefillPlan:
    """Validate one incremental prefill build without side effects.

    ``draft=True`` is the post-rejection MTP draft commit (shifted positions,
    at most ``ratio`` new tokens).  ``draft=False`` is a page-aligned
    prefix-reuse target prefill: the request starts at its absolute prefix
    position, every reused prefix block is a complete compressed block held
    by the side pool, and the writer seals complete blocks exactly as the
    draft path does.
    """
    checks = self.take_prewrite_checks()
    if draft_prefix_reuse:
        if draft or not self.is_mtp_draft:
            raise RuntimeError(
                "qwen4_exp QSA draft prefix-reuse requires an explicit MTP "
                "draft context"
            )
        label = "MTP draft prefix-reuse prefill"
    else:
        label = "draft incremental prefill" if draft else "prefix-reuse prefill"
    lengths = self._validate_mode_and_metadata(
        expect_prefill=True,
        allow_draft_incremental=draft,
        allow_prefix_reuse=not draft,
    )
    if self._target_verify_enabled():
        raise RuntimeError(
            "qwen4_exp QSA draft incremental prefill cannot be target verification"
        )
    if draft and not is_qsa_rope_style(rope_config, "Base"):
        # The MTP draft runs Base RoPE; the target keeps its own style
        # (MRoPE for the released checkpoint), which build_qsa_rope
        # dispatches on for both the writer and the scoring rotation.
        raise RuntimeError(
            "qwen4_exp QSA MTP draft incremental prefill requires Base RoPE"
        )

    batch_size = len(lengths)
    head_dim = int(indexer.head_dim)
    head_num = int(indexer.n_heads)
    ratio = int(indexer.compress_ratio)
    token_budget = int(indexer.token_budget)
    if min(head_dim, head_num, ratio) <= 0:
        raise RuntimeError(
            "qwen4_exp QSA draft incremental prefill has invalid indexer geometry"
        )
    if token_budget <= 0 or token_budget % ratio:
        raise RuntimeError(
            "qwen4_exp QSA draft incremental prefill token budget must be "
            "positive and divisible by the compression ratio"
        )
    if draft and max(lengths) > ratio:
        raise RuntimeError(
            "qwen4_exp QSA draft incremental prefill requires "
            f"max(input_lengths)={max(lengths)} <= compress_ratio={ratio}"
        )
    if sum(lengths) != token_count:
        raise RuntimeError(
            f"qwen4_exp QSA input_lengths={lengths} do not partition "
            f"the {token_count} packed draft tokens"
        )

    prefixes = self.main_inputs.prefix_lengths
    if (
        prefixes.dim() != 1
        or int(prefixes.numel()) != batch_size
        or prefixes.dtype != torch.int32
    ):
        raise RuntimeError(
            "qwen4_exp QSA draft incremental prefill requires int32 "
            "prefix_lengths [B]"
        )
    host_prefixes = self.host_values(prefixes)
    prefixes = prefixes.to(
        device=device, dtype=torch.int64, non_blocking=True
    ).contiguous()
    if any(value <= 0 if draft else value < 0 for value in host_prefixes):
        requirement = "nonzero" if draft else "non-negative"
        raise RuntimeError(
            f"qwen4_exp QSA {label} requires every prefix length to be "
            f"{requirement}"
        )
    sequence_lengths = self.main_inputs.sequence_lengths
    if sequence_lengths.numel() != 0:
        raise RuntimeError(
            "qwen4_exp QSA draft incremental prefill requires empty " "sequence_lengths"
        )

    cu_seqlens = self.main_inputs.cu_seqlens_device
    expected_cu = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    expected_cu[1:] = torch.tensor(lengths, dtype=torch.int32, device=device).cumsum(0)
    if (
        tuple(cu_seqlens.shape) != (batch_size + 1,)
        or cu_seqlens.dtype != torch.int32
        or cu_seqlens.device != device
    ):
        raise RuntimeError(
            "qwen4_exp QSA draft incremental prefill cu_seqlens do not "
            "match input_lengths"
        )
    checks.add(
        (cu_seqlens != expected_cu).any(),
        "qwen4_exp QSA draft incremental prefill cu_seqlens do not match input_lengths",
    )
    expected_cu_kv = torch.zeros_like(expected_cu)
    visible_ends = prefixes + torch.tensor(lengths, dtype=torch.int64, device=device)
    host_visible_ends = [
        prefix + length for prefix, length in zip(host_prefixes, lengths)
    ]
    if max(host_visible_ends) > torch.iinfo(torch.int32).max:
        raise RuntimeError(
            "qwen4_exp QSA draft incremental prefill positions exceed int32"
        )
    if sum(host_visible_ends) > torch.iinfo(torch.int32).max:
        raise RuntimeError(
            "qwen4_exp QSA draft incremental prefill cumulative KV lengths "
            "exceed int32"
        )
    expected_cu_kv[1:] = visible_ends.to(torch.int32).cumsum(0)
    cu_kv_seqlens = self.main_inputs.cu_kv_seqlens_device
    if (
        tuple(cu_kv_seqlens.shape) != (batch_size + 1,)
        or cu_kv_seqlens.dtype != torch.int32
        or cu_kv_seqlens.device != device
    ):
        raise RuntimeError(
            "qwen4_exp QSA draft incremental prefill cu_kv_seqlens do not "
            "provide device int32 [B + 1] metadata"
        )
    if draft_prefix_reuse:
        checks.add(
            (cu_kv_seqlens != expected_cu_kv).any()
            & (cu_kv_seqlens[1:] < cu_kv_seqlens[:-1]).any(),
            "qwen4_exp QSA MTP draft prefix-reuse has invalid local cu_kv_seqlens metadata",
        )
    else:
        checks.add(
            (cu_kv_seqlens != expected_cu_kv).any(),
            "qwen4_exp QSA draft incremental prefill cu_kv_seqlens do not match prefix+input lengths",
        )

    prefix_values = host_prefixes
    logical_positions = torch.cat(
        [
            torch.arange(prefix, prefix + length, device=device)
            for prefix, length in zip(prefix_values, lengths)
        ]
    ).to(torch.int64)
    position_ids = self.main_inputs.combo_position_ids
    index_factor = int(rope_config.index_factor)
    if position_ids is None or int(position_ids.numel()) != token_count * index_factor:
        actual = 0 if position_ids is None else int(position_ids.numel())
        raise RuntimeError(
            "qwen4_exp QSA draft incremental prefill position ids do not "
            f"match index_factor={index_factor}: got {actual} values for "
            f"{token_count} tokens"
        )
    try:
        current_cos, current_sin = build_qsa_rope(
            position_ids,
            rope_config,
            token_count=token_count,
            dtype=torch.bfloat16,
            device=device,
            logical_positions=logical_positions,
            checks=checks,
        )
    except ValueError as error:
        raise RuntimeError(
            f"qwen4_exp QSA draft incremental prefill {error}"
        ) from error

    kv_tokens_per_block = int(self.indexer_kv_cache.seq_size_per_block)
    state_tokens_per_block = int(self.indexer_state_cache.seq_size_per_block)
    if kv_tokens_per_block <= 0 or kv_tokens_per_block % ratio:
        raise RuntimeError(
            "qwen4_exp QSA indexer KV page size must be positive and divisible "
            f"by ratio={ratio}, got {kv_tokens_per_block}"
        )
    if state_tokens_per_block <= 0:
        raise RuntimeError("qwen4_exp QSA indexer state page size must be positive")
    if not draft and any(prefix % kv_tokens_per_block for prefix in host_prefixes):
        raise RuntimeError(
            "qwen4_exp QSA prefix reuse requires prefixes aligned to the "
            f"indexer KV page size ({kv_tokens_per_block})"
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
    k_norm_gamma = indexer.k_norm_gamma
    if (
        not isinstance(k_norm_gamma, torch.Tensor)
        or tuple(k_norm_gamma.shape) != (head_dim,)
        or k_norm_gamma.device != device
    ):
        raise RuntimeError(
            "qwen4_exp QSA draft incremental prefill k-norm gamma must be a "
            f"device-local [{head_dim}] tensor"
        )
    if kv_pool.device != device or state_pool.device != device:
        raise RuntimeError(
            "qwen4_exp QSA side pools and draft projection must share a device"
        )
    kv_table = _block_table(
        self.indexer_kv_inputs,
        "kv_cache_kernel_block_id_device",
        tag=INDEXER_KV_TAG,
        batch_size=batch_size,
        device=device,
    )
    state_table = _block_table(
        self.indexer_state_inputs,
        "kv_cache_block_id_device",
        tag=INDEXER_STATE_TAG,
        batch_size=batch_size,
        device=device,
    )

    row_offsets = torch.arange(max(lengths), dtype=torch.int64, device=device)
    visible_lengths = prefixes[:, None] + row_offsets[None, :] + 1
    valid_rows = (
        row_offsets[None, :]
        < torch.tensor(lengths, dtype=torch.int64, device=device)[:, None]
    )
    visible_lengths = torch.where(valid_rows, visible_lengths, 0)
    compressed_lengths = (visible_lengths // ratio).to(torch.int32)
    final_compressed_lengths = (visible_ends // ratio).to(torch.int32)
    required_kv_columns = (
        final_compressed_lengths + kv_entries_per_block - 1
    ) // kv_entries_per_block
    _validate_required_blocks(
        kv_table,
        required_kv_columns,
        pool_blocks=int(kv_pool.shape[0]),
        tag=INDEXER_KV_TAG,
        checks=checks,
    )

    # A completed group may read committed raw keys preceding the new rows.
    # Validate that source tail and every destination absolute state page.
    state_table_width = int(state_table.shape[1])
    state_pairs = []
    for request_idx, (prefix, length) in enumerate(zip(prefix_values, lengths)):
        first_needed = prefix - (prefix % ratio)
        logical_blocks = {
            position // state_tokens_per_block
            for position in range(first_needed, prefix + length)
        }
        for logical_block in logical_blocks:
            if logical_block < 0 or logical_block >= state_table_width:
                raise RuntimeError(
                    "QSA cache 'indexer_state' block table does not cover the "
                    "draft incremental tail"
                )
            state_pairs.append((request_idx, logical_block))
    from .qsa_cache_views import _validate_state_pairs

    _validate_state_pairs(
        state_table,
        state_pairs,
        int(state_pool.shape[0]),
        "draft incremental tail",
        checks=checks,
    )

    block_starts = logical_positions - logical_positions.remainder(ratio)
    try:
        rope_cos, rope_sin = build_qsa_rope_from_logical_positions(
            block_starts,
            rope_config,
            token_count=token_count,
            dtype=torch.bfloat16,
            device=device,
        )
    except ValueError as error:
        raise RuntimeError(
            f"qwen4_exp QSA draft incremental prefill {error}"
        ) from error

    try:
        checks.verify()
    except ValueError as error:
        raise RuntimeError(
            f"qwen4_exp QSA draft incremental prefill {error}"
        ) from error
    return IncrementalPrefillPlan(
        **{
            "lengths": lengths,
            "batch_size": batch_size,
            "head_dim": head_dim,
            "head_num": head_num,
            "ratio": ratio,
            "token_budget": token_budget,
            "prefixes": prefixes,
            "cu_seqlens": expected_cu,
            "current_cos": current_cos,
            "current_sin": current_sin,
            "visible_lengths": visible_lengths,
            "compressed_lengths": compressed_lengths,
            "max_ctx_len": max(
                (prefix + length) // ratio
                for prefix, length in zip(prefix_values, lengths)
            ),
            "kv_tokens_per_block": kv_tokens_per_block,
            "kv_entries_per_block": kv_entries_per_block,
            "state_tokens_per_block": state_tokens_per_block,
            "kv_pool": kv_pool,
            "state_pool": state_pool,
            "kv_table": kv_table,
            "state_table": state_table,
            "rope_cos": rope_cos,
            "rope_sin": rope_sin,
        }
    )


def _target_verify_geometry(
    self,
    *,
    indexer: Any,
    token_count: int,
    device: torch.device,
) -> TargetVerifyPlan:
    """Validate and materialise the bounded physical-tail overwrite plan.

    The model calls this before projection and shares the verified plan with
    its side writer. Direct component calls build and verify their own plan.
    """
    graph_target = bool(self.main_inputs.is_cuda_graph)
    checks = self.take_prewrite_checks()
    lengths = self._validate_mode_and_metadata(
        expect_prefill=True,
        allow_target_verify=True,
        allow_graph_target=graph_target,
    )
    if not self._target_verify_enabled():
        raise RuntimeError("qwen4_exp QSA expected a target-verify invocation")

    ratio = int(indexer.compress_ratio)
    head_dim = int(indexer.head_dim)
    head_num = int(indexer.n_heads)
    token_budget = int(indexer.token_budget)
    if min(ratio, head_dim, head_num) <= 0:
        raise RuntimeError("qwen4_exp QSA target-verify has invalid indexer geometry")
    if token_budget <= 0 or token_budget % ratio:
        raise RuntimeError(
            "qwen4_exp QSA target-verify token budget must be positive and "
            "divisible by the compression ratio"
        )
    k_norm_gamma = indexer.k_norm_gamma
    if (
        not isinstance(k_norm_gamma, torch.Tensor)
        or tuple(k_norm_gamma.shape) != (head_dim,)
        or k_norm_gamma.device != device
    ):
        raise RuntimeError(
            "qwen4_exp QSA target-verify k-norm gamma must be a device-local "
            f"[{head_dim}] tensor"
        )
    batch_size = len(lengths)
    query_len = lengths[0]
    if any(length != query_len for length in lengths):
        raise RuntimeError(
            "qwen4_exp QSA target-verify requires one uniform gamma+1 width"
        )
    if query_len > ratio:
        raise RuntimeError(
            "qwen4_exp QSA target-verify physical-tail overwrite requires "
            f"gamma+1={query_len} <= compress_ratio={ratio}"
        )
    if device.type != "cuda":
        raise RuntimeError(
            "qwen4_exp QSA target-verify paged scoring requires CUDA tensors"
        )
    if token_count != batch_size * query_len:
        raise RuntimeError(
            "qwen4_exp QSA target-verify input lengths do not partition "
            f"the {token_count} packed tokens"
        )

    prefixes = self.main_inputs.prefix_lengths
    sequence_lengths = self.main_inputs.sequence_lengths
    if prefixes.dim() != 1 or int(prefixes.numel()) != batch_size:
        raise RuntimeError(
            "qwen4_exp QSA target-verify requires one prefix length per request"
        )
    if prefixes.dtype != torch.int32:
        raise RuntimeError("qwen4_exp QSA target-verify prefix lengths must be int32")
    # GraphRunner retains a decode-length mirror for capture bookkeeping;
    # target scoring addresses history through prefix_lengths.
    if sequence_lengths.numel() != 0 and not graph_target:
        raise RuntimeError(
            "qwen4_exp QSA target-verify requires an empty sequence_lengths tensor"
        )
    if graph_target and prefixes.device.type != "cpu":
        raise RuntimeError("qwen4_exp QSA target Graph requires host prefixes")
    prefix_values = self.host_values(prefixes)
    if any(prefix < 0 for prefix in prefix_values):
        raise RuntimeError(
            "qwen4_exp QSA target-verify prefix lengths must be non-negative"
        )
    prefix_source = self.main_inputs.prefix_lengths_device if graph_target else prefixes
    prefixes_device = prefix_source.to(
        device=device, dtype=torch.int32, non_blocking=True
    ).contiguous()

    expected_cu = torch.arange(
        0,
        (batch_size + 1) * query_len,
        query_len,
        dtype=torch.int32,
        device=device,
    )
    cu_seqlens = self.main_inputs.cu_seqlens_device
    if (
        cu_seqlens.dtype != torch.int32
        or cu_seqlens.device != device
        or tuple(cu_seqlens.shape) != (batch_size + 1,)
    ):
        raise RuntimeError(
            "qwen4_exp QSA target-verify cu_seqlens do not match gamma+1"
        )
    if not graph_target:
        checks.add(
            (cu_seqlens != expected_cu).any(),
            "qwen4_exp QSA target-verify cu_seqlens do not match gamma+1",
        )
    cu_kv_seqlens = self.main_inputs.cu_kv_seqlens_device
    if (
        cu_kv_seqlens.dtype != torch.int32
        or cu_kv_seqlens.device != device
        or tuple(cu_kv_seqlens.shape) != (batch_size + 1,)
    ):
        raise RuntimeError(
            "qwen4_exp QSA target-verify cu_kv_seqlens do not match prefix+gamma+1"
        )
    if not graph_target:
        expected_cu_kv = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
        expected_cu_kv[1:] = (prefixes_device + query_len).cumsum(0)
        checks.add(
            (cu_kv_seqlens != expected_cu_kv).any(),
            "qwen4_exp QSA target-verify cu_kv_seqlens do not match prefix+gamma+1",
        )

    positions = self.main_inputs.combo_position_ids
    if positions is None or int(positions.numel()) != token_count * 3:
        actual = 0 if positions is None else int(positions.numel())
        raise RuntimeError(
            "qwen4_exp QSA target-verify requires three MRoPE positions per "
            f"token, got {actual} for {token_count} tokens"
        )
    offsets = None
    if not graph_target:
        offsets = torch.arange(query_len, dtype=torch.int32, device=device)
        positions = positions.to(
            device=device, dtype=torch.int64, non_blocking=True
        ).reshape(batch_size, query_len, 3)
        expected_positions = prefixes_device[:, None, None] + offsets[None, :, None]
        expected_positions = expected_positions.expand(-1, -1, 3)
        checks.add(
            (positions != expected_positions).any(),
            "qwen4_exp QSA target-verify only supports contiguous text-only "
            "MRoPE whose three axes equal prefix+j",
        )
    elif positions.device.type == "cpu":
        # Graph replay cannot re-read device capture buffers per step without a
        # host sync, so validate host-resident positions for free here. Device
        # positions rely on the position generator's uniform-axis contract;
        # see docs/design/qwen3.8_flash_next_tp8_bringup_handoff.md §3.
        host_offsets = torch.arange(query_len, dtype=torch.int32)
        host_positions = positions.to(dtype=torch.int32).reshape(
            batch_size, query_len, 3
        )
        expected_positions = prefixes.to(dtype=torch.int32).reshape(
            batch_size, 1, 1
        ) + host_offsets.reshape(1, query_len, 1)
        if not bool(torch.equal(host_positions, expected_positions.expand(-1, -1, 3))):
            raise RuntimeError(
                "qwen4_exp QSA target-verify only supports contiguous text-only "
                "MRoPE whose three axes equal prefix+j"
            )

    kv_tokens_per_block = int(self.indexer_kv_cache.seq_size_per_block)
    state_tokens_per_block = int(self.indexer_state_cache.seq_size_per_block)
    if kv_tokens_per_block <= 0 or kv_tokens_per_block % ratio:
        raise RuntimeError(
            "qwen4_exp QSA indexer KV page size must be positive and divisible "
            f"by ratio={ratio}, got {kv_tokens_per_block}"
        )
    if state_tokens_per_block <= 0:
        raise RuntimeError("qwen4_exp QSA indexer state page size must be positive")
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
    if kv_pool.device != device or state_pool.device != device:
        raise RuntimeError(
            "qwen4_exp QSA target-verify projection and side pools must share a device"
        )
    kv_table = _block_table(
        self.indexer_kv_inputs,
        "kv_cache_kernel_block_id_device",
        tag=INDEXER_KV_TAG,
        batch_size=batch_size,
        device=device,
    )
    state_table = _block_table(
        self.indexer_state_inputs,
        "kv_cache_block_id_device",
        tag=INDEXER_STATE_TAG,
        batch_size=batch_size,
        device=device,
    )

    from rtp_llm.models_py.modules.qwen4_exp.qsa.target_metadata_triton import (
        try_target_metadata,
    )

    metadata = try_target_metadata(prefixes_device, query_len, ratio)
    query_positions = block_starts = None
    if metadata is None:
        if offsets is None:
            offsets = torch.arange(query_len, dtype=torch.int32, device=device)
        visible_lengths = prefixes_device[:, None] + offsets[None, :] + 1
        compressed_lengths = visible_lengths // ratio
    else:
        visible_lengths, compressed_lengths, query_positions, block_starts = metadata
    if self.main_inputs.is_s_padded:
        active = state_table.ne(0).any(dim=1, keepdim=True)
        visible_lengths = torch.where(active, visible_lengths, 0)
        compressed_lengths = torch.where(active, compressed_lengths, 0)
    max_ctx_len = (max(prefix_values) + query_len) // ratio
    if graph_target:
        required_host = (max_ctx_len + kv_entries_per_block - 1) // kv_entries_per_block
        if required_host > int(kv_table.shape[1]):
            raise RuntimeError(
                "QSA cache 'indexer_kv' block table does not cover target verification"
            )
    else:
        required_kv_columns = (
            compressed_lengths[:, -1] + kv_entries_per_block - 1
        ) // kv_entries_per_block
        _validate_required_blocks(
            kv_table,
            required_kv_columns,
            pool_blocks=int(kv_pool.shape[0]),
            tag=INDEXER_KV_TAG,
            checks=checks,
        )

    # STATE_RING payload offsets wrap, but the framework block table keeps
    # absolute logical-page columns (including sentinel holes). Validate the
    # small committed-tail + candidate window explicitly before any write.
    state_table_width = int(state_table.shape[1])
    state_pairs = []
    for request_idx, prefix in enumerate(prefix_values):
        first_needed = max(0, prefix - (prefix % ratio))
        logical_blocks = {
            position // state_tokens_per_block
            for position in range(first_needed, prefix + query_len)
        }
        for logical_block in logical_blocks:
            if logical_block < 0 or logical_block >= state_table_width:
                raise RuntimeError(
                    "QSA cache 'indexer_state' block table does not cover the "
                    "target-verify tail"
                )
            if not graph_target:
                state_pairs.append((request_idx, logical_block))
    if not graph_target:
        from .qsa_cache_views import _validate_state_pairs

        _validate_state_pairs(
            state_table,
            state_pairs,
            int(state_pool.shape[0]),
            "target-verify tail",
            checks=checks,
        )

    checks.verify()
    return TargetVerifyPlan(
        **{
            "batch_size": batch_size,
            "query_len": query_len,
            "prefixes": prefixes_device,
            "cu_seqlens": expected_cu,
            "visible_lengths": visible_lengths,
            "compressed_lengths": compressed_lengths,
            "max_ctx_len": max_ctx_len,
            "query_positions": query_positions,
            "block_starts": block_starts,
            "kv_tokens_per_block": kv_tokens_per_block,
            "kv_entries_per_block": kv_entries_per_block,
            "state_tokens_per_block": state_tokens_per_block,
            "kv_pool": kv_pool,
            "state_pool": state_pool,
            "kv_table": kv_table,
            "state_table": state_table,
        },
        indexer_id=id(indexer),
    )
