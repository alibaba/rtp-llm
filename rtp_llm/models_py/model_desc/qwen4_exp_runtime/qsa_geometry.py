"""Pure validation and layout planning before projection or pool writes."""

from __future__ import annotations

from typing import Any

import torch

from rtp_llm.models.qwen4_exp.qwen4_exp_kv_cache import (
    INDEXER_KV_TAG,
    INDEXER_STATE_TAG,
)

from .qsa_cache_views import (
    _block_table,
    _required_blocks_verdict,
    _side_tables_verdict,
    _typed_2d_pool,
    _validate_required_blocks,
    qsa_assert_valid,
    qsa_strict_validation_enabled,
)
from .qsa_plans import IncrementalPrefillPlan, TargetVerifyPlan
from .qsa_step_plan import get_prefill_step_plan, host_input_mirror


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

    The layer-independent metadata (lengths, prefixes, cu boundaries, logical
    positions, visible/compressed lengths, both indexer RoPE tables) is
    frozen once per model step in a ``Qwen4ExpPrefillStepPlan``
    (:mod:`.qsa_step_plan`) and reused by every layer; revalidating the same
    GPU lengths and rebuilding the same device tensors per layer would add
    host work without seeing new metadata (the ``_target_verify_plan``
    precedent).  What stays per layer here is what IS layer-local: this
    layer's pool views and block tables, and their required-blocks verdicts
    enqueued before any pool mutation.
    """
    step = get_prefill_step_plan(
        self,
        indexer=indexer,
        token_count=token_count,
        device=device,
        rope_config=rope_config,
        draft=draft,
        draft_prefix_reuse=draft_prefix_reuse,
    )
    head_dim = step.head_dim
    kv_entries_per_block = step.kv_entries_per_block

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
        entries=2 * step.ratio,
        width=head_dim,
    )
    if kv_pool.device != device or state_pool.device != device:
        raise RuntimeError(
            "qwen4_exp QSA side pools and draft projection must share a device"
        )
    kv_table = _block_table(
        self.indexer_kv_inputs,
        "kv_cache_kernel_block_id_device",
        tag=INDEXER_KV_TAG,
        batch_size=step.batch_size,
        device=device,
    )
    state_table = _block_table(
        self.indexer_state_inputs,
        "kv_cache_block_id_device",
        tag=INDEXER_STATE_TAG,
        batch_size=step.batch_size,
        device=device,
    )

    if qsa_strict_validation_enabled():
        _validate_required_blocks(
            kv_table,
            step.required_kv_columns,
            pool_blocks=int(kv_pool.shape[0]),
            tag=INDEXER_KV_TAG,
        )
        # A completed group may read committed raw keys preceding the new
        # rows. Validate that source tail and every destination absolute
        # state page, keeping the original per-block diagnostics.
        state_table_width = int(state_table.shape[1])
        for request_idx, (prefix, length) in enumerate(
            zip(step.prefix_values, step.lengths)
        ):
            first_needed = prefix - (prefix % step.ratio)
            logical_blocks = {
                position // step.state_tokens_per_block
                for position in range(first_needed, prefix + length)
            }
            for logical_block in logical_blocks:
                if logical_block < 0 or logical_block >= state_table_width:
                    raise RuntimeError(
                        "QSA cache 'indexer_state' block table does not cover the "
                        "draft incremental tail"
                    )
                block_id = int(state_table[request_idx, logical_block].item())
                if block_id <= 0 or block_id >= int(state_pool.shape[0]):
                    raise RuntimeError(
                        "QSA cache 'indexer_state' draft incremental tail resolves "
                        "to an unallocated or out-of-range physical block"
                    )
    else:
        # Both verdict groups validate step-frozen inputs (the same table
        # tensors and required-column vectors for every layer of a step)
        # against this layer's pool block counts, so one dual-table
        # reduction plus one async assert replaces the two separate
        # verdict/assert pairs per layer (six device launches -> two).
        qsa_assert_valid(
            _side_tables_verdict(
                kv_table,
                step.required_kv_columns,
                state_table,
                step.required_state_columns_device,
                kv_pool_blocks=int(kv_pool.shape[0]),
                state_pool_blocks=int(state_pool.shape[0]),
            ),
            "QSA cache 'indexer_kv'/'indexer_state' block table does not cover "
            "or resolve the draft incremental tail",
        )

    return IncrementalPrefillPlan(
        **{
            "lengths": step.lengths,
            "batch_size": step.batch_size,
            "head_dim": step.head_dim,
            "head_num": step.head_num,
            "ratio": step.ratio,
            "token_budget": step.token_budget,
            "prefixes": step.prefixes_i64,
            "cu_seqlens": step.cu_seqlens,
            "logical_positions": step.logical_positions,
            "current_cos": step.current_cos,
            "current_sin": step.current_sin,
            "visible_lengths": step.visible_lengths,
            "compressed_lengths": step.compressed_lengths,
            "max_ctx_len": step.max_ctx_len,
            "kv_tokens_per_block": step.kv_tokens_per_block,
            "kv_entries_per_block": step.kv_entries_per_block,
            "state_tokens_per_block": step.state_tokens_per_block,
            "kv_pool": kv_pool,
            "state_pool": state_pool,
            "kv_table": kv_table,
            "state_table": state_table,
            "rope_cos": step.rope_cos,
            "rope_sin": step.rope_sin,
        },
        step=step,
    )


def _target_verify_geometry(
    self,
    *,
    indexer: Any,
    token_count: int,
    device: torch.device,
) -> TargetVerifyPlan:
    """Validate and materialise the bounded physical-tail overwrite plan.

    This method is intentionally side-effect free. The model calls it before
    either the QSA projection or the production main-cache writer, then the
    target selection path repeats it immediately before its side-pool write.
    """
    graph_target = bool(self.main_inputs.is_cuda_graph)
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
    # Request-data sign check on the host source/mirror: the graph branch
    # already read the pinned host tensor for free, and the eager branch now
    # does the same instead of synchronizing the device copy per layer. A
    # device source (RTP_LLM_DEVICE_INPUT) is read back once per model step
    # through the shared mirror cache.
    prefixes_host = host_input_mirror(self.main_inputs, "prefix_lengths")
    if prefixes_host is None:
        prefixes_host = prefixes
    if bool(torch.any(prefixes_host < 0).item()):
        raise RuntimeError(
            "qwen4_exp QSA target-verify prefix lengths must be non-negative"
        )
    prefixes_device = prefixes.to(
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
        if qsa_strict_validation_enabled():
            if not bool(torch.equal(cu_seqlens, expected_cu)):
                raise RuntimeError(
                    "qwen4_exp QSA target-verify cu_seqlens do not match gamma+1"
                )
        else:
            # Per eager target-verify layer: enqueue-only device verdict; a
            # mismatch traps fail-stop at the next host synchronization.
            qsa_assert_valid(
                (cu_seqlens == expected_cu).all(),
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
        if qsa_strict_validation_enabled():
            if not bool(torch.equal(cu_kv_seqlens, expected_cu_kv)):
                raise RuntimeError(
                    "qwen4_exp QSA target-verify cu_kv_seqlens do not match "
                    "prefix+gamma+1"
                )
        else:
            qsa_assert_valid(
                (cu_kv_seqlens == expected_cu_kv).all(),
                "qwen4_exp QSA target-verify cu_kv_seqlens do not match "
                "prefix+gamma+1",
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
            device=device, dtype=torch.int32, non_blocking=True
        ).reshape(batch_size, query_len, 3)
        expected_positions = prefixes_device[:, None, None] + offsets[None, :, None]
        expected_positions = expected_positions.expand(-1, -1, 3)
        message = (
            "qwen4_exp QSA target-verify only supports contiguous text-only "
            "MRoPE whose three axes equal prefix+j"
        )
        if qsa_strict_validation_enabled():
            if not bool(torch.equal(positions, expected_positions)):
                raise RuntimeError(message)
        else:
            qsa_assert_valid((positions == expected_positions).all(), message)

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
    # Host arithmetic on the prefix mirror: a device-resident source
    # (RTP_LLM_DEVICE_INPUT) is read back once per model step by the mirror
    # cache instead of reducing the device copy in every layer.  The mirror
    # max equals the legacy device ``compressed_lengths.max()`` because
    # floor-division is monotone in the visible length.
    max_ctx_len = (int(prefixes_host.max().item()) + query_len) // ratio
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
        if qsa_strict_validation_enabled():
            _validate_required_blocks(
                kv_table,
                required_kv_columns,
                pool_blocks=int(kv_pool.shape[0]),
                tag=INDEXER_KV_TAG,
            )
        else:
            qsa_assert_valid(
                _required_blocks_verdict(
                    kv_table,
                    required_kv_columns,
                    pool_blocks=int(kv_pool.shape[0]),
                ),
                "QSA cache 'indexer_kv' block table does not cover or resolve the "
                "target-verify tail",
            )

    # STATE_RING payload offsets wrap, but the framework block table keeps
    # absolute logical-page columns (including sentinel holes). Validate the
    # small committed-tail + candidate window explicitly before any write.
    # Production folds the per-request, per-block state_table[i, j].item()
    # loop into one required-blocks verdict over the full [0, required) prefix
    # window (a superset of the old tail window); strict mode keeps the
    # original per-block diagnostics. Host prefix values come from the pinned
    # prefix_lengths mirror for both graph and eager modes.
    # Host prefix values from the mirror: a device-resident source was already
    # read back once per model step by the sign check above.
    prefix_values = prefixes_host.tolist()
    # Pinned host staging: the eager branch below uploads it asynchronously, so
    # the verdict's ``required_columns.to(device)`` stays a no-op instead of a
    # pageable blocking H2D that synchronizes the stream in every layer.
    required_state_columns = torch.tensor(
        [
            (prefix + query_len + state_tokens_per_block - 1) // state_tokens_per_block
            for prefix in prefix_values
        ],
        dtype=torch.int32,
        pin_memory=True,
    )
    if qsa_strict_validation_enabled():
        state_table_width = int(state_table.shape[1])
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
                    block_id = int(state_table[request_idx, logical_block].item())
                    if block_id <= 0 or block_id >= int(state_pool.shape[0]):
                        raise RuntimeError(
                            "QSA cache 'indexer_state' target-verify tail resolves to "
                            "an unallocated or out-of-range physical block"
                        )
    elif graph_target:
        # Graph capture materialises synthetic block tables whose entries are
        # unallocated placeholders; only the table width is real. Keep the
        # capture-time coverage check host-side (required_state_columns is a
        # CPU tensor built from the pinned host prefix mirror, so this stays
        # capture-safe without a device readback) and skip the physical-block
        # resolution verdict, mirroring the legacy capture-time semantics.
        if int(required_state_columns.max()) > int(state_table.shape[1]):
            raise RuntimeError(
                "QSA cache 'indexer_state' block table does not cover the "
                "target-verify tail"
            )
    else:
        # Async upload from pinned staging: the blocking pageable H2D inside
        # the verdict synchronizes the stream in every eager target-verify
        # layer.
        required_state_columns_device = required_state_columns.to(
            device=device, non_blocking=True
        )
        qsa_assert_valid(
            _required_blocks_verdict(
                state_table,
                required_state_columns_device,
                pool_blocks=int(state_pool.shape[0]),
            ),
            "QSA cache 'indexer_state' target-verify tail is not covered or "
            "resolves to an unallocated or out-of-range physical block",
        )

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
