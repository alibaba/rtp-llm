"""Restricted Qwen4 QSA side-cache runtime.

This module wires pure prefill, page-aligned prefix reuse, MTP-draft incremental prefill,
ordinary single-token decode and bounded speculative target verification into
the independent ``indexer_kv`` and ``indexer_state`` pools. Decode is
deliberately limited to text-only MRoPE, whose three current position axes all
equal ``sequence_lengths``.  The side state stores raw keys but no historical
MRoPE positions, so accepting a multimodal/non-linear position stream would
make a compressed block that crosses the prefill/decode boundary incorrect.

Target verification may overwrite the uncommitted physical tail without an
accepted-length callback only when its width is at most one compression group.
With ``G <= ratio`` it can complete at most one new compressed entry, while the
``2 * ratio`` raw-key ring cannot alias the committed partial group. Rejected
tail entries therefore remain outside the logical length and are overwritten
before they can become visible. CP, PD, padding and CUDA Graph remain
unsupported; prefix-cache reuse is restricted to page-aligned prefixes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch

from rtp_llm.models.qwen4_exp.qwen4_exp_kv_cache import (
    INDEXER_KV_TAG,
    INDEXER_STATE_TAG,
)
from rtp_llm.models_py.modules.qwen4_exp.indexer import (
    apply_partial_rope,
    build_qsa_rope,
    is_qsa_rope_style,
)
from rtp_llm.models_py.modules.qwen4_exp.indexer_compressor import (
    IndexerCacheUndo,
    restore_indexer_cache,
    write_indexer_cache,
)
from rtp_llm.ops.compute_ops import LayerKVCache, PyAttentionInputs


def select_qsa_paged_tokens(
    block_logits: torch.Tensor,
    token_lengths: torch.Tensor,
    *,
    compress_ratio: int,
    token_budget: int,
) -> torch.Tensor:
    """Expand scored block top-k plus the unscored partial tail to token ids.

    ``block_logits`` has one row per query and one column per completed compressed
    block. ``token_lengths`` is the corresponding visible main-KV token count.
    The fixed output layout matches :meth:`Qwen4ExpQSAIndexer.forward_causal`:
    ``token_budget`` scored-block token slots followed by ``ratio - 1`` tail
    slots, with ``-1`` padding.
    """
    if block_logits.dim() != 2 or not block_logits.is_floating_point():
        raise ValueError("block_logits must be a floating-point [rows, blocks] tensor")
    if token_lengths.dim() != 1 or int(token_lengths.numel()) != int(
        block_logits.shape[0]
    ):
        raise ValueError("token_lengths must have one value per logits row")
    if (
        token_lengths.dtype != torch.int32
        or token_lengths.device != block_logits.device
    ):
        raise ValueError("token_lengths must be int32 on the logits device")
    if compress_ratio <= 0 or token_budget <= 0 or token_budget % compress_ratio:
        raise ValueError(
            "token_budget must be positive and divisible by compress_ratio"
        )
    complete_blocks = token_lengths // compress_ratio
    invalid_lengths = (token_lengths < 0) | (
        complete_blocks > int(block_logits.shape[1])
    )
    if bool(torch.any(invalid_lengths).item()):
        if bool(torch.any(token_lengths < 0).item()):
            raise ValueError("token_lengths must be non-negative")
        raise ValueError(
            "block_logits do not cover every completed block in token_lengths"
        )

    rows = int(block_logits.shape[0])
    block_topk = token_budget // compress_ratio
    output = torch.full(
        (rows, token_budget + compress_ratio - 1),
        -1,
        dtype=torch.int32,
        device=block_logits.device,
    )
    selected_block_count = min(block_topk, int(block_logits.shape[1]))
    if selected_block_count:
        block_ids = torch.arange(
            int(block_logits.shape[1]), device=block_logits.device
        ).unsqueeze(0)
        masked_logits = torch.where(
            block_ids < complete_blocks.unsqueeze(1),
            block_logits,
            torch.full_like(block_logits, float("-inf")),
        )
        values, blocks = torch.topk(masked_logits, selected_block_count, dim=-1)
        valid = torch.isfinite(values)
        block_tokens = blocks.unsqueeze(-1) * compress_ratio + torch.arange(
            compress_ratio, device=block_logits.device
        )
        block_tokens = torch.where(valid.unsqueeze(-1), block_tokens, -1)
        output[:, : selected_block_count * compress_ratio] = block_tokens.flatten(1).to(
            torch.int32
        )

    tail_width = compress_ratio - 1
    if tail_width:
        tail_count = token_lengths % compress_ratio
        tail_offsets = torch.arange(tail_width, device=block_logits.device)
        tail = complete_blocks.unsqueeze(1) * compress_ratio + tail_offsets
        tail = torch.where(
            tail_offsets.unsqueeze(0) < tail_count.unsqueeze(1), tail, -1
        )
        output[:, token_budget:] = tail.to(torch.int32)
    return output


def _same_tensor_metadata(lhs: Any, rhs: Any) -> bool:
    if lhs is None or rhs is None:
        return lhs is rhs
    if not isinstance(lhs, torch.Tensor) or not isinstance(rhs, torch.Tensor):
        return lhs == rhs
    return (
        lhs.shape == rhs.shape
        and lhs.dtype == rhs.dtype
        and lhs.device == rhs.device
        and bool(torch.equal(lhs, rhs))
    )


def _transported_logical_positions(
    logical_positions: torch.Tensor, rope_config: Any
) -> torch.Tensor:
    """Expand scalar cache positions to the configured position ABI."""
    index_factor = int(rope_config.index_factor)
    return logical_positions.reshape(-1, 1).expand(-1, index_factor).reshape(-1)


def _typed_2d_pool(
    cache: LayerKVCache,
    *,
    tag: str,
    storage_dtype: torch.dtype,
    payload_dtype: torch.dtype,
    entries: int,
    width: int,
) -> torch.Tensor:
    if str(cache.tag) != tag:
        raise RuntimeError(
            f"QSA cache tag mismatch: expected {tag!r}, got {cache.tag!r}"
        )
    base = cache.kv_cache_base
    if base is None or base.dim() != 2 or not base.is_contiguous():
        raise RuntimeError(f"QSA cache {tag!r} must be a contiguous 2-D pool")
    if base.dtype != storage_dtype:
        raise RuntimeError(
            f"QSA cache {tag!r} has dtype {base.dtype}, expected {storage_dtype}"
        )
    if entries <= 0 or width <= 0:
        raise RuntimeError(
            f"QSA cache {tag!r} has invalid geometry entries={entries}, width={width}"
        )
    raw = base.view(torch.uint8)
    expected_bytes = (
        entries * width * torch.empty((), dtype=payload_dtype).element_size()
    )
    if int(raw.shape[1]) != expected_bytes:
        raise RuntimeError(
            f"QSA cache {tag!r} row has {raw.shape[1]} bytes, expected {expected_bytes}"
        )
    return raw.view(payload_dtype).view(int(raw.shape[0]), entries, width)


def _block_table(
    inputs: PyAttentionInputs,
    name: str,
    *,
    tag: str,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    table = getattr(inputs, name, None)
    if table is None or table.numel() == 0:
        raise RuntimeError(f"QSA cache {tag!r} requires {name}")
    if table.dim() != 2 or int(table.shape[0]) != batch_size:
        raise RuntimeError(
            f"QSA cache {tag!r} {name} must be [B, max_blocks], got {table.shape}"
        )
    if table.dtype != torch.int32:
        raise RuntimeError(f"QSA cache {tag!r} {name} must be int32, got {table.dtype}")
    if table.device != device:
        raise RuntimeError(
            f"QSA cache {tag!r} {name} is on {table.device}, expected {device}"
        )
    return table


def _validate_required_blocks(
    table: torch.Tensor,
    required_columns: torch.Tensor,
    *,
    pool_blocks: int,
    tag: str,
) -> None:
    """Validate every table entry a paged reader may observe."""
    if required_columns.dim() != 1 or int(required_columns.numel()) != int(
        table.shape[0]
    ):
        raise RuntimeError(f"QSA cache {tag!r} has invalid required-column geometry")
    required_columns = required_columns.to(device=table.device)
    columns = torch.arange(int(table.shape[1]), device=table.device).unsqueeze(0)
    required = columns < required_columns.unsqueeze(1)
    negative_columns = torch.any(required_columns < 0)
    uncovered_columns = torch.any(required_columns > int(table.shape[1]))
    invalid_required = torch.any(required & ((table <= 0) | (table >= pool_blocks)))
    invalid_physical = torch.any(table >= pool_blocks)
    # Synchronize once on the normal path.  Keep the specific diagnostics on
    # the rare error path, where the extra synchronizations do not affect TPOT.
    if bool(
        (
            negative_columns
            | uncovered_columns
            | invalid_required
            | invalid_physical
        ).item()
    ):
        if bool(negative_columns.item()):
            raise RuntimeError(f"QSA cache {tag!r} has a negative required-column count")
        if bool(uncovered_columns.item()):
            raise RuntimeError(
                f"QSA cache {tag!r} block table does not cover the target-verify tail"
            )
        if bool(invalid_required.item()):
            raise RuntimeError(
                f"QSA cache {tag!r} target-verify tail resolves to an unallocated "
                "or out-of-range physical block"
            )
        raise RuntimeError(f"QSA cache {tag!r} contains an out-of-range physical block")


@dataclass(frozen=True)
class Qwen4ExpQSARuntimeContext:
    main_cache: LayerKVCache
    main_inputs: PyAttentionInputs
    indexer_kv_cache: LayerKVCache
    indexer_kv_inputs: PyAttentionInputs
    indexer_state_cache: LayerKVCache
    indexer_state_inputs: PyAttentionInputs
    is_mtp_draft: bool = False
    _side_cache_undo: IndexerCacheUndo | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def _write_indexer_cache_transactional(self, *args, **kwargs) -> dict:
        if self._side_cache_undo is not None:
            raise RuntimeError("qwen4_exp QSA side-cache transaction is already active")
        result = write_indexer_cache(*args, **kwargs, capture_undo=True)
        undo = result.pop("undo", None)
        if not isinstance(undo, IndexerCacheUndo):
            raise RuntimeError("qwen4_exp QSA writer did not return an undo record")
        object.__setattr__(self, "_side_cache_undo", undo)
        return result

    def rollback_side_cache(self) -> None:
        """Restore one failed QSA side write and make rollback idempotent."""
        undo = self._side_cache_undo
        if undo is None:
            return
        try:
            restore_indexer_cache(undo)
            if undo.state_pool.is_cuda:
                torch.cuda.current_stream(undo.state_pool.device).synchronize()
        finally:
            object.__setattr__(self, "_side_cache_undo", None)

    def finalize_side_cache(self) -> None:
        """Discard undo state after the paired main attention succeeds."""
        if self._side_cache_undo is None:
            raise RuntimeError(
                "qwen4_exp QSA has no active side-cache transaction to finalize"
            )
        object.__setattr__(self, "_side_cache_undo", None)

    def _target_verify_enabled(self) -> bool:
        all_inputs = (
            self.main_inputs,
            self.indexer_kv_inputs,
            self.indexer_state_inputs,
        )
        target_modes = tuple(bool(inputs.is_target_verify) for inputs in all_inputs)
        if any(mode != target_modes[0] for mode in target_modes[1:]):
            raise RuntimeError(
                "qwen4_exp QSA cache regions disagree about target-verify mode"
            )
        return target_modes[0]

    def _validate_mode_and_metadata(
        self,
        *,
        expect_prefill: bool,
        allow_target_verify: bool = False,
        allow_draft_incremental: bool = False,
        allow_prefix_reuse: bool = False,
    ) -> list[int]:
        if str(self.indexer_kv_cache.tag) != INDEXER_KV_TAG:
            raise RuntimeError("qwen4_exp QSA received the wrong indexer KV cache")
        if str(self.indexer_state_cache.tag) != INDEXER_STATE_TAG:
            raise RuntimeError("qwen4_exp QSA received the wrong indexer state cache")
        if str(self.main_cache.tag) in (INDEXER_KV_TAG, INDEXER_STATE_TAG):
            raise RuntimeError("qwen4_exp QSA main cache resolves to a side-pool tag")

        all_inputs = (
            self.main_inputs,
            self.indexer_kv_inputs,
            self.indexer_state_inputs,
        )
        is_target_verify = self._target_verify_enabled()
        if is_target_verify and not allow_target_verify:
            raise RuntimeError(
                "qwen4_exp QSA target-verify requires the bounded multi-token "
                "paged path"
            )
        if allow_draft_incremental and not self.is_mtp_draft:
            raise RuntimeError(
                "qwen4_exp QSA incremental prefill requires an explicit MTP "
                "draft context"
            )
        for inputs in all_inputs:
            if bool(inputs.is_cuda_graph):
                raise RuntimeError("qwen4_exp QSA does not support CUDA Graph")
            if bool(inputs.is_s_padded):
                raise RuntimeError("qwen4_exp QSA does not support padded execution")
            if inputs.context_parallel_info is not None:
                raise RuntimeError("qwen4_exp QSA does not support context parallelism")
            if inputs.cache_store_inputs is not None:
                raise RuntimeError("qwen4_exp QSA indexer state does not support PD")
            prefixes = inputs.prefix_lengths
            if (
                not is_target_verify
                and not allow_draft_incremental
                and not allow_prefix_reuse
                and prefixes.numel()
                and bool(torch.any(prefixes != 0).item())
            ):
                raise RuntimeError("qwen4_exp QSA prefix reuse is not implemented")

        modes = tuple(bool(inputs.is_prefill) for inputs in all_inputs)
        if any(mode != expect_prefill for mode in modes):
            expected = "prefill" if expect_prefill else "ordinary decode"
            raise RuntimeError(
                f"qwen4_exp QSA expected {expected} in every cache region, "
                f"got is_prefill={modes}"
            )

        for name in (
            "input_lengths",
            "prefix_lengths",
            "sequence_lengths",
            "cu_seqlens_device",
            "cu_kv_seqlens_device",
            "combo_position_ids",
        ):
            main_value = getattr(self.main_inputs, name, None)
            if any(
                not _same_tensor_metadata(main_value, getattr(inputs, name, None))
                for inputs in all_inputs[1:]
            ):
                raise RuntimeError(
                    f"qwen4_exp QSA cache-region metadata {name!r} is inconsistent"
                )

        lengths = [int(value) for value in self.main_inputs.input_lengths.tolist()]
        if not lengths or any(length <= 0 for length in lengths):
            raise RuntimeError(f"qwen4_exp QSA has invalid input_lengths={lengths}")
        return lengths

    def _draft_incremental_prefill_geometry(
        self,
        *,
        indexer: Any,
        token_count: int,
        device: torch.device,
        rope_config: Any,
        draft: bool = True,
        draft_prefix_reuse: bool = False,
    ) -> dict[str, Any]:
        """Validate one incremental prefill build without side effects.

        ``draft=True`` is the post-rejection MTP draft commit (shifted positions,
        at most ``ratio`` new tokens).  ``draft=False`` is a page-aligned
        prefix-reuse target prefill: the request starts at its absolute prefix
        position, every reused prefix block is a complete compressed block held
        by the side pool, and the writer seals complete blocks exactly as the
        draft path does.
        """
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
        prefixes = prefixes.to(
            device=device, dtype=torch.int64, non_blocking=True
        ).contiguous()
        invalid_prefixes = prefixes <= 0 if draft else prefixes < 0
        if bool(torch.any(invalid_prefixes).item()):
            requirement = "nonzero" if draft else "non-negative"
            raise RuntimeError(
                f"qwen4_exp QSA {label} requires every prefix length to be "
                f"{requirement}"
            )
        sequence_lengths = self.main_inputs.sequence_lengths
        if sequence_lengths.numel() != 0:
            raise RuntimeError(
                "qwen4_exp QSA draft incremental prefill requires empty "
                "sequence_lengths"
            )

        cu_seqlens = self.main_inputs.cu_seqlens_device
        expected_cu = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
        expected_cu[1:] = torch.tensor(
            lengths, dtype=torch.int32, device=device
        ).cumsum(0)
        if (
            cu_seqlens.dim() != 1
            or cu_seqlens.dtype != torch.int32
            or cu_seqlens.device != device
            or not bool(torch.equal(cu_seqlens, expected_cu))
        ):
            raise RuntimeError(
                "qwen4_exp QSA draft incremental prefill cu_seqlens do not "
                "match input_lengths"
            )
        expected_cu_kv = torch.zeros_like(expected_cu)
        visible_ends = prefixes + torch.tensor(
            lengths, dtype=torch.int64, device=device
        )
        if bool(torch.any(visible_ends > torch.iinfo(torch.int32).max).item()):
            raise RuntimeError(
                "qwen4_exp QSA draft incremental prefill positions exceed int32"
            )
        if int(visible_ends.sum().item()) > torch.iinfo(torch.int32).max:
            raise RuntimeError(
                "qwen4_exp QSA draft incremental prefill cumulative KV lengths "
                "exceed int32"
            )
        expected_cu_kv[1:] = visible_ends.to(torch.int32).cumsum(0)
        cu_kv_seqlens = self.main_inputs.cu_kv_seqlens_device
        if (
            cu_kv_seqlens.dim() != 1
            or cu_kv_seqlens.dtype != torch.int32
            or cu_kv_seqlens.device != device
        ):
            raise RuntimeError(
                "qwen4_exp QSA draft incremental prefill cu_kv_seqlens do not "
                "provide device int32 [B + 1] metadata"
            )
        if not bool(torch.equal(cu_kv_seqlens, expected_cu_kv)):
            if not draft_prefix_reuse:
                raise RuntimeError(
                    "qwen4_exp QSA draft incremental prefill cu_kv_seqlens do not "
                    "match prefix+input lengths"
                )
            if int(cu_kv_seqlens.numel()) != batch_size + 1 or bool(
                torch.any(cu_kv_seqlens[1:] < cu_kv_seqlens[:-1]).item()
            ):
                raise RuntimeError(
                    "qwen4_exp QSA MTP draft prefix-reuse has invalid local "
                    "cu_kv_seqlens metadata"
                )

        logical_positions = torch.cat(
            [
                torch.arange(prefix, prefix + length, device=device)
                for prefix, length in zip(prefixes.tolist(), lengths)
            ]
        ).to(torch.int64)
        position_ids = self.main_inputs.combo_position_ids
        index_factor = int(rope_config.index_factor)
        if (
            position_ids is None
            or int(position_ids.numel()) != token_count * index_factor
        ):
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
        if not draft and bool((prefixes % kv_tokens_per_block != 0).any().item()):
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
        )

        # A completed group may read committed raw keys preceding the new rows.
        # Validate that source tail and every destination absolute state page.
        state_table_width = int(state_table.shape[1])
        for request_idx, (prefix, length) in enumerate(zip(prefixes.tolist(), lengths)):
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
                block_id = int(state_table[request_idx, logical_block].item())
                if block_id <= 0 or block_id >= int(state_pool.shape[0]):
                    raise RuntimeError(
                        "QSA cache 'indexer_state' draft incremental tail resolves "
                        "to an unallocated or out-of-range physical block"
                    )

        block_starts = logical_positions - logical_positions.remainder(ratio)
        try:
            rope_cos, rope_sin = build_qsa_rope(
                _transported_logical_positions(block_starts, rope_config),
                rope_config,
                token_count=token_count,
                dtype=torch.bfloat16,
                device=device,
                logical_positions=block_starts,
            )
        except ValueError as error:
            raise RuntimeError(
                f"qwen4_exp QSA draft incremental prefill {error}"
            ) from error

        return {
            "lengths": lengths,
            "batch_size": batch_size,
            "head_dim": head_dim,
            "head_num": head_num,
            "ratio": ratio,
            "token_budget": token_budget,
            "prefixes": prefixes,
            "cu_seqlens": expected_cu,
            "logical_positions": logical_positions,
            "current_cos": current_cos,
            "current_sin": current_sin,
            "visible_lengths": visible_lengths,
            "compressed_lengths": compressed_lengths,
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

    def _target_verify_geometry(
        self,
        *,
        indexer: Any,
        token_count: int,
        device: torch.device,
    ) -> dict[str, Any]:
        """Validate and materialise the bounded physical-tail overwrite plan.

        This method is intentionally side-effect free. The model calls it before
        either the QSA projection or the production main-cache writer, then the
        target selection path repeats it immediately before its side-pool write.
        """
        lengths = self._validate_mode_and_metadata(
            expect_prefill=True, allow_target_verify=True
        )
        if not self._target_verify_enabled():
            raise RuntimeError("qwen4_exp QSA expected a target-verify invocation")

        ratio = int(indexer.compress_ratio)
        head_dim = int(indexer.head_dim)
        head_num = int(indexer.n_heads)
        token_budget = int(indexer.token_budget)
        if min(ratio, head_dim, head_num) <= 0:
            raise RuntimeError(
                "qwen4_exp QSA target-verify has invalid indexer geometry"
            )
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
            raise RuntimeError(
                "qwen4_exp QSA target-verify prefix lengths must be int32"
            )
        if sequence_lengths.numel() != 0:
            raise RuntimeError(
                "qwen4_exp QSA target-verify requires an empty sequence_lengths tensor"
            )
        prefixes_device = prefixes.to(
            device=device, dtype=torch.int32, non_blocking=True
        ).contiguous()
        if bool(torch.any(prefixes_device < 0).item()):
            raise RuntimeError(
                "qwen4_exp QSA target-verify prefix lengths must be non-negative"
            )

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
            or not bool(torch.equal(cu_seqlens, expected_cu))
        ):
            raise RuntimeError(
                "qwen4_exp QSA target-verify cu_seqlens do not match gamma+1"
            )
        expected_cu_kv = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
        expected_cu_kv[1:] = (prefixes_device + query_len).cumsum(0)
        cu_kv_seqlens = self.main_inputs.cu_kv_seqlens_device
        if (
            cu_kv_seqlens.dtype != torch.int32
            or cu_kv_seqlens.device != device
            or not bool(torch.equal(cu_kv_seqlens, expected_cu_kv))
        ):
            raise RuntimeError(
                "qwen4_exp QSA target-verify cu_kv_seqlens do not match prefix+gamma+1"
            )

        positions = self.main_inputs.combo_position_ids
        if positions is None or int(positions.numel()) != token_count * 3:
            actual = 0 if positions is None else int(positions.numel())
            raise RuntimeError(
                "qwen4_exp QSA target-verify requires three MRoPE positions per "
                f"token, got {actual} for {token_count} tokens"
            )
        positions = positions.to(
            device=device, dtype=torch.int32, non_blocking=True
        ).reshape(batch_size, query_len, 3)
        offsets = torch.arange(query_len, dtype=torch.int32, device=device)
        expected_positions = prefixes_device[:, None, None] + offsets[None, :, None]
        expected_positions = expected_positions.expand(-1, -1, 3)
        if not bool(torch.equal(positions, expected_positions)):
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

        visible_lengths = prefixes_device[:, None] + offsets[None, :] + 1
        compressed_lengths = visible_lengths // ratio
        required_kv_columns = (
            compressed_lengths[:, -1] + kv_entries_per_block - 1
        ) // kv_entries_per_block
        _validate_required_blocks(
            kv_table,
            required_kv_columns,
            pool_blocks=int(kv_pool.shape[0]),
            tag=INDEXER_KV_TAG,
        )

        # STATE_RING payload offsets wrap, but the framework block table keeps
        # absolute logical-page columns (including sentinel holes). Validate the
        # small committed-tail + candidate window explicitly before any write.
        state_table_width = int(state_table.shape[1])
        for request_idx, prefix in enumerate(prefixes_device.tolist()):
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
                block_id = int(state_table[request_idx, logical_block].item())
                if block_id <= 0 or block_id >= int(state_pool.shape[0]):
                    raise RuntimeError(
                        "QSA cache 'indexer_state' target-verify tail resolves to "
                        "an unallocated or out-of-range physical block"
                    )

        return {
            "batch_size": batch_size,
            "query_len": query_len,
            "prefixes": prefixes_device,
            "cu_seqlens": expected_cu,
            "visible_lengths": visible_lengths,
            "compressed_lengths": compressed_lengths,
            "kv_tokens_per_block": kv_tokens_per_block,
            "kv_entries_per_block": kv_entries_per_block,
            "state_tokens_per_block": state_tokens_per_block,
            "kv_pool": kv_pool,
            "state_pool": state_pool,
            "kv_table": kv_table,
            "state_table": state_table,
        }

    def validate_before_projection(
        self,
        *,
        indexer: Any,
        token_count: int,
        device: torch.device,
    ) -> None:
        """Validate target geometry before projection and every cache mutation."""
        if self._target_verify_enabled():
            self._target_verify_geometry(
                indexer=indexer, token_count=token_count, device=device
            )

    def write_prefill_indexer_cache(
        self,
        raw_keys: torch.Tensor,
        *,
        indexer: Any,
        rope_config: Any,
    ) -> tuple[list[int], torch.Tensor, torch.Tensor, dict]:
        """Write one ragged prefill projection to both side pools.

        Page-aligned non-zero prefixes are supported: the request starts at its
        absolute prefix position and the block tables already map the reused
        prefix pages, so the writer seals the prefix's last partial block with
        raw keys retained from the earlier request.
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
            raise RuntimeError(
                "qwen4_exp QSA prefill requires int32 prefix_lengths [B]"
            )
        start_positions = prefix_tensor.to(
            device=raw_keys.device, dtype=torch.int64, non_blocking=True
        ).contiguous()
        if bool((start_positions < 0).any().item()):
            raise RuntimeError("qwen4_exp QSA prefill prefixes must be non-negative")
        if bool((start_positions % kv_tokens_per_block != 0).any().item()):
            raise RuntimeError(
                "qwen4_exp QSA prefix reuse requires prefixes aligned to the "
                f"indexer KV page size ({kv_tokens_per_block})"
            )
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
            raise RuntimeError(
                "qwen4_exp QSA side pools and raw keys must share a device"
            )

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
        expected_cu = torch.tensor(
            [0] + list(torch.tensor(lengths).cumsum(0).tolist()),
            dtype=torch.int32,
            device=raw_keys.device,
        )
        if not bool(torch.equal(cu_seqlens, expected_cu)):
            raise RuntimeError("qwen4_exp QSA cu_seqlens do not match input_lengths")

        position_ids = self.main_inputs.combo_position_ids
        index_factor = int(rope_config.index_factor)
        if (
            position_ids is None
            or int(position_ids.numel()) != token_count * index_factor
        ):
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
                prefix_len = int(start_positions[request_idx].item())
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
        head_num = int(plan["head_num"])
        head_dim = int(plan["head_dim"])
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
            raise RuntimeError(
                f"qwen4_exp QSA {label} q/raw keys must share " "a device"
            )
        if not q.is_cuda:
            raise RuntimeError(
                f"qwen4_exp QSA {label} paged scoring requires " "CUDA tensors"
            )

        # All metadata, tables, physical destinations and position contracts
        # above are validated before this first side-cache mutation.
        self._write_indexer_cache_transactional(
            raw_keys,
            plan["cu_seqlens"],
            plan["prefixes"],
            plan["rope_cos"],
            plan["rope_sin"],
            indexer.k_norm_gamma,
            plan["kv_pool"],
            plan["kv_table"],
            plan["state_pool"],
            plan["state_table"],
            ratio=int(plan["ratio"]),
            kv_tokens_per_block=int(plan["kv_tokens_per_block"]),
            state_tokens_per_block=int(plan["state_tokens_per_block"]),
            norm_eps=float(indexer.norm_eps),
            rope_is_token_aligned=True,
        )

        rotated_q = apply_partial_rope(
            q,
            plan["current_cos"].unsqueeze(1),
            plan["current_sin"].unsqueeze(1),
        )
        score_weights = torch.full(
            (token_count, head_num),
            1.0 / math.sqrt(head_dim),
            dtype=torch.float32,
            device=q.device,
        )
        from rtp_llm.models_py.modules.qwen4_exp.indexer_paged_score import (
            qsa_paged_indexer_score,
        )

        selections = []
        row_offset = 0
        for request_idx, length in enumerate(plan["lengths"]):
            row_end = row_offset + length
            compressed_lengths = plan["compressed_lengths"][
                request_idx : request_idx + 1, :length
            ].contiguous()
            block_logits = qsa_paged_indexer_score(
                rotated_q[row_offset:row_end].unsqueeze(0).contiguous(),
                score_weights[row_offset:row_end],
                plan["kv_pool"].flatten(0, 1),
                plan["kv_table"][request_idx : request_idx + 1],
                compressed_lengths,
                block_size=int(plan["kv_entries_per_block"]),
                max_ctx_len=(
                    int(compressed_lengths.max().item())
                    if int(compressed_lengths.numel())
                    else 0
                ),
            )
            visible_lengths = plan["visible_lengths"][request_idx, :length].to(
                torch.int32
            )
            selections.append(
                select_qsa_paged_tokens(
                    block_logits,
                    visible_lengths,
                    compress_ratio=int(plan["ratio"]),
                    token_budget=int(plan["token_budget"]),
                )
            )
            row_offset = row_end
        return torch.cat(selections, dim=0)

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
        input_lengths = self._validate_mode_and_metadata(expect_prefill=False)
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

        sequence_lengths = sequence_lengths.to(
            device=q.device, dtype=torch.int32, non_blocking=True
        ).contiguous()
        if bool(torch.any(sequence_lengths < 0).item()):
            raise RuntimeError("qwen4_exp QSA sequence_lengths must be non-negative")
        position_ids = self.main_inputs.combo_position_ids
        index_factor = int(rope_config.index_factor)
        if (
            position_ids is None
            or int(position_ids.numel()) != batch_size * index_factor
        ):
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
        try:
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

        visible_token_lengths = sequence_lengths + 1
        compressed_lengths = visible_token_lengths // ratio
        compressed_capacity = int(kv_table.shape[1]) * kv_entries_per_block
        if bool(torch.any(compressed_lengths > compressed_capacity).item()):
            raise RuntimeError(
                "qwen4_exp QSA compressed decode length exceeds its tag-local "
                "indexer KV block table"
            )
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
        block_starts = sequence_lengths.to(torch.int64)
        block_starts = block_starts - block_starts.remainder(ratio)
        rope_cos, rope_sin = build_qsa_rope(
            _transported_logical_positions(block_starts, rope_config),
            rope_config,
            token_count=batch_size,
            dtype=raw_keys.dtype,
            device=raw_keys.device,
            logical_positions=block_starts,
        )
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
        score_weights = torch.full(
            (batch_size, head_num),
            1.0 / math.sqrt(head_dim),
            dtype=torch.float32,
            device=q.device,
        )
        from rtp_llm.models_py.modules.qwen4_exp.indexer_paged_score import (
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
                int(compressed_lengths.max().item())
                if int(compressed_lengths.numel())
                else 0
            ),
        )
        return select_qsa_paged_tokens(
            block_logits,
            visible_token_lengths,
            compress_ratio=ratio,
            token_budget=token_budget,
        )

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
        plan = self._target_verify_geometry(
            indexer=indexer, token_count=token_count, device=raw_keys.device
        )
        batch_size = int(plan["batch_size"])
        query_len = int(plan["query_len"])
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
            raise RuntimeError(
                "qwen4_exp QSA target-verify q/raw keys must share a device"
            )
        if not q.is_cuda:
            raise RuntimeError(
                "qwen4_exp QSA target-verify paged scoring requires CUDA tensors"
            )

        query_positions = plan["visible_lengths"] - 1
        flat_query_positions = query_positions.reshape(-1).to(torch.int64)
        block_starts = flat_query_positions - flat_query_positions.remainder(ratio)
        rope_cos, rope_sin = build_qsa_rope(
            _transported_logical_positions(block_starts, rope_config),
            rope_config,
            token_count=token_count,
            dtype=raw_keys.dtype,
            device=raw_keys.device,
            logical_positions=block_starts,
        )
        current_cos, current_sin = build_qsa_rope(
            _transported_logical_positions(flat_query_positions, rope_config),
            rope_config,
            token_count=token_count,
            dtype=raw_keys.dtype,
            device=raw_keys.device,
            logical_positions=flat_query_positions,
        )
        self._write_indexer_cache_transactional(
            raw_keys,
            plan["cu_seqlens"],
            plan["prefixes"].to(torch.int64),
            rope_cos,
            rope_sin,
            indexer.k_norm_gamma,
            plan["kv_pool"],
            plan["kv_table"],
            plan["state_pool"],
            plan["state_table"],
            ratio=ratio,
            kv_tokens_per_block=int(plan["kv_tokens_per_block"]),
            state_tokens_per_block=int(plan["state_tokens_per_block"]),
            norm_eps=float(indexer.norm_eps),
            rope_is_token_aligned=True,
        )

        q = q.view(batch_size, query_len, head_num, head_dim)
        rotated_q = apply_partial_rope(
            q,
            current_cos.view(batch_size, query_len, 1, -1),
            current_sin.view(batch_size, query_len, 1, -1),
        )
        score_weights = torch.full(
            (token_count, head_num),
            1.0 / math.sqrt(head_dim),
            dtype=torch.float32,
            device=q.device,
        )
        from rtp_llm.models_py.modules.qwen4_exp.indexer_paged_score import (
            qsa_paged_indexer_score,
        )

        compressed_lengths = plan["compressed_lengths"]
        block_logits = qsa_paged_indexer_score(
            rotated_q.contiguous(),
            score_weights,
            plan["kv_pool"].flatten(0, 1),
            plan["kv_table"],
            compressed_lengths,
            block_size=int(plan["kv_entries_per_block"]),
            max_ctx_len=(
                int(compressed_lengths.max().item())
                if int(compressed_lengths.numel())
                else 0
            ),
        )
        return select_qsa_paged_tokens(
            block_logits,
            plan["visible_lengths"].reshape(-1),
            compress_ratio=ratio,
            token_budget=token_budget,
        )
