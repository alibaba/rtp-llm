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
before they can become visible. CP, PD and padding remain unsupported; CUDA
Graph handles exact-batch decode and target verification while dynamic draft
incremental prefill remains eager. Prefix-cache reuse is page-aligned.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any

import torch

from rtp_llm.models.qwen4_exp.qwen4_exp_kv_cache import (
    INDEXER_KV_TAG,
    INDEXER_STATE_TAG,
)
from rtp_llm.models_py.modules.qwen4_exp.indexer import (
    apply_partial_rope,
    build_base_rope,
    build_qsa_rope,
    build_qsa_rope_from_logical_positions,
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
    validate_lengths: bool = True,
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
    if validate_lengths:
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
    if lhs is rhs:
        return True
    if lhs is None or rhs is None:
        return lhs is rhs
    if not isinstance(lhs, torch.Tensor) or not isinstance(rhs, torch.Tensor):
        return lhs == rhs
    if lhs.shape != rhs.shape or lhs.dtype != rhs.dtype or lhs.device != rhs.device:
        return False
    # Tagged attention inputs normally share the same tensor storage. Avoid a
    # CUDA equality kernel and host synchronization for that exact alias.
    if lhs.stride() == rhs.stride() and lhs.data_ptr() == rhs.data_ptr():
        return True
    return bool(torch.equal(lhs, rhs))


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
    # The bounded Triton path has one launch and one verdict synchronization.
    # The environment switch permits an immediate Torch fallback if needed.
    fused_validation = os.environ.get(
        "RTP_LLM_QWEN4_FUSED_CACHE_VALIDATE", "1"
    ).strip().lower() not in ("0", "false", "off", "no")
    if fused_validation and table.is_cuda and torch.version.hip is None:
        from rtp_llm.models_py.modules.qwen4_exp.qsa_validation_triton import (
            is_supported,
            required_blocks_are_valid,
        )

        if is_supported(table, required_columns) and required_blocks_are_valid(
            table, required_columns, pool_blocks
        ):
            return
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
            negative_columns | uncovered_columns | invalid_required | invalid_physical
        ).item()
    ):
        if bool(negative_columns.item()):
            raise RuntimeError(
                f"QSA cache {tag!r} has a negative required-column count"
            )
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
    _target_verify_plan: dict[str, Any] | None = field(
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
        allow_graph_decode: bool = False,
        allow_graph_target: bool = False,
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
            graph_decode = allow_graph_decode and not expect_prefill
            graph_target = allow_graph_target and expect_prefill and is_target_verify
            graph_mode = graph_decode or graph_target
            if bool(inputs.is_cuda_graph) and not graph_mode:
                raise RuntimeError("qwen4_exp QSA does not support CUDA Graph")
            if bool(inputs.is_s_padded) and not graph_mode:
                raise RuntimeError("qwen4_exp QSA does not support padded execution")
            if graph_mode and not bool(inputs.is_cuda_graph):
                raise RuntimeError("qwen4_exp QSA Graph cache modes disagree")
            if graph_mode and not bool(
                getattr(inputs, "is_exact_cuda_graph_batch", False)
            ):
                raise RuntimeError("qwen4_exp QSA Graph requires an exact batch graph")
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

        prefix_values = prefixes.tolist()
        logical_positions = torch.cat(
            [
                torch.arange(prefix, prefix + length, device=device)
                for prefix, length in zip(prefix_values, lengths)
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
                block_id = int(state_table[request_idx, logical_block].item())
                if block_id <= 0 or block_id >= int(state_pool.shape[0]):
                    raise RuntimeError(
                        "QSA cache 'indexer_state' draft incremental tail resolves "
                        "to an unallocated or out-of-range physical block"
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
        # GraphRunner retains a decode-length mirror for capture bookkeeping;
        # target scoring addresses history through prefix_lengths.
        if sequence_lengths.numel() != 0 and not graph_target:
            raise RuntimeError(
                "qwen4_exp QSA target-verify requires an empty sequence_lengths tensor"
            )
        if graph_target and prefixes.device.type != "cpu":
            raise RuntimeError("qwen4_exp QSA target Graph requires host prefixes")
        if graph_target and bool(torch.any(prefixes < 0).item()):
            raise RuntimeError(
                "qwen4_exp QSA target-verify prefix lengths must be non-negative"
            )
        prefixes_device = prefixes.to(
            device=device, dtype=torch.int32, non_blocking=True
        ).contiguous()
        if not graph_target and bool(torch.any(prefixes_device < 0).item()):
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
            or tuple(cu_seqlens.shape) != (batch_size + 1,)
            or (not graph_target and not bool(torch.equal(cu_seqlens, expected_cu)))
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
            or tuple(cu_kv_seqlens.shape) != (batch_size + 1,)
            or (
                not graph_target
                and not bool(torch.equal(cu_kv_seqlens, expected_cu_kv))
            )
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
        if not graph_target and not bool(torch.equal(positions, expected_positions)):
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
        max_ctx_len = (
            (int(prefixes.max().item()) + query_len) // ratio
            if prefixes.device.type == "cpu"
            else int(compressed_lengths.max().item())
        )
        required_kv_columns = (
            compressed_lengths[:, -1] + kv_entries_per_block - 1
        ) // kv_entries_per_block
        if graph_target:
            required_host = (
                max_ctx_len + kv_entries_per_block - 1
            ) // kv_entries_per_block
            if required_host > int(kv_table.shape[1]):
                raise RuntimeError(
                    "QSA cache 'indexer_kv' block table does not cover target verification"
                )
        else:
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
        for request_idx, prefix in enumerate(
            prefixes.tolist() if graph_target else prefixes_device.tolist()
        ):
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

        return {
            "batch_size": batch_size,
            "query_len": query_len,
            "prefixes": prefixes_device,
            "cu_seqlens": expected_cu,
            "visible_lengths": visible_lengths,
            "compressed_lengths": compressed_lengths,
            "max_ctx_len": max_ctx_len,
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
            plan = self._target_verify_geometry(
                indexer=indexer, token_count=token_count, device=device
            )
            plan["_indexer_id"] = id(indexer)
            # The same context immediately consumes this plan after its Q/K
            # projection. Revalidating GPU lengths and block IDs per layer
            # would add host synchronization without seeing new metadata.
            object.__setattr__(self, "_target_verify_plan", plan)

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

        if (
            os.environ.get("RTP_LLM_QWEN4_FUSED_INDEXER_PREFILL", "0") == "1"
            and ratio == 4
            and int(self.indexer_state_cache.seq_size_per_block) == kv_tokens_per_block
            and raw_keys.is_cuda
            and not bool(torch.any(start_positions != 0).item())
        ):
            from rtp_llm.models_py.modules.qwen4_exp.indexer_prefill_triton import (
                write_zero_prefix_prefill,
            )

            if self._side_cache_undo is not None:
                raise RuntimeError(
                    "qwen4_exp QSA side-cache transaction is already active"
                )
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
            )
            if fused_result is not None:
                object.__setattr__(self, "_side_cache_undo", fused_result.pop("undo"))
                return lengths, rope_cos, rope_sin, fused_result

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
        self, q: torch.Tensor, raw_keys: torch.Tensor, *, indexer: Any, rope_config: Any
    ) -> torch.Tensor:
        """Capture a fixed four-lane draft plan; replay validates live host mirrors."""
        from rtp_llm.models_py.modules.qwen4_exp.draft_prefill_graph import (
            draft_graph_rows,
        )
        from rtp_llm.models_py.modules.qwen4_exp.indexer_decode_triton import (
            write_draft_window_with_undo_,
        )
        from rtp_llm.models_py.modules.qwen4_exp.indexer_paged_score import (
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
            raise RuntimeError(
                "QSA draft graph requires ratio 4 and head dimension 128"
            )
        if (
            not q.is_cuda
            or q.dtype != torch.bfloat16
            or raw_keys.dtype != q.dtype
            or raw_keys.device != q.device
        ):
            raise RuntimeError("QSA draft graph requires device-local BF16 projections")
        tokens, heads, dim = q.shape
        batch = inputs.input_lengths_device.numel()
        if tuple(raw_keys.shape) != (tokens, dim) or tokens != batch * 4:
            raise RuntimeError(
                "QSA draft graph projection must use four-token batch capacity"
            )
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
                if not _same_tensor_metadata(
                    getattr(inputs, name), getattr(tagged, name)
                ):
                    raise RuntimeError("QSA draft graph cache-region metadata disagree")
        if (
            self.indexer_kv_cache.seq_size_per_block != 128
            or self.indexer_state_cache.seq_size_per_block != 128
        ):
            raise RuntimeError("QSA draft graph requires 128-token side pages")
        kv_pool = _typed_2d_pool(
            self.indexer_kv_cache,
            tag=INDEXER_KV_TAG,
            storage_dtype=torch.uint8,
            payload_dtype=torch.bfloat16,
            entries=32,
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
        sources, inverse, valid, positions, visible = draft_graph_rows(inputs, tokens)
        cos, sin = build_qsa_rope_from_logical_positions(
            positions, rope_config, token_count=tokens, dtype=q.dtype, device=q.device
        )
        block_starts = positions - positions.remainder(4)
        block_cos, block_sin = build_qsa_rope_from_logical_positions(
            block_starts,
            rope_config,
            token_count=tokens,
            dtype=q.dtype,
            device=q.device,
        )
        if self._side_cache_undo is not None:
            raise RuntimeError(
                "QSA draft graph side-cache transaction is already active"
            )
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
        rotated = apply_partial_rope(q, cos.unsqueeze(1), sin.unsqueeze(1))
        padded = rotated.index_select(0, sources).reshape(batch, 4, heads, dim)
        weights = torch.full(
            (batch * 4, heads),
            1.0 / math.sqrt(dim),
            device=q.device,
            dtype=torch.float32,
        )
        logits = qsa_paged_indexer_score(
            padded.contiguous(),
            weights,
            kv_pool.flatten(0, 1),
            kv_table,
            visible // 4,
            block_size=32,
            max_ctx_len=int(kv_table.shape[1]) * 32,
            validate_block_table=False,
        )
        selected = select_qsa_paged_tokens(
            logits,
            visible.reshape(-1),
            compress_ratio=4,
            token_budget=int(indexer.token_budget),
            validate_lengths=False,
        )
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
        draft_writer_enabled = os.environ.get(
            "RTP_LLM_QWEN4_DRAFT_WINDOW_WRITER", "1"
        ).strip().lower() in ("1", "true", "yes", "on")
        triton_draft_window = (
            draft
            and max(plan["lengths"]) <= 4
            and int(plan["ratio"]) == 4
            and int(plan["head_dim"]) == 128
            and int(plan["kv_tokens_per_block"]) == 128
            and int(plan["state_tokens_per_block"]) == 128
            and draft_writer_enabled
        )
        if triton_draft_window:
            from rtp_llm.models_py.modules.qwen4_exp.indexer_decode_triton import (
                write_draft_window_with_undo_,
            )

            if self._side_cache_undo is not None:
                raise RuntimeError(
                    "qwen4_exp QSA side-cache transaction is already active"
                )
            undo = write_draft_window_with_undo_(
                raw_keys,
                plan["cu_seqlens"],
                plan["prefixes"].to(torch.int32).contiguous(),
                self.main_inputs.input_lengths.to(
                    device=raw_keys.device, dtype=torch.int32
                ).contiguous(),
                plan["rope_cos"],
                plan["rope_sin"],
                indexer.k_norm_gamma,
                plan["kv_pool"],
                plan["kv_table"],
                plan["state_pool"],
                plan["state_table"],
                norm_eps=float(indexer.norm_eps),
            )
            object.__setattr__(self, "_side_cache_undo", undo)
        else:
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

        # The draft has at most one compression group's worth of rows per
        # request. Pad only that short dimension so all requests share one
        # paged-score launch and one top-k selection. Invalid rows have zero
        # visible context and are removed before returning packed token IDs.
        lengths = plan["lengths"]
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
        compressed_lengths = plan["compressed_lengths"]
        block_logits = qsa_paged_indexer_score(
            padded_q,
            padded_weights,
            plan["kv_pool"].flatten(0, 1),
            plan["kv_table"],
            compressed_lengths,
            block_size=int(plan["kv_entries_per_block"]),
            max_ctx_len=int(plan["max_ctx_len"]),
            validate_block_table=False,
        )
        selected = select_qsa_paged_tokens(
            block_logits,
            plan["visible_lengths"].reshape(-1).to(torch.int32),
            compress_ratio=int(plan["ratio"]),
            token_budget=int(plan["token_budget"]),
            validate_lengths=False,
        )
        return selected.index_select(
            0, torch.tensor(valid_rows, dtype=torch.int64, device=q.device)
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
        sequence_lengths = sequence_lengths.to(
            device=q.device, dtype=torch.int32, non_blocking=True
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
            if graph_decode and self.is_mtp_draft:
                if not is_qsa_rope_style(rope_config, "Base"):
                    raise RuntimeError(
                        "qwen4_exp draft Graph requires text-only Base RoPE"
                    )
                # MTP's text position generator repeats one scalar across its
                # axes. Eager checks equality; Graph capture cannot synchronize.
                current_cos, current_sin = build_base_rope(
                    position_ids,
                    rope_config,
                    token_count=batch_size,
                    dtype=raw_keys.dtype,
                    device=raw_keys.device,
                    validate_position_axes=False,
                )
            elif graph_decode:
                # Exact-batch Graph capture uses the engine's canonical text
                # positions. The C++ graph owner refreshes lengths before replay.
                current_cos, current_sin = build_qsa_rope_from_logical_positions(
                    sequence_lengths,
                    rope_config,
                    token_count=batch_size,
                    dtype=raw_keys.dtype,
                    device=raw_keys.device,
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

        visible_token_lengths = sequence_lengths + 1
        compressed_lengths = visible_token_lengths // ratio
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
        required_kv_columns = (
            compressed_lengths + kv_entries_per_block - 1
        ) // kv_entries_per_block
        if not graph_decode:
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
        rope_cos, rope_sin = build_qsa_rope_from_logical_positions(
            block_starts,
            rope_config,
            token_count=batch_size,
            dtype=raw_keys.dtype,
            device=raw_keys.device,
        )
        fused_writer = os.environ.get(
            "RTP_LLM_QWEN4_TRITON_DECODE_WRITER", "1"
        ).strip().lower() in ("1", "true", "yes", "on")
        if fused_writer:
            from rtp_llm.models_py.modules.qwen4_exp.indexer_decode_triton import (
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
                raise RuntimeError(
                    "qwen4_exp QSA side-cache transaction is already active"
                )
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
            cu_seqlens = torch.arange(
                batch_size + 1, dtype=torch.int32, device=q.device
            )
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
        plan = self._target_verify_plan
        object.__setattr__(self, "_target_verify_plan", None)
        if plan is None:
            plan = self._target_verify_geometry(
                indexer=indexer, token_count=token_count, device=raw_keys.device
            )
        elif (
            int(plan["batch_size"]) * int(plan["query_len"]) != token_count
            or plan["kv_pool"].device != raw_keys.device
            or plan["_indexer_id"] != id(indexer)
        ):
            raise RuntimeError("qwen4_exp QSA target verification plan changed")
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
        rope_cos, rope_sin = build_qsa_rope_from_logical_positions(
            block_starts,
            rope_config,
            token_count=token_count,
            dtype=raw_keys.dtype,
            device=raw_keys.device,
        )
        current_cos, current_sin = build_qsa_rope_from_logical_positions(
            flat_query_positions,
            rope_config,
            token_count=token_count,
            dtype=raw_keys.dtype,
            device=raw_keys.device,
        )
        target_writer = os.environ.get(
            "RTP_LLM_QWEN4_TRITON_TARGET_WRITER", "1"
        ).strip().lower() in ("1", "true", "yes", "on")
        target_writer = target_writer and (
            query_len == ratio == 4
            and head_dim == 128
            and int(plan["kv_tokens_per_block"]) == 128
            and int(plan["state_tokens_per_block"]) == 128
        )
        if bool(self.main_inputs.is_cuda_graph) and not target_writer:
            raise RuntimeError("qwen4_exp QSA target Graph requires Triton writer")
        if target_writer:
            from rtp_llm.models_py.modules.qwen4_exp.indexer_decode_triton import (
                write_target_window_with_undo_,
            )

            if self._side_cache_undo is not None:
                raise RuntimeError(
                    "qwen4_exp QSA side-cache transaction is already active"
                )
            undo = write_target_window_with_undo_(
                raw_keys,
                plan["prefixes"],
                rope_cos,
                rope_sin,
                indexer.k_norm_gamma,
                plan["kv_pool"],
                plan["kv_table"],
                plan["state_pool"],
                plan["state_table"],
                norm_eps=float(indexer.norm_eps),
            )
            object.__setattr__(self, "_side_cache_undo", undo)
        else:
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
            validate_block_table=False,
            max_ctx_len=int(plan["max_ctx_len"]),
        )
        return select_qsa_paged_tokens(
            block_logits,
            plan["visible_lengths"].reshape(-1),
            compress_ratio=ratio,
            token_budget=token_budget,
            validate_lengths=False,
        )
