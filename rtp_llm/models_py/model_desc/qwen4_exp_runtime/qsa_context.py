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
before they can become visible. CP and padding remain unsupported; PD requires
an explicit committed side-cache publication plan. CUDA
Graph handles exact-batch decode and target verification, plus bounded packed
MTP-draft prefill with device metadata. Prefix-cache reuse is page-aligned.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from rtp_llm.models.qwen4_exp.qwen4_exp_kv_cache import (
    INDEXER_KV_TAG,
    INDEXER_STATE_TAG,
)
from rtp_llm.models_py.modules.qwen4_exp.qsa.indexer_compressor import (
    IndexerCacheUndo,
    restore_indexer_cache,
    write_indexer_cache,
)
from rtp_llm.ops.compute_ops import LayerKVCache, PyAttentionInputs

from . import qsa_decode, qsa_geometry, qsa_prefill, qsa_target
from .qsa_cache_views import _same_tensor_metadata
from .qsa_plans import IncrementalPrefillPlan, TargetVerifyPlan


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
    _target_verify_plan: TargetVerifyPlan | None = field(
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
            prefixes = inputs.prefix_lengths
            if (
                not is_target_verify
                and not allow_draft_incremental
                and not allow_prefix_reuse
                and prefixes.numel()
                and bool(torch.any(prefixes != 0).item())
            ):
                raise RuntimeError("qwen4_exp QSA prefix reuse is not implemented")

        from rtp_llm.models_py.model_desc.qwen4_exp_runtime.pd_publication import (
            validate_pd_cache_inputs,
        )

        validate_pd_cache_inputs(all_inputs)
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
    ) -> IncrementalPrefillPlan:
        return qsa_geometry._draft_incremental_prefill_geometry(
            self,
            indexer=indexer,
            token_count=token_count,
            device=device,
            rope_config=rope_config,
            draft=draft,
            draft_prefix_reuse=draft_prefix_reuse,
        )

    def _target_verify_geometry(
        self,
        *,
        indexer: Any,
        token_count: int,
        device: torch.device,
    ) -> TargetVerifyPlan:
        return qsa_geometry._target_verify_geometry(
            self, indexer=indexer, token_count=token_count, device=device
        )

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
        return qsa_prefill.write_prefill_indexer_cache(
            self, raw_keys, indexer=indexer, rope_config=rope_config
        )

    def select_draft_graph_tokens(
        self, q: torch.Tensor, raw_keys: torch.Tensor, *, indexer: Any, rope_config: Any
    ) -> torch.Tensor:
        return qsa_prefill.select_draft_graph_tokens(
            self, q, raw_keys, indexer=indexer, rope_config=rope_config
        )

    def select_draft_incremental_prefill_tokens(
        self,
        q: torch.Tensor,
        raw_keys: torch.Tensor,
        *,
        indexer: Any,
        rope_config: Any,
    ) -> torch.Tensor:
        return qsa_prefill.select_draft_incremental_prefill_tokens(
            self, q, raw_keys, indexer=indexer, rope_config=rope_config
        )

    def select_prefix_reuse_prefill_tokens(
        self,
        q: torch.Tensor,
        raw_keys: torch.Tensor,
        *,
        indexer: Any,
        rope_config: Any,
    ) -> torch.Tensor:
        return qsa_prefill.select_prefix_reuse_prefill_tokens(
            self, q, raw_keys, indexer=indexer, rope_config=rope_config
        )

    def select_draft_prefix_reuse_prefill_tokens(
        self,
        q: torch.Tensor,
        raw_keys: torch.Tensor,
        *,
        indexer: Any,
        rope_config: Any,
    ) -> torch.Tensor:
        return qsa_prefill.select_draft_prefix_reuse_prefill_tokens(
            self, q, raw_keys, indexer=indexer, rope_config=rope_config
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
        return qsa_prefill._incremental_prefill_selection(
            self,
            q,
            raw_keys,
            indexer=indexer,
            rope_config=rope_config,
            draft=draft,
            draft_prefix_reuse=draft_prefix_reuse,
        )

    def select_decode_tokens(
        self,
        q: torch.Tensor,
        raw_keys: torch.Tensor,
        *,
        indexer: Any,
        rope_config: Any,
    ) -> torch.Tensor:
        return qsa_decode.select_decode_tokens(
            self, q, raw_keys, indexer=indexer, rope_config=rope_config
        )

    def select_target_verify_tokens(
        self,
        q: torch.Tensor,
        raw_keys: torch.Tensor,
        *,
        indexer: Any,
        rope_config: Any,
        raw_query: bool = False,
    ) -> torch.Tensor:
        return qsa_target.select_target_verify_tokens(
            self,
            q,
            raw_keys,
            indexer=indexer,
            rope_config=rope_config,
            raw_query=raw_query,
        )
