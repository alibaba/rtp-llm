"""Model-wide PLE staging, prepare consensus boundary, undo and finalization.

The C++ executor owns TP consensus. Captured stages retain their addresses in
``graph_registry``; releasing an active commit only releases its undo record.
"""

from collections import deque
from collections.abc import Callable, Mapping
import os
import threading
from typing import Optional

import torch

from rtp_llm.models.qwen4_exp.qwen4_exp_kv_cache import PLE_NGRAM_CTX_TAG, PLE_STATE_TAG
from rtp_llm.models_py.model_desc.qwen4_exp_runtime.qsa_step_plan import (
    host_input_mirror,
)
from rtp_llm.models_py.modules.qwen4_exp.ple.ple import Qwen4ExpPLELayer
from rtp_llm.ops.compute_ops import PyAttentionInputs

from .ple_cache_views import PLECacheViews
from .ple_graph_state import PLEGraphRegistry
from .ple_state import (
    _PLEBoundaryWrite,
    _PLEPreparedWrite,
    _PLETargetLayerStage,
    _PLETargetTransaction,
)


def async_ple_commit_disabled() -> bool:
    """Emergency kill-switch to revert PLE target commit to synchronous verdict."""
    return os.environ.get("RTP_QWEN4_DISABLE_ASYNC_PLE_COMMIT", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


class PLETransactionCoordinator:
    def __init__(self, layers: Mapping, fault_hook: Callable[[str], None]):
        self.ple_layers = layers
        self.fault_hook = fault_hook
        self.active_transaction: Optional[_PLETargetTransaction] = None
        self.graph_registry = PLEGraphRegistry()
        self._pending_verdicts: deque = deque()
        self._pending_verdicts_lock = threading.Lock()

    def close(self) -> None:
        if self.active_transaction is not None:
            if self.active_transaction.deferred_resolution is not None:
                self.active_transaction.deferred_resolution.release()
                self.active_transaction.deferred_resolution = None
            self._drain_ple_transaction(self.active_transaction)
        for transaction in self.graph_registry.captured.values():
            if transaction.deferred_resolution is not None:
                transaction.deferred_resolution.release()
                transaction.deferred_resolution = None
            self._drain_ple_transaction(transaction)
        with self._pending_verdicts_lock:
            pending = list(self._pending_verdicts)
            self._pending_verdicts.clear()
        for resolution in pending:
            resolution.release()
        self.active_transaction = None
        self.graph_registry.clear()

    _physical_blocks = staticmethod(PLECacheViews._physical_blocks)
    _commit_physical_blocks = staticmethod(PLECacheViews._commit_physical_blocks)

    def _maybe_inject_speculative_fault(self, phase: str) -> None:
        self.fault_hook(phase)

    def _stage_ple_target_verify(
        self,
        *,
        layer_idx: int,
        ple: Qwen4ExpPLELayer,
        hyper_states: torch.Tensor,
        ids: torch.Tensor,
        state_inputs: PyAttentionInputs,
        ctx_inputs: PyAttentionInputs,
        state_pool: torch.Tensor,
        ctx_pool: torch.Tensor,
        page_size: int,
        embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute all target rows while leaving the persistent PLE pools intact."""
        if not bool(state_inputs.is_prefill):
            raise RuntimeError(
                "qwen4_exp PLE target verification must use context-style prefill"
            )
        lengths = state_inputs.input_lengths
        prefixes = state_inputs.prefix_lengths
        sequence_lengths = state_inputs.sequence_lengths
        graph_capture = bool(getattr(state_inputs, "is_cuda_graph", False))
        if (
            lengths.dim() != 1
            or lengths.dtype not in (torch.int32, torch.int64)
            or not int(lengths.numel())
        ):
            raise RuntimeError(
                "qwen4_exp PLE target verify requires integer input_lengths [B]"
            )
        batch_size = int(lengths.numel())
        query_len = int(lengths[0].item())
        if query_len <= 0 or bool(torch.any(lengths != query_len).item()):
            raise RuntimeError(
                "qwen4_exp PLE target verify requires one uniform gamma+1 width"
            )
        # GraphRunner keeps a host sequence-length mirror even for context-style
        # target verification.  The target path uses prefix_lengths instead.
        if sequence_lengths.numel() != 0 and not graph_capture:
            raise RuntimeError(
                "qwen4_exp PLE target verify requires empty sequence_lengths"
            )
        if (
            prefixes.dim() != 1
            or prefixes.dtype != torch.int32
            or int(prefixes.numel()) != batch_size
        ):
            raise RuntimeError(
                "qwen4_exp PLE target verify requires int32 prefix_lengths [B]"
            )
        if (
            graph_capture
            and self.active_transaction is not None
            and self.active_transaction.prefixes.device == hyper_states.device
            and self.active_transaction.prefixes.dtype == torch.long
            and int(self.active_transaction.prefixes.numel()) == batch_size
        ):
            prefixes_device = self.active_transaction.prefixes
        else:
            prefixes_src = prefixes
            if graph_capture:
                device_candidate = getattr(state_inputs, "prefix_lengths_device", None)
                if (
                    isinstance(device_candidate, torch.Tensor)
                    and device_candidate.is_cuda
                    and device_candidate.device == hyper_states.device
                    and device_candidate.dim() == 1
                    and device_candidate.dtype == torch.int32
                    and int(device_candidate.numel()) == batch_size
                ):
                    prefixes_src = device_candidate
            prefixes_device = prefixes_src.to(
                device=hyper_states.device, dtype=torch.long, non_blocking=True
            ).contiguous()
        # Host-mirror read (a device source was read back once per step) keeps
        # this per-layer check free of a CUDA scalar readback.
        if not graph_capture and bool(
            torch.any(
                host_input_mirror(state_inputs, "prefix_lengths") <= 0
            ).item()
        ):
            raise RuntimeError(
                "qwen4_exp PLE target verify requires a non-empty committed history"
            )
        if int(ids.numel()) != batch_size * query_len:
            raise RuntimeError(
                "qwen4_exp PLE target verify input lengths do not partition "
                f"the {ids.numel()} packed ids"
            )
        if (
            state_pool.device != hyper_states.device
            or ctx_pool.device != hyper_states.device
        ):
            raise RuntimeError(
                "qwen4_exp PLE target projection and side pools must share a device"
            )

        gathered = None
        if hyper_states.is_cuda and torch.version.hip is None:
            from rtp_llm.models_py.modules.qwen4_exp.ple.ple_commit_triton import (
                gather_target_read_state,
            )

            try:
                state_table = PLECacheViews._resolve_block_table(state_inputs, PLE_STATE_TAG)
                ctx_table = PLECacheViews._resolve_block_table(ctx_inputs, PLE_NGRAM_CTX_TAG)
            except RuntimeError:
                # Preserve the original state-before-context validation order.
                pass
            else:
                gathered = gather_target_read_state(
                    prefixes_device,
                    ids,
                    state_table,
                    ctx_table,
                    state_pool,
                    ctx_pool,
                    page_size,
                    strict=not graph_capture,
                )
        candidate_ids = ids.reshape(batch_size, query_len)
        if gathered is not None:
            initial_state, initial_context, history = gathered
        else:
            read_pages = torch.div(
                prefixes_device - 1, page_size, rounding_mode="floor"
            )
            state_read_blocks = self._physical_blocks(
                state_inputs,
                PLE_STATE_TAG,
                read_pages,
                int(state_pool.shape[0]),
                state_pool.device,
                graph_capture=graph_capture,
            )
            ctx_read_blocks = self._physical_blocks(
                ctx_inputs,
                PLE_NGRAM_CTX_TAG,
                read_pages,
                int(ctx_pool.shape[0]),
                ctx_pool.device,
                graph_capture=graph_capture,
            )
            initial_state = state_pool.index_select(0, state_read_blocks)
            initial_context = ctx_pool.index_select(0, ctx_read_blocks)
            history = torch.cat([initial_context, candidate_ids], dim=1)
        output, candidate_state_inputs = ple.decode_chunk(
            hyper_states.reshape(batch_size, query_len, -1),
            history,
            initial_state,
            add_residual=True,
            **({"embeddings": embeddings} if embeddings is not None else {}),
        )

        stage = _PLETargetLayerStage(
            layer_idx=layer_idx,
            page_size=page_size,
            prefixes=prefixes_device,
            state_pool=state_pool,
            ctx_pool=ctx_pool,
            state_inputs=state_inputs,
            ctx_inputs=ctx_inputs,
            initial_state=initial_state,
            candidate_state_inputs=candidate_state_inputs,
            initial_context=initial_context,
            candidate_ids=candidate_ids,
        )
        transaction = self.active_transaction
        if transaction is None:
            expected_layers = frozenset(int(key) for key in self.ple_layers.keys())
            transaction = _PLETargetTransaction(
                batch_size=batch_size,
                query_len=query_len,
                prefixes=prefixes_device,
                expected_layers=expected_layers,
                stages={},
            )
            self.active_transaction = transaction
        else:
            if (
                transaction.prepared_writes is not None
                or transaction.commit_started
                or transaction.tentative_committed
            ):
                raise RuntimeError("qwen4_exp PLE target transaction was not finalized")
            if (
                transaction.batch_size != batch_size
                or transaction.query_len != query_len
                or transaction.prefixes.device != prefixes_device.device
                or (
                    not graph_capture
                    and not bool(torch.equal(transaction.prefixes, prefixes_device))
                )
            ):
                raise RuntimeError(
                    "qwen4_exp PLE layers disagree about target-verify geometry"
                )
        if layer_idx not in transaction.expected_layers:
            raise RuntimeError(
                f"qwen4_exp PLE target transaction received unexpected layer {layer_idx}"
            )
        if layer_idx in transaction.stages:
            raise RuntimeError(
                f"qwen4_exp PLE target layer {layer_idx} was staged twice"
            )
        transaction.stages[layer_idx] = stage
        return output.reshape_as(hyper_states)

    @staticmethod
    def _synchronize_ple_transaction(transaction: _PLETargetTransaction) -> None:
        if transaction.prefixes.is_cuda:
            torch.cuda.current_stream(transaction.prefixes.device).synchronize()

    def _drain_ple_transaction(self, transaction: _PLETargetTransaction) -> None:
        self._synchronize_ple_transaction(transaction)
        if transaction.prefixes.is_cuda:
            torch.cuda.synchronize(transaction.prefixes.device)

    def clear_speculative_target_graph_capture(self) -> None:
        """Discard a target stage left by capture warmup, never a live commit."""
        transaction = self.active_transaction
        if transaction is None:
            return
        if transaction.graph_key is not None or transaction.prepared_writes is not None:
            raise RuntimeError("cannot discard an active PLE target commit")
        self._drain_ple_transaction(transaction)
        self.active_transaction = None

    def save_speculative_target_graph_capture(self, graph_key: int) -> None:
        """Retain graph-owned stage tensors for later replay of this exact batch."""
        if graph_key <= 0 or graph_key in self.graph_registry.captured:
            raise RuntimeError("invalid or duplicate PLE target graph key")
        transaction = self.active_transaction
        if transaction is None:
            if self.ple_layers:
                raise RuntimeError("PLE target graph captured without a transaction")
            return
        if transaction.batch_size != graph_key or transaction.graph_key is not None:
            raise RuntimeError("PLE target graph batch does not match its stage")
        if set(transaction.stages) != set(transaction.expected_layers):
            raise RuntimeError("PLE target graph has incomplete layer coverage")
        if transaction.prepared_writes is not None or transaction.commit_started:
            raise RuntimeError("PLE target graph contains an active commit")
        transaction.graph_key = graph_key
        self.graph_registry.captured[graph_key] = transaction
        self.active_transaction = None

    def activate_speculative_target_graph_replay(
        self, graph_key: int, live_batch_size: Optional[int] = None
    ) -> None:
        """Select the captured stage whose kernels just replayed."""
        if self.active_transaction is not None:
            raise RuntimeError("previous PLE target transaction is still active")
        if not self.ple_layers:
            return
        transaction = self.graph_registry.captured.get(graph_key)
        if transaction is None or transaction.batch_size != graph_key:
            raise RuntimeError("PLE target graph stage is missing for replay batch")
        if transaction.prepared_writes is not None or transaction.commit_started:
            raise RuntimeError("PLE target graph stage was not finalized")
        live_batch = graph_key if live_batch_size is None else live_batch_size
        if type(live_batch) is not int or not (0 < live_batch <= graph_key):
            raise RuntimeError("PLE target graph live batch is out of range")
        transaction.live_batch_size = live_batch
        self.active_transaction = transaction

    def _release_ple_target_transaction(
        self, transaction: _PLETargetTransaction
    ) -> None:
        if transaction.deferred_resolution is not None:
            transaction.deferred_resolution.release()
            transaction.deferred_resolution = None
        if transaction.graph_key is not None:
            transaction.prepared_writes = None
            transaction.commit_started = False
            transaction.tentative_committed = False
            transaction.deferred_resolution = None
            transaction.live_batch_size = None
        self.active_transaction = None

    @staticmethod
    def _slice_stage_for_live_batch(
        stage: _PLETargetLayerStage, live_batch: int
    ) -> _PLETargetLayerStage:
        def slice_inputs(inputs):
            device_table = getattr(inputs, "kv_cache_block_id_device", None)
            host_table = getattr(inputs, "kv_cache_block_id", None)
            return type(
                "_LiveBlockTableInputs",
                (),
                {
                    "kv_cache_block_id_device": (
                        device_table[:live_batch]
                        if isinstance(device_table, torch.Tensor)
                        and device_table.dim() == 2
                        else device_table
                    ),
                    "kv_cache_block_id": (
                        host_table[:live_batch]
                        if isinstance(host_table, torch.Tensor)
                        and host_table.dim() == 2
                        else host_table
                    ),
                },
            )()

        return _PLETargetLayerStage(
            layer_idx=stage.layer_idx,
            page_size=stage.page_size,
            prefixes=stage.prefixes[:live_batch],
            state_pool=stage.state_pool,
            ctx_pool=stage.ctx_pool,
            state_inputs=slice_inputs(stage.state_inputs),
            ctx_inputs=slice_inputs(stage.ctx_inputs),
            initial_state=stage.initial_state[:live_batch],
            candidate_state_inputs=stage.candidate_state_inputs[:live_batch],
            initial_context=stage.initial_context[:live_batch],
            candidate_ids=stage.candidate_ids[:live_batch],
        )

    def prepare_speculative_target_commit(
        self, accept_len: torch.Tensor, defer_verdict: bool = False
    ) -> None:
        """Select accepted PLE snapshots and prepare undo without writing pools."""
        transaction = self.active_transaction
        if transaction is None:
            if len(getattr(self, "ple_layers", ())):
                raise RuntimeError(
                    "qwen4_exp PLE target commit has no staged transaction"
                )
            # QSA-only or PLE-disabled qwen4 targets need no PLE side-state commit.
            return
        self._maybe_inject_speculative_fault("prepare")
        if (
            transaction.prepared_writes is not None
            or transaction.commit_started
            or transaction.tentative_committed
        ):
            raise RuntimeError("qwen4_exp PLE target transaction is already prepared")
        if set(transaction.stages) != set(transaction.expected_layers):
            missing = sorted(transaction.expected_layers.difference(transaction.stages))
            extra = sorted(
                set(transaction.stages).difference(transaction.expected_layers)
            )
            raise RuntimeError(
                "qwen4_exp PLE target transaction has incomplete layer coverage: "
                f"missing={missing}, extra={extra}"
            )
        expected_batch = (
            transaction.batch_size
            if transaction.live_batch_size is None
            else transaction.live_batch_size
        )
        if (
            not isinstance(accept_len, torch.Tensor)
            or accept_len.dim() != 1
            or accept_len.dtype != torch.int32
            or int(accept_len.numel()) != expected_batch
            or accept_len.device != transaction.prefixes.device
        ):
            raise RuntimeError(
                "qwen4_exp PLE target commit requires device-local int32 accept_len [B]"
            )
        commit_transaction = transaction
        if expected_batch < transaction.batch_size:
            commit_transaction = _PLETargetTransaction(
                batch_size=expected_batch,
                query_len=transaction.query_len,
                prefixes=transaction.prefixes[:expected_batch],
                expected_layers=transaction.expected_layers,
                stages={
                    layer_idx: self._slice_stage_for_live_batch(stage, expected_batch)
                    for layer_idx, stage in transaction.stages.items()
                },
            )
        from rtp_llm.models_py.modules.qwen4_exp.ple.ple_commit_metadata_triton import (
            begin_accepted_pair_resolution,
        )
        from rtp_llm.models_py.modules.qwen4_exp.ple.ple_commit_prepare_triton import (
            can_apply_commit_values_predicated,
            prepare_commit_values,
        )

        resolution = begin_accepted_pair_resolution(commit_transaction, accept_len)
        try:
            if resolution is not None:
                # The verdict read is deferred: the gather launches below run while
                # the device computes it, and verify() re-checks before publishing
                # (or asynchronously after predicated commit when defer_verdict=True).
                accepted = resolution.accepted
                pending = resolution.pending
            else:
                if bool(
                    torch.any(
                        (accept_len < 1) | (accept_len > commit_transaction.query_len)
                    ).item()
                ):
                    raise RuntimeError(
                        "qwen4_exp PLE target accept_len must be in "
                        f"[1, {commit_transaction.query_len}]"
                    )

                accepted = accept_len.to(torch.long)
                pending = []
                destination_rows: set[tuple[int, int]] = set()
                for layer_idx in sorted(commit_transaction.stages):
                    stage = commit_transaction.stages[layer_idx]
                    destinations = PLECacheViews._commit_physical_block_pair(
                        stage, accepted
                    )
                    if destinations is not None:
                        state_blocks, state_host_blocks, ctx_blocks, ctx_host_blocks = (
                            destinations
                        )
                    else:
                        final_positions = stage.prefixes + accepted - 1
                        final_pages = torch.div(
                            final_positions, stage.page_size, rounding_mode="floor"
                        )
                        state_blocks, state_host_blocks = self._commit_physical_blocks(
                            stage.state_inputs,
                            PLE_STATE_TAG,
                            final_pages,
                            int(stage.state_pool.shape[0]),
                            stage.state_pool.device,
                        )
                        ctx_blocks, ctx_host_blocks = self._commit_physical_blocks(
                            stage.ctx_inputs,
                            PLE_NGRAM_CTX_TAG,
                            final_pages,
                            int(stage.ctx_pool.shape[0]),
                            stage.ctx_pool.device,
                        )
                    pending.append(
                        (
                            stage,
                            state_blocks,
                            ctx_blocks,
                        )
                    )

                    for pool, destination_ids, tag in (
                        (stage.state_pool, state_host_blocks, PLE_STATE_TAG),
                        (stage.ctx_pool, ctx_host_blocks, PLE_NGRAM_CTX_TAG),
                    ):
                        pool_ptr = int(pool.data_ptr())
                        for block in destination_ids:
                            row = (pool_ptr, int(block))
                            if row in destination_rows:
                                raise RuntimeError(
                                    f"qwen4_exp PLE cache {tag!r} has a duplicate target "
                                    "destination physical row"
                                )
                            destination_rows.add(row)

            # Every layer and destination has been validated (or guarded by the
            # GPU verdict flag). Select values and allocate immutable undo only
            # now, with no persistent-pool writes.
            prepared_writes = []
            for stage, state_blocks, ctx_blocks in pending:
                selected = prepare_commit_values(
                    stage, accepted, state_blocks, ctx_blocks
                )
                if selected is None:
                    selected = self._prepare_ple_values_reference(
                        stage, accepted, state_blocks, ctx_blocks
                    )
                state_values, context_values, original_state, original_context = (
                    selected
                )
                prepared_writes.append(
                    _PLEPreparedWrite(
                        state_pool=stage.state_pool,
                        state_blocks=state_blocks,
                        state_values=state_values,
                        original_state=original_state,
                        ctx_pool=stage.ctx_pool,
                        ctx_blocks=ctx_blocks,
                        ctx_values=context_values,
                        original_context=original_context,
                        boundary_slots=self._prepare_boundary_slots(
                            stage,
                            accepted,
                            state_blocks,
                            ctx_blocks,
                            int(transaction.query_len),
                        ),
                    )
                )

            can_defer = (
                bool(defer_verdict)
                and not async_ple_commit_disabled()
                and resolution is not None
                and all(
                    can_apply_commit_values_predicated(
                        write.state_pool,
                        write.state_blocks,
                        write.state_values,
                        write.ctx_pool,
                        write.ctx_blocks,
                        write.ctx_values,
                    )
                    and all(
                        can_apply_commit_values_predicated(
                            write.state_pool,
                            slot.state_blocks,
                            slot.state_values,
                            write.ctx_pool,
                            slot.ctx_blocks,
                            slot.ctx_values,
                        )
                        for slot in write.boundary_slots
                    )
                    for write in prepared_writes
                )
            )
            if can_defer:
                transaction.deferred_resolution = resolution
                transaction.prepared_writes = prepared_writes
                return

            # Synchronous prepare path: complete gathers and verify verdict before
            # publishing the plan so C++ prepare consensus observes any failure
            # before persistent writes begin.
            if resolution is not None:
                resolution.verify()
            self._synchronize_ple_transaction(transaction)
            transaction.deferred_resolution = None
            transaction.prepared_writes = prepared_writes
        except Exception:
            if (
                resolution is not None
                and transaction.deferred_resolution is not resolution
            ):
                resolution.release()
            raise

    def _prepare_boundary_slots(
        self,
        stage: _PLETargetLayerStage,
        accepted: torch.Tensor,
        state_blocks: torch.Tensor,
        ctx_blocks: torch.Tensor,
        query_len: int,
    ) -> list[_PLEBoundaryWrite]:
        """Stage the page checkpoints an accepted window completes on the way.

        A window that crosses a page boundary finishes that page, so the page
        must end up holding the state window ending at its last token -- a
        later prefix-cache hit on that boundary resumes from that checkpoint.
        Without this refresh the completed page keeps the window from before
        the window's first token. Slots whose boundary was not crossed park on
        the final destination; the final write runs after every slot, so such
        duplicates cannot survive. All geometry stays on the device and adds
        no host synchronization to the commit.
        """
        from rtp_llm.models_py.modules.qwen4_exp.ple.ple_commit_prepare_triton import (
            prepare_commit_values,
        )

        page_size = int(stage.page_size)
        slots = (query_len + page_size - 1) // page_size
        start_pages = torch.div(stage.prefixes - 1, page_size, rounding_mode="floor")
        final_pages = torch.div(
            stage.prefixes + accepted - 1, page_size, rounding_mode="floor"
        )
        prepared: list[_PLEBoundaryWrite] = []
        for offset in range(slots):
            boundary = (start_pages + offset + 1) * page_size
            crossed = boundary < stage.prefixes + accepted
            base = torch.where(crossed, boundary - stage.prefixes, accepted)
            pages = torch.where(crossed, start_pages + offset, final_pages)
            slot_state_blocks = PLECacheViews._physical_blocks(
                stage.state_inputs,
                PLE_STATE_TAG,
                pages,
                int(stage.state_pool.shape[0]),
                stage.state_pool.device,
                graph_capture=True,
            )
            slot_ctx_blocks = PLECacheViews._physical_blocks(
                stage.ctx_inputs,
                PLE_NGRAM_CTX_TAG,
                pages,
                int(stage.ctx_pool.shape[0]),
                stage.ctx_pool.device,
                graph_capture=True,
            )
            # A page the allocator left without a block never becomes a
            # prefix-reuse boundary, so park such writes on the final block.
            slot_state_blocks = torch.where(
                slot_state_blocks > 0, slot_state_blocks, state_blocks
            )
            slot_ctx_blocks = torch.where(
                slot_ctx_blocks > 0, slot_ctx_blocks, ctx_blocks
            )
            selected = prepare_commit_values(
                stage, base, slot_state_blocks, slot_ctx_blocks
            )
            if selected is None:
                selected = self._prepare_ple_values_reference(
                    stage, base, slot_state_blocks, slot_ctx_blocks
                )
            (
                slot_state_values,
                slot_ctx_values,
                slot_state_undo,
                slot_ctx_undo,
            ) = selected
            prepared.append(
                _PLEBoundaryWrite(
                    state_blocks=slot_state_blocks,
                    state_values=slot_state_values,
                    original_state=slot_state_undo,
                    ctx_blocks=slot_ctx_blocks,
                    ctx_values=slot_ctx_values,
                    original_context=slot_ctx_undo,
                )
            )
        return prepared

    @staticmethod
    def _prepare_ple_values_reference(stage, accepted, state_blocks, ctx_blocks):
        """Portable fallback with the original selection and immutable undo."""
        state_len = int(stage.initial_state.shape[1])
        state_timeline = torch.cat(
            [stage.initial_state, stage.candidate_state_inputs], dim=1
        )
        state_indices = accepted.unsqueeze(1) + torch.arange(
            state_len, device=accepted.device, dtype=torch.long
        ).unsqueeze(0)
        state_values = state_timeline.gather(
            1,
            state_indices.unsqueeze(-1).expand(-1, -1, int(state_timeline.shape[-1])),
        )

        context_len = int(stage.initial_context.shape[1])
        context_timeline = torch.cat(
            [stage.initial_context, stage.candidate_ids], dim=1
        )
        context_indices = accepted.unsqueeze(1) + torch.arange(
            context_len, device=accepted.device, dtype=torch.long
        ).unsqueeze(0)
        context_values = context_timeline.gather(1, context_indices)
        return (
            state_values,
            context_values,
            stage.state_pool.index_select(0, state_blocks).clone(),
            stage.ctx_pool.index_select(0, ctx_blocks).clone(),
        )

    def finish_speculative_target_commit(self, commit: bool) -> None:
        """Tentatively apply a prepared PLE plan or idempotently roll it back."""
        transaction = self.active_transaction
        if transaction is None:
            return
        if commit:
            self._maybe_inject_speculative_fault("commit")
        deferred = transaction.deferred_resolution
        if not commit:
            if transaction.commit_started:
                assert transaction.prepared_writes is not None
                if deferred is not None:
                    from rtp_llm.models_py.modules.qwen4_exp.ple.ple_commit_prepare_triton import (
                        apply_commit_values_predicated,
                    )

                    verdict_flag = deferred.verdict_flag
                    for write in transaction.prepared_writes:
                        for slot in write.boundary_slots:
                            if not apply_commit_values_predicated(
                                verdict_flag,
                                write.state_pool,
                                slot.state_blocks,
                                slot.original_state,
                                write.ctx_pool,
                                slot.ctx_blocks,
                                slot.original_context,
                            ):
                                raise RuntimeError(
                                    "qwen4_exp PLE predicated boundary rollback failed"
                                )
                        if not apply_commit_values_predicated(
                            verdict_flag,
                            write.state_pool,
                            write.state_blocks,
                            write.original_state,
                            write.ctx_pool,
                            write.ctx_blocks,
                            write.original_context,
                        ):
                            raise RuntimeError("qwen4_exp PLE predicated rollback failed")
                else:
                    for write in transaction.prepared_writes:
                        for slot in write.boundary_slots:
                            write.state_pool.index_copy_(
                                0, slot.state_blocks, slot.original_state
                            )
                            write.ctx_pool.index_copy_(
                                0, slot.ctx_blocks, slot.original_context
                            )
                        write.state_pool.index_copy_(
                            0, write.state_blocks, write.original_state
                        )
                        write.ctx_pool.index_copy_(
                            0, write.ctx_blocks, write.original_context
                        )
            # Also synchronize an uncommitted transaction: target staging and a
            # failed prepare may have queued work even though no pool write was
            # launched.  Do not clear its lifetime until that work is settled.
            self._drain_ple_transaction(transaction)
            if deferred is not None:
                deferred.release()
                transaction.deferred_resolution = None
            self._release_ple_target_transaction(transaction)
            return

        if transaction.prepared_writes is None:
            raise RuntimeError("qwen4_exp PLE target transaction was not prepared")
        if transaction.commit_started:
            raise RuntimeError("qwen4_exp PLE target transaction was already committed")
        # Mark before the first copy so finish(false) restores every destination
        # even if a later launch or synchronization reports an error.
        transaction.commit_started = True
        if deferred is not None:
            from rtp_llm.models_py.modules.qwen4_exp.ple.ple_commit_prepare_triton import (
                apply_commit_values_predicated,
            )

            verdict_flag = deferred.verdict_flag
            for write in transaction.prepared_writes:
                # Boundary slots first: a slot parked on the final destination must
                # not survive the final write below.
                for slot in write.boundary_slots:
                    if not apply_commit_values_predicated(
                        verdict_flag,
                        write.state_pool,
                        slot.state_blocks,
                        slot.state_values,
                        write.ctx_pool,
                        slot.ctx_blocks,
                        slot.ctx_values,
                    ):
                        raise RuntimeError(
                            "qwen4_exp PLE predicated boundary commit failed"
                        )
                if not apply_commit_values_predicated(
                    verdict_flag,
                    write.state_pool,
                    write.state_blocks,
                    write.state_values,
                    write.ctx_pool,
                    write.ctx_blocks,
                    write.ctx_values,
                ):
                    raise RuntimeError("qwen4_exp PLE predicated commit failed")
            # Advance the completion event past every predicated store so
            # settle_speculative_target_verdict() proves both verdict arrival
            # and full GPU commit completion without synchronizing here.
            deferred.record_commit_done()
            transaction.tentative_committed = True
            return

        for write in transaction.prepared_writes:
            # Boundary slots first: a slot parked on the final destination must
            # not survive the final write below.
            for slot in write.boundary_slots:
                write.state_pool.index_copy_(0, slot.state_blocks, slot.state_values)
                write.ctx_pool.index_copy_(0, slot.ctx_blocks, slot.ctx_values)
            write.state_pool.index_copy_(0, write.state_blocks, write.state_values)
            write.ctx_pool.index_copy_(0, write.ctx_blocks, write.ctx_values)
        # Verify staging is a dependency of this stream. Draft extend
        # runs on a different stream and does not read the target pools.
        self._synchronize_ple_transaction(transaction)
        transaction.tentative_committed = True

    def finalize_speculative_target_commit(self) -> None:
        """Release undo state after C++ confirms every TP rank committed."""
        transaction = self.active_transaction
        if transaction is None:
            return
        self._maybe_inject_speculative_fault("finalize")
        if not transaction.tentative_committed:
            raise RuntimeError(
                "qwen4_exp PLE target transaction cannot finalize before commit"
            )
        if transaction.deferred_resolution is not None:
            resolution = transaction.deferred_resolution
            transaction.deferred_resolution = None
            resolution.detach_stage()
            with self._pending_verdicts_lock:
                self._pending_verdicts.append(resolution)
        self._release_ple_target_transaction(transaction)

    def has_pending_speculative_target_verdict(self) -> bool:
        with self._pending_verdicts_lock:
            return bool(self._pending_verdicts)

    def settle_speculative_target_verdict(self) -> None:
        """Await and verify the oldest deferred PLE commit verdict off-path."""
        with self._pending_verdicts_lock:
            resolution = (
                self._pending_verdicts.popleft() if self._pending_verdicts else None
            )
        if resolution is None:
            return
        resolution.verify()
