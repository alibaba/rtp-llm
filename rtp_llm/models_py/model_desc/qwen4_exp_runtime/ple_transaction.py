"""Model-wide PLE staging, prepare consensus boundary, undo and finalization.

The C++ executor owns TP consensus. Captured stages retain their addresses in
``graph_registry``; releasing an active commit only releases its undo record.
"""

from collections.abc import Callable, Mapping
from typing import Optional

import torch

from rtp_llm.models.qwen4_exp.qwen4_exp_kv_cache import PLE_NGRAM_CTX_TAG, PLE_STATE_TAG
from rtp_llm.models_py.modules.qwen4_exp.ple.ple import Qwen4ExpPLELayer
from rtp_llm.ops.compute_ops import PyAttentionInputs

from .ple_cache_views import PLECacheViews
from .ple_graph_state import PLEGraphRegistry
from .ple_state import _PLEPreparedWrite, _PLETargetLayerStage, _PLETargetTransaction


class PLETransactionCoordinator:
    def __init__(self, layers: Mapping, fault_hook: Callable[[str], None]):
        self.ple_layers = layers
        self.fault_hook = fault_hook
        self.active_transaction: Optional[_PLETargetTransaction] = None
        self.graph_registry = PLEGraphRegistry()

    def close(self) -> None:
        if self.active_transaction is not None:
            self._synchronize_ple_transaction(self.active_transaction)
        for transaction in self.graph_registry.captured.values():
            self._synchronize_ple_transaction(transaction)
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
        prefixes_device = prefixes.to(
            device=hyper_states.device, dtype=torch.long, non_blocking=True
        ).contiguous()
        if not graph_capture and bool(torch.any(prefixes_device <= 0).item()):
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
        if graph_capture:
            from rtp_llm.models_py.modules.qwen4_exp.ple.ple_commit_triton import (
                gather_target_read_state,
            )

            gathered = gather_target_read_state(
                prefixes_device,
                ids,
                PLECacheViews._resolve_block_table(state_inputs, PLE_STATE_TAG),
                PLECacheViews._resolve_block_table(ctx_inputs, PLE_NGRAM_CTX_TAG),
                state_pool,
                ctx_pool,
                page_size,
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
        return hyper_states + output.reshape_as(hyper_states)

    @staticmethod
    def _synchronize_ple_transaction(transaction: _PLETargetTransaction) -> None:
        if transaction.prefixes.is_cuda:
            torch.cuda.synchronize(transaction.prefixes.device)

    def clear_speculative_target_graph_capture(self) -> None:
        """Discard a target stage left by capture warmup, never a live commit."""
        transaction = self.active_transaction
        if transaction is None:
            return
        if transaction.graph_key is not None or transaction.prepared_writes is not None:
            raise RuntimeError("cannot discard an active PLE target commit")
        self._synchronize_ple_transaction(transaction)
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

    def activate_speculative_target_graph_replay(self, graph_key: int) -> None:
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
        self.active_transaction = transaction

    def _release_ple_target_transaction(
        self, transaction: _PLETargetTransaction
    ) -> None:
        if transaction.graph_key is not None:
            transaction.prepared_writes = None
            transaction.commit_started = False
            transaction.tentative_committed = False
        self.active_transaction = None

    def prepare_speculative_target_commit(self, accept_len: torch.Tensor) -> None:
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
        if (
            not isinstance(accept_len, torch.Tensor)
            or accept_len.dim() != 1
            or accept_len.dtype != torch.int32
            or int(accept_len.numel()) != transaction.batch_size
            or accept_len.device != transaction.prefixes.device
        ):
            raise RuntimeError(
                "qwen4_exp PLE target commit requires device-local int32 accept_len [B]"
            )
        from rtp_llm.models_py.modules.qwen4_exp.ple.ple_commit_metadata_triton import (
            begin_accepted_pair_resolution,
        )

        resolution = begin_accepted_pair_resolution(transaction, accept_len)
        if resolution is not None:
            # The verdict read is deferred: the gather launches below run while
            # the device computes it, and verify() re-checks before publishing.
            accepted = resolution.accepted
            pending = resolution.pending
        else:
            if bool(
                torch.any(
                    (accept_len < 1) | (accept_len > transaction.query_len)
                ).item()
            ):
                raise RuntimeError(
                    "qwen4_exp PLE target accept_len must be in "
                    f"[1, {transaction.query_len}]"
                )

            accepted = accept_len.to(torch.long)
            pending = []
            destination_rows: set[tuple[int, int]] = set()
            for layer_idx in sorted(transaction.stages):
                stage = transaction.stages[layer_idx]
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

        # Every layer and destination has been validated. Select values and
        # allocate immutable undo only now, with no persistent-pool writes.
        prepared_writes = []
        for stage, state_blocks, ctx_blocks in pending:
            from rtp_llm.models_py.modules.qwen4_exp.ple.ple_commit_prepare_triton import (
                prepare_commit_values,
            )

            selected = prepare_commit_values(stage, accepted, state_blocks, ctx_blocks)
            if selected is None:
                selected = self._prepare_ple_values_reference(
                    stage, accepted, state_blocks, ctx_blocks
                )
            state_values, context_values, original_state, original_context = selected
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
                )
            )
        # Gather/clone launches above are part of prepare, not commit.  Complete
        # them before publishing the plan so the C++ prepare consensus observes
        # any asynchronous failure on this rank before persistent writes begin.
        # The gathers run on the current stream, so a current-stream
        # synchronize observes their failures without draining unrelated
        # streams (async bookkeeping) the way a device-wide sync does.
        if resolution is not None:
            resolution.verify()
        torch.cuda.current_stream(transaction.prefixes.device).synchronize()
        transaction.prepared_writes = prepared_writes

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
        if not commit:
            if transaction.commit_started:
                assert transaction.prepared_writes is not None
                for write in transaction.prepared_writes:
                    write.state_pool.index_copy_(
                        0, write.state_blocks, write.original_state
                    )
                    write.ctx_pool.index_copy_(
                        0, write.ctx_blocks, write.original_context
                    )
            # Also synchronize an uncommitted transaction: target staging and a
            # failed prepare may have queued work even though no pool write was
            # launched.  Do not clear its lifetime until that work is settled.
            self._synchronize_ple_transaction(transaction)
            self._release_ple_target_transaction(transaction)
            return

        if transaction.prepared_writes is None:
            raise RuntimeError("qwen4_exp PLE target transaction was not prepared")
        if transaction.commit_started:
            raise RuntimeError("qwen4_exp PLE target transaction was already committed")
        # Mark before the first copy so finish(false) restores every destination
        # even if a later launch or synchronization reports an error.
        transaction.commit_started = True
        for write in transaction.prepared_writes:
            write.state_pool.index_copy_(0, write.state_blocks, write.state_values)
            write.ctx_pool.index_copy_(0, write.ctx_blocks, write.ctx_values)
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
        self._release_ple_target_transaction(transaction)
