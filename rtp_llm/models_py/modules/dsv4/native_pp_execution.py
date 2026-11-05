"""Maintained eager PP layer execution after native WORLD admission.

The caller owns initialization, execution begin/completion, transport tickets,
allocation retention and retirement. This adapter preserves the worker and
transformer arithmetic owners; it does not schedule or admit public forwards.
"""

import os
import time

# R51: per-process guard so the candidate-TopK-v2 warmup + markers run once per
# rank, not on every per-chunk admission (4416 calls/run otherwise).
_V41_CTKV2_WARMED: set = set()
from collections.abc import Mapping
from contextlib import nullcontext

import torch

from rtp_llm.config.dsv41_config import dsv41_prefill_pp_cut
from rtp_llm.models_py.modules.dsv4.hc.pp_state import imported_delayed_pre_mix
from rtp_llm.models_py.modules.dsv4.native_l20_export import snapshot_native_l20_frame
from rtp_llm.models_py.modules.dsv4.native_l20_prefill import (
    imported_native_l20_prefill,
    prepare_native_l20_prefill_source,
)
from rtp_llm.models_py.modules.dsv4.native_l20_transfer import (
    NativeL20ReplicaReceiver,
    export_native_l20_delta,
)
from rtp_llm.models_py.modules.dsv4.native_pp_prefill import (
    NativePPPrefillDispatch,
    prepare_native_pp_prefill_dispatch,
)
from rtp_llm.models_py.modules.dsv4.prefill.forward import forward_prefill
from rtp_llm.models_py.modules.dsv4.v41_cache_transfer import V41L20Contract
from rtp_llm.models_py.modules.dsv4.v41_pp import (
    pack_v41_hidden_frame,
    pack_v41_pp_frame,
    validate_v41_hidden_frame,
    validate_v41_pp_frame,
)

# Test-only adversarial consumer delay (guide §8), default off. When set, the consumer stage
# sleeps this many milliseconds before each chunk's L20 import, so the producer runs ahead
# until the pipeline slot ring's reuse wait blocks it -- proving the backpressure bound
# (pp_size + 1 slots) rather than merely not hitting it. The perf and numerical harnesses
# never set it; only the delayed-consumer scenario does.
_CONSUMER_DELAY_MS = int(os.environ.get("DSV41_TEST_DELAY_CONSUMER_MS", "0"))


def l20_tables(dispatch):
    tables = {}
    for row, span in enumerate(dispatch.spans):
        tables[span.identity] = {}
        for tag in ("hca_kv", "indexer_kv"):
            pages = dispatch.tables[tag][row].tolist()
            # Preflight checks these rectangular tables against native leases.
            # Only the trailing zeros are padding, never allocated pages.
            while pages and pages[-1] == 0:
                pages.pop()
            tables[span.identity][tag] = tuple(pages)
    return tables


def _validate_execution(worker, dispatch, contract, transfer_round, receiver, tensors):
    if worker._v4_args.pp_size != 2:
        raise NotImplementedError("Native V4.1 PP execution requires CEP4PP2")
    if torch.cuda.is_current_stream_capturing() or (
        torch.cuda.current_stream(worker._weight_device()) != worker._worker_stream
    ):
        raise NotImplementedError(
            "Native V4.1 PP execution requires its eager worker stream"
        )
    if (
        not isinstance(dispatch, NativePPPrefillDispatch)
        or dispatch.worker_owner is not worker._native_pp_owner
        or not isinstance(contract, V41L20Contract)
        or type(transfer_round) is not int
        or transfer_round < 0
    ):
        raise ValueError(
            "Native V4.1 PP execution requires its admitted worker descriptor"
        )
    worker._check_execution_config()
    contract.require_attention_config(worker._v4_args.v41_config)
    # The admitted dispatch is immutable across the request's chunks, and
    # prepare_native_pp_prefill_dispatch is a pure function of the fixed
    # dispatch.native_inputs (it takes no transfer_round), so re-running it on
    # every chunk recomputes an identical dispatch. That redundant host work --
    # dozens of GPU syncs plus _require_cp_wire's O(tokens) Python list building
    # -- pins the host behind the GPU every chunk and is the serializer that
    # keeps the pp_intermediates pipeline from overlapping the two stages. Reuse
    # the admitted dispatch directly (admission already validated it). Set
    # DSV41_PP_SKIP_REPREP=0 to restore the defensive per-chunk re-preparation.
    if os.environ.get("DSV41_PP_SKIP_REPREP", "1") == "1":
        current = dispatch
    else:
        current = prepare_native_pp_prefill_dispatch(worker, dispatch.native_inputs)
        if (
            current.schedule != dispatch.schedule
            or current.tables.keys() != dispatch.tables.keys()
        ):
            raise ValueError(
                "Native V4.1 PP execution metadata changed after preflight"
            )
        pairs = [(current.input_ids, dispatch.input_ids), (current.rows, dispatch.rows)]
        pairs.extend(
            (current.tables[tag], dispatch.tables[tag]) for tag in current.tables
        )
        if current.token_windows is None or dispatch.token_windows is None:
            if current.token_windows is not dispatch.token_windows:
                raise ValueError("Native V4.1 PP history changed after preflight")
        else:
            pairs.append((current.token_windows, dispatch.token_windows))
        if any(
            a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a, b)
            for a, b in pairs
        ):
            raise ValueError(
                "Native V4.1 PP execution snapshot changed after preflight"
            )
    if worker._v4_args.pp_rank == 0:
        if receiver is not None or tensors is not None:
            raise ValueError("Native stage zero cannot import an upstream frame")
        return current, None
    if dsv41_prefill_pp_cut() == 20:
        if receiver is not None or not isinstance(tensors, Mapping):
            raise ValueError(
                "Native cut-20 stage one requires a hidden frame without a replica receiver"
            )
        frame = validate_v41_hidden_frame(
            tensors,
            contract,
            transfer_round,
            current.cp.cp_rank,
            current.spans,
            current.rows,
            worker._weight_device(),
        )
        return current, frame
    if (
        not isinstance(receiver, NativeL20ReplicaReceiver)
        or receiver._native_cache is not worker.kv_cache
        or receiver.contract != contract
        or receiver.cp_rank != current.cp.cp_rank
        or receiver._next_round != transfer_round
        or not isinstance(tensors, Mapping)
    ):
        raise ValueError("Native stage-one receiver belongs to another owner or lane")
    for identity, tables in l20_tables(current).items():
        if receiver.block_tables(identity) != tables:
            raise ValueError(
                "Native L20 replica tables disagree with the admitted allocation"
            )
    frame = validate_v41_pp_frame(tensors, receiver, current.spans, current.rows)
    receiver.validate(tensors, current.spans)
    return current, frame


def execute_native_pp_prefill(
    worker, dispatch, contract, transfer_round, *, receiver=None, tensors=None
):
    """Serialize at the worker; STAGE validates before either import or layers."""
    from rtp_llm.models_py.modules.dsv4._nvtx_phases import nvtx_phase
    from rtp_llm.models_py.modules.dsv4._v41_chunk_boundary import (
        chunk_boundary_overlap_enabled as _chunk_boundary_overlap_enabled,
    )
    from rtp_llm.models_py.modules.dsv4.fp8 import _v41_compact_cp_runtime as compact_cp

    current = frame = error = compact = None
    try:
        # R30/B: one-time per-device pre-warm (allocator pools + pinned
        # staging) before the first chunk's admission work.
        from rtp_llm.models_py.modules.dsv4._v41_staging_prewarm import (
            prewarm_device,
            staging_prewarm_enabled,
        )

        if staging_prewarm_enabled():
            prewarm_device(worker._weight_device())
        # R51: warm the SM120 fused candidate TopK once per rank (flag-gated)
        # so the first measured prefill doesn't pay the nvcc compile.
        _ctk_dev = worker._weight_device()
        _ctk_key = getattr(_ctk_dev, "index", None)
        if _ctk_key is None:
            _ctk_key = str(_ctk_dev)
        if _ctk_key not in _V41_CTKV2_WARMED:
            _V41_CTKV2_WARMED.add(_ctk_key)
            from rtp_llm.models_py.modules.dsv4.fp8._v41_candidate_topk_v2 import (
                _enabled as _ctk_v2_enabled,
            )

            if _ctk_v2_enabled():
                from rtp_llm.models_py.modules.dsv4.fp8._v41_candidate_topk_v2 import (
                    warmup as _ctk_v2_warmup,
                )

                _wr = _ctk_v2_warmup(_ctk_dev)
                print(
                    "[V41CTKV2] warmup returned %s (DSV41_CANDIDATE_TOPK_V2=1)"
                    % (_wr,),
                    flush=True,
                )
        agree_wait = None
        with nvtx_phase("v41.admission"):
            try:
                current, frame = _validate_execution(
                    worker, dispatch, contract, transfer_round, receiver, tensors
                )
                compact = compact_cp.prepare(worker, current, transfer_round)
            except Exception as exc:
                error = exc
            if _chunk_boundary_overlap_enabled():
                # R28/C: issue the DP agreement now; wait+check right before
                # the layers (still before any of this chunk's collectives).
                agree_wait = worker._agree_dp_async(
                    error,
                    (
                        "native-pp-execution",
                        transfer_round,
                        None if current is None else current.schedule,
                        None if compact is None else compact.signature,
                    ),
                    worker._weight_device(),
                    native_dispatch=True,
                )
                error = None
            else:
                try:
                    worker._agree_dp(
                        error,
                        (
                            "native-pp-execution",
                            transfer_round,
                            None if current is None else current.schedule,
                            None if compact is None else compact.signature,
                        ),
                        worker._weight_device(),
                        native_dispatch=True,
                    )
                finally:
                    error = None
        inputs = current.native_inputs
        previous_intermediates = inputs.pp_intermediates
        payload = {}

        def export(hidden):
            with nvtx_phase("v41.pp_export"):
                if dsv41_prefill_pp_cut() == 20:
                    payload.update(
                        pack_v41_hidden_frame(
                            worker.v4.layers[-1],
                            hidden,
                            contract,
                            transfer_round,
                            current.cp.cp_rank,
                            current.spans,
                            current.rows,
                        )
                    )
                    return
                state = snapshot_native_l20_frame(
                    worker.v4.layers[-1], hidden, current.spans, current.rows
                )
                delta = export_native_l20_delta(
                    worker._v4_args.v41_config,
                    worker.kv_cache,
                    contract,
                    current.cp.cp_rank,
                    transfer_round,
                    current.spans,
                    l20_tables(current),
                )
                payload.update(
                    pack_v41_pp_frame(delta, state, current.spans, current.rows)
                )

        try:
            with nvtx_phase("v41.chunk_setup"):
                if frame is None:
                    source_scope = premix_scope = nullcontext()
                    worker._prepare_engram(inputs)
                else:
                    if _CONSUMER_DELAY_MS:
                        time.sleep(_CONSUMER_DELAY_MS / 1000)
                    source_scope = nullcontext()
                    if dsv41_prefill_pp_cut() == 21:
                        receiver.apply(tensors, current.spans)
                        source = prepare_native_l20_prefill_source(
                            frame,
                            receiver,
                            current.spans,
                            current.rows,
                            cp_ctx=current.cp,
                        )
                        source_scope = imported_native_l20_prefill(
                            worker.v4.layers[0].attn, source
                        )
                    entry = worker.v4.layers[0]
                    # Maintained mHC post writes into its residual. The received
                    # frame remains owned by transport until consumer completion.
                    residual = frame.hidden_states.clone()
                    premix_scope = imported_delayed_pre_mix(
                        entry.attn_hc, residual, frame.prev_pre_mix
                    )
                    inputs.pp_intermediates = {"hidden_states": residual}
            if agree_wait is not None:
                with nvtx_phase("v41.agree_wait"):
                    agree_wait()
            with nvtx_phase("v41.layers"):
                with source_scope, premix_scope, compact_cp.execution_scope(
                    worker, compact
                ):
                    outputs = forward_prefill(
                        worker.v4,
                        worker.kv_cache,
                        worker.parallelism_config,
                        inputs,
                        after_layers_fn=export if frame is None else None,
                    )
            with nvtx_phase("v41.result"):
                if frame is None:
                    outputs.pp_intermediates = payload
            return outputs
        except BaseException:
            worker._worker_state = "failed"
            raise
        finally:
            inputs.pp_intermediates = previous_intermediates
    finally:
        worker._clear_forward_state()
        worker.v4._propagate_cp_ctx(None)
        worker.v4.set_cp_info(None, 1, 0)
        shared = worker.v4.layers[0].attn._shared_attention
        shared["global"] = {}
        shared["topk"] = {}
        shared["candidates"] = None
