"""Maintained eager PP layer execution after native WORLD admission.

The caller owns initialization, execution begin/completion, transport tickets,
allocation retention and retirement. This adapter preserves the worker and
transformer arithmetic owners; it does not schedule or admit public forwards.
"""

from collections.abc import Mapping
from contextlib import nullcontext

import torch

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
    pack_v41_pp_frame,
    validate_v41_pp_frame,
)


def l20_tables(dispatch):
    return {
        span.identity: {
            tag: tuple(dispatch.tables[tag][row].tolist())
            for tag in ("hca_kv", "indexer_kv")
        }
        for row, span in enumerate(dispatch.spans)
    }


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
    worker._check_rotary_config()
    contract.require_rotary_config(worker._v4_args.v41_config)
    current = prepare_native_pp_prefill_dispatch(worker, dispatch.native_inputs)
    if (
        current.schedule != dispatch.schedule
        or current.tables.keys() != dispatch.tables.keys()
    ):
        raise ValueError("Native V4.1 PP execution metadata changed after preflight")
    pairs = [(current.input_ids, dispatch.input_ids), (current.rows, dispatch.rows)]
    pairs.extend((current.tables[tag], dispatch.tables[tag]) for tag in current.tables)
    if current.token_windows is None or dispatch.token_windows is None:
        if current.token_windows is not dispatch.token_windows:
            raise ValueError("Native V4.1 PP history changed after preflight")
    else:
        pairs.append((current.token_windows, dispatch.token_windows))
    if any(
        a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a, b)
        for a, b in pairs
    ):
        raise ValueError("Native V4.1 PP execution snapshot changed after preflight")
    if worker._v4_args.pp_rank == 0:
        if receiver is not None or tensors is not None:
            raise ValueError("Native stage zero cannot import an upstream frame")
        return current, None
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
    current = frame = error = None
    try:
        try:
            current, frame = _validate_execution(
                worker, dispatch, contract, transfer_round, receiver, tensors
            )
        except Exception as exc:
            error = exc
        try:
            worker._agree_dp(
                error,
                (
                    "native-pp-execution",
                    transfer_round,
                    None if current is None else current.schedule,
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
            payload.update(pack_v41_pp_frame(delta, state, current.spans, current.rows))

        try:
            if frame is None:
                source_scope = premix_scope = nullcontext()
                worker._prepare_engram(inputs)
            else:
                receiver.apply(tensors, current.spans)
                source = prepare_native_l20_prefill_source(
                    frame, receiver, current.spans, current.rows, cp_ctx=current.cp
                )
                entry = worker.v4.layers[0]
                # Maintained mHC post writes into its residual. The received
                # frame remains owned by transport until consumer completion.
                residual = frame.hidden_states.clone()
                source_scope = imported_native_l20_prefill(entry.attn, source)
                premix_scope = imported_delayed_pre_mix(
                    entry.attn_hc, residual, frame.prev_pre_mix
                )
                inputs.pp_intermediates = {"hidden_states": residual}
            with source_scope, premix_scope:
                outputs = forward_prefill(
                    worker.v4,
                    worker.kv_cache,
                    worker.parallelism_config,
                    inputs,
                    after_layers_fn=export if frame is None else None,
                )
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
