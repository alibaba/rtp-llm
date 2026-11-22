"""Forward-local PPU FFN-post to next-attention MegaMHC boundaries.

The public decode loop continues to pass a real residual Tensor. A producer
may prepare the next attention input while completing its own pending post;
only that invocation's ``state.shared`` carries the prepared value. Graph
capture records both kernels and their live buffers. Replay never consumes a
Python value from an earlier request. New coverage is opt-in and unvalidated.
"""

from dataclasses import dataclass, replace

import torch

from rtp_llm.models_py.modules.dsv41.hc.decode_transition import PendingDecodePost
from rtp_llm.models_py.modules.dsv41.ppu.optimization_config import (
    CROSS_LAYER_MHC,
    enabled,
)
from rtp_llm.models_py.modules.dsv41.ppu.row_metadata import tensor_metadata_key


_SCOPE = "ppu_decode_mhc_scope"
_TICKET = "ppu_decode_mhc_ticket"
_PHASES = ("decode", "graph", "verify", "graph_verify")


@dataclass(frozen=True, eq=False)
class _Scope:
    token: object
    state: object
    metadata: object
    input_ids: torch.Tensor
    input_key: tuple
    cache: object


@dataclass(frozen=True, eq=False)
class _Ticket:
    scope: _Scope
    source: object
    target: object
    source_state: object
    source_mix_keys: tuple
    target_keys: tuple
    transition: object
    residual_key: tuple


def reshape_transition(transition, shape):
    """Restore the caller's [B,Q] token axes without an additional kernel."""
    if len(shape) == 3:
        return transition
    leading, hc, dim = tuple(shape[:-2]), shape[-2], shape[-1]
    return replace(
        transition,
        residual=transition.residual.view(*shape),
        normalized=transition.normalized.view(*leading, dim),
        pre_mix=transition.pre_mix.view(*leading, hc),
        post_mix=transition.post_mix.view(*leading, hc, 1),
        residual_mix=transition.residual_mix.view(*leading, hc, hc),
    )


def flatten_mix(state):
    return replace(
        state,
        hidden=None,
        pre_mix=state.pre_mix.reshape(-1, 4),
        post_mix=state.post_mix.reshape(-1, 4, 1),
        residual_mix=state.residual_mix.reshape(-1, 4, 4),
    )


def _mix_keys(mix):
    return tuple(tensor_metadata_key(getattr(mix, name)) for name in (
        "pre_mix", "post_mix", "residual_mix",
    ))


def _target_keys(target):
    return (
        *(tensor_metadata_key(value) for value in (
            target.attn_hc.fn, target.attn_hc.base, target.attn_hc.scale,
            target.attn_norm.weight,
        )),
        target.attn_norm.variance_epsilon,
    )


def _plain_block(block):
    # Restrict to the two existing PPU wrappers, not arbitrary user overrides.
    kind = type(block)
    return (
        (kind.__module__, kind.__name__) in (
            ("rtp_llm.models_py.modules.dsv41.ppu.block", "SharedPPUBlock"),
            ("rtp_llm.models_py.modules.dsv41.ppu.model", "SharedPPUTargetBlock"),
        )
        and not block.query_only
    )


def _call_eligible(block, state, x, metadata, input_ids, kwargs):
    if (
        not enabled(CROSS_LAYER_MHC)
        or not _plain_block(block)
        or block._mhc_phase not in _PHASES
        or getattr(state, "prefill", False)
        or getattr(state, "idle", False)
        or getattr(state, "proposal", False)
        or getattr(state, "is_graph_commit", False)
        or torch.is_grad_enabled()
        or any(name in kwargs for name in ("_prepared_attn", "_defer_ffn_post"))
        or kwargs.get("routing_metadata") is not None
        or kwargs.get("attn_fn") is not None
        or set(kwargs) - {"kv_cache", "routing_metadata", "attn_fn"}
    ):
        return False
    runtime = block.attn.runtime
    model = runtime.model
    if (
        runtime.current is not state
        or getattr(getattr(model, "parallelism_config", None), "tp_size", None) != 1
        or getattr(model.v4, "commit_only", False)
        or (getattr(model, "_is_speculative", False) and not getattr(state, "verify", False))
        or not isinstance(x, torch.Tensor)
        or x.ndim != 4
        or tuple(x.shape[-2:]) != (4, 5120)
        or tuple(x.shape[:2]) != (metadata.batch_size, metadata.q_len_per_req)
        or x.dtype != torch.bfloat16
        or not x.is_cuda
        or not x.is_contiguous()
        or x.is_neg() or x.is_conj()
        or not isinstance(input_ids, torch.Tensor)
        or input_ids.numel() != x.shape[0] * x.shape[1]
    ):
        return False
    from rtp_llm.models_py.modules.dsv4 import _record_tensor
    from rtp_llm.models_py.modules.dsv41.ppu.attention_layer import optimized_diagnostics_enabled

    return not (
        optimized_diagnostics_enabled()
        or _record_tensor.should_record_layer(block.layer_id)
        or (
            block._mhc_phase != "graph_verify"
            and block.layer_id in model.v4.capture_aux_hidden_layer_ids
        )
    )


def _adjacent(block, target):
    runtime = block.attn.runtime
    layers = runtime.model.v4.layers
    previous = target.attn_hc._previous_ref
    if (
        not _plain_block(target)
        or type(target) is not type(block)
        or target.layer_id != block.layer_id + 1
        or not 0 <= block.layer_id < target.layer_id < len(layers)
        or layers[block.layer_id] is not block
        or layers[target.layer_id] is not target
        or target.attn.runtime is not runtime
        or target.engram is not None
        or (
            block._mhc_phase != "graph_verify"
            and target.layer_id in runtime.model.v4.capture_aux_hidden_layer_ids
        )
        or previous is None or previous() is not block.ffn_hc
    ):
        return False
    from rtp_llm.models_py.modules.dsv4 import _record_tensor

    return not _record_tensor.should_record_layer(target.layer_id)


def prepare_decode_boundary(block, state, x, metadata, input_ids, kwargs):
    """Return (prepared_attention, next_layer, scope), declining before writes."""
    shared = getattr(state, "shared", None)
    if not isinstance(shared, dict):
        return None, None, None
    # Even a retried invocation with reused state cannot inherit an old ticket.
    if block.layer_id == 0:
        shared.pop(_SCOPE, None)
        shared.pop(_TICKET, None)
    ticket = shared.pop(_TICKET, None)
    if not _call_eligible(block, state, x, metadata, input_ids, kwargs):
        return None, None, None
    scope = shared.get(_SCOPE)
    input_key = tensor_metadata_key(input_ids)
    if (
        scope is None or scope.state is not state or scope.metadata is not metadata
        or scope.input_ids is not input_ids or scope.input_key != input_key
        or scope.cache is not state.cache
    ):
        scope = _Scope(object(), state, metadata, input_ids, input_key, state.cache)
        shared[_SCOPE] = scope
        ticket = None
    prepared = None
    if (
        isinstance(ticket, _Ticket)
        and ticket.scope is scope
        and ticket.target is block
        and _adjacent(ticket.source, block)
        and ticket.source.ffn_hc.state is ticket.source_state
        and _mix_keys(ticket.source_state) == ticket.source_mix_keys
        and _target_keys(block) == ticket.target_keys
        and x is ticket.transition.residual
        and tensor_metadata_key(x) == ticket.residual_key
    ):
        transition = ticket.transition
        block.attn_hc.accept_transition(transition)
        prepared = (
            transition.residual, transition.normalized,
            transition.post_mix, transition.residual_mix,
        )
    layers = block.attn.runtime.model.v4.layers
    target = layers[block.layer_id + 1] if block.layer_id + 1 < len(layers) else None
    if target is not None and not _adjacent(block, target):
        target = None
    return prepared, target, scope


def finish_decode_boundary(block, state, pending, target, scope):
    """Complete a deferred post, optionally preparing the adjacent attention."""
    if not isinstance(pending, PendingDecodePost):
        raise RuntimeError("PPU deferred decode must return PendingDecodePost")
    shared = state.shared
    mix = block.ffn_hc.state
    if (
        shared.get(_SCOPE) is not scope or _TICKET in shared
        or block.attn.runtime.current is not state
        or not _adjacent(block, target)
        or pending.previous is not block.ffn_hc
        or mix is None or pending.post is not mix.post_mix
        or pending.comb is not mix.residual_mix
        or mix.pre_mix is None
        or not pending.output.is_contiguous()
        or not pending.residual.is_contiguous()
    ):
        return pending.materialize()
    target_keys, source_keys = _target_keys(target), _mix_keys(mix)
    transition = block.ops.try_mhc_ffn_transition(
        pending.output.view(-1, pending.output.shape[-1]),
        pending.residual.view(-1, 4, 5120),
        flatten_mix(mix),
        target.attn_hc.parameters_map,
        target.attn_norm.weight,
        target.attn_norm.variance_epsilon,
        phase=block._mhc_phase + "_cross",
    )
    if transition is None:
        return pending.materialize()
    # Graph Verify captures auxiliary features from this completed residual
    # at layer entry, independently of the prepared attention input.
    # Native outputs are disjoint from the source residual. If the consumer
    # later declines this ticket, its ordinary pre recomputes from this correct
    # updated residual and the unchanged previous FFN coefficients.
    transition = reshape_transition(transition, pending.residual.shape)
    shared[_TICKET] = _Ticket(
        scope, block, target, mix, source_keys, target_keys, transition,
        tensor_metadata_key(transition.residual),
    )
    return transition.residual
