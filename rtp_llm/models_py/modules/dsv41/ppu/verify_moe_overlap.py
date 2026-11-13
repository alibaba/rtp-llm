"""An explicit fork/join plan for the fixed target Verify MoE surface.

The existing router, MegaMoE, shared expert and BF16 addition do all math.
This plan owns only stream ordering and its capture-external resources.
It does not establish that concurrent MegaMoE makes progress on WORLD8.
"""

import threading

import torch

from rtp_llm.models_py.modules.dsv41.ppu._quant import is_ppu
from rtp_llm.models_py.modules.dsv41.ppu.optimization_config import (
    enabled as optimization_enabled,
)
from rtp_llm.models_py.modules.dsv41.ppu.overlap_resources import (
    OverlapResources,
    record_stream,
    stream_key as _stream_key,
)


def _metadata(value):
    if isinstance(value, torch.Tensor):
        return (value.data_ptr(), tuple(value.shape), tuple(value.stride()), str(value.dtype), str(value.device))
    if isinstance(value, (tuple, list)):
        return tuple(_metadata(item) for item in value)
    if isinstance(value, dict):
        return tuple((str(key), _metadata(item)) for key, item in sorted(value.items(), key=lambda item: str(item[0])))
    return type(value).__name__, id(value)


class VerifyMoEOverlap:
    """One nonreentrant caller/side-stream DAG for target40, M32, Q4.

    Construct after ``engine.prepare_shared_graph``. Register the actual C++
    runner's nondefault capture stream with ``bind_caller`` in eager warmup;
    constructor-time current/default streams are not the capture caller.
    Hold this plan for the lifetime of every graph that records its resources.
    Replay must remain ordered on that caller and use the same engine storage.
    """

    def __init__(self, device, engine, *, target_layers=40, max_rows=32):
        self.device = torch.device(device)
        if self.device.type != "cuda" or not is_ppu():
            raise ValueError("Verify MoE overlap requires PPU CUDA")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("create Verify MoE overlap before capture")
        if target_layers != 40 or max_rows != 32:
            raise ValueError("Verify MoE overlap is fixed to target40/M32")
        self.engine = engine
        self.target_layers, self.max_rows = target_layers, max_rows
        if not self._engine_ready() or not all(
            (layer, self.device) in getattr(engine, "_processed", {}) for layer in range(target_layers)
        ):
            raise ValueError("prepare target MegaMoE W4A4 resources before overlap")
        self._owner_weights = {
            layer: (engine._processed[layer, self.device], _metadata(engine._processed[layer, self.device]))
            for layer in range(target_layers)
        }
        self._resources = OverlapResources(self.device, name="Verify MoE overlap")
        self._lock = self._resources.lock
        self._python_calls = self._forks = self._joins = 0
        self._initial_signature = self.storage_signature()

    @property
    def side_stream(self):
        return self._resources.side_stream

    @property
    def producer(self):
        return self._resources.producer

    @property
    def done(self):
        return self._resources.done

    @property
    def _caller(self):
        return self._resources.caller

    @property
    def _caller_key(self):
        return self._resources.caller_key

    @property
    def _closed(self):
        return self._resources.closed

    def _engine_ready(self):
        engine = self.engine
        return (
            (getattr(engine, "backend", None), getattr(engine, "precision", None)) == ("megamoe", "w4a4")
            and getattr(engine, "ep_size", None) == 8
            and getattr(engine, "hidden_size", None) == 5120
            and getattr(engine, "inter_dim", None) == 2304
            and getattr(engine, "num_routed_experts", None) == 384
            and getattr(engine, "_mega_buffer", None) is not None
            and getattr(engine, "_mega_capacity", 0) >= self.max_rows
            and getattr(engine, "_mega_device", None) == self.device
        )

    def storage_signature(self):
        """Read live host metadata, including replaced engine storage.

        Caller binding is deliberately absent: it first occurs during native
        warmup after the enclosing runtime has saved its initial fingerprint.
        Event object identities are stable across their first native record.
        """
        engine = self.engine
        return (
            id(engine), id(getattr(engine, "_mega_buffer", None)),
            getattr(engine, "_mega_generation", None), getattr(engine, "_mega_capacity", None),
            str(getattr(engine, "_mega_device", None)),
            getattr(engine, "backend", None), getattr(engine, "precision", None),
            _metadata(getattr(engine, "_mega_recv_stats", None)), id(getattr(engine, "_processed", None)),
            *self._resources.storage_signature(),
        )

    def validate(self):
        if self._closed or not self._engine_ready() or self.storage_signature() != self._initial_signature:
            raise RuntimeError("Verify MoE overlap resources changed; recapture required")

    def bind_caller(self, stream=None):
        """Initialize event handles once, outside capture, on the real caller."""
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("bind Verify MoE caller before capture")
        self.validate()
        stream = torch.cuda.current_stream(self.device) if stream is None else stream
        self._resources.bind_caller(stream)

    prepare_caller = bind_caller

    def supported(self, state, flat, owner, *, allowed=None):
        if allowed is None:
            allowed = getattr(state, "verify_moe_overlap_allowed", False)
        if (
            not allowed or self._closed or self._caller_key is None
            or not optimization_enabled("RTP_V41_PPU_VERIFY_DIRECT_MOE_ROUTES", default=True)
            or not self._engine_ready()
            or getattr(state, "is_graph", False) is not True
            or getattr(state, "verify", False) is not True
            or getattr(state, "prefill", True) is not False
            or getattr(state, "proposal", False)
            or getattr(state, "grouped_moe", None) is not None
            or getattr(state, "packed_attention", None) is None
            or not isinstance(flat, torch.Tensor)
            or flat.device != self.device or flat.dtype != torch.bfloat16
            or flat.shape != (32, 5120) or not flat.is_contiguous()
            or torch.is_grad_enabled() or flat.requires_grad
            or not 0 <= getattr(owner, "layer_id", -1) < self.target_layers
            or not getattr(owner, "routed_enabled", False)
            or getattr(owner, "shared_expert", None) is None
            or not callable(getattr(owner, "gate", None))
            or not callable(getattr(owner, "shared_expert", None))
            or getattr(getattr(owner, "ops", None), "moe_engine", None) is not self.engine
            or getattr(getattr(owner, "runtime", None), "current", None) is not state
        ):
            return False
        metadata = getattr(state, "metadata", None)
        original, signature = self._owner_weights[owner.layer_id]
        current = getattr(self.engine, "_processed", {}).get((owner.layer_id, self.device))
        if current is not original or _metadata(current) != signature:
            return False
        if getattr(metadata, "q_len_per_req", None) != 4 or getattr(metadata, "batch_size", None) != 8:
            return False
        positions, valid = getattr(metadata, "positions", None), getattr(metadata, "valid_mask", None)
        for value, dtype, shape in (
            (positions, torch.int64, (32,)), (valid, torch.bool, (32,)),
            (getattr(state, "moe_calls", None), torch.int64, (40,)),
            (getattr(state, "routed_rows_device", None), torch.int64, (40,)),
        ):
            if not isinstance(value, torch.Tensor) or value.device != self.device or value.dtype != dtype or value.shape != shape:
                return False
        return _stream_key(torch.cuda.current_stream(self.device)) == self._caller_key

    def forward(self, owner, flat, shape, state, expert_load=None, return_shared=False):
        """Enqueue the unchanged shared/routed branches and join before use."""
        self.validate()
        if not self.supported(state, flat, owner):
            raise ValueError("unqualified Verify MoE overlap input/caller")
        caller = torch.cuda.current_stream(self.device)
        with self._resources.submission():
            routed, shared = self._enqueue_branches(owner, flat, state, expert_load, caller)
        # Input/output allocator registrations remain effective after this
        # frame releases its references, including for return_shared views.
        if return_shared:
            return routed.reshape(shape), shared.reshape(shape)
        from rtp_llm.models_py.modules.dsv41.ppu.moe import combine_routed_and_shared

        return combine_routed_and_shared(routed, shared).reshape(shape)

    def _enqueue_branches(self, owner, flat, state, expert_load, caller):
        forked, failure = False, None
        try:
            self._python_calls += 1
            self._resources.fork(caller)
            forked = True
            self._forks += 1
            # Register the side read before either callback can raise. Stream
            # dependencies order execution; allocator ownership is separate.
            record_stream(flat, self.side_stream)
            # Match SGLang's enqueue order: shared work can start before the
            # full-residency routed pipeline is submitted on the side stream.
            shared = owner.shared_expert(flat, allow_quant=False)
            with torch.cuda.stream(self.side_stream):
                ids, weights = owner.gate(
                    flat, graph_routes=(state.metadata.valid_mask, state.routed_rows_device[owner.layer_id]),
                    allow_top3_graph_routes=False,
                )
                # Preserve the original gate-success counter boundary: a
                # router exception must not count an attempted MegaMoE call.
                if not getattr(state, "bulk_counters", False):
                    state.moe_calls[owner.layer_id].add_(1)
                routed = self.engine.forward_shared_graph(
                    flat, ids, weights, layer_id=owner.layer_id, expert_load=expert_load,
                    fused_expert_load=False,
                )
                # The result is allocated on side but its addition or tuple
                # consumer runs on caller, potentially after this frame dies.
                record_stream(routed, caller)
        except BaseException as error:
            failure = error
            raise
        finally:
            if forked:
                try:
                    self._resources.join(caller)
                    self._joins += 1
                except BaseException:
                    if failure is None:
                        raise
        return routed, shared

    def receipt(self):
        return dict(
            caller=self._caller_key, side=_stream_key(self.side_stream),
            python_calls=self._python_calls, forks=self._forks, joins=self._joins,
            closed=self._closed, storage_signature=self.storage_signature(),
            scope="stream DAG only; WORLD8 MegaMoE progress requires provider validation",
        )

    def close(self):
        """Reject new submissions; the runtime must retain/drain graph users."""
        self._resources.close()


class VerifyMoEOverlapPlans:
    """Capture-caller resources sharing one serially executed Mega workspace.

    Native dtype discovery and capture warmup use different caller streams.
    Each receives its own single-caller plan. This pool does not authorize
    concurrent graph replay: the native execution stream must order complete
    graph invocations, including every captured side-stream join.
    """

    def __init__(self, device, engine, *, target_layers=40, max_rows=32):
        self.device, self.engine = torch.device(device), engine
        self.target_layers, self.max_rows = target_layers, max_rows
        self._anchor = VerifyMoEOverlap(device, engine, target_layers=target_layers, max_rows=max_rows)
        self.device = self._anchor.device
        self._plans = {}
        self._closed = False
        self._prepare_lock = threading.Lock()

    def storage_signature(self):
        """Stable anchor fingerprint; dynamic membership is validated separately."""
        return self._anchor.storage_signature()

    def validate(self):
        if self._closed:
            raise RuntimeError("Verify MoE overlap pool is closed")
        self._anchor.validate()
        for key, plan in self._plans.items():
            if plan.engine is not self.engine or plan._caller_key != key:
                raise RuntimeError("Verify MoE caller plan changed; recapture required")
            if plan is not self._anchor:
                plan.validate()

    def prepare_current_stream(self):
        """Explicit eager warmup only: allocate/bind at most once per caller."""
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("prepare Verify MoE caller before capture")
        if not self._prepare_lock.acquire(blocking=False):
            raise RuntimeError("Verify MoE caller preparation is not reentrant")
        try:
            self.validate()
            stream = torch.cuda.current_stream(self.device)
            key = _stream_key(stream)
            plan = self._plans.get(key)
            if plan is None:
                plan = self._anchor if not self._plans else VerifyMoEOverlap(
                    self.device, self.engine, target_layers=self.target_layers, max_rows=self.max_rows,
                )
                plan.bind_caller(stream)
                self._plans[key] = plan
            else:
                plan.bind_caller(stream)
            return plan
        finally:
            self._prepare_lock.release()

    def lookup_current_stream(self, *, required=False):
        """Lookup without creating resources, including during capture."""
        self.validate()
        plan = self._plans.get(_stream_key(torch.cuda.current_stream(self.device)))
        if plan is None and (required or torch.cuda.is_current_stream_capturing()):
            raise RuntimeError("prepare Verify MoE caller before capture; no plan for this stream")
        return plan

    def receipt(self):
        return dict(
            callers={str(key): plan.receipt() for key, plan in self._plans.items()},
            prepared_callers=len(self._plans), closed=self._closed,
            execution_contract="one native execution stream serializes all graph replays",
        )

    def close(self):
        """Reject new use; the owner still retains/drains captured graph users."""
        if not self._prepare_lock.acquire(blocking=False):
            raise RuntimeError("cannot close active Verify MoE caller preparation")
        try:
            plans = [self._anchor] + [plan for plan in self._plans.values() if plan is not self._anchor]
            if any(plan._lock.locked() for plan in plans):
                raise RuntimeError("cannot close an active Verify MoE caller plan")
            for plan in plans:
                plan.close()
            self._closed = True
        finally:
            self._prepare_lock.release()


VerifyMoEOverlapPool = VerifyMoEOverlapPlans
