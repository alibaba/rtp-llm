"""Map Native connector load identities to the in-process Agent lifecycle.

Native owns worker counting and transfer failures. This adapter keeps one Agent
record per request; it neither enumerates shards nor communicates with workers.
"""

from dataclasses import dataclass
import threading
import time


@dataclass
class _AgentLoad:
    request_id: int
    session: int
    generation: int
    state: str = "PREPARING"
    native_load: object = None


class VLLMKVLoadReadiness:
    def __init__(self, agent, scheduler, request_id, lookup_complete, emit):
        self._agent = agent
        self._scheduler = scheduler
        self._request_id = request_id
        self._lookup_complete = lookup_complete
        self._emit = emit
        self._loads = {}
        self._lock = threading.RLock()
        self._closed = False
        self.source = None
        self.executor = None

    def bind(self, executor, connector):
        enable = getattr(executor, "enable_kv_load_notifications", None)
        if enable is None:
            return False
        connectors = [connector]
        sources = []
        while connectors:
            child = connectors.pop()
            if callable(getattr(child, "set_load_completion_observer", None)):
                sources.append(child.scheduler)
            else:
                connectors.extend(getattr(child, "_connectors", ()))
        if not sources:
            return False
        if len(sources) != 1:
            raise RuntimeError("KV load reporting requires one Native load source")
        if not enable(connector, self._on_started, self._on_finished):
            return False
        self.source, self.executor = sources[0], executor
        return True

    def _begin(self, external_id, request):
        if self._closed:
            return None
        request_id = self._request_id(request)
        if type(request_id) is not int or request_id <= 0:
            raise ValueError("KV loading requires the canonical Navi request ID")
        key = self._agent.begin_kv_preparation(request_id, time.monotonic_ns())
        if key is None:
            raise RuntimeError("Agent rejected a Native KV preparation")
        session, generation = key
        load = _AgentLoad(request_id, session, generation)
        self._loads[external_id] = load
        self._emit(dict(event="kv_load_preparing", request_id=external_id,
                        navi_request_id=request_id, agent_session=session,
                        load_generation=generation, event_time_us=time.time_ns() // 1000))
        return load

    def admitted(self, request, included):
        if included is not None and type(included) is not bool:
            raise TypeError("execution_load_included must be bool or absent")
        if included is False:
            with self._lock:
                external_id = str(request.request_id)
                if external_id not in self._loads:
                    self._begin(external_id, request)

    def _on_started(self, native_load):
        external_id = native_load.request_id
        with self._lock:
            request = self._scheduler.requests.get(external_id)
            if self._closed or request is None:
                return False
            load = self._loads.get(external_id)
            if load is not None and load.native_load == native_load:
                return True
            if load is not None and (load.native_load is not None or load.state != "PREPARING"):
                self._retire(external_id)
                load = None
            if load is None:
                load = self._begin(external_id, request)
            if load is None:
                return False
            load.native_load = native_load
            return True

    def _on_finished(self, native_load, success):
        if type(success) is not bool:
            raise TypeError("Native completion requires an explicit success bool")
        with self._lock:
            load = self._loads.get(native_load.request_id)
            if self._closed or load is None or load.native_load != native_load:
                return
            if not self._agent.complete_kv_preparation(
                load.session, load.request_id, load.generation, success, time.monotonic_ns()
            ):
                raise RuntimeError("Agent rejected the current Native KV completion")
            load.state = "READY" if success else "FAILED"
            self._emit(dict(event="kv_load_ready" if success else "kv_load_failed",
                request_id=native_load.request_id, navi_request_id=load.request_id,
                agent_session=load.session, load_generation=load.generation,
                native_engine_id=native_load.engine_id, native_load_index=native_load.load_index,
                event_time_us=time.time_ns() // 1000))

    def make_record(self, request, contribution, running_capacity, initial_included, module):
        external_id = str(request.request_id)
        with self._lock:
            native = self.source._request_states.get(external_id)
            pending = native is not None and bool(
                native.load_inflight or native.pending_load_meta
                or (native.resolved_match and native.resolved_match[1])
            )
            status = getattr(getattr(request, "status", None), "name", "")
            load = self._loads.get(external_id)
            # Native RUNNING after its load state is cleared proves recomputation
            # has resumed, even if the earlier external load failed.
            if status == "RUNNING" and not pending:
                state = "EXECUTING"
            elif load is not None and load.state in ("READY", "FAILED"):
                state = load.state
            elif pending or status == "WAITING_FOR_REMOTE_KVS":
                if load is None:
                    load = self._begin(external_id, request)
                state = "PREPARING"
            elif self._lookup_complete(request):
                # A predicted external hit can resolve to HBM or a miss. There
                # is then no worker load whose completion we should wait for.
                if load is not None and load.native_load is None:
                    self._retire(external_id)
                    load = None
                state = "READY"
            elif initial_included is False:
                if load is None:
                    load = self._begin(external_id, request)
                state = "PREPARING"
            elif initial_included is True:
                state = "READY"
            else:
                # Unknown input cannot be turned into an executable default.
                return None
            record = module.NaviSchedKVPreparation()
            record.request_id = self._request_id(request)
            record.local_batch_id = contribution.batch_id
            record.context_tokens = contribution.prefill_context_tokens
            record.compute_tokens = contribution.remaining_prefill_tokens
            record.running_capacity = running_capacity
            record.load_generation = load.generation if load is not None else 0
            record.state = getattr(module.NativeKVExecutionState, state)
            return record

    def _retire(self, external_id):
        load = self._loads.pop(external_id, None)
        if load is None:
            return
        if load.state == "PREPARING":
            self._agent.cancel_kv_preparation(load.session, load.request_id, load.generation)
        self._agent.retire_kv_preparation(load.session, load.request_id, load.generation)

    @staticmethod
    def make_prepared_record(request_id, contribution, running_capacity, included, module):
        # These requests have not entered Native yet. Keep their dispatch hint,
        # but do not invent a Native load generation or a measured KV latency.
        if included is None:
            return None
        if type(included) is not bool:
            raise TypeError("Prepared request inclusion must be explicit bool")
        record = module.NaviSchedKVPreparation()
        record.request_id = request_id
        record.local_batch_id = contribution.batch_id
        record.context_tokens = contribution.prefill_context_tokens
        record.compute_tokens = contribution.remaining_prefill_tokens
        record.running_capacity = running_capacity
        record.load_generation = 0
        record.state = module.NativeKVExecutionState.READY if included else module.NativeKVExecutionState.PREPARING
        return record

    def forget(self, external_id):
        with self._lock:
            self._retire(str(external_id))

    def forget_removed_requests(self, native_requests):
        # Normal subscribed/non-PP requests can have no forwarded batch record.
        # The Native request table is authoritative for their final cleanup too.
        with self._lock:
            for external_id in tuple(self._loads):
                if external_id not in native_requests:
                    self._retire(external_id)

    def close(self):
        with self._lock:
            self._closed = True
            for external_id in tuple(self._loads):
                self._retire(external_id)


def forwarded_execution_load_flags(native_batch, request_ids):
    """Read the full Native batch before selecting decoded/remaining requests."""
    batch_ids = tuple(native_batch.request_ids)
    values = tuple(native_batch.execution_load_included)
    if len(batch_ids) != len(values) or len(set(batch_ids)) != len(batch_ids):
        raise ValueError("Forwarded execution load flags do not match request IDs")
    if any(value is not None and type(value) is not bool for value in values):
        raise TypeError("Forwarded execution load flag must be bool or absent")
    by_id = dict(zip(batch_ids, values))
    return {request_id: by_id[request_id] for request_id in request_ids}
