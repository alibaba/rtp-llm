"""Authentic epoch-scoped resources, explicit evidence and bounded LIFO cleanup."""

from dataclasses import dataclass

from scenario.contracts import ResourceHandle
from runtime.deadline import Deadline, interruptible


@dataclass
class CleanupAction:
    name: str
    callback: object
    request_stop: object = None
    state: str = "ACTIVE"


class ResourceScope:
    def __init__(self, clock, sleeper, enforce_deadlines):
        self.env_epoch = 0
        self.clock, self.sleeper = clock, sleeper
        self.enforce_deadlines = enforce_deadlines
        self._resources = {}
        self._evidence_exporters = {}
        self._cleanup = []
        self.cleanup_results = []

    def resolve(self, value):
        return value

    def add_cleanup(self, name, callback, *, request_stop=None):
        self._cleanup.append(CleanupAction(name, callback, request_stop))

    @property
    def resource_count(self):
        return len(self._resources)

    def export_evidence(self, profile, deadline):
        from case_api.extension_contracts import ResourceEvidence
        for identity, exporter in self._evidence_exporters.items():
            deadline.check()
            handle, value, _ = self._resources[identity]
            snapshot = exporter(value, profile, deadline)
            if not isinstance(snapshot, ResourceEvidence):
                raise TypeError("resource exporter must return ResourceEvidence")
            deadline.check()
            yield dict(handle), snapshot

    def register_resource(self, kind, value, cleanup=None, historical=False, *, evidence=None, request_stop=None):
        if evidence is not None and not callable(evidence):
            raise TypeError("resource evidence exporter must be callable")
        handle = ResourceHandle(
            kind, f"resource_{len(self._resources) + 1}", self.env_epoch
        ).to_dict()
        self._resources[handle["id"]] = (handle, value, historical)
        if evidence is not None:
            self._evidence_exporters[handle["id"]] = evidence
        if cleanup is not None:
            self.add_cleanup(handle["id"], cleanup, request_stop=request_stop)
        return dict(handle)

    def resource(self, value, kind, allow_stale=False):
        handle = self.resolve(value)
        if not isinstance(handle, dict) or set(handle) != {"kind", "id", "env_epoch"}:
            raise ValueError("expected authentic resource handle")
        record = self._resources.get(handle["id"])
        if record is None or record[0] != handle or handle["kind"] != kind:
            raise ValueError("unknown or forged resource handle")
        if handle["env_epoch"] != self.env_epoch and not (allow_stale and record[2]):
            raise ValueError("stale environment epoch")
        return record[1]

    def cleanup(self, budget_s, retain_failed=True):
        # Instance expiry does not consume the separately reserved cleanup time.
        deadline = Deadline(self.clock() + budget_s, self.clock, self.sleeper)
        results, retry = [], []
        actions = list(reversed(self._cleanup))
        self._cleanup.clear()
        stop_errors = {}
        # Cancellation is a separate sweep: a slow join must not prevent later
        # resources from receiving their stop request. Stop hooks never join.
        for action in actions:
            if action.request_stop is not None:
                try:
                    deadline.check()
                    stop_deadline = Deadline(min(deadline.expires_at, self.clock() +
                        min(.05, deadline.remaining() / (len(actions) + 1))), self.clock, self.sleeper)
                    with interruptible(stop_deadline, self.enforce_deadlines):
                        action.request_stop()
                except Exception as exc:
                    stop_errors[id(action)] = exc
        for action in actions:
            name, callback = action.name, action.callback
            started = self.clock()
            action.state = "STOPPING"
            try:
                deadline.check()
                with interruptible(deadline, self.enforce_deadlines):
                    callback(deadline)
                deadline.check()
                if id(action) in stop_errors:
                    raise stop_errors[id(action)]
                action.state = "CLOSED"
                status, error = "PASS", None
            except Exception as exc:
                action.state = "CLOSE_FAILED"
                status = "TIMEOUT" if isinstance(exc, TimeoutError) else "ERROR"
                error = f"{type(exc).__name__}: {exc}"
                if retain_failed:
                    retry.append(action)
            results.append(
                {
                    "id": name,
                    "status": status,
                    "error": error,
                    "duration_ms": int((self.clock() - started) * 1000),
                }
            )
        # A failed intermediate teardown must remain reachable by the final
        # separately budgeted cleanup; do not retry it in this same loop.
        self._cleanup.extend(reversed(retry))
        self.cleanup_results.extend(results)
        return results
