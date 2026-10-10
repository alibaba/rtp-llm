"""Authentic epoch-scoped resources, explicit evidence and bounded LIFO cleanup."""

from scenario.contracts import ResourceHandle
from runtime.deadline import Deadline, interruptible


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

    def add_cleanup(self, name, callback):
        self._cleanup.append((name, callback))

    @property
    def resource_count(self):
        return len(self._resources)

    def export_evidence(self, profile, deadline):
        from runtime.resource_evidence import ResourceEvidence
        for identity, exporter in self._evidence_exporters.items():
            deadline.check()
            handle, value, _ = self._resources[identity]
            snapshot = exporter(value, profile)
            if not isinstance(snapshot, ResourceEvidence):
                raise TypeError("resource exporter must return ResourceEvidence")
            deadline.check()
            yield dict(handle), snapshot

    def register_resource(self, kind, value, cleanup=None, historical=False, *, evidence=None):
        if evidence is not None and not callable(evidence):
            raise TypeError("resource evidence exporter must be callable")
        handle = ResourceHandle(
            kind, f"resource_{len(self._resources) + 1}", self.env_epoch
        ).to_dict()
        self._resources[handle["id"]] = (handle, value, historical)
        if evidence is not None:
            self._evidence_exporters[handle["id"]] = evidence
        if cleanup is not None:
            self.add_cleanup(handle["id"], cleanup)
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

    def cleanup(self, budget_s, retain_failed=False):
        # Instance expiry does not consume the separately reserved cleanup time.
        deadline = Deadline(self.clock() + budget_s, self.clock, self.sleeper)
        results, retry = [], []
        while self._cleanup:
            name, callback = self._cleanup.pop()
            started = self.clock()
            try:
                # Even an expired callback gets the opportunity to cancel its
                # owned calls/processes before checking its remaining join time.
                if self.clock() < deadline.expires_at:
                    with interruptible(deadline, self.enforce_deadlines):
                        callback(deadline)
                else:
                    callback(deadline)
                deadline.check()
                status, error = "PASS", None
            except Exception as exc:
                status = "TIMEOUT" if isinstance(exc, TimeoutError) else "ERROR"
                error = f"{type(exc).__name__}: {exc}"
                if retain_failed:
                    retry.append((name, callback))
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
