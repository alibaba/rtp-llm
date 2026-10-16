"""Freeze scenario cohorts after the background sampler has joined."""
import json
import threading

from schema_contract import matches_schema
from adapters.debug_client import DebugUnavailable
from adapters.observation_sources import Sources, MAX_DATASET_BYTES
from runtime.observation import FrozenSnapshot
from runtime.sample_worker import SampleWorker, SamplingError


class Observer(SampleWorker):
    def __init__(self, ctx, params):
        self.ctx = ctx
        self.sources = Sources(ctx, params)
        super().__init__(self.sources.capture, ctx.clock, params)
        self.frozen = None
        self.handle = None
        self.stop_lock = threading.Lock()
        self.artifact = None

    def stop(self, deadline):
        if not self.stop_lock.acquire(timeout=deadline.remaining()):
            raise TimeoutError("concurrent observer stop did not finish")
        try:
            return self._stop(deadline)
        finally:
            self.stop_lock.release()

    def _stop(self, deadline):
        if self.frozen is not None:
            return self.frozen
        try:
            self.stop_sampling(deadline)
        except SamplingError as exc:
            raise DebugUnavailable(str(exc)) from exc
        samples = [json.loads(s) for s in self.samples]
        records = []
        for sample in samples:
            source = sample["sources"].get("client_records")
            if source and source["status"] == "ok":
                records = source["data"]
        # Freeze the pinned record provider after joining: records issued between
        # the last periodic sample and stop must remain in the cohort, including
        # unfinished attempts. Never consult the replacement environment.
        if self.sources.records is not None:
            if (
                self.ctx.env_epoch != self.sources.epoch
                or self.ctx.env is not self.sources.env
            ):
                self.error = "environment changed before cohort freeze"
            else:
                try:
                    records = self.sources.records.snapshot_records()
                    if not isinstance(records, list) or any(
                        not isinstance(r, dict)
                        or not matches_schema(r, "client_record_schema_version", 1)
                        or r.get("env_epoch") != self.sources.epoch
                        or type(r.get("wire_request_id")) is not int
                        or type(r.get("attempt")) is not int
                        for r in records
                    ):
                        raise DebugUnavailable("invalid final client records")
                except Exception as error:
                    self.error = f"cohort freeze: {type(error).__name__}: {error}"
                    records = []
        key = {
            "issued_in_window": lambda r: r.get("issued_s"),
            "submitted_in_window": lambda r: r.get("schedule", {}).get("started_s"),
            "terminal_in_window": lambda r: r.get("transport_terminal_s"),
        }[self.params["cohort"]]
        records = [
            r
            for r in records
            if key(r) is not None and self.started <= key(r) < self.stopped_at
        ]
        data = dict(
            observation_window_schema_version=1,
            phase="frozen",
            env_epoch=self.sources.epoch,
            window=[self.started, self.stopped_at],
            cohort_basis=self.params["cohort"],
            cohort_records=records,
            worker_exit_mono=self.worker_exit_mono,
            background_started=self.thread is not None,
            background_done=self.done_event.is_set(),
            cohort_settle="not_waited",
            samples=samples,
            sample_count=len(samples),
            error=self.error,
            status=(
                "ok"
                if samples
                and not self.error
                and all(s["status"] == "ok" for s in samples)
                else "partial"
            ),
        )
        encoded = json.dumps(data, allow_nan=False)
        if len(encoded.encode()) > MAX_DATASET_BYTES:
            raise DebugUnavailable("frozen cohort exceeds dataset byte budget")
        self.frozen = FrozenSnapshot(encoded)
        return self.frozen
