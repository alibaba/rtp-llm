"""Private legacy observation policy for the two KV-skew programs."""

import copy
import json
import threading
from pathlib import Path

from ..observed import ObservedRequestBatch
from ..runtime import Deadline, RuntimeContext, StageTimeout
from .elastic import BoundedFlow, RecordedRequests, request_success


class FixedRequestOps:
    def __init__(self, ops, rid):
        self._ops, self._rid, self._issued = ops, rid, False

    def next_request_id(self):
        if self._issued:
            raise RuntimeError("skew child attempted more than its one request")
        self._issued = True
        return self._rid

    def __getattr__(self, name):
        return getattr(self._ops, name)


def legacy_success(record):
    error = record.get("execution_error")
    if error:
        if error["kind"] == "StageTimeout":
            raise StageTimeout(error["detail"])
        raise RuntimeError(error["detail"])
    observation = record.get("rpc_observation", {})
    if observation.get("complete") is not True:
        raise ValueError("skew request lacks completed caller observation")
    if observation.get("schedule_failed") is True:
        if (
            observation.get("legacy_success") is not False
            or record["schedule"]["status"] == "OK"
            or record["stream"]["method"] is not None
            or record.get("transport_terminal_s") is None
            or not record.get("request_error")
        ):
            raise ValueError("invalid skew Schedule failure evidence")
        return False
    if observation.get("schedule_rejected") is not True and (
        record.get("consumer_done") is not True
        or record.get("consumer_completion_verified") is not True
        or record.get("consumer_exit_s") is None
        or record.get("transport_terminal_s") is None
    ):
        raise ValueError("skew request lacks consumer exit evidence")
    if type(observation.get("legacy_success")) is not bool:
        raise ValueError("skew legacy success is unavailable")
    return observation["legacy_success"]


def summary(records, *, raise_errors=True):
    errors, completed = [], 0
    for record in records:
        try:
            completed += int(legacy_success(record))
        except Exception as exc:
            errors.append(dict(request_id=record["wire_request_id"], error=repr(exc)))
            if raise_errors:
                raise
    missing = [
        r["wire_request_id"]
        for r in records
        if not r.get("rpc_observation", {}).get("complete")
    ]
    failed = [
        r["wire_request_id"]
        for r in records
        if r.get("rpc_observation", {}).get("legacy_success") is not True
    ]
    return dict(
        issued=len(records),
        completed=completed,
        legacy_completed=completed,
        legacy_error_count=len(records) - completed,
        sample_count=len(records),
        min_samples=1,
        complete=bool(records) and not missing and not errors,
        result_complete=bool(records) and not missing and not errors,
        zero_errors=bool(records) and not failed and not errors,
        failed_request_ids=failed,
        incomplete_request_ids=missing,
        execution_errors=errors,
        strict_transport_success=sum(request_success(r) for r in records),
        success_policy="legacy run_one_request plus verified consumer exit",
    )


class SkewRecordedRequests(RecordedRequests):
    def __init__(self, ctx, phase, end_wait_s, phase_deadline_s=None):
        if phase not in {"seed", "pump", "recovery"}:
            raise ValueError("unknown skew request phase")
        super().__init__(ctx.ops, ctx.env_epoch, ctx.clock)
        self.parent_ctx, self.phase, self.end_wait_s = ctx, phase, end_wait_s
        self.active_children, self.worker_done, self.started_rids = {}, {}, set()
        self.phase_deadline_s = phase_deadline_s or ctx.instance_deadline_s
        self.artifact_root = Path(ctx.artifact_dir) / "skew" / phase
        self.artifact_root.mkdir(parents=True, exist_ok=False)

    def issue(self, rid, clock):
        record = super().issue(rid, clock)
        with self._lock:
            record.update(rpc_observation={}, transport_record={})
            self.worker_done[rid] = threading.Event()
        return record

    def run(self, record, shape, timeout_s=125):
        rid, parent = record["wire_request_id"], self.parent_ctx
        child, batch, deadline = None, None, None
        error = request_error = failure_observation = None
        try:
            with self._lock:
                self.started_rids.add(rid)
                if self._cancelled:
                    raise RuntimeError(
                        "skew aggregate cancelled before child registration"
                    )
            if parent.env_epoch != self.env_epoch:
                raise ValueError("skew request belongs to an old environment epoch")
            directory = self.artifact_root / str(rid)
            directory.mkdir(exist_ok=False)
            child = RuntimeContext(
                {}, parent.backend, directory, self.clock, parent.sleeper
            )
            child.env_epoch = self.env_epoch
            child.ops = FixedRequestOps(self.ops, rid)
            child.instance_deadline_s = min(
                parent.instance_deadline_s,
                self.phase_deadline_s,
                self.clock() + timeout_s,
            )
            deadline = Deadline(child.instance_deadline_s, self.clock, parent.sleeper)
            batch = ObservedRequestBatch(
                child,
                dict(
                    shape,
                    count=1,
                    consume="immediate",
                    schedule_timeout_s=30,
                    stream_timeout_s=60,
                ),
                "request_total",
                deadline,
                end_wait_s=self.end_wait_s,
            )
            with self._lock:
                self.active_children[rid] = batch
                cancelled = self._cancelled
            if cancelled:
                batch.cancel("skew aggregate already cancelled")
            batch.submit(deadline)
        except Exception as exc:
            error = dict(kind=type(exc).__name__, detail=str(exc) or repr(exc))
            if batch is not None:
                import grpc

                rows = batch.snapshot_records()
                if isinstance(exc, (grpc.FutureTimeoutError, grpc.RpcError)) and rows:
                    raw = rows[0]
                    if self.clock() >= deadline.expires_at:
                        error = dict(
                            kind="StageTimeout",
                            detail="skew parent deadline expired during Schedule",
                        )
                    elif (
                        not self._cancelled
                        and raw["schedule"]["status"] != "OK"
                        and raw["schedule"]["ended_s"] is not None
                        and raw["stream"]["method"] is None
                        and batch.entries[0]["thread"] is None
                    ):
                        # Legacy run_one_request returns this as a failed
                        # request. Preserve raw transport evidence and keep
                        # subsequent recovery/verdict stages reachable.
                        request_error, error = error, None
                        failure_observation = dict(
                            mode="request_total",
                            complete=True,
                            legacy_success=False,
                            schedule_failed=True,
                            operation_started_s=raw["issued_s"],
                            operation_ended_s=self.clock(),
                        )
        finally:
            if batch is not None:
                try:
                    # This is cleanup, never additional business-observation time.
                    batch.cleanup(
                        Deadline(self.clock() + 5, self.clock, parent.sleeper)
                    )
                except Exception as exc:
                    error = dict(
                        kind=type(exc).__name__, detail=f"child cleanup: {exc}"
                    )
                rows = batch.snapshot_records()
                if rows:
                    transport = rows[0]
                    with self._lock:
                        original_issued = record["issued_s"]
                        record.update(copy.deepcopy(transport))
                        record["issued_s"] = original_issued
                        record["transport_record"] = copy.deepcopy(transport)
            with self._lock:
                if request_error:
                    record["request_error"] = request_error
                    record["rpc_observation"] = failure_observation
                if error:
                    record["execution_error"] = error
            try:
                if child is not None:
                    (child.artifact_dir / "aggregate-record.json").write_text(
                        json.dumps(record, indent=2)
                    )
            except Exception as exc:
                with self._lock:
                    record["execution_error"] = dict(
                        kind=type(exc).__name__, detail=f"archive: {exc}"
                    )
            finally:
                with self._lock:
                    self.active_children.pop(rid, None)
                    self.worker_done[rid].set()

    def cancel_active(self, reason="cleanup"):
        with self._lock:
            self._cancelled = True
            children = list(self.active_children.values())
        for child in children:
            child.cancel(reason)

    def cleanup(self, deadline):
        self.cancel_active("skew cleanup")
        with self._lock:
            children = list(self.active_children.values())
            for record in self._records:
                rid = record["wire_request_id"]
                if rid not in self.started_rids:
                    record["execution_error"] = dict(
                        kind="RuntimeError", detail="cancelled before worker start"
                    )
                    self.worker_done[rid].set()
            events = list(self.worker_done.values())
        for child in children:
            child.cleanup(deadline)
        for event in events:
            while not event.is_set():
                event.wait(min(0.1, deadline.remaining()))
        deadline.check()
        (self.artifact_root / "records.json").write_text(
            json.dumps(self.snapshot_records(), indent=2)
        )


class SkewFlow(BoundedFlow):
    def __init__(self, ctx, families):
        super().__init__(ctx.ops, ctx.env_epoch, families, clock=ctx.clock)
        self.observed = SkewRecordedRequests(ctx, "pump", 30)

    def issue(self, rid, clock):
        return self.observed.issue(rid, clock)

    def run(self, record, shape, *args, **kwargs):
        return self.observed.run(record, shape, *args, **kwargs)

    def snapshot_records(self):
        return self.observed.snapshot_records()

    def cancel_active(self, reason="cleanup"):
        return self.observed.cancel_active(reason)

    def stop(self, deadline, cancel=False):
        self._stop.set()
        if cancel:
            self.cancel_active("skew flow cleanup")
        if self.thread.ident is not None:
            if not self.done.wait(deadline.remaining()):
                self.cancel_active("skew flow stop deadline")
                raise StageTimeout("skew pump did not finish in its stop window")
            self.thread.join(deadline.remaining())
            if self.thread.is_alive():
                raise RuntimeError("skew pump exit signal preceded actual thread exit")
        if self.pump_error and not cancel:
            raise RuntimeError(self.pump_error)
        if cancel:
            self.observed.cleanup(deadline)
        return summary(self.snapshot_records(), raise_errors=not cancel)
