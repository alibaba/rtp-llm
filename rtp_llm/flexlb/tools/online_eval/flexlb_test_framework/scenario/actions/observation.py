"""Bounded, epoch-aware read-only scenario observations; no driver or global registry."""

from __future__ import annotations

import json
import math
import threading
import time
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path

from ...debug_client import DebugClient, DebugUnavailable
from ..contracts import CheckResult, StageHandler, StageOutput

SOURCES = frozenset(
    {"master_debug", "engine_snapshot", "engine_requests", "client_records"}
)
SOURCE_STATUSES = frozenset({"ok", "partial", "error", "epoch_changed"})
MAX_SOURCE_BYTES = 8 * 1024 * 1024
MAX_DATASET_BYTES = 32 * 1024 * 1024


def _error(plan, detail):
    raise ValueError(f"{plan.path}: {detail}")


def _number(value, plan, name, minimum, maximum, integer=False):
    types = (int,) if integer else (int, float)
    if (
        type(value) not in types
        or not math.isfinite(value)
        or not minimum <= value <= maximum
    ):
        _error(plan, f"{name} must be {minimum}..{maximum}")
    return value


def validate_snapshot(params, plan):
    allowed = {
        "sources",
        "required",
        "flow",
        "requests",
        "targets",
        "master_include",
        "limit",
        "scan_limit",
        "endpoint_limit",
    }
    if not isinstance(params, dict) or set(params) - allowed:
        _error(plan, "unknown snapshot parameters")
    sources = params.get("sources")
    if (
        not isinstance(sources, list)
        or not sources
        or any(not isinstance(s, str) or s not in SOURCES for s in sources)
    ):
        _error(plan, "sources must be a nonempty supported source list")
    if len(set(sources)) != len(sources):
        _error(plan, "duplicate sources")
    result = dict(params, sources=list(sources))
    if type(params.get("required", True)) is not bool:
        _error(plan, "required must be boolean")
    result["required"] = params.get("required", True)
    if "requests" in params and "flow" in params:
        _error(plan, "choose requests or flow, not both")
    for kind in ("requests", "flow"):
        if kind in params:
            plan.reference(params[kind], kind)
    if "client_records" in sources and not ({"requests", "flow"} & params.keys()):
        _error(plan, "client_records requires a requests or flow reference")
    targets = params.get("targets", [])
    if not isinstance(targets, list) or any(
        not isinstance(x, str) or not x for x in targets
    ):
        _error(plan, "targets must contain engine names")
    if len(targets) > 256 or len(set(targets)) != len(targets):
        _error(plan, "too many or duplicate targets")
    result["targets"] = list(targets)
    include = params.get(
        "master_include", ["scheduler", "queues", "prefill", "decode", "engine"]
    )
    if (
        not isinstance(include, list)
        or not include
        or any(
            not isinstance(x, str)
            or x not in {"scheduler", "queues", "prefill", "decode", "engine"}
            for x in include
        )
    ):
        _error(plan, "invalid master_include")
    result["master_include"] = list(include)
    result["limit"] = _number(params.get("limit", 500), plan, "limit", 1, 5000, True)
    result["scan_limit"] = _number(
        params.get("scan_limit", 2000), plan, "scan_limit", result["limit"], 20000, True
    )
    result["endpoint_limit"] = _number(
        params.get("endpoint_limit", 64), plan, "endpoint_limit", 1, 256, True
    )
    return result


def validate_observe(params, plan):
    if not isinstance(params, dict):
        _error(plan, "observe params must be a mapping")
    mode = params.get("mode", "window")
    if mode == "stop":
        if set(params) - {"mode", "observation"} or "observation" not in params:
            _error(plan, "stop requires only an observation reference")
        plan.reference(params["observation"], "observation")
        return dict(params, mode=mode)
    if mode not in {"start", "window"}:
        _error(plan, "observe mode must be start/window/stop")
    extras = {
        "mode",
        "interval_s",
        "duration_s",
        "max_duration_s",
        "max_samples",
        "max_bytes",
        "cohort",
    }
    base = validate_snapshot({k: v for k, v in params.items() if k not in extras}, plan)
    duration_key = "duration_s" if mode == "window" else "max_duration_s"
    forbidden = "max_duration_s" if mode == "window" else "duration_s"
    if forbidden in params:
        _error(plan, f"{mode} does not accept {forbidden}")
    duration = _number(params.get(duration_key), plan, duration_key, 0.05, 3600)
    interval = _number(params.get("interval_s", 0.5), plan, "interval_s", 0.05, 60)
    cohort = params.get("cohort", "issued_in_window")
    if cohort not in {"issued_in_window", "submitted_in_window", "terminal_in_window"}:
        _error(plan, "unsupported cohort basis")
    return dict(
        base,
        mode=mode,
        duration_s=duration,
        interval_s=interval,
        cohort=cohort,
        max_samples=_number(
            params.get("max_samples", 10000), plan, "max_samples", 1, 10000, True
        ),
        max_bytes=_number(
            params.get("max_bytes", MAX_DATASET_BYTES),
            plan,
            "max_bytes",
            1024,
            MAX_DATASET_BYTES,
            True,
        ),
    )


@dataclass(frozen=True)
class FrozenSnapshot:
    """Serialized canonical payload makes returned dictionaries independent deep copies."""

    encoded: str

    def to_dict(self):
        return json.loads(self.encoded)


def _read_json(url, deadline):
    deadline.check()
    with urllib.request.urlopen(
        url, timeout=min(5.0, deadline.remaining())
    ) as response:
        body = response.read(MAX_SOURCE_BYTES + 1)
    if len(body) > MAX_SOURCE_BYTES:
        raise DebugUnavailable("source response exceeds byte budget")
    return json.loads(body)


class Sources:
    def __init__(self, ctx, params):
        self.ctx, self.params = ctx, params
        self.epoch = ctx.env_epoch
        self.env = ctx.env
        if self.env is None:
            raise DebugUnavailable("snapshot requires setup")
        self.master_port = self.env.master_http_port
        self.mock_port = self.env.mock_http_port
        self.records = None
        for kind in ("requests", "flow"):
            if kind in params:
                self.records = ctx.resource(params[kind], kind)
                if not callable(getattr(self.records, "snapshot_records", None)):
                    raise DebugUnavailable(
                        f"{kind} does not implement snapshot_records"
                    )
        self.master_instance = None
        self.initial_members = {}

    def capture(self, deadline):
        started = self.ctx.clock()
        sources = {}
        for name in self.params["sources"]:
            deadline.check()
            sample = dict(
                source=name,
                env_epoch=self.epoch,
                started_mono=self.ctx.clock(),
                started_epoch_ms=int(time.time() * 1000),
                status="ok",
                errors=[],
                data=None,
            )
            try:
                if self.ctx.env_epoch != self.epoch or self.ctx.env is not self.env:
                    sample["status"] = "epoch_changed"
                    raise DebugUnavailable(
                        "environment changed; old cohort is not sampled from the new environment"
                    )
                if name == "master_debug":
                    result = DebugClient(
                        f"http://127.0.0.1:{self.master_port}",
                        timeout_s=min(5.0, deadline.remaining()),
                    ).snapshot(
                        include=",".join(self.params["master_include"]),
                        limit=self.params["limit"],
                        scan_limit=self.params["scan_limit"],
                        endpoint_limit=self.params["endpoint_limit"],
                    )
                    sample["data"] = result.payload
                    instance = result.payload["instanceId"]
                    sample["master_instance_id"] = instance
                    if self.master_instance is None:
                        self.master_instance = instance
                    elif self.master_instance != instance:
                        sample["status"] = "epoch_changed"
                        raise DebugUnavailable(
                            "Master instance changed; previous baseline is no longer comparable"
                        )
                    if result.payload["status"] != "ok" or any(
                        page["status"] != "ok" or page["truncated"]
                        for page in result.payload["components"].values()
                    ):
                        sample["status"] = "partial"
                        raise DebugUnavailable(
                            "Master snapshot has partial/busy/truncated components"
                        )
                elif name == "client_records":
                    records = self.records.snapshot_records()
                    if not isinstance(records, list) or any(
                        not isinstance(r, dict)
                        or r.get("schema_version") != 1
                        or r.get("env_epoch") != self.epoch
                        or type(r.get("wire_request_id")) is not int
                        or type(r.get("attempt")) is not int
                        for r in records
                    ):
                        raise DebugUnavailable("invalid or stale client_records v1")
                    sample["data"] = records  # Includes every nonterminal attempt.
                else:
                    route = "snapshot" if name == "engine_snapshot" else "requests"
                    data = _read_json(
                        f"http://127.0.0.1:{self.mock_port}/{route}", deadline
                    )
                    if name == "engine_snapshot":
                        if not isinstance(data, dict) or not isinstance(
                            data.get("engines"), list
                        ):
                            raise DebugUnavailable("missing engines snapshot")
                        members = {e["name"] for e in data["engines"]}
                    else:
                        if not isinstance(data, dict) or any(
                            not isinstance(v, dict) for v in data.values()
                        ):
                            raise DebugUnavailable("invalid engine requests snapshot")
                        members = set(data)
                    initial = self.initial_members.setdefault(name, frozenset(members))
                    sample["membership"] = dict(
                        initial=sorted(initial),
                        present=sorted(members),
                        added=sorted(members - initial),
                        removed=sorted(initial - members),
                    )
                    sample["engine_generation"] = (
                        None  # Mock HTTP does not provide restart identity.
                    )
                    sample["targets"] = self.params["targets"]
                    sample["data"] = data
                    missing = set(self.params["targets"]) - members
                    if missing:
                        sample["status"] = "partial"
                        raise DebugUnavailable(
                            f"requested engines missing: {sorted(missing)}"
                        )
            except Exception as error:
                if sample["status"] == "ok":
                    sample["status"] = "error"
                sample["errors"].append(
                    dict(type=type(error).__name__, message=str(error))
                )
            sample.update(
                finished_mono=self.ctx.clock(),
                finished_epoch_ms=int(time.time() * 1000),
            )
            sources[name] = sample
        payload = dict(
            schema_version=1,
            env_epoch=self.epoch,
            started_mono=started,
            finished_mono=self.ctx.clock(),
            sources=sources,
            status=(
                "ok"
                if all(s["status"] == "ok" for s in sources.values())
                else "partial"
            ),
        )
        encoded = json.dumps(payload, allow_nan=False)
        if len(encoded.encode()) > MAX_DATASET_BYTES:
            raise DebugUnavailable("snapshot exceeds dataset byte budget")
        return FrozenSnapshot(encoded)


def _artifact(ctx, name, data):
    path = Path(ctx.artifact_dir) / f"{name}-{uuid.uuid4().hex}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    return str(path)


def _coverage(required, data, artifact=None):
    return [
        CheckResult(
            "sources",
            "PASS" if not required or data["status"] == "ok" else "ERROR",
            (
                "required sources must be complete"
                if required
                else "optional observation recorded; PASS does not assert source availability or completeness"
            ),
            actual={
                "required": required,
                "source_status": data["status"],
                "sample_count": data.get("sample_count"),
            },
            evidence={"artifact": artifact} if artifact else {},
        )
    ]


def execute_snapshot(ctx, params, deadline):
    frozen = Sources(ctx, params).capture(deadline)
    handle = ctx.register_resource("snapshot", frozen, historical=True)
    data = frozen.to_dict()
    artifact = _artifact(ctx, "snapshot", data)
    checks = _coverage(params["required"], data, artifact)
    return StageOutput(output={"snapshot": handle}, checks=checks, artifacts=[artifact])


class CaptureDeadline:
    def __init__(self, clock, end, cancel):
        self.clock, self.end, self.cancel = clock, end, cancel

    def remaining(self):
        return max(0, self.end - self.clock())

    def check(self):
        if self.cancel.is_set() or self.remaining() <= 0:
            raise TimeoutError("observation deadline or cancellation")


class Observer:
    def __init__(self, ctx, params):
        self.ctx, self.params = ctx, params
        self.sources = Sources(ctx, params)
        self.started = ctx.clock()
        self.end = self.started + params["duration_s"]
        self.stop_event = threading.Event()
        self.done_event = threading.Event()
        self.worker_exit_mono = None
        self.thread = None
        self.samples = []
        self.bytes = 0
        self.error = None
        self.frozen = None
        self.stopped_at = None
        self.handle = None
        self.stop_lock = threading.Lock()
        self.artifact = None

    def append(self, sample):
        size = len(sample.encoded.encode())
        if (
            len(self.samples) >= self.params["max_samples"]
            or self.bytes + size > self.params["max_bytes"]
        ):
            self.error = "observation sample/byte budget exhausted"
            self.stop_event.set()
            return False
        self.samples.append(sample.encoded)
        self.bytes += size
        return True

    def start(self, deadline):
        self.append(self.sources.capture(deadline))
        self.thread = threading.Thread(
            target=self._run, name="flexlb-scenario-observer", daemon=True
        )
        self.thread.start()

    def _run(self):
        try:
            while not self.stop_event.wait(
                min(self.params["interval_s"], max(0, self.end - self.ctx.clock()))
            ):
                if self.ctx.clock() >= self.end:
                    break
                self.append(
                    self.sources.capture(
                        CaptureDeadline(self.ctx.clock, self.end, self.stop_event)
                    )
                )
        except Exception as error:
            self.error = f"incomplete sampling: {type(error).__name__}: {error}"
        finally:
            # Publish the exit record after the last possible sample/error write.
            self.worker_exit_mono = self.ctx.clock()
            self.done_event.set()

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
        self.stopped_at = (
            min(self.ctx.clock(), self.end)
            if self.stopped_at is None
            else self.stopped_at
        )
        self.stop_event.set()
        if self.thread is not None:
            while not self.done_event.is_set():
                deadline.check()
                self.done_event.wait(min(0.05, deadline.remaining()))
            if self.worker_exit_mono is None:
                raise DebugUnavailable("observer completion signal has no exit record")
            self.thread.join(timeout=deadline.remaining())
            if self.thread.is_alive():
                raise TimeoutError(
                    "observer thread still alive; cleanup has not completed"
                )
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
                        or r.get("schema_version") != 1
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
            schema_version=1,
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


def execute_observe(ctx, params, deadline):
    if params["mode"] == "stop":
        observer = ctx.resource(params["observation"], "observation", allow_stale=True)
        if not isinstance(observer, Observer):
            raise DebugUnavailable("observation handle does not belong to this adapter")
        frozen = observer.stop(deadline)
    else:
        observer = Observer(ctx, params)
        observer.handle = ctx.register_resource(
            "observation", observer, cleanup=observer.stop, historical=True
        )
        observer.start(deadline)
        if params["mode"] == "start":
            initial = (
                json.loads(observer.samples[0])
                if observer.samples
                else {"status": "partial"}
            )
            checks = _coverage(params["required"], initial)
            return StageOutput(output={"observation": observer.handle}, checks=checks)
        while not observer.done_event.is_set():
            deadline.check()
            observer.done_event.wait(timeout=min(0.1, deadline.remaining()))
        frozen = observer.stop(deadline)
    data = frozen.to_dict()
    if observer.artifact is None:
        observer.artifact = _artifact(ctx, "observation", data)
    artifact = observer.artifact
    checks = _coverage(observer.params["required"], data, artifact)
    return StageOutput(
        output={"observation": observer.handle}, checks=checks, artifacts=[artifact]
    )


HANDLERS = [
    StageHandler(
        "snapshot",
        validate_snapshot,
        execute_snapshot,
        {"snapshot": "snapshot"},
        checks=frozenset({"sources"}),
    ),
    StageHandler(
        "observe",
        validate_observe,
        execute_observe,
        {"observation": "observation"},
        checks=frozenset({"sources"}),
    ),
]
CHECK_HANDLERS = []
