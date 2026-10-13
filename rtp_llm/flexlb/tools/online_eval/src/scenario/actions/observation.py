"""Bounded, epoch-aware read-only scenario observations; no driver or global registry."""

from __future__ import annotations

import json
import math
import uuid
from pathlib import Path
from adapters.observation_sources import Sources, SOURCES, MAX_DATASET_BYTES
from adapters.observation_window import Observer

from adapters.debug_client import DebugUnavailable
from scenario.contracts import CheckResult, StageHandler, StageOutput

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
    available = set(SOURCES)
    if plan.case:
        from case_api.registry import registry
        available.update(registry().cases[plan.case].definition.snapshot_sources)
    if (
        not isinstance(sources, list)
        or not sources
        or any(not isinstance(s, str) or s not in available for s in sources)
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
