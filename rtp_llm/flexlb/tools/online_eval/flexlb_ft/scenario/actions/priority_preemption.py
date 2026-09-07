"""Explicit preemption predicates over owned cohorts and engine observations."""

import json
import uuid

from ..contracts import CheckResult, StageHandler, StageOutput
from .elastic import request_success
from .engine_control import _engines, _http
from .priority import PriorityWave, _number, _params


def _cohort(ctx, handle):
    wave = ctx.resource(handle, "requests")
    if not isinstance(wave, PriorityWave):
        raise ValueError("preemption expects an owned priority cohort")
    return wave


def _settled_params(p, plan):
    p = _params(p, {"requests"}, {"requests"})
    plan.reference(p["requests"], "requests")
    return p


def _settled(ctx, p, deadline):
    wave = _cohort(ctx, p["requests"])
    for entry in wave.entries:
        wave._join(entry, deadline)
        if entry["error"] is not None:
            raise entry["error"]
    wave.persist()
    rows = wave.records()
    if len(rows) != len(wave.entries):
        raise ValueError("missing settled Schedule evidence")
    return StageOutput(
        {"admitted": bool(rows) and all(r["schedule"]["status"] == "OK" for r in rows)},
        artifacts=[str(wave.path)],
    )


def _pending_params(p, plan):
    p = _params(p, {"target"}, {"target"})
    if isinstance(p["target"], dict):
        plan.reference(p["target"], "string")
    elif not isinstance(p["target"], str) or not p["target"]:
        raise ValueError("explicit Prefill owner required")
    return p


def _pending(ctx, p, deadline):
    target = ctx.resolve(p["target"])
    samples = []
    path = ctx.artifact_dir / f"preemption-pending-{uuid.uuid4().hex}.json"
    try:
        while True:
            deadline.check()
            raw = _http(ctx.ops, "snapshot", deadline)
            row = _engines(raw, [target])[target]
            if row.get("role") != "prefill":
                raise ValueError("pending precondition must observe Prefill")
            pending = sum(
                _number(row.get(k), 0, 1e9, True) for k in ("waiting", "running")
            )
            samples.append(
                dict(time_s=ctx.clock(), target=target, pending=pending, raw=raw)
            )
            if pending >= 1:
                break
            deadline.sleep(0.1)
    finally:
        path.write_text(json.dumps(samples, indent=2) + "\n")
    return StageOutput(
        checks=[CheckResult("pending", "PASS", actual=pending, expected=">=1")],
        artifacts=[str(path)],
    )


def _same_params(p, plan):
    p = _params(p, {"placeholder", "wave"}, {"placeholder", "wave"})
    for key in p:
        plan.reference(p[key], "requests")
    return p


def _outcome(entry, row):
    if request_success(row):
        return True, 200
    response = entry["batch"].entries[0]["response"]
    if response is not None and (response.code != 200 or not response.success):
        return False, int(response.code)
    # This first program requires every wave peer to succeed. Any stream
    # failure fails all_ok regardless of its code; full raw trailer policy is
    # reserved for the later victim-terminal programs, not claimed here.
    code = row.get("business_error_code")
    return False, 8429 if code == 2 else code


def _observations(ctx, p, deadline):
    placeholder, wave = [_cohort(ctx, p[key]) for key in ("placeholder", "wave")]
    if not placeholder.complete or not wave.complete:
        raise ValueError("preemption verdict requires drained owned cohorts")
    ph, rows = placeholder.records(), wave.records()
    if len(ph) != 1 or len(rows) != 9 or len(wave.entries) != 9:
        raise ValueError("preemption program needs placeholder plus nine peers")
    ordered = sorted(
        ph + rows, key=lambda r: (r["schedule"]["ended_s"], r["wire_request_id"])
    )
    ranks = {r["wire_request_id"]: i for i, r in enumerate(ordered)}
    raw = _http(ctx.ops, "snapshot", deadline)
    engines = raw.get("engines")
    if not isinstance(engines, list):
        raise ValueError("missing Prefill lifecycle snapshot")
    dispatch = []
    for row in rows:
        rid = row["wire_request_id"]
        matches = [
            e.get("request_lifecycle", {}).get(str(rid))
            for e in engines
            if e.get("role") == "prefill"
        ]
        matches = [m for m in matches if m is not None]
        if len(matches) != 1:
            raise ValueError("missing or ambiguous Prefill lifecycle")
        running = _number(matches[0].get("running_ms"), 0, 1e18)
        dispatch.append((running, ranks[rid], rid))
    actual = [r[2] for r in sorted(dispatch)]
    outcomes = [_outcome(e, r) for e, r in zip(wave.entries, rows)]
    return placeholder, wave, ph, rows, raw, dispatch, actual, outcomes


def _same_priority(ctx, p, deadline):
    _, _, ph, rows, raw, dispatch, actual, outcomes = _observations(ctx, p, deadline)
    expected = [r["wire_request_id"] for r in rows]
    zero_eviction = all(code not in (8400, 8429) for _, code in outcomes)
    all_ok = all(ok for ok, _ in outcomes)
    shape = actual == expected
    checks = [
        CheckResult(
            "PR4",
            "PASS" if zero_eviction and shape and all_ok else "FAIL",
            actual=dict(zero_eviction=zero_eviction, all_ok=all_ok, dispatch=actual),
            expected=expected,
        ),
        CheckResult(
            "AT3",
            "PASS" if outcomes[-1][1] == 200 else "FAIL",
            actual=outcomes[-1][1],
            expected=200,
        ),
        CheckResult(
            "P6_terminal",
            "PASS" if shape and all_ok else "FAIL",
            actual=dict(shape=shape, all_ok=all_ok),
        ),
    ]
    path = ctx.artifact_dir / f"preemption-same-priority-{uuid.uuid4().hex}.json"
    path.write_text(
        json.dumps(
            dict(
                placeholder=ph,
                wave=rows,
                snapshot=raw,
                dispatch=dispatch,
                outcomes=outcomes,
            ),
            indent=2,
        )
        + "\n"
    )
    return StageOutput(checks=checks, artifacts=[str(path)])


def _queued_params(p, plan):
    p = _params(p, {"placeholder", "wave", "round"}, {"placeholder", "wave", "round"})
    for key in ("placeholder", "wave"):
        plan.reference(p[key], "requests")
    _number(p["round"], 1, 2, True)
    return p


def _queued_first_params(p, plan):
    p = _queued_params(p, plan)
    if p["round"] != 1:
        raise ValueError("first queued action requires round one")
    return p


def _queued_second_params(p, plan):
    p = _queued_params(p, plan)
    if p["round"] != 2:
        raise ValueError("second queued action requires round two")
    return p


def _queued(ctx, p, deadline):
    placeholder, wave, ph, rows, raw, dispatch, actual, outcomes = _observations(
        ctx, p, deadline
    )
    priorities = [r["priority"] for r in wave.p["requests"]]
    required = (
        [30, 30, 40, 40, 30, 30, 30, 30, 70] if p["round"] == 1 else [70] * 8 + [90]
    )
    if priorities != required or placeholder.p["requests"][0]["priority"] != (
        50 if p["round"] == 1 else 70
    ):
        raise ValueError("queued preemption cohort differs from legacy round")
    indices = [0] + sorted(range(1, 9), key=lambda i: (-priorities[i], i))
    expected = [rows[i]["wire_request_id"] for i in indices]
    ph_ok, ph_code = _outcome(placeholder.entries[0], ph[0])
    zero = all(code not in (8400, 8429) for _, code in outcomes)
    if p["round"] == 2:
        zero = zero and ph_code not in (8400, 8429)
    shape, all_ok = actual == expected, all(ok for ok, _ in outcomes)
    if p["round"] == 1:
        predicates = dict(
            PR10=shape and zero and ph_ok,
            PR5=zero and shape,
            PR6=all_ok and shape,
            PR4=zero and ph_ok and shape and all_ok,
        )
    else:
        predicates = dict(PR10=zero and shape and all_ok and ph_ok)
    predicates["P6_terminal"] = shape and all_ok and ph_ok
    evidence = dict(
        round=p["round"],
        shape=shape,
        all_ok=all_ok,
        placeholder_ok=ph_ok,
        zero_eviction=zero,
        dispatch=actual,
    )
    path = ctx.artifact_dir / f"preemption-queued-{uuid.uuid4().hex}.json"
    path.write_text(
        json.dumps(
            dict(
                placeholder=ph,
                wave=rows,
                snapshot=raw,
                dispatch=dispatch,
                outcomes=outcomes,
                placeholder_outcome=[ph_ok, ph_code],
                expected=expected,
            ),
            indent=2,
        )
        + "\n"
    )
    return StageOutput(
        checks=[
            CheckResult(
                key, "PASS" if ok else "FAIL", actual=evidence, expected=expected
            )
            for key, ok in predicates.items()
        ],
        artifacts=[str(path)],
    )


HANDLERS = [
    StageHandler(
        "preemption_queued_first",
        _queued_first_params,
        _queued,
        {},
        checks=frozenset({"PR10", "PR5", "PR6", "PR4", "P6_terminal"}),
    ),
    StageHandler(
        "preemption_queued_second",
        _queued_second_params,
        _queued,
        {},
        checks=frozenset({"PR10", "P6_terminal"}),
    ),
    StageHandler(
        "preemption_settled", _settled_params, _settled, {"admitted": "boolean"}
    ),
    StageHandler(
        "preemption_pending",
        _pending_params,
        _pending,
        {},
        checks=frozenset({"pending"}),
    ),
    StageHandler(
        "preemption_same_priority",
        _same_params,
        _same_priority,
        {},
        checks=frozenset({"PR4", "AT3", "P6_terminal"}),
    ),
]
