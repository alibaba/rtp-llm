"""Explicit preemption predicates over owned cohorts and engine observations."""

import json
import uuid

from ..contracts import CheckResult, StageHandler, StageOutput
from ..runtime import Deadline
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
    stream = row.get("stream", {})
    if stream.get("status") != "OK":
        return False, _typed_stream_code(row)
    # Legacy maps the in-band enum only. A typed trailer's literal 2 stays 2.
    code = row.get("business_error_code")
    return False, 8429 if code == 2 else code


def _typed_stream_code(row):
    stream = row.get("stream", {})
    # Legacy client transport cancellation is not an engine terminal, even
    # when metadata is attached. Preserve transport and raw bytes separately.
    if stream.get("status") in (None, "OK", "CANCELLED"):
        return None
    code = stream.get("trailer_error_code")
    parsed = stream.get("error_trailer", {}).get("status") == "parsed"
    return code if parsed and type(code) is int else None


def _terminal_wait(ctx, p, deadline):
    wave = _cohort(ctx, p["requests"])
    for entry in wave.entries:
        wave._join(entry, deadline)
    for entry in wave.entries:
        if entry["error"] is not None:
            raise entry["error"]
        per_request = Deadline(
            min(deadline.expires_at, ctx.clock() + 35), ctx.clock, ctx.sleeper
        )
        try:
            entry["batch"].wait(per_request)
        except RuntimeError:
            # Permit only a fully consumed, actual typed server error to reach
            # this scenario's verdict. Stage/RPC deadlines still propagate.
            per_request.check()
            rows = entry["batch"].snapshot_records()
            if len(rows) != 1:
                raise
            row = rows[0]
            if not (
                row["schedule"]["status"] == "OK"
                and row.get("consumer_completion_verified") is True
                and row.get("consumer_done") is True
                and row.get("consumer_exit_s") is not None
                and row.get("transport_terminal_s") is not None
                and _typed_stream_code(row) is not None
            ):
                raise
    wave.complete = True
    wave.persist()
    return StageOutput(artifacts=[str(wave.path)])


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


def _expiry(ctx, p, deadline):
    placeholder, wave = [_cohort(ctx, p[key]) for key in ("placeholder", "wave")]
    if not placeholder.complete or not wave.complete:
        raise ValueError("expiry verdict requires drained cohorts")
    ph, rows = placeholder.records(), wave.records()
    expected_priorities = [30] * 8 + [70] + ([90, 90] if p["round"] == 1 else [])
    if len(ph) != 1 or len(rows) != len(expected_priorities):
        raise ValueError("expiry cohort size differs from legacy round")
    if [r["priority"] for r in wave.p["requests"]] != expected_priorities:
        raise ValueError("expiry cohort priorities differ from legacy round")
    if placeholder.p["requests"][0]["priority"] != (90 if p["round"] == 1 else 70):
        raise ValueError("expiry placeholder priority differs from legacy round")
    outcomes = [_outcome(e, r) for e, r in zip(wave.entries, rows)]
    reasons = []
    for entry in wave.entries:
        response = entry["batch"].entries[0]["response"]
        if response is None:
            raise ValueError("expiry requires actual Schedule response evidence")
        reasons.append(
            _number(
                getattr(response, "admission_reject_reason", None), 0, 2**31 - 1, True
            )
        )
    ph_ok, ph_code = _outcome(placeholder.entries[0], ph[0])
    victims = [rows[i]["wire_request_id"] for i in range(8) if outcomes[i][1] == 8400]
    expired = all(code == 8511 for _, code in outcomes)
    unspecified = all(reason == 0 for reason in reasons)
    incoming = rows[8]["schedule"]
    wall_ms = (incoming["ended_s"] - incoming["started_s"]) * 1000
    evidence = dict(
        expired=expired,
        reasons=reasons,
        victims8400=victims,
        placeholder_ok=ph_ok,
        incoming_schedule_wall_ms=wall_ms,
    )
    path = ctx.artifact_dir / f"preemption-expiry-{uuid.uuid4().hex}.json"
    path.write_text(
        json.dumps(
            dict(
                placeholder=ph,
                wave=rows,
                outcomes=outcomes,
                placeholder_outcome=[ph_ok, ph_code],
                evidence=evidence,
            ),
            indent=2,
        )
        + "\n"
    )
    return StageOutput(
        checks=[
            CheckResult(
                "PR7",
                "PASS" if expired and unspecified and not victims and ph_ok else "FAIL",
                actual=evidence,
            ),
            CheckResult(
                "P6_terminal",
                "PASS" if ph_ok and not victims else "FAIL",
                actual=evidence,
            ),
        ],
        artifacts=[str(path)],
    )


def _disabled(ctx, p, deadline):
    placeholder, wave = [_cohort(ctx, p[key]) for key in ("placeholder", "wave")]
    if not placeholder.complete or not wave.complete:
        raise ValueError("disabled verdict requires drained cohorts")
    ph, rows = placeholder.records(), wave.records()
    fill, incoming = (30, 70) if p["round"] == 1 else (70, 90)
    if len(ph) != 1 or len(rows) != 9:
        raise ValueError("disabled program needs one placeholder plus nine peers")
    if [r["priority"] for r in wave.p["requests"]] != [fill] * 8 + [incoming]:
        raise ValueError("disabled cohort priorities differ from legacy round")
    if placeholder.p["requests"][0]["priority"] != fill:
        raise ValueError("disabled placeholder priority differs from legacy round")
    outcomes = [_outcome(e, r) for e, r in zip(wave.entries, rows)]
    ph_outcome = _outcome(placeholder.entries[0], ph[0])
    # An unknown terminal cannot prove absence of a forbidden error family.
    if any(code is None for _, code in outcomes + [ph_outcome]):
        raise ValueError("disabled preemption requires typed terminal evidence")
    zero = all(
        code not in (8400, 8429, 8430) for _, code in outcomes[:8] + [ph_outcome]
    )
    inc_ok = outcomes[-1][1] in (200, 8511)
    evidence = dict(
        zero_preemption=zero,
        incoming_code=outcomes[-1][1],
        outcomes=outcomes,
        placeholder_outcome=ph_outcome,
    )
    path = ctx.artifact_dir / f"preemption-disabled-{uuid.uuid4().hex}.json"
    path.write_text(
        json.dumps(dict(placeholder=ph, wave=rows, evidence=evidence), indent=2) + "\n"
    )
    return StageOutput(
        checks=[
            CheckResult(key, "PASS" if zero and inc_ok else "FAIL", actual=evidence)
            for key in ("AT2", "P6_terminal")
        ],
        artifacts=[str(path)],
    )


HANDLERS = [
    StageHandler("preemption_wait", _settled_params, _terminal_wait, {}),
    StageHandler(
        "preemption_disabled",
        _queued_params,
        _disabled,
        {},
        checks=frozenset({"AT2", "P6_terminal"}),
    ),
    StageHandler(
        "preemption_expiry",
        _queued_params,
        _expiry,
        {},
        checks=frozenset({"PR7", "P6_terminal"}),
    ),
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
