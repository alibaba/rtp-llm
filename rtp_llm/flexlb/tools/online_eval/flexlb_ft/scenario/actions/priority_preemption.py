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


def _observations(ctx, p, deadline, size=9):
    placeholder, wave = [_cohort(ctx, p[key]) for key in ("placeholder", "wave")]
    if not placeholder.complete or not wave.complete:
        raise ValueError("preemption verdict requires drained owned cohorts")
    ph, rows = placeholder.records(), wave.records()
    if len(ph) != 1 or len(rows) != size or len(wave.entries) != size:
        raise ValueError("preemption program has an unexpected cohort size")
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


def _comparator_params(p, plan):
    p = _params(
        p, {"placeholder", "wave", "ordering"}, {"placeholder", "wave", "ordering"}
    )
    for key in ("placeholder", "wave"):
        plan.reference(p[key], "requests")
    if p["ordering"] not in ("priority", "fifo"):
        raise ValueError("comparator ordering must be priority or fifo")
    return p


def _comparator(ctx, p, deadline):
    ph_wave, wave, ph, rows, raw, dispatch, actual, outcomes = _observations(
        ctx, p, deadline, size=5
    )
    priorities = [r["priority"] for r in wave.p["requests"]]
    if priorities != [30, 30, 70, 70, 70] or ph_wave.p["requests"][0]["priority"] != 30:
        raise ValueError("comparator cohort differs from the legacy load shape")
    indices = [0, 2, 3, 4, 1] if p["ordering"] == "priority" else list(range(5))
    expected = [rows[i]["wire_request_id"] for i in indices]
    ph_ok = _outcome(ph_wave.entries[0], ph[0])[0]
    all_ok, shape = all(ok for ok, _ in outcomes), actual == expected
    passed = ph_ok and all_ok and shape
    path = ctx.artifact_dir / f"preemption-comparator-{uuid.uuid4().hex}.json"
    path.write_text(
        json.dumps(
            dict(
                ordering=p["ordering"],
                placeholder=ph,
                wave=rows,
                raw=raw,
                dispatch=dispatch,
                expected=expected,
                outcomes=outcomes,
                placeholder_ok=ph_ok,
                actual=actual,
                all_ok=all_ok,
                shape_ok=shape,
                passed=passed,
            ),
            indent=2,
        )
        + "\n"
    )
    return StageOutput(
        {"passed": passed},
        artifacts=[str(path)],
    )


def _comparator_pair_params(p, plan):
    p = _params(p, {"priority_half", "fifo_half"}, {"priority_half", "fifo_half"})
    for value in p.values():
        plan.reference(value, "boolean")
    return p


def _comparator_pair(ctx, p, deadline):
    values = {key: ctx.resolve(value) for key, value in p.items()}
    passed = all(value is True for value in values.values())
    return StageOutput(
        checks=[
            CheckResult(key, "PASS" if passed else "FAIL", actual=values)
            for key in ("PR9", "P6")
        ]
    )


def _config_reject_params(p, plan):
    p = _params(
        p, {"rejected", "environment_absent"}, {"rejected", "environment_absent"}
    )
    if not isinstance(p["rejected"], list) or len(p["rejected"]) != 3:
        raise ValueError("config rejection requires all three actual startup probes")
    for ref in p["rejected"] + [p["environment_absent"]]:
        plan.reference(ref, "boolean")
    return p


def _config_reject(ctx, p, deadline):
    rejected = [ctx.resolve(ref) for ref in p["rejected"]]
    absent = ctx.resolve(p["environment_absent"])
    return StageOutput(
        checks=[
            CheckResult(
                "AT1",
                "PASS" if all(v is True for v in rejected) else "FAIL",
                actual=rejected,
                expected=[True] * 3,
            ),
            CheckResult(
                "P6", "PASS" if absent is True else "FAIL", actual=absent, expected=True
            ),
        ]
    )


def _decode_fleet(ctx, p, deadline):
    raw = _http(ctx.ops, "snapshot", deadline)
    rows = raw.get("engines")
    if not isinstance(rows, list):
        raise ValueError("missing engine fleet")
    result = {}
    for role, count in (("prefill", 2), ("decode", 4)):
        names = [
            r.get("name")
            for r in rows
            if r.get("role") == role and r.get("stopped") is False
        ]
        if (
            len(names) != count
            or any(not isinstance(n, str) for n in names)
            or len(set(names)) != count
        ):
            raise ValueError(
                "decode EV2 requires exactly two Prefill and four Decode owners"
            )
        _engines(raw, names)
        result.update({f"{role}{i}": name for i, name in enumerate(names)})
    path = ctx.artifact_dir / f"preemption-fleet-{uuid.uuid4().hex}.json"
    path.write_text(json.dumps(raw, indent=2) + "\n")
    return StageOutput(result, artifacts=[str(path)])


def _decode_targets(p, plan, pressure=False):
    keys = {"targets", "tokens"} if pressure else {"targets"}
    p = _params(p, keys, keys)
    if not isinstance(p["targets"], list) or len(p["targets"]) != 4:
        raise ValueError("four explicit Decode targets required")
    for ref in p["targets"]:
        plan.reference(ref, "string")
    if pressure and (type(p["tokens"]) is not int or p["tokens"] not in (0, 6291456)):
        raise ValueError("EV2 pressure must be zero or the old mock total6291456")
    return p


def _decode_owners(ctx, p, deadline):
    names = [ctx.resolve(ref) for ref in p["targets"]]
    if len(set(names)) != 4:
        raise ValueError("Decode targets must be distinct")
    raw = _http(ctx.ops, "snapshot", deadline)
    owners = _engines(raw, names)
    if any(r["role"] != "decode" or r["stopped"] is not False for r in owners.values()):
        raise ValueError("pressure requires four live Decode owners")
    return names, raw, owners


def _decode_pressure(ctx, p, deadline):
    names, raw, owners = _decode_owners(ctx, p, deadline)
    path = ctx.artifact_dir / f"preemption-pressure-{uuid.uuid4().hex}.json"
    evidence = dict(tokens=p["tokens"], before=raw, responses=[])
    ops = ctx.ops

    def send(name, tokens, limit):
        response = _http(
            ops, "set_kv_pressure", limit, dict(engine=name, active_kv_tokens=tokens)
        )
        if response.get("status") != "ok" or response.get("engine") != name:
            raise ValueError("pressure control lacks target acknowledgement")
        return response

    try:
        for name in names:
            if p["tokens"]:
                ctx.add_cleanup(
                    f"preemption-pressure-{name}", lambda d, name=name: send(name, 0, d)
                )
            evidence["responses"].append(send(name, p["tokens"], deadline))
    finally:
        path.write_text(json.dumps(evidence, indent=2) + "\n")
    return StageOutput(artifacts=[str(path)])


def _decode_guard(ctx, p, deadline):
    names, raw, owners = _decode_owners(ctx, p, deadline)
    # The legacy helper maps missing counters to -1 and accidentally passes.
    # A missing owner counter cannot prove saturation in the explicit program.
    values = {
        name: _number(owners[name].get("available_kv_tokens"), 0, 1e18)
        for name in names
    }
    passed = all(value <= 0 for value in values.values())
    path = ctx.artifact_dir / f"preemption-decode-guard-{uuid.uuid4().hex}.json"
    path.write_text(json.dumps(raw, indent=2) + "\n")
    return StageOutput(
        checks=[
            CheckResult(
                "all_decode_saturated",
                "PASS" if passed else "FAIL",
                actual=values,
                expected="all<=0",
            )
        ],
        artifacts=[str(path)],
    )


def _decode_running(ctx, p, deadline):
    wave = _cohort(ctx, p["requests"])
    rows = wave.records()
    if len(rows) != 4 or len(wave.entries) != 4:
        raise ValueError("engine-owned wave requires four actual Schedule records")
    samples = []
    path = ctx.artifact_dir / f"preemption-decode-running-{uuid.uuid4().hex}.json"
    try:
        for row in rows:
            limit = Deadline(
                min(deadline.expires_at, ctx.clock() + 20),
                ctx.clock,
                ctx.sleeper,
                deadline.cancelled,
            )
            rid = str(row["wire_request_id"])
            while True:
                limit.check()
                raw = _http(ctx.ops, "snapshot", limit)
                engines = raw.get("engines")
                if not isinstance(engines, list):
                    raise ValueError("missing Decode lifecycle snapshot")
                samples.append(dict(request_id=rid, time_s=ctx.clock(), raw=raw))
                matches = [
                    e.get("request_lifecycle", {}).get(rid, {})
                    for e in engines
                    if e.get("role") == "decode"
                ]
                running = False
                for lc in matches:
                    if not isinstance(lc, dict):
                        raise ValueError("malformed Decode lifecycle")
                    if lc.get("end_state") == "running":
                        running = True
                    elif lc.get("running_ms") is not None and not lc.get("end_state"):
                        running |= _number(lc["running_ms"], 0, 1e18) > 0
                if running:
                    break
                limit.sleep(0.1)
    finally:
        path.write_text(json.dumps(samples, indent=2) + "\n")
    return StageOutput(
        checks=[CheckResult("all_running", "PASS", actual=4, expected=4)],
        artifacts=[str(path)],
    )


def _decode_half_params(p, plan):
    p = _params(
        p, {"occupants", "incoming", "phase"}, {"occupants", "incoming", "phase"}
    )
    for key in ("occupants", "incoming"):
        plan.reference(p[key], "requests")
    if p["phase"] not in ("reserved", "owned"):
        raise ValueError("unknown Decode phase")
    return p


def _decode_half(ctx, p, deadline):
    occupants, incoming = [_cohort(ctx, p[k]) for k in ("occupants", "incoming")]
    if not occupants.complete or not incoming.complete:
        raise ValueError("Decode verdict requires drained cohorts")
    rows, inc = occupants.records(), incoming.records()
    if (
        len(rows) != 4
        or len(inc) != 1
        or len(occupants.entries) != 4
        or len(incoming.entries) != 1
    ):
        raise ValueError("Decode EV2 requires four occupants plus one incoming")
    if any(
        (r.get("priority"), r["input_len"], r["output_len"]) != (30, 2048, 500)
        for r in occupants.p["requests"]
    ) or [
        (r.get("priority"), r["input_len"], r["output_len"])
        for r in incoming.p["requests"]
    ] != [
        (70, 2048, 2)
    ]:
        raise ValueError("Decode EV2 cohort differs from old load shape")
    outcomes = [_outcome(e, r) for e, r in zip(occupants.entries, rows)]
    incoming_ok, code = _outcome(incoming.entries[0], inc[0])
    if any(type(c) is not int for _, c in outcomes) or type(code) is not int:
        raise ValueError("Decode EV2 needs typed terminal evidence")
    forbidden = (8400, 8429) if p["phase"] == "reserved" else (8429,)
    zero = all(c not in forbidden for _, c in outcomes)
    survivors = all(ok for ok, c in outcomes if p["phase"] == "reserved" or c != 8429)
    rejected = not incoming_ok and code in (8403, 8402, 8510, 8431)
    passed = zero and survivors and rejected
    path = ctx.artifact_dir / f"preemption-decode-{uuid.uuid4().hex}.json"
    path.write_text(
        json.dumps(
            dict(
                phase=p["phase"],
                occupants=rows,
                incoming=inc,
                outcomes=outcomes,
                incoming_outcome=(incoming_ok, code),
            ),
            indent=2,
        )
        + "\n"
    )
    return StageOutput(
        dict(zero_eviction=zero, incoming_rejected=rejected, survivors_ok=survivors),
        checks=[
            CheckResult(
                "PR6",
                "PASS" if passed else "FAIL",
                actual=dict(
                    zero_eviction=zero,
                    survivors_ok=survivors,
                    incoming_rejected=rejected,
                ),
            )
        ],
        artifacts=[str(path)],
    )


def _decode_final_params(p, plan):
    keys = {"reserved_zero", "owned_zero", "incoming_rejected", "survivors_ok"}
    p = _params(p, keys, keys)
    for ref in p.values():
        plan.reference(ref, "boolean")
    return p


def _decode_final(ctx, p, deadline):
    values = {k: ctx.resolve(v) for k, v in p.items()}
    pr10 = values["reserved_zero"] is True and values["owned_zero"] is True
    p6 = values["incoming_rejected"] is True and values["survivors_ok"] is True
    return StageOutput(
        checks=[
            CheckResult("PR10", "PASS" if pr10 else "FAIL", actual=values),
            CheckResult("P6", "PASS" if p6 else "FAIL", actual=values),
        ]
    )


def _error_segment_params(p, plan):
    p = _params(
        p, {"placeholder", "wave", "segment"}, {"placeholder", "wave", "segment"}
    )
    for key in ("placeholder", "wave"):
        plan.reference(p[key], "requests")
    if p["segment"] not in ("outstanding", "park", "expiry"):
        raise ValueError("unknown error-family segment")
    return p


def _error_segment(ctx, p, deadline):
    ph_wave, wave = [_cohort(ctx, p[k]) for k in ("placeholder", "wave")]
    if not ph_wave.complete or not wave.complete:
        raise ValueError("error-family verdict requires drained cohorts")
    ph, rows = ph_wave.records(), wave.records()
    first = p["segment"] == "outstanding"
    ph_count, count = (2, 2) if first else (1, 9)
    if (len(ph), len(ph_wave.entries), len(rows), len(wave.entries)) != (
        ph_count,
        ph_count,
        count,
        count,
    ):
        raise ValueError("error-family cohort size differs from old segment")
    expected_ph = [50, 50] if first else [70]
    expected_wave = (
        [30, 70]
        if first
        else ([70] * 8 + [90] if p["segment"] == "park" else [30] * 8 + [90])
    )
    for cohort, priorities in ((ph_wave, expected_ph), (wave, expected_wave)):
        if [r.get("priority") for r in cohort.p["requests"]] != priorities or any(
            (r["input_len"], r["output_len"]) != (2048, 2) for r in cohort.p["requests"]
        ):
            raise ValueError("error-family request shape differs from old segment")
    ph_out = [_outcome(e, r) for e, r in zip(ph_wave.entries, ph)]
    outcomes = [_outcome(e, r) for e, r in zip(wave.entries, rows)]
    all_codes = [c for _, c in ph_out + outcomes]
    if any(type(c) is not int for c in all_codes):
        raise ValueError("error-family verdict lacks typed terminal evidence")
    ph_ok = all(ok for ok, _ in ph_out)
    evidence = dict(
        segment=p["segment"],
        placeholder=ph,
        wave=rows,
        placeholder_outcomes=ph_out,
        outcomes=outcomes,
    )
    codes = [c for _, c in outcomes]
    if first:
        wall = [
            _number(r["schedule"]["ended_s"], 0, 1e18)
            - _number(e.get("submitted_s"), 0, 1e18)
            for e, r in zip(wave.entries, rows)
        ]
        if any(v < 0 for v in wall):
            raise ValueError("Schedule settlement precedes submission")
        fast = all(v < 3 for v in wall)
        isolated = all(c not in (8402, 8403, 8431, 8400, 8429, 8511) for c in all_codes)
        passed = codes == [8502, 8502] and fast and ph_ok and isolated
        evidence.update(
            schedule_wall_s=wall,
            fast=fast,
            reasons=[
                getattr(
                    e["batch"].entries[0]["response"], "admission_reject_reason", None
                )
                for e in wave.entries
            ],
        )
    elif p["segment"] == "park":
        _, _, _, _, raw, dispatch, actual, _ = _observations(ctx, p, deadline)
        expected = [rows[i]["wire_request_id"] for i in [0, 8, 1, 2, 3, 4, 5, 6, 7]]
        isolated = all(c not in (8502, 8403, 8431, 8400, 8429, 8511) for c in all_codes)
        passed = (
            codes[-1] == 200
            and all(c not in (8400, 8429) for c in codes)
            and actual == expected
            and all(ok for ok, _ in outcomes)
            and ph_ok
            and isolated
        )
        evidence.update(raw=raw, dispatch=dispatch, actual=actual, expected=expected)
    else:
        isolated = all(c not in (8502, 8403, 8400, 8429) for c in all_codes)
        passed = (
            all(c == 8511 for c in codes)
            and 8400 not in codes[:8]
            and ph_ok
            and isolated
        )
        evidence["incoming_reason"] = getattr(
            wave.entries[-1]["batch"].entries[0]["response"],
            "admission_reject_reason",
            None,
        )
    evidence.update(passed=passed, isolated=isolated)
    path = ctx.artifact_dir / f"preemption-error-family-{uuid.uuid4().hex}.json"
    path.write_text(json.dumps(evidence, indent=2) + "\n")
    return StageOutput({"passed": passed}, artifacts=[str(path)])


def _error_recovery(ctx, p, deadline):
    from .engine_recovery import RecoveryRequests

    cohort = ctx.resource(p["requests"], "requests")
    if not isinstance(cohort, RecoveryRequests):
        raise ValueError("error-family recovery requires owned recovery workers")
    rows = cohort.snapshot_records()
    if len(rows) != 1:
        raise ValueError("error-family recovery requires one request")
    passed = (
        request_success(rows[0]) and rows[0].get("recovery_observed_success") is True
    )
    path = ctx.artifact_dir / f"preemption-error-recovery-{uuid.uuid4().hex}.json"
    path.write_text(json.dumps(rows, indent=2) + "\n")
    return StageOutput({"passed": passed}, artifacts=[str(path)])


def _error_final_params(p, plan):
    p = _params(p, {"segments", "recovery"}, {"segments", "recovery"})
    if not isinstance(p["segments"], list) or len(p["segments"]) != 3:
        raise ValueError("all three error-family segments required")
    for ref in p["segments"] + [p["recovery"]]:
        plan.reference(ref, "boolean")
    return p


def _error_final(ctx, p, deadline):
    values = dict(
        segments=[ctx.resolve(v) for v in p["segments"]],
        recovery=ctx.resolve(p["recovery"]),
    )
    passed = all(v is True for v in values["segments"]) and values["recovery"] is True
    return StageOutput(
        checks=[
            CheckResult(k, "PASS" if passed else "FAIL", actual=values)
            for k in ("AT4", "P6")
        ]
    )


HANDLERS = [
    StageHandler(
        "preemption_error_segment",
        _error_segment_params,
        _error_segment,
        {"passed": "boolean"},
    ),
    StageHandler(
        "preemption_error_recovery",
        _settled_params,
        _error_recovery,
        {"passed": "boolean"},
    ),
    StageHandler(
        "preemption_error_final",
        _error_final_params,
        _error_final,
        {},
        checks=frozenset({"AT4", "P6"}),
    ),
    StageHandler(
        "preemption_decode_fleet",
        lambda p, plan: _params(p, (), ()),
        _decode_fleet,
        {
            **{f"prefill{i}": "string" for i in range(2)},
            **{f"decode{i}": "string" for i in range(4)},
        },
    ),
    StageHandler(
        "preemption_decode_pressure",
        lambda p, plan: _decode_targets(p, plan, True),
        _decode_pressure,
        {},
    ),
    StageHandler(
        "preemption_decode_guard",
        _decode_targets,
        _decode_guard,
        {},
        checks=frozenset({"all_decode_saturated"}),
    ),
    StageHandler(
        "preemption_decode_running",
        _settled_params,
        _decode_running,
        {},
        checks=frozenset({"all_running"}),
    ),
    StageHandler(
        "preemption_decode_half",
        _decode_half_params,
        _decode_half,
        {
            "zero_eviction": "boolean",
            "incoming_rejected": "boolean",
            "survivors_ok": "boolean",
        },
        checks=frozenset({"PR6"}),
    ),
    StageHandler(
        "preemption_decode_final",
        _decode_final_params,
        _decode_final,
        {},
        checks=frozenset({"PR10", "P6"}),
    ),
    StageHandler(
        "preemption_config_reject",
        _config_reject_params,
        _config_reject,
        {},
        checks=frozenset({"AT1", "P6"}),
    ),
    StageHandler(
        "preemption_comparator_pair",
        _comparator_pair_params,
        _comparator_pair,
        {},
        checks=frozenset({"PR9", "P6"}),
    ),
    StageHandler(
        "preemption_comparator_half",
        _comparator_params,
        _comparator,
        {"passed": "boolean"},
    ),
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
