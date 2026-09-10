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


def _first_peer_params(p):
    if type(p.setdefault("first_peer_exempt", True)) is not bool:
        raise ValueError("first_peer_exempt must be boolean")
    return p


def _priority_indices(priorities, first_peer_exempt):
    first = [0] if first_peer_exempt else []
    return first + sorted(
        range(len(first), len(priorities)), key=lambda i: (-priorities[i], i)
    )


def _queued_params(p, plan):
    p = _params(
        p,
        {"placeholder", "wave", "round", "first_peer_exempt"},
        {"placeholder", "wave", "round"},
    )
    for key in ("placeholder", "wave"):
        plan.reference(p[key], "requests")
    _number(p["round"], 1, 2, True)
    return _first_peer_params(p)


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
    indices = _priority_indices(priorities, p.get("first_peer_exempt", True))
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
                first_peer_exempt=p.get("first_peer_exempt", True),
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
        p,
        {"placeholder", "wave", "ordering", "first_peer_exempt"},
        {"placeholder", "wave", "ordering"},
    )
    for key in ("placeholder", "wave"):
        plan.reference(p[key], "requests")
    if p["ordering"] not in ("priority", "fifo"):
        raise ValueError("comparator ordering must be priority or fifo")
    return _first_peer_params(p)


def _comparator(ctx, p, deadline):
    ph_wave, wave, ph, rows, raw, dispatch, actual, outcomes = _observations(
        ctx, p, deadline, size=5
    )
    priorities = [r["priority"] for r in wave.p["requests"]]
    if priorities != [30, 30, 70, 70, 70] or ph_wave.p["requests"][0]["priority"] != 30:
        raise ValueError("comparator cohort differs from the legacy load shape")
    indices = (
        _priority_indices(priorities, p.get("first_peer_exempt", True))
        if p["ordering"] == "priority"
        else list(range(5))
    )
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
                first_peer_exempt=p.get("first_peer_exempt", True),
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
        p,
        {"placeholder", "wave", "segment", "first_peer_exempt"},
        {"placeholder", "wave", "segment"},
    )
    for key in ("placeholder", "wave"):
        plan.reference(p[key], "requests")
    if p["segment"] not in ("outstanding", "park", "expiry"):
        raise ValueError("unknown error-family segment")
    return _first_peer_params(p)


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
        first_peer_exempt=p.get("first_peer_exempt", True),
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
        expected = [
            rows[i]["wire_request_id"]
            for i in _priority_indices(expected_wave, p.get("first_peer_exempt", True))
        ]
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


def _reservation_metric_params(p, plan):
    p = _params(p, {"labels"}, {"labels"})
    if p["labels"] not in ({}, {"victim_priority": "30", "incoming_priority": "70"}):
        raise ValueError("reservation metric labels must match a legacy wave")
    return p


def _strict_victim_sample(line):
    """Validate an emitted classic-name victim sample before label selection."""
    import re

    sample = re.fullmatch(
        r"([A-Za-z_:][A-Za-z0-9_:]*)(?:[ \t]*\{(.*)\})?[ \t]+"
        r"([+-]?(?:(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?|Inf|NaN))"
        r"(?:[ \t]+([+-]?[0-9]+))?[ \t]*",
        line,
    )
    if sample is None:
        raise ValueError("malformed victim metric sample")
    name, block, value, timestamp = sample.groups()
    labels = {}
    rest = "" if block is None else block.strip(" \t")
    label_pattern = re.compile(
        r'([A-Za-z_][A-Za-z0-9_]*)[ \t]*=[ \t]*"((?:[^"\\\n]|\\[\\"n])*)"'
    )
    while rest:
        label = label_pattern.match(rest)
        if label is None:
            raise ValueError("malformed victim metric label")
        key, raw = label.groups()
        if key in labels:
            raise ValueError("duplicate victim metric label")
        labels[key] = re.sub(
            r'\\([\\"n])', lambda match: "\n" if match[1] == "n" else match[1], raw
        )
        rest = rest[label.end() :].lstrip(" \t")
        if rest:
            if rest[0] != ",":
                raise ValueError("trailing victim metric label content")
            rest = rest[1:].lstrip(" \t")
    if timestamp is not None and not -(2**63) <= int(timestamp) < 2**63:
        raise ValueError("victim metric timestamp outside int64")
    return name, labels, _number(float(value), 0, 1e18)


def _reservation_metric(ctx, p, deadline):
    from ...engine_ops import parse_prometheus_samples
    from .status_protocol import _http as metric_http

    source = getattr(ctx, "preemption_victim_metric_source", None)
    if source is not None and source[0] != ctx.env_epoch:
        raise ValueError("victim metric source belongs to an old environment")
    paths = [source[1]] if source else ["actuator/prometheus", "prometheus"]
    evidence = dict(env_epoch=ctx.env_epoch, labels=p["labels"], attempts=[])
    path = ctx.artifact_dir / f"preemption-victim-metric-{uuid.uuid4().hex}.json"
    try:
        for endpoint in paths:
            status, body = metric_http(
                ctx, "metrics", endpoint, deadline, allowed=(200, 404), text=True
            )
            evidence["attempts"].append(
                dict(endpoint=endpoint, status=status, body=body)
            )
            if status == 404:
                continue
            samples = parse_prometheus_samples(body, "")
            if not samples:
                raise ValueError("missing valid Prometheus exposition")
            # Validate complete matching samples before applying a label subset.
            # The shared permissive parser can discard malformed label pairs.
            import re

            victim_samples = []
            identities = set()
            for line in body.splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                name = re.split(r"[{\s]", stripped, maxsplit=1)[0]
                if "auto_tpm_victim" not in name:
                    continue
                sample = _strict_victim_sample(stripped)
                identity = (sample[0], tuple(sorted(sample[1].items())))
                if identity in identities:
                    raise ValueError("duplicate victim metric series")
                identities.add(identity)
                victim_samples.append(sample)
            selected = [
                sample
                for sample in victim_samples
                if all(sample[1].get(k) == v for k, v in p["labels"].items())
            ]
            value = sum(row[2] for row in selected) if selected else None
            evidence.update(
                endpoint=endpoint,
                selected=selected,
                value=value,
                missing_series=not selected,
                legacy_effective_value=value or 0.0,
            )
            ctx.preemption_victim_metric_source = (ctx.env_epoch, endpoint)
            break
        else:
            raise ValueError("no available Prometheus endpoint")
    except Exception as exc:
        evidence["error"] = repr(exc)
        raise
    finally:
        path.write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")
    return StageOutput(
        {"snapshot": ctx.register_resource("snapshot", evidence, historical=True)},
        artifacts=[str(path)],
    )


def _reservation_half_params(p, plan):
    p = _params(
        p,
        {"occupants", "incoming", "wave", "baseline", "after"},
        {"occupants", "incoming", "wave"},
    )
    for key in ("occupants", "incoming"):
        plan.reference(p[key], "requests")
    if p["wave"] not in ("lower", "same", "kvbucket"):
        raise ValueError("unknown reservation wave")
    if p["wave"] == "kvbucket":
        if "baseline" in p or "after" in p:
            raise ValueError("legacy kvbucket wave has no metric predicate")
    else:
        for key in ("baseline", "after"):
            plan.reference(p.get(key), "snapshot")
    return p


def _reservation_half(ctx, p, deadline):
    occupants, incoming = [_cohort(ctx, p[k]) for k in ("occupants", "incoming")]
    rows, inc = occupants.records(), incoming.records()
    if (
        not occupants.complete
        or not incoming.complete
        or (len(rows), len(occupants.entries), len(inc), len(incoming.entries))
        != (4, 4, 1, 1)
    ):
        raise ValueError("reservation verdict requires drained four-plus-one cohorts")
    priority = 50 if p["wave"] == "same" else 30
    lengths = [2048, 2048, 16384, 16384] if p["wave"] == "kvbucket" else [2048] * 4
    incoming_shape = (
        (50, 2048, 2)
        if p["wave"] == "same"
        else (70, 8192 if p["wave"] == "kvbucket" else 2048, 2)
    )
    if [
        (r.get("priority"), r["input_len"], r["output_len"])
        for r in occupants.p["requests"]
    ] != [(priority, n, 500) for n in lengths] or [
        (r.get("priority"), r["input_len"], r["output_len"])
        for r in incoming.p["requests"]
    ] != [
        incoming_shape
    ]:
        raise ValueError("reservation wave differs from old priority/token shape")
    outcomes = [_outcome(e, r) for e, r in zip(occupants.entries, rows)]
    incoming_ok, code = _outcome(incoming.entries[0], inc[0])
    if type(code) is not int or any(type(c) is not int for _, c in outcomes):
        raise ValueError("reservation verdict needs typed terminals")
    victim_codes = (8429,) if p["wave"] == "lower" else (8400, 8429)
    victims = [
        rows[i]["wire_request_id"]
        for i, (_, c) in enumerate(outcomes)
        if c in victim_codes
    ]
    survivors = all(ok for ok, c in outcomes if c not in victim_codes)
    legal = (
        code in (200, 8511, 8403, 8402, 8510, 8431)
        if p["wave"] == "same"
        else incoming_ok or code in (8403, 8402, 8510, 8431)
    )
    evidence = dict(
        wave=p["wave"],
        occupants=rows,
        incoming=inc,
        outcomes=outcomes,
        incoming_outcome=(incoming_ok, code),
        victims=victims,
        survivors_ok=survivors,
        incoming_legal=legal,
    )
    metric_ok = True
    if p["wave"] != "kvbucket":
        before, after = [ctx.resource(p[k], "snapshot") for k in ("baseline", "after")]
        labels = (
            {}
            if p["wave"] == "same"
            else {"victim_priority": "30", "incoming_priority": "70"}
        )
        for observation in (before, after):
            if (
                observation.get("env_epoch") != ctx.env_epoch
                or observation.get("labels") != labels
                or "value" not in observation
                or observation.get("error")
            ):
                raise ValueError("wrong victim metric observation")
        if before["endpoint"] != after["endpoint"]:
            raise ValueError("victim metric source changed")
        delta = (after["value"] or 0.0) - (before["value"] or 0.0)
        metric_ok = delta == 0.0
        evidence.update(
            baseline=before,
            after=after,
            victim_delta=delta,
            missing_series_policy="legacy None-or-zero; not proof of an exposed zero series",
        )
    passed = not victims and survivors and legal and metric_ok
    evidence["passed"] = passed
    path = ctx.artifact_dir / f"preemption-reservation-{uuid.uuid4().hex}.json"
    path.write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")
    return StageOutput({"passed": passed}, artifacts=[str(path)])


def _reservation_final_params(p, plan):
    p = _params(p, {"waves"}, {"waves"})
    if not isinstance(p["waves"], list) or len(p["waves"]) != 3:
        raise ValueError("all three reservation waves required")
    for ref in p["waves"]:
        plan.reference(ref, "boolean")
    return p


def _reservation_final(ctx, p, deadline):
    values = [ctx.resolve(ref) for ref in p["waves"]]
    passed = all(v is True for v in values)
    return StageOutput(
        checks=[
            CheckResult(k, "PASS" if passed else "FAIL", actual=values)
            for k in ("AT7", "P6")
        ]
    )


def _observability_duplicate(ctx, p, deadline):
    from ..runtime import StageTimeout

    cohort = _cohort(ctx, p["requests"])
    rows = cohort.records()
    if len(rows) != 1 or rows[0]["schedule"]["status"] != "OK":
        raise ValueError("duplicate probe requires one admitted placeholder")
    rid = rows[0]["wire_request_id"]
    evidence = dict(wire_request_id=rid, priority=40, code=None, error=None)
    holder = {}

    def cleanup(d):
        call = holder.get("call")
        if call is not None:
            call.cancel()

    ctx.add_cleanup("observability-duplicate-client", cleanup)
    path = (
        ctx.artifact_dir / f"preemption-observability-duplicate-{uuid.uuid4().hex}.json"
    )
    try:
        limit = min(30.0, deadline.remaining())
        stub = ctx.ops.schedule_pb2_grpc.FlexlbServiceStub(
            ctx.ops._channel(ctx.ops.master_target())
        )
        evidence["started_s"] = ctx.clock()
        holder["call"] = stub.Schedule.future(
            ctx.ops.build_schedule_request(
                rid, priority=40, input_len=2048, output_len=2
            ),
            timeout=limit,
        )
        response = holder["call"].result(timeout=limit)
        evidence["code"] = int(response.code)
    except StageTimeout:
        raise
    except Exception as exc:
        # The legacy probe records RPC failure then continues the main wave.
        deadline.check()
        evidence["error"] = repr(exc)
    finally:
        cleanup(deadline)
        holder.clear()
        evidence["ended_s"] = ctx.clock()
        path.write_text(json.dumps(evidence, indent=2) + "\n")
    return StageOutput({"rejected": evidence["code"] == 8406}, artifacts=[str(path)])


def _observability_params(p, plan):
    p = _params(
        p, {"placeholder", "wave", "first_peer_exempt"}, {"placeholder", "wave"}
    )
    _same_params({k: p[k] for k in ("placeholder", "wave")}, plan)
    return _first_peer_params(p)


def _observability(ctx, p, deadline):
    from pathlib import Path

    from ...engine_ops import parse_prometheus_samples

    ph_wave, wave = [_cohort(ctx, p[k]) for k in ("placeholder", "wave")]
    if not ph_wave.complete or not wave.complete:
        raise ValueError("observability requires drained owned cohorts")
    ph, rows = ph_wave.records(), wave.records()
    tags = ["30a", "30b", "50a", "50b", "70a", "70b", "30c", "30d", "90"]
    if len(ph) != 1 or len(rows) != 9 or len(wave.entries) != 9:
        raise ValueError("observability cohort size differs from O1")
    for cohort, priorities in (
        (ph_wave, [50]),
        (wave, [30, 30, 50, 50, 70, 70, 30, 30, 90]),
    ):
        if [r.get("priority") for r in cohort.p["requests"]] != priorities or any(
            (r["input_len"], r["output_len"]) != (2048, 2) for r in cohort.p["requests"]
        ):
            raise ValueError("observability request shape differs from O1")
    if [r["tag"] for r in wave.p["requests"]] != tags:
        raise ValueError("observability tags differ from O1")
    ph_ok, ph_code = _outcome(ph_wave.entries[0], ph[0])
    outcomes = [_outcome(e, r) for e, r in zip(wave.entries, rows)]
    if any(type(code) is not int for code in [ph_code] + [c for _, c in outcomes]):
        raise ValueError("observability lacks typed terminal evidence")
    raw = _http(ctx.ops, "snapshot", deadline)
    # Keep the original cohort and snapshot even when validation fails before
    # StageOutput can be returned. Missing lifecycle evidence must be diagnosable.
    input_evidence = ctx.artifact_dir / f"preemption-o1-input-{uuid.uuid4().hex}.json"
    input_evidence.write_text(
        json.dumps(
            dict(
                placeholder=ph,
                wave=rows,
                outcomes=outcomes,
                snapshot=raw,
            ),
            indent=2,
        )
        + "\n"
    )
    engines = raw.get("engines")
    if not isinstance(engines, list):
        raise ValueError(
            f"missing Prefill lifecycle snapshot; evidence={input_evidence}"
        )
    ranks = {
        r["wire_request_id"]: i
        for i, r in enumerate(
            sorted(
                ph + rows,
                key=lambda r: (r["schedule"]["ended_s"], r["wire_request_id"]),
            )
        )
    }
    # Only the completed pair needs dispatch evidence; expired requests need not run.
    exempt = p.get("first_peer_exempt", True)
    completed_indices = (0, 8) if exempt else (4, 8)
    dispatch = []
    for index in completed_indices:
        row = rows[index]
        rid = row["wire_request_id"]
        matches = [
            e.get("request_lifecycle", {}).get(str(rid))
            for e in engines
            if e.get("role") == "prefill"
        ]
        matches = [m for m in matches if m is not None]
        if (
            not matches
            and row["schedule"]["status"] == "REJECTED"
            and outcomes[index] == (False, 8511)
        ):
            # A witnessed queue expiry never entered Prefill. It violates the
            # expected completed pair, but is not missing telemetry for a
            # completed request. Keep it as a failed business predicate below.
            dispatch.append(None)
            continue
        if len(matches) != 1:
            raise ValueError(
                f"missing or ambiguous O1 Prefill lifecycle: request_id={rid}, "
                f"matches={len(matches)}; evidence={input_evidence}"
            )
        dispatch.append(
            (_number(matches[0].get("running_ms"), 0, 1e18), ranks[rid], rid)
        )
    completed = ["ph"] + [tag for tag, (ok, _) in zip(tags, outcomes) if ok]
    expired = [tag for tag, (_, code) in zip(tags, outcomes) if code == 8511]
    rejected = [tag for tag, (_, code) in zip(tags, outcomes) if code in (8402, 8510)]
    client = (
        ph_ok
        and completed == (["ph", "30a", "90"] if exempt else ["ph", "70a", "90"])
        and len(expired) == 7
        and not rejected
        and all(item is not None for item in dispatch)
        and (dispatch[0] < dispatch[1] if exempt else dispatch[1] < dispatch[0])
    )

    metric = _reservation_metric(ctx, {"labels": {}}, deadline)
    snapshot = ctx.resource(metric.output["snapshot"], "snapshot")
    body = snapshot["attempts"][-1]["body"]
    samples = parse_prometheus_samples(body, "")

    def metric_sum(name, labels):
        values = [
            _number(value, 0, 1e18)
            for metric_name, metric_labels, value in samples
            if name in metric_name
            and all(metric_labels.get(k) == v for k, v in labels.items())
        ]
        return sum(values) if values else None

    buckets = {
        str(prio): metric_sum("auto_tpm_request", {"priority": str(prio)})
        for prio in (30, 50, 70, 90)
    }
    latency = metric_sum("auto_tpm_schedule", {"result": "success"})
    victim = snapshot["value"]
    metrics_ok = (
        buckets == {"30": 4.0, "50": 3.0, "70": 2.0, "90": 1.0}
        and latency is not None
        and (victim or 0.0) == 0.0
    )
    private = getattr(ctx, "master_log_dir", None)
    if private is None or private != getattr(ctx.env, "master_log_dir", None):
        raise ValueError("O1 requires this environment's private Master log directory")
    private = Path(private).resolve()
    root = ctx.artifact_dir.resolve()
    if root not in private.parents:
        raise ValueError("O1 Master log directory lies outside instance artifacts")
    logs = {}

    def read_log(path):
        path = Path(path)
        if root not in path.resolve().parents:
            raise ValueError("O1 log path escapes owned instance")
        try:
            with path.open("rb") as stream:
                data = stream.read(16 * 1024 * 1024 + 1)
            if len(data) > 16 * 1024 * 1024:
                raise ValueError("O1 log exceeds bounded observation size")
            text = data.decode("utf-8", errors="replace")
            logs[str(path)] = dict(text=text, missing=False)
            return text
        except FileNotFoundError:
            logs[str(path)] = dict(text="", missing=True)
            return ""

    master = (
        read_log(Path(ctx.env.run_dir) / "flexlb_master.log")
        + "\n"
        + read_log(private / "flexlb.log")
    )
    pv = read_log(private / "pv.log")
    wanted = {r["wire_request_id"] for r in ph + rows}
    selected = []
    for line in pv.splitlines():
        start = line.find("{")
        if start < 0:
            continue
        try:
            record = json.loads(line[start:])
        except ValueError:
            continue
        if (
            isinstance(record, dict)
            and type(record.get("requestId")) is int
            and record["requestId"] in wanted
        ):
            selected.append(record)
    selected = selected[-400:]
    scheduler_log = "[request-scheduler]" in master
    pv_field = any("admissionRejectReason" in row for row in selected)
    evidence = dict(
        first_peer_exempt=exempt,
        completed_indices=completed_indices,
        placeholder=ph,
        wave=rows,
        outcomes=outcomes,
        completed=completed,
        expired=expired,
        rejected=rejected,
        raw=raw,
        dispatch=dispatch,
        client=client,
        buckets=buckets,
        latency_success=latency,
        victim_total=victim,
        metrics_ok=metrics_ok,
        logs=logs,
        pv_selected=selected,
        scheduler_log=scheduler_log,
        pv_field=pv_field,
    )
    path = ctx.artifact_dir / f"preemption-observability-{uuid.uuid4().hex}.json"
    path.write_text(json.dumps(evidence, indent=2) + "\n")
    return StageOutput(
        {"client": client, "planes": metrics_ok and scheduler_log and pv_field},
        artifacts=metric.artifacts + [str(path)],
    )


def _observability_final_params(p, plan):
    p = _params(p, {"client", "planes", "duplicate"}, {"client", "planes", "duplicate"})
    for key in p:
        plan.reference(p[key], "boolean")
    return p


def _observability_final(ctx, p, deadline):
    client, planes, duplicate = [
        ctx.resolve(p[k]) for k in ("client", "planes", "duplicate")
    ]
    return StageOutput(
        checks=[
            CheckResult(key, "PASS" if passed else "FAIL", actual=passed, expected=True)
            for key, passed in (
                ("AT8", client and planes),
                ("P6", client),
                ("AT6", duplicate and client),
            )
        ]
    )


def _live_pressure_params(p, plan):
    p = _params(p, {"tokens"}, {"tokens"})
    if type(p["tokens"]) is not int or p["tokens"] < 0:
        raise ValueError("live pressure requires nonnegative token count")
    return p


def _live_pressure(ctx, p, deadline):
    """Use the existing reporting-only control; retain physical-pool evidence."""
    path = ctx.artifact_dir / f"preemption-live-pressure-{uuid.uuid4().hex}.json"
    evidence = {"tokens": p["tokens"], "complete": False}
    ops = ctx.ops

    def send(name, tokens, limit):
        response = _http(
            ops, "set_kv_pressure", limit, dict(engine=name, active_kv_tokens=tokens)
        )
        if response.get("status") != "ok" or response.get("engine") != name:
            raise ValueError("live pressure lacks target acknowledgement")
        return response

    try:
        raw, engines = _live_engine_rows(ctx, deadline)
        decodes = [e for e in engines if e["role"] == "decode"]
        if len(decodes) != 1 or decodes[0].get("stopped") is not False:
            raise ValueError("live pressure requires one live Decode")
        before = decodes[0]
        evidence["before"] = raw
        for field in (
            "available_blocks",
            "cache_blocks",
            "total_kv_tokens",
            "held_blocks",
            "referenced_blocks",
        ):
            if type(before.get(field)) is not int:
                raise ValueError(f"live pressure lacks {field}")
        if (
            before["available_blocks"] != before["cache_blocks"]
            or before["held_blocks"]
            or before["referenced_blocks"]
        ):
            raise ValueError("live pressure must be changed while Decode pool is idle")
        name = before["name"]
        if p["tokens"]:
            ctx.add_cleanup(f"live-pressure-{name}", lambda d: send(name, 0, d))
        evidence["response"] = send(name, p["tokens"], deadline)
        raw_after, engines = _live_engine_rows(ctx, deadline)
        evidence["after"] = raw_after
        after = _engines(raw_after, [name])[name]
        if any(
            after.get(k) != before[k]
            for k in (
                "available_blocks",
                "cache_blocks",
                "total_kv_tokens",
                "held_blocks",
                "referenced_blocks",
            )
        ):
            raise ValueError("reporting pressure changed physical Decode capacity")
        if after.get("available_kv_tokens") != max(
            0, before["total_kv_tokens"] - p["tokens"]
        ):
            raise ValueError("reporting pressure did not change reported availability")
        evidence["complete"] = True
    finally:
        path.write_text(json.dumps(evidence, indent=2) + "\n")
    return StageOutput(artifacts=[str(path)])


def _reserved_start_params(p, plan):
    guard = p.get("guard")
    wave = _live_start_params({k: v for k, v in p.items() if k != "guard"}, plan)
    guard = _params(
        guard,
        {
            "poll_s",
            "timeout_s",
            "http_timeout_s",
            "limit",
            "scan_limit",
            "endpoint_limit",
        },
        {
            "poll_s",
            "timeout_s",
            "http_timeout_s",
            "limit",
            "scan_limit",
            "endpoint_limit",
        },
    )
    for key, value in guard.items():
        if type(value) not in (int, float) or value <= 0:
            raise ValueError("master-local guard budgets must be positive")
    if len(wave["requests"]) != 2:
        raise ValueError("reserved live wave requires victim and incoming")
    return dict(wave, guard=guard)


def _reserved_start(ctx, p, deadline):
    from ...debug_client import DebugClient

    path = ctx.artifact_dir / f"preemption-live-master-local-{uuid.uuid4().hex}.json"
    evidence = {"attempts": [], "master_local": False}
    guard = p["guard"]

    def before_next(wave, item):
        limit = Deadline(
            min(deadline.expires_at, ctx.clock() + guard["timeout_s"]),
            ctx.clock,
            ctx.sleeper,
        )
        while True:
            limit.check()
            if item["error"] is not None:
                raise item["error"]
            entries = item["batch"].entries
            if entries:
                rid = entries[0]["record"]["wire_request_id"]
                capture = DebugClient(
                    f"http://127.0.0.1:{ctx.env.master_http_port}",
                    timeout_s=min(guard["http_timeout_s"], limit.remaining()),
                ).snapshot(
                    request_id=rid,
                    include="decode",
                    limit=guard["limit"],
                    scan_limit=guard["scan_limit"],
                    endpoint_limit=guard["endpoint_limit"],
                )
                raw, engines = _live_engine_rows(ctx, limit)
                evidence["attempts"].append(
                    dict(master=capture.payload, engines=raw, request_id=rid)
                )
                if (
                    capture.payload["status"] != "ok"
                    or capture.payload["endpointDirectoryTruncated"]
                ):
                    raise ValueError(
                        "master-local proof has incomplete endpoint coverage"
                    )
                pages = [
                    capture.component(k)
                    for k in capture.payload["components"]
                    if k.startswith("decode/")
                ]
                rows = [
                    row
                    for page in pages
                    for row in page["rows"]
                    if row["request_id"] == str(rid)
                ]
                if any(
                    not isinstance(e.get("request_lifecycle"), dict) for e in engines
                ):
                    raise ValueError(
                        "master-local proof lacks engine lifecycle inventory"
                    )
                if any(str(rid) in e["request_lifecycle"] for e in engines):
                    raise ValueError("victim reached engine before incoming was issued")
                if (
                    len(rows) == 1
                    and rows[0].get("queued") is True
                    and rows[0].get("owns_request") is True
                    and rows[0].get("ownership") == "reserved"
                    and rows[0].get("has_dispatch_permit") is False
                    and rows[0].get("has_protocol_owner") is False
                ):
                    evidence["master_local"] = True
                    return
                if item["done"].is_set():
                    raise ValueError(
                        "victim Schedule settled before master-local proof"
                    )
            limit.sleep(guard["poll_s"])

    try:
        result = _live_start(
            ctx, {k: v for k, v in p.items() if k != "guard"}, deadline, before_next
        )
        result.artifacts.append(str(path))
        return result
    finally:
        path.write_text(json.dumps(evidence, indent=2) + "\n")


def _live_start_params(p, plan):
    from .priority import _wave_params

    p = _wave_params(p, plan)
    if not p["defer_batch"] or p["serial_schedule"]:
        raise ValueError("live eviction requires concurrent deferred Schedule")
    return p


def _live_start(ctx, p, deadline, before_next=None):
    import threading

    from ..backend import RequestBatch

    if ctx.instance["environment"]["resolved_config"]["dispatcher"]["type"] != "BATCH":
        raise ValueError("live eviction requires BATCH dispatcher")
    wave = PriorityWave(ctx, p)
    handle = ctx.register_resource("requests", wave, cleanup=wave.cleanup)
    try:
        for index, shape in enumerate(p["requests"]):
            deadline.check()
            params = {k: v for k, v in shape.items() if k != "tag"}
            params.update(
                count=1, consume="deferred", schedule_timeout_s=90, stream_timeout_s=60
            )
            batch = RequestBatch(ctx, params)
            batch.artifact = (
                ctx.artifact_dir / f"priority-request-{uuid.uuid4().hex}.json"
            )
            item = dict(
                tag=shape["tag"], batch=batch, done=threading.Event(), error=None
            )
            wave.entries.append(item)
            end = min(ctx.instance_deadline_s, ctx.clock() + 90)
            item["thread"] = threading.Thread(
                target=wave._submit, args=(item, end), daemon=True
            )
            issued_at = ctx.clock()
            item["thread"].start()
            # The legacy live ThreadPoolExecutor sleeps only BETWEEN submissions.
            if index + 1 < len(p["requests"]):
                if before_next is not None:
                    before_next(wave, item)
                deadline.sleep(max(0, p["gap_s"] - (ctx.clock() - issued_at)))
    finally:
        wave.persist()
    return StageOutput({"requests": handle}, artifacts=[str(wave.path)])


def _live_drain_params(p, plan):
    p = _params(p, {"requests", "tags"}, {"requests", "tags"})
    plan.reference(p["requests"], "requests")
    if (
        not isinstance(p["tags"], list)
        or not p["tags"]
        or any(not isinstance(t, str) for t in p["tags"])
        or len(set(p["tags"])) != len(p["tags"])
    ):
        raise ValueError("live drain requires distinct survivor tags")
    return p


def _live_drain(ctx, p, deadline):
    wave = _cohort(ctx, p["requests"])
    if not wave.p.get("defer_batch"):
        raise ValueError("live eviction requires deferred BATCH consumers")
    for item in wave.entries:
        wave._join(item, deadline)
        if item["error"] is not None:
            raise item["error"]
    if any(r.get("fetch_invocations", 0) for r in wave.records()):
        raise ValueError("live eviction consumed before survivor drain")
    by_tag = {e["tag"]: e for e in wave.entries}
    if set(p["tags"]) - set(by_tag):
        raise ValueError("unknown live survivor tag")
    for tag in p["tags"]:
        batch = by_tag[tag]["batch"]
        if len(batch.entries) != 1:
            raise ValueError("live survivor requires one settled Schedule")
        entry = batch.entries[0]
        if entry["record"]["schedule"]["status"] != "OK":
            continue
        if not entry["response"].enqueued_by_master:
            raise ValueError("live survivor was not enqueued by Master")
        # Old start_stream has a 60s transport timeout and a separate 45s wait.
        batch.params["stream_timeout_s"] = 60
        batch._start_consumer(entry, min(ctx.instance_deadline_s, ctx.clock() + 60))
        batch._await_consumer(
            entry,
            Deadline(
                min(deadline.expires_at, ctx.clock() + 45), ctx.clock, ctx.sleeper
            ),
        )
        batch.persist()
    wave.complete = True
    wave.persist()
    return StageOutput(artifacts=[str(wave.path)])


def _live_engine_rows(ctx, deadline):
    raw = _http(ctx.ops, "snapshot", deadline)
    engines = raw.get("engines")
    env = ctx.instance["environment"]
    if (
        not isinstance(engines, list)
        or len(engines) != env["n_prefill"] + env["n_decode"]
    ):
        raise ValueError("live proof requires the complete declared engine fleet")
    if any(not isinstance(e.get("name"), str) or not e["name"] for e in engines) or len(
        {e.get("name") for e in engines}
    ) != len(engines):
        raise ValueError("duplicate live engine owner")
    for role in ("prefill", "decode"):
        if sum(e.get("role") == role for e in engines) != env["n_" + role]:
            raise ValueError("live engine role inventory mismatch")
    return raw, engines


def _live_prefill(ctx, p, deadline):
    ph_wave, wave = [_cohort(ctx, p[k]) for k in ("placeholder", "wave")]
    if not ph_wave.complete or not wave.complete:
        raise ValueError("live verdict requires drained survivor cohorts")
    ph, rows = ph_wave.records(), wave.records()
    if len(ph) != 1 or len(rows) != 3:
        raise ValueError(
            "live Prefill cohort must contain placeholder plus three requests"
        )
    for cohort, tags, priorities in (
        (ph_wave, ["placeholder"], [50]),
        (wave, ["victim_a", "victim_b", "incoming"], [30, 30, 70]),
    ):
        shapes = cohort.p["requests"]
        if (
            [r["tag"] for r in shapes] != tags
            or [r.get("priority") for r in shapes] != priorities
            or any((r["input_len"], r["output_len"]) != (2048, 2) for r in shapes)
        ):
            raise ValueError("live Prefill request shape differs from old contract")
    responses = [e["batch"].entries[0]["response"] for e in wave.entries]
    if any(r is None for r in responses):
        raise ValueError("live verdict lacks Schedule response")
    va, vb, inc = responses
    evicted = vb.code == 8400 and not vb.success
    survivors = [ph[0], rows[0], rows[2]]
    completed = all(request_success(r) for r in survivors)
    raw, engines = _live_engine_rows(ctx, deadline)
    if any(not isinstance(e.get("request_lifecycle"), dict) for e in engines):
        raise ValueError("live never-delivered proof lacks lifecycle inventory")
    rid = rows[1]["wire_request_id"]
    never_seen = all(str(rid) not in e["request_lifecycle"] for e in engines)
    metric = _reservation_metric(ctx, {"labels": {}}, deadline)
    snapshot = ctx.resource(metric.output["snapshot"], "snapshot")
    from ...engine_ops import parse_prometheus_samples

    samples = parse_prometheus_samples(snapshot["attempts"][-1]["body"], "")

    def total(name, labels):
        values = [
            _number(v, 0, 1e18)
            for n, ls, v in samples
            if name in n and all(ls.get(k) == v for k, v in labels.items())
        ]
        return sum(values) if values else None

    stage = {"stage": "prefill_queued"}
    victim = total("auto_tpm_victim_count", stage)
    tagged = total(
        "auto_tpm_victim_count",
        dict(stage, victim_priority="30", incoming_priority="70"),
    )
    preempt = total("auto_tpm_priority_preempt_count", stage)
    pr10 = (
        evicted
        and va.code == 200
        and va.success
        and inc.code == 200
        and inc.success
        and completed
    )
    pr5 = never_seen and evicted
    pr6 = evicted and victim == 1.0 and (preempt or 0.0) >= 1.0
    evidence = dict(
        placeholder=ph,
        wave=rows,
        schedule_codes=[r.code for r in responses],
        survivors_completed=completed,
        never_seen=never_seen,
        raw=raw,
        victim_total=victim,
        tagged_victim_diagnostic=tagged,
        preempt_total=preempt,
        PR10=pr10,
        PR5=pr5,
        PR6=pr6,
    )
    path = ctx.artifact_dir / f"preemption-live-prefill-{uuid.uuid4().hex}.json"
    path.write_text(json.dumps(evidence, indent=2) + "\n")
    return StageOutput(
        {"pr10": pr10, "pr5": pr5, "pr6": pr6}, artifacts=metric.artifacts + [str(path)]
    )


def _live_engine_clean(ctx, p, deadline):
    end = min(deadline.expires_at, ctx.clock() + p.get("seconds", 30))
    samples = []
    passed = False
    while ctx.clock() < end:
        raw, engines = _live_engine_rows(ctx, deadline)
        if any(type(e.get("leak_detected")) is not bool for e in engines):
            raise ValueError("missing engine leak flag")
        passed = all(
            _number(e.get("inflight"), 0, 1e9, True) == 0 and not e["leak_detected"]
            for e in engines
        )
        samples.append(dict(at_s=ctx.clock(), raw=raw, passed=passed))
        if passed:
            break
        deadline.sleep(min(0.5, max(0, end - ctx.clock())))
    path = ctx.artifact_dir / f"preemption-live-engine-clean-{uuid.uuid4().hex}.json"
    path.write_text(json.dumps(samples, indent=2) + "\n")
    return StageOutput({"passed": passed}, artifacts=[str(path)])


def _live_final_params(p, plan):
    keys = {"pr10", "pr5", "pr6", "engine_clean", "recovery"}
    p = _params(p, keys, keys)
    for k in p:
        plan.reference(p[k], "boolean")
    return p


def _live_final(ctx, p, deadline):
    actual = {k: ctx.resolve(v) for k, v in p.items()}
    checks = [
        ("PR10", actual["pr10"]),
        ("PR5", actual["pr5"]),
        ("PR6", actual["pr6"]),
        ("P6", actual["engine_clean"] and actual["recovery"]),
    ]
    return StageOutput(
        checks=[
            CheckResult(k, "PASS" if v else "FAIL", actual=v, expected=True)
            for k, v in checks
        ]
    )


def _live_reserved(ctx, p, deadline):
    ph_wave, wave = [_cohort(ctx, p[k]) for k in ("placeholder", "wave")]
    if not ph_wave.complete or not wave.complete:
        raise ValueError("live reserved verdict requires drained survivor cohorts")
    ph, rows = ph_wave.records(), wave.records()
    if len(ph) != 1 or len(rows) != 2:
        raise ValueError(
            "live reserved cohort must contain placeholder, victim and incoming"
        )
    for cohort, tags in (
        (ph_wave, ["placeholder"]),
        (wave, ["victim", "incoming"]),
    ):
        if [r["tag"] for r in cohort.p["requests"]] != tags:
            raise ValueError("live reserved cohort roles are out of order")
    responses = [e["batch"].entries[0]["response"] for e in wave.entries]
    if any(r is None for r in responses):
        raise ValueError("live reserved verdict lacks Schedule response")
    victim, inc = responses
    evicted = victim.code == 8400 and not victim.success
    completed = all(request_success(r) for r in (ph[0], rows[1]))
    raw, engines = _live_engine_rows(ctx, deadline)
    if any(not isinstance(e.get("request_lifecycle"), dict) for e in engines):
        raise ValueError("live never-delivered proof lacks lifecycle inventory")
    rid = rows[0]["wire_request_id"]
    never_seen = all(str(rid) not in e["request_lifecycle"] for e in engines)
    metric = _reservation_metric(ctx, {"labels": {}}, deadline)
    snapshot = ctx.resource(metric.output["snapshot"], "snapshot")
    from ...engine_ops import parse_prometheus_samples

    samples = parse_prometheus_samples(snapshot["attempts"][-1]["body"], "")

    def total(names, scale=1.0):
        values = [
            _number(value, 0, 1e18) * scale
            for metric_name, labels, value in samples
            if metric_name in names
            and labels.get("stage") == "decode_reserved"
            and "quantile" not in labels
            and "le" not in labels
        ]
        return sum(values) if values else None

    count = total(
        {"flexlb_auto_tpm_victim_count", "flexlb_auto_tpm_victim_count_total"}
    )
    # RequestSchedulerReporter registers KV tokens as TIMER; Micrometer records
    # that numeric value in milliseconds and exports the sum in seconds.
    # Buckets/count/max/quantiles are not additional tokens.
    kv = total({"flexlb_auto_tpm_victim_kv_tokens_seconds_sum"}, 1000.0)
    if kv is None:
        kv = total({"flexlb_auto_tpm_victim_kv_tokens"})
    pr10 = (
        evicted
        and victim.code != 8429
        and inc.code == 200
        and inc.success
        and completed
    )
    pr5 = never_seen and evicted
    pr6 = evicted and count == 1.0 and (kv or 0.0) >= 428.0
    evidence = dict(
        placeholder=ph,
        wave=rows,
        schedule_codes=[r.code for r in responses],
        survivors_completed=completed,
        never_seen=never_seen,
        raw=raw,
        victim_total=count,
        victim_kv_total=kv,
        PR10=pr10,
        PR5=pr5,
        PR6=pr6,
    )
    path = ctx.artifact_dir / f"preemption-live-reserved-{uuid.uuid4().hex}.json"
    path.write_text(json.dumps(evidence, indent=2) + "\n")
    return StageOutput(
        {"pr10": pr10, "pr5": pr5, "pr6": pr6}, artifacts=metric.artifacts + [str(path)]
    )


def _nf_state_params(p, plan):
    p = _params(p, {"requests", "state"}, {"requests", "state"})
    plan.reference(p["requests"], "requests")
    if p["state"] not in ("running", "finished"):
        raise ValueError("unknown NF construction state")
    return p


def _nf_state(ctx, p, deadline):
    wave = _cohort(ctx, p["requests"])
    rows = wave.records()
    if len(rows) != 1 or rows[0]["schedule"]["status"] != "OK":
        raise ValueError("NF construction requires one admitted victim")
    rid = str(rows[0]["wire_request_id"])
    samples = []
    path = ctx.artifact_dir / f"preemption-nf-{p['state']}-{uuid.uuid4().hex}.json"
    try:
        while True:
            deadline.check()
            raw, engines = _live_engine_rows(ctx, deadline)
            owners = [e for e in engines if e["role"] == "decode"]
            if len(owners) != 1 or not isinstance(
                owners[0].get("request_lifecycle"), dict
            ):
                raise ValueError("NF needs one real Decode lifecycle owner")
            lc = owners[0]["request_lifecycle"].get(rid, {})
            if not isinstance(lc, dict):
                raise ValueError("malformed NF victim lifecycle")
            end = lc.get("end_state")
            if end is not None and not isinstance(end, str):
                raise ValueError("malformed NF terminal state")
            if p["state"] == "running":
                matched = end == "running" or (
                    not end
                    and lc.get("running_ms") is not None
                    and _number(lc["running_ms"], 0, 1e18) > 0
                )
            else:
                # Old engine-finished probe accepts any non-running end state;
                # normal output and absence of cancellation are separate PR6 facts.
                matched = bool(end) and end != "running"
            samples.append(dict(at_s=ctx.clock(), raw=raw, matched=matched))
            if matched:
                break
            deadline.sleep(0.1 if p["state"] == "running" else 0.05)
    finally:
        path.write_text(json.dumps(samples, indent=2) + "\n")
    return StageOutput(artifacts=[str(path)])


def _cancel_census(ctx, p, deadline):
    path = ctx.artifact_dir / f"preemption-cancel-census-{uuid.uuid4().hex}.json"
    evidence = dict(env_epoch=ctx.env_epoch, counts={}, missing_cancel_keys=[])
    try:
        raw, engines = _live_engine_rows(ctx, deadline)
        evidence["raw"] = raw
        for engine in engines:
            rpc = engine.get("rpc_counts")
            if not isinstance(rpc, dict):
                raise ValueError("Cancel census lacks per-engine RPC map")
            if "cancel" not in rpc:
                evidence["missing_cancel_keys"].append(engine["name"])
            evidence["counts"][engine["name"]] = _number(
                rpc.get("cancel", 0), 0, 1e18, True
            )
        evidence["total"] = sum(evidence["counts"].values())
    except Exception as exc:
        evidence["error"] = repr(exc)
        raise
    finally:
        path.write_text(json.dumps(evidence, indent=2) + "\n")
    return StageOutput(
        {"snapshot": ctx.register_resource("snapshot", evidence, historical=True)},
        artifacts=[str(path)],
    )


def _nf_verdict_params(p, plan):
    p = _params(
        p,
        {"victim", "incoming", "before", "after"},
        {"victim", "incoming", "before", "after"},
    )
    for key in ("victim", "incoming"):
        plan.reference(p[key], "requests")
    for key in ("before", "after"):
        plan.reference(p[key], "snapshot")
    return p


def _nf_verdict(ctx, p, deadline):
    victim, incoming = [_cohort(ctx, p[k]) for k in ("victim", "incoming")]
    vr, ir = victim.records(), incoming.records()
    if not victim.complete or len(vr) != 1 or len(ir) != 1:
        raise ValueError(
            "NF verdict requires one drained victim and one settled incoming"
        )
    for wave, shape in ((victim, (30, 512, 200)), (incoming, (70, 512, 2))):
        shapes = wave.p["requests"]
        if (
            len(shapes) != 1
            or (
                shapes[0].get("priority"),
                shapes[0]["input_len"],
                shapes[0]["output_len"],
            )
            != shape
        ):
            raise ValueError("NF request shape differs from old contract")
    response = incoming.entries[0]["batch"].entries[0]["response"]
    if response is None:
        raise ValueError("NF incoming lacks Schedule response")
    before, after = [ctx.resource(p[k], "snapshot") for k in ("before", "after")]
    if (
        before.get("env_epoch") != ctx.env_epoch
        or after.get("env_epoch") != ctx.env_epoch
        or set(before["counts"]) != set(after["counts"])
    ):
        raise ValueError("Cancel census belongs to a different environment/fleet")
    delta = after["total"] - before["total"]
    rid = vr[0]["wire_request_id"]
    cancelled_by = []
    for engine in after["raw"]["engines"]:
        cancelled = engine.get("cancelled_rids")
        lc = engine.get("request_lifecycle")
        if (
            not isinstance(cancelled, list)
            or any(type(r) is not int for r in cancelled)
            or not isinstance(lc, dict)
        ):
            raise ValueError("NF cancellation proof lacks typed engine evidence")
        row = lc.get(str(rid), {})
        if not isinstance(row, dict):
            raise ValueError("malformed cancellation lifecycle")
        if rid in cancelled or row.get("end_state") == "cancelled":
            cancelled_by.append(engine["name"])
    completed = request_success(vr[0])
    rejected = response.code == 8431 and not response.success
    settled = rejected and completed and not cancelled_by
    evidence = dict(
        victim=vr,
        incoming=ir,
        before=before,
        after=after,
        incoming_code=response.code,
        victim_completed=completed,
        cancelled_by=cancelled_by,
        cancel_delta=delta,
        settled=settled,
        cancel_seen=delta >= 1,
    )
    path = ctx.artifact_dir / f"preemption-nf-verdict-{uuid.uuid4().hex}.json"
    path.write_text(json.dumps(evidence, indent=2) + "\n")
    return StageOutput(
        {"settled": settled, "cancel_seen": delta >= 1}, artifacts=[str(path)]
    )


def _nf_final_params(p, plan):
    keys = {"settled", "cancel_seen", "engine_clean", "recovery"}
    p = _params(p, keys, keys)
    for key in p:
        plan.reference(p[key], "boolean")
    return p


def _nf_final(ctx, p, deadline):
    values = {k: ctx.resolve(v) for k, v in p.items()}
    return StageOutput(
        checks=[
            CheckResult(k, "PASS" if passed else "FAIL", actual=passed, expected=True)
            for k, passed in (
                ("PR6", values["settled"]),
                ("AT5", values["cancel_seen"]),
                ("P6", values["engine_clean"] and values["recovery"]),
            )
        ]
    )


def _ts_first_output(ctx, p, deadline):
    wave = _cohort(ctx, p["requests"])
    rows = wave.records()
    shapes = wave.p["requests"]
    if (
        len(rows) != 1
        or len(shapes) != 1
        or (shapes[0].get("priority"), shapes[0]["input_len"], shapes[0]["output_len"])
        != (30, 512, 5000)
    ):
        raise ValueError("tombstoned victim shape differs from old contract")
    entry = wave.entries[0]["batch"].entries[0]
    if (
        entry["record"]["schedule"]["status"] != "OK"
        or not entry["response"].enqueued_by_master
    ):
        raise ValueError("tombstoned victim needs an admitted BATCH route")
    batch = wave.entries[0]["batch"]
    batch._start_consumer(entry, min(ctx.instance_deadline_s, ctx.clock() + 60))
    try:
        while (
            batch.snapshot_records()[0].get("stream", {}).get("first_output_s") is None
        ):
            deadline.sleep(0.02)
    finally:
        wave.persist()
    return StageOutput(artifacts=[str(wave.path)])


def _ts_restore_guard(ctx, p, deadline):
    owner = "prefill-0"
    before = _engines(_http(ctx.ops, "snapshot", deadline), [owner])[owner]
    if before["role"] != "prefill" or before["stopped"]:
        raise ValueError("tombstoned crash guard needs the original live prefill-0")
    epoch, env, ops = ctx.env_epoch, ctx.env, ctx.ops

    def restore(limit):
        if ctx.env_epoch != epoch or ctx.env is not env:
            raise ValueError("cannot restore a different environment")
        row = _engines(_http(ops, "snapshot", limit), [owner])[owner]
        if row["stopped"]:
            response = _http(ops, "start_engine", limit, dict(engine=owner))
            if response.get("status") != "ok" or response.get("engine") != owner:
                raise ValueError("crash cleanup lacks restart acknowledgement")

    ctx.add_cleanup("tombstoned-original-prefill-restore", restore)
    return StageOutput()


def _ts_trigger(ctx, p, deadline):
    from ..runtime import StageTimeout

    rid = ctx.ops.next_request_id()
    evidence = dict(
        request_id=rid, input_len=2048, output_len=10, code=None, error=None
    )
    holder = {}

    def cleanup(limit):
        if holder.get("call") is not None:
            holder["call"].cancel()

    ctx.add_cleanup("tombstoned-sacrificial-client", cleanup)
    path = ctx.artifact_dir / f"preemption-ts-trigger-{uuid.uuid4().hex}.json"
    try:
        stub = ctx.ops.schedule_pb2_grpc.FlexlbServiceStub(
            ctx.ops._channel(ctx.ops.master_target())
        )
        holder["call"] = stub.Schedule.future(
            ctx.ops.build_schedule_request(rid, input_len=2048, output_len=10),
            timeout=min(8, deadline.remaining()),
        )
        response = holder["call"].result(timeout=min(8, deadline.remaining()))
        evidence["code"] = int(response.code)
    except StageTimeout:
        raise
    except Exception as exc:
        deadline.check()
        evidence["error"] = repr(exc)
    finally:
        cleanup(deadline)
        holder.clear()
        path.write_text(json.dumps(evidence, indent=2) + "\n")
    return StageOutput(artifacts=[str(path)])


def _ts_health_params(p, plan):
    p = _params(p, {"state"}, {"state"})
    if p["state"] not in ("dropped", "restored"):
        raise ValueError("invalid crash health transition")
    return p


def _ts_health(ctx, p, deadline):
    from .status_protocol import _alive_count
    from .status_protocol import _http as status_http

    end = ctx.clock() + 30
    samples = []
    matched = False
    path = ctx.artifact_dir / f"preemption-ts-health-{uuid.uuid4().hex}.json"
    try:
        while ctx.clock() < end:
            _, raw = status_http(
                ctx, "master", "rtp_llm/master/info", deadline, body={}
            )
            sample = dict(at_s=ctx.clock(), raw=raw, matched=False)
            samples.append(sample)
            alive = _number(
                _alive_count(raw, "PREFILL"),
                0,
                1e9,
                True,
            )
            matched = alive <= 0 if p["state"] == "dropped" else alive >= 1
            sample.update(alive=alive, matched=matched)
            if matched:
                break
            deadline.sleep(min(0.5, max(0, end - ctx.clock())))
    finally:
        path.write_text(json.dumps(samples, indent=2) + "\n")
    return StageOutput(
        checks=[
            CheckResult(
                "health_transition",
                "PASS" if matched else "FAIL",
                actual=matched,
                expected=True,
            )
        ],
        artifacts=[str(path)],
    )


def _ts_cut(ctx, p, deadline):
    wave = _cohort(ctx, p["requests"])
    if len(wave.entries) != 1:
        raise ValueError("cut proof requires one victim")
    batch = wave.entries[0]["batch"]
    try:
        batch._await_consumer(batch.entries[0], deadline)
    finally:
        wave.persist()
    row = batch.snapshot_records()[0]
    cut = (
        row.get("business_finished") is False
        and row.get("cancel", {}).get("requested_s") is None
    )
    wave.complete = True
    path = ctx.artifact_dir / f"preemption-ts-cut-{uuid.uuid4().hex}.json"
    path.write_text(json.dumps(dict(row=row, cut=cut), indent=2) + "\n")
    return StageOutput({"cut": cut}, artifacts=[str(wave.path), str(path)])


def _ts_incoming(ctx, p, deadline):
    wave = _cohort(ctx, p["requests"])
    rows = wave.records()
    shapes = wave.p["requests"]
    if (
        not wave.complete
        or len(rows) != 1
        or len(shapes) != 1
        or (shapes[0].get("priority"), shapes[0]["input_len"], shapes[0]["output_len"])
        != (70, 512, 2)
    ):
        raise ValueError("tombstoned incoming lacks original shape and drain")
    return StageOutput(
        # DecodePreemptionCoordinator aborts before Cancel when the crashed
        # victim has a cancellation first cause (victim_inflight_gone).
        {
            "completed": request_success(rows[0]),
            "rejected": (
                rows[0]["schedule"]["status"] == "REJECTED"
                and rows[0]["schedule"]["error"] == "victim_inflight_gone"
                and _outcome(wave.entries[0], rows[0]) == (False, 8431)
            ),
        },
        artifacts=[str(wave.path)],
    )


def _ts_cancel_params(p, plan):
    p = _params(p, {"before"}, {"before"})
    plan.reference(p["before"], "snapshot")
    return p


def _ts_cancel(ctx, p, deadline):
    before = ctx.resource(p["before"], "snapshot")
    if before.get("env_epoch") != ctx.env_epoch:
        raise ValueError("Cancel baseline belongs to another environment")
    samples = []
    reached = False
    end = ctx.clock() + 15
    artifacts = []
    while ctx.clock() < end:
        result = _cancel_census(ctx, {}, deadline)
        artifacts.extend(result.artifacts)
        current = ctx.resource(result.output["snapshot"], "snapshot")
        if set(current["counts"]) != set(before["counts"]):
            raise ValueError("Cancel census engine fleet changed")
        samples.append(current)
        reached = current["total"] > before["total"]
        if reached:
            break
        deadline.sleep(min(0.5, max(0, end - ctx.clock())))
    # Preserve the legacy extra sample after wait_for, separate from reached.
    result = _cancel_census(ctx, {}, deadline)
    artifacts.extend(result.artifacts)
    after = ctx.resource(result.output["snapshot"], "snapshot")
    if set(after["counts"]) != set(before["counts"]):
        raise ValueError("Cancel census engine fleet changed")
    delta = after["total"] - before["total"]
    path = ctx.artifact_dir / f"preemption-ts-cancel-{uuid.uuid4().hex}.json"
    path.write_text(
        json.dumps(
            dict(
                before=before, polls=samples, after=after, reached=reached, delta=delta
            ),
            indent=2,
        )
        + "\n"
    )
    return StageOutput(
        # auto_tpm_cancel_final_design: retirement is ledger-only; Cancel is
        # exclusively a preemption transaction on the original Prefill.
        {
            "delta_zero": delta == 0,
            "census_stable": all(
                sample["counts"] == before["counts"] for sample in samples + [after]
            ),
        },
        artifacts=artifacts + [str(path)],
    )


def _ts_fence(ctx, p, deadline):
    from ..runtime import StageTimeout

    wave = _cohort(ctx, p["requests"])
    if len(wave.entries) != 1:
        raise ValueError("fence requires one original route")
    entry = wave.entries[0]["batch"].entries[0]
    rid = entry["record"]["wire_request_id"]
    response = entry["response"]
    if response is None:
        raise ValueError("fence lacks original Schedule route")
    target = ctx.ops.prefill_addr(response)
    if not target:
        raise ValueError("fence lacks original Prefill address")
    evidence = dict(request_id=rid, target=target, batch_id=rid * 10 + 1, passed=False)
    path = ctx.artifact_dir / f"preemption-ts-fence-{uuid.uuid4().hex}.json"
    try:
        inp = ctx.ops.build_generate_input(rid, output_len=2)
        ctx.ops._copy_role_addrs(inp, response)
        req = ctx.ops.pb2.EnqueueBatchRequestPB(
            batch_id=rid * 10 + 1,
            dp_slots=[
                ctx.ops.pb2.EnqueueBatchDpSlotPB(
                    dp_rank=0,
                    requests=[ctx.ops.pb2.EnqueueBatchExternalInputPB(input=inp)],
                )
            ],
            fetch_attach_timeout_ms=30000,
        )
        stub = ctx.ops.pb2_grpc.RpcServiceStub(ctx.ops._channel(target))
        ack = stub.EnqueueBatch(req, timeout=min(10, deadline.remaining()))
        errors = [
            dict(request_id=e.request_id, error_code=int(e.error_info.error_code))
            for e in ack.errors
        ]
        passed = (
            not ack.successes
            and len(errors)
            == 1  # No Cancel means no ABSENT_FENCE (model_rpc_service.proto).
            and
            # JavaMockEngineCluster's Decode cancel marker instead rejects
            # pre-alignment with 8211 after the Prefill restart.
            errors[0] == dict(request_id=rid, error_code=8211)
        )
        evidence.update(successes=len(ack.successes), errors=errors, passed=passed)
    except StageTimeout:
        raise
    except Exception as exc:
        deadline.check()
        evidence["error"] = repr(exc)
    finally:
        path.write_text(json.dumps(evidence, indent=2) + "\n")
    return StageOutput({"passed": evidence["passed"]}, artifacts=[str(path)])


def _ts_residue(ctx, p, deadline):
    from .status_protocol import _http as status_http

    samples = []
    first = None
    second = None
    end = ctx.clock() + 20
    path = ctx.artifact_dir / f"preemption-ts-residue-{uuid.uuid4().hex}.json"

    def sample():
        _, raw = status_http(ctx, "master", "rtp_llm/inflight_status", deadline)
        count = _number(raw.get("scheduler_inflight"), 0, 1e9, True)
        samples.append(dict(at_s=ctx.clock(), raw=raw, count=count))
        return count

    try:
        while ctx.clock() < end:
            first = sample()
            if first <= 1:
                break
            deadline.sleep(min(1, max(0, end - ctx.clock())))
        if first is not None and first <= 1:
            deadline.sleep(8)
            second = sample()
        passed = (
            first is not None and first <= 1 and second is not None and second <= first
        )
    finally:
        path.write_text(
            json.dumps(dict(samples=samples, first=first, second=second), indent=2)
            + "\n"
        )
    return StageOutput({"passed": passed}, artifacts=[str(path)])


def _ts_final_params(p, plan):
    keys = {
        "cut",
        "incoming",
        "delta_zero",
        "census_stable",
        "fence",
        "engine_clean",
        "residue",
        "recovery",
    }
    p = _params(p, keys, keys)
    for k in p:
        plan.reference(p[k], "boolean")
    return p


def _ts_final(ctx, p, deadline):
    v = {k: ctx.resolve(value) for k, value in p.items()}
    checks = [
        ("PR10", v["cut"] and v["incoming"] and v["delta_zero"]),
        ("PR6", v["fence"] and v["census_stable"]),
        ("P6", v["engine_clean"] and v["residue"] and v["recovery"]),
    ]
    return StageOutput(
        checks=[
            CheckResult(k, "PASS" if passed else "FAIL", actual=passed, expected=True)
            for k, passed in checks
        ]
    )


HANDLERS = [
    StageHandler("preemption_live_pressure", _live_pressure_params, _live_pressure, {}),
    StageHandler(
        "preemption_live_reserved_start",
        _reserved_start_params,
        _reserved_start,
        {"requests": "requests"},
    ),
    StageHandler("preemption_ts_first_output", _settled_params, _ts_first_output, {}),
    StageHandler(
        "preemption_ts_restore_guard",
        lambda p, plan: _params(p, (), ()),
        _ts_restore_guard,
        {},
    ),
    StageHandler(
        "preemption_ts_trigger", lambda p, plan: _params(p, (), ()), _ts_trigger, {}
    ),
    StageHandler(
        "preemption_ts_health",
        _ts_health_params,
        _ts_health,
        {},
        checks=frozenset({"health_transition"}),
    ),
    StageHandler("preemption_ts_cut", _settled_params, _ts_cut, {"cut": "boolean"}),
    StageHandler(
        "preemption_ts_incoming",
        _settled_params,
        _ts_incoming,
        {"completed": "boolean", "rejected": "boolean"},
    ),
    StageHandler(
        "preemption_ts_cancel",
        _ts_cancel_params,
        _ts_cancel,
        {"delta_zero": "boolean", "census_stable": "boolean"},
    ),
    StageHandler(
        "preemption_ts_fence", _settled_params, _ts_fence, {"passed": "boolean"}
    ),
    StageHandler(
        "preemption_ts_engine_clean",
        lambda p, plan: _params(p, (), ()),
        lambda ctx, p, d: _live_engine_clean(ctx, {"seconds": 60}, d),
        {"passed": "boolean"},
    ),
    StageHandler(
        "preemption_ts_residue",
        lambda p, plan: _params(p, (), ()),
        _ts_residue,
        {"passed": "boolean"},
    ),
    StageHandler(
        "preemption_ts_final",
        _ts_final_params,
        _ts_final,
        {},
        checks=frozenset({"PR10", "PR6", "P6"}),
    ),
    StageHandler("preemption_nf_state", _nf_state_params, _nf_state, {}),
    StageHandler(
        "preemption_cancel_census",
        lambda p, plan: _params(p, (), ()),
        _cancel_census,
        {"snapshot": "snapshot"},
    ),
    StageHandler(
        "preemption_nf_verdict",
        _nf_verdict_params,
        _nf_verdict,
        {"settled": "boolean", "cancel_seen": "boolean"},
    ),
    StageHandler(
        "preemption_nf_engine_clean",
        lambda p, plan: _params(p, (), ()),
        lambda ctx, p, d: _live_engine_clean(ctx, {"seconds": 20}, d),
        {"passed": "boolean"},
    ),
    StageHandler(
        "preemption_nf_final",
        _nf_final_params,
        _nf_final,
        {},
        checks=frozenset({"PR6", "AT5", "P6"}),
    ),
    StageHandler(
        "preemption_live_reserved",
        _same_params,
        _live_reserved,
        {"pr10": "boolean", "pr5": "boolean", "pr6": "boolean"},
    ),
    StageHandler(
        "preemption_live_start",
        _live_start_params,
        _live_start,
        {"requests": "requests"},
    ),
    StageHandler("preemption_live_drain", _live_drain_params, _live_drain, {}),
    StageHandler(
        "preemption_live_prefill",
        _same_params,
        _live_prefill,
        {"pr10": "boolean", "pr5": "boolean", "pr6": "boolean"},
    ),
    StageHandler(
        "preemption_live_engine_clean",
        lambda p, plan: _params(p, (), ()),
        _live_engine_clean,
        {"passed": "boolean"},
    ),
    StageHandler(
        "preemption_live_final",
        _live_final_params,
        _live_final,
        {},
        checks=frozenset({"PR10", "PR5", "PR6", "P6"}),
    ),
    StageHandler(
        "preemption_observability_duplicate",
        _settled_params,
        _observability_duplicate,
        {"rejected": "boolean"},
    ),
    StageHandler(
        "preemption_observability",
        _observability_params,
        _observability,
        {"client": "boolean", "planes": "boolean"},
    ),
    StageHandler(
        "preemption_observability_final",
        _observability_final_params,
        _observability_final,
        {},
        checks=frozenset({"AT8", "P6", "AT6"}),
    ),
    StageHandler(
        "preemption_reservation_metric",
        _reservation_metric_params,
        _reservation_metric,
        {"snapshot": "snapshot"},
    ),
    StageHandler(
        "preemption_reservation_half",
        _reservation_half_params,
        _reservation_half,
        {"passed": "boolean"},
    ),
    StageHandler(
        "preemption_reservation_final",
        _reservation_final_params,
        _reservation_final,
        {},
        checks=frozenset({"AT7", "P6"}),
    ),
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


def _absent_params(p, plan):
    keys = {
        "requests",
        "input_len",
        "output_len",
        "rpc_timeout_s",
        "poll_s",
        "fetch_attach_timeout_ms",
    }
    p = _params(p, keys, keys)
    plan.reference(p["requests"], "requests")
    for key in keys - {"requests"}:
        _number(p[key], 0.001, 60000)
    return p


def _absent_contract(ctx, p, deadline):
    """Engine fidelity guard, not a master behavior test.

    PrefillBatchRpcServer.cancelByPriorityPreemption and CancelStatusPB define
    never-seen -> TOMBSTONED -> 8429; completed -> NOT_FOUND with no fence.
    Production tombstones expire after ten minutes; mock uses a bounded set.
    This test covers the immediate contract, not TTL equivalence.
    """
    import re
    import time

    from google.protobuf.json_format import MessageToDict

    wave = _cohort(ctx, p["requests"])
    rows = wave.records()
    if len(rows) != 1 or not wave.complete or not request_success(rows[0]):
        raise ValueError("absent fence needs a completed control request")
    entry = wave.entries[0]["batch"].entries[0]
    route = entry["response"]
    done_rid = rows[0]["wire_request_id"]
    unknown_rid = ctx.ops.next_request_id()
    stub = ctx.ops.pb2_grpc.RpcServiceStub(
        ctx.ops._channel(ctx.ops.prefill_addr(route))
    )
    path = ctx.artifact_dir / f"preemption-absent-fence-{uuid.uuid4().hex}.json"
    evidence = {"completed_rid": done_rid, "never_seen_rid": unknown_rid}

    def census(after=None):
        # Existing aggregate telemetry is the sole unknown-branch counter;
        # snapshot.rpc_counts exposes only total Cancel RPCs.
        while True:
            deadline.check()
            lines = (ctx.env.run_dir / "mock_engine.log").read_text().splitlines()
            for line in reversed(lines):
                if not line.startswith("java_mock_stats "):
                    continue
                data = dict(re.findall(r"(\w+)=(\S+)", line))
                if after is None or int(data["ts_epoch_ms"]) > after:
                    return {"unknown": int(data["cancel_census_unknown"]), "raw": line}
            deadline.sleep(p["poll_s"])

    def enqueue(rid):
        inp = ctx.ops.build_generate_input(
            rid, input_len=p["input_len"], output_len=p["output_len"]
        )
        ctx.ops._copy_role_addrs(inp, route)
        req = ctx.ops.pb2.EnqueueBatchRequestPB(
            batch_id=ctx.ops.next_request_id(),
            dp_slots=[
                ctx.ops.pb2.EnqueueBatchDpSlotPB(
                    dp_rank=0,
                    requests=[ctx.ops.pb2.EnqueueBatchExternalInputPB(input=inp)],
                )
            ],
            fetch_attach_timeout_ms=p["fetch_attach_timeout_ms"],
        )
        return stub.EnqueueBatch(
            req, timeout=min(p["rpc_timeout_s"], deadline.remaining())
        )

    try:
        evidence["before"] = census()
        cancel = stub.Cancel(
            ctx.ops.pb2.CancelRequestPB(request_id=unknown_rid),
            timeout=min(p["rpc_timeout_s"], deadline.remaining()),
        )
        ack = enqueue(unknown_rid)
        # Wait for telemetry published after the RPC has installed the fence.
        sent_ms = time.time() * 1000
        evidence["unknown_cancel"] = MessageToDict(
            cancel, preserving_proto_field_name=True
        )
        evidence["unknown_enqueue"] = MessageToDict(
            ack, preserving_proto_field_name=True
        )
        evidence["after"] = census(sent_ms)
        absent = (
            cancel.status == ctx.ops.pb2.CANCEL_STATUS_TOMBSTONED
            and not ack.successes
            and len(ack.errors) == 1
            and ack.errors[0].request_id == unknown_rid
            and ack.errors[0].error_info.error_code == 8429
            and evidence["after"]["unknown"] - evidence["before"]["unknown"] == 1
        )
        cancel = stub.Cancel(
            ctx.ops.pb2.CancelRequestPB(request_id=done_rid),
            timeout=min(p["rpc_timeout_s"], deadline.remaining()),
        )
        ack = enqueue(done_rid)
        evidence["completed_cancel"] = MessageToDict(
            cancel, preserving_proto_field_name=True
        )
        evidence["completed_enqueue"] = MessageToDict(
            ack, preserving_proto_field_name=True
        )
        # Completed lifecycle installs no cancel marker on either role. A short
        # replay on the healthy, drained pool must be admitted, not just !=8429.
        completed = (
            cancel.status == ctx.ops.pb2.CANCEL_STATUS_NOT_FOUND
            and len(ack.successes) == 1
            and ack.successes[0].request_id == done_rid
            and not ack.errors
        )
        if ack.successes:
            call = stub.FetchResponse(
                ctx.ops.pb2.FetchRequestPB(request_id=done_rid),
                timeout=min(p["rpc_timeout_s"], deadline.remaining()),
            )
            try:
                outputs = list(call)
                evidence["completed_fetch"] = [
                    MessageToDict(x, preserving_proto_field_name=True) for x in outputs
                ]
                completed = (
                    completed
                    and any(any(x.flatten_output.finished) for x in outputs)
                    and not any(x.HasField("error_info") for x in outputs)
                )
            finally:
                call.cancel()
        evidence.update(absent=absent, completed=completed)
    finally:
        path.write_text(json.dumps(evidence, indent=2) + "\n")
    return StageOutput(
        checks=[
            CheckResult("absent_fence", "PASS" if absent else "FAIL"),
            CheckResult("completed_not_found", "PASS" if completed else "FAIL"),
        ],
        artifacts=[str(path)],
    )


HANDLERS.append(
    StageHandler(
        "preemption_absent_fence",
        _absent_params,
        _absent_contract,
        {},
        checks=frozenset({"absent_fence", "completed_not_found"}),
    )
)
