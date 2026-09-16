"""KV-full Decode scale-in construction, terminal classes and balance bounds."""

import json
import math
import threading
import time

from ..contracts import CheckResult, StageHandler, StageOutput


def fill_validate(params, plan):
    from .elastic import _validate
    from .elastic_lifecycle import _name

    p = _validate(params, plan, {"victim", "survivor"}, {"victim", "survivor"})
    for value in p.values():
        _name(value, plan)
    return p


def saturated(entry):
    fields = [
        "cache_blocks",
        "available_blocks",
        "block_size",
        "total_kv_tokens",
        "running",
    ]
    if any(type(entry.get(k)) is not int or entry[k] < 0 for k in fields):
        raise ValueError("full-shrink requires actual pool and running counters")
    total, available, block = (
        entry["cache_blocks"],
        entry["available_blocks"],
        entry["block_size"],
    )
    if (
        not 0 <= available <= total
        or block <= 0
        or entry["total_kv_tokens"] != total * block
    ):
        raise ValueError("inconsistent full-shrink pool capacity")
    return (
        total == 24
        and available >= 2
        and entry["running"] > 0
        and ((total - available) * block + 2048) * 100 > 90 * entry["total_kv_tokens"]
    )


def fill(ctx, params, deadline):
    from .elastic import RecordedRequests, _snapshot

    victim, survivor = (ctx.resolve(params[k]) for k in ("victim", "survivor"))
    records = RecordedRequests(ctx.ops, ctx.env_epoch, ctx.clock)
    records.full_events = []
    records.full_timeouts = []
    path = ctx.artifact_dir / f"elastic-full-fill-{time.time_ns()}.json"
    end = ctx.clock() + 60

    class FillDeadline:
        def remaining(self):
            outer = deadline.remaining()
            value = min(outer, end - ctx.clock())
            if value <= 0:
                raise TimeoutError("full-shrink fill budget expired")
            return value

        def check(self):
            self.remaining()

    local = FillDeadline()

    def finish(d):
        records.cancel_active("full_fill_cleanup")
        try:
            for event in records.full_events:
                if not event.wait(d.remaining()):
                    raise TimeoutError("full fill consumer did not exit")
        finally:
            path.with_name(path.stem + "-cleanup.json").write_text(
                json.dumps(records.snapshot_records(), indent=2)
            )

    handle = ctx.register_resource("requests", records, cleanup=finish)
    latest = {}
    reached = False
    try:
        # Eight submissions then 100ms pause, with a snapshot after every
        # admitted Schedule. 4800 is the maximum implied by 60s/.1s*8.
        for index in range(4800):
            if ctx.clock() >= end:
                break
            schedule_cap = min(30, local.remaining())
            rid = ctx.ops.next_request_id()
            record = records.issue(rid, ctx.clock)
            event = threading.Event()
            records.full_events.append(event)

            def run(record=record, event=event, schedule_cap=schedule_cap):
                try:
                    rid = record["wire_request_id"]
                    records.run(
                        record,
                        dict(
                            input_len=2035,
                            output_len=13,
                            block_keys=[rid * 100, rid * 100 + 1],
                        ),
                        timeout_s=90,
                        schedule_timeout_s=schedule_cap,
                        stream_timeout_s=60,
                    )
                finally:
                    event.set()

            try:
                threading.Thread(
                    target=run, name=f"elastic-full-fill-{rid}", daemon=True
                ).start()
            except BaseException:
                event.set()
                raise
            while True:
                row = records.snapshot_records()[-1]
                if row.get("prefill_addr") or row["consumer_exit_s"] is not None:
                    break
                remaining = local.remaining()
                deadline.sleep(min(0.01, remaining))
            if row.get("prefill_addr"):
                latest = _snapshot(ctx, local)
                if victim not in latest or survivor not in latest:
                    raise ValueError("full-shrink victim/survivor missing")
                if (
                    latest[victim].get("role") != "decode"
                    or latest[survivor].get("role") != "decode"
                ):
                    raise ValueError("full-shrink targets must be Decode workers")
                reached = saturated(latest[victim])
                if reached:
                    break
            if index % 8 == 7:
                remaining = end - ctx.clock()
                if remaining <= 0:
                    break
                deadline.sleep(min(0.1, remaining))
        result = dict(
            victim=victim,
            survivor=survivor,
            snapshot=latest,
            filled_s=ctx.clock(),
            saturated=reached,
        )
        return StageOutput(
            output=dict(
                requests=handle,
                fill=ctx.register_resource("snapshot", result, historical=True),
            ),
            checks=[
                CheckResult("saturated", "PASS" if reached else "FAIL", evidence=result)
            ],
            artifacts=[str(path)],
        )
    finally:
        path.write_text(
            json.dumps(
                dict(
                    victim=victim,
                    survivor=survivor,
                    snapshot=latest,
                    saturated=reached,
                    records=records.snapshot_records(),
                ),
                indent=2,
            )
        )


def collect_validate(params, plan):
    from .elastic import _validate

    p = _validate(params, plan, {"requests", "mutation"}, {"requests", "mutation"})
    plan.reference(p["requests"], "requests")
    plan.reference(p["mutation"], "snapshot")
    return p


def terminal_kind(record):
    from .elastic import request_success

    if (
        record["consumer_exit_s"] is None
        or record["transport_terminal_s"] is None
        or record["cancel"]["requested_s"] is not None
    ):
        return "hang"
    if request_success(record):
        return "completed"
    if record["stream"]["status"] not in (None, "OK"):
        return "rpc_error"
    if record["business_error_code"] not in (None, 0):
        return "error"
    return "empty"


def collect(ctx, params, deadline):
    records = ctx.resource(params["requests"], "requests")
    mutation = ctx.resource(params["mutation"], "snapshot")
    for record, event in zip(records.snapshot_records(), records.full_events):
        if not event.wait(min(50, deadline.remaining())):
            records.full_timeouts.append(record["wire_request_id"])
            with records._lock:
                active = records._calls.get(record["wire_request_id"])
                if active:
                    records._cancel_call(*active, "full_collection_timeout")
            if not event.wait(min(5, deadline.remaining())):
                raise TimeoutError("full collector lacks consumer exit")
    rows = records.snapshot_records()
    result = dict(
        records=rows,
        timed_out=list(records.full_timeouts),
        mutation=mutation,
        collected_s=ctx.clock(),
    )
    path = ctx.artifact_dir / f"elastic-full-collected-{time.time_ns()}.json"
    path.write_text(json.dumps(result, indent=2))
    return StageOutput(
        output=dict(result=ctx.register_resource("snapshot", result, historical=True)),
        artifacts=[str(path)],
    )


def terminal_validate(params, plan):
    from .elastic import _validate

    p = _validate(
        params,
        plan,
        {"result", "branch", "data_error_codes", "data_error_tokens"},
        {"result", "branch"},
    )
    plan.reference(p["result"], "snapshot")
    if p["branch"] not in ("drain_ok", "drain_timeout"):
        raise ValueError("unsupported drain branch")
    codes, tokens = p.get("data_error_codes", []), p.get("data_error_tokens", [])
    if not isinstance(codes, list) or any(type(c) is not int or c < 1 for c in codes):
        raise ValueError("data_error_codes must be typed codes")
    if not isinstance(tokens, list) or any(
        not isinstance(t, str) or not t for t in tokens
    ):
        raise ValueError("data_error_tokens must be nonempty strings")
    if bool(codes) != bool(tokens) or (codes and p["branch"] != "drain_timeout"):
        raise ValueError(
            "data-plane errors require a drain_timeout code/message contract"
        )
    return p


def terminal(ctx, params, deadline):
    deadline.check()
    data = ctx.resource(params["result"], "snapshot")
    mutation = data["mutation"]
    removed = mutation["started_s"]
    response = mutation["response"]
    admitted = [
        r
        for r in data["records"]
        if r["schedule"]["status"] == "OK" and r.get("prefill_addr")
    ]
    rows = []
    for r in admitted:
        kind = "hang" if r["wire_request_id"] in data["timed_out"] else terminal_kind(r)
        stamp = r["transport_terminal_s"]
        code = r["business_error_code"]
        message = r["business_error_message"] or ""
        pre_refusal = (
            kind == "error" and code == 8211 and stamp is not None and stamp <= removed
        )
        retired = (
            kind == "error"
            and code == 8510
            and "Decode endpoint generation retired" in message
        )
        data_error = (
            params["branch"] == "drain_timeout"
            and kind == "error"
            and code in params.get("data_error_codes", [])
            and any(token in message for token in params.get("data_error_tokens", []))
            and stamp is not None
            and stamp >= removed
        )
        allowed = (
            kind == "completed"
            or pre_refusal
            or (params["branch"] == "drain_timeout" and (retired or data_error))
        )
        rows.append(
            dict(
                rid=r["wire_request_id"],
                kind=kind,
                code=code,
                message=message,
                pre_fill_refusal=pre_refusal,
                retired=retired,
                data_error=data_error,
                allowed=allowed,
                within_40s=stamp is not None
                and stamp - removed <= 40
                and kind != "hang",
            )
        )
    drain_ms = response.get("drain_ms")
    if (
        type(drain_ms) not in (int, float)
        or not math.isfinite(drain_ms)
        or drain_ms < 0
    ):
        raise ValueError("remove response lacks drain milliseconds")
    branch_ok = (
        response.get("drained") is True
        if params["branch"] == "drain_ok"
        else response.get("drained") is False and 5000 <= drain_ms <= 10000
    )
    checks = dict(
        nonempty_admitted=bool(rows),
        drain_branch=branch_ok,
        terminal_40s=bool(rows) and all(r["within_40s"] for r in rows),
        terminal_family=bool(rows) and all(r["allowed"] for r in rows),
        retirement_contract=(
            all(not r["retired"] for r in rows)
            if params["branch"] == "drain_ok"
            else all(r["allowed"] for r in rows)
            and any(r["retired"] or r["data_error"] for r in rows)
        ),
    )
    evidence = dict(
        branch=params["branch"],
        drain_response=response,
        outcomes=rows,
        schedule_rejects=[
            r["wire_request_id"] for r in data["records"] if r not in admitted
        ],
    )
    path = ctx.artifact_dir / f"elastic-full-terminal-{time.time_ns()}.json"
    path.write_text(json.dumps(evidence, indent=2))
    return StageOutput(
        checks=[
            CheckResult(k, "PASS" if ok else "FAIL", evidence=evidence)
            for k, ok in checks.items()
        ],
        artifacts=[str(path)],
    )


def steady_validate(params, plan):
    from .elastic import _validate
    from .elastic_lifecycle import _name

    p = _validate(
        params,
        plan,
        {"baseline", "steady", "newcomer"},
        {"baseline", "steady", "newcomer"},
    )
    for k in ("baseline", "steady"):
        plan.reference(p[k], "snapshot")
    _name(p["newcomer"], plan)
    return p


def steady(ctx, params, deadline):
    from .elastic_balance import optional_stat, points, shares, spread

    deadline.check()
    base, ss = (ctx.resource(params[k], "snapshot") for k in ("baseline", "steady"))
    names = ["decode-1", ctx.resolve(params["newcomer"])]
    bs = shares(base, ["decode-0", "decode-1"])
    if bs is None:
        raise ValueError("full-shrink baseline has no Decode traffic")
    ss_share = shares(ss, names)
    tail = ss["start_s"] + 40
    occ = points(ss, names, "occupancy", tail)
    depth = points(ss, names, "mock_engine_waiting", tail)
    occ_cap = spread(points(base, ["decode-0", "decode-1"], "occupancy")) + 0.05
    cap = max(max(bs.values()) + 0.10, 0.65)
    checks = dict(
        nonempty_share=ss_share is not None,
        share_max=ss_share is not None and max(ss_share.values()) <= cap,
        share_min=ss_share is not None and min(ss_share.values()) >= 0.10,
        occupancy_spread=spread(occ) <= occ_cap,
        waiting_peak=max(v for seq in depth.values() for _, v in seq) <= 2,
    )

    def tps():
        total = 0
        for name, role in {
            n: r["role"]
            for s in ss["data"]["samples"]
            if tail <= s["time_s"] <= ss["end_s"]
            for n, r in s["engines"].items()
        }.items():
            metric = (
                "rtp_llm_context_tps" if role == "prefill" else "rtp_llm_generate_tps"
            )
            vals = points(ss, [name], metric, tail)[name]
            total += sum(v for _, v in vals) / len(vals)
        return total

    evidence = dict(
        share=ss_share,
        share_cap=cap,
        occupancy_spread=spread(occ),
        occupancy_cap=occ_cap,
        steady_cluster_tps=optional_stat(tps),
    )
    path = ctx.artifact_dir / f"elastic-full-steady-{time.time_ns()}.json"
    path.write_text(json.dumps(evidence, indent=2))
    return StageOutput(
        checks=[
            CheckResult(k, "PASS" if ok else "FAIL", evidence=evidence)
            for k, ok in checks.items()
        ],
        artifacts=[str(path)],
    )


def transient_validate(params, plan):
    from .elastic import _validate

    p = _validate(
        params,
        plan,
        {"observation", "fill", "mutation"},
        {"observation", "fill", "mutation"},
    )
    for k in p:
        plan.reference(p[k], "observation" if k == "observation" else "snapshot")
    return p


def transient(ctx, params, deadline):
    from .elastic_balance import deltas, points

    deadline.check()
    fill = ctx.resource(params["fill"], "snapshot")
    mutation = ctx.resource(params["mutation"], "snapshot")
    start = mutation["started_s"]
    end = start + 20
    data = ctx.resource(params["observation"], "observation").snapshot()
    if ctx.clock() < end:
        raise ValueError("full transient window has not elapsed")
    window = dict(start_s=start, end_s=end, data=data)
    name = fill["survivor"]
    snapshot = fill["snapshot"]
    victim = snapshot[fill["victim"]]
    survivor = snapshot[name]
    demand = victim["cache_blocks"] - victim["available_blocks"]
    free = survivor["available_blocks"]
    cap = math.ceil(max(0, demand - free))
    occ = points(window, [name], "occupancy")[name]
    rejects = sum(
        deltas(points(window, [name], metric))[name]
        for metric in [
            "mock_engine_lack_mem_rejects_total",
            "mock_engine_kv_admission_fails_total",
        ]
    )
    evidence = dict(
        window=[start, end],
        survivor=name,
        victim_occupied=demand,
        survivor_free=free,
        k_reject=cap,
        reject_delta=rejects,
        occupancy_peak=max(v for _, v in occ),
    )
    path = ctx.artifact_dir / f"elastic-full-transient-{time.time_ns()}.json"
    path.write_text(json.dumps(evidence, indent=2))
    return StageOutput(
        checks=[
            CheckResult(
                "occupancy_peak",
                "PASS" if evidence["occupancy_peak"] <= 0.95 else "FAIL",
                evidence=evidence,
            ),
            CheckResult(
                "reject_bound", "PASS" if rejects <= cap else "FAIL", evidence=evidence
            ),
        ],
        artifacts=[str(path)],
    )


HANDLERS = [
    StageHandler(
        "elastic_decode_fill",
        fill_validate,
        fill,
        {"requests": "requests", "fill": "snapshot"},
        checks=frozenset({"saturated"}),
    ),
    StageHandler(
        "elastic_full_collect", collect_validate, collect, {"result": "snapshot"}
    ),
    StageHandler(
        "elastic_full_terminal",
        terminal_validate,
        terminal,
        {},
        checks=frozenset(
            {
                "nonempty_admitted",
                "drain_branch",
                "terminal_40s",
                "terminal_family",
                "retirement_contract",
            }
        ),
    ),
    StageHandler(
        "elastic_full_steady",
        steady_validate,
        steady,
        {},
        checks=frozenset(
            {
                "nonempty_share",
                "share_max",
                "share_min",
                "occupancy_spread",
                "waiting_peak",
            }
        ),
    ),
    StageHandler(
        "elastic_full_transient",
        transient_validate,
        transient,
        {},
        checks=frozenset({"occupancy_peak", "reject_bound"}),
    ),
]
