"""Live HA checkpoints with request accounting and separate owner measurements."""

import math
import time
from collections import Counter

from ...ha import LiveClientEvents
from ..contracts import CheckResult, StageHandler, StageOutput
from .engine_control import _http
from .kv import _artifact
from .master import _endpoint_loads, _master_json, _params, _process


def _journal(client):
    if client.flow._overrides.get("LIVE_CLIENT_EVENTS") != "true":
        raise ValueError("checkpoint requires live client events")
    if not hasattr(client, "journal"):
        client.journal = LiveClientEvents(
            client.flow.out_dir / "client_lifecycle.jsonl"
        )
    client.journal.read()
    return client.journal


def _number(p, name, default, lower, upper):
    p.setdefault(name, default)
    if (
        type(p[name]) not in (int, float)
        or not math.isfinite(p[name])
        or not lower <= p[name] <= upper
    ):
        raise ValueError(f"invalid checkpoint {name}")


def validate(params, plan):
    p = _params(
        params,
        plan,
        {
            "client",
            "target",
            "duration_s",
            "min_samples",
            "min_success",
            "min_target_share",
            "max_prefill_share",
            "max_owner_load",
            "fault",
            "baseline",
        },
        {"client", "target"},
    )
    plan.reference(p["client"], "ha_client")
    if getattr(plan, "environment", {}).get("master_layout") != "dual_standalone" or p[
        "target"
    ] not in {"A", "B"}:
        raise ValueError("checkpoint requires explicit dual master target")
    for field in ("fault", "baseline"):
        if field in p:
            plan.reference(p[field], "master_fault" if field == "fault" else "snapshot")
    for field, default, low, high in [
        ("duration_s", 5, 2, 30),
        ("min_samples", 30, 10, 1000),
        ("min_success", 0.95, 0, 1),
        ("min_target_share", 1, 0, 1),
        ("max_prefill_share", 0.75, 0.5, 1),
        ("max_owner_load", 128, 1, 5000),
    ]:
        _number(p, field, default, low, high)
    return p


def _percentile(values, fraction):
    return (
        sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)]
        if values
        else None
    )


def measurements(rows, pools):
    successful = [r for r in rows if r["status"] == "ok"]
    distribution = {}
    for role in ("prefill", "decode"):
        counts = dict.fromkeys(pools[role], 0)
        for row in successful:
            address = row.get(role)
            if address not in counts:
                raise ValueError(f"successful request has unknown {role} endpoint")
            counts[address] += 1
        total = sum(counts.values())
        distribution[role] = {
            "counts": counts,
            "max_share": max(counts.values()) / total if total else None,
        }
    ttft = [r.get("ttft_ms") for r in successful]
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in ttft):
        raise ValueError("successful request lacks valid TTFT")
    return {
        "sample_count": len(rows),
        "success_rate": len(successful) / len(rows) if rows else 0,
        "distribution": distribution,
        "ttft_p50_ms": _percentile(ttft, 0.5),
        "ttft_p95_ms": _percentile(ttft, 0.95),
        "failed": [r["rid"] for r in rows if r["status"] != "ok"],
    }


def _pools(ctx, deadline, address_field="grpc_addr"):
    raw = _http(ctx.ops, "snapshot", deadline)
    pools = {
        role: sorted(r[address_field] for r in raw["engines"] if r["role"] == role)
        for role in ("prefill", "decode")
    }
    expected = {"prefill": ctx.env.spec.n_prefill, "decode": ctx.env.spec.n_decode}
    if any(
        len(pools[role]) != expected[role] or len(set(pools[role])) != expected[role]
        for role in pools
    ):
        raise ValueError("checkpoint mock pool differs from configured topology")
    return pools


def checkpoint(ctx, p, deadline):
    client = ctx.resource(p["client"], "ha_client")
    journal = _journal(client)
    started = time.time()
    fault = ctx.resource(p["fault"], "master_fault") if "fault" in p else None
    lower = fault.started_epoch_s if fault else started
    if lower <= 0:
        raise ValueError("fault has no wall-clock boundary")
    end_clock = ctx.clock() + p["duration_s"]
    upper = None
    samples, cohort, rows = [], {}, []
    evidence = {"target": p["target"], "lower_epoch_s": lower, "samples": samples}
    path = None
    try:
        pools = _pools(ctx, deadline)
        http_pools = _pools(ctx, deadline, "http_addr")
        evidence.update(rpc_pool=pools, http_pool=http_pools)
        info = _master_json(ctx, p["target"], "/rtp_llm/master/info", deadline, True)
        expected = {"PREFILL": ctx.env.spec.n_prefill, "DECODE": ctx.env.spec.n_decode}
        topology = info.get("ready") is True and all(
            info["worker_summary"][role]["alive"]
            == info["worker_summary"][role]["discovered"]
            == count
            for role, count in expected.items()
        )
        while True:
            deadline.check()
            _process(ctx, p["target"])
            journal.read()
            if upper is None and ctx.clock() >= end_clock:
                upper = time.time()
            inflight = _master_json(
                ctx, p["target"], "/rtp_llm/inflight_status", deadline
            )
            scheduler = inflight["scheduler_inflight"]
            if type(scheduler) is not int or scheduler < 0:
                raise ValueError("missing scheduler owner count")
            loads = _endpoint_loads(inflight)
            if any(len(loads[role]) != expected[role.upper()] for role in loads):
                raise ValueError("active master endpoint pool is incomplete")
            evidence["latest_inflight"] = inflight
            for role in pools:
                if (
                    sorted(r["ip_port"] for r in inflight[role + "_endpoints"])
                    != http_pools[role]
                ):
                    raise ValueError(
                        "active master endpoint addresses differ from the mock pool"
                    )
            samples.append(
                {
                    "epoch_s": time.time(),
                    "scheduler_inflight": scheduler,
                    "prefill_batches": loads["prefill"],
                    "decode_loads": loads["decode"],
                    "raw": inflight,
                }
            )
            cohort = journal.cohort(
                lower * 1000,
                (upper or time.time()) * 1000,
                transition=fault is not None,
            )
            missing = sorted(set(cohort) - set(journal.terminal))
            if upper is not None and (not missing or deadline.remaining() < 1):
                break
            if client.flow.proc is None or not client.flow.proc.alive():
                raise RuntimeError("HA client ended before its checkpoint completed")
            deadline.sleep(min(0.5, deadline.remaining()))
        rows = [journal.terminal[rid] for rid in cohort if rid in journal.terminal]
        values = measurements(rows, pools)
        target = ctx.backend.manager.master_instance_target(ctx.env, p["target"])
        share = (
            sum(
                r["route_path"] == "master" and r["master_target"] == target
                for r in rows
            )
            / len(rows)
            if rows
            else 0
        )
        retry = sum(r["failover"] for r in rows)
        peaks = {
            "scheduler_requests": max(s["scheduler_inflight"] for s in samples),
            "prefill_batches": max(sum(s["prefill_batches"]) for s in samples),
            "decode_load": max(sum(s["decode_loads"]) for s in samples),
        }
        values.update(
            target_share=share,
            failover_count=retry,
            owner_peaks=peaks,
            missing_terminals=missing,
            pool=pools,
        )
        if "baseline" in p:
            baseline = ctx.resource(p["baseline"], "snapshot")
            if baseline["pool"] != pools:
                raise ValueError("HA masters did not observe the same engine pool")
            values["relative_to_baseline"] = {
                "ttft_p95_ratio": (
                    values["ttft_p95_ms"] / baseline["ttft_p95_ms"]
                    if values["ttft_p95_ms"] is not None and baseline["ttft_p95_ms"]
                    else None
                ),
                "success_rate_delta": values["success_rate"] - baseline["success_rate"],
                "prefill_max_share_delta": (
                    values["distribution"]["prefill"]["max_share"]
                    - baseline["distribution"]["prefill"]["max_share"]
                    if values["distribution"]["prefill"]["max_share"] is not None
                    and baseline["distribution"]["prefill"]["max_share"] is not None
                    else None
                ),
            }
        balance = values["distribution"]["prefill"]["max_share"]
        checks = [
            ("topology", topology, info["worker_summary"], expected),
            (
                "accounting",
                not missing and len(cohort) >= p["min_samples"],
                {"issued": len(cohort), "terminal": len(rows), "missing": missing},
                {"minimum": p["min_samples"], "missing": 0},
            ),
            ("route", share >= p["min_target_share"], share, p["min_target_share"]),
            (
                "success",
                values["success_rate"] >= p["min_success"],
                values["success_rate"],
                p["min_success"],
            ),
            (
                "balance",
                balance is not None and balance <= p["max_prefill_share"],
                values["distribution"],
                p["max_prefill_share"],
            ),
            (
                "owner_bound",
                all(v <= p["max_owner_load"] for v in peaks.values()),
                peaks,
                p["max_owner_load"],
            ),
            (
                "retry",
                retry >= (1 if fault else 0),
                retry,
                ">=1" if fault else "observed",
            ),
        ]
        evidence.update(
            upper_epoch_s=upper,
            metrics=values,
            issued=list(cohort.values()),
            terminal=rows,
            checks=[
                {"id": name, "passed": passed, "actual": actual, "expected": want}
                for name, passed, actual, want in checks
            ],
        )
    finally:
        evidence.setdefault("issued", list(cohort.values()))
        evidence.setdefault("terminal", rows)
        path = _artifact(ctx, "ha-checkpoint", evidence)
    return StageOutput(
        {"snapshot": ctx.register_resource("snapshot", values, historical=True)},
        [
            CheckResult(
                name, "PASS" if passed else "FAIL", actual=actual, expected=want
            )
            for name, passed, actual, want in checks
        ],
        [path],
    )


def _probe_validate(params, plan):
    p = _params(
        params,
        plan,
        {"requests", "max_prefill_share", "mode", "max_consecutive"},
        {"requests"},
    )
    plan.reference(p["requests"], "requests")
    _number(p, "max_prefill_share", 0.75, 0.5, 1)
    p.setdefault("mode", "share")
    if p["mode"] not in {"share", "avoidance", "rotation", "recovery"}:
        raise ValueError("invalid probe distribution mode")
    if p["mode"] == "rotation" and (
        type(p.get("max_consecutive")) is not int or p["max_consecutive"] < 1
    ):
        raise ValueError("rotation requires explicit max_consecutive")
    return p


def probe_distribution(ctx, p, deadline):
    from .elastic import request_success

    rows = ctx.resource(p["requests"], "requests").snapshot_records()
    pools = _pools(ctx, deadline)
    counts = dict.fromkeys(pools["prefill"], 0)
    for row in rows:
        if request_success(row):
            if row.get("prefill_addr") not in counts:
                raise ValueError("probe has unknown Prefill endpoint")
            counts[row["prefill_addr"]] += 1
    total = sum(counts.values())
    share = max(counts.values()) / total if total else 1
    actual = {
        "issued": len(rows),
        "completed": total,
        "prefill": counts,
        "max_share": share,
    }
    good = (
        len(rows) >= 20
        and total / len(rows) >= 0.95
        and share <= p["max_prefill_share"]
    )
    if p.get("mode") == "recovery":
        # Concurrent traffic from the other master makes engine-arrival busy
        # intervals unsuitable as a scheduling-decision oracle. This phase
        # proves the restarted master can serve requests; the later isolated
        # rotation phase separately checks distribution.
        good = len(rows) >= 20 and total / len(rows) >= 0.95
    if p.get("mode", "share") in {"avoidance", "rotation"}:
        from .execution_evidence import (
            avoidance,
            execution_batches,
            read_events,
            rotation,
        )

        events = read_events(ctx.env.run_dir / "engine_events.jsonl")
        if events is None:
            raise ValueError("probe distribution requires execution sidecar")
        ids = {r["wire_request_id"] for r in rows if request_success(r)}
        batches = execution_batches(events, ids)
        healthy = len(rows) >= 20 and total / len(rows) >= 0.95
        if p["mode"] == "avoidance":
            engine_names = sorted(
                {e["engine_name"] for e in events if e.get("event") == "prefill_done"}
            )
            detail = avoidance(events, ids, engine_names)
            good = healthy and detail["violations"] == 0
        else:
            detail = rotation(batches, p["max_consecutive"])
            detail["batches"] = batches
            good = good and healthy and detail["passed"]
        actual["execution"] = detail
    return StageOutput(
        checks=[
            CheckResult(
                "balance",
                "PASS" if good else "FAIL",
                actual=actual,
                expected={
                    "min_success": 0.95,
                    **(
                        {"valid_prefill_endpoints": True}
                        if p.get("mode") == "recovery"
                        else (
                            {"violations": 0}
                            if p.get("mode") == "avoidance"
                            else {"max_share": p["max_prefill_share"]}
                        )
                    ),
                    "mode": p.get("mode", "share"),
                    **(
                        {"max_consecutive": p["max_consecutive"]}
                        if p.get("mode") == "rotation"
                        else {}
                    ),
                },
            )
        ],
        artifacts=[_artifact(ctx, "ha-recovery-probes", rows)],
    )


def _reconcile_validate(params, plan):
    p = _params(params, plan, {"client", "rows"}, {"client", "rows"})
    plan.reference(p["client"], "ha_client")
    plan.reference(p["rows"], "ha_rows")
    return p


def reconcile(ctx, p, deadline):
    journal = _journal(ctx.resource(p["client"], "ha_client"))
    rows = ctx.resource(p["rows"], "ha_rows")
    ids = [r["rid"] for r in rows]
    missing = sorted(set(journal.issued) - set(journal.terminal))
    good = (
        bool(ids)
        and not missing
        and not journal.pending
        and len(ids) == len(set(ids))
        and set(ids) == set(journal.issued) == set(journal.terminal)
    )
    if good:
        good = all(
            all(
                row[field] == journal.terminal[row["rid"]][field]
                for field in ("status", "master_target", "failover", "route_path")
            )
            for row in rows
        )
    actual = {
        "issued": len(journal.issued),
        "live_terminal": len(journal.terminal),
        "final_terminal": len(rows),
        "missing": missing,
    }
    return StageOutput(
        checks=[
            CheckResult(
                "accounting",
                "PASS" if good else "FAIL",
                actual=actual,
                expected="one matching terminal per issued request",
            )
        ],
        artifacts=[str(journal.path), _artifact(ctx, "ha-reconcile", actual)],
    )


HANDLERS = [
    StageHandler(
        "master_client_checkpoint",
        validate,
        checkpoint,
        {"snapshot": "snapshot"},
        checks=frozenset(
            {
                "topology",
                "accounting",
                "route",
                "success",
                "balance",
                "owner_bound",
                "retry",
            }
        ),
    ),
    StageHandler(
        "master_probe_distribution",
        _probe_validate,
        probe_distribution,
        {},
        checks=frozenset({"balance"}),
    ),
    StageHandler(
        "master_client_reconcile",
        _reconcile_validate,
        reconcile,
        {},
        checks=frozenset({"accounting"}),
    ),
]


def _idle_validate(params, plan):
    from .master import _owner_gate_validate

    return _owner_gate_validate(params, plan)


def owned_counts(raw):
    counts = {"scheduler_requests": raw["scheduler_inflight"]}
    for side, fields in (
        (
            "prefill",
            ("inflight_batches", "inflight_requests", "inflight_route_requests"),
        ),
        ("decode", ("reserved_total", "active_dispatch_permits")),
    ):
        endpoints = raw[side + "_endpoints"]
        if not isinstance(endpoints, list) or not endpoints:
            raise ValueError("missing owned endpoint counts")
        for field in fields:
            values = [r[field] for r in endpoints]
            if any(type(v) is not int or v < 0 for v in values):
                raise ValueError("invalid owned endpoint count")
            counts[side + "." + field] = sum(values)
    if (
        type(counts["scheduler_requests"]) is not int
        or counts["scheduler_requests"] < 0
    ):
        raise ValueError("invalid owned scheduler count")
    return counts


def idle_owner(ctx, p, deadline):
    samples = []
    try:
        while True:
            deadline.check()
            raw = _master_json(ctx, p["target"], "/rtp_llm/inflight_status", deadline)
            counts = owned_counts(raw)
            samples.append({"epoch_s": time.time(), "owned": counts, "raw": raw})
            if not any(counts.values()):
                break
            deadline.sleep(0.5)
    finally:
        path = _artifact(ctx, "ha-idle-owner", samples)
    return StageOutput(
        checks=[
            CheckResult(
                "owned_idle",
                "PASS",
                actual=counts,
                expected="local ownership zero; shared engine load observed separately",
            )
        ],
        artifacts=[path],
    )


HANDLERS.append(
    StageHandler(
        "master_owner_idle",
        _idle_validate,
        idle_owner,
        {},
        checks=frozenset({"owned_idle"}),
    )
)
