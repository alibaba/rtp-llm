"""HA client lifecycle, window selection and case-specific adjudication."""

import math
from pathlib import Path

from adapters.master_control import require_process
from cases.master_ha_failover.analysis import measure_client_metric, row_ts_ms
from cases.master_ha_failover.inputs import validate_client_criterion, validate_wait
from scenario.contracts import StageHandler, StageOutput
from scenario.parameters import validate_fields


def _ha_validate(params, plan):
    p = validate_fields(
        params,
        plan,
        {
            "duration_s",
            "timeout_ms",
            "fallback",
            "live_events",
            "max_concurrency",
            "replay_speed",
            "source",
            "max_requests",
            "loop",
            "capture",
        },
        {"source", "capture"},
    )
    if "capture" in p:
        from runtime.observation import capture_limits

        capture_limits(p["capture"], plan.path + ".capture")
    environment = getattr(plan, "environment", {})
    if environment.get("master_layout", "single") != "dual_zookeeper":
        raise ValueError("HA traffic requires a dual-master environment with ZK (dual_zookeeper)")
    for key, default, lower, upper in (("duration_s", 60, 1, 3600), ("timeout_ms", 30000, 100, 120000)):
        p.setdefault(key, default)
        if type(p[key]) is not int or not lower <= p[key] <= upper:
            raise ValueError(f"{key} is outside the bounded HA range")
    if "replay_speed" in p and (
        type(p["replay_speed"]) not in (int, float) or not math.isfinite(p["replay_speed"]) or p["replay_speed"] <= 0
    ):
        raise ValueError("invalid HA replay_speed")
    if "max_concurrency" in p and (type(p["max_concurrency"]) is not int or p["max_concurrency"] < 1):
        raise ValueError("invalid HA max_concurrency")
    if "source" in p:
        from traffic.contracts import source_priority

        source_priority(p["source"])
        from traffic.traffic_source import SourceSpecification

        if not SourceSpecification.read(p["source"]).is_lineage:
            raise ValueError("HA source must be a pinned real prefix lineage trace")
    if "max_requests" in p and ("source" not in p or type(p["max_requests"]) is not int or not 1 <= p["max_requests"] <= 20000):
        raise ValueError("HA max_requests requires a bounded real trace")
    p.setdefault("loop", False)
    if type(p["loop"]) is not bool:
        raise ValueError("HA loop must be boolean")
    p.setdefault("live_events", False)
    if type(p["live_events"]) is not bool:
        raise ValueError("live_events must be boolean")
    p.setdefault("fallback", False)
    if type(p["fallback"]) is not bool:
        raise ValueError("fallback must be boolean")
    return p


def _ha_start(ctx, params, deadline):
    from cases.master_ha_failover.client import HaReplayClient

    if set(ctx.env.master_specs) != {"A", "B"}:
        raise ValueError("HA restart cycle requires registered Masters A and B")
    for target in ctx.env.master_specs:
        require_process(ctx, target)
    directory = ctx.artifact_dir / f"ha-client-{ctx.resource_count}"
    directory.mkdir(parents=True, exist_ok=True)
    from artifacts.startup import startup_phase

    with startup_phase(ctx.artifact_dir, "traffic/input", deadline):
        client = HaReplayClient(
            ctx.services.environment,
            ctx.env,
            directory,
            "traffic",
            duration_s=params["duration_s"],
            sampler_limits=params["capture"],
            monitor=ctx.monitor,
            timeout_ms=params["timeout_ms"],
            enable_fallback=params["fallback"],
            live_events=params["live_events"],
            collection_profile=ctx.instance.get("collection_profile", "request"),
            source=params["source"],
            source_dir=(Path(ctx.instance["source_path"]).parent),
            max_requests=params.get("max_requests"),
            loop=params["loop"],
            **({"replay_speed": params["replay_speed"]} if "replay_speed" in params else {}),
            **({"max_concurrency": params["max_concurrency"]} if "max_concurrency" in params else {}),
        )

    def export_client(value, profile, deadline):
        deadline.check()
        from adapters.resource_evidence import ResourceEvidence

        snapshot = value.evidence_snapshot()
        manifest = snapshot if profile == "diagnostic" else {k: v for k, v in snapshot.items() if k != "records"}
        manifest["request_journal"] = snapshot["path"]
        return ResourceEvidence(
            requests=manifest,
            producer=id(value),
            errors=() if snapshot["complete"] else tuple(snapshot["errors"] or ["incomplete HA client"]),
        )

    handle = ctx.register_resource("ha_client", client, client.cleanup, evidence=export_client)
    deadline.check()
    client.start(deadline)
    deadline.check()
    return StageOutput({"client": handle})


def _ha_finish_validate(params, plan):
    p = validate_fields(params, plan, {"client", "stop_sending"}, {"client"})
    p.setdefault("stop_sending", False)
    if type(p["stop_sending"]) is not bool:
        raise ValueError("stop_sending must be boolean")
    plan.reference(p["client"], "ha_client")
    return p


def _ha_finish(ctx, params, deadline):
    client = ctx.resource(params["client"], "ha_client")
    rows, path = client.finish(deadline, stop_sending=params.get("stop_sending", False))
    sampler = getattr(client, "state_sampler", None)
    return StageOutput(
        {"rows": ctx.register_resource("ha_rows", rows, historical=True)},
        artifacts=[str(path)] + ([str(sampler.path), str(sampler.path.with_suffix(".prometheus.json"))] if sampler is not None else []),
    )


def _period_validate(params, plan):
    p = validate_fields(
        params, plan, {"name", "settle_s", "duration_s", "operation_timeout_s"}, {"name", "settle_s", "duration_s", "operation_timeout_s"}
    )
    if p["name"] not in {"baseline", "a", "b", "both", "outage"}:
        raise ValueError("unknown HA period")
    for key in ("settle_s", "duration_s"):
        validate_wait({"wait_s": p[key]}, path=plan.path + "." + key)
    from scenario.validation import number

    number(p["operation_timeout_s"], plan.path, minimum=0.001)
    return p


def _period(ctx, params, deadline):
    def wait(seconds):
        deadline.child(params["operation_timeout_s"]).sleep(seconds)

    wait(params["settle_s"])
    name = params["name"]
    start = ctx.record_event(name + "_start")["epoch_s"] if name != "baseline" else ctx.wall_clock()
    wait(params["duration_s"])
    end = ctx.record_event(name + "_end")["epoch_s"]
    return StageOutput({"start": start, "end": end})


def _select_rows(ctx, rows, params, deadline):
    from analysis.statistics import select_window

    deadline.check()
    lower = ctx.resolve(params["from"]) if "from" in params else None
    upper = ctx.resolve(params["until"]) if "until" in params else None
    from scenario.windows import resolve_window

    lower, upper = resolve_window(
        lower, upper, lower_offset_s=params.get("from_offset_s", 0), upper_offset_s=params.get("until_offset_s", 0)
    )
    selected = select_window(
        rows, -math.inf if lower is None else lower, math.inf if upper is None else upper, time=lambda row: row_ts_ms(row) / 1000
    )
    for param, field in (("route", "route_path"), ("status", "status"), ("error_kind", "error_kind"), ("failover", "failover")):
        if param in params:
            selected = [r for r in selected if r[field] == params[param]]
    return selected


def _client_check(ctx, params, deadline, rows=None):
    deadline.check()
    if rows is None:
        rows = ctx.resource(params["rows"], "ha_rows")
    n = len(rows)
    metric = params["metric"].split("/", 1)[1]
    if n < params["min_samples"]:
        from analysis.checks import invalid_check

        return StageOutput(
            {"actual": None},
            checks=[
                invalid_check(
                    "criterion",
                    detail="insufficient request samples",
                    expected=params["expected"],
                    evidence=dict(metric=params["metric"], sample_count=n, min_samples=params["min_samples"]),
                )
            ],
        )

    target = None
    prefill_pool = ()
    if metric == "target_share":
        target = ctx.services.environment.master_instance_target(ctx.env, params["target"])
    elif metric == "prefill_max_share":
        from adapters.mock_control import topology_pools

        prefill_pool = topology_pools(ctx, deadline)["prefill"]
    actual = measure_client_metric(params, rows, target=target, prefill_pool=prefill_pool)
    from analysis.checks import evaluate
    from cases.master_ha_failover.metrics import gate_labels

    stamps = [row_ts_ms(row) / 1000 for row in rows if "send_start_epoch_ms" in row]
    result = evaluate(
        "criterion",
        actual,
        params["op"],
        params["expected"],
        advisory=bool(params.get("warning_profiles")) and ctx.instance["profile"] in params["warning_profiles"],
        evidence=dict(
            metric=params["metric"],
            op=params["op"],
            env_epoch=ctx.env_epoch,
            input_resource=params["rows"],
            labels=gate_labels(params),
            sample_count=n,
            min_samples=params["min_samples"],
            observed_request_bounds=[min(stamps), max(stamps)] if stamps else None,
        ),
    )
    if result.status == "WARNING":
        import logging

        logging.getLogger(__name__).warning(
            "HA advisory criterion: profile=%s metric=%s actual=%s expected=%s samples=%s",
            ctx.instance["profile"],
            metric,
            actual,
            params["expected"],
            n,
        )
        from dataclasses import replace

        result = replace(result, detail="Known issue: advisory criterion")
    return StageOutput({"actual": actual}, [result])


def _analyze_validate(params, plan):
    p = validate_fields(
        params, plan, {"rows", "windows", "checks", "operation_timeout_s"}, {"rows", "windows", "checks", "operation_timeout_s"}
    )
    plan.reference(p["rows"], "ha_rows")
    from scenario.validation import number

    number(p["operation_timeout_s"], plan.path, minimum=0.001)
    if not isinstance(p["windows"], dict) or not isinstance(p["checks"], dict) or not p["checks"]:
        raise ValueError("HA analysis requires windows and checks")
    from monitoring.identity import NAME

    for name, window in p["windows"].items():
        if not isinstance(name, str) or not NAME.fullmatch(name):
            raise ValueError("invalid window name")
        from scenario.inputs import fields

        fields(window, (), plan.path, optional={"from", "until", "from_offset_s", "until_offset_s"})
        for side in ("from", "until"):
            if side in window:
                plan.reference(window[side], "number")
            if side + "_offset_s" in window:
                offset = window[side + "_offset_s"]
                if side not in window or type(offset) not in (int, float) or not math.isfinite(offset):
                    raise ValueError("window offset requires finite offset and boundary reference")
    for name, criterion in p["checks"].items():
        if not isinstance(name, str) or not NAME.fullmatch(name):
            raise ValueError("invalid check name")
        criterion = dict(criterion)
        window = criterion.pop("window")
        if window not in {*p["windows"], "full_run"}:
            raise ValueError("unknown analysis window")
        p["checks"][name] = dict(validate_client_criterion(criterion, path=plan.path), rows=p["rows"], window=window)
    return p


def _analyze(ctx, params, deadline):
    from dataclasses import replace

    rows = ctx.resource(params["rows"], "ha_rows")
    windows = {"full_run": rows}
    for name, window in params["windows"].items():
        with deadline.operation(params["operation_timeout_s"], ctx.enforce_deadlines) as operation:
            windows[name] = _select_rows(ctx, rows, window, operation)
    checks = []
    for name, criterion in params["checks"].items():
        p = {k: v for k, v in criterion.items() if k != "window"}
        # A named cohort replaces the old intermediate resource identity.
        cohort = criterion["window"]
        with deadline.operation(params["operation_timeout_s"], ctx.enforce_deadlines) as operation:
            result = _client_check(ctx, p, operation, windows[cohort]).checks[0]
        evidence = dict(result.evidence or {})
        if "labels" in evidence:
            evidence["labels"] = dict(evidence["labels"], window=cohort)
        checks.append(replace(result, id=name, evidence=evidence))
    return StageOutput(checks=checks)


HANDLERS = [
    StageHandler("master_client_start", _ha_validate, _ha_start, {"client": "ha_client"}),
    StageHandler("master_client_finish", _ha_finish_validate, _ha_finish, {"rows": "ha_rows"}),
    StageHandler("master_period", _period_validate, _period, {"start": "number", "end": "number"}),
    StageHandler("master_client_analyze", _analyze_validate, _analyze, {}, checks=lambda p: frozenset(p["checks"])),
]
