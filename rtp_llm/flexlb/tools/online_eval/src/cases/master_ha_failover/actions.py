"""HA client lifecycle, window selection and case-specific adjudication."""

import math
from pathlib import Path

from cases.master_ha_failover.analysis import measure_client_metric, row_ts_ms
from cases.master_ha_failover.inputs import validate_client_criterion, validate_wait
from runtime.master_control import require_process
from scenario.parameters import validate_fields
from scenario.contracts import StageHandler, StageOutput


def _ha_validate(params, plan):
    p = validate_fields(
        params,
        plan,
        {
            "targets",
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
    if environment.get("master_layout", "single") != "dual_standalone":
        raise ValueError("HA traffic requires compiled dual_standalone environment")
    p.setdefault("targets", ["A", "B"])
    if p["targets"] not in (["A", "B"], ["B", "A"]):
        raise ValueError("HA targets must explicitly order A and B")
    for key, default, lower, upper in (
        ("duration_s", 60, 1, 3600),
        ("timeout_ms", 30000, 100, 120000),
    ):
        p.setdefault(key, default)
        if type(p[key]) is not int or not lower <= p[key] <= upper:
            raise ValueError(f"{key} is outside the bounded HA range")
    if "replay_speed" in p and (
        type(p["replay_speed"]) not in (int, float)
        or not math.isfinite(p["replay_speed"])
        or p["replay_speed"] <= 0
    ):
        raise ValueError("invalid HA replay_speed")
    if "max_concurrency" in p and (
        type(p["max_concurrency"]) is not int or p["max_concurrency"] < 1
    ):
        raise ValueError("invalid HA max_concurrency")
    if "source" in p:
        from traffic.contracts import source_priority
        source_priority(p["source"])
        source = p["source"]
        if (
            not isinstance(source, dict)
            or set(source) != {"kind", "model", "version", "parameters"}
            or (source["kind"], source["model"], source["version"])
            not in {("trace", "prefix_lineage", "2"), ("trace", "prefix_lineage", "3")}
        ):
            raise ValueError("HA source must be a pinned real prefix lineage trace")
    if "max_requests" in p and (
        "source" not in p
        or type(p["max_requests"]) is not int
        or not 1 <= p["max_requests"] <= 20000
    ):
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

    for target in params["targets"]:
        require_process(ctx, target)
    targets = [
        ctx.backend.manager.master_instance_target(ctx.env, target)
        for target in params["targets"]
    ]
    directory = ctx.artifact_dir / f"ha-client-{len(ctx._resources)}"
    directory.mkdir(parents=True, exist_ok=True)
    client = HaReplayClient(
        ctx.backend.manager,
        ctx.env,
        directory,
        "traffic",
        targets,
        duration_s=params["duration_s"],
        sampler_limits=params["capture"],
        clock=ctx.clock, wall_clock=ctx.wall_clock,
        timeout_ms=params["timeout_ms"],
        enable_fallback=params["fallback"],
        live_events=params["live_events"],
        collection_profile=ctx.instance.get("collection_profile", "request"),
        source=params["source"],
        source_dir=(
            Path(ctx.instance["source_path"]).parent
        ),
        max_requests=params.get("max_requests"),
        loop=params["loop"],
        **(
            {"replay_speed": params["replay_speed"]} if "replay_speed" in params else {}
        ),
        **(
            {"max_concurrency": params["max_concurrency"]}
            if "max_concurrency" in params
            else {}
        ),
    )
    handle = ctx.register_resource("ha_client", client, client.cleanup)
    deadline.check()
    client.start()
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
        artifacts=[str(path)] + ([str(sampler.path)] if sampler is not None else []),
    )


def _mark_validate(params, plan):
    p = validate_fields(params, plan, {"wait_s", "event"}, {"wait_s", "event"})
    validate_wait({"wait_s": p["wait_s"]}, path=plan.path)
    from monitoring.identity import NAME
    if type(p["event"]) is not str or not NAME.fullmatch(p["event"]):
        raise ValueError("invalid mark event identity")
    return p


def _mark(ctx, params, deadline):
    deadline.sleep(params["wait_s"])
    return StageOutput({"epoch_s": ctx.record_event(params["event"])["epoch_s"]})


def _window_validate(params, plan):
    p = validate_fields(
        params,
        plan,
        {
            "rows",
            "from",
            "until",
            "route",
            "status",
            "error_kind",
            "failover",
            "from_offset_s",
            "until_offset_s",
        },
        {"rows"},
    )
    plan.reference(p["rows"], "ha_rows")
    for key in ("from", "until"):
        if key in p:
            plan.reference(p[key], "number")
    if "route" in p and p["route"] not in {"master", "fallback", "failed"}:
        raise ValueError("invalid window route filter")
    if "failover" in p and type(p["failover"]) is not bool:
        raise ValueError("window failover filter must be boolean")
    if "status" in p and p["status"] not in {"ok", "schedule_error"}:
        raise ValueError("unsupported window status filter")
    if "error_kind" in p and p["error_kind"] not in {
        "none",
        "transport",
        "business",
        "deadline",
    }:
        raise ValueError("invalid window error filter")
    for field in ("from_offset_s", "until_offset_s"):
        if field in p:
            if (
                field.split("_")[0] not in p
                or type(p[field]) not in (int, float)
                or not math.isfinite(p[field])
            ):
                raise ValueError(
                    "window offset requires finite offset and boundary reference"
                )
    return p


def _window(ctx, params, deadline):
    from analysis.statistics import select_window

    deadline.check()
    rows = ctx.resource(params["rows"], "ha_rows")
    lower = ctx.resolve(params["from"]) if "from" in params else None
    upper = ctx.resolve(params["until"]) if "until" in params else None
    from cases.windows import resolve_window
    lower, upper = resolve_window(lower, upper,
        lower_offset_s=params.get("from_offset_s", 0), upper_offset_s=params.get("until_offset_s", 0))
    selected = select_window(rows, -math.inf if lower is None else lower,
                             math.inf if upper is None else upper, time=lambda row: row_ts_ms(row) / 1000)
    for param, field in (
        ("route", "route_path"),
        ("status", "status"),
        ("error_kind", "error_kind"),
        ("failover", "failover"),
    ):
        if param in params:
            selected = [r for r in selected if r[field] == params[param]]
    return StageOutput(
        {"rows": ctx.register_resource("ha_rows", selected, historical=True)}
    )


def _client_check_validate(params, plan):
    p = validate_fields(
        params,
        plan,
        {
            "rows",
            "metric",
            "op",
            "expected",
            "target",
            "route",
            "min_samples",
            "warning_profiles",
        },
        {"rows", "metric", "op", "expected"},
    )
    plan.reference(p["rows"], "ha_rows")
    criterion = validate_client_criterion({key: value for key, value in p.items() if key != "rows"})
    return dict(criterion, rows=p["rows"])


def _client_check(ctx, params, deadline):
    deadline.check()
    rows = ctx.resource(params["rows"], "ha_rows")
    n = len(rows)
    metric = params["metric"].split("/", 1)[1]
    if n < params["min_samples"]:
        from analysis.checks import invalid_check
        return StageOutput(checks=[invalid_check("criterion", detail="insufficient request samples",
            expected=params["expected"], evidence=dict(metric=params["metric"],
                sample_count=n, min_samples=params["min_samples"]))])

    target = None
    prefill_pool = ()
    if metric == "target_share":
        target = ctx.backend.manager.master_instance_target(ctx.env, params["target"])
    elif metric == "prefill_max_share":
        from runtime.mock_control import topology_pools

        prefill_pool = topology_pools(ctx, deadline)["prefill"]
    actual = measure_client_metric(params, rows, target=target, prefill_pool=prefill_pool)
    from cases.master_ha_failover.metrics import publish_gate, gate_labels
    from analysis.checks import check_metric

    store = publish_gate(ctx, params, actual, rows)
    result = check_metric(store, "criterion", params["metric"],
        labels=gate_labels(params), source="ha_gate", epoch=ctx.env_epoch, reduction="last",
        op=params["op"], expected=params["expected"],
        advisory=bool(params.get("warning_profiles")) and ctx.instance["profile"] in params["warning_profiles"],
        evidence=dict(sample_count=n, min_samples=params["min_samples"]))
    if result.status == "WARNING":
        import logging
        logging.getLogger(__name__).warning(
            "HA advisory criterion: profile=%s metric=%s actual=%s expected=%s samples=%s",
            ctx.instance["profile"], metric, actual, params["expected"], n)
        from dataclasses import replace
        result = replace(result, detail="Known issue: advisory criterion")
    return StageOutput(
        {"actual": actual},
        [result],
    )


HANDLERS = [
    StageHandler(
                "master_client_start", _ha_validate, _ha_start, {"client": "ha_client"}
            ),
    StageHandler(
                "master_client_finish", _ha_finish_validate, _ha_finish, {"rows": "ha_rows"}
            ),
    StageHandler("master_mark", _mark_validate, _mark, {"epoch_s": "number"}),
    StageHandler(
                "master_client_window", _window_validate, _window, {"rows": "ha_rows"}
            ),
    StageHandler(
                "master_client_check",
                _client_check_validate,
                _client_check,
                {"actual": "number"},
                checks=frozenset({"criterion"}),
            ),
]
