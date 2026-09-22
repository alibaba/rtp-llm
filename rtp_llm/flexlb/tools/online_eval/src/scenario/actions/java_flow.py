"""Java-generated traffic groups with Python-owned preconditions and checkpoints."""

import json
import math
import re
import time
from pathlib import Path

from runtime.java_flow import JavaFlowGroup
from runtime.load_client import LOAD_CLIENT_ENV_VARS
from traffic.traffic_source import materialize
from traffic.playback import normalize

from runtime.harness import ClientOps
from scenario.contracts import CheckResult, StageHandler, StageOutput
from scenario.actions.elastic import _snapshot, _validate


def _start_validate(params, plan):
    fields = {
        "group_id",
        "phase_id",
        "poll_s",
        "source",
        "trace",
        "client",
        "jvm_xms",
        "jvm_xmx",
    }
    p = _validate(params, plan, fields, fields - {"source", "trace"})
    if ("source" in p) == ("trace" in p):
        raise ValueError("specify exactly one traffic source")
    if "trace" in p:
        raise ValueError("legacy trace shorthand retired; use synthetic/realistic/1")
    if not all(
        isinstance(p[k], str) and p[k]
        for k in ("group_id", "phase_id", "jvm_xms", "jvm_xmx")
    ):
        raise ValueError("flow identity and JVM sizing must be explicit")
    if re.fullmatch(r"[A-Za-z0-9_-]+", p["group_id"]) is None:
        raise ValueError("group_id must be a safe directory component")
    if (
        type(p["poll_s"]) not in (int, float)
        or not math.isfinite(p["poll_s"])
        or p["poll_s"] <= 0
    ):
        raise ValueError("invalid flow polling interval")
    client, _ = normalize(p["client"])
    if p["source"].get("kind") == "synthetic" and client.get("SEND_MODE") == "replay":
        raise ValueError("statistical source requires client-paced uniform/burst/gradient playback")
    if (
        not isinstance(client, dict)
        or client.get("REPLAY_UNIQUE_PREFIX") != "false"
        or client.get("FETCH_OUTPUT_STREAM") != "true"
    ):
        raise ValueError(
            "scenario flow requires faithful prefix replay and response consumption"
        )
    if set(client) - set(LOAD_CLIENT_ENV_VARS):
        raise ValueError("unknown Java client configuration names")
    if (
        int(client.get("DURATION_S", 0)) <= 0
        or int(client.get("MAX_CONCURRENCY", 0)) <= 0
    ):
        raise ValueError("scenario flow requires bounded duration and concurrency")
    if set(client) & {
        "FLOW_CONTROL_DIR",
        "FLOW_RUN_ID",
        "FLOW_GROUP_ID",
        "FLOW_PHASE_ID",
        "TRACE_FILE",
        "OUTPUT_DIR",
        "GRPC_TARGET",
        "GRPC_TARGETS",
    }:
        raise ValueError(
            "client endpoint and runtime identities are resolved by framework"
        )
    return p


def _start(ctx, p, deadline):
    directory = ctx.artifact_dir / "flows" / p["group_id"]
    directory.parent.mkdir(exist_ok=True)
    trace = materialize(
        directory.parent / (p["group_id"] + ".jsonl"),
        p["source"],
        ctx.instance["id"] + ":" + p["group_id"],
        Path(ctx.instance["source_path"]).parent,
    )
    environment, playback = normalize(p["client"])
    client = ClientOps(ctx.backend.manager, p["jvm_xms"], p["jvm_xmx"])
    count = json.loads(trace.with_suffix('.manifest.json').read_text())['request_count']
    laps = int(environment.get('MAX_LAPS', 0 if environment.get('LOOP')=='true' else 1))
    from traffic.playback_controls import integral, poisson_count
    curve = playback.get('rate_curve')
    if laps:
        event_budget = max(50_000, 2*count*laps)
    elif environment.get('SEND_MODE')=='uniform':
        rate = float(environment['SEND_MODE_QPS'])
        modulation = float(environment.get('BURST_FACTOR',1))*(1+float(environment.get('DIURNAL_AMPLITUDE',0)))
        intensity = rate*modulation*integral(curve,int(environment['DURATION_S']))
        planned = (poisson_count(intensity, int(environment['PLAYBACK_SEED']))
                   if playback.get('arrival') == 'poisson' else math.ceil(intensity)+1)
        event_budget = max(50_000, 2*planned)
    else:
        # A finite model can contain coincident timestamps. Use its full cycle
        # span rather than average rate, including one partial final lap.
        first = last = None
        with trace.open() as stream:
            for line in stream:
                last = json.loads(line)['ts']
                if first is None: first=last
        span_ms=max(1,last-first)+1
        laps=math.ceil(integral(curve,int(environment['DURATION_S']))*1000*float(environment.get('REPLAY_SPEED',1))/span_ms)+1
        event_budget=max(50_000,2*count*laps)
    flow = JavaFlowGroup(
        client,
        directory,
        run_id=str(ctx.artifact_dir),
        group_id=p["group_id"],
        phase_id=p["phase_id"],
        poll_s=p["poll_s"],
        max_events=event_budget,
        collection_profile=ctx.instance.get("collection_profile", "request"),
        monitor=getattr(ctx, "monitor", None),
    )

    def cleanup(d):
        if flow.proc is not None:
            flow.stop_sending(d)
            flow.drain(d)

    handle = ctx.register_resource("java_flow", flow, cleanup=cleanup)
    environment = dict(
        environment, GRPC_TARGET=f"127.0.0.1:{ctx.env.master_http_port + 2}"
    )
    flow.start(trace, environment, deadline)
    return StageOutput({"flow": handle}, artifacts=[str(directory / "flow-input.json")])


def _flow_validate(params, plan):
    p = _validate(params, plan, {"flow"}, {"flow"})
    plan.reference(p["flow"], "java_flow")
    return p


def _stop(ctx, p, deadline):
    flow = ctx.resource(p["flow"], "java_flow")
    state = flow.stop_sending(deadline)
    return StageOutput(
        {"snapshot": ctx.register_resource("snapshot", state, historical=True)}
    )


def _drain(ctx, p, deadline):
    flow = ctx.resource(p["flow"], "java_flow")
    state = flow.drain(deadline)
    return StageOutput(
        {"snapshot": ctx.register_resource("snapshot", state, historical=True)}
    )


def _checkpoint_validate(params, plan):
    fields = {"flow", "min_started", "min_terminal", "min_decode_inflight", "engine"}
    p = _validate(params, plan, fields, fields - {"engine"})
    plan.reference(p["flow"], "java_flow")
    for k in fields - {"flow", "engine"}:
        if type(p[k]) is not int or p[k] < 0:
            raise ValueError("checkpoint bounds must be explicit nonnegative integers")
    if "engine" in p:
        plan.reference(p["engine"], "string")
    return p


def _checkpoint(ctx, p, deadline):
    flow = ctx.resource(p["flow"], "java_flow")
    samples = []
    path = ctx.artifact_dir / f"flow-checkpoint-{time.time_ns()}.json"
    try:
        while True:
            deadline.check()
            state = flow.status()
            engines = _snapshot(ctx, deadline)
            inflight = sum(
                e["running"] + e["waiting"]
                for e in engines.values()
                if e["role"] == "decode"
            )
            engine = ctx.resolve(p["engine"]) if "engine" in p else None
            accepted = engines.get(engine, {}).get("accepted") if engine else None
            actual = dict(
                epoch_s=time.time(),
                state=state,
                decode_inflight=inflight,
                engine=engine,
                accepted=accepted,
            )
            samples.append(actual)
            good = (
                state["observed_started"] >= p["min_started"]
                and state["observed_terminal"] >= p["min_terminal"]
                and inflight >= p["min_decode_inflight"]
                and (engine is None or (type(accepted) is int and accepted > 0))
            )
            if good:
                break
            if state["process_returncode"] is not None:
                raise RuntimeError("flow ended before checkpoint precondition")
            deadline.sleep(flow.poll_s)
    finally:
        path.write_text(json.dumps(samples, indent=2))
    return StageOutput(
        {"snapshot": ctx.register_resource("snapshot", actual, historical=True)},
        [CheckResult("precondition", "PASS", actual=actual)],
        [str(path)],
    )


def _check_validate(params, plan):
    p = _validate(
        params, plan, {"flow", "min_success_rate"}, {"flow", "min_success_rate"}
    )
    plan.reference(p["flow"], "java_flow")
    if (
        type(p["min_success_rate"]) not in (int, float)
        or not 0 <= p["min_success_rate"] <= 1
    ):
        raise ValueError("invalid explicit success rate")
    return p


def _check(ctx, p, deadline):
    flow = ctx.resource(p["flow"], "java_flow")
    snapshot = flow.evidence_snapshot()
    rows = snapshot["records"]
    good = sum(row.get("status") == "ok" for row in rows)
    rate = good / len(rows) if rows else None
    path = flow.directory / "verified-evidence.json"
    path.write_text(json.dumps(snapshot, indent=2))
    return StageOutput(
        checks=[
            CheckResult(
                "complete",
                "PASS" if snapshot["complete"] else "FAIL",
                actual=snapshot["errors"],
            ),
            CheckResult(
                "success_rate",
                (
                    "PASS"
                    if rate is not None and rate >= p["min_success_rate"]
                    else "FAIL"
                ),
                actual=rate,
                expected=p["min_success_rate"],
            ),
        ],
        artifacts=[str(path)],
    )


HANDLERS = [
    StageHandler("java_flow_start", _start_validate, _start, {"flow": "java_flow"}),
    StageHandler("java_flow_stop", _flow_validate, _stop, {"snapshot": "snapshot"}),
    StageHandler("java_flow_drain", _flow_validate, _drain, {"snapshot": "snapshot"}),
    StageHandler(
        "java_flow_checkpoint",
        _checkpoint_validate,
        _checkpoint,
        {"snapshot": "snapshot"},
        checks=frozenset({"precondition"}),
    ),
    StageHandler(
        "java_flow_check",
        _check_validate,
        _check,
        {},
        checks=frozenset({"complete", "success_rate"}),
    ),
]
