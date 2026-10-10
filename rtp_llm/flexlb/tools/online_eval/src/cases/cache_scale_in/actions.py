"""Continuous Java traffic across a concurrent Prefill scale-in."""

import math
from concurrent.futures import ThreadPoolExecutor

from scenario.contracts import StageHandler, StageOutput
from scenario.parameters import validate_fields
from runtime.observation import ObservationClock, SampleBudget, poll_samples
from artifacts.json_io import write_json
from workload.gate_evidence import new_evidence
from workload.run_provenance import gate_provenance
from runtime.mock_control import mock_json
from cases.cache_scale_in.analysis import (
    MEASUREMENT_POLICY,
    align_send_counters,
    analyze,
    attribute_client,
    scope_contract,
    topology_ready,
    window,
)
from workload.gate_result import freeze_gate, gate_check

from cases.cache_scale_in.inputs import validate_criteria


def validate(params, plan):
    grouped = validate_fields(params, plan,
        {"flow", "criteria", "gate_input", "observation"}, {"flow", "criteria", "gate_input", "observation"})
    from cases.cache_scale_in.inputs import observation_contract
    durations = observation_contract(grouped["observation"])
    if any(grouped["criteria"].get(key) != value for key, value in durations.items()):
        raise ValueError("observation windows disagree with frozen criteria")
    validate_criteria(dict(grouped["criteria"], flow=grouped["flow"], gate_input=grouped["gate_input"]), plan)
    return grouped


def _master_count(ctx, deadline):
    from runtime.network import master_url, http_post_json
    code, data = http_post_json(master_url("127.0.0.1", ctx.env.master_http_port, "info"),
                                {}, timeout=min(3, deadline.remaining()))
    if code != 200 or not isinstance(data, dict):
        raise ValueError("master topology HTTP unavailable")
    value = data["worker_summary"]["PREFILL"]["alive"]
    if type(value) is not int:
        raise ValueError("master topology missing")
    return value


def observe(ctx, p, deadline):
    observation = p["observation"]
    p = dict(p["criteria"], flow=p["flow"], gate_input=p["gate_input"])
    flow = ctx.resource(p["flow"], "java_flow")
    clock = ObservationClock.start(ctx)
    origin = clock.origin_monotonic_s
    budget = SampleBudget(observation["capture"])
    evidence = new_evidence("cache_evidence_schema_version", clock,
        {k: v for k, v in p.items() if k not in ("flow", "gate_input")},
        instance=ctx.instance["id"], env_epoch=ctx.env_epoch,
        measurement_policy=MEASUREMENT_POLICY,
        window_declarations=observation["windows"],
        observation_origin_epoch_s=clock.origin_epoch_s,
        gate_input=p["gate_input"],
        events=[],
        initial_engines=[],
        survivors=[],
        baseline_start=0,
        baseline_end=0,
        post_start=0,
        post_end=0,
        max_pacing_lag_ms=None,
    )
    from cases.cache_scale_in import analysis as cache_gate, program, inputs
    path = ctx.artifact_dir / "cache-gate-evidence.json"

    def event(name, row=None):
        timestamp = None if row is None else {key: row[key] for key in ("epoch_s", "monotonic_s")}
        recorded = ctx.record_event(name, timestamp=timestamp)
        evidence["events"].append(dict(recorded, t=recorded["monotonic_s"] - origin))

    def sample():
        deadline.check()
        from cases.cache_scale_in.inputs import engine_metric_snapshot
        engines = engine_metric_snapshot(ctx.monitor, p["gate_input"]["fields"],
                                  timeout=min(3, deadline.remaining()))
        state = flow.status()
        stamp = clock.stamp(ctx)
        row = dict(
            t=stamp["elapsed_s"],
            **stamp,
            engines=engines,
            started=state["observed_started"],
            terminal=state["observed_terminal"],
            master_p=_master_count(ctx, deadline),
            waiting=sum(e["waiting"] for e in engines.values()),
            running=sum(e["running"] for e in engines.values()),
        )
        budget.append(row)
        evidence["samples"].append(row)
        if state["process_returncode"] is not None or state.get("state") != "SENDING":
            raise ValueError("Java traffic stopped before observation completed")
        return row

    pool = None
    futures = []
    try:
        evidence["provenance"].update(gate_provenance(ctx, flow, source_files=(__file__, cache_gate.__file__, program.__file__, inputs.__file__), analyzer_file=cache_gate.__file__))
        first = sample()
        initial = sorted(first["engines"])
        evidence["initial_engines"] = initial
        evidence["survivors"] = initial[: p["target_p"]]
        if first["master_p"] != len(initial):
            raise ValueError("initial discovery not converged")
        event("warmup_start")
        for row in poll_samples(deadline, p["sample_s"], sample):
            t = row["t"]
            if t >= p["baseline_s"]:
                left = window(
                    evidence["samples"],
                    t - p["baseline_s"],
                    t - p["baseline_s"] / 2,
                    initial,
                    p["max_gap_s"],
                )
                right = window(
                    evidence["samples"],
                    t - p["baseline_s"] / 2,
                    t,
                    initial,
                    p["max_gap_s"],
                )
                ready = all(
                    not w["errors"]
                    and w["hit"] is not None
                    and w["hit"] >= p["baseline_min_hit"]
                    and w["completed"] >= p["min_completed"]
                    and w["sent_qps"] is not None
                    and abs(w["sent_qps"] / p["qps"] - 1) <= p["qps_tolerance"]
                    for w in (left, right)
                )
                if (
                    ready
                    and abs(left["hit"] - right["hit"]) <= p["baseline_max_spread"]
                ):
                    evidence.update(baseline_start=t - p["baseline_s"], baseline_end=t)
                    event("baseline_ready", row)
                    break
            if t >= p["warmup_timeout_s"]:
                raise ValueError("stable warm baseline not reached")
        removed = initial[p["target_p"] :]
        event("withdraw_start")
        # Every removal withdraws discovery before waiting. Parallel requests do
        # not serialize the drain periods; the Java mutation lock only covers rewrite.
        pool = ThreadPoolExecutor(
            max_workers=len(removed), thread_name_prefix="scale-in"
        )
        futures = [
            pool.submit(
                mock_json,
                ctx.ops,
                "remove_engine",
                deadline,
                dict(engine=n, mode=p["removal_mode"], drain_timeout_ms=p["drain_timeout_ms"]),
            )
            for n in removed
        ]
        topology_end = ctx.clock() + p["topology_timeout_s"]
        for row in poll_samples(deadline, p["sample_s"], sample):
            if topology_ready(row, evidence["survivors"], evidence["initial_engines"]):
                evidence["post_start"] = row["t"]
                event("target_topology_observed", row)
                break
            if ctx.clock() >= topology_end:
                raise ValueError("target topology did not converge")
        end = evidence["post_start"] + p["observe_s"]
        for row in poll_samples(deadline, p["sample_s"], sample, until=origin + end):
            pass
        evidence["post_end"] = row["t"]
        event("observation_end", row)
        flow.stop_sending(deadline)
        event("sending_stopped")
        evidence["removals"] = [
            f.result(timeout=max(0.01, deadline.remaining())) for f in futures
        ]
    except Exception as exc:
        evidence["errors"].append(str(exc))
    finally:
        if pool:
            # The underlying HTTP calls and server drains are deadline-bounded.
            pool.shutdown(wait=True, cancel_futures=True)
        # Preserve individual drain outcomes even if topology observation failed.
        evidence["removals"] = []
        for index, future in enumerate(futures):
            try:
                evidence["removals"].append(future.result())
            except Exception as exc:
                evidence.setdefault("removal_errors", []).append(dict(index=index, error=str(exc)))
        evidence["measurement_scope"] = scope_contract(evidence)
        write_json(path, evidence)
    return StageOutput(
        {"evidence": ctx.register_resource("gate_evidence", evidence, historical=True)},
        artifacts=[str(path)],
    )


def check_validate(params, plan):
    p = validate_fields(params, plan, {"flow", "evidence"}, {"flow", "evidence"})
    plan.reference(p["flow"], "java_flow")
    plan.reference(p["evidence"], "gate_evidence")
    return p


def check(ctx, p, deadline):
    evidence = ctx.resource(p["evidence"], "gate_evidence")
    flow = ctx.resource(p["flow"], "java_flow")
    snapshot = flow.evidence_snapshot()
    # Terminal request failures do not decide whether survivor cache counters
    # are measurable. Issued sends still need complete accounting for QPS.
    if len(snapshot["issued"]) != snapshot.get("status", {}).get("submitted"):
        evidence["errors"].append("issued send accounting incomplete")
    else:
        try:
            align_send_counters(evidence, snapshot["issued"])
        except ValueError as exc:
            evidence["errors"].append(str(exc))
    attribute_client(evidence, snapshot)
    from traffic.traffic_source import sha256_file
    evidence["client_attribution"]["artifacts"] = [
        dict(path=str(path.resolve()), sha256=sha256_file(path))
        for path in (flow.directory / "client_lifecycle.jsonl", flow.directory / "control/status.json")
        if path.is_file()
    ]
    times = [r["epoch_s"] for r in evidence["samples"]]
    issued = [
        r
        for r in snapshot["issued"]
        if times
        and times[0] * 1000 <= r.get("send_start_epoch_ms", -1) <= times[-1] * 1000
    ]
    lags = [r.get("pacing_lag_ms") for r in issued]
    if lags and all(type(v) in (int, float) and math.isfinite(v) for v in lags):
        evidence["max_pacing_lag_ms"] = max(lags)
    try:
        ctx.monitor.archive()
    except Exception as exc:
        evidence["errors"].append("monitor archive: " + str(exc))
    write_json(ctx.artifact_dir / "cache-gate-evidence.json", evidence)
    evidence["curve_source"] = "prometheus"
    result = analyze(evidence)
    path = freeze_gate(ctx.artifact_dir, "cache", evidence, result)
    return StageOutput(
        checks=[
            gate_check("cache_stability", result, path,
                       "valid survivor cache measurement without sustained hit collapse")
        ],
        artifacts=[
            str(ctx.artifact_dir / name)
            for name in (
                "cache-gate-evidence.json",
                "cache-gate-result.json",
                "cache-gate-manifest.json",
            )
        ],
    )


HANDLERS = [
    StageHandler("cache_scale_in_observe", validate, observe, {"evidence": "gate_evidence"}),
    StageHandler(
        "cache_scale_in_check",
        check_validate,
        check,
        {},
        checks=frozenset({"cache_stability"}),
    ),
]
