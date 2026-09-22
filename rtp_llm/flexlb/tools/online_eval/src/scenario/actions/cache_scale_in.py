"""Continuous Java traffic across a concurrent graceful Prefill scale-in."""

import json
import math
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from scenario.contracts import CheckResult, StageHandler, StageOutput
from scenario.actions.elastic import _validate, _http
from workload.cache_gate import align_send_counters, analyze, window, write_report

FIELDS = {
    "flow",
    "target_p",
    "warmup_timeout_s",
    "baseline_s",
    "observe_s",
    "sample_s",
    "window_s",
    "step_s",
    "sustain_s",
    "max_gap_s",
    "qps",
    "qps_tolerance",
    "baseline_min_hit",
    "baseline_max_spread",
    "absolute_min_hit",
    "max_drop",
    "min_completed",
    "min_waiting",
    "max_pacing_lag_ms",
    "drain_timeout_ms",
    "topology_timeout_s",
}
OPTIONAL_FIELDS = {"intermediate_p", "intermediate_hold_s"}


def validate(params, plan):
    p = _validate(params, plan, FIELDS | OPTIONAL_FIELDS, FIELDS)
    plan.reference(p["flow"], "java_flow")
    if plan.environment.get("discovery") != "discovery_file":
        raise ValueError("scale-in requires dynamic discovery_file")
    for k in FIELDS - {"flow"}:
        if type(p[k]) not in (int, float) or not math.isfinite(p[k]) or p[k] < 0:
            raise ValueError(k + " must be finite and nonnegative")
    for k in ("target_p", "min_completed", "min_waiting", "drain_timeout_ms"):
        if type(p[k]) is not int or p[k] < 1:
            raise ValueError(k + " must be a positive integer")
    if not 1 <= p["target_p"] < plan.environment["n_prefill"] <= 512:
        raise ValueError("scale-in requires fewer target P and at most 512 initial P")
    if OPTIONAL_FIELDS & p.keys():
        if not OPTIONAL_FIELDS <= p.keys():
            raise ValueError(
                "intermediate P and hold duration must be specified together"
            )
        if (
            type(p["intermediate_p"]) is not int
            or not p["target_p"]
            < p["intermediate_p"]
            < plan.environment["n_prefill"]
        ):
            raise ValueError(
                "intermediate P must lie strictly between initial and target P"
            )
        hold = p["intermediate_hold_s"]
        if (
            type(hold) not in (int, float)
            or not math.isfinite(hold)
            or hold < p["baseline_s"]
        ):
            raise ValueError("intermediate hold must cover a full baseline window")
    for k in (
        "qps_tolerance",
        "baseline_min_hit",
        "baseline_max_spread",
        "absolute_min_hit",
        "max_drop",
    ):
        if p[k] > 1:
            raise ValueError(k + " must be a fraction")
    if not 0 < p["sample_s"] <= p["step_s"] <= p["window_s"] <= p["baseline_s"] / 2:
        raise ValueError("sampling/window/baseline durations are inconsistent")
    if p["max_gap_s"] < p["sample_s"] or p["warmup_timeout_s"] < p["baseline_s"]:
        raise ValueError("insufficient warmup or sample gap budget")
    if p["observe_s"] < p["window_s"] + p["sustain_s"] or p["qps"] <= 0:
        raise ValueError("observation cannot cover sustained collapse")
    if p["drain_timeout_ms"] > 30000 or p["topology_timeout_s"] <= 0:
        raise ValueError("removal must have bounded drain and topology budgets")
    from scenario.compiler import environment

    for profile in plan.profiles:
        resolved = environment(plan.environment, plan.path, profile)["resolved_config"]
        stale_ms = resolved["workerRegistry"]["health"]["statusStaleAfterMs"]
        if (
            p["drain_timeout_ms"] <= stale_ms
            or p["topology_timeout_s"] * 1000 <= stale_ms
        ):
            raise ValueError(
                "drain and topology budgets must exceed Master status staleness"
            )
    return p


def _master_count(ctx, deadline):
    req = urllib.request.Request(
        f"http://127.0.0.1:{ctx.env.master_http_port}/rtp_llm/master/info",
        data=b"{}",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=min(3, deadline.remaining())) as response:
        data = json.load(response)
    value = data["worker_summary"]["PREFILL"]["alive"]
    if type(value) is not int:
        raise ValueError("master topology missing")
    return value


def observe(ctx, p, deadline):
    flow = ctx.resource(p["flow"], "java_flow")
    origin = ctx.clock()
    evidence = dict(
        criteria={k: v for k, v in p.items() if k != "flow"},
        samples=[],
        events=[],
        errors=[],
        initial_engines=[],
        survivors=[],
        baseline_start=0,
        baseline_end=0,
        post_start=0,
        post_end=0,
        max_pacing_lag_ms=None,
    )
    from traffic.traffic_source import sha256_file
    from runtime.harness import API_JAR, MOCK_JAR
    from workload import cache_gate

    evidence["provenance"] = dict(
        instance=ctx.instance["id"],
        topology=dict(prefill=ctx.env.spec.n_prefill, decode=ctx.env.spec.n_decode),
        configuration_sha256=ctx.instance.get("implementation", {}).get(
            "configuration_sha256"
        ),
        trace=flow.trace_manifest,
        files={
            str(path): sha256_file(path)
            for path in (API_JAR, MOCK_JAR, __file__, cache_gate.__file__)
        },
        performance=json.loads(ctx.env.perf_file.read_text()),
        master_config=json.loads((ctx.env.run_dir / "master_config.json").read_text()),
    )
    actual_config = ctx.env.run_dir / "actual-master-config.json"
    if actual_config.exists():
        evidence["provenance"]["mock_formula_config"] = evidence["provenance"][
            "master_config"
        ]
        evidence["provenance"]["master_config"] = json.loads(actual_config.read_text())
        evidence["provenance"]["historical_master"] = json.loads(
            (ctx.env.run_dir / "historical-master.json").read_text()
        )
    path = ctx.artifact_dir / "cache-gate-evidence.json"

    def event(name):
        evidence["events"].append(
            dict(name=name, t=ctx.clock() - origin, epoch_s=time.time())
        )

    def sample():
        deadline.check()
        from monitoring.session import engine_sample
        engines = engine_sample(
            f"http://127.0.0.1:{ctx.env.mock_http_port}/metrics?per_engine=true",
            timeout=min(3, deadline.remaining()),
        )
        state = flow.status()
        row = dict(
            t=ctx.clock() - origin,
            epoch_s=time.time(),
            engines=engines,
            started=state["observed_started"],
            terminal=state["observed_terminal"],
            master_p=_master_count(ctx, deadline),
            waiting=sum(e["waiting"] for e in engines.values()),
            running=sum(e["running"] for e in engines.values()),
        )
        evidence["samples"].append(row)
        if state["process_returncode"] is not None or state.get("state") != "SENDING":
            raise ValueError("Java traffic stopped before observation completed")
        return row

    pool = None
    futures = []
    try:
        first = sample()
        initial = sorted(first["engines"])
        evidence["initial_engines"] = initial
        evidence["survivors"] = initial[: p["target_p"]]
        if first["master_p"] != len(initial):
            raise ValueError("initial discovery not converged")
        event("warmup_start")
        while True:
            deadline.sleep(p["sample_s"])
            row = sample()
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
                    event("baseline_ready")
                    break
            if t >= p["warmup_timeout_s"]:
                raise ValueError("stable warm baseline not reached")
        intermediate_removals = []
        if "intermediate_p" in p:
            intermediate = p["intermediate_p"]
            event("intermediate_withdraw_start")
            with ThreadPoolExecutor(
                max_workers=len(initial) - intermediate,
                thread_name_prefix="scale-in-intermediate",
            ) as intermediate_pool:
                intermediate_futures = [
                    intermediate_pool.submit(
                        _http,
                        ctx.ops,
                        "remove_engine",
                        deadline,
                        dict(
                            engine=name,
                            mode="graceful",
                            drain_timeout_ms=p["drain_timeout_ms"],
                        ),
                    )
                    for name in initial[intermediate:]
                ]
                topology_end = ctx.clock() + p["topology_timeout_s"]
                while True:
                    deadline.sleep(p["sample_s"])
                    row = sample()
                    if set(row["engines"]) == set(initial[:intermediate]):
                        event("intermediate_topology_observed")
                        break
                    if ctx.clock() >= topology_end:
                        raise ValueError("intermediate topology did not converge")
                hold_end = row["t"] + p["intermediate_hold_s"]
                while row["t"] < hold_end:
                    deadline.sleep(min(p["sample_s"], hold_end - row["t"]))
                    row = sample()
                    if set(row["engines"]) != set(initial[:intermediate]):
                        raise ValueError(
                            "intermediate topology changed during hold"
                        )
                evidence["intermediate_window"] = window(
                    evidence["samples"],
                    row["t"] - p["baseline_s"],
                    row["t"],
                    initial[:intermediate],
                    p["max_gap_s"],
                )
                intermediate_removals = [
                    future.result(timeout=max(0.01, deadline.remaining()))
                    for future in intermediate_futures
                ]
                if any(
                    not removal.get("drained", False)
                    for removal in intermediate_removals
                ):
                    evidence["errors"].append(
                        "intermediate graceful drain timed out; continuing observation"
                    )
            initial = initial[:intermediate]

        removed = initial[p["target_p"] :]
        event("withdraw_start")
        # Every removal withdraws discovery before waiting. Parallel requests do
        # not serialize the drain periods; the Java mutation lock only covers rewrite.
        pool = ThreadPoolExecutor(
            max_workers=len(removed), thread_name_prefix="scale-in"
        )
        futures = [
            pool.submit(
                _http,
                ctx.ops,
                "remove_engine",
                deadline,
                dict(engine=n, mode="graceful", drain_timeout_ms=p["drain_timeout_ms"]),
            )
            for n in removed
        ]
        topology_end = ctx.clock() + p["topology_timeout_s"]
        while True:
            deadline.sleep(p["sample_s"])
            row = sample()
            if set(row["engines"]) == set(evidence["survivors"]):
                evidence["post_start"] = row["t"]
                event("target_topology_observed")
                break
            if ctx.clock() >= topology_end:
                raise ValueError("target topology did not converge")
        end = evidence["post_start"] + p["observe_s"]
        while row["t"] < end:
            deadline.sleep(min(p["sample_s"], end - row["t"]))
            row = sample()
        evidence["post_end"] = row["t"]
        event("observation_end")
        evidence["removals"] = intermediate_removals + [
            f.result(timeout=max(0.01, deadline.remaining())) for f in futures
        ]
        if any(not r.get("drained", False) for r in evidence["removals"]):
            evidence["errors"].append(
                "graceful drain timed out; removal introduced request loss"
            )
    except Exception as exc:
        evidence["errors"].append(str(exc))
    finally:
        if pool:
            # The underlying HTTP calls and server drains are deadline-bounded.
            pool.shutdown(wait=True, cancel_futures=True)
        path.write_text(json.dumps(evidence, indent=2))
    return StageOutput(
        {"evidence": ctx.register_resource("snapshot", evidence, historical=True)},
        artifacts=[str(path)],
    )


def check_validate(params, plan):
    p = _validate(params, plan, {"flow", "evidence"}, {"flow", "evidence"})
    plan.reference(p["flow"], "java_flow")
    plan.reference(p["evidence"], "snapshot")
    return p


def check(ctx, p, deadline):
    evidence = ctx.resource(p["evidence"], "snapshot")
    flow = ctx.resource(p["flow"], "java_flow")
    snapshot = flow.evidence_snapshot()
    if not snapshot["complete"]:
        evidence["errors"].extend(snapshot["errors"])
    else:
        align_send_counters(evidence, snapshot["issued"])
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
    if getattr(ctx, "monitor", None) is not None:
        ctx.monitor.archive()
    evidence["curve_source"] = "prometheus"
    result = analyze(evidence)
    write_report(ctx.artifact_dir, evidence, result)
    status = {"INVALID": "ERROR", "FAIL": "FAIL", "PASS": "PASS"}[result["verdict"]]
    return StageOutput(
        checks=[
            CheckResult(
                "cache_stability",
                status,
                actual=result,
                expected="valid overload without sustained cache collapse",
            )
        ],
        artifacts=[
            str(ctx.artifact_dir / name)
            for name in (
                "cache-gate-evidence.json",
                "reports/run/cache-scale-in/analysis.json",
                "reports/run/cache-scale-in/report.html",
            )
        ],
    )


HANDLERS = [
    StageHandler("cache_scale_in_observe", validate, observe, {"evidence": "snapshot"}),
    StageHandler(
        "cache_scale_in_check",
        check_validate,
        check,
        {},
        checks=frozenset({"cache_stability"}),
    ),
]
