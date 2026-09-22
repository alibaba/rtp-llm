"""Fixed-window observation over the existing Java flow and runtime lifecycle."""

import json
import math
import time
from pathlib import Path

from scenario.contracts import CheckResult, StageHandler, StageOutput
from scenario.actions.elastic import _validate
from workload.performance_gate import validate, report, analyze, trace_workload_sha, ENGINE_TPS, write_evidence
from traffic.traffic_source import sha256_file


def observe_validate(params, plan):
    p = _validate(params, plan, {"flow", "criteria"}, {"flow", "criteria"})
    plan.reference(p["flow"], "java_flow")
    validate(p["criteria"])
    return p


def provenance(ctx, flow, criteria):
    from runtime.harness import MOCK_JAR

    trace = dict(flow.trace_manifest)
    trace["workload_sha256"] = trace_workload_sha(trace["path"])
    env = json.loads((flow.directory / "flow-input.json").read_text())["environment"]
    return dict(
        benchmark_id=criteria["benchmark_id"],
        master_artifact=json.loads(
            (ctx.env.run_dir / "master-artifact.json").read_text()
        ),
        actual_master_config=json.loads(
            (ctx.env.run_dir / "actual-master-config.json").read_text()
        ),
        mock_jar_sha256=sha256_file(MOCK_JAR),
        performance=json.loads(ctx.env.perf_file.read_text()),
        topology=dict(prefill=ctx.env.spec.n_prefill, decode=ctx.env.spec.n_decode),
        capacity=dict(
            prefill_cache_blocks=ctx.env.spec.prefill_cache_blocks,
            decode_cache_blocks=ctx.env.spec.decode_cache_blocks,
            mock_extra_args=ctx.env.spec.mock_extra_args,
        ),
        trace=trace,
        client_environment={
            k: v
            for k, v in env.items()
            if k
            not in {
                "TRACE_FILE",
                "FLOW_CONTROL_DIR",
                "FLOW_RUN_ID",
                "GRPC_TARGET",
                "OUTPUT_DIR",
            }
        },
        analyzer_sha256=sha256_file(
            Path(__file__).parents[2] / "workload/performance_gate.py"
        ),
    )


def observe(ctx, p, deadline):
    flow = ctx.resource(p["flow"], "java_flow")
    c = p["criteria"]
    origin = time.time() * 1000
    lo = origin + c["warmup_s"] * 1000
    hi = lo + c["measure_s"] * 1000
    evidence = dict(
        schema_version=1,
        criteria=c,
        errors=[],
        samples=[],
        window=dict(start_epoch_ms=lo, end_epoch_ms=hi),
        provenance={},
    )
    try:
        evidence["provenance"] = provenance(ctx, flow, c)
        while True:
            deadline.check()
            state = flow.status()
            epoch = time.time() * 1000
            evidence["samples"].append(
                dict(
                    epoch_ms=epoch,
                    state=state.get("state"),
                    process_returncode=state.get("process_returncode"),
                )
            )
            if (
                state.get("state") != "SENDING"
                or state.get("process_returncode") is not None
            ):
                raise ValueError("traffic ended before measurement completed")
            if epoch >= hi:
                break
            deadline.sleep(max(0, min(c["sample_s"], (hi - time.time() * 1000) / 1000)))
    except Exception as exc:
        evidence["errors"].append(str(exc))
    path = ctx.artifact_dir / "performance-gate-evidence.json"
    path.write_text(json.dumps(evidence, indent=2))
    return StageOutput(
        {"evidence": ctx.register_resource("snapshot", evidence, historical=True)},
        artifacts=[str(path)],
    )


def finish_validate(params, plan):
    p = _validate(params, plan, {"flow", "evidence"}, {"flow", "evidence"})
    plan.reference(p["flow"], "java_flow")
    plan.reference(p["evidence"], "snapshot")
    return p


def finish(ctx, p, deadline):
    flow = ctx.resource(p["flow"], "java_flow")
    e = ctx.resource(p["evidence"], "snapshot")
    try:
        flow.stop_sending(deadline)
        flow.drain(deadline)
    except Exception as exc:
        e["errors"].append("drain: " + str(exc))
    e["flow"] = flow.evidence_snapshot()
    if "engine_tps" in e["criteria"]:
        try:
            # Query only the contract metrics; fetching all 240 engines' series
            # and discarding most of them after JSON decoding is unbounded work.
            e["engine_tps_samples"] = []
            cursor = e["window"]["start_epoch_ms"] / 1000
            end = e["window"]["end_epoch_ms"] / 1000
            selector = '{job="mock",__name__=~"' + "|".join(ENGINE_TPS) + '"}'
            while cursor < end:
                right = min(end, cursor + 60)
                expression = selector + f"[{math.ceil((right - cursor) * 1000)}ms]"
                e["engine_tps_samples"].extend(ctx.monitor.query(expression, right))
                cursor = right
        except Exception as exc:
            e["errors"].append("engine TPS collection: " + str(exc))
    # Preserve terminal evidence even if analysis or presentation later times out.
    write_evidence(ctx.artifact_dir / "performance-gate-evidence.json", e)
    result = analyze(e)
    bundle = report(ctx.artifact_dir, e, result)
    return StageOutput(
        checks=[
            CheckResult(
                "absolute_performance",
                "ERROR" if result["verdict"] == "INVALID" else result["verdict"],
                actual=result,
                expected="single run satisfies all absolute criteria",
            )
        ],
        artifacts=[
            str(ctx.artifact_dir / "performance-gate-evidence.json"),
            str(bundle / "analysis.json"),
            str(bundle / "report.html"),
        ],
    )


HANDLERS = [
    StageHandler(
        "performance_observe", observe_validate, observe, {"evidence": "snapshot"}
    ),
    StageHandler(
        "performance_finish",
        finish_validate,
        finish,
        {},
        checks=frozenset({"absolute_performance"}),
    ),
]
