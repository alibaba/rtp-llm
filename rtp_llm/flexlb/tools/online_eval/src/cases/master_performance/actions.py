"""Fixed-window observation over the existing Java flow and runtime lifecycle."""

import json
import math
import time
from pathlib import Path

from scenario.contracts import CheckResult, StageHandler, StageOutput
from scenario.parameters import validate_fields
from cases.master_performance.analysis import analyze, validate, for_profile
from workload.gate_evidence import compact_flow, trace_workload_sha, write_evidence
from cases.master_performance.publication import publish_performance
from cases.master_performance.inputs import engine_tps
from traffic.traffic_source import sha256_file


def observe_validate(params, plan):
    fields = {"flow", "criteria", "gate_input"}
    p = validate_fields(params, plan, fields, fields)
    plan.reference(p["flow"], "java_flow")
    validate(p["criteria"], p["gate_input"])
    from flexlb_profile_data import PROFILES
    if set(p["criteria"].get("engine_tps_by_profile", {})) - set(PROFILES):
        raise ValueError("engine_tps_by_profile requires registered profiles")
    return p


def provenance(ctx, flow, criteria):
    from runtime.paths import MOCK_JAR

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
            Path(__file__).parents[2] / "cases/master_performance/analysis.py"
        ),
    )


def observe(ctx, p, deadline):
    flow = ctx.resource(p["flow"], "java_flow")
    c = for_profile(p["criteria"], ctx.instance["profile"], p["gate_input"])
    origin = time.time() * 1000
    lo = origin + c["warmup_s"] * 1000
    hi = lo + c["measure_s"] * 1000
    evidence = dict(
        performance_evidence_schema_version=1,
        criteria=c,
        gate_input=p["gate_input"],
        errors=[],
        samples=[],
        window=dict(start_epoch_ms=lo, end_epoch_ms=hi),
        provenance=dict(instance=ctx.instance["id"]),
    )
    try:
        evidence["provenance"].update(provenance(ctx, flow, c))
        while True:
            deadline.check()
            state = flow.control_status()
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
    p = validate_fields(params, plan, {"flow", "evidence"}, {"flow", "evidence"})
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
    e["flow"] = compact_flow(flow.evidence_snapshot())
    journal = flow.directory / "client_lifecycle.jsonl"
    e["flow"]["journal"] = dict(path=str(journal), sha256=sha256_file(journal))
    if "engine_tps" in e["criteria"]:
        try:
            # Query only the contract metrics; fetching all 240 engines' series
            # and discarding most of them after JSON decoding is unbounded work.
            e["engine_tps_samples"] = []
            names = engine_tps(e["gate_input"], e["criteria"]["engine_tps"])["metric_roles"]
            for metric_id in names:
                e["engine_tps_samples"].extend(ctx.monitor.metric_rows(
                    metric_id, source="mock", start=e["window"]["start_epoch_ms"] / 1000,
                    end=e["window"]["end_epoch_ms"] / 1000))
        except Exception as exc:
            e["errors"].append("engine TPS collection: " + str(exc))
    try:
        ctx.monitor.archive()
    except Exception as exc:
        e["errors"].append("monitor archive: " + str(exc))
    # Preserve terminal evidence even if analysis or presentation later times out.
    write_evidence(ctx.artifact_dir / "performance-gate-evidence.json", e)
    result = analyze(e)
    bundle = publish_performance(ctx.artifact_dir, e, result)
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
