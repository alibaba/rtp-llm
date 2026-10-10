"""Fixed-window observation over the existing Java flow and runtime lifecycle."""

from scenario.contracts import CheckResult, StageHandler, StageOutput
from scenario.parameters import validate_fields
from cases.master_performance.analysis import analyze
from cases.master_performance.inputs import validate, for_profile
from workload.gate_evidence import compact_flow, write_evidence, new_evidence
from cases.master_performance.publication import publish_performance
from cases.master_performance.inputs import engine_roles
from traffic.traffic_source import sha256_file
from runtime.observation import ObservationClock, SampleBudget, poll_samples, verdict_status


def observe_validate(params, plan):
    fields = {"flow", "criteria", "gate_input", "observation"}
    p = validate_fields(params, plan, fields, fields)
    plan.reference(p["flow"], "java_flow")
    validate(p["criteria"], p["gate_input"])
    from cases.master_performance.inputs import observation_contract
    durations = observation_contract(p["observation"])
    if any(p["criteria"][key] != value for key, value in durations.items()):
        raise ValueError("observation windows disagree with frozen criteria")
    from flexlb_profile_data import PROFILES
    if set(p["criteria"].get("engine_tps_by_profile", {})) - set(PROFILES):
        raise ValueError("engine_tps_by_profile requires registered profiles")
    return p


def provenance(ctx, flow):
    from workload.run_provenance import gate_provenance
    from cases.master_performance import analysis, program, inputs
    return gate_provenance(ctx, flow,
        source_files=(__file__, analysis.__file__, program.__file__, inputs.__file__),
        analyzer_file=analysis.__file__)


def observe(ctx, p, deadline):
    flow = ctx.resource(p["flow"], "java_flow")
    c = for_profile(p["criteria"], ctx.instance["profile"], p["gate_input"])
    clock = ObservationClock.start(ctx)
    origin = clock.origin_epoch_s * 1000
    lo = origin + c["warmup_s"] * 1000
    hi = lo + c["measure_s"] * 1000
    evidence = new_evidence("performance_evidence_schema_version", clock, c,
        instance=ctx.instance["id"], env_epoch=ctx.env_epoch, version=2,
        gate_input=p["gate_input"],
        window=dict(start_epoch_ms=lo, end_epoch_ms=hi),
    )
    try:
        evidence["provenance"].update(provenance(ctx, flow))
        budget = SampleBudget(p["observation"]["capture"])
        evidence["window_declarations"] = p["observation"]["windows"]
        def sample():
            state = flow.control_status()
            row = dict(clock.stamp(ctx), state=state.get("state"),
                       process_returncode=state.get("process_returncode"))
            row["epoch_ms"] = row["epoch_s"] * 1000
            budget.append(row)
            evidence["samples"].append(row)
            return row
        end = clock.origin_monotonic_s + c["warmup_s"] + c["measure_s"]
        for row in poll_samples(deadline, c["sample_s"], sample, until=end, immediate=True):
            state = row
            epoch = row["epoch_ms"]
            if (
                state.get("state") != "SENDING"
                or state.get("process_returncode") is not None
            ):
                raise ValueError("traffic ended before measurement completed")
            if epoch >= hi:
                break
        ctx.record_event("observation_end")
    except Exception as exc:
        evidence["errors"].append(str(exc))
    path = ctx.artifact_dir / "performance-gate-evidence.json"
    write_evidence(path, evidence)
    return StageOutput(
        {"evidence": ctx.register_resource("gate_evidence", evidence, historical=True)},
        artifacts=[str(path)],
    )


def finish_validate(params, plan):
    p = validate_fields(params, plan, {"flow", "evidence"}, {"flow", "evidence"})
    plan.reference(p["flow"], "java_flow")
    plan.reference(p["evidence"], "gate_evidence")
    return p


def finish(ctx, p, deadline):
    flow = ctx.resource(p["flow"], "java_flow")
    e = ctx.resource(p["evidence"], "gate_evidence")
    e["flow"] = compact_flow(flow.evidence_snapshot())
    journal = flow.directory / "client_lifecycle.jsonl"
    e["flow"]["journal"] = dict(path=str(journal), sha256=sha256_file(journal))
    if "engine_tps" in e["criteria"]:
        try:
            # Query only the contract metrics; fetching all 240 engines' series
            # and discarding most of them after JSON decoding is unbounded work.
            e["engine_tps_samples"] = []
            names = engine_roles(e["gate_input"], e["criteria"]["engine_tps"])
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
                verdict_status(result["verdict"]),
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
        "performance_observe", observe_validate, observe, {"evidence": "gate_evidence"}
    ),
    StageHandler(
        "performance_finish",
        finish_validate,
        finish,
        {},
        checks=frozenset({"absolute_performance"}),
    ),
]
