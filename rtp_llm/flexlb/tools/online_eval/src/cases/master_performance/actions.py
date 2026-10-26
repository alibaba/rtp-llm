"""Fixed-window observation over the existing Java flow and runtime lifecycle."""

from functools import partial

from artifacts.gate import gate_check
from artifacts.gate_inputs import compact_flow, new_evidence
from cases.master_performance.inputs import effective_profile, engine_roles
from cases.master_performance.inputs import validate as validate_criteria
from file_digest import sha256_file
from file_io import write_json
from runtime.observation import SampleBudget, poll_samples
from scenario.contracts import StageHandler, StageOutput
from scenario.parameters import reference_parameters, validate_fields


def observe_validate(params, plan):
    fields = {"flow", "criteria", "gate_input", "observation"}
    params = validate_fields(params, plan, fields, fields)
    plan.reference(params["flow"], "java_flow")
    validate_criteria(params["criteria"], params["gate_input"])
    from cases.master_performance.inputs import observation_contract

    durations = observation_contract(params["observation"])
    if any(params["criteria"][key] != value for key, value in durations.items()):
        raise ValueError("observation windows disagree with frozen criteria")
    from flexlb_profile_data import PROFILES

    if set(params["criteria"].get("engine_tps_by_profile", {})) - set(PROFILES):
        raise ValueError("engine_tps_by_profile requires registered profiles")
    return params


def provenance(ctx, flow):
    from artifacts.provenance import gate_provenance
    from cases.master_performance import analysis

    return gate_provenance(ctx, flow, source_files=ctx.services.source_files(), analyzer_file=analysis.__file__)


def observe(ctx, params, deadline):
    flow = ctx.resource(params["flow"], "java_flow")
    criteria = effective_profile(params["criteria"], ctx.instance["profile"])
    # With no warmup the first observed state defines the window origin;
    # provenance/HTTP work must not create an uncovered gap before that sample.
    initial_state = initial_provenance = initial_error = None
    if criteria["warmup_s"] == 0:
        try:
            initial_provenance = provenance(ctx, flow)
            initial_state = flow.control_status()
        except Exception as exc:
            initial_error = exc
    clock = ctx.start_observation()
    origin_epoch_ms = clock.origin_epoch_s * 1000
    window_start_ms = origin_epoch_ms + criteria["warmup_s"] * 1000
    window_end_ms = window_start_ms + criteria["measure_s"] * 1000
    evidence = new_evidence(
        "performance_evidence_schema_version",
        clock,
        criteria,
        instance=ctx.instance["id"],
        env_epoch=ctx.env_epoch,
        version=2,
        gate_input=params["gate_input"],
        capture_limits=params["observation"]["capture"],
        window=dict(start_epoch_ms=window_start_ms, end_epoch_ms=window_end_ms),
    )
    try:
        if initial_error is not None:
            raise initial_error
        evidence["provenance"].update(initial_provenance if initial_provenance is not None else provenance(ctx, flow))
        budget = SampleBudget(params["observation"]["capture"])
        evidence["window_declarations"] = params["observation"]["windows"]

        def sample():
            nonlocal initial_state
            if initial_state is not None:
                state, initial_state = initial_state, None
                timestamp = clock.origin_monotonic_s
            else:
                state = flow.control_status()
                timestamp = ctx.clock()
            row = dict(clock.stamp(timestamp), state=state.get("state"), process_returncode=state.get("process_returncode"))
            row["epoch_ms"] = row["epoch_s"] * 1000
            budget.append(row)
            evidence["samples"].append(row)
            return row

        end = clock.origin_monotonic_s + criteria["warmup_s"] + criteria["measure_s"]
        measurement_started = False
        for row in poll_samples(deadline, criteria["sample_s"], sample, until=end, immediate=True):
            state = row
            epoch = row["epoch_ms"]
            if state.get("state") != "SENDING" or state.get("process_returncode") is not None:
                raise ValueError("traffic ended before measurement completed")
            if epoch >= window_start_ms and not measurement_started:
                ctx.record_event(
                    "measurement_started",
                    timestamp=dict(epoch_s=window_start_ms / 1000, monotonic_s=clock.origin_monotonic_s + criteria["warmup_s"]),
                )
                measurement_started = True
            if epoch >= window_end_ms:
                ctx.record_event("measurement_ended", timestamp=dict(epoch_s=window_end_ms / 1000, monotonic_s=end))
                break
        ctx.record_event("observation_end")
    except Exception as exc:
        evidence["errors"].append(str(exc))
    path = ctx.artifact_dir / "performance-gate-evidence.json"
    write_json(path, evidence)
    return StageOutput({"evidence": ctx.register_resource("gate_evidence", evidence, historical=True)}, artifacts=[str(path)])


def finish(ctx, params, deadline):
    flow = ctx.resource(params["flow"], "java_flow")
    evidence = ctx.resource(params["evidence"], "gate_evidence")
    evidence["flow"] = compact_flow(flow.evidence_snapshot())
    journal = flow.directory / "client_lifecycle.jsonl"
    evidence["flow"]["journal"] = dict(path=str(journal), sha256=sha256_file(journal))
    if "engine_tps" in evidence["criteria"]:
        try:
            # Query only the contract metrics; fetching all 240 engines' series
            # and discarding most of them after JSON decoding is unbounded work.
            from monitoring.windows import capture_window

            engine_roles(evidence["gate_input"], evidence["criteria"]["engine_tps"])
            raw = capture_window(
                ctx.monitor.client,
                evidence["gate_input"]["fields"],
                source="mock",
                windows=[(evidence["window"]["start_epoch_ms"] / 1000, evidence["window"]["end_epoch_ms"] / 1000)],
                epoch=evidence["provenance"]["env_epoch"],
                deadline=deadline,
                limits=evidence["capture_limits"],
                directory=ctx.artifact_dir / "performance-measurement",
            )
            evidence["engine_tps_samples"] = [row for rows in raw.field_rows().values() for row in rows]
            evidence["raw_evidence"] = dict(
                path="performance-measurement/manifest.json", sha256=sha256_file(ctx.artifact_dir / "performance-measurement/manifest.json")
            )
        except Exception as exc:
            evidence["errors"].append("engine TPS collection: " + str(exc))
    # Preserve terminal evidence even if analysis or presentation later times out.
    write_json(ctx.artifact_dir / "performance-gate-evidence.json", evidence)
    from artifacts.gate import commit_gate
    from cases.master_performance.analysis import analyze

    result, path = commit_gate(ctx.artifact_dir, "performance", evidence, analyze)
    return StageOutput(
        checks=[gate_check("absolute_performance", result, path, "single run satisfies all absolute criteria")],
        artifacts=[
            str(ctx.artifact_dir / "performance-gate-evidence.json"),
            str(path),
            str(ctx.artifact_dir / "performance-gate-manifest.json"),
        ],
    )


HANDLERS = [
    StageHandler("performance_observe", observe_validate, observe, {"evidence": "gate_evidence"}),
    StageHandler(
        "performance_finish",
        partial(reference_parameters, resources={"flow": "java_flow", "evidence": "gate_evidence"}),
        finish,
        {},
        checks=frozenset({"absolute_performance"}),
    ),
]
