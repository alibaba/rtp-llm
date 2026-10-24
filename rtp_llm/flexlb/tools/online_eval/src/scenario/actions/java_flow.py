"""Java-generated traffic groups with Python-owned preconditions and checkpoints."""

import json
from pathlib import Path

from adapters.client_journal import MAX_EVIDENCE_EVENTS
from adapters.java_flow import JavaFlowGroup
from adapters.java_flow_spec import JavaFlowSpec
from traffic.traffic_source import materialize
from traffic.contracts import source_priority

from scenario.contracts import StageHandler, StageOutput
from scenario.parameters import validate_fields

def _start_validate(params, plan):
    if isinstance(params, dict) and "trace" in params:
        raise ValueError("legacy trace shorthand retired; use trace/prefix_lineage")
    JavaFlowSpec.read(params)
    return params


def _start(ctx, p, deadline):
    if ctx.instance["environment"].get("master_test_control", False):
        from adapters.master_control import affinity_request
        initial = affinity_request(ctx, deadline)
        expected = ctx.instance["environment"]["resolved_config"]["router"]["roles"]["prefill"]["cacheAffinity"]["maxExtraTtftMs"]
        if initial.get("maxExtraTtftMs") != expected:
            raise ValueError("unexpected initial affinity test policy")
        (ctx.artifact_dir / "master-test-control.json").write_text(json.dumps(initial, indent=2))
    directory = ctx.artifact_dir / "flows" / p["group_id"]
    directory.parent.mkdir(exist_ok=True)
    from artifacts.startup import startup_phase
    with startup_phase(ctx.artifact_dir, 'traffic/input', deadline):
        trace = materialize(
            directory.parent / (p["group_id"] + ".jsonl"),
            p["source"],
            ctx.instance["id"] + ":" + p["group_id"],
            Path(ctx.instance["source_path"]).parent,
            max_requests=p.get("max_requests"),
        )
    environment = p["client"]
    client = ctx.services.environment.client_operations(p["jvm_xms"], p["jvm_xmx"])
    # This is a retained-evidence ceiling, not a prediction of Java scheduling.
    # The journal grows on demand and fails explicitly when the ceiling is exceeded.
    event_budget = MAX_EVIDENCE_EVENTS
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

    from adapters.resource_evidence import flow_evidence
    handle = ctx.register_resource("java_flow", flow, cleanup=cleanup, evidence=flow_evidence, request_stop=flow.request_stop)
    flow.start(trace, environment, deadline,
               target=f"127.0.0.1:{ctx.env.master_http_port + 2}",
               priority=source_priority(p["source"]))
    return StageOutput({"flow": handle}, artifacts=[str(directory / "flow-input.json")])


def _flow_validate(params, plan):
    p = validate_fields(params, plan, {"flow"}, {"flow"})
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



HANDLERS = [
    StageHandler("java_flow_start", _start_validate, _start, {"flow": "java_flow"}),
    StageHandler("java_flow_stop", _flow_validate, _stop, {"snapshot": "snapshot"}),
    StageHandler("java_flow_drain", _flow_validate, _drain, {"snapshot": "snapshot"}),
]
