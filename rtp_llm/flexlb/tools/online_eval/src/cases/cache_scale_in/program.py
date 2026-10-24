"""Warm once, shrink in one step, keep the Java sender running, adjudicate offline."""

from adapters.client_configuration import ClientConfiguration
from cases.cache_scale_in.actions import HANDLERS as ACTION_HANDLERS
from cases.cache_scale_in.inputs import (
    CRITERIA,
    ENGINE_FIELDS,
    INPUTS,
    compile_checks,
    engine_counters,
    observation_contract,
)
from cases.cache_scale_in.report import validate_view, write_report
from scenario.builder import output
from scenario.metric_inputs import bind_mock_engine_metrics
from traffic.contracts import driver, uniform_gate_flow


def default(case):
    data = case.config
    flow = driver(data["traffic"], "java_flow")
    inputs = data["observation"]["inputs"]
    if not isinstance(inputs, dict) or set(inputs) != {"engine_counters"}:
        raise ValueError("cache gate requires engine_counters input")
    engine_counters(inputs["engine_counters"])
    bind_mock_engine_metrics(case, inputs["engine_counters"], ENGINE_FIELDS)
    client = ClientConfiguration.read(flow["client"])
    gate = CRITERIA.project(
        procedure=data["procedure"],
        observation=data["observation"],
        windows=observation_contract(data["observation"]),
        checks=compile_checks(case, data["checks"], data["analysis"]["collapse"]),
        runtime={"qps": client.playback.qps},
    )
    budget = gate["warmup_timeout_s"] + gate["topology_timeout_s"] + gate["observe_s"] + 10
    if gate["removal_mode"] != "graceful":
        raise ValueError("step requires one graceful scale-in; different fault sequences require a separate case")
    uniform_gate_flow(flow, client, duration_s=budget, trace_duration_s=budget)
    case.step("setup", "setup", timeout_s=180)
    case.step("traffic", "java_flow_start", params=flow, timeout_s=data["procedure"]["startup_timeout_s"])
    case.step(
        "scale_in",
        "cache_scale_in_observe",
        params=dict(
            flow=output("traffic", "flow"),
            criteria=gate,
            gate_input=inputs["engine_counters"],
            observation=data["observation"],
        ),
        timeout_s=budget + 40,
    )
    case.step("stop", "java_flow_stop", params={"flow": output("traffic", "flow")})
    case.step(
        "drain",
        "java_flow_drain",
        params={"flow": output("traffic", "flow")},
        timeout_s=client.limits.drain_timeout_s + 30,
    )
    case.observe(
        "gate",
        "cache_scale_in_check",
        timeout_s=data["procedure"]["analysis_timeout_s"],
        params={
            "flow": output("traffic", "flow"),
            "evidence": output("scale_in", "evidence"),
        },
    )
    case.step("teardown", "teardown")


from functools import partial

from artifacts.gate import render_gate
from cases.cache_scale_in import metrics
from scenario.registry import ReportView

render = partial(render_gate, name="cache", write_report=write_report)
VIEW = ReportView(validate_view, render, frozenset({"expected_behavior", "observed_behavior"}))


from scenario.contracts import ExecutionRequirements
from scenario.registry import CaseDefinition

CASE = CaseDefinition(
    execution_requirements=ExecutionRequirements(request_evidence=True, monitoring=True),
    builders={"default": default},
    input_schema=INPUTS,
    metrics=metrics,
    report_views={"cache_scale_in.yaml": VIEW},
    actions=tuple(ACTION_HANDLERS),
)
