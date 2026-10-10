"""Warm once, shrink in one step, keep the Java sender running, adjudicate offline."""

from cases.cache_scale_in.actions import HANDLERS as ACTION_HANDLERS

from cases.config import output
from traffic.playback_config import normalize
from cases.cache_scale_in.comparison import validate_policy
from cases.cache_scale_in.inputs import (
    engine_counters, ENGINE_COUNTER_UNITS, PROCEDURE_FIELDS, OBSERVATION_FIELDS, RULES, compile_checks,
)
from runtime.java_flow import JAVA_FLOW_INPUT_FIELDS
from traffic.contracts import driver

ANALYSIS_POLICY_VALIDATOR = validate_policy


def default(case):
    data = case.inputs(
        traffic=JAVA_FLOW_INPUT_FIELDS | {"kind"},
        procedure=PROCEDURE_FIELDS | {"analysis_timeout_s"},
        observation=OBSERVATION_FIELDS | {"inputs", "collapse"},
        checks=set(RULES) | {"collapse"},
    )
    flow = driver(data.traffic, "java_flow")
    inputs = data.observation["inputs"]
    gate = dict({k: v for k, v in data.procedure.items() if k != "analysis_timeout_s"},
                **{k: data.observation[k] for k in OBSERVATION_FIELDS},
                **compile_checks(case, data.checks, data.observation["collapse"]))
    if not isinstance(inputs, dict) or set(inputs) != {"engine_counters"}:
        raise ValueError("cache gate requires engine_counters input")
    engine_counters(inputs["engine_counters"])
    for field, binding in inputs["engine_counters"]["fields"].items():
        identity = binding["metric"]
        if field not in ENGINE_COUNTER_UNITS:
            raise ValueError("unknown engine gate field: " + field)
        case.metric(identity, unit=ENGINE_COUNTER_UNITS[field],
                    labels=("role", "engine_name", "engine_incarnation"), mode="scrape")
    client, _ = normalize(flow["client"])
    gate["qps"] = float(client["SEND_MODE_QPS"])
    if (
        client.get("LOOP") != "false"
        or client.get("SEND_MODE") != "uniform"
    ):
        raise ValueError("scale-in requires a nonlooping uniform Java workload")
    budget = (
        gate["warmup_timeout_s"] + gate["topology_timeout_s"] + gate["observe_s"] + 10
    )
    if gate["removal_mode"] != "graceful":
        raise ValueError("step requires one graceful scale-in; different fault sequences require a separate case")
    if (
        int(client["DURATION_S"]) < budget
        or flow["source"]["parameters"]["count"] < budget * gate["qps"]
    ):
        raise ValueError("traffic plan must cover worst-case observation duration")
    case.step("setup", "setup", timeout_s=180)
    case.step("traffic", "java_flow_start", params=flow, timeout_s=300)
    case.step(
        "scale_in",
        "cache_scale_in_observe",
        params=dict(gate, flow=output("traffic", "flow"), gate_input=inputs["engine_counters"]),
        timeout_s=budget + 40,
    )
    case.step("stop", "java_flow_stop", params={"flow": output("traffic", "flow")})
    case.step(
        "drain",
        "java_flow_drain",
        params={"flow": output("traffic", "flow")},
        timeout_s=max(120, int(client.get("TIMEOUT_MS", "10000")) // 1000 + 30),
    )
    case.observe(
        "gate",
        "cache_scale_in_check",
        timeout_s=data.procedure["analysis_timeout_s"],
        params={
            "flow": output("traffic", "flow"),
            "evidence": output("scale_in", "evidence"),
        },
    )
    case.step("teardown", "teardown")
