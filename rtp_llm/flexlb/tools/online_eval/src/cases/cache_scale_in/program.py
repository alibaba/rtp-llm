"""Warm once, shrink in one step, keep the Java sender running, adjudicate offline."""

from cases.cache_scale_in.actions import HANDLERS as ACTION_HANDLERS

from cases.config import output
from traffic.playback_config import normalize
from cases.cache_scale_in.comparison import validate_policy
from cases.cache_scale_in.inputs import engine_counters, ENGINE_COUNTER_UNITS

ANALYSIS_POLICY_VALIDATOR = validate_policy


def default(case):
    flow = case.value("flow")
    gate = case.value("gate")
    inputs = case.value("gate_inputs")
    if not isinstance(inputs, dict) or set(inputs) != {"engine_counters"}:
        raise ValueError("cache gate requires engine_counters input")
    engine_counters(inputs["engine_counters"])
    for field, identity in inputs["engine_counters"]["fields"].items():
        if field not in ENGINE_COUNTER_UNITS:
            raise ValueError("unknown engine gate field: " + field)
        case.metric(identity, unit=ENGINE_COUNTER_UNITS[field],
                    labels=("role", "engine_name", "engine_incarnation"), mode="scrape")
    client, playback = normalize(flow["client"])
    if (
        client.get("LOOP") != "false"
        or client.get("SEND_MODE") != "uniform"
    ):
        raise ValueError("scale-in requires a nonlooping uniform Java workload")
    if float(client["SEND_MODE_QPS"]) != gate["qps"]:
        raise ValueError("client and gate QPS must agree")
    budget = (
        gate["warmup_timeout_s"] + gate["topology_timeout_s"] + gate["observe_s"] + 10
    )
    if {"intermediate_p", "intermediate_hold_s"} & gate.keys() or gate.get("removal_mode") != "graceful":
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
        timeout_s=case.value("analysis_timeout_s"),
        params={
            "flow": output("traffic", "flow"),
            "evidence": output("scale_in", "evidence"),
        },
    )
    case.step("teardown", "teardown")
