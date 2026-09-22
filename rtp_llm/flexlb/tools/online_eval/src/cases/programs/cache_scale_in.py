"""Warm once, shrink in one step, keep the Java sender running, adjudicate offline."""

from cases.config import output
from traffic.playback import normalize


def step(case):
    flow = case.value("flow")
    gate = case.value("gate")
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
    if "intermediate_p" in gate:
        budget += gate["intermediate_hold_s"] + gate["topology_timeout_s"]
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
        params=dict(gate, flow=output("traffic", "flow")),
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
        params={
            "flow": output("traffic", "flow"),
            "evidence": output("scale_in", "evidence"),
        },
    )
    case.step("teardown", "teardown")
