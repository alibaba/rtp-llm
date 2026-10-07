"""Single-run absolute gate; profiles are separate runs, never an A/B dependency."""

from cases.config import output
from traffic.playback_config import normalize
from workload.performance_gate import validate


def steady(case):
    c = validate(case.value("criteria"))
    flow = case.value("flow")
    client, _ = normalize(flow["client"])
    if (
        client.get("SEND_MODE") != "uniform"
        or float(client["SEND_MODE_QPS"]) != c["qps"]
    ):
        raise ValueError("steady performance gate requires matching uniform QPS")
    budget = c["warmup_s"] + c["measure_s"]
    if int(client["DURATION_S"]) < budget + 10:
        raise ValueError("flow must cover warmup, measurement and startup margin")
    if (
        client.get("LOOP") != "false"
        or flow["source"]["parameters"]["count"] < int(client["DURATION_S"]) * c["qps"]
    ):
        raise ValueError("performance gate requires sufficient nonlooping input")
    case.step("setup", "setup", timeout_s=180)
    case.step("traffic", "java_flow_start", params=flow, timeout_s=120)
    case.step(
        "measure",
        "performance_observe",
        params=dict(flow=output("traffic", "flow"), criteria=c),
        timeout_s=budget + 30,
    )
    case.observe(
        "gate",
        "performance_finish",
        params=dict(
            flow=output("traffic", "flow"), evidence=output("measure", "evidence")
        ),
        timeout_s=int(client["TIMEOUT_MS"]) / 1000 + 45,
    )
    case.step("teardown", "teardown")
