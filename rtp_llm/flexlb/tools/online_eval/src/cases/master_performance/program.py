"""Single-run absolute gate; profiles are separate runs, never an A/B dependency."""

from cases.master_performance.actions import HANDLERS as ACTION_HANDLERS

from cases.config import output
from traffic.playback_config import normalize
from cases.master_performance.analysis import validate


def default(case):
    inputs = case.value("gate_inputs")
    if not isinstance(inputs, dict) or set(inputs) != {"engine_tps"}:
        raise ValueError("performance gate requires engine_tps input")
    c = validate(case.value("criteria"), inputs["engine_tps"])
    for identity in inputs["engine_tps"]["metric_roles"]:
        case.metric(identity, unit="tokens/s",
                    labels=("role", "engine_name", "engine_incarnation"), mode="scrape")
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
        params=dict(flow=output("traffic", "flow"), criteria=c, gate_input=inputs["engine_tps"]),
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


def REPORT_FINALIZER(directory):
    from cases.master_performance.report import refresh_report

    refresh_report(directory)
