"""Single-run absolute gate; profiles are separate runs, never an A/B dependency."""

from cases.master_performance.actions import HANDLERS as ACTION_HANDLERS

from cases.config import output
from traffic.playback_config import normalize
from cases.master_performance.analysis import validate
from cases.master_performance.inputs import OBSERVATION_FIELDS, CHECK_FIELDS
from runtime.java_flow import JAVA_FLOW_INPUT_FIELDS


def default(case):
    data = case.inputs(
        traffic=JAVA_FLOW_INPUT_FIELDS,
        observation=OBSERVATION_FIELDS | {"inputs"},
        checks=CHECK_FIELDS,
        optional={"checks": {"engine_tps", "engine_tps_by_profile"}},
    )
    inputs = data.observation["inputs"]
    if not isinstance(inputs, dict) or set(inputs) != {"engine_tps"}:
        raise ValueError("performance gate requires engine_tps input")
    flow = data.traffic
    client, _ = normalize(flow["client"])
    c = validate(dict({k: v for k, v in data.observation.items() if k != "inputs"}, **data.checks,
                      qps=float(client["SEND_MODE_QPS"])), inputs["engine_tps"])
    for identity in inputs["engine_tps"]["metric_roles"]:
        case.metric(identity, unit="tokens/s",
                    labels=("role", "engine_name", "engine_incarnation"), mode="scrape")
    if client.get("SEND_MODE") != "uniform":
        raise ValueError("steady performance gate requires uniform QPS")
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
