"""Single-run absolute gate; profiles are separate runs, never an A/B dependency."""

from cases.master_performance.actions import HANDLERS as ACTION_HANDLERS

from cases.config import output
from traffic.playback_config import normalize
from cases.master_performance.analysis import validate
from cases.master_performance.inputs import OBSERVATION_FIELDS, RULES, compile_checks
from runtime.java_flow import JAVA_FLOW_INPUT_FIELDS
from traffic.contracts import driver


def default(case):
    data = case.inputs(
        traffic=JAVA_FLOW_INPUT_FIELDS | {"kind"},
        observation=OBSERVATION_FIELDS | {"inputs", "slo"},
        checks=set(RULES) | {"engine_tps"},
    )
    inputs = data.observation["inputs"]
    if not isinstance(inputs, dict) or set(inputs) != {"engine_tps"}:
        raise ValueError("performance gate requires engine_tps input")
    flow = driver(data.traffic, "java_flow")
    client, _ = normalize(flow["client"])
    from cases.inputs import fields
    slo = fields(data.observation["slo"], {"ttft_ms", "e2e_ms", "tpot_ms"}, "parameters.observation.slo")
    c = validate(dict({k: data.observation[k] for k in OBSERVATION_FIELDS},
                      **compile_checks(case, data.checks, inputs["engine_tps"]),
                      **{"slo_"+key: value for key, value in slo.items()},
                      qps=float(client["SEND_MODE_QPS"])), inputs["engine_tps"])
    allowed = c["qps"] * (1 + c["qps_tolerance"])
    if c["min_requests"] > allowed * c["measure_s"] or c["min_goodput_rps"] > allowed:
        raise ValueError("request/goodput floors exceed the declared offered-load envelope; review thresholds after changing playback.qps")
    for binding in inputs["engine_tps"]["fields"].values():
        identity = binding["metric"]
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
