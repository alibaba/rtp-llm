"""Single-run absolute gate; profiles are separate runs, never an A/B dependency."""

from cases.registry import ReportView
from cases.master_performance.report import validate_view, render_view

from cases.master_performance.actions import HANDLERS as ACTION_HANDLERS

from cases.config import output
from cases.master_performance.inputs import NUMERIC_PARAMETERS
from traffic.playback_config import normalize
from cases.master_performance.inputs import validate
from cases.master_performance.inputs import OBSERVATION_FIELDS, RULES, compile_checks, observation_contract
from runtime.java_flow import JAVA_FLOW_INPUT_FIELDS
from traffic.contracts import driver, uniform_gate_flow
from cases.metric_inputs import bind_engine_metrics


def default(case):
    data = case.inputs(
        traffic=JAVA_FLOW_INPUT_FIELDS | {"kind"},
        procedure={"analysis_timeout_s"},
        observation=OBSERVATION_FIELDS | {"inputs", "windows", "capture"},
        analysis={"slo"},
        checks=set(RULES) | {"engine_tps"},
    )
    from input_contract import finite_number
    analysis_timeout = data.procedure["analysis_timeout_s"]
    if not finite_number(analysis_timeout) or analysis_timeout <= 0:
        raise ValueError("analysis_timeout_s must be finite and positive")
    inputs = data.observation["inputs"]
    if not isinstance(inputs, dict) or set(inputs) != {"engine_tps"}:
        raise ValueError("performance gate requires engine_tps input")
    flow = driver(data.traffic, "java_flow")
    client, _ = normalize(flow["client"])
    from cases.inputs import fields
    slo = fields(data.analysis["slo"], {"ttft_ms", "e2e_ms", "tpot_ms"}, "parameters.analysis.slo")
    c = validate(dict({k: data.observation[k] for k in OBSERVATION_FIELDS},
                      **observation_contract(data.observation),
                      **compile_checks(case, data.checks, inputs["engine_tps"]),
                      **{"slo_"+key: value for key, value in slo.items()},
                      qps=float(client["SEND_MODE_QPS"])), inputs["engine_tps"])
    allowed = c["qps"] * (1 + c["qps_tolerance"])
    if c["min_requests"] > allowed * c["measure_s"] or c["min_goodput_rps"] > allowed:
        raise ValueError("request/goodput floors exceed the declared offered-load envelope; review thresholds after changing playback.qps")
    bind_engine_metrics(case, inputs["engine_tps"],
        {field: "tokens/s" for field in inputs["engine_tps"]["fields"]})
    budget = c["warmup_s"] + c["measure_s"]
    uniform_gate_flow(flow, duration_s=budget + 10, trace_duration_s=int(client["DURATION_S"]))
    case.step("setup", "setup", timeout_s=180)
    case.step("traffic", "java_flow_start", params=flow, timeout_s=120)
    case.step(
        "measure",
        "performance_observe",
        params=dict(flow=output("traffic", "flow"), criteria=c, gate_input=inputs["engine_tps"],
                    observation=data.observation),
        timeout_s=budget + 30,
    )
    case.step("stop", "java_flow_stop", params={"flow": output("traffic", "flow")})
    case.step("drain", "java_flow_drain", params={"flow": output("traffic", "flow")},
              timeout_s=int(client["TIMEOUT_MS"]) / 1000 + 30)
    case.observe(
        "gate",
        "performance_finish",
        params=dict(
            flow=output("traffic", "flow"), evidence=output("measure", "evidence")
        ),
        timeout_s=analysis_timeout,
    )
    case.step("teardown", "teardown")


REPORT_VIEWS = {
    "master_performance.yaml": ReportView(validate_view, render_view),
}


def produce_gate_metrics(directory):
    from workload.gate_result import load_gate
    from cases.master_performance.metrics import produce
    from pathlib import Path
    if (Path(directory) / "performance-gate-manifest.json").is_file():
        evidence, result = load_gate(directory, "performance")
        produce(directory, evidence, result)


from cases.registry import CaseDefinition, MetricProducer, ProducerPhase
from cases.master_performance.metrics import metric_contract

CASE = CaseDefinition(
    builders={"default": default},
    numeric_parameters=NUMERIC_PARAMETERS,
    actions=tuple(ACTION_HANDLERS),
    report_views=REPORT_VIEWS,
    producers={
        "performance_requests": MetricProducer(metric_contract, produce_gate_metrics, ProducerPhase.GATE),
    },
)
