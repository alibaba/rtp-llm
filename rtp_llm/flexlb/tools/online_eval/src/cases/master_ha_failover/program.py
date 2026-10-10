"""One continuous real-trace replay across a dual-master restart cycle."""

from cases.registry import ReportView
from cases.master_ha_failover.report import validate_view, write_report

from cases.master_ha_failover.actions import HANDLERS as ACTION_HANDLERS

from cases.config import output
from cases.master_ha_failover.inputs import NUMERIC_PARAMETERS
from cases.master_ha_failover.inputs import read_cycle
from scenario.loader import ScenarioError
from traffic.contracts import driver


FLOW_PROGRAMS = ("non_rolling",)


def default(case):
    _cycle(case, "rolling")


def non_rolling(case):
    _cycle(case, "non_rolling")


def _cycle(case, restart_mode):
    data, windows = read_cycle(case)
    case.step("setup", "setup", timeout_s=data.procedure["setup_timeout_s"])
    case.step("flow", "master_client_start", params=dict(driver(data.traffic, "ha_replay"), capture=data.observation["capture"]))

    # The producer keeps running while A and B are killed and restarted in order.
    case.step("baseline_end", "master_mark", params=dict(data.procedure["baseline_wait"], event="baseline_end"))
    case.step("kill_a", "master_fault", params=data.procedure["kill_a"])
    case.step("b_ready", "master_ready", params=data.procedure["b_ready"])
    case.step("b_start", "master_mark", params=dict(data.procedure["settle"], event="b_start"))
    case.step("b_end", "master_mark", params=dict(data.procedure["survivor_wait"], event="b_end"))
    if restart_mode == "non_rolling":
        case.step("kill_b", "master_fault", params=data.procedure["kill_b"])
        case.step("outage_start", "master_mark", params=dict(data.procedure["settle"], event="outage_start"))
        case.step("outage_end", "master_mark", params=dict(data.procedure["outage_wait"], event="outage_end"))
    case.step("restart_a", "master_restore", timeout_s=data.procedure["restart_timeout_s"],
              params={"fault": output("kill_a", "fault")})
    case.step("a_ready", "master_ready", params=data.procedure["a_ready"])
    if restart_mode == "rolling":
        case.step("kill_b", "master_fault", params=data.procedure["kill_b"])
    case.step("a_start", "master_mark", params=dict(data.procedure["settle"], event="a_start"))
    case.step("a_end", "master_mark", params=dict(data.procedure["survivor_wait"], event="a_end"))
    case.step("restart_b", "master_restore", timeout_s=data.procedure["restart_timeout_s"],
              params={"fault": output("kill_b", "fault")})
    case.step("a_ready_final", "master_ready", params=data.procedure["a_ready"])
    case.step("b_ready_final", "master_ready", params=data.procedure["b_ready"])
    case.step("both_start", "master_mark", params=dict(data.procedure["settle"], event="both_start"))
    case.step("both_end", "master_mark", params=dict(data.procedure["both_wait"], event="both_end"))
    case.step("finish", "master_client_finish", timeout_s=data.procedure["finish_timeout_s"],
              params={"client": output("flow", "client"), "stop_sending": True})

    selected_windows = ["baseline", "b_only", "a_only", "both"]
    if restart_mode == "non_rolling":
        selected_windows.append("outage")
    else:
        selected_windows.extend(["a_handover", "post_recovery", "all_requests"])
    rows = {"full_run": output("finish", "rows")}
    for name in selected_windows:
        case.step(name, "master_client_window", params={"rows": output("finish", "rows"), **windows[name].boundaries})
        rows[name] = output(name, "rows")

    checks = ["baseline_success", "b_success", "b_route", "b_balance", "a_success",
              "a_route", "a_balance", "both_success", "both_balance"]
    if restart_mode == "non_rolling":
        checks.extend(["outage_failures", "outage_no_master", "outage_terminal"])
    else:
        checks.extend(["handover_errors", "late_errors", "rolling_errors"])
    checks.append("unique_requests")
    for name in checks:
        params = dict(data.checks[name])
        window, = params.pop("windows")
        params.pop("unit")
        if window not in rows:
            raise ScenarioError(f"parameters.checks.{name}: window unavailable in {restart_mode}")
        params["rows"] = rows[window]
        case.observe(name, "master_client_check", params=params)
    case.step("clean_a", "master_inflight_clean", params={"target": "A"})
    case.step("clean_b", "master_inflight_clean", params={"target": "B"})
    case.step("cleanup", "teardown")


REPORT_VIEWS = {
    "master_ha_failover.yaml": ReportView(validate_view, write_report),
}


def produce_gate_metrics(directory):
    from cases.master_ha_failover.metrics import produce_gates
    produce_gates(directory)


from cases.registry import CaseDefinition, MetricProducer, ProducerPhase
from cases.master_ha_failover.metrics import metric_contract
from cases.registry import EvidenceSource
from cases.master_ha_failover.metrics import produce as produce_evidence
from cases.master_ha_failover.observation import master_adapters, STATE_FIELDS

CASE = CaseDefinition(
    builders={"default": default, "non_rolling": non_rolling},
    numeric_parameters=NUMERIC_PARAMETERS,
    actions=tuple(ACTION_HANDLERS),
    report_views=REPORT_VIEWS,
    producers={
        "ha_gates": MetricProducer(metric_contract, produce_gate_metrics, ProducerPhase.GATE),
        "ha_evidence": MetricProducer(metric_contract, produce_evidence, ProducerPhase.FINALIZE),
    },
    sources={"master_inflight": EvidenceSource(master_adapters, frozenset(STATE_FIELDS),
               "master", frozenset({"http_up"}))},
)
