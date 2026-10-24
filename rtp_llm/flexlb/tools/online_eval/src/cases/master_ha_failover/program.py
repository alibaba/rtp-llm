"""One continuous real-trace replay across a dual-master restart cycle."""

from cases.master_ha_failover.actions import HANDLERS as ACTION_HANDLERS
from cases.master_ha_failover.inputs import INPUTS, read_cycle
from cases.master_ha_failover.report import validate_view, write_report
from document_io import DocumentError as ScenarioError
from scenario.builder import output
from scenario.registry import ReportView
from traffic.contracts import driver


def rolling(case):
    _cycle(case, "rolling")


def non_rolling(case):
    _cycle(case, "non_rolling")


def _cycle(case, restart_mode):
    data, windows = read_cycle(case)
    case.step("setup", "setup", timeout_s=data["procedure"]["setup_timeout_s"])
    elected = case.env.master_layout == "dual_zookeeper"
    if elected:
        case.step("baseline_leadership", "master_leadership", params={"targets": ["A", "B"]})
    case.step("flow", "master_client_start", params=dict(driver(data["traffic"], "ha_replay"), capture=data["observation"]["capture"]))

    # The producer keeps running while A and B are killed and restarted in order.
    period(case, "baseline", data, "baseline_wait", baseline=True)
    case.step("kill_a", "master_fault", params=data["procedure"]["kill_a"])
    case.step("b_ready", "master_ready", params=data["procedure"]["b_ready"])
    if elected:
        case.step("b_leadership", "master_leadership", params={"targets": ["B"], "leader": "B"})
    period(case, "b", data, "survivor_wait")
    if restart_mode == "non_rolling":
        case.step("kill_b", "master_fault", params=data["procedure"]["kill_b"])
        period(case, "outage", data, "outage_wait")
    case.step("restart_a", "master_restore", timeout_s=data["procedure"]["restart_timeout_s"], params={"fault": output("kill_a", "fault")})
    case.step("a_ready", "master_ready", params=data["procedure"]["a_ready"])
    if elected:
        case.step(
            "rejoin_leadership",
            "master_leadership",
            params={"targets": ["A", "B"], "leader": "B"} if restart_mode == "rolling" else {"targets": ["A"], "leader": "A"},
        )
    if restart_mode == "rolling":
        case.step("kill_b", "master_fault", params=data["procedure"]["kill_b"])
    if elected:
        case.step("a_leadership", "master_leadership", params={"targets": ["A"], "leader": "A"})
    period(case, "a", data, "survivor_wait")
    case.step("restart_b", "master_restore", timeout_s=data["procedure"]["restart_timeout_s"], params={"fault": output("kill_b", "fault")})
    case.step("a_ready_final", "master_ready", params=data["procedure"]["a_ready"])
    case.step("b_ready_final", "master_ready", params=data["procedure"]["b_ready"])
    if elected:
        case.step("final_leadership", "master_leadership", params={"targets": ["A", "B"], "leader": "A"})
    period(case, "both", data, "both_wait")
    case.step(
        "finish",
        "master_client_finish",
        timeout_s=data["procedure"]["finish_timeout_s"],
        params={"client": output("flow", "client"), "stop_sending": True},
    )

    selected_windows = ["baseline", "b_only", "a_only", "both"]
    if restart_mode == "non_rolling":
        selected_windows.append("outage")
    else:
        selected_windows.extend(["a_handover", "post_recovery", "all_requests"])
    selected_checks = [
        "baseline_success",
        "b_success",
        "b_route",
        "b_balance",
        "a_success",
        "a_route",
        "a_balance",
        "both_success",
        "both_balance",
    ]
    selected_checks.extend(
        ["outage_failures", "outage_no_master", "outage_terminal"]
        if restart_mode == "non_rolling"
        else ["handover_errors", "late_errors", "rolling_errors"]
    )
    selected_checks.append("unique_requests")
    boundaries = {}
    for name in selected_windows:
        boundaries[name] = {}
        for key, value in windows[name].boundaries.items():
            if key in {"from", "until"}:
                event = value["$ref"].split(".")[1]
                phase, side = event.rsplit("_", 1)
                boundaries[name][key] = output(phase + "_period", "end" if side == "end" else "start")
            else:
                boundaries[name][key] = value
    criteria = {}
    for name in selected_checks:
        criterion = dict(data["checks"][name])
        (window,) = criterion.pop("windows")
        criterion.pop("unit")
        if window not in {*selected_windows, "full_run"}:
            raise ScenarioError(f"parameters.checks.{name}: window unavailable in {restart_mode}")
        criteria[name] = dict(criterion, window=window)
    case.observe(
        "analysis",
        "master_client_analyze",
        params=dict(rows=output("finish", "rows"), windows=boundaries, checks=criteria, operation_timeout_s=case.stage_timeout_s),
        timeout_s=(len(selected_windows) + len(criteria)) * case.stage_timeout_s,
    )
    case.step("clean_a", "master_inflight_clean", params={"target": "A"})
    case.step("clean_b", "master_inflight_clean", params={"target": "B"})
    case.step("cleanup", "teardown")


def period(case, name, data, duration, *, baseline=False):
    case.step(
        name + "_period",
        "master_period",
        params=dict(
            name=name,
            settle_s=0 if baseline else data["procedure"]["settle"]["wait_s"],
            duration_s=data["procedure"][duration]["wait_s"],
            operation_timeout_s=case.stage_timeout_s,
        ),
        timeout_s=(1 if baseline else 2) * case.stage_timeout_s,
    )


REPORT_VIEWS = {"master_ha_failover.yaml": ReportView(validate_view, write_report)}


from cases.master_ha_failover import metrics
from cases.master_ha_failover.observation import STATE_FIELDS, master_adapters
from scenario.contracts import ExecutionRequirements
from scenario.registry import CaseDefinition, EvidenceSource

CASE = CaseDefinition(
    execution_requirements=ExecutionRequirements(request_evidence=True, monitoring=True),
    builders={"default": rolling, "non_rolling": non_rolling},
    input_schema=INPUTS,
    actions=tuple(ACTION_HANDLERS),
    report_views=REPORT_VIEWS,
    metrics=metrics,
    sources={"master_inflight": EvidenceSource(master_adapters, frozenset(STATE_FIELDS), "master", frozenset({"http_up"}))},
)
