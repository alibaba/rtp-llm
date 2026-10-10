"""Strict HA program inputs; flow sequencing and measurement remain separate."""

import copy
import math

from cases.inputs import fields
from cases.master_ha_failover.analysis import HA_METRICS
from analysis.checks import validate_comparison
from scenario.loader import ScenarioError
from cases.numeric_parameters import (
    SOURCE_NUMBERS, OUTPUT_DISTRIBUTION_NUMBERS, CAPTURE_NUMBERS, FRACTION,
    POSITIVE_COUNT, COUNT, OFFSET, NONNEGATIVE,
    number_fields,
)


NUMERIC_PARAMETERS = {
    **SOURCE_NUMBERS,
    **OUTPUT_DISTRIBUTION_NUMBERS,
    **CAPTURE_NUMBERS,
    **number_fields(FRACTION,
        'checks.a_balance.expected',
        'checks.a_route.expected',
        'checks.a_success.expected',
        'checks.b_balance.expected',
        'checks.b_route.expected',
        'checks.b_success.expected',
        'checks.baseline_success.expected',
        'checks.both_balance.expected',
        'checks.both_success.expected',
        'checks.outage_terminal.expected',
    ),
    **number_fields(POSITIVE_COUNT,
        'checks.a_balance.min_samples',
        'checks.a_route.min_samples',
        'checks.a_success.min_samples',
        'checks.b_balance.min_samples',
        'checks.b_route.min_samples',
        'checks.b_success.min_samples',
        'checks.baseline_success.min_samples',
        'checks.both_balance.min_samples',
        'checks.both_success.min_samples',
        'checks.handover_errors.min_samples',
        'checks.late_errors.min_samples',
        'checks.outage_failures.min_samples',
        'checks.outage_no_master.min_samples',
        'checks.outage_terminal.min_samples',
        'checks.rolling_errors.min_samples',
        'checks.unique_requests.min_samples',
    ),
    **number_fields(COUNT,
        'checks.handover_errors.expected',
        'checks.late_errors.expected',
        'checks.outage_failures.expected',
        'checks.outage_no_master.expected',
        'checks.rolling_errors.expected',
        'checks.unique_requests.expected',
        'traffic.max_concurrency',
        'traffic.max_requests',
        'traffic.replay_speed',
        'traffic.timeout_ms',
    ),
    **number_fields(OFFSET,
        'observation.windows.baseline.until.offset_s',
    ),
    **number_fields(NONNEGATIVE,
        'procedure.baseline_wait.wait_s',
        'procedure.both_wait.wait_s',
        'procedure.finish_timeout_s',
        'procedure.outage_wait.wait_s',
        'procedure.restart_timeout_s',
        'procedure.settle.wait_s',
        'procedure.setup_timeout_s',
        'procedure.survivor_wait.wait_s',
        'traffic.duration_s',
    ),
}


def validate_client_criterion(params, *, path="client criterion"):
    """Validate selected and inactive YAML criteria without accessing live resources."""
    p = copy.deepcopy(fields(params, {"metric", "op", "expected"}, path,
        optional={"target", "route", "error_kind", "code", "min_samples", "warning_profiles"}))
    if type(p["metric"]) is not str or p["metric"] not in {"ha_gate/" + name for name in HA_METRICS}:
        raise ValueError("unknown client metric/comparison")
    if type(p["expected"]) not in (int, float) or not math.isfinite(p["expected"]):
        raise ValueError("client comparison needs finite numeric expected value")
    validate_comparison(p["op"], p["expected"])
    from flexlb_profile_data import PROFILES
    warnings = p.get("warning_profiles", [])
    if not isinstance(warnings, list) or any(type(profile) is not str or profile not in PROFILES for profile in warnings):
        raise ValueError("warning_profiles must contain registered profiles")
    if "min_samples" not in p:
        raise ValueError(path + ": min_samples is required")
    if type(p["min_samples"]) is not int or p["min_samples"] < 1:
        raise ValueError("client check must require actual samples")
    required = {
        "target_share": "target",
        "target_count": "target",
        "route_share": "route",
        "route_count": "route",
        "error_kind_count": "error_kind",
        "wrong_error_code": "code",
    }.get(p["metric"].split("/", 1)[1])
    if required and required not in p:
        raise ValueError(f"{p['metric']} requires {required}")
    if "target" in p and p["target"] not in ("A", "B"):
        raise ValueError("client target must be A or B")
    if "route" in p and p["route"] not in {"master", "fallback", "failed"}:
        raise ValueError("invalid expected route")
    if "error_kind" in p and p["error_kind"] not in {
        "none",
        "transport",
        "business",
        "deadline",
    }:
        raise ValueError("invalid expected error kind")
    if "code" in p and (type(p["code"]) is not int or p["code"] <= 0):
        raise ValueError("error code must be positive integer")
    return p


def read_cycle(case):
    from cases.windows import ObservationWindow, check_windows

    if isinstance(case.parameters.get("procedure"), dict) and "restart_mode" in case.parameters["procedure"]:
        raise ScenarioError("restart_mode is owned by flow identity, not parameters")
    data = case.inputs(
        traffic={"kind", "source", "targets", "duration_s", "timeout_ms", "replay_speed", "loop",
                 "max_concurrency", "max_requests", "fallback"},
        procedure={"setup_timeout_s", "kill_a", "kill_b", "b_ready", "a_ready",
                   "restart_timeout_s", "finish_timeout_s",
                   "baseline_wait", "settle", "survivor_wait", "outage_wait", "both_wait"},
        observation={"windows", "capture"},
        checks={"baseline_success", "b_success", "b_route", "b_balance", "outage_failures",
                "outage_no_master", "outage_terminal", "a_success", "a_route", "a_balance",
                "both_success", "both_balance", "handover_errors", "late_errors",
                "rolling_errors", "unique_requests"},
    )
    from runtime.observation import capture_limits
    capture_limits(data.observation["capture"], "parameters.observation.capture")
    for name in ("baseline_wait", "settle", "survivor_wait", "outage_wait", "both_wait"):
        fields(data.procedure[name], {"wait_s"}, "parameters.procedure." + name)
        validate_wait(data.procedure[name], path="parameters.procedure." + name)
    window_names = {"baseline", "b_only", "a_only", "both", "outage",
                    "a_handover", "post_recovery", "all_requests"}
    fields(data.observation["windows"], window_names, "parameters.observation.windows")
    windows = {name: ObservationWindow.read(spec, "parameters.observation.windows." + name,
        timestamp_stages={"baseline_end", "b_start", "b_end", "a_start", "a_end",
                          "both_start", "both_end", "outage_start", "outage_end"})
        for name, spec in data.observation["windows"].items()}
    for name, criterion in data.checks.items():
        fields(criterion, {"windows", "metric", "unit", "op", "expected", "min_samples"},
               "parameters.checks." + name,
               optional={"target", "route", "error_kind", "code", "warning_profiles"})
        check_windows(criterion["windows"], available=window_names | {"full_run"},
                      path=f"parameters.checks.{name}.windows", single=True)
        validate_client_criterion({k: v for k, v in criterion.items() if k not in {"windows", "unit"}},
                                  path="parameters.checks." + name)
        case.metric(criterion["metric"], unit=criterion["unit"])
    return data, windows


def validate_wait(params, *, path):
    p = copy.deepcopy(fields(params, {"wait_s"}, path))
    if (type(p["wait_s"]) not in (int, float) or not math.isfinite(p["wait_s"])
            or not 0 <= p["wait_s"] <= 180):
        raise ValueError("wait_s must be finite in [0,180]")
    return p
