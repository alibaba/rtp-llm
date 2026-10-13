"""Strict HA program inputs; flow sequencing and measurement remain separate."""

import copy
import math

from case_api.inputs import fields
from cases.master_ha_failover.analysis import HA_METRICS
from analysis.checks import validate_comparison
from document_io import DocumentError as ScenarioError
from case_api.numeric_parameters import SOURCE_NUMBERS
from case_api.numeric_parameters import OUTPUT_DISTRIBUTION_NUMBERS
from case_api.numeric_parameters import CAPTURE_NUMBERS
from case_api.numeric_parameters import FRACTION
from case_api.numeric_parameters import POSITIVE_COUNT
from case_api.numeric_parameters import COUNT
from case_api.numeric_parameters import OFFSET
from case_api.numeric_parameters import NONNEGATIVE
from case_api.numeric_parameters import number_fields


from case_api.input_contract import InputContract, section_fields

# Each check declares its numeric leaves once. Adding a check derives both
# section admission and numeric validation, including the common sample floor.
CHECK_INPUTS = {
    **dict.fromkeys(('a_balance', 'a_route', 'a_success', 'b_balance', 'b_route',
                    'b_success', 'baseline_success', 'both_balance', 'both_success',
                    'outage_terminal'), FRACTION),
    **dict.fromkeys(('handover_errors', 'late_errors', 'outage_failures',
                    'outage_no_master', 'rolling_errors', 'unique_requests'), COUNT),
}
INPUTS = InputContract.from_fields({
    'traffic': section_fields('traffic',
        {'kind', 'source', 'duration_s', 'timeout_ms', 'replay_speed', 'loop',
         'max_concurrency', 'max_requests', 'fallback'}, {
            **SOURCE_NUMBERS, **OUTPUT_DISTRIBUTION_NUMBERS,
            **number_fields(COUNT, 'traffic.max_concurrency', 'traffic.max_requests',
                            'traffic.replay_speed', 'traffic.timeout_ms'),
            'traffic.duration_s': NONNEGATIVE,
        }),
    'procedure': {
        **dict.fromkeys(('setup_timeout_s', 'restart_timeout_s', 'finish_timeout_s'), NONNEGATIVE),
        **dict.fromkeys(('kill_a', 'kill_b', 'b_ready', 'a_ready')),
        **{name: {'wait_s': NONNEGATIVE} for name in
           ('baseline_wait', 'settle', 'survivor_wait', 'outage_wait', 'both_wait')},
    },
    'observation': {
        'windows': {'baseline': {'until': {'offset_s': OFFSET}}},
        'capture': section_fields('observation', {'capture'}, CAPTURE_NUMBERS)['capture'],
    },
    'checks': {name: {'expected': rule, 'min_samples': POSITIVE_COUNT}
               for name, rule in CHECK_INPUTS.items()},
})


def validate_client_criterion(params, *, path="client criterion"):
    """Validate selected and inactive YAML criteria without accessing live resources."""
    p = copy.deepcopy(fields(params, {"metric", "op", "expected"}, path,
        optional={"target", "route", "min_samples", "warning_profiles"}))
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
        "route_count": "route",
    }.get(p["metric"].split("/", 1)[1])
    if required and required not in p:
        raise ValueError(f"{p['metric']} requires {required}")
    if "target" in p and p["target"] not in ("A", "B"):
        raise ValueError("client target must be A or B")
    if "route" in p and p["route"] not in {"master", "fallback", "failed"}:
        raise ValueError("invalid expected route")
    unused = ({"target", "route"} & set(p)) - ({required} if required else set())
    if unused:
        raise ValueError(f"{p['metric']} does not consume {sorted(unused)}")
    return p


def read_cycle(case):
    from case_api.windows import ObservationWindow
    from case_api.windows import check_windows

    if isinstance(case.parameters.get("procedure"), dict) and "restart_mode" in case.parameters["procedure"]:
        raise ScenarioError("restart_mode is owned by flow identity, not parameters")
    data = case.bind(INPUTS)
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
               optional={"target", "route", "warning_profiles"})
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
