"""Compile declared comparisons into case-owned measurement contracts."""

from analysis.checks import validate_comparison
from cases.inputs import fields


def metric_criterion(case, spec, *, metric, unit, window, op, path,
                     profiles=False):
    fields(spec, {"metric", "unit", "window", "op", "expected"}, path,
           optional={"expected_by_profile"} if profiles else ())
    if (spec["metric"] != metric or spec["unit"] != unit
            or spec["window"] != window or spec["op"] != op):
        raise ValueError(path + ": metric, unit, window or comparison does not match the measurement contract")
    validate_comparison(spec["op"], spec["expected"])
    if type(spec["expected"]) not in (int, float):
        raise ValueError(path + ": metric threshold must be numeric")
    case.metric(metric, unit=unit)
    overrides = spec.get("expected_by_profile", {})
    if not isinstance(overrides, dict):
        raise ValueError(path + ": expected_by_profile must be a mapping")
    from flexlb_profile_data import PROFILES
    for profile, expected in overrides.items():
        if type(profile) is not str or profile not in PROFILES:
            raise ValueError(path + ": expected_by_profile requires registered profiles")
        validate_comparison(op, expected)
        if type(expected) not in (int, float):
            raise ValueError(path + ": profile threshold must be numeric")
    return spec["expected"], overrides
