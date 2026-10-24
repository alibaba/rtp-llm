"""Compile declared comparisons into case-owned measurement contracts."""

from analysis.checks import validate_comparison
from scenario.inputs import fields
from scenario.windows import check_windows


def metric_criterion(case, spec, *, metric, unit, windows, op, path, profiles=False, dimensions=None):
    binding = dict(metric=metric, unit=unit, windows=list(windows), op=op)
    fields(spec, {"expected"}, path, optional=set(binding) | ({"expected_by_profile"} if profiles else set()))
    declared = {**binding, **spec}
    check_windows(declared["windows"], available=windows, path=path + ".windows")
    if (
        declared["metric"] != metric
        or declared["unit"] != unit
        or set(declared["windows"]) != set(windows)
        or declared["op"] != op
    ):
        raise ValueError(path + ": metric, unit, windows or comparison does not match the measurement contract")
    validate_comparison(op, spec["expected"])
    if type(spec["expected"]) not in (int, float):
        raise ValueError(path + ": metric threshold must be numeric")
    case.metric(metric, unit=unit)
    overrides = spec.get("expected_by_profile", {})
    if not isinstance(overrides, dict):
        raise ValueError(path + ": expected_by_profile must be a mapping")
    allowed = frozenset(case.profiles if dimensions is None else dimensions)
    for profile, expected in overrides.items():
        if type(profile) is not str or profile not in allowed:
            raise ValueError(path + ": expected_by_profile requires registered profiles")
        validate_comparison(op, expected)
        if type(expected) not in (int, float):
            raise ValueError(path + ": profile threshold must be numeric")
    return spec["expected"], overrides


def compile_metric_checks(case, checks, rules, *, namespace, path="parameters.checks"):
    """Project a fixed measurement table; case-specific policies stay with the case."""
    result = {}
    for name, (key, metric, op, unit, windows) in rules.items():
        result[key], _ = metric_criterion(
            case,
            checks[name],
            metric=namespace + "/" + metric,
            unit=unit,
            windows=windows,
            op=op,
            path=path + "." + name,
        )
    return result


from dataclasses import dataclass

from scenario.numeric_parameters import NONNEGATIVE


@dataclass(frozen=True)
class CheckSpec:
    key: str
    metric: str
    op: str
    unit: str
    windows: tuple
    number: object = NONNEGATIVE

    def __post_init__(self):
        from scenario.numeric_parameters import parameter_rules

        parameter_rules({"checks.threshold.expected": self.number})
        if self.op not in {"eq", "ne", "gt", "ge", "lt", "le"}:
            raise ValueError("invalid check comparison")
        if not self.key or not self.metric or not self.unit or not self.windows:
            raise ValueError("check requires key, metric, unit and windows")
        object.__setattr__(self, "windows", tuple(self.windows))

    @property
    def rule(self):
        return self.key, self.metric, self.op, self.unit, self.windows


class CheckSet:
    """Declare threshold type and measurement binding together, once per check."""

    def __init__(self, namespace, specs):
        from types import MappingProxyType

        if not namespace or any(not isinstance(value, CheckSpec) for value in specs.values()):
            raise ValueError("invalid check declarations")
        self.namespace = namespace
        self.specs = MappingProxyType(dict(specs))

    @property
    def rules(self):
        return {name: spec.rule for name, spec in self.specs.items()}

    @property
    def numeric_parameters(self):
        return {"checks." + name + ".expected": spec.number for name, spec in self.specs.items()}

    @property
    def input_fields(self):
        return {name: {"expected": spec.number} for name, spec in self.specs.items()}

    @property
    def criteria_fields(self):
        return {spec.key: spec.number for spec in self.specs.values()}

    @property
    def fields(self):
        return frozenset(spec.key for spec in self.specs.values())

    def compile(self, case, checks):
        return compile_metric_checks(case, checks, self.rules, namespace=self.namespace)
