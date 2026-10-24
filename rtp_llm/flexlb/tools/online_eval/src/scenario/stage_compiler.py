"""Compile ordered stages with typed references and explicit action contracts."""

import copy
import math
from dataclasses import dataclass
from typing import Callable

from analysis.checks import validate_comparison
from document_io import DocumentError as ScenarioError
from scenario.contracts import PlanContext
from scenario.plan import StagePlan
from scenario.program import StageDefinition
from scenario.references import enter_stage, reference, validate_references
from scenario.request_parameters import normalize_request_parameters
from scenario.stage_parameters import decode_parameters
from scenario.validation import fail, identifier, mapping, number


def _empty_params(params, loc, outputs):
    mapping(params, loc + ".params", set())


def _request_reference(params, loc, outputs):
    mapping(params, loc + ".params", {"requests"}, {"requests"})
    reference(params["requests"], loc + ".params.requests", outputs, "requests")


def _check_params(params, loc, outputs):
    mapping(
        params,
        loc + ".params",
        {"actual", "op", "expected"},
        {"actual", "op", "expected"},
    )
    kind = reference(params["actual"], loc + ".params.actual", outputs)
    expected_types = {
        "boolean": (bool,),
        "integer": (int,),
        "number": (int, float),
        "string": (str,),
    }.get(kind, ())
    if type(params["expected"]) not in expected_types or (
        type(params["expected"]) is float and not math.isfinite(params["expected"])
    ):
        fail(
            loc,
            "checks compare scalar values with matching types, not live handles",
        )
    try:
        validate_comparison(params["op"], params["expected"])
    except ValueError:
        fail(loc, "invalid comparison for output type")


@dataclass(frozen=True)
class _CoreAction:
    outputs: dict
    validate: Callable


_CORE_ACTIONS = {
    "setup": _CoreAction({"environment": "environment"}, _empty_params),
    "request": _CoreAction(
        {"requests": "requests", "count": "integer"},
        lambda params, loc, outputs: normalize_request_parameters(params, loc),
    ),
    "wait": _CoreAction({"completed": "boolean", "error_count": "integer"}, _request_reference),
    "cancel": _CoreAction({"issued": "integer"}, _request_reference),
    "check": _CoreAction({"passed": "boolean"}, _check_params),
    "teardown": _CoreAction({"clean": "boolean"}, _empty_params),
}
OUTPUTS = {name: action.outputs for name, action in _CORE_ACTIONS.items()}


class _StageCompiler:
    """State belongs to one stage sequence; handlers receive isolated environment data."""

    def __init__(self, default_timeout, handlers, env, profiles, case):
        self.default_timeout = default_timeout
        self.handlers = handlers
        self.environment = copy.deepcopy(env or {})
        self.profiles = tuple(profiles)
        self.case = case
        self.outputs = {}
        self.setup_seen = False
        self.torn_down = False

    def compile(self, values, path):
        if not isinstance(values, tuple) or not values:
            fail(path, "expected nonempty typed stage sequence")
        return tuple(self._stage(value, index, f"{path}[{index}]") for index, value in enumerate(values))

    def _handler_params(self, descriptor, params, loc, action):
        if descriptor.owners and self.case not in descriptor.owners:
            fail(loc + ".action", f"action {action!r} belongs to cases {sorted(descriptor.owners)!r}")
        try:
            params = descriptor.validate(
                params,
                PlanContext(
                    loc + ".params",
                    dict(self.outputs),
                    copy.deepcopy(self.environment),
                    self.profiles,
                    self.case,
                ),
            )
        except ScenarioError:
            raise
        except ValueError as exc:
            detail = str(exc)
            path = loc + ".params"
            raise ScenarioError(detail if detail.startswith(path) else f"{path}: {detail}") from exc
        if not isinstance(params, dict):
            fail(loc, "adapter validate must return a mapping")
        return params

    def _stage(self, value, index, loc):
        if not isinstance(value, StageDefinition):
            fail(loc, "expected StageDefinition")
        if (value.purpose if value.purpose is not None else "operation") not in {"operation", "observation"}:
            fail(loc + ".purpose", "expected operation or observation")
        sid = identifier(value.id, loc + ".id")
        if sid in self.outputs:
            fail(loc, f"duplicate stage id {sid}")
        action = value.action
        if not isinstance(action, str) or action not in (set(OUTPUTS) | set(self.handlers)):
            fail(loc + ".action", f"unsupported action {action!r}")
        self.setup_seen = enter_stage(action, index, self.setup_seen, self.torn_down, loc)
        timeout = number(
            value.timeout_s if value.timeout_s is not None else self.default_timeout, loc + ".timeout_s", minimum=0.001
        )
        params = copy.deepcopy(value.params)
        descriptor = self.handlers.get(action)
        if descriptor is not None:
            params = self._handler_params(descriptor, params, loc, action)
        else:
            _CORE_ACTIONS[action].validate(params, loc, self.outputs)
            if action == "teardown":
                self.torn_down = True
        validate_references(params, loc + ".params", self.outputs)
        self.outputs[sid] = descriptor.outputs if descriptor is not None else OUTPUTS[action]
        if descriptor is not None and descriptor.next_environment is not None:
            self.environment = copy.deepcopy(descriptor.next_environment(params))
        checks = descriptor.checks if descriptor else ({"comparison"} if action == "check" else set())
        if callable(checks):
            checks = checks(params)
        return StagePlan(sid, action, timeout, decode_parameters(action, params), tuple(sorted(checks)), value.purpose)


def stages(values, path, default_timeout, handlers, env=None, profiles=(), case=None):
    return _StageCompiler(default_timeout, handlers, env, profiles, case).compile(values, path)
