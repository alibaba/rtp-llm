"""Compile ordered stages with typed references and explicit action contracts."""

import copy
import math
from dataclasses import dataclass
from typing import Callable

from scenario.contracts import PlanContext
from scenario.validation import fail, mapping, identifier, number
from analysis.checks import validate_comparison


def reference(value, path, outputs, expected=None):
    mapping(value, path, {"$ref"}, {"$ref"})
    ref = value["$ref"]
    if not isinstance(ref, str):
        fail(path, "$ref must be a string")
    parts = ref.split(".")
    if len(parts) != 4 or parts[0] != "stages" or parts[2] != "output":
        fail(path, "expected stages.<earlier_stage>.output.<field>")
    kind = outputs.get(parts[1], {}).get(parts[3])
    if kind is None:
        fail(path, f"unknown or forward reference {ref!r}")
    if expected is not None and kind != expected:
        fail(path, f"reference is {kind}, expected {expected}")
    return kind


def _empty_params(params, loc, outputs):
    mapping(params, loc + ".params", set())


def _request_params(params, loc, outputs):
    mapping(
        params,
        loc + ".params",
        {
            "input_len",
            "output_len",
            "count",
            "consume",
            "block_keys",
            "priority",
            "qos_level",
            "schedule_timeout_s",
            "stream_timeout_s",
            "post_issue_delay_s",
        },
    )
    for key, default in (("input_len", 2048), ("output_len", 10), ("count", 1)):
        params[key] = number(
            params.get(key, default),
            loc + ".params." + key,
            minimum=1,
            integer=True,
        )
    params.setdefault("consume", "immediate")
    if params["consume"] not in ("immediate", "deferred"):
        fail(loc, "consume must be immediate or deferred")
    if "post_issue_delay_s" in params:
        pause = number(
            params["post_issue_delay_s"], loc + ".params.post_issue_delay_s"
        )
        if pause > 2:
            fail(loc, "post_issue_delay_s cannot exceed two seconds")
    if "block_keys" in params:
        keys = params["block_keys"]
        if (
            not isinstance(keys, list)
            or not 1 <= len(keys) <= 4096
            or any(
                type(k) is not int or not -(2**63) <= k < 2**63 for k in keys
            )
        ):
            fail(loc, "block_keys must be 1..4096 explicit int64 keys")
    for key in ("priority", "qos_level"):
        if key in params and (
            type(params[key]) is not int or not -(2**31) <= params[key] < 2**31
        ):
            fail(loc, f"{key} must be an explicit int32 protocol value")
    for key in ("schedule_timeout_s", "stream_timeout_s"):
        if key in params:
            params[key] = number(
                params[key], loc + ".params." + key, minimum=0.001
            )
            if params[key] > 60:
                fail(loc, f"{key} cannot exceed 60 seconds")


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
        type(params["expected"]) is float
        and not math.isfinite(params["expected"])
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
    "request": _CoreAction({"requests": "requests", "count": "integer"}, _request_params),
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
        if not isinstance(values, list) or not values:
            fail(path, "expected nonempty stage list")
        return [self._stage(value, index, f"{path}[{index}]")
                for index, value in enumerate(values)]

    def _enter(self, action, index, loc):
        if self.torn_down:
            fail(loc, "no stage may follow teardown")
        if action == "setup":
            if self.setup_seen or index != 0:
                fail(loc, "setup must occur exactly once as the first stage")
            self.setup_seen = True
        elif not self.setup_seen:
            fail(loc, "setup must precede actions")

    def _handler_params(self, descriptor, params, loc, action):
        if descriptor.owners and self.case not in descriptor.owners:
            fail(loc + ".action", f"action {action!r} belongs to cases {sorted(descriptor.owners)!r}")
        params = descriptor.validate(params, PlanContext(
            loc + ".params", dict(self.outputs), copy.deepcopy(self.environment), self.profiles,
        ))
        if not isinstance(params, dict):
            fail(loc, "adapter validate must return a mapping")
        return params

    def _stage(self, value, index, loc):
        mapping(value, loc, {"id", "action", "timeout_s", "params", "purpose"}, {"id", "action"})
        if value.get("purpose", "operation") not in {"operation", "observation"}:
            fail(loc + ".purpose", "expected operation or observation")
        sid = identifier(value["id"], loc + ".id")
        if sid in self.outputs:
            fail(loc, f"duplicate stage id {sid}")
        action = value["action"]
        if not isinstance(action, str) or action not in (set(OUTPUTS) | set(self.handlers)):
            fail(loc + ".action", f"unsupported action {action!r}")
        self._enter(action, index, loc)
        timeout = number(value.get("timeout_s", self.default_timeout), loc + ".timeout_s", minimum=0.001)
        params = copy.deepcopy(value.get("params", {}))
        descriptor = self.handlers.get(action)
        if descriptor is not None:
            params = self._handler_params(descriptor, params, loc, action)
        else:
            _CORE_ACTIONS[action].validate(params, loc, self.outputs)
            if action == "teardown":
                self.torn_down = True
        result = dict(id=sid, action=action, timeout_s=timeout, params=params)
        if "purpose" in value:
            result["purpose"] = value["purpose"]
        self.outputs[sid] = descriptor.outputs if descriptor is not None else OUTPUTS[action]
        if descriptor is not None and descriptor.next_environment is not None:
            self.environment = copy.deepcopy(descriptor.next_environment(params))
        return result


def stages(values, path, default_timeout, handlers, env=None, profiles=(), case=None):
    return _StageCompiler(default_timeout, handlers, env, profiles, case).compile(values, path)
