"""Compile explicit instances and typed stage references without process imports."""

import copy
import json
import math
import re

from flexlb_cfg import OMIT, PROFILE_CAPS, PROFILES, ConfigOverride, render_env

from .contracts import PlanContext
from .loader import ScenarioError

CATEGORIES = {
    "cancel",
    "status",
    "kv",
    "balance",
    "elastic",
    "engine_fault",
    "master",
    "admission",
    "priority",
}
ID = re.compile(r"[a-z][a-z0-9_]*\Z")
# A deliberately bounded first compilation vocabulary. Other actions require an
# adapter and result contract before their names can be accepted by compilation.
OUTPUTS = {
    "setup": {"environment": "environment"},
    "request": {"requests": "requests", "count": "integer"},
    "wait": {"completed": "boolean", "error_count": "integer"},
    "cancel": {"issued": "integer"},
    "check": {"passed": "boolean"},
    "teardown": {"clean": "boolean"},
}
INTEGER_OVERRIDES = {
    "default_priority",
    "max_requests",
    "max_collection_wait_ms",
    "max_predicted_execution_ms",
    "queue_timeout_ms",
    "max_outstanding",
    "stale_inflight_ms",
    "delivered_not_accepted_timeout_ms",
    "max_delivered_not_accepted",
    "max_waiting_requests_per_prefill_worker",
    "max_inflight_batches",
    "enqueue_rpc_timeout_ms",
    "max_inflight_requests_per_worker",
    "status_rpc_ms",
    "decode_max_engine_requests",
}


def fail(path, message):
    raise ScenarioError(f"{path}: {message}")


def mapping(value, path, allowed, required=()):
    if not isinstance(value, dict):
        fail(path, "expected mapping")
    if any(not isinstance(key, str) for key in value):
        fail(path, "mapping keys must be strings")
    extra, missing = set(value) - set(allowed), set(required) - set(value)
    if extra or missing:
        fail(path, f"unknown fields {sorted(extra)}, missing fields {sorted(missing)}")
    return value


def identifier(value, path):
    if not isinstance(value, str) or not ID.fullmatch(value):
        fail(path, "expected lower-case identifier")
    return value


def number(value, path, minimum=0, integer=False):
    if (
        type(value) not in ((int,) if integer else (int, float))
        or value < minimum
        or (isinstance(value, float) and not math.isfinite(value))
    ):
        fail(path, f"expected {'integer' if integer else 'number'} >= {minimum}")
    return value


def names(value, path, vocabulary):
    if not isinstance(value, list) or any(not isinstance(x, str) for x in value):
        fail(path, "expected string list")
    if len(set(value)) != len(value) or set(value) - set(vocabulary):
        fail(path, f"duplicate or unknown values: {value}")
    return value


def environment(value, path, profile):
    value = mapping(
        value,
        path,
        {
            "backend",
            "n_prefill",
            "n_decode",
            "prefill_cache_blocks",
            "decode_cache_blocks",
            "config_overrides",
        },
    )
    if value.get("backend", "java_mock") != "java_mock":
        fail(
            path + ".backend",
            "only java_mock is implemented; GPU/multi-host/TP/DP layouts need separate capabilities",
        )
    result = {"backend": "java_mock", "n_prefill": 2, "n_decode": 4}
    for key in ("n_prefill", "n_decode", "prefill_cache_blocks", "decode_cache_blocks"):
        if key in value:
            result[key] = number(value[key], path + "." + key, minimum=1, integer=True)
    overrides = mapping(
        value.get("config_overrides", {}),
        path + ".config_overrides",
        INTEGER_OVERRIDES | {"ordering"},
    )
    kwargs = {}
    for key, val in overrides.items():
        field = path + ".config_overrides." + key
        if key == "ordering":
            if val not in ("fifo", "priority"):
                fail(field, "expected fifo or priority")
        elif isinstance(val, dict):
            if val != {"omit": True} or type(val.get("omit")) is not bool:
                fail(field, "expected exact {omit: true}")
            val = OMIT
        else:
            number(val, field, integer=True)
        kwargs[key] = val
    try:
        result["resolved_config"] = json.loads(
            render_env(profile, ConfigOverride(**kwargs))
        )
    except (TypeError, ValueError) as exc:
        fail(path + ".config_overrides", str(exc))
    result["config_overrides"] = copy.deepcopy(overrides)
    return result


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


def stages(values, path, default_timeout, handlers):
    if not isinstance(values, list) or not values:
        fail(path, "expected nonempty stage list")
    outputs, compiled = {}, []
    setup_seen, torn_down = False, False
    for i, value in enumerate(values):
        loc = f"{path}[{i}]"
        mapping(value, loc, {"id", "action", "timeout_s", "params"}, {"id", "action"})
        sid = identifier(value["id"], loc + ".id")
        if sid in outputs:
            fail(loc, f"duplicate stage id {sid}")
        action = value["action"]
        if not isinstance(action, str) or action not in (set(OUTPUTS) | set(handlers)):
            fail(loc + ".action", f"unsupported action {action!r}")
        if torn_down:
            fail(loc, "no stage may follow teardown")
        if action == "setup":
            if setup_seen or i != 0:
                fail(loc, "setup must occur exactly once as the first stage")
            setup_seen = True
        elif not setup_seen:
            fail(loc, "setup must precede actions")
        timeout = number(
            value.get("timeout_s", default_timeout), loc + ".timeout_s", minimum=0.001
        )
        params = copy.deepcopy(value.get("params", {}))
        if action in handlers:
            descriptor = handlers[action]
            params = descriptor.validate(
                params, PlanContext(loc + ".params", dict(outputs))
            )
            if not isinstance(params, dict):
                fail(loc, "adapter validate must return a mapping")
        elif action in ("setup", "teardown"):
            mapping(params, loc + ".params", set())
            torn_down = action == "teardown"
        elif action == "request":
            mapping(
                params, loc + ".params", {"input_len", "output_len", "count", "consume"}
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
        elif action in ("wait", "cancel"):
            mapping(params, loc + ".params", {"requests"}, {"requests"})
            reference(params["requests"], loc + ".params.requests", outputs, "requests")
        elif action == "check":
            mapping(
                params,
                loc + ".params",
                {"actual", "op", "expected"},
                {"actual", "op", "expected"},
            )
            kind = reference(params["actual"], loc + ".params.actual", outputs)
            expected_type = {"boolean": bool, "integer": int}.get(kind)
            if expected_type is None or type(params["expected"]) is not expected_type:
                fail(
                    loc,
                    "checks compare scalar values with matching types, not live handles",
                )
            if params["op"] not in ("eq", "le", "ge") or (
                kind == "boolean" and params["op"] != "eq"
            ):
                fail(loc, "invalid comparison for output type")
        compiled.append(
            {"id": sid, "action": action, "timeout_s": timeout, "params": params}
        )
        outputs[sid] = (
            handlers[action].outputs if action in handlers else OUTPUTS[action]
        )
    return compiled


def compile_scenarios(documents, profile=None, handlers=None):
    """Expand named variants and selected profiles, preserving declaration order."""
    handlers = dict(handlers or {})
    if set(handlers) & set(OUTPUTS):
        fail("handlers", "adapters cannot override core action names")
    if profile is not None and profile not in PROFILES:
        fail("profile", f"unknown profile {profile}")
    instances, seen = [], set()
    for source, doc in documents:
        mapping(
            doc,
            source,
            {
                "schema_version",
                "id",
                "description",
                "category",
                "profiles",
                "requires",
                "environment",
                "execution",
                "variants",
                "stages",
                "findings",
                "tags",
                "legacy_case_ids",
                "estimated_duration_s",
            },
            {
                "schema_version",
                "id",
                "description",
                "category",
                "environment",
                "stages",
            },
        )
        if type(doc["schema_version"]) is not int or doc["schema_version"] != 1:
            fail(source + ".schema_version", "only schema_version 1 is supported")
        sid = identifier(doc["id"], source + ".id")
        if sid in seen:
            fail(source, f"duplicate scenario id {sid}")
        seen.add(sid)
        if not isinstance(doc["description"], str) or not doc["description"].strip():
            fail(source + ".description", "contract description required")
        if not isinstance(doc["category"], str) or doc["category"] not in CATEGORIES:
            fail(source + ".category", "unknown category")
        tags = doc.get("tags", [])
        legacy = doc.get("legacy_case_ids", [])
        for key, items in (("tags", tags), ("legacy_case_ids", legacy)):
            if (
                not isinstance(items, list)
                or any(not isinstance(x, str) for x in items)
                or len(set(items)) != len(items)
            ):
                fail(source + "." + key, "expected unique string list")
        estimate = number(
            doc.get("estimated_duration_s", 60),
            source + ".estimated_duration_s",
            minimum=0.001,
        )
        profiles = names(
            doc.get("profiles", list(PROFILES)), source + ".profiles", PROFILES
        )
        if not profiles:
            fail(source + ".profiles", "must not be empty")
        requires = names(
            doc.get("requires", []),
            source + ".requires",
            set().union(*PROFILE_CAPS.values()),
        )
        execution = mapping(
            doc.get("execution", {}),
            source + ".execution",
            {"timeout_s", "stage_timeout_s", "cleanup_timeout_s"},
        )
        budgets = {
            key: number(
                execution.get(key, default), source + ".execution." + key, minimum=0.001
            )
            for key, default in (
                ("timeout_s", 600),
                ("stage_timeout_s", 60),
                ("cleanup_timeout_s", 120),
            )
        }
        variants = doc.get("variants", [{"id": "default"}])
        if not isinstance(variants, list) or not variants:
            fail(source + ".variants", "expected nonempty explicit variant list")
        variant_ids = set()
        for variant in variants:
            loc = source + ".variants"
            mapping(
                variant,
                loc,
                {"id", "profiles", "environment_overrides", "stage_overrides"},
                {"id"},
            )
            vid = identifier(variant["id"], loc + ".id")
            if vid in variant_ids:
                fail(loc, f"duplicate variant {vid}")
            variant_ids.add(vid)
            selected = names(
                variant.get("profiles", profiles), loc + ".profiles", profiles
            )
            if not selected:
                fail(loc + ".profiles", "must not be empty")
            env = copy.deepcopy(doc["environment"])
            if not isinstance(env, dict):
                fail(source + ".environment", "expected mapping")
            patch = mapping(
                variant.get("environment_overrides", {}),
                loc + ".environment_overrides",
                {
                    "n_prefill",
                    "n_decode",
                    "prefill_cache_blocks",
                    "decode_cache_blocks",
                    "config_overrides",
                },
            )
            for key, value in patch.items():
                if key == "config_overrides":
                    if not isinstance(value, dict) or not isinstance(
                        env.get(key, {}), dict
                    ):
                        fail(loc, "config_overrides must be mapping")
                    env[key] = {**env.get(key, {}), **copy.deepcopy(value)}
                else:
                    env[key] = copy.deepcopy(value)
            steps = copy.deepcopy(doc["stages"])
            if not isinstance(steps, list):
                fail(source + ".stages", "expected stage list")
            patches = mapping(
                variant.get("stage_overrides", {}),
                loc + ".stage_overrides",
                {
                    s.get("id")
                    for s in steps
                    if isinstance(s, dict) and isinstance(s.get("id"), str)
                },
            )
            for step in steps:
                if (
                    isinstance(step, dict)
                    and isinstance(step.get("id"), str)
                    and step.get("id") in patches
                ):
                    patch = patches[step["id"]]
                    if not isinstance(patch, dict) or not isinstance(
                        step.get("params", {}), dict
                    ):
                        fail(loc, "stage overrides replace named params only")
                    step["params"] = {**step.get("params", {}), **copy.deepcopy(patch)}
            compiled = stages(
                steps, source + f"::{vid}.stages", budgets["stage_timeout_s"], handlers
            )
            check_ids = set()
            action_requires = set(requires)
            additions = 0
            for stage in compiled:
                action = stage["action"]
                checks = (
                    handlers[action].checks
                    if action in handlers
                    else ({"comparison"} if action == "check" else set())
                )
                stage["check_ids"] = sorted(checks)
                check_ids.update(stage["id"] + "." + check for check in checks)
                if action in handlers:
                    descriptor = handlers[action]
                    action_requires.update(descriptor.requires)
                    bound = descriptor.max_dynamic_additions
                    bound = bound(stage["params"]) if callable(bound) else bound
                    additions += number(
                        bound,
                        source + f"::{vid}.{stage['id']}.max_dynamic_additions",
                        integer=True,
                    )
            findings = names(
                doc.get("findings", []),
                source + ".findings",
                check_ids,
            )
            for p in selected:
                # Validate all declared variants, even when CLI selects a subset.
                if action_requires - PROFILE_CAPS[p]:
                    fail(
                        source,
                        f"profile {p} lacks capabilities {sorted(action_requires - PROFILE_CAPS[p])}",
                    )
                if (
                    any(
                        s["action"] == "request"
                        and s["params"]["consume"] == "deferred"
                        for s in compiled
                    )
                    and "enqueue_batch" not in PROFILE_CAPS[p]
                ):
                    fail(
                        source,
                        f"profile {p}: deferred consumption requires enqueue_batch",
                    )
                resolved = environment(env, source + f"::{vid}.environment", p)
                if resolved["n_prefill"] + resolved["n_decode"] + additions > 149:
                    fail(
                        source,
                        "initial workers plus cumulative additions overlap the reserved victim port range",
                    )
                if profile is not None and p != profile:
                    continue
                instances.append(
                    {
                        "id": f"{sid}::{vid}::{p}",
                        "scenario": sid,
                        "scenario_id": sid,
                        "variant": vid,
                        "variant_id": vid,
                        "profile": p,
                        "category": doc["category"],
                        "description": doc["description"],
                        "requires": sorted(action_requires),
                        "source": "yaml",
                        "source_path": source,
                        "tags": list(tags),
                        "legacy_case_ids": list(legacy),
                        "estimated_duration_s": estimate,
                        "resource_budget": {
                            "backend": "java_mock",
                            "bounded": True,
                            "initial_workers": resolved["n_prefill"]
                            + resolved["n_decode"],
                            "max_dynamic_additions": additions,
                            "mock_control_offset": -1,
                            "victim_control_offset": 149,
                            "victim_grpc_offset": 150,
                            "reserved_tail_offset": 151,
                        },
                        "environment": resolved,
                        "execution": dict(budgets),
                        "stages": copy.deepcopy(compiled),
                        "findings": list(findings),
                    }
                )
    return instances
