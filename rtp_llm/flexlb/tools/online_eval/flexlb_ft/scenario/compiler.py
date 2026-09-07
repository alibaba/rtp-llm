"""Compile explicit instances and typed stage references without process imports."""

import copy
import json
import math
import re

from flexlb_cfg import (
    OMIT,
    PROFILE_CAPS,
    PROFILES,
    VICTIM_STAGES,
    ConfigOverride,
    render_env,
)

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


def plan_counts(plans):
    """Counts describe the selected declaration set, never successful execution."""
    return {
        "logical_scenarios": len({p["scenario_id"] for p in plans}),
        "variants": len({(p["scenario_id"], p["variant_id"]) for p in plans}),
        "instances": len(plans),
        "checks": sum(len(s["check_ids"]) for p in plans for s in p["stages"]),
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
CAPABILITIES = set().union(*PROFILE_CAPS.values()) | {
    "priority",
    "preemption",
    "engine_cancellation",
}


def effective_capabilities(config):
    scheduler, dispatcher = config["scheduler"], config["dispatcher"]
    axes = dict(
        scheduler=scheduler["type"],
        ordering=scheduler["ordering"]["type"],
        decision=scheduler["decision"]["type"],
        dispatcher=dispatcher["type"],
    )
    caps = {"queue", axes["ordering"].lower(), axes["decision"].lower()}
    if dispatcher["type"] == "BATCH":
        caps.update({"batch_dispatch", "enqueue_batch", "fetch_response"})
    elif dispatcher["type"] == "NON_BATCH":
        caps.update({"non_batch_dispatch", "frontend_send", "generate_stream"})
    else:
        raise ValueError("unrecognized effective dispatcher")
    preemption = scheduler["ordering"].get("preemption")
    if preemption:
        caps.add("preemption")
        if preemption.get("engineCancellation"):
            caps.add("engine_cancellation")
    return axes, sorted(caps)


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
            "discovery",
            "perf_preset",
            "debug_enabled",
            "master_layout",
            "master_stable_window_s",
        },
    )
    if value.get("backend", "java_mock") != "java_mock":
        fail(
            path + ".backend",
            "only java_mock is implemented; GPU/multi-host/TP/DP layouts need separate capabilities",
        )
    result = {"backend": "java_mock", "n_prefill": 2, "n_decode": 4}
    for key, default, allowed in (
        ("discovery", "file", ("file", "discovery_file")),
        ("perf_preset", "default", ("default", "fault_env")),
        ("master_layout", "single", ("single", "dual_standalone")),
    ):
        val = value.get(key, default)
        if val not in allowed:
            fail(path + "." + key, f"expected one of {allowed}")
        result[key] = val
    if type(value.get("debug_enabled", False)) is not bool:
        fail(path + ".debug_enabled", "expected boolean")
    result["debug_enabled"] = value.get("debug_enabled", False)
    result["master_stable_window_s"] = number(
        value.get("master_stable_window_s", 3), path + ".master_stable_window_s"
    )
    for key in ("n_prefill", "n_decode", "prefill_cache_blocks", "decode_cache_blocks"):
        if key in value:
            result[key] = number(value[key], path + "." + key, minimum=1, integer=True)
    overrides = mapping(
        value.get("config_overrides", {}),
        path + ".config_overrides",
        INTEGER_OVERRIDES | {"ordering", "decision", "dispatcher", "preemption"},
    )
    kwargs = {}
    for key, val in overrides.items():
        field = path + ".config_overrides." + key
        if key in {"ordering", "decision", "dispatcher"}:
            choices = {
                "ordering": ("fifo", "priority"),
                "decision": ("single", "fixed_window"),
                "dispatcher": ("batch", "non_batch"),
            }[key]
            if val not in choices:
                fail(field, f"expected one of {choices}")
        elif key == "preemption":
            mapping(
                val,
                field,
                {"allowed_victim_stages", "engine_cancellation"},
                {"allowed_victim_stages"},
            )
            victim_stages = names(
                val["allowed_victim_stages"],
                field + ".allowed_victim_stages",
                VICTIM_STAGES,
            )
            if not victim_stages:
                fail(field, "preemption victim stages cannot be empty")
            if "engine_cancellation" in val:
                cancellation = mapping(
                    val["engine_cancellation"],
                    field + ".engine_cancellation",
                    {"ack_timeout_ms", "completion_timeout_ms"},
                    {"ack_timeout_ms", "completion_timeout_ms"},
                )
                for name, timeout in cancellation.items():
                    number(
                        timeout,
                        field + ".engine_cancellation." + name,
                        minimum=1,
                        integer=True,
                    )
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
    result["effective_axes"], result["effective_capabilities"] = effective_capabilities(
        result["resolved_config"]
    )
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


def stages(values, path, default_timeout, handlers, env=None, profiles=()):
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
                params,
                PlanContext(
                    loc + ".params",
                    dict(outputs),
                    copy.deepcopy(env or {}),
                    tuple(profiles),
                ),
            )
            if not isinstance(params, dict):
                fail(loc, "adapter validate must return a mapping")
        elif action in ("setup", "teardown"):
            mapping(params, loc + ".params", set())
            torn_down = action == "teardown"
        elif action == "request":
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
            if params["op"] not in ("eq", "le", "ge") or (
                kind in ("boolean", "string") and params["op"] != "eq"
            ):
                fail(loc, "invalid comparison for output type")
        compiled.append(
            {"id": sid, "action": action, "timeout_s": timeout, "params": params}
        )
        outputs[sid] = (
            handlers[action].outputs if action in handlers else OUTPUTS[action]
        )
    return compiled


def compile_scenarios(documents, profile=None, handlers=None, grade="normal"):
    """Expand named variants and selected profiles, preserving declaration order."""
    handlers = dict(handlers or {})
    if grade not in ("strict", "normal", "loose"):
        fail("grade", "must be strict, normal or loose")
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
            CAPABILITIES,
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
                {
                    "id",
                    "profiles",
                    "environment_overrides",
                    "stage_overrides",
                    "stages",
                    "execution",
                    "requires",
                    "findings",
                    "legacy_case_ids",
                },
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
            variant_budgets = dict(budgets)
            for key, value in mapping(
                variant.get("execution", {}), loc + ".execution", set(budgets)
            ).items():
                variant_budgets[key] = number(
                    value, loc + ".execution." + key, minimum=0.001
                )
            variant_requires = set(requires) | set(
                names(
                    variant.get("requires", []),
                    loc + ".requires",
                    CAPABILITIES,
                )
            )
            variant_legacy = names(
                variant.get("legacy_case_ids", legacy), loc + ".legacy_case_ids", legacy
            )
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
                    "discovery",
                    "perf_preset",
                    "debug_enabled",
                    "master_layout",
                    "master_stable_window_s",
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
            if "stages" in variant and "stage_overrides" in variant:
                fail(
                    loc,
                    "explicit variant stages and stage_overrides are mutually exclusive",
                )
            steps = copy.deepcopy(variant.get("stages", doc.get("stages")))
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
                steps,
                source + f"::{vid}.stages",
                variant_budgets["stage_timeout_s"],
                handlers,
                env,
                selected,
            )
            check_ids = set()
            action_requires = set(variant_requires)
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
            if not check_ids:
                fail(source + f"::{vid}", "scenario must declare at least one check")
            findings = names(
                variant.get("findings", doc.get("findings", [])),
                source + ".findings",
                check_ids,
            )
            for p in selected:
                # Validate all declared variants, even when CLI selects a subset.
                resolved = environment(env, source + f"::{vid}.environment", p)
                caps = set(resolved["effective_capabilities"])
                if action_requires - caps:
                    fail(
                        source,
                        f"profile {p} effective environment lacks capabilities {sorted(action_requires - caps)}",
                    )
                if (
                    any(
                        s["action"] == "request"
                        and s["params"]["consume"] == "deferred"
                        for s in compiled
                    )
                    and "enqueue_batch" not in caps
                ):
                    fail(
                        source,
                        f"profile {p}: deferred consumption requires enqueue_batch",
                    )
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
                        "effective_axes": resolved["effective_axes"],
                        "effective_capabilities": resolved["effective_capabilities"],
                        "grade": grade,
                        "category": doc["category"],
                        "description": doc["description"],
                        "requires": sorted(action_requires),
                        "source": "yaml",
                        "source_path": source,
                        "tags": list(tags),
                        "legacy_case_ids": list(variant_legacy),
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
                        "execution": dict(variant_budgets),
                        "stages": copy.deepcopy(compiled),
                        "findings": list(findings),
                    }
                )
    return instances
