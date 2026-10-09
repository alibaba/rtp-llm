"""Validate and resolve profile, preset and case environment settings."""

import copy
import json
import re
from flexlb_cfg import (
    OMIT, PROFILE_CAPS, ProfileIdentityError, PROFILES, VICTIM_STAGES,
    ConfigOverride, render_env, validate_profile_identity,
)
from runtime.perf_presets import capture_defaults, load_preset, preset_names
from scenario.loader import ScenarioError
from scenario.validation import fail, mapping, number, names

INTEGER_OVERRIDES = {
    "cleanup_interval_ms",
    "status_rpc_ms",
    "max_requests",
    "default_priority",
    "queue_timeout_ms",
    "decode_max_engine_requests",
    "decode_max_kv_usage_percent",
    "max_predicted_execution_ms",
    "request_timeout_ms",
    "max_collection_wait_ms",
    "status_stale_after_ms",
    "max_inflight_per_prefill_worker",
}

_SPECIAL_ENVIRONMENT_FIELDS = frozenset({
    "backend", "n_prefill", "n_decode", "prefill_cache_blocks",
    "decode_cache_blocks", "config_overrides", "profile_overrides",
    "discovery", "perf_preset", "model_override", "prefill_perf",
    "prefill_cache_policy", "master_layout", "master_stable_window_s",
    "metric_whitelist",
})

_ABSENT = object()

OPTIONAL_SCALARS = {
    "debug_enabled": (bool, False, None),
    "mock_auto_fetch": (bool, _ABSENT, None),
    "master_debug_log": (bool, _ABSENT, None),
    "master_sync_log": (bool, False, None),
    "mock_fetch_attach_timeout_ms": (int, _ABSENT, 1),
    "prefill_max_waiting_batches": (int, _ABSENT, 0),
}


def environment_fields():
    """One vocabulary for both top-level fields and simple scalar declarations."""
    return _SPECIAL_ENVIRONMENT_FIELDS | OPTIONAL_SCALARS.keys()


def variant_environment_fields():
    # A variant may change declared data, but never switch its case's backend.
    return environment_fields() - {"backend"}


def optional_scalars(value, path, result, declarations=None):
    """Apply simple scalar rules; retain each field's old omission semantics."""
    if declarations is None:
        declarations = OPTIONAL_SCALARS
    for key, (kind, default, minimum) in declarations.items():
        if key not in value and default is _ABSENT:
            continue
        val = value.get(key, default)
        if kind is bool:
            if type(val) is not bool:
                fail(path + "." + key, "expected boolean")
            result[key] = val
        else:
            result[key] = number(val, path + "." + key, minimum=minimum, integer=True)

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
    if preemption is None and scheduler["ordering"]["type"] == "PRIORITY":
        preemption = {"allowedVictimStages": list(VICTIM_STAGES)}
    if preemption:
        caps.add("preemption")
        if "DECODE_ENGINE_OWNED" in preemption.get("allowedVictimStages", []):
            caps.add("engine_cancellation")
    return axes, sorted(caps)


def environment(value, path, profile):
    value = mapping(
        value,
        path,
        environment_fields(),
    )
    if value.get("backend", "java_mock") != "java_mock":
        fail(
            path + ".backend",
            "only java_mock is implemented; GPU/multi-host/TP/DP layouts need separate capabilities",
        )
    result = {"backend": "java_mock", "n_prefill": 2, "n_decode": 4}
    for key, default, allowed in (
        ("discovery", "file", ("file", "discovery_file")),
        ("perf_preset", "default", preset_names()),
        ("master_layout", "single", ("single", "dual_standalone")),
    ):
        val = value.get(key, default)
        if val not in allowed:
            fail(path + "." + key, f"invalid value {val!r}; expected one of {allowed}")
        result[key] = val
    captured = capture_defaults(result["perf_preset"])
    for field, capture_field in (
        ("n_prefill", "n_prefill"), ("n_decode", "n_decode"),
        ("prefill_cache_blocks", "prefill_kv_pool_blocks"),
        ("decode_cache_blocks", "decode_kv_pool_blocks"),
    ):
        if field not in value and capture_field in captured:
            result[field] = captured[capture_field]
    if "model_override" in value:
        override = mapping(value["model_override"], path + ".model_override",
            {"baseline", "reason"}, {"baseline", "reason"})
        if override["baseline"] != result["perf_preset"] or not isinstance(override["reason"], str) or not override["reason"].strip():
            fail(path + ".model_override", "baseline must name perf_preset and reason must be nonempty")
        result["model_override"] = dict(override)
    optional_scalars(value, path, result)
    if "metric_whitelist" in value:
        whitelist = value["metric_whitelist"]
        if not isinstance(whitelist, str) or not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*(,[A-Za-z_][A-Za-z0-9_]*){0,15}", whitelist
        ):
            fail(
                path + ".metric_whitelist",
                "expected 1..16 comma-separated metric identifiers",
            )
        if len(whitelist) > 1024:
            fail(path + ".metric_whitelist", "metric whitelist exceeds byte budget")
        result["metric_whitelist"] = whitelist
    if result["master_sync_log"] and result["master_layout"] != "single":
        fail(
            path + ".master_sync_log",
            "isolated sync log currently requires single master",
        )
    result["master_stable_window_s"] = number(
        value.get("master_stable_window_s", 3), path + ".master_stable_window_s"
    )
    if "prefill_cache_policy" in value:
        field = path + ".prefill_cache_policy"
        keys = {"device_tree", "memory_tree", "memory_blocks"}
        cache = mapping(value["prefill_cache_policy"], field, keys, {"device_tree", "memory_tree"})
        for key in ("device_tree", "memory_tree"):
            if type(cache[key]) is not bool:
                fail(field + "." + key, "expected boolean")
        if "memory_blocks" in cache:
            number(cache["memory_blocks"], field + ".memory_blocks", integer=True)
        result["prefill_cache_policy"] = dict(cache)
    if "prefill_perf" in value:
        field = path + ".prefill_perf"
        required = {"fixed_ms", "scale", "max_batch_tokens", "max_batch_requests"}
        perf = mapping(value["prefill_perf"], field, required, required)
        result["prefill_perf"] = {
            key: number(val, field + "." + key, integer=key.startswith("max_batch_"))
            for key, val in perf.items()
        }
    for key in ("n_prefill", "n_decode", "prefill_cache_blocks", "decode_cache_blocks"):
        if key in value:
            result[key] = number(value[key], path + "." + key, minimum=1, integer=True)
    _, preset_runtime = load_preset(result["perf_preset"])
    paired_settings = preset_runtime.get("paired_master")
    paired_master = paired_settings.for_profile(profile) if paired_settings else {}
    profile_overrides = mapping(value.get("profile_overrides", {}),
                                path + ".profile_overrides", PROFILES)
    # Validate every keyed entry, including profiles excluded by CLI selection.
    for target, patch in profile_overrides.items():
        mapping(patch, path + ".profile_overrides." + target, INTEGER_OVERRIDES | {
            "ordering", "decision", "dispatcher", "preemption", "decision_lifetime",
            "prefill_expression", "cache_affinity_max_extra_ttft_ms",
            "cache_affinity_min_prefix_hit_percent",
        })
        if target != profile:
            environment({**value, "profile_overrides": {target: patch}}, path, target)
    common = value.get("config_overrides", {})
    if not isinstance(common, dict):
        fail(path + ".config_overrides", "expected mapping")
    for target, patch in [(profile, common), (profile, paired_master), *profile_overrides.items()]:
        try:
            validate_profile_identity(target, patch)
        except ProfileIdentityError as exc:
            raise ScenarioError(f"{path}: {exc}") from exc
    overrides = mapping(
        {**common, **{k: v for k, v in profile_overrides.get(profile, {}).items() if v is not None}},
        path + ".config_overrides",
        INTEGER_OVERRIDES
        | {
            "ordering",
            "decision",
            "dispatcher",
            "preemption",
            "decision_lifetime",
            "prefill_expression",
            "cache_affinity_max_extra_ttft_ms",
            "cache_affinity_min_prefix_hit_percent",
        },
    )
    declared_overrides = dict(overrides)
    overrides = {**paired_master, **{k: v for k, v in declared_overrides.items() if v is not None}}
    kwargs = {}
    for key, val in overrides.items():
        field = path + ".config_overrides." + key
        if val is None:
            continue
        if key in {"ordering", "decision", "dispatcher"}:
            choices = {
                "ordering": ("fifo", "priority"),
                "decision": ("single", "fixed_window"),
                "dispatcher": ("batch", "non_batch"),
            }[key]
            if val not in choices:
                fail(field, f"expected one of {choices}")
        elif key == "cache_affinity_max_extra_ttft_ms":
            number(val, field, minimum=0, integer=True)
        elif key == "cache_affinity_min_prefix_hit_percent":
            number(val, field, minimum=0)
        elif key == "prefill_expression":
            if not isinstance(val, str) or not val.strip() or len(val) > 4096:
                fail(field, "expected a nonempty formula of at most 4096 characters")
        elif key == "preemption":
            mapping(
                val,
                field,
                {"allowed_victim_stages", "timeout_ms"},
                {"allowed_victim_stages"},
            )
            victim_stages = names(
                val["allowed_victim_stages"],
                field + ".allowed_victim_stages",
                VICTIM_STAGES,
            )
            if not victim_stages:
                fail(field, "preemption victim stages cannot be empty")
            if "timeout_ms" in val:
                number(
                    val["timeout_ms"], field + ".timeout_ms", minimum=1, integer=True
                )
        elif key == "decision_lifetime":
            number(val, field, minimum=1)
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
    if captured:
        changed = [field for field, source_field in (
            ("n_prefill", "n_prefill"), ("n_decode", "n_decode"),
            ("prefill_cache_blocks", "prefill_kv_pool_blocks"),
            ("decode_cache_blocks", "decode_kv_pool_blocks"),
        ) if field in value and value[field] != captured[source_field]]
        if ("prefill_expression" in declared_overrides and declared_overrides["prefill_expression"] != paired_master.get("prefill_expression")) or "prefill_perf" in value:
            changed.append("prefill model")
        changed.extend("master." + key for key, val in declared_overrides.items()
                       if key in paired_master and val != paired_master[key])
        if "prefill_cache_policy" in value:
            performance, _ = load_preset(result["perf_preset"])
            policy = value["prefill_cache_policy"]
            original = performance.get("prefill", {})
            memory = original.get("memory_cache", {})
            if (policy["device_tree"] != original.get("enable_gpu_prefix_tree")
                or policy["memory_tree"] != memory.get("enable_prefix_tree")
                or ("memory_blocks" in policy and policy["memory_blocks"] != memory.get("capacity_blocks"))):
                changed.append("prefill_cache_policy")
        if changed and "model_override" not in result:
            fail(path + ".model_override", "test deviation requires baseline and reason: " + ", ".join(changed))
    result["config_overrides"] = copy.deepcopy(overrides)
    result["effective_axes"], result["effective_capabilities"] = effective_capabilities(
        result["resolved_config"]
    )
    return result
