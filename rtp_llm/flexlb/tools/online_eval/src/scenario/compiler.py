"""Compile explicit instances and typed stage references without process imports."""

import copy

from flexlb_cfg import PROFILES

from runtime.resource_plan import VICTIM_OFFSETS
from runtime.perf_presets import capture_defaults
from scenario.validation import fail, mapping, identifier, number, names
from scenario.stage_compiler import OUTPUTS, stages
from scenario.environment_config import environment, variant_environment_fields, CAPABILITIES

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
        for key, items in (("tags", tags),):
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
                    "test",
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
            env = copy.deepcopy(doc["environment"])
            if not isinstance(env, dict):
                fail(source + ".environment", "expected mapping")
            patch = mapping(
                variant.get("environment_overrides", {}),
                loc + ".environment_overrides",
                variant_environment_fields(),
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
            capture = capture_defaults(env.get("perf_preset", "default"))
            for field, source_field in (
                ("n_prefill", "n_prefill"), ("n_decode", "n_decode"),
                ("prefill_cache_blocks", "prefill_kv_pool_blocks"),
                ("decode_cache_blocks", "decode_kv_pool_blocks"),
            ):
                if field not in env and source_field in capture:
                    env[field] = capture[source_field]
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
                getattr(doc, "implementation", {}).get("program"),
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
                initial_workers = resolved["n_prefill"] + resolved["n_decode"]
                max_environment_workers = initial_workers
                for stage in compiled:
                    if stage["action"] in handlers:
                        bound = handlers[stage["action"]].max_environment_workers
                        bound = bound(stage["params"], p) if callable(bound) else bound
                        max_environment_workers = max(
                            max_environment_workers,
                            number(
                                bound,
                                source
                                + f"::{vid}.{stage['id']}.max_environment_workers",
                                integer=True,
                            ),
                        )
                caps = set(resolved["effective_capabilities"])
                stage_caps = caps
                for stage in compiled:
                    descriptor = handlers.get(stage["action"])
                    stage_requires = set(variant_requires) | (
                        set(descriptor.requires) if descriptor else set()
                    )
                    if stage_requires - stage_caps:
                        fail(
                            source,
                            f"profile {p} stage {stage['id']} effective environment lacks capabilities {sorted(stage_requires - stage_caps)}",
                        )
                    if (
                        stage["action"] == "request"
                        and stage["params"]["consume"] == "deferred"
                        and "enqueue_batch" not in stage_caps
                    ):
                        fail(
                            source,
                            f"profile {p} stage {stage['id']}: deferred consumption requires enqueue_batch",
                        )
                    if descriptor and descriptor.next_environment is not None:
                        next_env = environment(
                            descriptor.next_environment(stage["params"]),
                            source + f"::{vid}.{stage['id']}.environment",
                            p,
                        )
                        stage_caps = set(next_env["effective_capabilities"])
                if max_environment_workers + additions > VICTIM_OFFSETS[0]:
                    fail(
                        source,
                        "maximum environment workers plus cumulative additions overlap the reserved victim port range",
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
                        **({"test": copy.deepcopy(variant["test"])} if "test" in variant else {}),
                        "effective_axes": resolved["effective_axes"],
                        "effective_capabilities": resolved["effective_capabilities"],
                        "grade": grade,
                        "category": doc["category"],
                        "description": doc["description"],
                        "requires": sorted(action_requires),
                        "source": "yaml",
                        "source_path": source,
                        **(
                            {"implementation": copy.deepcopy(doc.implementation)}
                            if hasattr(doc, "implementation")
                            else {}
                        ),
                        "tags": list(tags),
                        "estimated_duration_s": estimate,
                        "resource_budget": {
                            "backend": "java_mock",
                            "bounded": True,
                            "initial_workers": initial_workers,
                            **(
                                {"max_environment_workers": max_environment_workers}
                                if max_environment_workers > initial_workers
                                else {}
                            ),
                            "max_dynamic_additions": additions,
                            "mock_control_offset": -1,
                            "victim_control_offset": VICTIM_OFFSETS[0],
                            "victim_grpc_offset": VICTIM_OFFSETS[1],
                            "reserved_tail_offset": VICTIM_OFFSETS[2],
                        },
                        "environment": resolved,
                        "execution": dict(variant_budgets),
                        "stages": copy.deepcopy(compiled),
                        "findings": list(findings),
                    }
                )
    return instances
