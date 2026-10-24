"""Compile explicit instances and typed stage references without process imports."""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace

from adapters.java_mock_resources import VICTIM_OFFSETS, JavaMockBudget
from flexlb_cfg import PROFILES
from scenario.environment_config import environment as _environment
from scenario.environment_config import read_environment
from scenario.environment_declaration import EnvironmentDeclaration, variant_environment_fields
from scenario.environment_resolver import CAPABILITIES, EnvironmentResolver
from scenario.execution_policy import DEFAULT_BUDGETS, EXECUTION_BUDGETS, EXECUTION_FIELDS, normalize_execution, normalize_metadata
from scenario.plan import CompiledInstance, EnvironmentPlan, ExecutionBudgets, StagePlan, json_document
from scenario.program import ProgramPlan, VariantPlan
from scenario.stage_compiler import OUTPUTS, stages
from scenario.stage_parameters import AdapterParameters
from scenario.validation import fail, identifier, mapping, names, number

CATEGORIES = {"cancel", "status", "kv", "balance", "elastic", "engine_fault", "master", "admission", "priority"}


def plan_counts(plans):
    """Counts describe the selected declaration set, never successful execution."""
    return {
        "logical_scenarios": len({p["scenario_id"] for p in plans}),
        "variants": len({(p["scenario_id"], p["variant_id"]) for p in plans}),
        "instances": len(plans),
        "checks": sum(len(s["check_ids"]) for p in plans for s in p["stages"]),
    }


@dataclass(frozen=True)
class _DocumentCompilation:
    source: str
    doc: ProgramPlan
    profile: str | None
    handlers: dict
    grade: str
    sid: str
    tags: list[str]
    estimate: float
    profiles: list[str]
    requires: list[str]
    budgets: dict[str, float]


@dataclass(frozen=True)
class _VariantCompilation:
    document: _DocumentCompilation
    variant: VariantPlan
    vid: str
    selected: list[str]
    variant_budgets: dict[str, float]
    variant_execution: dict
    declaration: EnvironmentDeclaration
    resolver: EnvironmentResolver
    env: dict
    compiled: tuple[StagePlan, ...]
    variant_requires: set[str]
    action_requires: set[str]
    additions: int
    findings: list[str]


def _compile_documents(documents, profile=None, handlers=None, grade="normal"):
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
        instances.extend(_compile_document(source, doc, profile, handlers, grade, seen))
    return instances


def _compile_document(source, doc, profile, handlers, grade, seen):
    if not isinstance(doc, ProgramPlan):
        fail(source, "expected ProgramPlan from a Python builder")
    sid = identifier(doc.id, source + ".id")
    if sid in seen:
        fail(source, f"duplicate scenario id {sid}")
    seen.add(sid)
    if not isinstance(doc.description, str) or not doc.description.strip():
        fail(source + ".description", "contract description required")
    if not isinstance(doc.category, str) or doc.category not in CATEGORIES:
        fail(source + ".category", "unknown category")
    tags = doc.tags
    for key, items in (("tags", tags),):
        if not isinstance(items, tuple) or any(not isinstance(x, str) for x in items) or len(set(items)) != len(items):
            fail(source + "." + key, "expected unique string list")
    estimate = number(doc.estimated_duration_s, source + ".estimated_duration_s", minimum=0.001)
    profiles = names(list(doc.profiles), source + ".profiles", PROFILES)
    if not profiles:
        fail(source + ".profiles", "must not be empty")
    requires = names(list(doc.requires), source + ".requires", CAPABILITIES)
    execution = mapping(doc.execution, source + ".execution", EXECUTION_BUDGETS)
    budgets = {
        key: number(execution.get(key, default), source + ".execution." + key, minimum=0.001) for key, default in DEFAULT_BUDGETS.items()
    }
    variants = doc.variants
    if not isinstance(variants, tuple) or not variants:
        fail(source + ".variants", "expected nonempty typed variant sequence")
    context = _DocumentCompilation(source, doc, profile, handlers, grade, sid, tags, estimate, profiles, requires, budgets)
    variant_ids = set()
    instances = []
    for variant in variants:
        instances.extend(_compile_variant(context, variant, variant_ids))
    return instances


def _compile_variant(document, variant, variant_ids):
    source = document.source
    doc = document.doc
    handlers = document.handlers
    profiles = document.profiles
    requires = document.requires
    budgets = document.budgets
    loc = source + ".variants"
    if not isinstance(variant, VariantPlan):
        fail(loc, "expected VariantPlan")
    vid = identifier(variant.id, loc + ".id")
    if vid in variant_ids:
        fail(loc, f"duplicate variant {vid}")
    variant_ids.add(vid)
    selected = names(list(variant.profiles) if variant.profiles is not None else profiles, loc + ".profiles", profiles)
    if not selected:
        fail(loc + ".profiles", "must not be empty")
    variant_budgets = dict(budgets)
    variant_execution = mapping(variant.execution, loc + ".execution", EXECUTION_FIELDS)
    for key, value in variant_execution.items():
        if key not in budgets:
            continue
        variant_budgets[key] = number(value, loc + ".execution." + key, minimum=0.001)
    variant_requires = set(requires) | set(names(list(variant.requires), loc + ".requires", CAPABILITIES))
    env = copy.deepcopy(doc.environment)
    if not isinstance(env, dict):
        fail(source + ".environment", "expected mapping")
    from scenario.config_data import merge_environment

    env = merge_environment(env, mapping(variant.environment_overrides, loc + ".environment_overrides", variant_environment_fields()))
    declaration, resolver = read_environment(env, source + f"::{vid}.environment")
    # Action validation receives the same resolved topology as execution.
    for field in ("n_prefill", "n_decode", "prefill_cache_blocks", "decode_cache_blocks"):
        value = getattr(declaration, field)
        if value is not None:
            env[field] = value
    steps = tuple(
        replace(step, timeout_s=variant_budgets["cleanup_timeout_s"]) if step.action == "teardown" and step.timeout_s is None else step
        for step in variant.stages
    )
    compiled = stages(
        steps, source + f"::{vid}.stages", variant_budgets["stage_timeout_s"], handlers, env, selected, doc.implementation.get("program")
    )
    check_ids = set()
    action_requires = set(variant_requires)
    additions = 0
    for stage in compiled:
        action = stage.action
        check_ids.update(stage.id + "." + check for check in stage.check_ids)
        if action in handlers:
            descriptor = handlers[action]
            action_requires.update(descriptor.requires)
            bound = descriptor.max_dynamic_additions
            bound = bound(stage.parameters.to_dict()) if callable(bound) else bound
            additions += number(bound, source + f"::{vid}.{stage.id}.max_dynamic_additions", integer=True)
    if not check_ids:
        fail(source + f"::{vid}", "scenario must declare at least one check")
    findings = names(list(variant.findings if variant.findings is not None else doc.findings), source + ".findings", check_ids)
    context = _VariantCompilation(
        document,
        variant,
        vid,
        selected,
        variant_budgets,
        variant_execution,
        declaration,
        resolver,
        env,
        compiled,
        variant_requires,
        action_requires,
        additions,
        findings,
    )
    instances = []
    for p in selected:
        instances.extend(_compile_profile(context, p))
    return instances


def _compile_profile(context, p):
    document = context.document
    source = document.source
    doc = document.doc
    profile = document.profile
    handlers = document.handlers
    grade = document.grade
    sid = document.sid
    tags = document.tags
    estimate = document.estimate
    budgets = document.budgets
    variant = context.variant
    vid = context.vid
    variant_budgets = context.variant_budgets
    variant_execution = context.variant_execution
    declaration = context.declaration
    resolver = context.resolver
    compiled = context.compiled
    variant_requires = context.variant_requires
    action_requires = context.action_requires
    additions = context.additions
    findings = context.findings
    instances = []
    # Validate all declared variants, even when CLI selects a subset.
    resolved = resolver.resolve(declaration, source + f"::{vid}.environment", p)
    initial_workers = resolved.declaration.n_prefill + resolved.declaration.n_decode
    max_environment_workers = initial_workers
    for stage in compiled:
        if stage.action in handlers:
            bound = handlers[stage.action].max_environment_workers
            bound = bound(stage.parameters.to_dict(), p) if callable(bound) else bound
            max_environment_workers = max(
                max_environment_workers, number(bound, source + f"::{vid}.{stage.id}.max_environment_workers", integer=True)
            )
    caps = set(resolved.effective_capabilities)
    stage_caps = caps
    for stage in compiled:
        descriptor = handlers.get(stage.action)
        if descriptor:
            try:
                descriptor.execution_requirements.validate(variant_execution)
            except ValueError as exc:
                fail(source + f"::{vid}.{stage.id}.execution", str(exc))
        stage_requires = set(variant_requires) | (set(descriptor.requires) if descriptor else set())
        if stage_requires - stage_caps:
            fail(source, f"profile {p} stage {stage.id} effective environment lacks capabilities {sorted(stage_requires - stage_caps)}")
        if stage.action == "request" and stage.parameters.to_dict()["consume"] == "deferred" and "enqueue_batch" not in stage_caps:
            fail(source, f"profile {p} stage {stage.id}: deferred consumption requires enqueue_batch")
        if descriptor and descriptor.next_environment is not None:
            next_env = _environment(descriptor.next_environment(stage.parameters.to_dict()), source + f"::{vid}.{stage.id}.environment", p)
            stage_caps = set(next_env["effective_capabilities"])
    if max_environment_workers + additions > VICTIM_OFFSETS[0]:
        fail(source, "maximum environment workers plus cumulative additions overlap the reserved victim port range")
    if profile is not None and p != profile:
        return []
    metadata = normalize_metadata(
        variant.metadata if variant.metadata is not None else {"kind": "functional", "description": doc.description}
    )
    policy_data = {**variant_budgets, **{key: copy.deepcopy(value) for key, value in variant_execution.items() if key not in budgets}}
    policy_data.setdefault("collection", "diagnostic")
    policy = normalize_execution(policy_data, kind=metadata["kind"])
    from scenario.registry import registry

    registered = registry().cases.get(doc.implementation.get("program"))
    try:
        if registered is not None:
            registered.definition.execution_requirements.validate(policy)
    except ValueError as exc:
        fail(source + f"::{vid}.execution", str(exc))
    attachments = dict(implementation=copy.deepcopy(doc.implementation))
    for key in ("reports", "reporting"):
        value = getattr(variant, key)
        if value is not None:
            attachments[key] = copy.deepcopy(value)
    if variant.monitoring_query_plan is not None:
        attachments["implementation"]["monitoring_query_plan"] = copy.deepcopy(variant.monitoring_query_plan)
    instances.append(
        CompiledInstance(
            id=f"{sid}::{vid}::{p}",
            scenario_id=sid,
            variant_id=vid,
            profile=p,
            grade=grade,
            category=doc.category,
            description=doc.description,
            requires=tuple(sorted(action_requires)),
            source="yaml",
            source_path=source,
            tags=tuple(tags),
            estimated_duration_s=estimate,
            resource_budget=JavaMockBudget(
                initial_workers, additions, max_environment_workers if max_environment_workers > initial_workers else None
            ),
            environment=EnvironmentPlan(
                resolved.declaration,
                resolved.snapshot,
                json_document(resolved.master_patch.declared),
                tuple(resolved.effective_axes.items()),
                resolved.effective_capabilities,
            ),
            execution=ExecutionBudgets.read(policy),
            stages=compiled,
            findings=tuple(findings),
            metadata=AdapterParameters(metadata),
            **{key: tuple(value) if key == "reports" else AdapterParameters(value) for key, value in attachments.items()},
        )
    )
    return instances


class ScenarioCompiler:
    """Compile Python program declarations into immutable execution objects."""

    def __init__(self, handlers=None):
        self.handlers = dict(handlers or {})

    def compile(self, documents, profile=None, grade="normal") -> list[CompiledInstance]:
        return _compile_documents(documents, profile, self.handlers, grade)


def compile_scenarios(documents, profile=None, handlers=None, grade="normal"):
    """JSON projection for CLI/catalog consumers; the compiler returns domain objects."""
    return [instance.to_dict() for instance in ScenarioCompiler(handlers).compile(documents, profile, grade)]
