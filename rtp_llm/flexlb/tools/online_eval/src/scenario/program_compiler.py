"""Data-only case configuration. All control flow comes from registered Python programs."""

import copy
import hashlib
import json
from dataclasses import asdict

from document_io import DocumentError as ScenarioError
from monitoring.identity import NAME
from scenario.builder import CaseBuilder
from scenario.config_data import leaf_paths, merge_data, merge_environment, path_in_scope
from scenario.declarations import CaseDeclaration, VariantDeclaration
from scenario.execution_policy import DEFAULT_BUDGETS, EXECUTION_BUDGETS
from scenario.program import ProgramPlan, VariantPlan


def program_module(name, source):
    from scenario.registry import registry

    if not isinstance(name, str) or name not in registry().cases:
        raise ScenarioError(f"{source}: unknown registered Python case {name!r}")
    return registry().cases[name]


def configure_program(config, source) -> ProgramPlan:
    """Build a typed program using a registered Python entry point, without starting services."""
    config = CaseDeclaration.read(config, source)
    name = config.case
    module = program_module(name, source)
    if config.program != "default":
        raise ScenarioError(f"{source}: base case program must be default")
    from scenario.environment_config import environment as validate_environment
    from scenario.validation import identifier

    identifier(config.identity, source + ".id")
    for profile in config.profiles:
        validate_environment(config.environment, source + ".environment", profile)
    variants = []
    seen = set()
    numeric_contracts: dict = {}
    default = VariantDeclaration.read({"id": config.default_variant}, source)
    from scenario.data_archives import archive_variants

    axes = {}
    for row, archive, behavior in archive_variants(config, (default, *config.variants), source):
        identity = row.identity
        if not isinstance(identity, str) or not NAME.fullmatch(identity) or identity in seen:
            raise ScenarioError(f"{source}: missing or duplicate configuration id {identity!r}")
        seen.add(identity)
        if archive is not None:
            axes[identity] = dict(data_archive=archive, variant=behavior)
        variants.append(_build_variant(config, row, identity, module, config.profiles, source, numeric_contracts))
    implementation = _implementation(config, module, name)
    implementation["numeric_parameters"] = numeric_contracts
    if axes:
        implementation["variant_axes"] = axes
    metadata = {key: copy.deepcopy(value) for key, value in config.metadata.items() if key != "kind"}
    metadata.setdefault("category", "")
    for key in ("tags", "requires", "findings"):
        if key in metadata:
            if not isinstance(metadata[key], list):
                raise ScenarioError(source + ".metadata." + key + ": expected string list")
            metadata[key] = tuple(metadata[key])
    return ProgramPlan(
        id=config.identity,
        environment=copy.deepcopy(config.environment),
        variants=tuple(variants),
        profiles=config.profiles,
        implementation=implementation,
        execution={key: value for key, value in config.execution.to_dict().items() if key in EXECUTION_BUDGETS},
        **metadata,
    )


def _select_program(module, program, source):
    build = module.definition.builders.get(program) if isinstance(program, str) else None
    if build is None:
        raise ScenarioError(f"{source}: unknown Python case program {program!r}")
    return build


def _build_variant(config, row, identity, module, selected_profiles, source, numeric_contracts):
    environment = config.environment
    parameters = config.parameters
    program = row.program if row.program is not None else config.program
    build = _select_program(module, program, source)
    profiles = list(selected_profiles)
    if not isinstance(profiles, list) or not profiles or any(not isinstance(p, str) for p in profiles):
        raise ScenarioError(f"{source}: profiles must be a nonempty string list in YAML")
    variant_parameters = row.parameters
    patch = row.environment
    from scenario.numeric_parameters import narrow_parameters, parameter_rules

    schema = module.definition.input_schema
    defaults = parameter_rules(schema.numeric_parameters if schema is not None else {})
    base_rules = narrow_parameters(defaults, config.parameter_schema)
    builder = CaseBuilder(
        merge_environment(environment, patch),
        merge_data(parameters, variant_parameters),
        number_rules=narrow_parameters(base_rules, row.parameter_schema),
        profiles=profiles,
        stage_timeout_s=config.execution.stage_timeout_s or DEFAULT_BUDGETS["stage_timeout_s"],
        optional_numbers=(schema.optional_numbers if schema is not None else ()),
    )
    if schema is not None:
        try:
            builder.config = schema.parse(builder.parameters, builder.number_rules)
        except ValueError as exc:
            raise ScenarioError(f"{source}.parameters: {exc}") from exc
        builder.read_parameters.update(builder.config)
    else:
        builder.validate_numbers()
    numeric_contracts[identity] = {path: asdict(rule) for path, rule in builder.number_rules.items()}
    try:
        build(builder)
    except ValueError as exc:
        raise ScenarioError(f"{source}.parameters: {exc}") from exc
    for field in leaf_paths(builder.parameters):
        if not path_in_scope(field, builder.read_parameters):
            raise ScenarioError(f"{source}: unused YAML parameter {field!r}")
    variant = _variant_contract(config, builder, source)
    if "reports" in variant:
        variant["reports"] = tuple(variant["reports"])
    return VariantPlan(
        id=identity, profiles=tuple(profiles), stages=builder.finish(), environment_overrides=copy.deepcopy(patch), **variant
    )


def _variant_contract(config, builder, source):
    metadata = config.metadata
    execution = config.execution.to_dict()
    from scenario.registry import registry

    try:
        registry().cases[config.case].definition.execution_requirements.validate(execution)
    except ValueError as exc:
        raise ScenarioError(f"{source}.execution: {exc}") from exc
    if (
        builder.metric_dependencies
        or "metric_whitelist" in builder.env
        or any("metric_whitelist" in patch for patch in builder.env.get("profile_overrides", {}).values())
    ):
        _bind_metric_dependencies(builder.metric_dependencies, execution)
    variant = dict(metadata=metadata, execution=execution)
    _bind_reports(config, variant, source, builder.steps)
    if config.reporting is not None:
        if metadata["kind"] != "workload":
            raise ScenarioError("reporting.time_axis requires a workload case")
        from reporting.timeline import validate_configuration

        try:
            variant["reporting"] = validate_configuration(config.reporting, {stage.id for stage in builder.steps})
        except ValueError as exc:
            raise ScenarioError(f"{source}.reporting: {exc}") from exc
    if metadata["kind"] == "workload":
        from monitoring.collection_plan import frozen_plan, select_plan
        from monitoring.query_plan import DEFAULT_PLAN, definitions, validate_export_filter
        from scenario.registry import load_metric_plan as load_plan
        from scenario.registry import view as view

        name = execution["monitoring"].get("query_plan", DEFAULT_PLAN)
        selected = select_plan(load_plan(name), builder.metric_dependencies, [view(report) for report in variant.get("reports", [])])
        for definition in definitions(selected).values():
            if definition.get("producer"):
                source = definition["producer"]
                metrics = registry().cases[config.case].definition.metrics
                if metrics is None or source not in metrics.CONTRACTS:
                    raise ScenarioError("metric source is not declared by case: " + source)
            collection = definition.get("collection", {})
            if collection.get("source"):
                registry().capability("source", collection["source"], case=config.case)
        validate_export_filter(selected, builder.env)
        variant["monitoring_query_plan"] = frozen_plan(name, selected, registry().sources)
    return variant


def _bind_metric_dependencies(requirements, execution):
    from monitoring.query_plan import DEFAULT_PLAN, definitions
    from scenario.registry import load_metric_plan as load_plan

    plan = load_plan(execution["monitoring"].get("query_plan", DEFAULT_PLAN))
    declared = definitions(plan)
    missing = set(requirements) - set(declared)
    if missing:
        raise ScenarioError("undeclared metric ids: " + ", ".join(sorted(missing)))
    for metric_id, requirement in requirements.items():
        definition = declared[metric_id]
        if (
            (requirement["unit"] is not None and definition["unit"] != requirement["unit"])
            or (requirement["mode"] is not None and definition.get("mode") != requirement["mode"])
            or not set(requirement["labels"]) <= set(definition["labels"])
        ):
            raise ScenarioError("metric dependency unit, mode or identity labels mismatch: " + metric_id)


def _bind_reports(config, variant, source, stages):
    # Resolve case views after program discovery, avoiding recursive case initialization.
    from reporting.events import validate_stage_sources
    from reporting.view_config import DEFAULT_VIEW
    from scenario.registry import registry, view
    from scenario.registry import report_declaration as declaration

    reports = list(config.reports) if config.reports is not None else None
    kind = variant["metadata"]["kind"]
    if reports is not None:
        variant["reports"] = declaration(reports, kind=kind, path=source + ".reports")
        for report_name in reports:
            if report_name != DEFAULT_VIEW:
                registry().capability("view", report_name, case=config.case)
            presentation = view(report_name)
            validate_stage_sources(presentation, {stage.id for stage in stages}, source + ".reports")
            required_plan = presentation.metrics.query_plan if presentation.metrics else None
            if required_plan and required_plan != variant["execution"]["monitoring"].get("query_plan"):
                raise ScenarioError(f"{source}.reports: {report_name} requires monitoring query plan {required_plan}")


def _implementation(config, module, name):
    program_path = module.path.resolve()
    case_sources = {
        str(path.relative_to(program_path.parent)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(program_path.parent.rglob("*.py"))
    }
    implementation = {
        "case_sources": case_sources,
        "case_sources_sha256": hashlib.sha256(json.dumps(case_sources, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "configuration": copy.deepcopy(config.declaration),
        "language": "python",
        "program": name,
        "path": str(program_path),
        "sha256": hashlib.sha256(program_path.read_bytes()).hexdigest(),
        "configuration_sha256": hashlib.sha256(
            json.dumps(config.declaration, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest(),
    }
    return implementation
