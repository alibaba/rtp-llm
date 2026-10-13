"""Data-only case configuration. All control flow comes from registered Python programs."""

import copy
from dataclasses import asdict
import hashlib
import json
from monitoring.identity import NAME
from pathlib import Path

from document_io import DocumentError as ScenarioError
from scenario.config_data import merge_data
from scenario.config_data import leaf_paths
from scenario.config_data import merge_environment
from scenario.config_data import path_in_scope
from scenario.declarations import CaseDeclaration
from scenario.declarations import VariantDeclaration


from case_api.builder import CaseBuilder


def program_module(name, source):
    from case_api.registry import registry
    if not isinstance(name, str) or name not in registry().cases:
        raise ScenarioError(f"{source}: unknown registered Python case {name!r}")
    return registry().cases[name]


def configure_program(config, source):
    """Build an internal plan using an allowlisted Python entry point, without starting services."""
    config = CaseDeclaration.read(config, source)
    name = config.case
    module = program_module(name, source)
    if config.program != "default":
        raise ScenarioError(f"{source}: every case must declare program: default")
    document = _program_document(config, name, source)
    seen = set()
    numeric_contracts = {}
    default = VariantDeclaration.read({"id": "default"}, source)
    for row in (default, *config.variants):
        identity = row.identity
        if not isinstance(identity, str) or not NAME.fullmatch(identity) or identity in seen:
            raise ScenarioError(f"{source}: missing or duplicate configuration id {identity!r}")
        seen.add(identity)
        document["variants"].append(_build_variant(
            config, row, identity, module, config.profiles, source, numeric_contracts,
        ))
    document["implementation"] = _implementation(config, module, name)
    document["implementation"]["numeric_parameters"] = numeric_contracts
    return document


def _program_document(config, name, source):
    environment = config.environment
    metadata = config.metadata
    if not isinstance(metadata, dict):
        raise ScenarioError(f"{source}.metadata: expected mapping")
    document = dict({
        key: copy.deepcopy(value) for key, value in metadata.items() if key != "kind"
    })
    document.update(
        program_schema_version=1, environment=copy.deepcopy(environment), variants=[]
    )
    document["id"] = config.identity
    from scenario.execution_policy import EXECUTION_BUDGETS

    document["profiles"] = list(config.profiles)
    document["execution"] = {
        key: copy.deepcopy(value) for key, value in config.execution.to_dict().items()
        if key in EXECUTION_BUDGETS
    }
    selected_profiles = document.get("profiles")
    from scenario.validation import identifier, names
    from scenario.environment_config import environment as validate_environment
    from flexlb_cfg import PROFILES
    identifier(document["id"], source + ".id")
    if isinstance(selected_profiles, list):
        names(selected_profiles, source + ".profiles", PROFILES)
        for profile in selected_profiles:
            validate_environment(environment, source + ".environment", profile)
    return document


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
    if (
        not isinstance(profiles, list)
        or not profiles
        or any(not isinstance(p, str) for p in profiles)
    ):
        raise ScenarioError(
            f"{source}: profiles must be a nonempty string list in YAML"
        )
    variant_parameters = row.parameters
    patch = row.environment
    from case_api.numeric_parameters import parameter_rules
    from case_api.numeric_parameters import narrow_parameters
    defaults = parameter_rules(module.definition.numeric_parameters)
    base_rules = narrow_parameters(defaults, config.parameter_schema)
    builder = CaseBuilder(
        merge_environment(environment, patch),
        merge_data(parameters, variant_parameters),
        number_rules=narrow_parameters(base_rules, row.parameter_schema),
    )
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
    variant.update(
        id=identity, profiles=copy.deepcopy(profiles), stages=builder.finish()
    )
    if patch:
        variant["environment_overrides"] = copy.deepcopy(patch)
    return variant


def _variant_contract(config, builder, source):
    metadata = config.metadata
    execution = config.execution.to_dict()
    if (builder.metric_dependencies or "metric_whitelist" in builder.environment
            or any("metric_whitelist" in patch for patch in builder.environment.get("profile_overrides", {}).values())):
        _bind_metric_dependencies(builder.metric_dependencies, execution)
    variant = dict(metadata=metadata, execution=execution)
    _bind_reports(config, variant, source, builder.steps)
    if config.reporting is not None:
        if metadata["kind"] != "workload":
            raise ScenarioError("reporting.time_axis requires a workload case")
        from reporting.timeline import validate_configuration
        try:
            variant["reporting"] = validate_configuration(config.reporting, {stage["id"] for stage in builder.steps})
        except ValueError as exc:
            raise ScenarioError(f"{source}.reporting: {exc}") from exc
    if metadata["kind"] == "workload":
        from monitoring.query_plan import load_plan, DEFAULT_PLAN, validate_export_filter
        from monitoring.collection_plan import select_plan, frozen_plan
        from reporting.view_config import view
        name = execution["monitoring"].get("query_plan", DEFAULT_PLAN)
        selected = select_plan(load_plan(name), builder.metric_dependencies,
                               [view(report) for report in variant.get("reports", [])])
        from case_api.registry import registry
        from monitoring.query_plan import definitions
        for definition in definitions(selected).values():
            if definition.get("producer"):
                registry().capability("producer", definition["producer"], case=config.case)
            collection = definition.get("collection", {})
            if collection.get("source"):
                registry().capability("source", collection["source"], case=config.case)
        validate_export_filter(selected, builder.environment)
        variant["monitoring_query_plan"] = frozen_plan(name, selected)
    return variant


def _bind_metric_dependencies(requirements, execution):
    from monitoring.query_plan import DEFAULT_PLAN, load_plan, definitions
    plan = load_plan(execution["monitoring"].get("query_plan", DEFAULT_PLAN))
    declared = definitions(plan)
    missing = set(requirements) - set(declared)
    if missing:
        raise ScenarioError("undeclared metric ids: " + ", ".join(sorted(missing)))
    for metric_id, requirement in requirements.items():
        definition = declared[metric_id]
        if ((requirement["unit"] is not None and definition["unit"] != requirement["unit"])
                or (requirement["mode"] is not None and definition.get("mode") != requirement["mode"])
                or not set(requirement["labels"]) <= set(definition["labels"])):
            raise ScenarioError("metric dependency unit, mode or identity labels mismatch: " + metric_id)


def _bind_reports(config, variant, source, stages):
    from reporting.view_config import declaration

    reports = list(config.reports) if config.reports is not None else None
    kind = variant["metadata"]["kind"]
    if reports is not None:
        variant["reports"] = declaration(
            reports, kind=kind, path=source + ".reports"
        )
        from reporting.view_config import view

        for report_name in reports:
            from case_api.registry import registry
            from reporting.view_config import DEFAULT_VIEW
            if report_name != DEFAULT_VIEW:
                registry().capability("view", report_name, case=config.case)
            presentation = view(report_name)
            from reporting.events import validate_stage_sources
            validate_stage_sources(presentation, {stage["id"] for stage in stages}, source + ".reports")
            required_plan = presentation.metrics.query_plan if presentation.metrics else None
            if required_plan and required_plan != variant["execution"]["monitoring"].get("query_plan"):
                raise ScenarioError(
                    f"{source}.reports: {report_name} requires monitoring query plan {required_plan}"
                )


def _implementation(config, module, name):
    program_path = module.path.resolve()
    case_sources = {
        str(path.relative_to(program_path.parent)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(program_path.parent.rglob("*.py"))
    }
    implementation = {
        "case_sources": case_sources,
        "case_sources_sha256": hashlib.sha256(
            json.dumps(case_sources, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "configuration": copy.deepcopy(config.declaration),
        "language": "python",
        "program": name,
        "path": str(program_path),
        "sha256": hashlib.sha256(program_path.read_bytes()).hexdigest(),
        "configuration_sha256": hashlib.sha256(
            json.dumps(
                config.declaration, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        ).hexdigest(),
    }
    return implementation
