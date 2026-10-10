"""Data-only case configuration. All control flow comes from registered Python programs."""

import copy
from dataclasses import asdict
import hashlib
import importlib
import json
from monitoring.identity import METRIC_ID, NAME
from pathlib import Path

from scenario.loader import ScenarioError
from cases.config_data import (
    merge_data, mapping, data_only, leaf_paths, merge_environment, path_in_scope,
)
from cases.variants import VariantAxis


class ProgramDocument(dict):
    """An internal Python plan, with provenance kept outside the plan vocabulary."""


def output(stage, field):
    """Refer to an earlier Python step's typed output; never evaluated as Python."""
    return {"$ref": f"stages.{stage}.output.{field}"}


class CaseBuilder:
    """Python owns ordering, branches and checks; YAML supplies declared parameters."""

    def __init__(self, environment, parameters, *, number_rules):
        self.environment = copy.deepcopy(environment)
        self.parameters = copy.deepcopy(parameters)
        self.number_rules = dict(number_rules)
        self.steps = []
        self.read_parameters = set()
        self.metric_dependencies = {}

    def value(self, path):
        """Read required YAML data; programs provide no hidden fallback values."""
        self.read_parameters.add(path)
        value = self.parameters
        for part in path.split("."):
            if not isinstance(value, dict) or part not in value:
                raise ScenarioError(f"missing YAML parameter {path!r}")
            value = value[part]
        return copy.deepcopy(value)

    def inputs(self, **contract):
        from cases.inputs import ProgramInputs

        return ProgramInputs.read(self, **contract)

    def metric(self, identity, *, unit=None, labels=(), mode=None):
        """Declare a numeric dependency; compilation binds it to the selected plan."""
        if type(identity) is not str or not METRIC_ID.fullmatch(identity):
            raise ScenarioError("invalid metric id: " + str(identity))
        requirement = dict(unit=unit, labels=tuple(labels), mode=mode)
        if identity in self.metric_dependencies and self.metric_dependencies[identity] != requirement:
            raise ScenarioError("conflicting metric dependency: " + identity)
        self.metric_dependencies[identity] = requirement
        return identity

    def params(self, path, dynamic):
        values = self.value(path)
        if not isinstance(values, dict):
            raise ScenarioError(f"YAML parameter {path!r} must be a mapping")
        return merge_data(values, dynamic)

    def number(self, name):
        if name not in self.number_rules:
            raise ScenarioError("undeclared numeric parameter: " + name)
        try:
            return self.number_rules[name].validate(self.value(name), name)
        except ValueError as exc:
            raise ScenarioError(str(exc)) from exc

    def validate_numbers(self):
        """Apply program types and YAML narrowing without consuming business inputs."""
        reads = set(self.read_parameters)
        try:
            for name in self.number_rules:
                self.number(name)
        finally:
            self.read_parameters = reads

    def step(self, name, action, *, params=None, timeout_s=None):
        step = {"id": name, "action": action}
        if params is not None:
            step["params"] = copy.deepcopy(params)
        if timeout_s is not None:
            step["timeout_s"] = timeout_s
        self.steps.append(step)

    def observe(self, name, action, *, params=None, timeout_s=None):
        """Independent observation: a FAIL may continue in the workload executor.

        Do not use for prerequisites, injections or checks authorizing a mutation.
        ERROR/TIMEOUT still abort dependent execution in either test class.
        """
        self.step(name, action, params=params, timeout_s=timeout_s)
        self.steps[-1]["purpose"] = "observation"

    def finish(self):
        return copy.deepcopy(self.steps)


def program_module(name, source):
    from cases.registry import PROGRAMS

    if not isinstance(name, str) or name not in PROGRAMS:
        raise ScenarioError(f"{source}: unknown registered Python case {name!r}")
    return importlib.import_module(PROGRAMS[name])


def configure_program(config, source):
    """Build an internal plan using an allowlisted Python entry point, without I/O."""
    _validate_configuration(config, source)
    name = config.get("case")
    module = program_module(name, source)
    parameters = config.get("parameters", {})
    if not isinstance(parameters, dict):
        raise ScenarioError(f"{source}.parameters: expected mapping")
    if config.get("program") != "default":
        raise ScenarioError(f"{source}: every case must declare program: default")
    axis = VariantAxis.from_config(config, source)
    document = _program_document(config, name, source)
    seen = set()
    numeric_contracts = {}
    for index, row in enumerate([{"id": "default"}, *config.get("variants", [])]):
        mapping(row, {
            "id", "program", "profiles", "environment", "execution", "parameters",
            "metadata", "parameter_schema", "reports",
        }, source + ".variants")
        if index and axis is not None:
            axis.validate_patch(row, source)
        identity = row.get("id")
        if not isinstance(identity, str) or not NAME.fullmatch(identity) or identity in seen:
            raise ScenarioError(f"{source}: missing or duplicate configuration id {identity!r}")
        seen.add(identity)
        document["variants"].append(_build_variant(
            config, row, identity, module, axis if index else None, document.get("profiles"), source, numeric_contracts,
        ))
    document.implementation = _implementation(config, module, name)
    document.implementation["numeric_parameters"] = numeric_contracts
    return document


def _validate_configuration(config, source):
    data_only(config, source)
    mapping(
        config,
        {
            "case_schema_version",
            "case",
            "program",
            "variant_axis",
            "id",
            "profiles",
            "environment",
            "execution",
            "parameters",
            "variants",
            "metadata",
            "parameter_schema",
            "reports",
        },
        source,
    )
    from scenario.suites import normalize_metadata, normalize_execution

    metadata = normalize_metadata(config.get("metadata"))
    normalize_execution(config.get("execution"), kind=metadata["kind"])
    if type(config.get("case_schema_version")) is not int or config["case_schema_version"] != 2:
        raise ScenarioError(
            f"{source}: only data-only case_schema_version 2 is accepted; move orchestration into Python"
        )


def _program_document(config, name, source):
    environment = config.get("environment", {})
    metadata = config.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ScenarioError(f"{source}.metadata: expected mapping")
    document = ProgramDocument({
        key: copy.deepcopy(value) for key, value in metadata.items() if key != "kind"
    })
    document.update(
        program_schema_version=1, environment=copy.deepcopy(environment), variants=[]
    )
    document["id"] = config.get("id", name)
    from scenario.suites import EXECUTION_BUDGETS

    if "profiles" in config:
        document["profiles"] = copy.deepcopy(config["profiles"])
    document["execution"] = {
        key: copy.deepcopy(value) for key, value in config["execution"].items()
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
    build = vars(module).get(program) if isinstance(program, str) else None
    if (
        not isinstance(program, str)
        or program.startswith("_")
        or not callable(build)
        or getattr(build, "__module__", None) != module.__name__
    ):
        raise ScenarioError(f"{source}: unknown Python case program {program!r}")
    return build


def _build_variant(config, row, identity, module, axis, selected_profiles, source, numeric_contracts):
    environment = config.get("environment", {})
    parameters = config.get("parameters", {})
    program = row.get("program", config.get("program"))
    if axis is not None:
        axis.validate_flow(program, identity, module, source)
    build = _select_program(module, program, source)
    profiles = selected_profiles
    if (
        not isinstance(profiles, list)
        or not profiles
        or any(not isinstance(p, str) for p in profiles)
    ):
        raise ScenarioError(
            f"{source}: profiles must be a nonempty string list in YAML"
        )
    variant_parameters = row.get("parameters", {})
    if not isinstance(variant_parameters, dict):
        raise ScenarioError(f"{source}: variant parameters must be a mapping")
    patch = row.get("environment", {})
    from cases.numeric_parameters import parameter_rules, narrow_parameters
    defaults = parameter_rules(getattr(module, "NUMERIC_PARAMETERS", {}))
    base_rules = narrow_parameters(defaults, config.get("parameter_schema", {}))
    builder = CaseBuilder(
        merge_environment(environment, patch),
        merge_data(parameters, variant_parameters),
        number_rules=narrow_parameters(base_rules, row.get("parameter_schema", {})),
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
    from scenario.suites import normalize_metadata, normalize_execution

    metadata = normalize_metadata(config.get("metadata"))
    execution = normalize_execution(config.get("execution"), kind=metadata["kind"])
    if (builder.metric_dependencies or "metric_whitelist" in builder.environment
            or any("metric_whitelist" in patch for patch in builder.environment.get("profile_overrides", {}).values())):
        _bind_metric_dependencies(builder.metric_dependencies, execution)
    variant = dict(metadata=metadata, execution=execution)
    _bind_reports(config, variant, source, builder.steps)
    if metadata["kind"] == "workload":
        from monitoring.query_plan import load_plan, DEFAULT_PLAN, validate_export_filter
        from monitoring.collection_plan import select_plan, frozen_plan
        from reporting.view_config import view
        name = execution["monitoring"].get("query_plan", DEFAULT_PLAN)
        selected = select_plan(load_plan(name), builder.metric_dependencies,
                               [view(report) for report in variant.get("reports", [])])
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

    reports = config.get("reports")
    kind = variant["metadata"]["kind"]
    if reports is not None:
        variant["reports"] = declaration(
            reports, kind=kind, path=source + ".reports"
        )
        from reporting.view_config import view

        for report_name in reports:
            presentation = view(report_name)
            from reporting.events import validate_stage_sources
            validate_stage_sources(presentation, {stage["id"] for stage in stages}, source + ".reports")
            required_plan = presentation.get("metrics", {}).get("query_plan")
            if required_plan and required_plan != variant["execution"]["monitoring"].get("query_plan"):
                raise ScenarioError(
                    f"{source}.reports: {report_name} requires monitoring query plan {required_plan}"
                )


def _implementation(config, module, name):
    program_path = Path(module.__file__).resolve()
    implementation = {
        "configuration": copy.deepcopy(config),
        "language": "python",
        "program": name,
        "path": str(program_path),
        "sha256": hashlib.sha256(program_path.read_bytes()).hexdigest(),
        "configuration_sha256": hashlib.sha256(
            json.dumps(
                config, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        ).hexdigest(),
    }
    return implementation
