"""Data-only case configuration. All control flow comes from registered Python programs."""

import copy
import hashlib
import importlib
import json
import math
import re
from pathlib import Path

from scenario.loader import ScenarioError


class ProgramDocument(dict):
    """An internal Python plan, with provenance kept outside the plan vocabulary."""


def output(stage, field):
    """Refer to an earlier Python step's typed output; never evaluated as Python."""
    return {"$ref": f"stages.{stage}.output.{field}"}


class CaseBuilder:
    """Python owns ordering, branches and checks; YAML supplies declared parameters."""

    def __init__(self, environment, parameters, parameter_schema=None):
        self.environment = copy.deepcopy(environment)
        self.parameters = copy.deepcopy(parameters)
        self.parameter_schema = copy.deepcopy(parameter_schema or {})
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

    def metric(self, identity, *, unit=None, labels=(), mode=None):
        """Declare a numeric dependency; compilation binds it to the selected plan."""
        if type(identity) is not str or not re.fullmatch(r"[a-z][a-z0-9_]*/[a-z][a-z0-9_]*", identity):
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
        return _merge_data(values, dynamic)

    def number(self, name):
        value = self.value(name)
        rule = self.parameter_schema.get(name, {})
        integer = rule.get("integer", True)
        minimum, maximum = rule.get("minimum"), rule.get("maximum")
        if (
            type(value) not in ((int,) if integer else (int, float))
            or (isinstance(value, float) and not math.isfinite(value))
            or (minimum is not None and value < minimum)
            or (maximum is not None and value > maximum)
        ):
            raise ScenarioError(f"parameter {name!r}: invalid value {value!r}")
        return value

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


def _merge_data(base, patch):
    """Mappings inherit recursively; lists and scalars are replaced by YAML."""
    result = copy.deepcopy(base)
    for key, value in patch.items():
        if isinstance(result.get(key), dict) and isinstance(value, dict):
            result[key] = _merge_data(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _mapping(value, allowed, path):
    if not isinstance(value, dict):
        raise ScenarioError(f"{path}: expected mapping")
    extra = set(value) - set(allowed)
    if extra:
        raise ScenarioError(f"{path}: unknown configuration fields {sorted(extra)}")
    return value


def _data_only(value, path):
    if isinstance(value, dict):
        forbidden = set(value) & {
            "stages",
            "stage_overrides",
            "steps",
            "action",
            "$ref",
            "needs",
            "when",
        }
        if forbidden:
            raise ScenarioError(
                f"{path}: YAML cannot orchestrate steps ({sorted(forbidden)}); edit the Python case"
            )
        for key, child in value.items():
            _data_only(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _data_only(child, f"{path}[{index}]")


def _leaf_paths(value, prefix=""):
    for key, child in value.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(child, dict) and child:
            yield from _leaf_paths(child, path)
        else:
            yield path


def _merge_environment(base, patch):
    if not isinstance(base, dict) or not isinstance(patch, dict):
        raise ScenarioError("environment must be a mapping")
    result = {**copy.deepcopy(base), **copy.deepcopy(patch)}
    if "config_overrides" in patch:
        if not isinstance(base.get("config_overrides", {}), dict) or not isinstance(
            patch["config_overrides"], dict
        ):
            raise ScenarioError("config_overrides must be a mapping")
        result["config_overrides"] = {
            **copy.deepcopy(base.get("config_overrides", {})),
            **copy.deepcopy(patch["config_overrides"]),
        }
    return result


def program_module(name, source):
    from cases.registry import PROGRAMS

    if not isinstance(name, str) or name not in PROGRAMS:
        raise ScenarioError(f"{source}: unknown registered Python case {name!r}")
    return importlib.import_module(PROGRAMS[name])


def validate_analysis(config, source, *, module=None):
    """Resolve analysis capability from registered Python code, never YAML flags."""
    _data_only(config, source)
    if type(config.get("schema_version")) is not int or config["schema_version"] != 2:
        raise ScenarioError(f"{source}: analysis requires schema_version 2")
    module = module or program_module(config.get("case"), source)
    validator = getattr(module, "ANALYSIS_POLICY_VALIDATOR", None)
    if not callable(validator):
        raise ScenarioError(f"{source}: Python program does not support analysis")
    try:
        return validator(config["analysis"])
    except ValueError as exc:
        raise ScenarioError(f"{source}.analysis: {exc}") from exc


def configure_program(config, source):
    """Build an internal plan using an allowlisted Python entry point, without I/O."""
    _data_only(config, source)
    _mapping(
        config,
        {
            "schema_version",
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
            "analysis",
            "test",
            "reports",
        },
        source,
    )
    if type(config.get("schema_version")) is not int or config["schema_version"] != 2:
        raise ScenarioError(
            f"{source}: only data-only schema_version 2 is accepted; move orchestration into Python"
        )
    name = config.get("case")
    module = program_module(name, source)
    if "analysis" in config:
        validate_analysis(config, source, module=module)
    environment = config.get("environment", {})
    parameters = config.get("parameters", {})
    if not isinstance(parameters, dict):
        raise ScenarioError(f"{source}.parameters: expected mapping")
    if config.get("program") != "default":
        raise ScenarioError(f"{source}: every case must declare program: default")
    variants = config.get("variants", [])
    if "variants" in config and (not isinstance(variants, list) or not variants):
        raise ScenarioError(f"{source}.variants: expected a nonempty list")
    axis = config.get("variant_axis")
    if "variants" in config:
        _mapping(axis, {"kind", "fields"}, source + ".variant_axis")
        if axis.get("kind") not in {"data", "scale", "flow"}:
            raise ScenarioError(f"{source}: variant_axis.kind must be data, scale or flow")
        axis_fields = axis.get("fields", [])
        if (not isinstance(axis_fields, list) or not axis_fields
                or any(not isinstance(f, str) for f in axis_fields)
                or len(set(axis_fields)) != len(axis_fields)):
            raise ScenarioError(f"{source}: variant_axis.fields must be unique nonempty paths")
        for field in axis_fields:
            if not re.fullmatch(r"[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)*", field):
                raise ScenarioError(f"{source}: invalid variant field path {field!r}")
            if any(other != field and field.startswith(other + ".") for other in axis_fields):
                raise ScenarioError(f"{source}: overlapping variant field paths")
            valid = (field == "program" if axis["kind"] == "flow" else
                     field.startswith("parameters.") if axis["kind"] == "data" else
                     field in {"environment.n_prefill", "environment.n_decode",
                               "environment.prefill_cache_blocks", "environment.decode_cache_blocks"})
            if not valid:
                raise ScenarioError(f"{source}: field {field!r} does not belong to declared variant dimension")
    elif axis is not None:
        raise ScenarioError(f"{source}: variant_axis requires variants")
    metadata = config.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ScenarioError(f"{source}.metadata: expected mapping")
    document = ProgramDocument(copy.deepcopy(metadata))
    document.update(
        schema_version=1, environment=copy.deepcopy(environment), variants=[]
    )
    document["id"] = config.get("id", name)
    for key in ("profiles", "execution"):
        if key in config:
            document[key] = copy.deepcopy(config[key])
    selected_profiles = document.get("profiles")
    from scenario.compiler import identifier, environment as validate_environment, names
    from flexlb_cfg import PROFILES
    identifier(document["id"], source + ".id")
    if isinstance(selected_profiles, list):
        names(selected_profiles, source + ".profiles", PROFILES)
        for profile in selected_profiles:
            validate_environment(environment, source + ".environment", profile)
    seen = set()
    # The root program is always built first; variants only append isolated points.
    for index, row in enumerate([{"id": "default"}, *variants]):
        _mapping(
            row,
            {
                "id",
                "program",
                "profiles",
                "environment",
                "execution",
                "parameters",
                "metadata",
                "parameter_schema",
                "test",
                "reports",
            },
            source + ".variants",
        )
        if index and axis is not None:
            for field in _leaf_paths({k: v for k, v in row.items() if k != "id"}):
                if not any(field == f or field.startswith(f + ".") for f in axis_fields):
                    raise ScenarioError(f"{source}: variant field {field!r} outside declared dimension")
        identity = row.get("id")
        if not isinstance(identity, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", identity) or identity in seen:
            raise ScenarioError(
                f"{source}: missing or duplicate configuration id {identity!r}"
            )
        seen.add(identity)
        program = row.get("program", config.get("program"))
        if index and axis is not None and axis["kind"] == "flow":
            if program != identity or program not in getattr(module, "FLOW_PROGRAMS", ()):
                raise ScenarioError(f"{source}: flow identity must equal a registered flow program")
        build = vars(module).get(program) if isinstance(program, str) else None
        if (
            not isinstance(program, str)
            or program.startswith("_")
            or not callable(build)
            or getattr(build, "__module__", None) != module.__name__
        ):
            raise ScenarioError(f"{source}: unknown Python case program {program!r}")
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
        builder = CaseBuilder(
            _merge_environment(environment, patch),
            _merge_data(parameters, variant_parameters),
            config.get("parameter_schema", {}),
        )
        build(builder)
        for field in _leaf_paths(variant_parameters):
            if not any(field == used or field.startswith(used + ".") for used in builder.read_parameters):
                raise ScenarioError(f"{source}: unused variant parameter {field!r}")
        variant = {}
        from scenario.suites import normalize_test

        if not isinstance(config.get("test", {}), dict):
            raise ScenarioError(f"{source}.test: expected mapping")
        variant["test"] = normalize_test(copy.deepcopy(config.get("test", {})))
        if builder.metric_dependencies:
            from monitoring.query_plan import load_plan, definitions
            declared = definitions(load_plan(variant["test"]["monitoring"].get("query_plan", "workload.yaml")))
            missing = set(builder.metric_dependencies) - set(declared)
            if missing:
                raise ScenarioError("undeclared metric ids: " + ", ".join(sorted(missing)))
            for metric_id, requirement in builder.metric_dependencies.items():
                definition = declared[metric_id]
                if ((requirement["unit"] is not None and definition["unit"] != requirement["unit"])
                        or (requirement["mode"] is not None and definition.get("mode") != requirement["mode"])
                        or not set(requirement["labels"]) <= set(definition["labels"])):
                    raise ScenarioError("metric dependency unit, mode or identity labels mismatch: " + metric_id)

        from reporting.view_config import declaration

        reports = config.get("reports") if variant["test"]["kind"] == "workload" else None
        if reports is not None:
            variant["test"]["reports"] = declaration(
                reports, kind=variant["test"]["kind"], path=source + ".reports"
            )
            from reporting.view_config import view

            for report_name in reports:
                required_plan = view(report_name).get("monitoring_query_plan")
                if required_plan and required_plan != variant["test"]["monitoring"].get("query_plan"):
                    raise ScenarioError(
                        f"{source}.reports: {report_name} requires monitoring query plan {required_plan}"
                    )
        variant.update(
            id=identity, profiles=copy.deepcopy(profiles), stages=builder.finish()
        )
        if patch:
            variant["environment_overrides"] = copy.deepcopy(patch)
        document["variants"].append(variant)
    program_path = Path(module.__file__).resolve()
    document.implementation = {
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
    query_plan = config.get("test", {}).get("monitoring", {}).get("query_plan")
    if query_plan is not None:
        from monitoring.query_plan import load_plan, plan_hash

        metric_plan = load_plan(query_plan)
        document.implementation["monitoring_query_plan"] = {
            "name": query_plan,
            "sha256": plan_hash(metric_plan),
            "definition": metric_plan,
        }
    return document
