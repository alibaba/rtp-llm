"""Data-only case configuration. All control flow comes from registered Python programs."""

import copy
import hashlib
import importlib
import json
import math
from pathlib import Path

from .scenario.loader import ScenarioError


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

    def value(self, path):
        """Read required YAML data; programs provide no hidden fallback values."""
        value = self.parameters
        for part in path.split("."):
            if not isinstance(value, dict) or part not in value:
                raise ScenarioError(f"missing YAML parameter {path!r}")
            value = value[part]
        return copy.deepcopy(value)

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


def configure_program(config, source):
    """Build an internal plan using an allowlisted Python entry point, without I/O."""
    from .case_programs import PROGRAMS

    _data_only(config, source)
    _mapping(
        config,
        {
            "schema_version",
            "case",
            "id",
            "profiles",
            "environment",
            "execution",
            "parameters",
            "variants",
            "metadata",
            "parameter_schema",
        },
        source,
    )
    if type(config.get("schema_version")) is not int or config["schema_version"] != 2:
        raise ScenarioError(
            f"{source}: only data-only schema_version 2 is accepted; move orchestration into Python"
        )
    name = config.get("case")
    if not isinstance(name, str) or name not in PROGRAMS:
        raise ScenarioError(f"{source}: unknown registered Python case {name!r}")
    # The module path is code-owned. Configuration cannot import arbitrary modules.
    module = importlib.import_module(PROGRAMS[name])
    environment = config.get("environment", {})
    parameters = config.get("parameters", {})
    if not isinstance(parameters, dict):
        raise ScenarioError(f"{source}.parameters: expected mapping")
    variants = config.get("variants")
    if not isinstance(variants, list) or not variants:
        raise ScenarioError(f"{source}.variants: expected a nonempty list")
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
    seen = set()
    for row in variants:
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
            },
            source + ".variants",
        )
        identity = row.get("id")
        if not isinstance(identity, str) or identity in seen:
            raise ScenarioError(
                f"{source}: missing or duplicate configuration id {identity!r}"
            )
        seen.add(identity)
        program = row.get("program", identity)
        build = vars(module).get(program) if isinstance(program, str) else None
        if (
            not isinstance(program, str)
            or program.startswith("_")
            or not callable(build)
            or getattr(build, "__module__", None) != module.__name__
        ):
            raise ScenarioError(f"{source}: unknown Python case program {program!r}")
        profiles = row.get("profiles", selected_profiles)
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
            _merge_data(
                config.get("parameter_schema", {}), row.get("parameter_schema", {})
            ),
        )
        build(builder)
        variant = copy.deepcopy(row.get("metadata", {}))
        variant.update(
            id=identity, profiles=copy.deepcopy(profiles), stages=builder.finish()
        )
        if patch:
            variant["environment_overrides"] = copy.deepcopy(patch)
        if "execution" in row:
            variant["execution"] = copy.deepcopy(row["execution"])
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
    return document
