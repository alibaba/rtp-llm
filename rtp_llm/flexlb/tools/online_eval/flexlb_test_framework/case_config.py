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

    def __init__(self, environment, parameters):
        self.environment = copy.deepcopy(environment)
        self.parameters = copy.deepcopy(parameters)
        self.steps = []
        self._used = set()

    def number(self, name, default, *, minimum=1, maximum=None, integer=True):
        self._used.add(name)
        value = self.parameters.get(name, default)
        if (
            type(value) not in ((int,) if integer else (int, float))
            or (isinstance(value, float) and not math.isfinite(value))
            or value < minimum
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

    def finish(self):
        unused = set(self.parameters) - self._used
        if unused:
            raise ScenarioError(
                f"Python case does not declare parameters {sorted(unused)}"
            )
        return copy.deepcopy(self.steps)


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
    variants = config.get("variants", [{"id": key} for key in module.VARIANTS])
    if not isinstance(variants, list) or not variants:
        raise ScenarioError(f"{source}.variants: expected a nonempty list")
    document = ProgramDocument(copy.deepcopy(module.METADATA))
    document.update(
        schema_version=1, environment=copy.deepcopy(environment), variants=[]
    )
    document["id"] = config.get("id", name)
    for key in ("profiles", "execution"):
        if key in config:
            document[key] = copy.deepcopy(config[key])
    selected_profiles = document.get("profiles", module.PROFILES)
    seen = set()
    for row in variants:
        _mapping(
            row,
            {"id", "use", "profiles", "environment", "execution", "parameters"},
            source + ".variants",
        )
        identity = row.get("id")
        if not isinstance(identity, str) or identity in seen:
            raise ScenarioError(
                f"{source}: missing or duplicate configuration id {identity!r}"
            )
        seen.add(identity)
        use = row.get("use", identity)
        if not isinstance(use, str) or use not in module.VARIANTS:
            raise ScenarioError(f"{source}: unknown Python case variant {use!r}")
        definition = module.VARIANTS[use]
        profiles = row.get(
            "profiles",
            selected_profiles if "profiles" in config else definition["profiles"],
        )
        if not isinstance(profiles, list) or any(
            not isinstance(p, str) for p in profiles
        ):
            raise ScenarioError(f"{source}: profiles must be a string list")
        if set(profiles) - set(definition["profiles"]):
            raise ScenarioError(
                f"{source}: variant {use!r} supports only {definition['profiles']}"
            )
        variant_parameters = row.get("parameters", {})
        if not isinstance(variant_parameters, dict):
            raise ScenarioError(f"{source}: variant parameters must be a mapping")
        patch = row.get("environment", {})
        builder = CaseBuilder(
            _merge_environment(environment, patch), {**parameters, **variant_parameters}
        )
        definition["build"](builder)
        variant = copy.deepcopy(definition["metadata"])
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
