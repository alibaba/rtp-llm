"""Data-only guards and isolated merges shared by case configuration and variants."""

import copy
from scenario.loader import ScenarioError


def merge_data(base, patch):
    """Mappings inherit recursively; lists and scalars are replaced by YAML."""
    result = copy.deepcopy(base)
    for key, value in patch.items():
        if isinstance(result.get(key), dict) and isinstance(value, dict):
            result[key] = merge_data(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def mapping(value, allowed, path, *, required=()):
    from input_contract import mapping_fields
    try:
        return mapping_fields(value, allowed, path, required=required)
    except ValueError as exc:
        raise ScenarioError(str(exc)) from exc


def data_only(value, path):
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
            data_only(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            data_only(child, f"{path}[{index}]")


def leaf_paths(value, prefix=""):
    for key, child in value.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(child, dict) and child:
            yield from leaf_paths(child, path)
        else:
            yield path


def merge_environment(base, patch):
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


def path_in_scope(path, declared):
    return any(path == field or path.startswith(field + ".") for field in declared)
