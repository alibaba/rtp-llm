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
    for name in ("config_overrides", "profile_overrides"):
        if name not in patch:
            continue
        previous, changed = base.get(name, {}), patch[name]
        if not isinstance(previous, dict) or not isinstance(changed, dict):
            raise ScenarioError(name + " must be a mapping")
        if name == "config_overrides":
            # Each configuration field is atomic, including omit and model objects.
            result[name] = {**copy.deepcopy(previous), **copy.deepcopy(changed)}
        else:
            profiles = copy.deepcopy(previous)
            for profile, overrides in changed.items():
                if not isinstance(overrides, dict) or not isinstance(profiles.get(profile, {}), dict):
                    raise ScenarioError("profile_overrides entries must be mappings")
                profiles[profile] = {**profiles.get(profile, {}), **copy.deepcopy(overrides)}
            result[name] = profiles
    return result


def path_in_scope(path, declared):
    return any(path == field or path.startswith(field + ".") for field in declared)
