"""Strict JSON/YAML input with duplicate-key and structural validation."""

import json
import math
from pathlib import Path


class ScenarioError(ValueError):
    pass


def strict_pairs(pairs):
    result = {}
    for key, value in pairs:
        if not isinstance(key, str):
            raise ScenarioError("mapping keys must be strings")
        if key in result:
            raise ScenarioError(f"duplicate key {key!r}")
        result[key] = value
    return result


def validate_data_tree(value, depth=0):
    if depth > 32:
        raise ScenarioError("document exceeds maximum depth 32")
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ScenarioError("mapping keys must be strings")
            validate_data_tree(child, depth + 1)
    elif isinstance(value, list):
        for child in value:
            validate_data_tree(child, depth + 1)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ScenarioError("NaN and Infinity are not allowed")
    elif value is not None and type(value) not in (str, int, float, bool):
        raise ScenarioError(f"unsupported value type {type(value).__name__}")


def load_document(path):
    path = Path(path)
    try:
        with path.open("rb") as stream:
            raw = stream.read(1024 * 1024 + 1)
        if len(raw) > 1024 * 1024:
            raise ScenarioError("document exceeds 1 MiB")
        text = raw.decode("utf-8")
        if path.suffix.lower() == ".json":
            doc = json.loads(text, object_pairs_hook=strict_pairs)
        elif path.suffix.lower() in (".yaml", ".yml"):
            import yaml

            class Loader(yaml.SafeLoader):
                def construct_mapping(self, node, deep=False):
                    return strict_pairs(
                        (
                            self.construct_object(k, deep=deep),
                            self.construct_object(v, deep=deep),
                        )
                        for k, v in node.value
                    )

            for token in yaml.scan(text):
                if isinstance(
                    token,
                    (
                        yaml.tokens.AnchorToken,
                        yaml.tokens.AliasToken,
                        yaml.tokens.TagToken,
                    ),
                ):
                    raise ScenarioError(
                        f"anchors, aliases and tags are not supported at line {token.start_mark.line + 1}"
                    )
            doc = yaml.load(text, Loader=Loader)
        else:
            raise ScenarioError("expected .json, .yaml or .yml")
        validate_data_tree(doc)
        if not isinstance(doc, dict):
            raise ScenarioError("document root must be a mapping")
        return doc
    except (OSError, UnicodeError, ValueError, RecursionError) as exc:
        raise ScenarioError(f"{path}: {exc}") from exc
    except ImportError as exc:
        raise ScenarioError(
            f"{path}: YAML input requires PyYAML; JSON needs no optional parser"
        ) from exc
    except Exception as exc:
        # PyYAML parser errors carry line/column; preserve them for diagnostics.
        if type(exc).__module__.startswith("yaml"):
            raise ScenarioError(f"{path}: {exc}") from exc
        raise


def load_scenarios(root, *, experiment_params=None):
    """Load data-only configurations and invoke their registered Python builders."""
    from scenario.program_compiler import configure_program

    from scenario.experiment_params import ExperimentParams
    overrides = ExperimentParams.read(experiment_params)
    root = Path(root).resolve()
    if root.is_file():
        raw = load_document(root)
        overrides.require_cases([raw.get('case')])
        return [(str(root), configure_program(overrides.apply(raw), str(root)))]
    if not root.is_dir():
        raise ScenarioError(f"scenario root does not exist: {root}")
    result = []
    cases = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in (".json", ".yaml", ".yml") or not path.is_file():
            continue
        if root not in path.resolve().parents:
            raise ScenarioError(f"{path}: scenario path escapes root")
        raw = load_document(path)
        cases.append(raw.get("case"))
        result.append((str(path), configure_program(overrides.apply(raw), str(path))))
    if not result:
        raise ScenarioError(f"no scenario definitions in {root}")
    overrides.require_cases(cases)
    return result
