"""Shared strict JSON/YAML input, independent of scenario compilation."""

import json
import math
from pathlib import Path


class DocumentError(ValueError):
    pass


def strict_pairs(pairs):
    result = {}
    for key, value in pairs:
        if not isinstance(key, str):
            raise DocumentError("mapping keys must be strings")
        if key in result:
            raise DocumentError(f"duplicate key {key!r}")
        result[key] = value
    return result


def validate_data_tree(value, depth=0):
    if depth > 32:
        raise DocumentError("document exceeds maximum depth 32")
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise DocumentError("mapping keys must be strings")
            validate_data_tree(child, depth + 1)
    elif isinstance(value, list):
        for child in value:
            validate_data_tree(child, depth + 1)
    elif isinstance(value, float) and not math.isfinite(value):
        raise DocumentError("NaN and Infinity are not allowed")
    elif value is not None and type(value) not in (str, int, float, bool):
        raise DocumentError(f"unsupported value type {type(value).__name__}")


def load_document(path):
    path = Path(path)
    try:
        with path.open("rb") as stream:
            raw = stream.read(1024 * 1024 + 1)
        if len(raw) > 1024 * 1024:
            raise DocumentError("document exceeds 1 MiB")
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
                    raise DocumentError(
                        f"anchors, aliases and tags are not supported at line {token.start_mark.line + 1}"
                    )
            doc = yaml.load(text, Loader=Loader)
        else:
            raise DocumentError("expected .json, .yaml or .yml")
        validate_data_tree(doc)
        if not isinstance(doc, dict):
            raise DocumentError("document root must be a mapping")
        return doc
    except (OSError, UnicodeError, ValueError, RecursionError) as exc:
        raise DocumentError(f"{path}: {exc}") from exc
    except ImportError as exc:
        raise DocumentError(
            f"{path}: YAML input requires PyYAML; JSON needs no optional parser"
        ) from exc
    except Exception as exc:
        # PyYAML parser errors carry line/column; preserve them for diagnostics.
        if type(exc).__module__.startswith("yaml"):
            raise DocumentError(f"{path}: {exc}") from exc
        raise

