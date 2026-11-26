import json
import os
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Tuple,
    TypeAlias,
    cast,
    get_args,
)

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException

GrammarFieldName: TypeAlias = Literal[
    "json_schema",
    "regex",
    "ebnf",
    "structural_tag",
]

# Derive runtime iteration from the type definition so the field list has one source.
GRAMMAR_FIELD_NAMES = cast(
    Tuple[GrammarFieldName, ...],
    get_args(GrammarFieldName),
)

SCHEMA_MAP_KEYWORDS = (
    "properties",
    "patternProperties",
    "definitions",
    "$defs",
    "dependencies",
    "dependentSchemas",
)
_SCHEMA_ARRAY_KEYWORDS = {"allOf", "anyOf", "oneOf", "prefixItems", "items"}
_SCHEMA_SINGLE_KEYWORDS = {
    "items",
    "additionalItems",
    "contains",
    "additionalProperties",
    "propertyNames",
    "not",
    "if",
    "then",
    "else",
    "unevaluatedProperties",
    "unevaluatedItems",
}
# Draft 7 allows either a schema or an array of schemas for items.
SCHEMA_VALUE_KEYWORDS = _SCHEMA_SINGLE_KEYWORDS | _SCHEMA_ARRAY_KEYWORDS

STRIP_STRING_PATTERNS = os.getenv(
    "DS_LLM_GRAMMAR_STRIP_STRING_PATTERNS", "0"
).strip().lower() in ("1", "true", "on", "yes")


def load_json_field(name: str, value: Any) -> Any:
    if not isinstance(value, (str, bytes, bytearray)):
        return value
    try:
        return json.loads(value)
    except RecursionError as e:
        raise FtRuntimeException(
            ExceptionType.ERROR_INPUT_FORMAT_ERROR,
            f"{name} exceeds the supported JSON nesting depth",
        ) from e
    except (json.JSONDecodeError, TypeError, UnicodeDecodeError) as e:
        raise FtRuntimeException(
            ExceptionType.ERROR_INPUT_FORMAT_ERROR,
            f"{name} must be valid JSON: {str(e)}",
        ) from e


def parse_json_grammar_value(name: GrammarFieldName, value: Any) -> Any:
    """Parse one JSON-valued grammar field into its canonical Python type."""
    if value is None:
        return None

    value = load_json_field(name, value)
    if name == "json_schema":
        if isinstance(value, (dict, bool)):
            return value
        expected = "a JSON object or boolean"
    elif name == "structural_tag":
        if isinstance(value, dict):
            return value
        expected = "a JSON object"
    else:
        raise ValueError(f"{name} is not a JSON-valued grammar field")

    raise FtRuntimeException(
        ExceptionType.ERROR_INPUT_FORMAT_ERROR,
        f"{name} must be {expected}",
    )


def normalize_grammar_value(name: GrammarFieldName, value: Any) -> Any:
    if name in ("json_schema", "structural_tag"):
        value = parse_json_grammar_value(name, value)
    if name == "structural_tag":
        if (
            isinstance(value, dict)
            and "type" not in value
            and ("format" in value or ("structures" in value and "triggers" in value))
        ):
            value = {"type": "structural_tag", **value}
    elif name in ("regex", "ebnf") and not isinstance(value, str):
        raise FtRuntimeException(
            ExceptionType.ERROR_INPUT_FORMAT_ERROR,
            f"{name} must be a string",
        )
    return value


def _has_regex_lookahead(pattern: str) -> bool:
    """Match xgrammar's escape and character-class boundaries."""
    if "(?" not in pattern:
        return False
    escaped = False
    in_character_class = False
    for index, char in enumerate(pattern):
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == "[":
            in_character_class = True
        elif char == "]":
            in_character_class = False
        elif not in_character_class and pattern.startswith(("(?=", "(?!"), index):
            return True
    return False


def _grammar_schema_roots(kind: str, value: dict) -> Iterator[Any]:
    if kind == "json_schema":
        yield value
        return

    structures = value.get("structures")
    if isinstance(structures, list):
        for structure in structures:
            if isinstance(structure, dict):
                yield structure.get("schema")

    pending = [value.get("format")]
    seen = set()
    while pending:
        fmt = pending.pop()
        if not isinstance(fmt, dict) or id(fmt) in seen:
            continue
        seen.add(id(fmt))
        kind = fmt.get("type")
        if kind in ("json_schema", "qwen_xml_parameter"):
            yield fmt.get("json_schema")
        elif kind in ("tag", "optional", "plus", "star", "repeat"):
            pending.append(fmt.get("content"))
        elif kind in (
            "sequence",
            "or",
            "triggered_tags",
            "token_triggered_tags",
            "tags_with_separator",
        ):
            children = fmt.get("elements" if kind in ("sequence", "or") else "tags")
            if isinstance(children, list):
                pending.extend(children)
        elif kind in ("dispatch", "token_dispatch"):
            rules = fmt.get("rules")
            if isinstance(rules, list):
                for rule in rules:
                    if isinstance(rule, (list, tuple)) and len(rule) == 2:
                        pending.append(rule[1])


def _string_patterns_to_strip(kind: str, value: dict) -> Iterator[dict]:
    pending = list(_grammar_schema_roots(kind, value))
    seen = set()
    while pending:
        node = pending.pop()
        if not isinstance(node, dict) or id(node) in seen:
            continue
        seen.add(id(node))
        pattern = node.get("pattern")
        if node.get("type") == "string" and isinstance(pattern, str):
            if STRIP_STRING_PATTERNS or _has_regex_lookahead(pattern):
                yield node

        for key, child in node.items():
            if key in SCHEMA_MAP_KEYWORDS and isinstance(child, dict):
                pending.extend(child.values())
            elif key in _SCHEMA_ARRAY_KEYWORDS and isinstance(child, list):
                pending.extend(child)
            elif key in _SCHEMA_SINGLE_KEYWORDS or key == "contentSchema":
                pending.append(child)


def without_string_patterns(kind: str, value: Any) -> Any:
    """Strip lookahead patterns, or every explicit string pattern when configured."""
    if kind not in ("json_schema", "structural_tag") or not isinstance(value, dict):
        return value
    if not any(_string_patterns_to_strip(kind, value)):
        return value
    # A Python caller can reuse one dict as a subschema and as an enum literal.
    # Materialize independent JSON values before editing just the schema positions.
    result = json.loads(json.dumps(value))
    for node in _string_patterns_to_strip(kind, result):
        del node["pattern"]
    return result


def has_bounded_region(node: Any) -> bool:
    pending = [node]
    while pending:
        current = pending.pop()
        if isinstance(current, dict):
            # JSON Schema is opaque data, not part of the structural grammar AST.
            if current.get("type") == "json_schema":
                continue
            if (
                current.get("type") in ("any_text", "any_tokens")
                and current.get("max_tokens") is not None
            ):
                return True
            pending.extend(current.values())
        elif isinstance(current, list):
            pending.extend(current)
    return False


@dataclass(frozen=True)
class GrammarConstraint:
    """Canonical one-of constraint and owner of GenerateConfig grammar fields."""

    name: GrammarFieldName
    value: Any

    @classmethod
    def from_response_format(
        cls, response_format: Any
    ) -> Optional["GrammarConstraint"]:
        if response_format is None or response_format.type == "text":
            return None
        if response_format.type == "json_schema":
            return cls("json_schema", response_format.json_schema.schema_)
        if response_format.type == "json_object":
            # OpenAI-compatible json_object accepts any top-level JSON container.
            return cls(
                "json_schema", {"anyOf": [{"type": "object"}, {"type": "array"}]}
            )
        if response_format.type == "regex":
            return cls("regex", response_format.pattern)
        if response_format.type == "ebnf":
            return cls("ebnf", response_format.grammar)
        if response_format.type == "structural_tag":
            return cls("structural_tag", response_format.structural_tag)
        raise FtRuntimeException(
            ExceptionType.ERROR_INPUT_FORMAT_ERROR,
            f"unsupported response_format type {response_format.type}",
        )

    @classmethod
    def collect_from_config(cls, config: Any) -> List["GrammarConstraint"]:
        return [
            cls(name, value)
            for name in GRAMMAR_FIELD_NAMES
            if (value := getattr(config, name)) is not None
        ]

    @classmethod
    def clear_from_config(cls, config: Any) -> None:
        for name in GRAMMAR_FIELD_NAMES:
            setattr(config, name, None)

    @classmethod
    def normalize_config(cls, config: Any) -> None:
        for constraint in cls.collect_from_config(config):
            normalized = constraint.normalized()
            setattr(config, normalized.name, normalized.value)

    @classmethod
    def resolve_from_config(cls, config: Any) -> Optional["GrammarConstraint"]:
        constraints = cls.collect_from_config(config)
        for constraint in constraints:
            constraint.validate_not_empty()
        if len(constraints) > 1:
            field_names = " / ".join(GRAMMAR_FIELD_NAMES)
            raise FtRuntimeException(
                ExceptionType.UNSUPPORTED_OPERATION,
                f"only one grammar constraint ({field_names}) may be set per request",
            )
        return constraints[0] if constraints else None

    def apply_to_config(self, config: Any) -> None:
        if self.name not in GRAMMAR_FIELD_NAMES:
            raise FtRuntimeException(
                ExceptionType.ERROR_INPUT_FORMAT_ERROR,
                f"unsupported grammar field {self.name}",
            )
        normalized = self.normalized()
        self.clear_from_config(config)
        setattr(config, normalized.name, normalized.value)

    def normalized(self) -> "GrammarConstraint":
        return GrammarConstraint(
            self.name, normalize_grammar_value(self.name, self.value)
        )

    def validate_not_empty(self) -> None:
        if isinstance(self.value, str) and not self.value.strip():
            raise FtRuntimeException(
                ExceptionType.ERROR_INPUT_FORMAT_ERROR,
                f"{self.name} must not be empty",
            )

    def _json_schema_format_node(self) -> Dict[str, Any]:
        schema = self.value
        if not isinstance(schema, (dict, bool)):
            raise FtRuntimeException(
                ExceptionType.ERROR_INPUT_FORMAT_ERROR,
                "json_schema must be a JSON object or boolean",
            )
        return {"type": "json_schema", "json_schema": schema, "style": "json"}

    def _structural_tag_format_node(self) -> Dict[str, Any]:
        structural_tag = self.value
        if not isinstance(structural_tag, dict):
            raise FtRuntimeException(
                ExceptionType.ERROR_INPUT_FORMAT_ERROR,
                "structural_tag must be a JSON object",
            )

        if "type" not in structural_tag or structural_tag["type"] != "structural_tag":
            raise FtRuntimeException(
                ExceptionType.ERROR_INPUT_FORMAT_ERROR,
                "structural_tag must have type='structural_tag' and a format object",
            )

        if "format" not in structural_tag or not isinstance(
            structural_tag["format"], dict
        ):
            raise FtRuntimeException(
                ExceptionType.ERROR_INPUT_FORMAT_ERROR,
                "structural_tag must have type='structural_tag' and a format object",
            )

        format_node = structural_tag["format"]
        if has_bounded_region(format_node):
            raise FtRuntimeException(
                ExceptionType.UNSUPPORTED_OPERATION,
                "reasoning grammar cannot wrap a final structural_tag that "
                "already contains any_text/any_tokens max_tokens",
            )
        return format_node

    def final_format_node(self) -> Dict[str, Any]:
        if self.name == "json_schema":
            return self._json_schema_format_node()
        if self.name == "regex":
            return {"type": "regex", "pattern": self.value}
        if self.name == "ebnf":
            return {"type": "grammar", "grammar": self.value}
        if self.name == "structural_tag":
            return self._structural_tag_format_node()
        raise FtRuntimeException(
            ExceptionType.ERROR_INPUT_FORMAT_ERROR,
            f"unsupported grammar field {self.name}",
        )
