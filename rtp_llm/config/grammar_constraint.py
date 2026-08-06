import json
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, TypeAlias, cast, get_args

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
            return cls("json_schema", {"type": "object"})
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
