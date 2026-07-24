import json
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, TypeAlias

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException

GrammarFieldName: TypeAlias = Literal[
    "json_schema",
    "regex",
    "ebnf",
    "structural_tag",
]


def dump_compact_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except RecursionError as e:
        raise FtRuntimeException(
            ExceptionType.ERROR_INPUT_FORMAT_ERROR,
            "JSON value exceeds the supported nesting depth",
        ) from e


def load_json_field(name: str, value: Any) -> Any:
    if isinstance(value, dict):
        return value
    try:
        return json.loads(value)
    except RecursionError as e:
        raise FtRuntimeException(
            ExceptionType.ERROR_INPUT_FORMAT_ERROR,
            f"{name} exceeds the supported JSON nesting depth",
        ) from e
    except (json.JSONDecodeError, TypeError) as e:
        raise FtRuntimeException(
            ExceptionType.ERROR_INPUT_FORMAT_ERROR,
            f"{name} must be valid JSON: {str(e)}",
        )


def normalize_grammar_value(name: GrammarFieldName, value: Any) -> Any:
    if name in ("json_schema", "structural_tag") and isinstance(value, dict):
        return dump_compact_json(value)
    return value


def has_bounded_region(node: Any) -> bool:
    pending = [node]
    while pending:
        current = pending.pop()
        if isinstance(current, dict):
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
    """Canonical one-of grammar constraint before it is written to GenerateConfig fields."""

    name: GrammarFieldName
    value: Any

    @classmethod
    def from_response_format(
        cls, response_format: Any
    ) -> Optional["GrammarConstraint"]:
        if response_format is None or response_format.type == "text":
            return None
        if response_format.type == "json_schema":
            return cls("json_schema", response_format.json_schema.schema)
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
        constraints = []
        if config.json_schema is not None:
            constraints.append(cls("json_schema", config.json_schema))
        if config.regex is not None:
            constraints.append(cls("regex", config.regex))
        if config.ebnf is not None:
            constraints.append(cls("ebnf", config.ebnf))
        if config.structural_tag is not None:
            constraints.append(cls("structural_tag", config.structural_tag))
        return constraints

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
        schema: Any
        if self.value == "$$ANY$$":
            schema = True
        else:
            schema = load_json_field("json_schema", self.value)
        if not isinstance(schema, (dict, bool)):
            raise FtRuntimeException(
                ExceptionType.ERROR_INPUT_FORMAT_ERROR,
                "json_schema must be a JSON object or boolean",
            )
        return {"type": "json_schema", "json_schema": schema, "style": "json"}

    def _structural_tag_format_node(self) -> Dict[str, Any]:
        structural_tag = load_json_field("structural_tag", self.value)
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
