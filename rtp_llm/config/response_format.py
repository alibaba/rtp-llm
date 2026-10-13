import json
from typing import Any, Dict, Literal, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    model_serializer,
    model_validator,
)


def normalize_think_tag(value: str) -> str:
    """Convert the literal newline escapes accepted by THINK_*_TAG."""
    return value.replace(r"\n", "\n")


class ResponseFormatJSONSchema(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    name: Optional[str] = None
    description: Optional[str] = None
    schema_: Optional[Dict[str, Any]] = Field(default=None, alias="schema")
    strict: Optional[bool] = None

    @model_serializer(mode="wrap")
    def _serialize(self, handler: SerializerFunctionWrapHandler) -> Dict[str, Any]:
        """Keep the OpenAI wire field named ``schema`` in nested dumps."""
        data = handler(self)
        if "schema_" in data:
            data["schema"] = data.pop("schema_")
        return data


class ResponseFormat(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal[
        "text", "json_schema", "json_object", "regex", "ebnf", "structural_tag"
    ]
    json_schema: Optional[ResponseFormatJSONSchema] = None  # for type=json_schema
    pattern: Optional[str] = None  # for type=regex
    grammar: Optional[str] = None  # for type=ebnf
    structural_tag: Optional[Dict[str, Any]] = None  # for type=structural_tag

    @model_validator(mode="after")
    def _check_payload(self) -> "ResponseFormat":
        payload_field = {
            "json_schema": "json_schema",
            "regex": "pattern",
            "ebnf": "grammar",
            "structural_tag": "structural_tag",
        }.get(self.type)
        for field_name in ("json_schema", "pattern", "grammar", "structural_tag"):
            if field_name != payload_field and getattr(self, field_name) is not None:
                raise ValueError(
                    f"response_format.type={self.type} does not allow {field_name}"
                )
        if self.type == "json_schema":
            if self.json_schema is None or self.json_schema.schema_ is None:
                raise ValueError(
                    "response_format.type=json_schema requires json_schema.schema"
                )
        elif self.type == "regex":
            if not self.pattern:
                raise ValueError("response_format.type=regex requires pattern")
        elif self.type == "ebnf":
            if not self.grammar:
                raise ValueError("response_format.type=ebnf requires grammar")
        elif self.type == "structural_tag":
            if (
                not isinstance(self.structural_tag, dict)
                or self.structural_tag.get("type") != "structural_tag"
                or not isinstance(self.structural_tag.get("format"), dict)
            ):
                raise ValueError(
                    "response_format.type=structural_tag requires "
                    "structural_tag.type='structural_tag' and a format object"
                )
        return self


def parse_response_format(value: Any) -> Optional[ResponseFormat]:
    """Normalize legacy wire shapes to the canonical response-format model."""
    if value is None or isinstance(value, ResponseFormat):
        return value
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        if value == "text":
            return ResponseFormat(type="text")
        try:
            value = json.loads(value)
        except RecursionError as e:
            raise ValueError(
                "response_format exceeds the supported JSON nesting depth"
            ) from e
    return ResponseFormat.model_validate(value)
