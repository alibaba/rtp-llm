import json
from typing import Any, Dict, Literal, Optional, TypeAlias, Union

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
    model_config = ConfigDict(populate_by_name=True)

    name: Optional[str] = None
    schema_: Optional[Dict[str, Any]] = Field(default=None, alias="schema")

    @model_serializer(mode="wrap")
    def _serialize(self, handler: SerializerFunctionWrapHandler) -> Dict[str, Any]:
        """Keep the OpenAI wire field named ``schema`` in nested dumps."""
        data = handler(self)
        if "schema_" in data:
            data["schema"] = data.pop("schema_")
        return data


class ResponseFormat(BaseModel):
    type: Literal[
        "text", "json_schema", "json_object", "regex", "ebnf", "structural_tag"
    ]
    json_schema: Optional[ResponseFormatJSONSchema] = None  # for type=json_schema
    pattern: Optional[str] = None  # for type=regex
    grammar: Optional[str] = None  # for type=ebnf
    structural_tag: Optional[Dict[str, Any]] = None  # for type=structural_tag

    @model_validator(mode="after")
    def _check_payload(self) -> "ResponseFormat":
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
            if not self.structural_tag:
                raise ValueError(
                    "response_format.type=structural_tag requires structural_tag"
                )
        return self


def parse_response_format(value: Any) -> Optional[ResponseFormat]:
    """Parse loose request payloads into a validated ResponseFormat envelope."""
    if value is None:
        return None
    if isinstance(value, ResponseFormat):
        return value
    if isinstance(value, BaseModel):
        # The OpenAI request layer owns a wire-model ResponseFormat class that
        # is intentionally separate from this canonical config model. Convert
        # any validated Pydantic envelope back to its wire shape before
        # canonical validation instead of coupling the config layer to the
        # OpenAI module.
        value = value.model_dump(exclude_none=True)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        if stripped == "text":
            return ResponseFormat(type="text")
        value = json.loads(stripped)
    if isinstance(value, dict):
        if not value:
            return None
        return ResponseFormat(**value)
    raise TypeError(f"response_format has unsupported type {type(value).__name__}")


ResponseFormatInput: TypeAlias = Union[ResponseFormat, Dict[str, Any], str]
