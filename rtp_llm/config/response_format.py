from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, model_validator


class ResponseFormatJSONSchema(BaseModel):
    # `schema` shadows BaseModel.schema() in pydantic v1 and triggers a v2
    # protected_namespaces UserWarning. The OpenAI wire field is literally
    # named "schema", so we keep the alias and silence the namespace.
    model_config = ConfigDict(protected_namespaces=())

    name: Optional[str] = None
    schema: Optional[Dict[str, Any]] = None
    strict: Optional[bool] = None


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
            if self.json_schema is None or self.json_schema.schema is None:
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
