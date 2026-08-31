"""DashSc grammar spec adapters.

This module is intentionally narrow: it only handles DashSc/DashScope wire
compatibility and normalization of a grammar spec before the payload is passed
to GenerateConfig. The grammar compiler remains responsible for full DSL
validity.
"""

from __future__ import annotations

import json
from typing import Any

# JSON Schema keywords whose value is an instance rather than a schema. xgrammar
# serializes such a value into the grammar verbatim, so rewriting it would change
# the literal the model is required to emit.
_INSTANCE_VALUED_KEYWORDS = frozenset({"const", "enum", "default", "examples"})

# Keys under which a schema may arrive as a JSON string instead of an object. The
# engine unwraps those before compiling, so admission has to unwrap them too.
_SCHEMA_BEARING_KEYS = ("json_schema", "schema")


class DashScStructuralTagError(ValueError):
    """Invalid dash-sc structural_tag request payload."""


def strip_string_length_bounds(spec: Any, *, in_json_schema: bool = False) -> bool:
    """Remove string ``minLength``/``maxLength`` in place; report whether anything went.

    xgrammar lowers a string length bound into a counted repetition, so the
    remaining-length counter joins the grammar state: every token generated
    inside the string lands in a state its token mask cache has never seen and
    costs a full vocabulary scan, even though away from the bounds the mask is
    the one the unbounded field yields. The lowering also drops the escape
    branch of the string rule on the pinned version, making legal escaped
    content unemittable. The engine strips the bound before compiling; doing it
    here too means admission validates the spec the engine will actually
    compile, instead of paying a pathological trial compile and possibly
    rejecting the request over its own budget.

    A member only counts as the keyword when its value is a number. Under
    ``properties`` the same name denotes a field and carries a schema instead,
    and the keyword is inert on non-string types, so that test alone decides and
    no type check is needed -- which also covers the string nodes whose type
    xgrammar infers rather than reads, such as the one under ``propertyNames``.

    ``in_json_schema`` tracks whether xgrammar reads this node as JSON Schema.
    The structural-tag DSL around it may carry unrelated payloads, so keywords
    only count once a ``json_schema`` value or a legacy StructuralTagItem
    ``schema`` value has been entered.
    """
    if isinstance(spec, list):
        stripped = False
        for item in spec:
            stripped |= strip_string_length_bounds(item, in_json_schema=in_json_schema)
        return stripped
    if not isinstance(spec, dict):
        return False

    stripped = False
    if in_json_schema:
        for key in ("minLength", "maxLength"):
            bound = spec.get(key)
            if isinstance(bound, (int, float)) and not isinstance(bound, bool):
                del spec[key]
                stripped = True

    legacy_schema_item = "schema" in spec and "begin" in spec and "end" in spec
    for key, value in spec.items():
        if key in _INSTANCE_VALUED_KEYWORDS:
            continue
        child_in_json_schema = (
            in_json_schema
            or key == "json_schema"
            or (legacy_schema_item and key == "schema")
        )
        if (
            child_in_json_schema
            and key in _SCHEMA_BEARING_KEYS
            and isinstance(value, str)
        ):
            stripped |= _strip_encoded_schema(spec, key, value)
            continue
        stripped |= strip_string_length_bounds(
            value, in_json_schema=child_in_json_schema
        )
    return stripped


def _strip_encoded_schema(parent: dict[str, Any], key: str, encoded: str) -> bool:
    try:
        schema = json.loads(encoded)
    except Exception:
        return False
    if not strip_string_length_bounds(schema, in_json_schema=True):
        return False
    parent[key] = json.dumps(schema, ensure_ascii=False)
    return True


def _raise_invalid(path: str, message: str, field_name: str) -> None:
    raise DashScStructuralTagError(f"invalid {field_name}: {path} {message}")


def validate_structural_tag_shape(
    value: Any, field_name: str = "tool_call_structural_tag"
) -> None:
    if not isinstance(value, dict) or not value:
        _raise_invalid("$", "must be a non-empty object", field_name)
    if "format" in value:
        return
    if "structures" in value and "triggers" in value:
        return
    _raise_invalid("$", "must contain format or legacy structures/triggers", field_name)


def adapt_dashscope_tool_call_wrapper_to_tag(
    value: dict[str, Any],
) -> dict[str, Any]:
    fmt = value.get("format")
    if not isinstance(fmt, dict) or fmt.get("type") != "sequence":
        return value
    elements = fmt.get("elements")
    if not isinstance(elements, list) or len(elements) != 3:
        return value
    begin, content, end = elements
    if (
        not isinstance(begin, dict)
        or begin.get("type") != "const_string"
        or not isinstance(begin.get("value"), str)
        or not isinstance(content, dict)
        or content.get("type") != "tags_with_separator"
        or not isinstance(end, dict)
        or end.get("type") != "const_string"
        or not isinstance(end.get("value"), str)
    ):
        return value

    adapted = dict(value)
    adapted["format"] = {
        "type": "tag",
        "begin": begin["value"],
        "content": content,
        "end": end["value"],
    }
    return adapted


def structural_tag_from_response_format(
    value: dict[str, Any], field_name: str = "response_format"
) -> dict[str, Any]:
    if value.get("type") != "structural_tag":
        _raise_invalid("$.type", "must be 'structural_tag'", field_name)
    if isinstance(value.get("format"), dict):
        structural_tag = {"format": value["format"]}
    elif isinstance(value.get("structural_tag"), dict):
        structural_tag = value["structural_tag"]
    else:
        _raise_invalid("$", "must contain format or structural_tag object", field_name)
    validate_structural_tag_shape(structural_tag, field_name)
    return structural_tag
