"""Compatibility parsing for DashSc structural_tag request controls.

These adapters are retained only to recognize legacy Dash wire shapes and report
stable request errors. DashSc rejects every valid structured-output control before
Model RPC until the backend grammar path is implemented.
"""

from __future__ import annotations

from typing import Any


class DashScStructuralTagError(ValueError):
    """Invalid dash-sc structural_tag request payload."""


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
