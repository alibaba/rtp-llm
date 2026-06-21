"""DashSc structural_tag request adapters.

This module is intentionally narrow: it only handles DashSc/DashScope wire
compatibility before the payload is passed to GenerateConfig.structural_tag.
The grammar compiler remains responsible for full DSL validity.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

FORCE_AT_LEAST_ONE_ENV_KEY = "DASH_SC_FORCE_STRUCTURAL_TAG_AT_LEAST_ONE"


def force_at_least_one_enabled() -> bool:
    """Truthy values: ``1`` / ``true`` / ``yes`` / ``on`` (case-insensitive)."""
    raw = os.environ.get(FORCE_AT_LEAST_ONE_ENV_KEY, "").strip().lower()
    return raw in ("1", "true", "yes", "on")


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


def _flip_at_least_one_anywhere(node: Any) -> int:
    """Walk ``node`` recursively and force every ``at_least_one`` key to
    ``True`` regardless of the surrounding shape. Returns the number of keys
    actually flipped (already-``True`` keys count as 0 so the caller can decide
    whether to re-serialize).

    Intentionally schema-blind: no format-type filter, no validation. If the
    request carries an ``at_least_one`` key anywhere in any nested JSON, it
    gets flipped. The walker is the entire policy.
    """
    count = 0
    if isinstance(node, dict):
        for k, v in list(node.items()):
            if k == "at_least_one":
                if v is not True:
                    node[k] = True
                    count += 1
            else:
                count += _flip_at_least_one_anywhere(v)
    elif isinstance(node, list):
        for item in node:
            count += _flip_at_least_one_anywhere(item)
    return count


def maybe_force_at_least_one_on_request_proto(request) -> int:
    """Env-gated: scan every ``string_param`` on the request, JSON-decode it,
    and flip every ``at_least_one`` key to ``True`` in place. Returns the total
    number of keys flipped.

    No format-type checks, no shape validation, no parameter-key allowlist —
    if the field exists in the request, the override fires. Best-effort:
    non-JSON / non-dict-or-list payloads are skipped silently so unrelated
    parameters can't break a request.
    """
    if not force_at_least_one_enabled():
        return 0
    parameters = getattr(request, "parameters", None)
    if parameters is None:
        return 0
    total = 0
    try:
        for _key, param in parameters.items():
            if not param.HasField("string_param"):
                continue
            raw = param.string_param
            if not raw or not raw.strip():
                continue
            try:
                decoded = json.loads(raw)
            except Exception:
                continue
            if not isinstance(decoded, (dict, list)):
                continue
            flipped = _flip_at_least_one_anywhere(decoded)
            if flipped > 0:
                param.string_param = json.dumps(
                    decoded, ensure_ascii=False, separators=(",", ":")
                )
                total += flipped
    except Exception as e:
        logging.warning(
            "[DashSc] force_at_least_one request mutation failed: %s", e
        )
        return total
    if total > 0:
        logging.debug(
            "[DashSc] force_at_least_one flipped %d at_least_one key(s) on request",
            total,
        )
    return total


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
