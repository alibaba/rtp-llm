"""Reject XGrammar request shapes known to make decode CPU-bound.

The parser mirror in this module is aligned with XGrammar v0.2.3, pinned at
557becfb64c503ae9c04344b0047661f43f44320 in WORKSPACE.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

_INT32_MAX = 2_147_483_647
_DEEPSEEK_XML_LENGTH_LIMIT = 128
_XGRAMMAR_REGEX_STRING_FORMATS = frozenset(
    {
        "date",
        "date-time",
        "duration",
        "email",
        "hostname",
        "ipv4",
        "ipv6",
        "json-pointer",
        "relative-json-pointer",
        "time",
        "uri",
        "uri-reference",
        "uri-template",
        "uuid",
    }
)
# JSON Schema minLength counts Unicode characters, not model tokens. Keep the
# unbounded plain-JSON lower bound capped without applying the unrelated
# deepseek_xml 128 boundary to ordinary JSON.
_PLAIN_JSON_UNBOUNDED_MIN_LENGTH_LIMIT = 2_000


class XGrammarSafetyError(ValueError):
    """The request enters a known pathological XGrammar length path."""


def _is_non_empty_object_list(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(item, dict) for item in value)
    )


def _is_integral_int32(value: Any, *, non_negative: bool) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    if isinstance(value, float) and not value.is_integer():
        return False
    return (-_INT32_MAX - 1 if not non_negative else 0) <= value <= _INT32_MAX


def _is_structural_token(value: Any) -> bool:
    return _is_integral_int32(value, non_negative=True) or (
        isinstance(value, str) and bool(value)
    )


def _is_structural_token_format(value: Any) -> bool:
    return isinstance(value, dict) and _is_structural_token(value.get("token"))


def _is_valid_structural_format(
    value: dict[str, Any],
    format_type: str,
    classified: dict[int, str | None],
) -> bool:
    def is_valid_child(child: Any) -> bool:
        return isinstance(child, dict) and classified.get(id(child)) is not None

    def has_valid_optional_bool(name: str) -> bool:
        return name not in value or isinstance(value[name], bool)

    if format_type == "tag":
        if "type" in value and value.get("type") != "tag":
            return False
        begin = value.get("begin")
        end = value.get("end")
        valid_begin = isinstance(begin, str) or _is_structural_token_format(begin)
        valid_end = (
            isinstance(end, str)
            or _is_structural_token_format(end)
            or (
                isinstance(end, list)
                and bool(end)
                and all(isinstance(item, str) for item in end)
            )
        )
        return valid_begin and valid_end and is_valid_child(value.get("content"))
    if format_type == "const_string":
        return isinstance(value.get("value"), str)
    if format_type in ("json_schema", "qwen_xml_parameter"):
        schema = value.get("json_schema")
        style = value.get("style")
        valid_style = (
            format_type == "qwen_xml_parameter"
            or not isinstance(style, str)
            or style in {"json", "qwen_xml", "minimax_xml", "deepseek_xml", "glm_xml"}
        )
        return (
            (
                isinstance(schema, (dict, bool))
                or (
                    "json_schema" not in value
                    and value.get("type") in ("json_schema", "qwen_xml_parameter")
                )
            )
            and valid_style
            and has_valid_optional_bool("any_order")
        )
    if format_type == "any_text":
        excludes = value.get("excludes")
        return "excludes" not in value or (
            isinstance(excludes, list)
            and all(isinstance(item, str) for item in excludes)
        )
    if format_type in ("sequence", "or"):
        elements = value.get("elements")
        return _is_non_empty_object_list(elements) and all(
            is_valid_child(element) for element in elements
        )
    if format_type in (
        "triggered_tags",
        "tags_with_separator",
        "token_triggered_tags",
    ):
        tags = value.get("tags")
        if not _is_non_empty_object_list(tags) or not all(
            _is_valid_structural_format(tag, "tag", classified) for tag in tags
        ):
            return False
        if format_type == "triggered_tags":
            triggers = value.get("triggers")
            excludes = value.get("excludes")
            return (
                isinstance(triggers, list)
                and bool(triggers)
                and all(isinstance(item, str) and item for item in triggers)
                and (
                    "excludes" not in value
                    or (
                        isinstance(excludes, list)
                        and all(isinstance(item, str) and item for item in excludes)
                    )
                )
                and has_valid_optional_bool("at_least_one")
                and has_valid_optional_bool("stop_after_first")
            )
        if format_type == "tags_with_separator":
            return (
                isinstance(value.get("separator"), str)
                and has_valid_optional_bool("at_least_one")
                and has_valid_optional_bool("stop_after_first")
            )
        trigger_tokens = value.get("trigger_tokens")
        exclude_tokens = value.get("exclude_tokens")
        return (
            isinstance(trigger_tokens, list)
            and bool(trigger_tokens)
            and all(_is_structural_token(item) for item in trigger_tokens)
            and (
                "exclude_tokens" not in value
                or (
                    isinstance(exclude_tokens, list)
                    and all(_is_structural_token(item) for item in exclude_tokens)
                )
            )
            and has_valid_optional_bool("at_least_one")
            and has_valid_optional_bool("stop_after_first")
        )
    if format_type in ("optional", "plus", "star"):
        return is_valid_child(value.get("content"))
    if format_type == "repeat":
        minimum = value.get("min")
        maximum = value.get("max")
        return (
            isinstance(minimum, int)
            and not isinstance(minimum, bool)
            and 0 <= minimum <= _INT32_MAX
            and isinstance(maximum, int)
            and not isinstance(maximum, bool)
            and maximum >= -1
            and (maximum == -1 or minimum <= maximum)
            and is_valid_child(value.get("content"))
        )
    if format_type in ("dispatch", "token_dispatch"):
        rules = value.get("rules")
        if (
            not isinstance(rules, list)
            or not rules
            or not has_valid_optional_bool("loop")
        ):
            return False

        def is_valid_trigger(trigger: Any) -> bool:
            if isinstance(trigger, str):
                return True
            return format_type == "token_dispatch" and _is_integral_int32(
                trigger, non_negative=False
            )

        valid_rules = all(
            isinstance(rule, list)
            and len(rule) == 2
            and is_valid_trigger(rule[0])
            and is_valid_child(rule[1])
            for rule in rules
        )
        if not valid_rules:
            return False
        if format_type == "dispatch":
            excludes = value.get("excludes")
            return "excludes" not in value or (
                isinstance(excludes, list)
                and all(isinstance(item, str) and item for item in excludes)
            )
        exclude_tokens = value.get("exclude_tokens")
        return "exclude_tokens" not in value or (
            isinstance(exclude_tokens, list)
            and all(_is_structural_token(item) for item in exclude_tokens)
        )
    if format_type == "grammar":
        return isinstance(value.get("grammar"), str) and bool(value["grammar"])
    if format_type == "regex":
        return isinstance(value.get("pattern"), str) and bool(value["pattern"])
    if format_type == "token":
        return _is_structural_token(value.get("token"))
    if format_type in ("exclude_token", "any_tokens"):
        exclude_tokens = value.get("exclude_tokens")
        return "exclude_tokens" not in value or (
            isinstance(exclude_tokens, list)
            and all(_is_structural_token(item) for item in exclude_tokens)
        )
    return False


def _implicit_structural_format_type(
    value: dict[str, Any], classified: dict[int, str | None]
) -> str | None:
    """Infer an omitted format type in XGrammar's parser precedence order."""
    if _is_valid_structural_format(value, "tag", classified):
        return "tag"
    if _is_valid_structural_format(value, "const_string", classified):
        return "const_string"
    if _is_valid_structural_format(value, "json_schema", classified):
        return "json_schema"
    excludes = value.get("excludes")
    if excludes is not None and _is_valid_structural_format(
        value, "any_text", classified
    ):
        return "any_text"
    if _is_valid_structural_format(value, "sequence", classified):
        return "sequence"
    if _is_valid_structural_format(value, "triggered_tags", classified):
        return "triggered_tags"
    if _is_valid_structural_format(value, "tags_with_separator", classified):
        return "tags_with_separator"
    if _is_valid_structural_format(value, "optional", classified):
        return "optional"
    if _is_valid_structural_format(value, "dispatch", classified):
        return "dispatch"
    if _is_valid_structural_format(value, "token_dispatch", classified):
        return "token_dispatch"
    return None


def _structural_format_children(value: dict[str, Any]) -> Iterator[dict[str, Any]]:
    format_type = value.get("type")
    if format_type is not None and not isinstance(format_type, str):
        return
    if format_type is None or format_type in ("sequence", "or"):
        elements = value.get("elements")
        if isinstance(elements, list):
            yield from (child for child in elements if isinstance(child, dict))
    if format_type is None or format_type in (
        "triggered_tags",
        "tags_with_separator",
        "token_triggered_tags",
    ):
        tags = value.get("tags")
        if isinstance(tags, list):
            yield from (child for child in tags if isinstance(child, dict))
    if format_type is None or format_type in (
        "tag",
        "optional",
        "plus",
        "star",
        "repeat",
    ):
        content = value.get("content")
        if isinstance(content, dict):
            yield content
    if format_type is None or format_type in ("dispatch", "token_dispatch"):
        rules = value.get("rules")
        if isinstance(rules, list):
            for rule in rules:
                if (
                    isinstance(rule, list)
                    and len(rule) == 2
                    and isinstance(rule[1], dict)
                ):
                    yield rule[1]


def _classify_structural_formats(root: Any) -> dict[int, str | None]:
    """Classify one structural DSL tree iteratively in O(number of formats)."""
    if not isinstance(root, dict):
        return {}
    classified: dict[int, str | None] = {}
    stack: list[tuple[dict[str, Any], bool, int]] = [(root, False, 0)]
    while stack:
        current, expanded, depth = stack.pop()
        current_id = id(current)
        if current_id in classified:
            continue
        if depth > 10_000:
            raise XGrammarSafetyError(
                "invalid structural_tag: format nesting exceeds 10000"
            )
        if not expanded:
            stack.append((current, True, depth))
            for child in _structural_format_children(current):
                if id(child) not in classified:
                    stack.append((child, False, depth + 1))
            continue

        explicit_type = current.get("type")
        if explicit_type is None:
            format_type = _implicit_structural_format_type(current, classified)
        elif isinstance(explicit_type, str) and _is_valid_structural_format(
            current, explicit_type, classified
        ):
            format_type = explicit_type
        else:
            format_type = None
        classified[current_id] = format_type
    return classified


def _iter_structural_json_schema_formats(
    structural_tag: dict[str, Any],
) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield JSON-schema formats reachable through the structural-tag DSL."""
    root = structural_tag.get("format")
    classified = _classify_structural_formats(root)
    stack = [("structural_tag.format", root)]
    while stack:
        path, current = stack.pop()
        if not isinstance(current, dict):
            continue
        format_type = classified.get(id(current))
        if format_type == "json_schema":
            yield path, current
            continue
        if format_type in ("sequence", "or"):
            children = current.get("elements")
            child_name = "elements"
        elif format_type in (
            "triggered_tags",
            "tags_with_separator",
            "token_triggered_tags",
        ):
            children = current.get("tags")
            child_name = "tags"
        else:
            children = None
            child_name = ""
        if isinstance(children, list):
            for index in range(len(children) - 1, -1, -1):
                stack.append((f"{path}.{child_name}[{index}]", children[index]))

        if format_type in ("tag", "optional", "plus", "star", "repeat"):
            stack.append((f"{path}.content", current.get("content")))
        elif format_type in ("dispatch", "token_dispatch"):
            rules = current.get("rules")
            if isinstance(rules, list):
                for index in range(len(rules) - 1, -1, -1):
                    rule = rules[index]
                    if isinstance(rule, list) and len(rule) == 2:
                        stack.append((f"{path}.rules[{index}][1]", rule[1]))


def _decode_json_schema_string(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return value


def _response_format_json_schema(container: Any) -> Any:
    """Mirror the C++ response-format envelope extraction semantics."""
    if isinstance(container, dict) and "schema" in container:
        return _decode_json_schema_string(container["schema"])
    return _decode_json_schema_string(container)


def _resolve_local_json_schema_ref(root_schema: Any, ref: Any) -> Any:
    """Resolve the local references supported by XGrammar's schema parser."""
    if ref == "#":
        return root_schema
    if not isinstance(ref, str) or not ref.startswith("#/"):
        return None
    current = root_schema
    for part in (part for part in ref[2:].split("/") if part):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _iter_xgrammar_string_schemas(
    schema: Any,
    root_path: str,
    *,
    min_container_depth: int = 0,
    max_container_depth: int | None = None,
) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield string specs that XGrammar actually compiles from a JSON Schema."""
    root_schema = schema
    stack: list[tuple[str, Any, int, str | None]] = [(root_path, schema, 0, None)]
    visited: set[tuple[int, int, str | None]] = set()

    while stack:
        path, current, container_depth, default_type = stack.pop()
        if not isinstance(current, dict):
            continue
        if max_container_depth is not None and container_depth > max_container_depth:
            continue
        if max_container_depth is not None:
            visit_depth = container_depth
        elif min_container_depth > 0:
            visit_depth = min(container_depth, min_container_depth)
        else:
            visit_depth = 0
        visit_key = (id(current), visit_depth, default_type)
        if visit_key in visited:
            continue
        visited.add(visit_key)

        if "$ref" in current:
            resolved = _resolve_local_json_schema_ref(root_schema, current["$ref"])
            if resolved is not None:
                stack.append(
                    (
                        f"{path}.$ref({current['$ref']})",
                        resolved,
                        container_depth,
                        None,
                    )
                )
            continue
        if "const" in current or "enum" in current:
            continue
        union_keyword = "anyOf" if "anyOf" in current else "oneOf"
        if union_keyword in current:
            options = current[union_keyword]
            if isinstance(options, list):
                for index in range(len(options) - 1, -1, -1):
                    stack.append(
                        (
                            f"{path}.{union_keyword}[{index}]",
                            options[index],
                            container_depth,
                            None,
                        )
                    )
            continue
        if "allOf" in current:
            schemas = current["allOf"]
            if isinstance(schemas, list) and len(schemas) == 1:
                stack.append((f"{path}.allOf[0]", schemas[0], container_depth, None))
            continue

        raw_type = current.get("type")
        if isinstance(raw_type, list):
            if raw_type:
                schema_types: list[Any] = list(raw_type)
            elif any(
                keyword in current
                for keyword in (
                    "properties",
                    "additionalProperties",
                    "unevaluatedProperties",
                )
            ):
                schema_types = ["object"]
            elif any(
                keyword in current
                for keyword in ("items", "prefixItems", "unevaluatedItems")
            ):
                schema_types = ["array"]
            else:
                schema_types = []
        elif isinstance(raw_type, str):
            schema_types = [raw_type]
        elif "type" in current:
            schema_types = []
        elif default_type is not None:
            schema_types = [default_type]
        elif any(
            keyword in current
            for keyword in (
                "properties",
                "additionalProperties",
                "unevaluatedProperties",
            )
        ):
            schema_types = ["object"]
        elif any(
            keyword in current
            for keyword in ("items", "prefixItems", "unevaluatedItems")
        ):
            schema_types = ["array"]
        else:
            schema_types = []

        for schema_type in schema_types:
            if schema_type == "string":
                if container_depth >= min_container_depth:
                    yield path, current
                continue
            child_depth = container_depth + 1
            if schema_type == "object":
                for keyword in ("properties", "patternProperties"):
                    children = current.get(keyword)
                    if isinstance(children, dict):
                        for name, child in children.items():
                            stack.append(
                                (
                                    f"{path}.{keyword}.{name}",
                                    child,
                                    child_depth,
                                    None,
                                )
                            )
                if "propertyNames" in current:
                    stack.append(
                        (
                            f"{path}.propertyNames",
                            current["propertyNames"],
                            child_depth,
                            "string",
                        )
                    )
                if "additionalProperties" in current:
                    stack.append(
                        (
                            f"{path}.additionalProperties",
                            current["additionalProperties"],
                            child_depth,
                            None,
                        )
                    )
                elif "unevaluatedProperties" in current:
                    stack.append(
                        (
                            f"{path}.unevaluatedProperties",
                            current["unevaluatedProperties"],
                            child_depth,
                            None,
                        )
                    )
            elif schema_type == "array":
                prefix_items = current.get("prefixItems")
                if isinstance(prefix_items, list):
                    for index in range(len(prefix_items) - 1, -1, -1):
                        stack.append(
                            (
                                f"{path}.prefixItems[{index}]",
                                prefix_items[index],
                                child_depth,
                                None,
                            )
                        )
                if "items" in current:
                    stack.append(
                        (
                            f"{path}.items",
                            current["items"],
                            child_depth,
                            None,
                        )
                    )
                elif "unevaluatedItems" in current:
                    stack.append(
                        (
                            f"{path}.unevaluatedItems",
                            current["unevaluatedItems"],
                            child_depth,
                            None,
                        )
                    )


def _is_length_above(value: Any, limit: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > limit


def _is_xgrammar_unbounded_max_length(value: Any) -> bool:
    if not isinstance(value, int) or isinstance(value, bool):
        return False
    # XGrammar v0.2.3 narrows the JSON int64 to a 32-bit int; -1 is its
    # unbounded sentinel.  Include values that narrow to the same sentinel.
    return value & 0xFFFFFFFF == 0xFFFFFFFF


def _xgrammar_uses_string_length(schema: dict[str, Any]) -> bool:
    """Whether GenerateString reaches its length-constraint branch."""
    if "pattern" in schema:
        return False
    string_format = schema.get("format")
    if "format" in schema and not isinstance(string_format, str):
        return False
    return string_format not in _XGRAMMAR_REGEX_STRING_FORMATS


def validate_xgrammar_length_safety(
    response_format: Any, structural_tag: dict[str, Any] | None
) -> None:
    """Reject the two length patterns known to make XGrammar decode CPU-bound."""
    if structural_tag is not None:
        for tag_path, tag_node in _iter_structural_json_schema_formats(structural_tag):
            schema = tag_node.get("json_schema")
            style_value = tag_node.get("style")
            style = style_value.lower() if isinstance(style_value, str) else "json"
            if style == "deepseek_xml":
                for schema_path, schema_node in _iter_xgrammar_string_schemas(
                    schema,
                    f"{tag_path}.json_schema",
                    max_container_depth=1,
                ):
                    if not _xgrammar_uses_string_length(schema_node):
                        continue
                    for keyword in ("minLength", "maxLength"):
                        value = schema_node.get(keyword)
                        if _is_length_above(value, _DEEPSEEK_XML_LENGTH_LIMIT):
                            raise XGrammarSafetyError(
                                "unsupported deepseek_xml length constraint: "
                                f"{schema_path}.{keyword}={value}; must be <= "
                                f"{_DEEPSEEK_XML_LENGTH_LIMIT}"
                            )
                _validate_plain_json_length_safety(
                    schema,
                    f"{tag_path}.json_schema",
                    min_container_depth=2,
                )
            elif style == "json":
                _validate_plain_json_length_safety(schema, f"{tag_path}.json_schema")

    if not isinstance(response_format, dict):
        return
    if response_format.get("type") != "json_schema":
        return
    schema = _response_format_json_schema(response_format.get("json_schema"))
    _validate_plain_json_length_safety(schema, "response_format.json_schema")


def _validate_plain_json_length_safety(
    schema: Any,
    root_path: str,
    *,
    min_container_depth: int = 0,
) -> None:
    for schema_path, schema_node in _iter_xgrammar_string_schemas(
        schema,
        root_path,
        min_container_depth=min_container_depth,
    ):
        if not _xgrammar_uses_string_length(schema_node):
            continue
        min_length = schema_node.get("minLength")
        max_length = schema_node.get("maxLength")
        if (
            isinstance(min_length, int)
            and not isinstance(min_length, bool)
            and min_length > _PLAIN_JSON_UNBOUNDED_MIN_LENGTH_LIMIT
            and (
                "maxLength" not in schema_node
                or _is_xgrammar_unbounded_max_length(max_length)
            )
        ):
            raise XGrammarSafetyError(
                "unsupported unbounded JSON schema length constraint: "
                f"{schema_path}.minLength={min_length}; minLength must be <= "
                f"{_PLAIN_JSON_UNBOUNDED_MIN_LENGTH_LIMIT} when maxLength is "
                "absent or unbounded"
            )
