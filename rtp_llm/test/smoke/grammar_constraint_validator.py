"""Constraint validators for grammar smoke (stdlib only, no jsonschema dependency).

Single source of truth for "does the model output satisfy what xgrammar was asked
to enforce?". Each response_format type has a verifier; `validate_constraint` is the
dispatcher used by OpenaiComparer when a task is marked grammar_constraint_only.

The intent is to validate the *constraint*, never byte-equality with a golden:
- regex          -> re.fullmatch(pattern, content)
- json_schema    -> parse JSON + recursive schema-subset check
- structural_tag -> triggered_tags / glm_xml structural checks (incl. trigger invariant)
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict

Schema = Dict[str, Any]

_GLM_XML_PAIR = r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>"


# ---------------------------------------------------------------------------
# json_schema
# ---------------------------------------------------------------------------
def validate_json_schema_instance(
    instance: Any, schema: Schema, path: str = "$"
) -> None:
    """Validate *instance* against the JSON Schema subset used in smoke response_format."""
    if not schema:
        return

    schema_type = schema.get("type")
    if schema_type == "object":
        if not isinstance(instance, dict):
            raise ValueError(f"{path}: expected object, got {type(instance).__name__}")
        properties = schema.get("properties") or {}
        if schema.get("additionalProperties") is False:
            extra = set(instance.keys()) - set(properties.keys())
            if extra:
                raise ValueError(f"{path}: unexpected keys {sorted(extra)}")
        for key in schema.get("required") or []:
            if key not in instance:
                raise ValueError(f"{path}: missing required field {key!r}")
        for key, prop_schema in properties.items():
            if key in instance:
                validate_json_schema_instance(
                    instance[key], prop_schema, f"{path}.{key}"
                )
        return

    if schema_type == "string":
        if not isinstance(instance, str):
            raise ValueError(f"{path}: expected string, got {type(instance).__name__}")
        enum = schema.get("enum")
        if enum is not None and instance not in enum:
            raise ValueError(f"{path}: {instance!r} not in enum {enum}")
        pattern = schema.get("pattern")
        if pattern is not None and re.fullmatch(pattern, instance) is None:
            raise ValueError(f"{path}: {instance!r} does not match pattern {pattern!r}")
        max_length = schema.get("maxLength")
        if max_length is not None and len(instance) > max_length:
            raise ValueError(
                f"{path}: length {len(instance)} exceeds maxLength {max_length}"
            )
        return

    if schema_type == "integer":
        if isinstance(instance, bool) or not isinstance(instance, int):
            raise ValueError(f"{path}: expected integer, got {type(instance).__name__}")
        return

    if schema_type == "number":
        if isinstance(instance, bool) or not isinstance(instance, (int, float)):
            raise ValueError(f"{path}: expected number, got {type(instance).__name__}")
        return

    if schema_type == "boolean":
        if not isinstance(instance, bool):
            raise ValueError(f"{path}: expected boolean, got {type(instance).__name__}")
        return

    if schema_type == "array":
        if not isinstance(instance, list):
            raise ValueError(f"{path}: expected array, got {type(instance).__name__}")
        min_items = schema.get("minItems")
        if min_items is not None and len(instance) < min_items:
            raise ValueError(f"{path}: length {len(instance)} < minItems {min_items}")
        max_items = schema.get("maxItems")
        if max_items is not None and len(instance) > max_items:
            raise ValueError(f"{path}: length {len(instance)} > maxItems {max_items}")
        item_schema = schema.get("items")
        if item_schema:
            for i, item in enumerate(instance):
                validate_json_schema_instance(item, item_schema, f"{path}[{i}]")
        return

    raise ValueError(f"{path}: unsupported or missing schema type {schema_type!r}")


def validate_response_json_schema(
    content: str, response_format: Dict[str, Any], idx: int
) -> None:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"choice[{idx}] content is not valid JSON: {e}; content={content!r}"
        ) from e
    schema = (response_format.get("json_schema") or {}).get("schema")
    if not schema:
        raise ValueError(f"choice[{idx}] response_format.json_schema.schema missing")
    validate_json_schema_instance(parsed, schema, f"choice[{idx}]")


# ---------------------------------------------------------------------------
# structural_tag
# ---------------------------------------------------------------------------
def validate_glm_xml_payload(
    content: str, fmt: Dict[str, Any], idx: int, *, allow_outside: bool
) -> None:
    """Validate (<arg_key>K</arg_key>\\s*<arg_value>V</arg_value>)+ payload.

    allow_outside=True tolerates surrounding text (used for tag inner bodies whose
    delimiters were already stripped by the caller).
    """
    style = fmt.get("style", "json")
    if style != "glm_xml":
        raise ValueError(
            f"choice[{idx}] structural_tag validator does not yet support style={style!r}"
        )
    schema = fmt.get("json_schema") or {}
    pairs = re.findall(_GLM_XML_PAIR, content, re.DOTALL)
    if not pairs:
        raise ValueError(
            f"choice[{idx}] glm_xml content has no arg_key/arg_value pairs: {content!r}"
        )
    if not allow_outside:
        stripped = re.sub(_GLM_XML_PAIR, "", content, flags=re.DOTALL)
        if stripped.strip():
            raise ValueError(
                f"choice[{idx}] glm_xml content has extraneous text outside "
                f"arg_key/arg_value pairs: {stripped!r} (full: {content!r})"
            )
    keys = [k for k, _ in pairs]
    required = schema.get("required") or list((schema.get("properties") or {}).keys())
    missing = set(required) - set(keys)
    if missing:
        raise ValueError(
            f"choice[{idx}] glm_xml missing required keys {sorted(missing)} in {keys}"
        )


def validate_triggered_tags(content: str, fmt: Dict[str, Any], idx: int) -> None:
    """Validate a triggered_tags output against the xgrammar guarantee.

    xgrammar enforces: free text anywhere, EXCEPT (1) `excludes` strings never appear,
    (2) once a `trigger` prefix is emitted the grammar forces a complete, schema-valid
    tag (begin..inner..end), and (3) `at_least_one` => >=1 tag. We verify all three:
      - excludes absent
      - every well-formed tag's inner payload matches its content schema
      - after removing all valid tags, NO bare trigger remains in the free text
        (a dangling trigger means a token escaped the mask)
      - at_least_one honoured
    """
    triggers = fmt.get("triggers") or []
    tags = fmt.get("tags") or []
    excludes = fmt.get("excludes") or []
    at_least_one = bool(fmt.get("at_least_one", False))

    for ex in excludes:
        if ex and ex in content:
            raise ValueError(
                f"choice[{idx}] triggered_tags content contains excluded marker "
                f"{ex!r}: {content!r}"
            )

    matched = 0
    remaining = content
    for tag in tags:
        begin = tag.get("begin") or ""
        end = tag.get("end") or ""
        tag_content = tag.get("content") or {}
        if not begin or not end:
            continue
        pat = re.escape(begin) + r"(.*?)" + re.escape(end)
        for m in re.finditer(pat, content, re.DOTALL):
            matched += 1
            inner = m.group(1)
            inner_type = tag_content.get("type")
            if inner_type == "json_schema":
                validate_glm_xml_payload(inner, tag_content, idx, allow_outside=True)
            else:
                raise ValueError(
                    f"choice[{idx}] triggered_tags inner content.type={inner_type!r} "
                    f"not supported by validator"
                )
        remaining = re.sub(pat, "", remaining, flags=re.DOTALL)

    for trig in triggers:
        if trig and trig in remaining:
            raise ValueError(
                f"choice[{idx}] triggered_tags emitted a bare trigger {trig!r} that did "
                f"not complete into a valid tag: {remaining!r}"
            )

    if at_least_one and matched == 0:
        raise ValueError(
            f"choice[{idx}] triggered_tags requires at_least_one match but found none "
            f"in {content!r}"
        )


def validate_structural_tag(
    content: str, response_format: Dict[str, Any], idx: int
) -> None:
    stag = response_format.get("structural_tag") or {}
    fmt = stag.get("format") or {}
    fmt_type = fmt.get("type")
    if fmt_type == "json_schema":
        validate_glm_xml_payload(content, fmt, idx, allow_outside=False)
    elif fmt_type == "triggered_tags":
        validate_triggered_tags(content, fmt, idx)
    else:
        raise ValueError(
            f"choice[{idx}] structural_tag validator does not handle format.type={fmt_type!r}"
        )


# ---------------------------------------------------------------------------
# dispatcher
# ---------------------------------------------------------------------------
def validate_constraint(
    content: str, response_format: Dict[str, Any], idx: int
) -> None:
    """Validate one choice's content against its response_format. Raises ValueError on violation."""
    if content is None:
        raise ValueError(f"choice[{idx}].message.content is None")
    rf_type = response_format.get("type")
    if rf_type == "json_schema":
        validate_response_json_schema(content, response_format, idx)
    elif rf_type == "regex":
        pattern = response_format.get("pattern")
        if not pattern:
            raise ValueError("response_format.pattern missing")
        if re.fullmatch(pattern, content) is None:
            raise ValueError(
                f"choice[{idx}] content {content!r} does not match {pattern!r}"
            )
    elif rf_type == "structural_tag":
        validate_structural_tag(content, response_format, idx)
    else:
        raise ValueError(f"grammar_constraint_only does not support type={rf_type}")
