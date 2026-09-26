"""K3 grammar ownership and adaptation of xgrammar's model format."""

import copy
from typing import Any, Dict, List, Optional

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.config.generate_config import GenerateConfig
from rtp_llm.config.grammar_constraint import GrammarConstraint
from rtp_llm.config.response_format_compiler import (
    GrammarScope,
    ResponseFormatPlan,
    resolve_final_constraint,
)
from rtp_llm.config.thinking_mode import INT32_MAX
from rtp_llm.models.kimi_k3.kimi_k3_request_contract import (
    CONTROL_TOKENS,
    RESPONSE_CLOSE,
    RESPONSE_OPEN,
    THINK_CLOSE,
    THINK_OPEN,
    THINK_TO_RESPONSE,
    kimi_k3_pending_prompt_token_ids,
    kimi_k3_prompt_channel,
)


def _elements(node: Dict[str, Any]) -> List[Dict[str, Any]]:
    if node.get("type") != "sequence":
        return [node]
    elements = node.get("elements")
    if (
        not isinstance(elements, list)
        or not elements
        or not all(isinstance(element, dict) for element in elements)
    ):
        raise FtRuntimeException(
            ExceptionType.INVALID_PARAMS, "invalid Kimi K3 grammar sequence"
        )
    return elements


def structural_tag_scope(tag: Dict[str, Any]) -> GrammarScope:
    """Recognize legacy K3 envelopes without treating arbitrary text as syntax."""
    node = tag.get("format")
    if not isinstance(node, dict):
        return "response_content"
    first = _elements(node)[0]
    if first.get("type") == "optional" and isinstance(first.get("content"), dict):
        first = first["content"]
    if first.get("type") == "const_string" and first.get("value") in (
        THINK_CLOSE,
        THINK_TO_RESPONSE,
        THINK_OPEN + THINK_CLOSE,
    ):
        return "full_turn"
    if first.get("type") == "tag":
        if first.get("end") in (THINK_CLOSE, THINK_TO_RESPONSE):
            return "full_turn"
        if first.get("end") == RESPONSE_CLOSE or str(first.get("begin", "")).startswith(
            RESPONSE_CLOSE
        ):
            return "response_tail"
    return "response_content"


def _install_boundaries(config: GenerateConfig, tokenizer: Any) -> None:
    if tokenizer is not None:
        config.begin_think_token_ids = kimi_k3_pending_prompt_token_ids(tokenizer, True)
        # Usage and XTML parsing still switch channels after this full transition.
        config.end_think_token_ids = list(tokenizer.encode(THINK_TO_RESPONSE))


def build_kimi_k3_response_format_plan(
    config: GenerateConfig,
    *,
    tokenizer: Any = None,
    input_ids: Optional[List[int]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    parallel_tool_calls: bool = True,
    scope: Optional[GrammarScope] = None,
    required_tools_tail: Optional[Dict[str, Any]] = None,
    prompt_info: Optional[tuple[Optional[str], int]] = None,
) -> ResponseFormatPlan:
    """Preserve an upstream envelope, or compose one model-owned assistant turn."""
    final_constraint = resolve_final_constraint(config)
    if config.max_thinking_tokens < 0:
        config.max_thinking_tokens = INT32_MAX
    _install_boundaries(config, tokenizer)
    if (
        scope is None
        and final_constraint is not None
        and final_constraint.name == "structural_tag"
    ):
        scope = structural_tag_scope(final_constraint.value)
    scope = scope or "response_content"
    if scope == "full_turn":
        if final_constraint is None or final_constraint.name != "structural_tag":
            raise FtRuntimeException(
                ExceptionType.INVALID_PARAMS, "full_turn requires a structural_tag"
            )
        return ResponseFormatPlan(final_constraint, final_constraint)

    from xgrammar.builtin_structural_tag import get_model_structural_tag

    channel, _ = prompt_info or kimi_k3_prompt_channel(tokenizer, input_ids)
    if config.in_think_mode and channel == "response":
        raise FtRuntimeException(
            ExceptionType.INVALID_PARAMS,
            "Kimi K3 thinking is enabled but the prompt opens the response channel",
        )
    empty_think = not config.in_think_mode and channel == "think"
    reasoning = config.in_think_mode or empty_think
    budget = config.max_thinking_tokens if config.in_think_mode else 0
    try:
        tag = get_model_structural_tag(
            "kimi_k3",
            tools=tools,
            tool_choice="auto",
            reasoning="enabled" if reasoning else "disabled",
            parallel_tool_calls=parallel_tool_calls,
        ).model_dump(mode="json", exclude_none=True)
    except ValueError as exc:
        raise FtRuntimeException(
            ExceptionType.INVALID_PARAMS, f"invalid Kimi K3 tool schema: {exc}"
        ) from exc

    elements = tag["format"]["elements"]
    response_index = 1 if reasoning else 0
    if reasoning:
        think_tag = elements[0]
        think_tag["content"]["max_tokens"] = budget
        if input_ids is not None and channel is None:
            think_tag["begin"] = THINK_OPEN
    elif input_ids is not None and channel is None:
        elements[0]["begin"] = RESPONSE_OPEN

    if required_tools_tail is not None:
        # Retain RTP's JSON argument blocks and consecutive bounded call indices.
        tools_tag = copy.deepcopy(required_tools_tail["format"])
        tools_tag["begin"] = tools_tag["begin"][len(RESPONSE_CLOSE) :]
        elements[response_index]["content"] = {"type": "const_string", "value": ""}
        elements.insert(response_index + 1, tools_tag)
    elif final_constraint is not None:
        if scope == "response_tail":
            tail = final_constraint.final_format_node()
            response_begin = elements[response_index]["begin"]
            tail_elements = copy.deepcopy(_elements(tail))
            first = tail_elements[0]
            if response_begin and first.get("begin") != response_begin:
                tail_elements.insert(
                    0, {"type": "const_string", "value": response_begin}
                )
            elements[response_index:] = tail_elements
        else:
            content = copy.deepcopy(final_constraint.final_format_node())
            if content.get("type") in ("any_text", "json_schema"):
                content["excludes"] = list(
                    dict.fromkeys(content.get("excludes", []) + CONTROL_TOKENS)
                )
            elements[response_index]["content"] = content

    return ResponseFormatPlan(
        final_constraint, GrammarConstraint("structural_tag", tag).normalized()
    )
