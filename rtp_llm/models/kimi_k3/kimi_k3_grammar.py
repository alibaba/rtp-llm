"""K3 grammar ownership and adaptation of xgrammar's model format."""

import copy
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.config.generate_config import GenerateConfig, ThinkingMode
from rtp_llm.config.grammar_constraint import GrammarConstraint
from rtp_llm.config.response_format_compiler import ReasoningFormat
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


def _exclude_controls(node: Dict[str, Any]) -> None:
    """Reserve channel markers for enclosing tags, never JSON Schema data."""
    if node.get("type") in ("any_text", "json_schema"):
        node["excludes"] = list(
            dict.fromkeys(node.get("excludes", []) + CONTROL_TOKENS)
        )
        return
    for key in ("content", "elements"):
        children = node.get(key, [])
        for child in children if isinstance(children, list) else [children]:
            _exclude_controls(child)


def _constrain_response(node: Dict[str, Any]) -> None:
    if node.get("type") == "json_schema":
        return
    if node.get("type") == "tag" and node.get("end") == RESPONSE_CLOSE:
        _exclude_controls(node["content"])
        return
    for child in node.values():
        if isinstance(child, dict):
            _constrain_response(child)
        elif isinstance(child, list):
            for element in child:
                if isinstance(element, dict):
                    _constrain_response(element)


def _adapt_upstream_grammar(tag: Dict[str, Any]) -> Dict[str, Any]:
    elements = _elements(tag["format"])
    if elements[-1].get("type") == "any_text":
        # DashScope's unconstrained suffix includes response, tools and message
        # markers. Split only the channel boundaries; tool syntax stays upstream-owned.
        tail = elements[-1]
        previous = elements[-2] if len(elements) > 1 else {}
        if previous.get("type") == "optional":
            previous = previous["content"]
        transition = previous.get("end", previous.get("value", ""))
        response_begin = (
            RESPONSE_OPEN
            if len(elements) > 1 and not transition.endswith(RESPONSE_OPEN)
            else ""
        )
        elements[-1:] = [
            {
                "type": "tag",
                "begin": response_begin,
                "content": tail,
                "end": RESPONSE_CLOSE,
            },
            {
                "type": "tag",
                "begin": "",
                "content": {
                    "type": "any_text",
                    "excludes": list(
                        dict.fromkeys(
                            tail.get("excludes", [])
                            + [
                                marker.removesuffix("<|sep|>")
                                for marker in (
                                    THINK_OPEN,
                                    THINK_CLOSE,
                                    RESPONSE_OPEN,
                                    RESPONSE_CLOSE,
                                )
                            ]
                        )
                    ),
                },
                "end": "<|close|>message<|sep|>",
            },
        ]
        tag["format"] = {"type": "sequence", "elements": elements}
    _constrain_response(tag["format"])
    return tag


@dataclass(frozen=True)
class KimiK3GrammarFormat:
    """K3 channel syntax; the shared compiler owns plans and thinking limits."""

    tools: Optional[List[Dict[str, Any]]] = None
    parallel_tool_calls: bool = True
    upstream_grammar: bool = False
    required_tools_tail: Optional[Dict[str, Any]] = None
    prompt_info: Optional[tuple[Optional[str], int]] = None
    has_prompt: bool = False

    @property
    def reasoning_format(self) -> ReasoningFormat:
        return ReasoningFormat("", THINK_CLOSE, content_excludes=tuple(CONTROL_TOKENS))

    def prepare(
        self,
        config: GenerateConfig,
        tokenizer: Any,
        input_ids: Optional[List[int]] = None,
    ) -> "KimiK3GrammarFormat":
        channel, offset = self.prompt_info or kimi_k3_prompt_channel(
            tokenizer, input_ids
        )
        if config.in_think_mode and channel == "response":
            raise FtRuntimeException(
                ExceptionType.INVALID_PARAMS,
                "Kimi K3 thinking is enabled but the prompt opens the response channel",
            )
        if config.thinking_mode == ThinkingMode.ADAPTIVE and channel == "think":
            config.thinking_mode = ThinkingMode.ENABLED
            config.in_think_mode = True
        if tokenizer is not None:
            config.begin_think_token_ids = kimi_k3_pending_prompt_token_ids(
                tokenizer, True
            )
            config.end_think_token_ids = list(tokenizer.encode(THINK_TO_RESPONSE))
        return replace(
            self,
            prompt_info=(channel, offset),
            has_prompt=self.has_prompt or input_ids is not None,
        )

    def build_format(
        self, config: GenerateConfig, constraint: Optional[GrammarConstraint]
    ) -> Dict[str, Any]:
        from xgrammar.builtin_structural_tag import get_model_structural_tag

        channel, _ = self.prompt_info or (None, 0)
        reasoning = config.in_think_mode or channel == "think"
        reasoning_mode = (
            "auto"
            if config.thinking_mode == ThinkingMode.ADAPTIVE
            else "enabled" if reasoning else "disabled"
        )
        upstream = None
        if self.upstream_grammar and constraint is not None:
            upstream = copy.deepcopy(constraint.value)
            upstream_elements = _elements(upstream["format"])
            first = upstream_elements[0]
            if first.get("type") == "optional":
                first = first["content"]
            has_think = str(first.get("end", "")).startswith(
                THINK_CLOSE
            ) or THINK_CLOSE in str(first.get("value", ""))
            if has_think or reasoning_mode == "disabled":
                return _adapt_upstream_grammar(upstream)["format"]

        try:
            tag = get_model_structural_tag(
                "kimi_k3",
                tools=self.tools,
                tool_choice="auto",
                reasoning=reasoning_mode,
                parallel_tool_calls=self.parallel_tool_calls,
            ).model_dump(mode="json", exclude_none=True)
        except ValueError as exc:
            raise FtRuntimeException(
                ExceptionType.INVALID_PARAMS, f"invalid Kimi K3 tool schema: {exc}"
            ) from exc
        elements = tag["format"]["elements"]
        response_index = 1 if reasoning_mode != "disabled" else 0
        if self.has_prompt and channel is None and reasoning_mode != "auto":
            elements[0]["begin"] = THINK_OPEN if reasoning else RESPONSE_OPEN

        if upstream is not None:
            # The upstream may deliberately leave thinking to the engine.
            if upstream_elements[0].get("end") == RESPONSE_CLOSE:
                upstream_elements[0]["begin"] = RESPONSE_OPEN
            upstream["format"] = {
                "type": "sequence",
                "elements": [elements[0]] + upstream_elements,
            }
            return _adapt_upstream_grammar(upstream)["format"]

        if self.required_tools_tail is not None:
            tools_tag = copy.deepcopy(self.required_tools_tail["format"])
            tools_tag["begin"] = tools_tag["begin"][len(RESPONSE_CLOSE) :]
            elements[response_index]["content"] = {"type": "const_string", "value": ""}
            elements.insert(response_index + 1, tools_tag)
        elif constraint is not None:
            content = copy.deepcopy(constraint.final_format_node())
            _exclude_controls(content)
            elements[response_index]["content"] = content
        return tag["format"]
