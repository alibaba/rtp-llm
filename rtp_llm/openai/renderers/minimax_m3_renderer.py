import json
import logging
from typing import Any, List, Optional

from jinja2 import Environment
from typing_extensions import override

from rtp_llm.openai.api_datatype import ChatCompletionRequest
from rtp_llm.openai.renderer_factory_register import register_renderer
from rtp_llm.openai.renderers.reasoning_tool_base_renderer import (
    ReasoningToolBaseRenderer,
)
from rtp_llm.openai.renderers.sglang_helpers.function_call.base_format_detector import (
    BaseFormatDetector,
)
from rtp_llm.openai.renderers.sglang_helpers.function_call.minimax_m3_detector import (
    MiniMaxM3Detector,
)
from rtp_llm.openai.renderers.sglang_helpers.reasoning_parser import (
    ReasoningParser,
    normalize_mm_think_tags,
)


class MiniMaxM3Renderer(ReasoningToolBaseRenderer):
    """
    OpenAI chat renderer for text-only MiniMax-M3.

    The HF `chat_template.jinja` already renders the `# Tools` section and the
    `<mm:think>` thinking markers, so the base class's `_build_prompt` (which
    feeds the whole request dict to the template, `tools` included) covers the
    input side. This class adds the output-side parsers plus the M3-specific
    template and stop-word wiring.
    """

    @override
    def _setup_stop_words(self):
        # CustomChatRenderer does not register the tokenizer's special tokens as
        # stop words the way BasicRenderer did, and without eos every request
        # would run to max_new_tokens. Deliberately excluding
        # `additional_special_tokens`: on M3 that would risk registering the tool
        # call markers (`]<]minimax[>[`, `<mm:think>`) and truncate every tool call.
        special_tokens_map = getattr(self.tokenizer, "special_tokens_map", None) or {}
        for name, token in special_tokens_map.items():
            if isinstance(token, str):
                tokens = [token]
            elif isinstance(token, list):
                tokens = token
            else:
                continue
            logging.info(
                f"minimax_m3: special token {tokens} ({name}) added as stop words."
            )
            self.add_extra_stop_words(tokens)

    @override
    def _customize_jinja_env(self, env: Environment) -> None:
        # The M3 template calls `tojson(ensure_ascii=False)`; the base class installs
        # a single-argument lambda, which would raise TypeError.
        def to_json(value: Any, ensure_ascii: bool = False) -> str:
            if isinstance(value, str):
                return value
            return json.dumps(value, sort_keys=False, ensure_ascii=ensure_ascii)

        env.filters["tojson"] = to_json

    @override
    def in_think_mode(self, request: ChatCompletionRequest) -> bool:
        # M3's default thinking mode is adaptive, so reasoning output is always
        # possible and must always be routed to `reasoning_content`.
        return True

    @override
    def _preprocess_messages(self, messages: List[dict]) -> List[dict]:
        # The template iterates `tool_call.arguments.items()`, but the OpenAI wire
        # format carries arguments as a JSON string.
        for message in messages:
            self._normalize_think_markers(message)
            for tool_call in message.get("tool_calls") or []:
                function = tool_call.get("function")
                if not isinstance(function, dict):
                    continue
                arguments = function.get("arguments")
                if not isinstance(arguments, str):
                    continue
                try:
                    decoded = json.loads(arguments) if arguments.strip() else {}
                except json.JSONDecodeError:
                    logging.warning(
                        f"minimax_m3: tool call arguments are not valid JSON, "
                        f"rendering as empty object: {arguments!r}"
                    )
                    decoded = {}
                function["arguments"] = decoded if isinstance(decoded, dict) else {}
        return messages

    @staticmethod
    def _normalize_think_markers(message: dict) -> None:
        """Fold zero-width-escaped think markers in a history message.

        The template decides whether a past assistant turn contained reasoning by
        looking for a literal `</mm:think>` in its content. An escaped marker fails
        that check, so the template prepends a fresh marker while keeping the escaped
        one, and the markers accumulate turn after turn until they crowd out the real
        context.
        """
        content = message.get("content")
        if isinstance(content, str):
            message["content"] = normalize_mm_think_tags(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    part["text"] = normalize_mm_think_tags(part["text"])
        reasoning = message.get("reasoning_content")
        if isinstance(reasoning, str):
            message["reasoning_content"] = normalize_mm_think_tags(reasoning)

    @override
    def _create_detector(
        self, request: ChatCompletionRequest
    ) -> Optional[BaseFormatDetector]:
        return MiniMaxM3Detector() if request.tools else None

    @override
    def _create_reasoning_parser(
        self, request: ChatCompletionRequest
    ) -> Optional[ReasoningParser]:
        return ReasoningParser(model_type="minimax_m3")


register_renderer("minimax_m3", MiniMaxM3Renderer)
