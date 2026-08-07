"""Chat renderer for MiniMax-M2.

Reuses the bundled `chat_template.jinja` via the HF tokenizer (no custom
prompt construction needed) and plugs in `MinimaxM2Detector` for tool-call
parsing. M2's chat template emits `]~b]ai\\n<think>` immediately after the
assistant marker so reasoning is on by default; we register `[e~[` (the
message-end marker) as a stop word so generation halts at turn boundaries.
"""

from __future__ import annotations

import json
import logging
from typing import List, Optional

from typing_extensions import override

from rtp_llm.openai.api_datatype import ChatCompletionRequest
from rtp_llm.openai.renderer_factory_register import register_renderer
from rtp_llm.openai.renderers.reasoning_tool_base_renderer import (
    ReasoningToolBaseRenderer,
)
from rtp_llm.openai.renderers.sglang_helpers.function_call.base_format_detector import (
    BaseFormatDetector,
)
from rtp_llm.openai.renderers.sglang_helpers.function_call.minimax_m2_detector import (
    MinimaxM2Detector,
)
from rtp_llm.openai.renderers.sglang_helpers.reasoning_parser import ReasoningParser


class MinimaxM2Renderer(ReasoningToolBaseRenderer):
    """Renderer for MiniMaxAI/MiniMax-M2*."""

    @override
    def in_think_mode(self, request: ChatCompletionRequest):
        return True

    @override
    def _setup_stop_words(self):
        # `[e~[` is the per-message end marker emitted by chat_template.jinja
        # at the end of any role's segment. Without it the model keeps
        # generating into the next pseudo-turn.
        self.add_extra_stop_words(["[e~[", "<|endoftext|>"])

    @override
    def _create_reasoning_parser(
        self, request: ChatCompletionRequest
    ) -> Optional[ReasoningParser]:
        return ReasoningParser(model_type="minimax_m2", stream_reasoning=True)

    @override
    def _create_detector(
        self, request: ChatCompletionRequest
    ) -> Optional[BaseFormatDetector]:
        if request.tools:
            return MinimaxM2Detector()
        return None

    @override
    def _preprocess_messages(self, messages: List[dict]) -> List[dict]:
        # The OpenAI schema stores `tool_calls[*].function.arguments` as a JSON
        # *string*, but M2's chat_template iterates `arguments.items()`, so we
        # parse the string back into a dict for prior assistant tool_calls
        # carried in multi-turn history.
        out: List[dict] = []
        for msg in messages:
            if not msg.get("tool_calls"):
                out.append(msg)
                continue
            new_msg = dict(msg)
            new_calls = []
            for tc in msg["tool_calls"]:
                new_tc = dict(tc)
                fn = new_tc.get("function")
                if isinstance(fn, dict) and isinstance(fn.get("arguments"), str):
                    new_fn = dict(fn)
                    raw_args = new_fn["arguments"]
                    try:
                        new_fn["arguments"] = json.loads(raw_args)
                    except (TypeError, ValueError) as exc:
                        logging.warning(
                            "MinimaxM2Renderer: failed to JSON-parse tool_call "
                            "arguments (tool_call_id=%s, function=%s, raw=%r): %s",
                            new_tc.get("id"),
                            fn.get("name"),
                            raw_args[:200],
                            exc,
                        )
                        new_fn["arguments"] = raw_args
                    new_tc["function"] = new_fn
                new_calls.append(new_tc)
            new_msg["tool_calls"] = new_calls
            out.append(new_msg)
        return out


register_renderer("minimax_m2", MinimaxM2Renderer)
