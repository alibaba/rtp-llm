"""Gemma4 chat renderer: canonical <|turn> protocol with thinking and tool calls.

Wires the checkpoint's own chat_template.jinja (Google 2026-07-09) into the
OpenAI-compatible serving path:

- Prompt side: renders the ckpt jinja template verbatim. The template needs
  `bos_token` in the render context (the tokenizer post_processor is empty, so
  the template is the sole BOS source) and a `raise_exception` jinja global.
  `enable_thinking` defaults from the request's thinking mode and can be
  overridden per request via chat_template_kwargs.
- Output side: parses the thinking channel `<|channel>thought\n...<channel|>`
  into reasoning_content and `<|tool_call>call:name{args}<tool_call|>` blocks
  into structured tool_calls. Args use the template's custom grammar: unquoted
  keys, strings wrapped as <|\"|>...<|\"|>, nested objects/arrays.
- Stop protocol: generation stops at <eos> (1), <turn|> (106) and the
  tool-response opener <|tool_response> (50) per generation_config
  eos_token_id=[1, 106, 50]; the latter two are ensured here so the model
  never free-runs into a tool response it must not fabricate.
- Vision: image content parts are rendered by the template as <|image|>
  placeholders and their urls are forwarded as multimodal inputs in order.
"""

import json
import logging
from typing import List, Optional, Tuple

from jinja2 import BaseLoader, Environment
from rtp_llm.openai.api_datatype import (
    ChatCompletionRequest,
    ContentPartTypeEnum,
    GPTToolDefinition,
)
from rtp_llm.openai.renderer_factory_register import register_renderer
from rtp_llm.openai.renderers.custom_renderer import CustomChatRenderer, RenderedInputs
from rtp_llm.openai.renderers.reasoning_tool_base_renderer import (
    ReasoningToolBaseRenderer,
)
from rtp_llm.openai.renderers.sglang_helpers.entrypoints.openai.protocol import Tool
from rtp_llm.openai.renderers.sglang_helpers.function_call.base_format_detector import (
    BaseFormatDetector,
    _forward_unknown_tools,
)
from rtp_llm.openai.renderers.sglang_helpers.function_call.core_types import (
    StreamingParseResult,
    StructureInfo,
    ToolCallItem,
    _GetInfoFunc,
)
from rtp_llm.openai.renderers.sglang_helpers.reasoning_parser import (
    BaseReasoningFormatDetector,
)
from rtp_llm.utils.base_model_datatypes import MMUrlType
from typing_extensions import override

logger = logging.getLogger(__name__)

THINK_OPEN = "<|channel>thought\n"
THINK_CLOSE = "<channel|>"
TOOL_CALL_OPEN = "<|tool_call>"
TOOL_CALL_CLOSE = "<tool_call|>"
TOOL_RESPONSE_OPEN = "<|tool_response>"
TURN_CLOSE = "<turn|>"
STR_DELIM = '<|"|>'


class _ArgsParser:
    """Recursive-descent parser for the gemma4 tool-call argument grammar.

    Mirrors the chat template's format_argument with escape_keys=False:
    unquoted keys at all depths, strings wrapped in <|\"|>...<|\"|>, arrays,
    nested objects, and bare scalars (null/true/false/numbers).
    """

    def __init__(self, text: str):
        self.s = text
        self.i = 0

    def parse(self):
        value = self._parse_value()
        self._skip_ws()
        if self.i != len(self.s):
            raise ValueError(f"trailing characters at {self.i}: {self.s[self.i:]!r}")
        return value

    def _skip_ws(self):
        while self.i < len(self.s) and self.s[self.i] in " \t\n":
            self.i += 1

    def _parse_value(self):
        self._skip_ws()
        if self.i >= len(self.s):
            raise ValueError("unexpected end of args")
        if self.s.startswith(STR_DELIM, self.i):
            return self._parse_string()
        c = self.s[self.i]
        if c == "{":
            return self._parse_object()
        if c == "[":
            return self._parse_array()
        return self._parse_scalar()

    def _parse_string(self) -> str:
        self.i += len(STR_DELIM)
        end = self.s.find(STR_DELIM, self.i)
        if end < 0:
            raise ValueError('unterminated <|"|> string')
        value = self.s[self.i : end]
        self.i = end + len(STR_DELIM)
        return value

    def _parse_key(self) -> str:
        self._skip_ws()
        start = self.i
        while self.i < len(self.s) and self.s[self.i] not in ":":
            self.i += 1
        if self.i >= len(self.s):
            raise ValueError(f"key without ':' at {start}")
        key = self.s[start : self.i].strip()
        self.i += 1  # consume ':'
        return key

    def _parse_object(self) -> dict:
        self.i += 1  # consume '{'
        out = {}
        self._skip_ws()
        if self.i < len(self.s) and self.s[self.i] == "}":
            self.i += 1
            return out
        while True:
            key = self._parse_key()
            out[key] = self._parse_value()
            self._skip_ws()
            if self.i >= len(self.s):
                raise ValueError("unterminated object")
            if self.s[self.i] == ",":
                self.i += 1
                continue
            if self.s[self.i] == "}":
                self.i += 1
                return out
            raise ValueError(f"unexpected {self.s[self.i]!r} in object at {self.i}")

    def _parse_array(self) -> list:
        self.i += 1  # consume '['
        out = []
        self._skip_ws()
        if self.i < len(self.s) and self.s[self.i] == "]":
            self.i += 1
            return out
        while True:
            out.append(self._parse_value())
            self._skip_ws()
            if self.i >= len(self.s):
                raise ValueError("unterminated array")
            if self.s[self.i] == ",":
                self.i += 1
                continue
            if self.s[self.i] == "]":
                self.i += 1
                return out
            raise ValueError(f"unexpected {self.s[self.i]!r} in array at {self.i}")

    def _parse_scalar(self):
        start = self.i
        while self.i < len(self.s) and self.s[self.i] not in ",}]":
            self.i += 1
        token = self.s[start : self.i].strip()
        if token == "null":
            return None
        if token == "true":
            return True
        if token == "false":
            return False
        try:
            return int(token)
        except ValueError:
            pass
        try:
            return float(token)
        except ValueError:
            pass
        return token


class Gemma4ToolCallDetector(BaseFormatDetector):
    """Detector for `<|tool_call>call:name{args}<tool_call|>` blocks.

    Complete segments are parsed with _ArgsParser and emitted once with full
    JSON parameters; text outside tool-call blocks passes through as normal
    content. Streaming buffers partial blocks (and partial marker prefixes)
    until the closing marker arrives.
    """

    def __init__(self):
        super().__init__()
        self.bot_token = TOOL_CALL_OPEN
        self.eot_token = TOOL_CALL_CLOSE

    def has_tool_call(self, text: str) -> bool:
        return self.bot_token in text

    def structure_info(self) -> _GetInfoFunc:
        return lambda name: StructureInfo(
            begin=f"<|tool_call>call:{name}{{",
            end="}<tool_call|>",
            trigger="<|tool_call>",
        )

    def _parse_segment(
        self, segment: str, tool_index: int, tools: List[Tool]
    ) -> Optional[ToolCallItem]:
        if not segment.startswith("call:"):
            logger.warning(
                f"gemma4 tool_call segment without 'call:' prefix: {segment!r}"
            )
            return None
        body = segment[len("call:") :]
        brace = body.find("{")
        if brace < 0:
            name, args = body.strip(), {}
        else:
            name = body[:brace].strip()
            try:
                args = _ArgsParser(body[brace:]).parse()
            except ValueError as e:
                logger.warning(f"gemma4 tool_call args parse failed: {e}; raw={body!r}")
                return None
        tool_indices = self._get_tool_indices(tools)
        if name not in tool_indices:
            logger.warning(f"Model attempted to call undefined function: {name}")
            if not _forward_unknown_tools():
                return None
        return ToolCallItem(
            tool_index=tool_index,
            name=name,
            parameters=json.dumps(args, ensure_ascii=False),
        )

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        calls: List[ToolCallItem] = []
        normal_parts: List[str] = []
        idx = 0
        tool_index = 0
        while True:
            start = text.find(self.bot_token, idx)
            if start < 0:
                normal_parts.append(text[idx:])
                break
            normal_parts.append(text[idx:start])
            end = text.find(self.eot_token, start + len(self.bot_token))
            if end < 0:
                normal_parts.append(text[start:])
                break
            call = self._parse_segment(
                text[start + len(self.bot_token) : end], tool_index, tools
            )
            if call is not None:
                calls.append(call)
                tool_index += 1
            idx = end + len(self.eot_token)
        return StreamingParseResult(normal_text="".join(normal_parts), calls=calls)

    def parse_streaming_increment(
        self, chunk: str, tools: List[Tool]
    ) -> StreamingParseResult:
        self._buffer += chunk
        out_calls: List[ToolCallItem] = []
        out_text = ""
        while True:
            start = self._buffer.find(self.bot_token)
            if start < 0:
                keep = self._partial_marker_len(self._buffer)
                out_text += self._buffer[: len(self._buffer) - keep]
                self._buffer = self._buffer[len(self._buffer) - keep :]
                break
            out_text += self._buffer[:start]
            end = self._buffer.find(self.eot_token, start + len(self.bot_token))
            if end < 0:
                self._buffer = self._buffer[start:]
                break
            segment = self._buffer[start + len(self.bot_token) : end]
            self.current_tool_id += 1
            call = self._parse_segment(segment, self.current_tool_id, tools)
            if call is not None:
                out_calls.append(call)
            self._buffer = self._buffer[end + len(self.eot_token) :]
        return StreamingParseResult(normal_text=out_text, calls=out_calls)

    def _partial_marker_len(self, s: str) -> int:
        for k in range(min(len(s), len(self.bot_token) - 1), 0, -1):
            if self.bot_token.startswith(s[-k:]):
                return k
        return 0


class Gemma4ReasoningParser:
    """Duck-typed drop-in for ReasoningParser with the gemma4 channel markers.

    The sglang base detector streams buffered text out immediately and only
    protects a buffer that is ENTIRELY a partial marker; a marker split after
    other content in the same chunk would leak through. parse_stream_chunk
    therefore holds back the longest trailing suffix that is a proper prefix
    of either marker and only feeds unambiguous text. A suffix still held at
    stream end is an incomplete protocol marker and is dropped, matching the
    base detector's own handling of partial prefixes.
    """

    _HOLD_MARKERS = (THINK_OPEN, THINK_CLOSE)

    def __init__(self, force_reasoning: bool = False):
        self.detector = BaseReasoningFormatDetector(
            think_start_token=THINK_OPEN,
            think_end_token=THINK_CLOSE,
            force_reasoning=force_reasoning,
        )
        self._hold = ""

    def parse_non_stream(self, full_text: str) -> Tuple[str, str]:
        ret = self.detector.detect_and_parse(full_text)
        return ret.reasoning_text, ret.normal_text

    def parse_stream_chunk(self, chunk_text: str) -> Tuple[str, str]:
        self._hold += chunk_text
        feed_len = len(self._hold)
        max_hold = max(len(m) for m in self._HOLD_MARKERS) - 1
        for k in range(min(len(self._hold), max_hold), 0, -1):
            suffix = self._hold[-k:]
            if any(m.startswith(suffix) for m in self._HOLD_MARKERS):
                feed_len = len(self._hold) - k
                break
        feed, self._hold = self._hold[:feed_len], self._hold[feed_len:]
        if not feed:
            return "", ""
        ret = self.detector.parse_streaming_increment(feed)
        return ret.reasoning_text, ret.normal_text


class Gemma4Renderer(ReasoningToolBaseRenderer):
    @override
    def _setup_stop_words(self):
        have = {tuple(w) for w in self.stop_words_id_list}
        for word in (TOOL_RESPONSE_OPEN, TURN_CLOSE):
            for ids in self.tokenize_words([word]):
                if tuple(ids) not in have:
                    self.add_extra_stop_word_ids([ids])
                    have.add(tuple(ids))

    @override
    def _customize_jinja_env(self, env: Environment) -> None:
        super()._customize_jinja_env(env)

        def raise_exception(message):
            raise Exception(message)

        env.globals["raise_exception"] = raise_exception

    @override
    def in_think_mode(self, request: ChatCompletionRequest) -> bool:
        thinking_enabled = super().in_think_mode(request)
        if request.chat_template_kwargs and request.chat_template_kwargs.get(
            "enable_thinking"
        ):
            thinking_enabled = True
        if (
            request.extra_configs
            and request.extra_configs.chat_template_kwargs
            and isinstance(request.extra_configs.chat_template_kwargs, dict)
            and request.extra_configs.chat_template_kwargs.get("enable_thinking")
        ):
            thinking_enabled = True
        return thinking_enabled

    @override
    def _build_prompt(self, request: ChatCompletionRequest) -> str:
        context = request.model_dump(exclude_none=True, mode="json")
        context["add_generation_prompt"] = True
        # The gemma4 tokenizer post_processor is empty: the chat template's
        # bos_token is the sole BOS source and must be provided explicitly.
        context["bos_token"] = self.tokenizer.bos_token or "<bos>"

        messages = self._preprocess_messages(context["messages"])
        context.update({"messages": messages})

        if request.chat_template_kwargs is not None:
            context.update(request.chat_template_kwargs)
        if (
            request.extra_configs is not None
            and request.extra_configs.chat_template_kwargs is not None
            and isinstance(request.extra_configs.chat_template_kwargs, dict)
        ):
            context.update(request.extra_configs.chat_template_kwargs)
        context.setdefault("enable_thinking", self.in_think_mode(request))

        env = Environment(
            loader=BaseLoader(),
            trim_blocks=True,
            lstrip_blocks=True,
            extensions=["jinja2.ext.do", "jinja2.ext.loopcontrols"],
        )
        self._customize_jinja_env(env)

        try:
            template = env.from_string(self.chat_template)
            return template.render(**context)
        except Exception as e:
            logging.error(f"gemma4 prompt template render failed: {e}")
            raise ValueError(f"Error rendering gemma4 prompt template: {e}")

    def _preprocess_messages(self, messages: List[dict]) -> List[dict]:
        # The template requires tool_calls[].function.arguments as a mapping
        # (it raises on strings); OpenAI clients send a JSON string.
        out = []
        for msg in messages:
            msg = dict(msg)
            content = msg.get("content")
            if isinstance(content, list):
                normalized_content = []
                for part in content:
                    part = dict(part)
                    if part.get("type") == "video_url":
                        part["type"] = "video"
                    normalized_content.append(part)
                msg["content"] = normalized_content
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                fixed_calls = []
                for tc in tool_calls:
                    tc = dict(tc)
                    fn = dict(tc.get("function") or {})
                    args = fn.get("arguments")
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except (json.JSONDecodeError, TypeError):
                            args = {}
                    fn["arguments"] = args if isinstance(args, dict) else {}
                    tc["function"] = fn
                    fixed_calls.append(tc)
                msg["tool_calls"] = fixed_calls
            out.append(msg)
        return out

    @override
    def render_chat(self, request: ChatCompletionRequest) -> RenderedInputs:
        prompt = self._build_prompt(request)
        input_ids = self.tokenizer.encode(prompt)
        urls, types = self._extract_mm_inputs(request)
        return RenderedInputs(
            input_ids=input_ids,
            rendered_prompt=prompt,
            input_urls=urls,
            input_urls_type=types,
        )

    def _extract_mm_inputs(
        self, request: ChatCompletionRequest
    ) -> Tuple[List[str], List[MMUrlType]]:
        urls: List[str] = []
        types: List[MMUrlType] = []
        for msg in request.messages:
            content = msg.content
            if not isinstance(content, list):
                continue
            for part in content:
                if part.type == ContentPartTypeEnum.image_url and part.image_url:
                    urls.append(part.image_url.url)
                    types.append(MMUrlType.IMAGE)
                elif part.type == ContentPartTypeEnum.video_url and part.video_url:
                    urls.append(part.video_url.url)
                    types.append(MMUrlType.VIDEO)
                elif part.type == ContentPartTypeEnum.audio_url:
                    raise ValueError("Gemma4 checkpoint has no audio tower")
        return urls, types

    @override
    def _create_reasoning_parser(
        self, request: ChatCompletionRequest
    ) -> Optional[Gemma4ReasoningParser]:
        # Thinking requests need full channel parsing. Tool requests need it
        # too: after a tool response with thinking disabled the prompt ends at
        # <tool_response|> without a pre-emitted empty thought block, so the
        # model emits `<|channel>thought\n<channel|>` itself before answering;
        # per the checkpoint response_template that block is protocol framing,
        # not content, and must be stripped into reasoning_content.
        if not (self.in_think_mode(request) or request.tools):
            return None
        # After a tool response the template leaves the thought channel open;
        # generation then continues inside the channel without re-emitting the
        # open marker, so reasoning must be assumed from the first token.
        try:
            force = self._build_prompt(request).endswith(THINK_OPEN)
        except Exception:
            force = False
        return Gemma4ReasoningParser(force_reasoning=force)

    @override
    def _create_detector(
        self, request: ChatCompletionRequest
    ) -> Optional[BaseFormatDetector]:
        if request.tools:
            return Gemma4ToolCallDetector()
        return None


register_renderer("gemma4", Gemma4Renderer)
