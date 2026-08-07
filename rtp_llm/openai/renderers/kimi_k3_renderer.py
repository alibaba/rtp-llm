import json
import logging
import re
import uuid
from typing import Any, Dict, List, Optional

from typing_extensions import override

from rtp_llm.config.generate_config import GenerateConfig
from rtp_llm.openai.api_datatype import (
    ChatCompletionRequest,
    DeltaMessage,
    FinisheReason,
    FunctionCall,
    ToolCall,
)
from rtp_llm.openai.renderer_factory_register import register_renderer
from rtp_llm.openai.renderers.custom_renderer import (
    CustomChatRenderer,
    OutputDelta,
    RenderedInputs,
    StreamStatus,
)


class _KimiK3StreamStatus(StreamStatus):
    """Per-choice state for parsing K3's generated XTML channels."""

    def __init__(self, request: ChatCompletionRequest):
        super().__init__(request)
        self.xtml_pending = ""
        self.in_reasoning = not request.disable_thinking()
        self.response_closed = False
        # XTML tools channel state (emitted after <|close|>response<|sep|>)
        self.tools_pending = ""
        self.in_tools = False
        self.tool_calls_seen = 0


class KimiK3Renderer(CustomChatRenderer):
    """Render Kimi K3's Python-defined XTML chat encoding.

    K3 deliberately has no Jinja ``chat_template``.  Its remote tokenizer
    renders a sequence of trusted structural segments and untrusted text
    segments, then encodes the two with different special-token policies.
    Calling ``encode`` on the final debug string would lose that distinction,
    so this renderer consumes the tokenizer's tokenized result directly.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_extra_stop_words(["<|end_of_msg|>"])

    _TOOLS_OPEN = "<|open|>tools<|sep|>"
    _TOOLS_CLOSE = "<|close|>tools<|sep|>"
    _XTML_CALL_RE = re.compile(
        r"<\|open\|>call tool=\"(?P<tool>[^\"]*)\" index=\"(?P<index>\d+)\"<\|sep\|>"
        r"(?P<body>.*?)<\|close\|>call<\|sep\|>",
        re.S,
    )
    _XTML_JSON_RE = re.compile(
        r"<\|open\|>json(?: type=\"[^\"]*\")?<\|sep\|>(?P<body>.*?)<\|close\|>json<\|sep\|>",
        re.S,
    )
    _XTML_ARG_RE = re.compile(
        r"<\|open\|>argument key=\"(?P<key>[^\"]*)\" type=\"(?P<type>[^\"]*)\"<\|sep\|>"
        r"(?P<body>.*?)<\|close\|>argument<\|sep\|>",
        re.S,
    )

    @staticmethod
    def _split_marker_prefix(text: str, marker: str) -> tuple[str, str]:
        """Keep the longest suffix that may be a split XTML marker."""

        max_prefix = min(len(text), len(marker) - 1)
        for length in range(max_prefix, 0, -1):
            if text.endswith(marker[:length]):
                return text[:-length], text[-length:]
        return text, ""

    @classmethod
    def _parse_xtml_delta(
        cls, status: _KimiK3StreamStatus, text: str, flush: bool = False
    ) -> DeltaMessage:
        """Split K3 reasoning/content channels and remove their XTML envelope.

        In thinking mode the generation prompt has already opened the
        ``think`` channel. The model then emits this transition before the
        visible answer::

            <|close|>think<|sep|><|open|>response<|sep|>

        Both thinking and non-thinking modes finish the visible channel with
        ``<|close|>response<|sep|>``. Tool calls follow in a separate
        ``<|open|>tools<|sep|> ... <|close|>tools<|sep|>`` channel.
        ``<|end_of_msg|>`` is removed by the generic stop path, but the
        other XTML tokens are ordinary generated tokens. Parse the exact
        channel boundaries here so they never leak into OpenAI ``content``,
        the reasoning text is exposed through ``reasoning_content``, and the
        tools channel is surfaced as ``tool_calls``. Partial markers are
        buffered across streaming chunks.
        """

        if status.response_closed:
            tool_calls = cls._parse_tools_delta(status, text, flush)
            return DeltaMessage(
                reasoning_content="", content="", tool_calls=tool_calls
            )

        think_to_response = "<|close|>think<|sep|><|open|>response<|sep|>"
        response_closure = "<|close|>response<|sep|>"
        combined = status.xtml_pending + text
        status.xtml_pending = ""
        reasoning = ""
        content = ""
        tool_calls: Optional[List[ToolCall]] = None

        if status.in_reasoning:
            transition_at = combined.find(think_to_response)
            if transition_at < 0:
                if flush:
                    reasoning = combined
                else:
                    reasoning, status.xtml_pending = cls._split_marker_prefix(
                        combined, think_to_response
                    )
                return DeltaMessage(reasoning_content=reasoning, content="")
            reasoning = combined[:transition_at]
            combined = combined[transition_at + len(think_to_response) :]
            status.in_reasoning = False

        closure_at = combined.find(response_closure)
        if closure_at >= 0:
            content = combined[:closure_at]
            status.response_closed = True
            remainder = combined[closure_at + len(response_closure) :]
            tool_calls = cls._parse_tools_delta(status, remainder, flush)
        elif flush:
            content = combined
        else:
            content, status.xtml_pending = cls._split_marker_prefix(
                combined, response_closure
            )

        return DeltaMessage(
            reasoning_content=reasoning, content=content, tool_calls=tool_calls
        )

    @classmethod
    def _parse_tools_delta(
        cls, status: _KimiK3StreamStatus, text: str, flush: bool = False
    ) -> Optional[List[ToolCall]]:
        """Buffer the XTML tools channel and emit complete tool calls.

        Tool call blocks are small, so buffer until the channel closes (or
        the stream flushes) and parse the whole block at once.
        """

        status.tools_pending += text
        buf = status.tools_pending

        if not status.in_tools:
            open_at = buf.find(cls._TOOLS_OPEN)
            if open_at < 0:
                if flush:
                    # Stream ended without a tools channel; drop leftovers
                    # (e.g. stray channel tokens must not surface as content).
                    status.tools_pending = ""
                return None
            status.in_tools = True
            buf = buf[open_at + len(cls._TOOLS_OPEN) :]

        close_at = buf.find(cls._TOOLS_CLOSE)
        if close_at < 0 and not flush:
            status.tools_pending = buf
            return None
        block = buf[:close_at] if close_at >= 0 else buf
        status.tools_pending = (
            buf[close_at + len(cls._TOOLS_CLOSE) :] if close_at >= 0 else ""
        )

        calls = cls._parse_tools_block(block)
        if calls:
            status.tool_calls_seen += len(calls)
        return calls or None

    @staticmethod
    def _unescape_attr(value: str) -> str:
        # Mirror of the tokenizer's _escape_attr_value
        return value.replace("&quot;", '"').replace("&amp;", "&")

    @staticmethod
    def _coerce_argument(value_type: str, body: str) -> Any:
        if value_type in ("number", "boolean", "null", "object", "array"):
            try:
                return json.loads(body)
            except ValueError:
                return body
        return body

    @classmethod
    def _parse_tools_block(cls, block: str) -> List[ToolCall]:
        calls: List[ToolCall] = []
        for match in cls._XTML_CALL_RE.finditer(block):
            name = cls._unescape_attr(match.group("tool"))
            body = match.group("body")
            json_match = cls._XTML_JSON_RE.search(body)
            if json_match:
                try:
                    arguments = json.loads(json_match.group("body"))
                except ValueError:
                    arguments = {}
            else:
                arguments = {}
                for arg_match in cls._XTML_ARG_RE.finditer(body):
                    key = cls._unescape_attr(arg_match.group("key"))
                    arguments[key] = cls._coerce_argument(
                        arg_match.group("type"), arg_match.group("body")
                    )
            calls.append(
                ToolCall(
                    index=int(match.group("index")) - 1,
                    id=f"call_{uuid.uuid4().hex[:24]}",
                    type="function",
                    function=FunctionCall(
                        name=name,
                        arguments=json.dumps(arguments, ensure_ascii=False),
                    ),
                )
            )
        return calls

    @override
    async def _create_status_list(
        self, n: int, request: ChatCompletionRequest
    ) -> List[StreamStatus]:
        return [_KimiK3StreamStatus(request) for _ in range(n)]

    @override
    async def _update_single_status(
        self,
        status: StreamStatus,
        output,
        max_new_tokens: int,
        stop_words_str: List[str],
        stop_word_slice_list: List[str],
        is_streaming: bool,
    ) -> OutputDelta:
        delta = await super()._update_single_status(
            status,
            output,
            max_new_tokens,
            stop_words_str,
            stop_word_slice_list,
            is_streaming,
        )
        if isinstance(status, _KimiK3StreamStatus) and isinstance(
            delta.output_str, str
        ):
            delta.output_str = self._parse_xtml_delta(
                status,
                delta.output_str,
                flush=status.finish_reason is not None,
            )
            # The engine only knows stop/length; report tool_calls when the
            # tools channel produced at least one call.
            if (
                status.finish_reason == FinisheReason.stop
                and status.tool_calls_seen > 0
            ):
                status.finish_reason = FinisheReason.tool_calls
        return delta

    @override
    def in_think_mode(self, request: ChatCompletionRequest) -> bool:
        return not request.disable_thinking()

    @override
    def should_process_think(self, request: ChatCompletionRequest) -> bool:
        del request
        # _parse_xtml_delta returns DeltaMessage with the channels already
        # separated; the generic <think> tag parser must not process it again.
        return False

    @staticmethod
    def _request_dict(request: ChatCompletionRequest) -> Dict[str, Any]:
        return request.model_dump(exclude_none=True, mode="json")

    @staticmethod
    def _ensure_text_only(messages: List[Dict[str, Any]]) -> None:
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            unsupported = [
                part.get("type")
                for part in content
                if isinstance(part, dict) and part.get("type") != "text"
            ]
            if unsupported:
                raise ValueError(
                    "Kimi K3 RTP-LLM bring-up is text-only; unsupported content "
                    f"types: {unsupported}"
                )

    @staticmethod
    def _tools(request_dict: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        merged: List[Dict[str, Any]] = []
        seen = set()

        def _extend(tools: List[Dict[str, Any]]) -> None:
            for tool in tools:
                key = json.dumps(tool, sort_keys=True, ensure_ascii=False)
                if key not in seen:
                    seen.add(key)
                    merged.append(tool)

        if request_dict.get("tools"):
            _extend(request_dict["tools"])
        elif request_dict.get("functions"):
            _extend(
                [
                    {"type": "function", "function": function}
                    for function in request_dict["functions"]
                ]
            )
        # KVV dynamic tools arrive embedded in messages (usually the system
        # message) instead of the request-level tools field.
        for message in request_dict.get("messages") or []:
            if message.get("tools"):
                _extend(message["tools"])
        return merged or None

    @staticmethod
    def _template_kwargs(
        request: ChatCompletionRequest, request_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}
        if request.chat_template_kwargs:
            kwargs.update(request.chat_template_kwargs)
        if (
            request.extra_configs is not None
            and request.extra_configs.chat_template_kwargs is not None
        ):
            kwargs.update(request.extra_configs.chat_template_kwargs)

        # RTP's public spelling is enable_thinking; K3's tokenizer spelling is
        # simply thinking.  An explicit request flag wins over template kwargs.
        if "enable_thinking" in kwargs and "thinking" not in kwargs:
            kwargs["thinking"] = bool(kwargs.pop("enable_thinking"))
        else:
            kwargs.pop("enable_thinking", None)
        kwargs.setdefault("thinking", not request.disable_thinking())
        if request.enable_thinking is not None:
            kwargs["thinking"] = request.enable_thinking

        if request.reasoning_effort is not None:
            kwargs["thinking_effort"] = request.reasoning_effort
        if request_dict.get("tool_choice") is not None:
            kwargs["tool_choice"] = request_dict["tool_choice"]
        if request_dict.get("response_format") is not None:
            kwargs["response_format"] = request_dict["response_format"]
        return kwargs

    @staticmethod
    def _as_token_ids(value: Any) -> List[int]:
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, list) and value and isinstance(value[0], list):
            if len(value) != 1:
                raise ValueError(
                    "Kimi K3 renderer expected one conversation, got a token batch"
                )
            value = value[0]
        if not isinstance(value, list) or not all(
            isinstance(token_id, int) for token_id in value
        ):
            raise TypeError(
                "Kimi K3 tokenizer.apply_chat_template must return List[int] "
                f"for a single request, got {type(value).__name__}"
            )
        return value

    @override
    def render_chat(self, request: ChatCompletionRequest) -> RenderedInputs:
        request_dict = self._request_dict(request)
        messages = request_dict["messages"]
        self._ensure_text_only(messages)
        tools = self._tools(request_dict)
        # The message-level tools key is an RTP extension; the tokenizer
        # template only understands the merged tools parameter.
        messages = [
            {key: value for key, value in message.items() if key != "tools"}
            for message in messages
        ]
        template_kwargs = self._template_kwargs(request, request_dict)

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=True,
            add_generation_prompt=True,
            **template_kwargs,
        )
        rendered_prompt = self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
            **template_kwargs,
        )
        if not isinstance(rendered_prompt, str):
            raise TypeError(
                "Kimi K3 tokenizer.apply_chat_template must return str when "
                f"tokenize=False, got {type(rendered_prompt).__name__}"
            )
        token_ids = self._as_token_ids(input_ids)
        logging.debug("Kimi K3 rendered %d XTML prompt tokens", len(token_ids))
        return RenderedInputs(input_ids=token_ids, rendered_prompt=rendered_prompt)

    @override
    def apply_chat_completion_constraints(
        self, request: ChatCompletionRequest, generate_config: GenerateConfig
    ) -> None:
        del request, generate_config
        # tool_choice (auto/none/required/named) is forwarded to the
        # tokenizer template through _template_kwargs; nothing to reject.


register_renderer("kimi_k3", KimiK3Renderer)
