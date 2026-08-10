import json
import logging
import re
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional

from typing_extensions import override

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.config.generate_config import GenerateConfig
from rtp_llm.openai.api_datatype import (
    ChatCompletionRequest,
    DeltaMessage,
    FinisheReason,
    FunctionCall,
    ToolCall,
    get_tool_choice_function_name,
)
from rtp_llm.openai.renderer_factory_register import register_renderer
from rtp_llm.openai.renderers.custom_renderer import (
    CustomChatRenderer,
    OutputDelta,
    RenderedInputs,
    StreamResponseObject,
    StreamStatus,
)
from rtp_llm.ops import MultimodalInput
from rtp_llm.server.backend_rpc_server_visitor import BackendRPCServerVisitor


_GRAMMAR_RESPONSE_FORMAT_TYPES = {
    "json_object",
    "json_schema",
    "regex",
    "ebnf",
    "structural_tag",
}


def _thinking_enabled(
    request: ChatCompletionRequest,
    template_kwargs: Optional[Dict[str, Any]] = None,
) -> bool:
    """Resolve K3 thinking controls in request-precedence order."""

    if request.thinking is not None:
        return request.thinking.type == "enabled"
    if request.enable_thinking is not None:
        return request.enable_thinking

    kwargs = template_kwargs
    if kwargs is None:
        kwargs = request.get_chat_template_kwargs() or {}
    if "thinking" in kwargs:
        return bool(kwargs["thinking"])
    if "enable_thinking" in kwargs:
        return bool(kwargs["enable_thinking"])

    if request.thinking_budget == 0:
        return False
    if (
        request.extra_configs is not None
        and request.extra_configs.max_thinking_tokens == 0
    ):
        return False

    if isinstance(request.reasoning_effort, str):
        effort = request.reasoning_effort.lower()
        if effort == "none":
            return False
        if effort in {"low", "high", "max"}:
            return True

    response_format = request.response_format
    if response_format is not None and response_format.type != "text":
        return False
    return True


def _uses_reasoning_channel(request: ChatCompletionRequest) -> bool:
    return _thinking_enabled(request)


class _KimiK3StreamStatus(StreamStatus):
    """Per-choice state for parsing K3's generated XTML channels."""

    def __init__(self, request: ChatCompletionRequest):
        super().__init__(request)
        self.xtml_pending = ""
        self.in_reasoning = _uses_reasoning_channel(request)
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
    _THINK_TO_RESPONSE = "<|close|>think<|sep|><|open|>response<|sep|>"
    _RESPONSE_CLOSE = "<|close|>response<|sep|>"
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
    def _reject_sampling_param(name: str, value: Any, expected: str) -> None:
        raise FtRuntimeException(
            ExceptionType.INVALID_PARAMS,
            f"Kimi K3 requires {name} {expected}, got {value!r}",
        )

    @classmethod
    def _apply_sampling_contract(
        cls,
        request: ChatCompletionRequest,
        generate_config: GenerateConfig,
        thinking: bool,
    ) -> None:
        fields_set = request.model_fields_set

        if "temperature" not in fields_set or request.temperature is None:
            generate_config.temperature = 1.0 if thinking else 0.6
        elif not 0.0 <= request.temperature <= 1.0:
            cls._reject_sampling_param("temperature", request.temperature, "in [0, 1]")
        else:
            generate_config.temperature = request.temperature

        if "top_p" not in fields_set or request.top_p is None:
            generate_config.top_p = 0.95
        else:
            effort = (
                request.thinking.effort
                if request.thinking is not None and request.thinking.effort is not None
                else request.reasoning_effort
            )
            effort_uses_top_p_one = (
                thinking
                and isinstance(effort, str)
                and effort.lower() in {"low", "high", "max"}
                and request.top_p == 1.0
            )
            if request.top_p != 0.95 and not effort_uses_top_p_one:
                cls._reject_sampling_param(
                    "top_p",
                    request.top_p,
                    "to be 0.95 (or 1.0 with thinking effort)",
                )
            generate_config.top_p = request.top_p

        for name in ("presence_penalty", "frequency_penalty"):
            value = getattr(request, name)
            if name in fields_set and value not in (None, 0, 0.0):
                cls._reject_sampling_param(name, value, "to be 0")
            setattr(generate_config, name, 0.0)

        if "n" in fields_set and request.n not in (None, 1):
            cls._reject_sampling_param("n", request.n, "to be 1")
        if "n" in fields_set and request.n == 1:
            generate_config.num_return_sequences = 1

    @staticmethod
    def _pending_prompt_token_count(tokenizer, thinking: bool) -> int:
        """Count the open generation channel excluded from tokenism usage."""

        channel = "think" if thinking else "response"
        channel_ids = tokenizer.encode(channel, add_special_tokens=False)
        if not isinstance(channel_ids, list) or not all(
            isinstance(token_id, int) for token_id in channel_ids
        ):
            raise TypeError(
                "Kimi K3 tokenizer.encode must return List[int], got "
                f"{type(channel_ids).__name__}"
            )
        # The opening XTML tag is <|open|>, ordinary channel text, <|sep|>.
        return len(channel_ids) + 2

    @staticmethod
    def _subtract_pending_prompt_tokens(
        response: StreamResponseObject, pending_tokens: int
    ) -> None:
        usage = response.usage
        if usage is None:
            return
        if usage.prompt_tokens < pending_tokens or usage.total_tokens < pending_tokens:
            raise RuntimeError(
                "Kimi K3 usage is shorter than its pending generation prompt: "
                f"prompt={usage.prompt_tokens}, total={usage.total_tokens}, "
                f"pending={pending_tokens}"
            )
        usage.prompt_tokens -= pending_tokens
        usage.total_tokens -= pending_tokens

    @override
    async def generate_choice(
        self,
        request_id: int,
        input_ids: List[int],
        mm_inputs: List[MultimodalInput],
        generate_config: GenerateConfig,
        backend_rpc_server_visitor: BackendRPCServerVisitor,
        request: ChatCompletionRequest,
        headers: Optional[Dict[str, str]] = None,
    ) -> AsyncGenerator[StreamResponseObject, None]:
        request_dict = self._request_dict(request)
        thinking = bool(self._template_kwargs(request, request_dict)["thinking"])
        pending_tokens = self._pending_prompt_token_count(self.tokenizer, thinking)
        async for response in super().generate_choice(
            request_id,
            input_ids,
            mm_inputs,
            generate_config,
            backend_rpc_server_visitor,
            request,
            headers,
        ):
            self._subtract_pending_prompt_tokens(response, pending_tokens)
            yield response

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
            return DeltaMessage(reasoning_content="", content="", tool_calls=tool_calls)

        think_to_response = cls._THINK_TO_RESPONSE
        response_closure = cls._RESPONSE_CLOSE
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
        return _uses_reasoning_channel(request)

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
        if request_dict.get("tools"):
            return request_dict["tools"]
        if request_dict.get("functions"):
            return [
                {"type": "function", "function": function}
                for function in request_dict["functions"]
            ]
        return None

    @classmethod
    def _all_tools(cls, request: ChatCompletionRequest) -> List[Dict[str, Any]]:
        request_dict = cls._request_dict(request)
        tools = list(cls._tools(request_dict) or [])
        for message in request_dict.get("messages") or []:
            tools.extend(message.get("tools") or [])
        return tools

    @staticmethod
    def _tool_choice_name(request: ChatCompletionRequest) -> Optional[str]:
        return get_tool_choice_function_name(request.tool_choice)

    @classmethod
    def _tool_choice_forces_tool(cls, request: ChatCompletionRequest) -> bool:
        return (
            request.tool_choice == "required"
            or cls._tool_choice_name(request) is not None
        )

    @classmethod
    def _active_tools(cls, request: ChatCompletionRequest) -> List[Dict[str, Any]]:
        tools = cls._all_tools(request)
        name = cls._tool_choice_name(request)
        if name is None:
            return tools

        selected = [tool for tool in tools if tool["function"]["name"] == name]
        if not selected:
            raise ValueError(f"tool_choice function {name!r} is not in tools")
        return selected

    @staticmethod
    def _escape_xtml_attr(value: str) -> str:
        return value.replace("&", "&amp;").replace('"', "&quot;")

    @classmethod
    def _build_tool_call_structural_tag(
        cls, request: ChatCompletionRequest
    ) -> Optional[Dict[str, Any]]:
        if not cls._tool_choice_forces_tool(request):
            return None

        tools = cls._active_tools(request)
        if not tools:
            raise ValueError("tool_choice requires at least one tool")

        call_tags = []
        for tool in tools:
            function = tool["function"]
            name = cls._escape_xtml_attr(function["name"])
            call_tags.append(
                {
                    "type": "tag",
                    "begin": (
                        f'<|open|>call tool="{name}" index="1"<|sep|>'
                        '<|open|>json type="object"<|sep|>'
                    ),
                    "content": {
                        "type": "json_schema",
                        "json_schema": function.get("parameters") or {},
                    },
                    "end": "<|close|>json<|sep|><|close|>call<|sep|>",
                }
            )

        return {
            "format": {
                "type": "tag",
                "begin": cls._RESPONSE_CLOSE + cls._TOOLS_OPEN,
                "content": {
                    "type": "tags_with_separator",
                    "tags": call_tags,
                    "separator": "",
                    "at_least_one": True,
                    # K3 call indices are embedded in the opening tag. Emit one
                    # schema-valid call so every alternative can use index 1.
                    "stop_after_first": True,
                },
                "end": cls._TOOLS_CLOSE,
            }
        }

    @staticmethod
    def _response_format_has_grammar(response_format: Any) -> bool:
        if response_format is None:
            return False
        if isinstance(response_format, str):
            try:
                response_format = json.loads(response_format)
            except ValueError:
                return True
        return not isinstance(response_format, dict) or response_format.get(
            "type"
        ) in _GRAMMAR_RESPONSE_FORMAT_TYPES

    @classmethod
    def _grammar_constraint_fields(cls, config: GenerateConfig) -> List[str]:
        fields = []
        if config.json_format:
            fields.append("json_format")
        if config.json_schema is not None:
            fields.append("json_schema")
        if config.regex is not None:
            fields.append("regex")
        if config.ebnf is not None:
            fields.append("ebnf")
        if config.structural_tag is not None:
            fields.append("structural_tag")
        if cls._response_format_has_grammar(config.response_format):
            fields.append("response_format")
        return fields

    @staticmethod
    def _clear_response_format_constraint(
        request: ChatCompletionRequest, config: GenerateConfig
    ) -> None:
        response_format = request.response_format
        if response_format is None or response_format.type == "text":
            return

        # A required tool call has no assistant content for response_format to
        # constrain. Its arguments are constrained by the tool schema instead.
        config.json_format = False
        config.json_schema = None
        config.regex = None
        config.ebnf = None
        config.response_format = None

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

        kwargs["thinking"] = _thinking_enabled(request, kwargs)
        kwargs.pop("enable_thinking", None)

        if request.thinking is not None and request.thinking.effort is not None:
            kwargs["thinking_effort"] = request.thinking.effort
        elif (
            request.reasoning_effort is not None
            and request.reasoning_effort.lower() != "none"
        ):
            kwargs["thinking_effort"] = request.reasoning_effort
        else:
            kwargs.pop("thinking_effort", None)
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
        thinking = _thinking_enabled(request)
        self._apply_sampling_contract(request, generate_config, thinking)
        generate_config.in_think_mode = thinking
        if not thinking:
            generate_config.max_thinking_tokens = 0

        structural_tag = self._build_tool_call_structural_tag(request)
        if structural_tag is not None:
            self._clear_response_format_constraint(request, generate_config)
            conflicts = self._grammar_constraint_fields(generate_config)
            if conflicts:
                raise FtRuntimeException(
                    ExceptionType.INVALID_PARAMS,
                    "tool_choice forced tool-call decoding conflicts with existing "
                    f"grammar constraint(s): {', '.join(conflicts)}",
                )

            generate_config.structural_tag = json.dumps(
                structural_tag, ensure_ascii=False, separators=(",", ":")
            )

        if generate_config.in_think_mode and self._grammar_constraint_fields(
            generate_config
        ):
            boundary_ids = self.tokenizer.encode(
                self._THINK_TO_RESPONSE, add_special_tokens=False
            )
            generate_config.end_think_token_ids = self._as_token_ids(boundary_ids)


register_renderer("kimi_k3", KimiK3Renderer)
