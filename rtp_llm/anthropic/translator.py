import asyncio
import json
import logging
import uuid
from typing import (
    Any,
    AsyncGenerator,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import orjson

from rtp_llm.anthropic.api_datatype import (
    AnthropicContentBlock,
    AnthropicCountTokensRequest,
    AnthropicImageBlock,
    AnthropicImageSource,
    AnthropicInputContentBlock,
    AnthropicMessage,
    AnthropicMessagesRequest,
    AnthropicOutputTextBlock,
    AnthropicOutputThinkingBlock,
    AnthropicOutputToolUseBlock,
    AnthropicRedactedThinkingBlock,
    AnthropicSystemTextBlock,
    AnthropicTextBlock,
    AnthropicThinkingBlock,
    AnthropicToolChoiceAny,
    AnthropicToolChoiceAuto,
    AnthropicToolChoiceNone,
    AnthropicToolChoiceTool,
    AnthropicToolResultBlock,
    AnthropicToolUseBlock,
    AnthropicUsage,
)
from rtp_llm.anthropic.errors import (
    AnthropicTranslationError,
    exception_to_anthropic_http,
)
from rtp_llm.config.generate_config import GenerateConfig, ThinkingMode
from rtp_llm.config.response_format import ResponseFormat, ResponseFormatJSONSchema
from rtp_llm.openai.api_datatype import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
    ChatMessage,
    ContentPart,
    ContentPartTypeEnum,
    DebugInfo,
    FinisheReason,
    FunctionCall,
    GPTFunctionDefinition,
    GPTToolDefinition,
    ImageURL,
    RoleEnum,
    ThinkingConfig,
    ToolCall,
    UsageInfo,
)

AnthropicRequestType = Union[AnthropicMessagesRequest, AnthropicCountTokensRequest]
AnthropicStopReason = Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"]


def _image_source_to_url(source: AnthropicImageSource) -> str:
    if source.type == "base64":
        return f"data:{source.media_type};base64,{source.data}"
    return source.url or ""


def _collapse_parts(parts: List[ContentPart]) -> Union[str, List[ContentPart]]:
    """Collapse text-only blocks into a plain string so text-only Jinja templates succeed."""
    if all(part.type == ContentPartTypeEnum.text for part in parts):
        if not parts:
            return ""
        if len(parts) == 1:
            return parts[0].text or ""
        return "\n\n".join(part.text or "" for part in parts)
    return parts


def _extract_system_text(
    system: Optional[Union[str, Sequence[AnthropicSystemTextBlock]]],
) -> Optional[str]:
    if system is None:
        return None
    if isinstance(system, str):
        return system if system != "" else None
    texts = [block.text for block in system if block.text != ""]
    if not texts:
        return None
    return texts[0] if len(texts) == 1 else "\n\n".join(texts)


def _apply_tool_error_marker(
    translated: Union[str, List[ContentPart]],
) -> Union[str, List[ContentPart]]:
    if isinstance(translated, str):
        return f"[Tool Error]: {translated}" if translated else "[Tool Error]"
    if not translated:
        return "[Tool Error]"
    parts = list(translated)
    first = parts[0]
    if first.type == ContentPartTypeEnum.text:
        prefixed = f"[Tool Error]: {first.text}" if first.text else "[Tool Error]"
        parts[0] = ContentPart(type=ContentPartTypeEnum.text, text=prefixed)
    else:
        parts.insert(
            0, ContentPart(type=ContentPartTypeEnum.text, text="[Tool Error]: ")
        )
    return parts


def _translate_tool_result_content(
    block: AnthropicToolResultBlock,
) -> Union[str, List[ContentPart]]:
    content = block.content
    if content is None:
        translated: Union[str, List[ContentPart]] = ""
    elif isinstance(content, str):
        translated = content
    else:
        parts: List[ContentPart] = []
        for item in content:
            if isinstance(item, AnthropicTextBlock):
                parts.append(ContentPart(type=ContentPartTypeEnum.text, text=item.text))
            elif isinstance(item, AnthropicImageBlock):
                parts.append(
                    ContentPart(
                        type=ContentPartTypeEnum.image_url,
                        image_url=ImageURL(url=_image_source_to_url(item.source)),
                    )
                )
        translated = _collapse_parts(parts)

    if block.is_error is True:
        return _apply_tool_error_marker(translated)
    return translated


def _translate_user_message(
    content: Union[str, Sequence[AnthropicInputContentBlock]],
) -> List[ChatMessage]:
    if isinstance(content, str):
        return [ChatMessage(role=RoleEnum.user, content=content)]

    tool_messages: List[ChatMessage] = []
    user_parts: List[ContentPart] = []
    for block in content:
        if isinstance(block, AnthropicToolResultBlock):
            tool_messages.append(
                ChatMessage(
                    role=RoleEnum.tool,
                    tool_call_id=block.tool_use_id,
                    content=_translate_tool_result_content(block),
                )
            )
        elif isinstance(block, AnthropicTextBlock):
            user_parts.append(
                ContentPart(type=ContentPartTypeEnum.text, text=block.text)
            )
        elif isinstance(block, AnthropicImageBlock):
            user_parts.append(
                ContentPart(
                    type=ContentPartTypeEnum.image_url,
                    image_url=ImageURL(url=_image_source_to_url(block.source)),
                )
            )
        else:
            raise AnthropicTranslationError(
                f"Block type '{block.type}' is not allowed in 'user' messages"
            )

    messages: List[ChatMessage] = list(tool_messages)
    has_meaningful_user_content = any(
        part.type != ContentPartTypeEnum.text or bool(part.text) for part in user_parts
    )
    if not messages or has_meaningful_user_content:
        messages.append(
            ChatMessage(
                role=RoleEnum.user,
                content=_collapse_parts(user_parts),
            )
        )
    return messages


def _translate_assistant_message(
    content: Union[str, Sequence[AnthropicInputContentBlock]],
) -> ChatMessage:
    if isinstance(content, str):
        return ChatMessage(role=RoleEnum.assistant, content=content)

    reasoning_parts: List[str] = []
    text_parts: List[str] = []
    tool_calls: List[ToolCall] = []

    for block in content:
        if isinstance(block, AnthropicThinkingBlock):
            if block.thinking:
                reasoning_parts.append(block.thinking)
        elif isinstance(block, AnthropicRedactedThinkingBlock):
            # Redacted thinking payloads are opaque and not plaintext prompt tokens.
            continue
        elif isinstance(block, AnthropicTextBlock):
            text_parts.append(block.text)
        elif isinstance(block, AnthropicToolUseBlock):
            tool_calls.append(
                ToolCall(
                    index=len(tool_calls),
                    id=block.id,
                    type="function",
                    function=FunctionCall(
                        name=block.name,
                        arguments=orjson.dumps(block.input).decode("utf-8"),
                    ),
                )
            )
        else:
            raise AnthropicTranslationError(
                f"Block type '{block.type}' is not allowed in 'assistant' messages"
            )

    reasoning_content = "\n".join(reasoning_parts) if reasoning_parts else None
    if text_parts:
        message_text: Optional[str] = (
            text_parts[0] if len(text_parts) == 1 else "".join(text_parts)
        )
    elif tool_calls:
        message_text = None
    else:
        message_text = ""

    return ChatMessage(
        role=RoleEnum.assistant,
        content=message_text,
        reasoning_content=reasoning_content,
        tool_calls=tool_calls or None,
    )


def translate_anthropic_request(request: AnthropicRequestType) -> ChatCompletionRequest:
    openai_messages: List[ChatMessage] = []

    system_text = _extract_system_text(request.system)
    if system_text is not None:
        openai_messages.append(ChatMessage(role=RoleEnum.system, content=system_text))

    for msg in request.messages:
        if msg.role == "user":
            openai_messages.extend(_translate_user_message(msg.content))
        else:
            openai_messages.append(_translate_assistant_message(msg.content))

    openai_tools: Optional[List[GPTToolDefinition]] = None
    if request.tools:
        openai_tools = [
            GPTToolDefinition(
                type="function",
                function=GPTFunctionDefinition(
                    name=tool.name,
                    description=tool.description or "",
                    parameters=tool.input_schema,
                ),
            )
            for tool in request.tools
        ]

    openai_tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    parallel_tool_calls: Optional[bool] = None
    if request.tool_choice is not None:
        if isinstance(request.tool_choice, AnthropicToolChoiceAuto):
            openai_tool_choice = "auto"
            if request.tool_choice.disable_parallel_tool_use is not None:
                parallel_tool_calls = not request.tool_choice.disable_parallel_tool_use
        elif isinstance(request.tool_choice, AnthropicToolChoiceAny):
            openai_tool_choice = "required"
            if request.tool_choice.disable_parallel_tool_use is not None:
                parallel_tool_calls = not request.tool_choice.disable_parallel_tool_use
        elif isinstance(request.tool_choice, AnthropicToolChoiceNone):
            openai_tool_choice = "none"
            # Omit tool definitions when tool calling is explicitly disabled so renderers
            # without tool_choice='none' gating do not expose tools to the model.
            openai_tools = None
        elif isinstance(request.tool_choice, AnthropicToolChoiceTool):
            openai_tool_choice = {
                "type": "function",
                "function": {"name": request.tool_choice.name},
            }
            if request.tool_choice.disable_parallel_tool_use is not None:
                parallel_tool_calls = not request.tool_choice.disable_parallel_tool_use

    openai_thinking: Optional[ThinkingConfig] = None
    enable_thinking: Optional[bool] = None
    thinking_budget: Optional[int] = None
    chat_template_kwargs: Optional[Dict[str, Any]] = None
    extra_configs: Optional[GenerateConfig] = None

    if request.thinking is not None:
        if request.thinking.type == "enabled":
            openai_thinking = ThinkingConfig(type="enabled")
            enable_thinking = True
            thinking_budget = request.thinking.budget_tokens
            chat_template_kwargs = {"enable_thinking": True}
        elif request.thinking.type == "disabled":
            openai_thinking = ThinkingConfig(type="disabled")
            enable_thinking = False
            thinking_budget = 0
            chat_template_kwargs = {"enable_thinking": False}
        elif request.thinking.type == "adaptive":
            thinking_budget = request.thinking.budget_tokens
            extra_configs = GenerateConfig(thinking_mode=ThinkingMode.ADAPTIVE)
        else:
            raise AnthropicTranslationError(
                f"Unsupported thinking.type: {request.thinking.type!r}"
            )

    response_format: Optional[ResponseFormat] = None
    reasoning_effort: Optional[str] = None
    if request.output_config is not None:
        if request.output_config.effort is not None:
            reasoning_effort = request.output_config.effort
        if request.output_config.format is not None:
            fmt = request.output_config.format
            if fmt.type == "json_schema":
                if not fmt.schema_:
                    raise AnthropicTranslationError(
                        "output_config.format.schema must be non-empty for type 'json_schema'"
                    )
                response_format = ResponseFormat(
                    type="json_schema",
                    json_schema=ResponseFormatJSONSchema(
                        name=fmt.name or "response_schema",
                        schema_=fmt.schema_,
                        strict=fmt.strict,
                    ),
                )
            elif fmt.type == "text":
                response_format = ResponseFormat(type="text")

    return ChatCompletionRequest(
        model=request.model,
        messages=openai_messages,
        tools=openai_tools,
        tool_choice=openai_tool_choice,
        parallel_tool_calls=parallel_tool_calls,
        reasoning_effort=reasoning_effort,
        thinking=openai_thinking,
        enable_thinking=enable_thinking,
        thinking_budget=thinking_budget,
        chat_template_kwargs=chat_template_kwargs,
        extra_configs=extra_configs,
        temperature=request.temperature,
        top_p=request.top_p if request.top_p is not None else 1.0,
        top_k=request.top_k,
        max_tokens=request.max_tokens,
        stop=list(request.stop_sequences) if request.stop_sequences else [],
        stream=request.stream,
        user=request.metadata.user_id if request.metadata else None,
        seed=request.seed,
        response_format=response_format,
    )


def build_anthropic_usage(
    usage: Optional[UsageInfo],
    *,
    output_tokens_override: Optional[int] = None,
) -> AnthropicUsage:
    if usage is None:
        return AnthropicUsage(
            input_tokens=0,
            output_tokens=output_tokens_override or 0,
            cache_read_input_tokens=0,
        )
    cached_tokens = 0
    if (
        usage.prompt_tokens_details is not None
        and usage.prompt_tokens_details.cached_tokens is not None
    ):
        cached_tokens = max(0, int(usage.prompt_tokens_details.cached_tokens))
    prompt_tokens = max(0, int(usage.prompt_tokens or 0))
    if cached_tokens > 0:
        input_tokens = max(0, prompt_tokens - cached_tokens)
    else:
        input_tokens = prompt_tokens
    output_tokens = (
        max(0, int(output_tokens_override))
        if output_tokens_override is not None
        else max(0, int(usage.completion_tokens or 0))
    )
    return AnthropicUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_input_tokens=cached_tokens,
    )


def _parse_tool_arguments(
    raw_args: Optional[Union[str, Dict[str, Any]]],
) -> Dict[str, Any]:
    if isinstance(raw_args, dict):
        return raw_args
    if not isinstance(raw_args, str) or not raw_args.strip():
        return {}
    try:
        parsed = json.loads(raw_args)
    except (json.JSONDecodeError, TypeError):
        logging.warning("Failed to parse tool_call arguments as JSON: %r", raw_args)
        return {}
    if isinstance(parsed, dict):
        return parsed
    return {"value": parsed}


def _normalize_finish_reason(
    finish_reason: Optional[Union[FinisheReason, str]],
) -> Optional[str]:
    if finish_reason is None:
        return None
    if isinstance(finish_reason, FinisheReason):
        return finish_reason.value
    return str(finish_reason)


def _filter_matched_stop_sequence(
    matched: Optional[str],
    stop_sequences: Optional[Sequence[str]],
) -> Optional[str]:
    if not matched:
        return None
    if stop_sequences is not None and matched not in stop_sequences:
        return None
    return matched


def _detect_stop_sequence_from_debug_info(
    resp: Optional[Union[ChatCompletionResponse, ChatCompletionStreamResponse]],
    stop_sequences: Optional[Sequence[str]],
) -> Optional[str]:
    if not stop_sequences or resp is None or not isinstance(resp.debug_info, DebugInfo):
        return None
    raw_outputs = resp.debug_info.raw_output
    if not raw_outputs:
        return None
    raw_text = raw_outputs[0]
    if not isinstance(raw_text, str):
        return None
    for seq in stop_sequences:
        if seq and raw_text.endswith(seq):
            return seq
    return None


def _map_stop_reason(
    finish_reason: Optional[str],
    has_tool_use: bool,
    matched_stop_sequence: Optional[str] = None,
) -> Tuple[AnthropicStopReason, Optional[str]]:
    if finish_reason == FinisheReason.length.value:
        return "max_tokens", None
    if matched_stop_sequence is not None:
        return "stop_sequence", matched_stop_sequence
    if (
        finish_reason
        in (FinisheReason.tool_calls.value, FinisheReason.function_call.value)
        or has_tool_use
    ):
        return "tool_use", None
    return "end_turn", None


def translate_openai_response(
    response: Union[ChatCompletionResponse, Dict[str, Any]],
    model: str = "",
    stop_sequences: Optional[Sequence[str]] = None,
) -> AnthropicMessage:
    if isinstance(response, dict):
        response = ChatCompletionResponse.model_validate(response)
    choice = response.choices[0] if response.choices else None
    blocks: List[AnthropicContentBlock] = []
    has_tool_use = False
    finish_reason: Optional[str] = None
    matched_stop: Optional[str] = None

    if choice is not None:
        finish_reason = _normalize_finish_reason(choice.finish_reason)
        matched_stop = _filter_matched_stop_sequence(
            choice.matched_stop_sequence, stop_sequences
        )
        msg = choice.message
        if msg.reasoning_content:
            blocks.append(
                AnthropicOutputThinkingBlock(
                    thinking=msg.reasoning_content, signature=""
                )
            )
        if msg.content:
            blocks.append(AnthropicOutputTextBlock(text=msg.content))

        tool_calls = list(msg.tool_calls or [])
        if not tool_calls and msg.function_call is not None:
            tool_calls = [
                ToolCall(
                    index=0,
                    id=f"toolu_{uuid.uuid4().hex[:24]}",
                    type="function",
                    function=msg.function_call,
                )
            ]

        for tc in tool_calls:
            has_tool_use = True
            tool_id = tc.id or f"toolu_{uuid.uuid4().hex[:24]}"
            tool_name = tc.function.name if tc.function and tc.function.name else ""
            raw_args = tc.function.arguments if tc.function else None
            blocks.append(
                AnthropicOutputToolUseBlock(
                    id=tool_id,
                    name=tool_name,
                    input=_parse_tool_arguments(raw_args),
                )
            )

    if not blocks:
        blocks.append(AnthropicOutputTextBlock(text=""))

    if matched_stop is None:
        matched_stop = _detect_stop_sequence_from_debug_info(response, stop_sequences)
    stop_reason, stop_sequence = _map_stop_reason(
        finish_reason, has_tool_use, matched_stop
    )

    return AnthropicMessage(
        id=f"msg_{uuid.uuid4().hex[:24]}",
        model=model or response.model or "",
        content=blocks,
        stop_reason=stop_reason,
        stop_sequence=stop_sequence,
        usage=build_anthropic_usage(response.usage),
    )


def format_anthropic_sse(event: str, data: Dict[str, Any]) -> str:
    payload = orjson.dumps(data).decode("utf-8")
    return f"event: {event}\ndata: {payload}\n\n"


class AnthropicStreamTranslator:
    """Stateful translator converting OpenAI ChatCompletionStreamResponse chunks into Anthropic SSE events."""

    def __init__(
        self,
        model: str = "",
        message_id: Optional[str] = None,
        stop_sequences: Optional[Sequence[str]] = None,
    ) -> None:
        self.model = model
        self.message_id = message_id or f"msg_{uuid.uuid4().hex[:24]}"
        self._stop_sequences: Optional[List[str]] = (
            list(stop_sequences) if stop_sequences is not None else None
        )
        self._started = False
        self._initial_usage: Optional[UsageInfo] = None
        self._latest_usage: Optional[UsageInfo] = None
        self._finish_reason: Optional[str] = None
        self._matched_stop_sequence: Optional[str] = None
        self._next_block_index = 0
        self._active_block_type: Optional[str] = None
        self._active_block_index: Optional[int] = None
        self._active_tool_index: Optional[int] = None
        self._active_tool_id: Optional[str] = None
        self._active_tool_name: Optional[str] = None
        self._emitted_tool_use = False

    def _emit_message_start(self) -> List[str]:
        if self._started:
            return []
        self._started = True
        usage = build_anthropic_usage(
            self._initial_usage or self._latest_usage,
            output_tokens_override=0,
        )
        return [
            format_anthropic_sse(
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": self.message_id,
                        "type": "message",
                        "role": "assistant",
                        "model": self.model,
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": usage.model_dump(exclude_none=True),
                    },
                },
            )
        ]

    def _close_active_block(self) -> List[str]:
        if self._active_block_type is None or self._active_block_index is None:
            return []
        events: List[str] = []
        if self._active_block_type == "thinking":
            events.append(
                format_anthropic_sse(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": self._active_block_index,
                        "delta": {"type": "signature_delta", "signature": ""},
                    },
                )
            )
        events.append(
            format_anthropic_sse(
                "content_block_stop",
                {
                    "type": "content_block_stop",
                    "index": self._active_block_index,
                },
            )
        )
        self._active_block_type = None
        self._active_block_index = None
        self._active_tool_index = None
        self._active_tool_id = None
        self._active_tool_name = None
        return events

    def _open_block(
        self, block_type: str, content_block: Dict[str, Any]
    ) -> Tuple[int, List[str]]:
        events = self._close_active_block()
        block_index = self._next_block_index
        self._next_block_index += 1
        self._active_block_type = block_type
        self._active_block_index = block_index
        events.append(
            format_anthropic_sse(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": block_index,
                    "content_block": content_block,
                },
            )
        )
        return block_index, events

    def _process_tool_calls(self, tool_calls: List[ToolCall]) -> List[str]:
        events: List[str] = []
        for pos, tc in enumerate(tool_calls):
            tc_index = tc.index if tc.index is not None else pos
            tc_name = tc.function.name if tc.function and tc.function.name else None
            tc_id = tc.id
            raw_args = tc.function.arguments if tc.function else None

            same_active_tool_identity = (
                tc_id is None
                or self._active_tool_id is None
                or tc_id == self._active_tool_id
            ) and (
                tc_name is None
                or self._active_tool_name is None
                or tc_name == self._active_tool_name
            )
            # Continue the open tool_use block whenever index matches and neither
            # id nor name conflicts with the active block.
            is_continuation = (
                self._active_block_type == "tool_use"
                and self._active_tool_index == tc_index
                and same_active_tool_identity
            )
            if not is_continuation:
                self._emitted_tool_use = True
                tool_id = tc_id or f"toolu_{uuid.uuid4().hex[:24]}"
                _, open_events = self._open_block(
                    "tool_use",
                    {
                        "type": "tool_use",
                        "id": tool_id,
                        "name": tc_name or "",
                        "input": {},
                    },
                )
                self._active_tool_index = tc_index
                self._active_tool_id = tool_id
                self._active_tool_name = tc_name
                events.extend(open_events)

            if raw_args is not None:
                if isinstance(raw_args, dict):
                    partial_json = orjson.dumps(raw_args).decode("utf-8")
                else:
                    partial_json = str(raw_args)
                if partial_json != "":
                    events.append(
                        format_anthropic_sse(
                            "content_block_delta",
                            {
                                "type": "content_block_delta",
                                "index": self._active_block_index,
                                "delta": {
                                    "type": "input_json_delta",
                                    "partial_json": partial_json,
                                },
                            },
                        )
                    )
        return events

    def process_chunk(self, chunk: ChatCompletionStreamResponse) -> List[str]:
        events: List[str] = []
        if chunk.usage is not None:
            self._latest_usage = chunk.usage
            if self._initial_usage is None:
                self._initial_usage = chunk.usage

        choice = chunk.choices[0] if chunk.choices else None
        has_payload = False
        if choice is not None:
            delta = choice.delta
            has_payload = bool(
                delta.reasoning_content
                or delta.content
                or delta.tool_calls
                or delta.function_call is not None
                or choice.finish_reason is not None
            )

        if not self._started:
            # _generate_first yields a role-only chunk with usage=None; wait for the next
            # chunk when possible so message_start reports real prompt input_tokens.
            if has_payload or chunk.usage is not None:
                events.extend(self._emit_message_start())

        if choice is None:
            return events

        delta = choice.delta
        if delta.reasoning_content:
            events.extend(self._emit_message_start())
            if self._active_block_type != "thinking":
                _, open_events = self._open_block(
                    "thinking", {"type": "thinking", "thinking": ""}
                )
                events.extend(open_events)
            events.append(
                format_anthropic_sse(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": self._active_block_index,
                        "delta": {
                            "type": "thinking_delta",
                            "thinking": delta.reasoning_content,
                        },
                    },
                )
            )

        if delta.content:
            events.extend(self._emit_message_start())
            if self._active_block_type != "text":
                _, open_events = self._open_block("text", {"type": "text", "text": ""})
                events.extend(open_events)
            events.append(
                format_anthropic_sse(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": self._active_block_index,
                        "delta": {
                            "type": "text_delta",
                            "text": delta.content,
                        },
                    },
                )
            )

        tool_calls = list(delta.tool_calls or [])
        if not tool_calls and delta.function_call is not None:
            tool_calls = [
                ToolCall(
                    index=0,
                    id=f"toolu_{uuid.uuid4().hex[:24]}",
                    type="function",
                    function=delta.function_call,
                )
            ]
        if tool_calls:
            events.extend(self._emit_message_start())
            events.extend(self._process_tool_calls(tool_calls))

        if (
            choice.matched_stop_sequence is not None
            and self._matched_stop_sequence is None
        ):
            self._matched_stop_sequence = _filter_matched_stop_sequence(
                choice.matched_stop_sequence, self._stop_sequences
            )

        if self._matched_stop_sequence is None:
            self._matched_stop_sequence = _detect_stop_sequence_from_debug_info(
                chunk, self._stop_sequences
            )

        if choice.finish_reason is not None:
            events.extend(self._emit_message_start())
            self._finish_reason = _normalize_finish_reason(choice.finish_reason)

        return events

    def finalize(self) -> List[str]:
        events: List[str] = []
        if not self._started:
            events.extend(self._emit_message_start())

        if self._next_block_index == 0:
            idx, open_events = self._open_block("text", {"type": "text", "text": ""})
            events.extend(open_events)
            events.append(
                format_anthropic_sse(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": idx,
                        "delta": {"type": "text_delta", "text": ""},
                    },
                )
            )

        events.extend(self._close_active_block())

        stop_reason, stop_sequence = _map_stop_reason(
            self._finish_reason, self._emitted_tool_use, self._matched_stop_sequence
        )
        output_tokens = (
            max(0, int(self._latest_usage.completion_tokens or 0))
            if self._latest_usage is not None
            else 0
        )
        events.append(
            format_anthropic_sse(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {
                        "stop_reason": stop_reason,
                        "stop_sequence": stop_sequence,
                    },
                    "usage": {
                        "output_tokens": output_tokens,
                    },
                },
            )
        )
        events.append(format_anthropic_sse("message_stop", {"type": "message_stop"}))
        return events

    async def stream(
        self,
        response_generator: AsyncGenerator[ChatCompletionStreamResponse, None],
    ) -> AsyncGenerator[str, None]:
        try:
            async for chunk in response_generator:
                for event_str in self.process_chunk(chunk):
                    yield event_str
            for event_str in self.finalize():
                yield event_str
        except (asyncio.CancelledError, GeneratorExit):
            raise
        except BaseException as exc:
            _, error_payload = exception_to_anthropic_http(exc)
            yield format_anthropic_sse("error", error_payload)


def anthropic_stream_error_event(exc: BaseException) -> str:
    _, error_payload = exception_to_anthropic_http(exc)
    return format_anthropic_sse("error", error_payload)
