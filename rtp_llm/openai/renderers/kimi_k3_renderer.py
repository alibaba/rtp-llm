import copy
import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, AsyncGenerator, Dict, List, Mapping, Optional

import torch
from typing_extensions import override

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.config.generate_config import GenerateConfig, ThinkingMode
from rtp_llm.config.response_format import ResponseFormat
from rtp_llm.config.response_format_compiler import ReasoningFormat
from rtp_llm.models.kimi_k3.kimi_k3_request_contract import (
    apply_kimi_k3_request_contract,
    kimi_k3_pending_prompt_token_ids,
    validate_kimi_k3_tool_history,
)
from rtp_llm.utils.base_model_datatypes import MMUrlType
from rtp_llm.openai.api_datatype import (
    ChatCompletionRequest,
    ChatCompletionResponseStreamChoice,
    ChatCompletionTokenLogprob,
    ChoiceLogprobs,
    DeltaMessage,
    FinisheReason,
    FunctionCall,
    RoleEnum,
    ToolCall,
    TopLogprob,
    UsageInfo,
    get_tool_choice_function_name,
)
from rtp_llm.openai.renderer_factory_register import register_renderer
from rtp_llm.openai.renderers.basic_renderer import PromptWithMMInput
from rtp_llm.openai.renderers.custom_renderer import (
    CustomChatRenderer,
    OutputDelta,
    RenderedInputs,
    StreamResponseObject,
    StreamStatus,
    ThinkStatus,
)
from rtp_llm.ops import MultimodalInput
from rtp_llm.server.backend_rpc_server_visitor import BackendRPCServerVisitor
from rtp_llm.utils.word_util import truncate_response_with_stop_words

_GRAMMAR_RESPONSE_FORMAT_TYPES = {
    "json_object",
    "json_schema",
    "regex",
    "ebnf",
    "structural_tag",
}

_K3_THINKING_EFFORTS = ("low", "high", "max")
_K3_MAX_CONSTRAINED_PARALLEL_TOOL_CALLS = 8


def _normalize_reasoning_effort(effort: str) -> str:
    normalized = effort.strip().lower()
    if normalized not in _K3_THINKING_EFFORTS:
        allowed = ", ".join(repr(value) for value in _K3_THINKING_EFFORTS)
        raise FtRuntimeException(
            ExceptionType.INVALID_PARAMS,
            f"'reasoning_effort' must be one of: {allowed}",
        )
    return normalized


def _resolve_thinking_enabled(
    request: ChatCompletionRequest,
    template_kwargs: Mapping[str, Any],
) -> bool:
    """Resolve K3 thinking controls in request-precedence order."""

    def template_bool(name: str) -> bool:
        value = template_kwargs[name]
        if not isinstance(value, bool):
            raise FtRuntimeException(
                ExceptionType.INVALID_PARAMS,
                f"Kimi K3 chat_template_kwargs.{name} must be a boolean",
            )
        return value

    if request.thinking is not None:
        return request.thinking.type == "enabled"
    if request.enable_thinking is not None:
        return request.enable_thinking

    if "thinking" in template_kwargs:
        return template_bool("thinking")
    if "enable_thinking" in template_kwargs:
        return template_bool("enable_thinking")

    if request.thinking_budget == 0:
        return False
    if (
        request.extra_configs is not None
        and request.extra_configs.max_thinking_tokens == 0
    ):
        return False

    if isinstance(request.reasoning_effort, str):
        return True

    response_format = request.response_format
    if response_format is not None and response_format.type != "text":
        return False
    if request.json_format:
        return False
    return True


def _validate_k3_dynamic_tools(request: ChatCompletionRequest) -> None:
    """Check K3's per-message declarations before the tokenizer sees them."""
    names = [tool.function.name for tool in request.tools or []]
    for index, message in enumerate(request.messages):
        if not message.tools:
            continue
        if message.role != RoleEnum.system or message.content not in (None, "", []):
            raise ValueError(
                f"messages[{index}].tools requires a system message with empty content"
            )
        for tool in message.tools:
            if tool.get("type") != "function" or not isinstance(
                tool.get("function"), dict
            ):
                raise ValueError("dynamic tool must be a function object")
            function = tool["function"]
            name = function.get("name")
            if not isinstance(name, str) or not re.fullmatch(
                r"[A-Za-z_][A-Za-z0-9_]{0,255}", name
            ):
                raise ValueError("dynamic tool.function.name is invalid")
            if not isinstance(function.get("parameters"), dict):
                raise ValueError("dynamic tool.function.parameters must be an object")
            if "strict" in function and not isinstance(function["strict"], bool):
                raise ValueError("dynamic tool.function.strict must be a boolean")
            names.append(name)
    if len(names) != len(set(names)):
        raise ValueError("tool names must be unique across global and dynamic tools")


@dataclass(frozen=True)
class K3RequestOptions:
    """Canonical, immutable view of K3 request-level rendering controls."""

    thinking: bool
    template_kwargs: Mapping[str, Any]

    @classmethod
    def from_request(cls, request: ChatCompletionRequest) -> "K3RequestOptions":
        _validate_k3_dynamic_tools(request)
        normalized_reasoning_effort = (
            _normalize_reasoning_effort(request.reasoning_effort)
            if request.reasoning_effort is not None
            else None
        )
        template_kwargs: Dict[str, Any] = {}
        if request.chat_template_kwargs:
            template_kwargs.update(request.chat_template_kwargs)
        if (
            request.extra_configs is not None
            and request.extra_configs.chat_template_kwargs is not None
        ):
            template_kwargs.update(request.extra_configs.chat_template_kwargs)

        thinking = _resolve_thinking_enabled(request, template_kwargs)
        effort: Optional[str] = None
        if thinking:
            if request.thinking is not None and request.thinking.effort is not None:
                effort = request.thinking.effort
            elif normalized_reasoning_effort is not None:
                effort = normalized_reasoning_effort

        template_kwargs["thinking"] = thinking
        template_kwargs.pop("enable_thinking", None)
        if effort is None:
            template_kwargs.pop("thinking_effort", None)
        else:
            template_kwargs["thinking_effort"] = effort
        return cls(
            thinking=thinking,
            template_kwargs=MappingProxyType(template_kwargs),
        )


K3ChannelLogprobs = Dict[str, List[ChatCompletionTokenLogprob]]


@dataclass(frozen=True)
class K3ParsedDelta:
    delta: DeltaMessage
    channel_logprobs: K3ChannelLogprobs


@dataclass
class _K3OutputDelta(OutputDelta):
    channel_logprobs: K3ChannelLogprobs = field(default_factory=dict)

    @classmethod
    def from_parsed(
        cls, source: OutputDelta, parsed: K3ParsedDelta
    ) -> "_K3OutputDelta":
        return cls(
            output_str=parsed.delta,
            logprobs=None,
            input_length=source.input_length,
            output_length=source.output_length,
            reuse_length=source.reuse_length,
            multimodal_lengths=source.multimodal_lengths,
            extra_outputs=source.extra_outputs,
            output_ids=source.output_ids,
            channel_logprobs=parsed.channel_logprobs,
        )


@dataclass
class _K3StreamResponseObject(StreamResponseObject):
    channel_logprobs: Dict[int, K3ChannelLogprobs] = field(default_factory=dict)


class K3XtmlDecoder:
    """Own all per-choice XTML state and decode one engine delta atomically."""

    def __init__(self, thinking: bool, request: ChatCompletionRequest):
        self.xtml_pending = ""
        self.in_reasoning = thinking
        self.emitted_reasoning = False
        self.response_closed = False
        self.tools_pending = ""
        self.in_tools = False
        self.tool_calls_seen = 0
        self.logprobs_channel: Optional[str] = (
            "reasoning_content" if thinking else "content"
        )
        self.logprobs_marker_pending: List[ChatCompletionTokenLogprob] = []
        self.next_tool_id_index = 0
        self.used_tool_ids: set[str] = set()
        self.used_tool_hex: set[str] = set()
        for message in request.messages:
            for call in message.tool_calls or []:
                name = call.function.name
                call_id = call.id
                if not isinstance(name, str) or not isinstance(call_id, str):
                    continue
                self.used_tool_ids.add(call_id)
                prefix = re.escape(name)
                modern = re.fullmatch(
                    rf"{prefix}_(?P<index>[0-9]+)_(?P<hex>[0-9a-fA-F]{{8}})",
                    call_id,
                )
                legacy = re.fullmatch(
                    rf"(?:functions\.)?{prefix}:(?P<index>[0-9]+)",
                    call_id,
                )
                if legacy is None:
                    legacy = re.fullmatch(
                        rf"{prefix}_(?P<index>[0-9]+)", call_id
                    )
                if modern:
                    self.used_tool_hex.add(modern.group("hex").lower())
                match = modern or legacy
                if match:
                    self.next_tool_id_index = max(
                        self.next_tool_id_index, int(match.group("index")) + 1
                    )

    def new_tool_call_id(self, name: str) -> str:
        while True:
            suffix = uuid.uuid4().hex[:8]
            call_id = f"{name}_{self.next_tool_id_index}_{suffix}"
            if suffix not in self.used_tool_hex and call_id not in self.used_tool_ids:
                break
        self.used_tool_hex.add(suffix)
        self.used_tool_ids.add(call_id)
        self.next_tool_id_index += 1
        return call_id

    def decode(
        self,
        text: str,
        logprobs: Optional[List[ChatCompletionTokenLogprob]],
        *,
        flush: bool = False,
    ) -> K3ParsedDelta:
        # Preserve token routing before text routing: both inputs describe the
        # same engine delta, but have independent partial-marker buffers.
        channel_logprobs = KimiK3Renderer._decode_logprobs_by_channel(
            self, logprobs, flush=flush
        )
        delta = KimiK3Renderer._decode_xtml_text(self, text, flush=flush)
        return K3ParsedDelta(delta=delta, channel_logprobs=channel_logprobs)


class _KimiK3StreamStatus(StreamStatus):
    """Per-choice state for parsing K3's generated XTML channels."""

    def __init__(self, request: ChatCompletionRequest, *, thinking: bool):
        super().__init__(request)
        self.xtml_decoder = K3XtmlDecoder(thinking, request)
        self.pending_token_probs: List[tuple[int, torch.Tensor]] = []


class KimiK3Renderer(CustomChatRenderer):
    """Render Kimi K3's Python-defined XTML and collect image inputs.

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
    _XTML_CONTROL_TOKENS = {"<|open|>", "<|close|>"}
    _XTML_CHANNELS = {
        "think": "reasoning_content",
        "response": "content",
        "tools": "tool_calls",
        "message": None,
    }

    @override
    def resolve_thinking_mode(self, request: ChatCompletionRequest) -> ThinkingMode:
        return (
            ThinkingMode.ENABLED
            if K3RequestOptions.from_request(request).thinking
            else ThinkingMode.DISABLED
        )

    @override
    def get_reasoning_format(self) -> ReasoningFormat:
        return ReasoningFormat(
            tag_begin="",
            tag_end=self._THINK_TO_RESPONSE,
            tag_end_native_encoding=True,
        )

    @staticmethod
    def _pending_prompt_token_count(tokenizer, thinking: bool) -> int:
        """Count the open generation channel excluded from tokenism usage."""

        return len(kimi_k3_pending_prompt_token_ids(tokenizer, thinking))

    @staticmethod
    def _subtract_pending_prompt_tokens(
        response: StreamResponseObject, pending_tokens: int
    ) -> None:
        def adjust(usage: Optional[UsageInfo]) -> None:
            if usage is None:
                return
            if (
                usage.prompt_tokens < pending_tokens
                or usage.total_tokens < pending_tokens
            ):
                raise RuntimeError(
                    "Kimi K3 usage is shorter than its pending generation prompt: "
                    f"prompt={usage.prompt_tokens}, total={usage.total_tokens}, "
                    f"pending={pending_tokens}"
                )
            usage.prompt_tokens -= pending_tokens
            usage.total_tokens -= pending_tokens
            details = usage.prompt_tokens_details
            if details is not None and details.cached_tokens is not None:
                details.cached_tokens = min(
                    details.cached_tokens, usage.prompt_tokens
                )

        adjust(response.usage)
        for choice in response.choices:
            adjust(choice.usage)

    @staticmethod
    def _has_logprobs(choice: ChatCompletionResponseStreamChoice) -> bool:
        logprobs = choice.logprobs
        return logprobs is not None and bool(logprobs.content or logprobs.refusal)

    @staticmethod
    def _stream_includes_usage(request: ChatCompletionRequest) -> bool:
        return bool(
            request.stream_options is not None
            and request.stream_options.include_usage
        )

    @override
    async def _generate_final(
        self,
        buffer_list: List[StreamStatus],
        request: ChatCompletionRequest,
        think_status_list: List[ThinkStatus],
    ) -> StreamResponseObject:
        # A non-streaming response may receive the entire tools channel in one
        # merged output. Its finish reason is still unset while XTML is parsed;
        # the base finalizer would otherwise default it to "stop".
        for status in buffer_list:
            if (
                isinstance(status, _KimiK3StreamStatus)
                and status.xtml_decoder.tool_calls_seen > 0
                and status.finish_reason in (None, FinisheReason.stop)
            ):
                status.finish_reason = FinisheReason.tool_calls
        response = await super()._generate_final(
            buffer_list, request, think_status_list
        )
        if response.usage is None:
            return response
        for choice, buffer, think_status in zip(
            response.choices, buffer_list, think_status_list
        ):
            assert buffer.output is not None
            choice_usage = copy.deepcopy(response.usage)
            output_tokens = buffer.output.aux_info.output_len
            choice_usage.completion_tokens = output_tokens
            choice_usage.total_tokens = choice_usage.prompt_tokens + output_tokens
            if choice_usage.completion_tokens_details is not None:
                choice_usage.completion_tokens_details.reasoning_tokens = (
                    think_status.think_tokens
                )
            choice.usage = choice_usage
        return response

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
        thinking = bool(generate_config.in_think_mode)
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
    def _split_exclusive_delta(delta: DeltaMessage) -> List[DeltaMessage]:
        """Split channel fields so each serialized delta carries only one."""

        if delta.role is not None:
            return [
                delta.model_copy(
                    update={"reasoning_content": None, "tool_calls": None}
                )
            ]

        channel_fields = ("reasoning_content", "content", "tool_calls")
        present = [
            name
            for name in channel_fields
            if getattr(delta, name) is not None
            and (name != "tool_calls" or bool(delta.tool_calls))
        ]
        if len(present) <= 1:
            return [delta]

        result = []
        for name in present:
            updates = {field: None for field in channel_fields}
            updates[name] = getattr(delta, name)
            split_delta = delta.model_copy(update=updates)
            result.append(split_delta)
            if name == "reasoning_content" and delta.reasoning_content:
                result.append(DeltaMessage(reasoning_content=""))
        return result

    @classmethod
    def _split_exclusive_response(
        cls, response: StreamResponseObject
    ) -> List[StreamResponseObject]:
        """Turn a batched response into ordered single-choice channel frames."""

        result = []
        response_channel_logprobs = (
            response.channel_logprobs
            if isinstance(response, _K3StreamResponseObject)
            else {}
        )
        for choice in response.choices:
            channel_logprobs = response_channel_logprobs.get(choice.index)
            emitted_logprob_channels = set()
            is_final = choice.finish_reason is not None
            deltas = (
                [DeltaMessage()]
                if is_final
                else cls._split_exclusive_delta(choice.delta)
            )
            for index, delta in enumerate(deltas):
                is_last_delta = index == len(deltas) - 1
                split_choice = choice.model_copy(
                    update={
                        "delta": delta,
                        "finish_reason": choice.finish_reason if is_last_delta else None,
                        "usage": choice.usage if is_last_delta else None,
                    }
                )
                split_response = StreamResponseObject(
                    choices=[split_choice],
                    usage=None,
                    aux_info=response.aux_info if is_last_delta else None,
                    extra_outputs=response.extra_outputs if is_last_delta else None,
                    prompt_logits=response.prompt_logits if is_last_delta else None,
                )

                if channel_logprobs is not None:
                    present_channel = next(
                        (
                            name
                            for name in ("reasoning_content", "content", "tool_calls")
                            if getattr(delta, name) is not None
                            and (name != "tool_calls" or bool(delta.tool_calls))
                        ),
                        None,
                    )
                    logprobs = (
                        channel_logprobs.get(present_channel)
                        if present_channel not in emitted_logprob_channels
                        else None
                    )
                    if present_channel is not None:
                        emitted_logprob_channels.add(present_channel)
                    split_choice.logprobs = (
                        ChoiceLogprobs(content=logprobs)
                        if logprobs
                        else None
                    )
                elif not is_last_delta:
                    split_choice.logprobs = None
                elif delta.role is not None or not cls._has_logprobs(split_choice):
                    split_choice.logprobs = None
                result.append(split_response)
        return result

    @classmethod
    def _transition_logprobs_channel(
        cls, control: str, tag: str
    ) -> Optional[str]:
        channel = cls._XTML_CHANNELS[tag]
        if control == "<|open|>":
            return channel
        if tag == "think":
            # A think budget can inject only the close marker; response text
            # may follow without an explicit response-open marker.
            return "content"
        return None

    @classmethod
    def _decode_logprobs_by_channel(
        cls,
        decoder: K3XtmlDecoder,
        logprobs: Optional[List[ChatCompletionTokenLogprob]],
        *,
        flush: bool = False,
    ) -> K3ChannelLogprobs:
        """Drop XTML envelopes and group visible token probabilities by channel."""

        result: Dict[str, List[ChatCompletionTokenLogprob]] = {
            "reasoning_content": [],
            "content": [],
            "tool_calls": [],
        }

        def route(item: ChatCompletionTokenLogprob) -> None:
            channel = decoder.logprobs_channel
            if channel in ("reasoning_content", "content"):
                result[channel].append(item)

        def consume(item: ChatCompletionTokenLogprob) -> None:
            pending = decoder.logprobs_marker_pending
            if not pending:
                if item.token in cls._XTML_CONTROL_TOKENS:
                    pending.append(item)
                elif item.token != "<|end_of_msg|>":
                    route(item)
                return

            pending.append(item)
            tokens = [entry.token for entry in pending]
            if len(tokens) == 2 and tokens[1] not in cls._XTML_CHANNELS:
                buffered = list(pending)
                pending.clear()
                route(buffered[0])
                consume(buffered[1])
                return
            if len(tokens) < 3:
                return
            if tokens[2] == "<|sep|>":
                decoder.logprobs_channel = cls._transition_logprobs_channel(
                    tokens[0], tokens[1]
                )
                pending.clear()
                return

            buffered = list(pending)
            pending.clear()
            route(buffered[0])
            for buffered_item in buffered[1:]:
                consume(buffered_item)

        for item in logprobs or []:
            consume(item)

        if flush and decoder.logprobs_marker_pending:
            buffered = list(decoder.logprobs_marker_pending)
            decoder.logprobs_marker_pending.clear()
            for item in buffered:
                route(item)
        return result

    @classmethod
    def _format_stream_frames(
        cls, response: StreamResponseObject, include_usage: bool
    ) -> List[StreamResponseObject]:
        final_usage = (
            copy.deepcopy(response.usage)
            if include_usage
            and any(choice.finish_reason is not None for choice in response.choices)
            else None
        )
        result = cls._split_exclusive_response(response)
        if final_usage is not None:
            result.append(StreamResponseObject(choices=[], usage=final_usage))
        return result

    @override
    async def render_response_stream(
        self,
        output_generator,
        request: ChatCompletionRequest,
        generate_config: GenerateConfig,
    ) -> AsyncGenerator[StreamResponseObject, None]:
        """Serialize mixed XTML channels as consecutive OpenAI stream frames."""

        async for response in super().render_response_stream(
            output_generator, request, generate_config
        ):
            if not generate_config.is_streaming:
                yield response
                continue
            for split_response in self._format_stream_frames(
                response, self._stream_includes_usage(request)
            ):
                yield split_response

    @staticmethod
    def _split_marker_prefix(text: str, marker: str) -> tuple[str, str]:
        """Keep the longest suffix that may be a split XTML marker."""

        max_prefix = min(len(text), len(marker) - 1)
        for length in range(max_prefix, 0, -1):
            if text.endswith(marker[:length]):
                return text[:-length], text[-length:]
        return text, ""

    @classmethod
    def _decode_xtml_text(
        cls, decoder: K3XtmlDecoder, text: str, flush: bool = False
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

        if decoder.response_closed:
            tool_calls = cls._parse_tools_delta(decoder, text, flush)
            return DeltaMessage(tool_calls=tool_calls)

        think_to_response = cls._THINK_TO_RESPONSE
        response_closure = cls._RESPONSE_CLOSE
        combined = decoder.xtml_pending + text
        decoder.xtml_pending = ""
        reasoning = ""
        content = ""
        tool_calls: Optional[List[ToolCall]] = None
        saw_transition = False

        if decoder.in_reasoning:
            transition_at = combined.find(think_to_response)
            if transition_at < 0:
                if flush:
                    reasoning = combined
                else:
                    reasoning, decoder.xtml_pending = cls._split_marker_prefix(
                        combined, think_to_response
                    )
                if reasoning:
                    decoder.emitted_reasoning = True
                return DeltaMessage(reasoning_content=reasoning or None)
            reasoning = combined[:transition_at]
            combined = combined[transition_at + len(think_to_response) :]
            decoder.in_reasoning = False
            saw_transition = True
            if reasoning:
                decoder.emitted_reasoning = True

        closure_at = combined.find(response_closure)
        if closure_at >= 0:
            content = combined[:closure_at]
            decoder.response_closed = True
            remainder = combined[closure_at + len(response_closure) :]
            tool_calls = cls._parse_tools_delta(decoder, remainder, flush)
        elif flush:
            content = combined
        else:
            content, decoder.xtml_pending = cls._split_marker_prefix(
                combined, response_closure
            )

        if content:
            content_value: Optional[str] = content
        else:
            content_value = None
        return DeltaMessage(
            reasoning_content=(
                reasoning
                if reasoning
                else "" if saw_transition and decoder.emitted_reasoning else None
            ),
            content=content_value,
            tool_calls=tool_calls,
        )

    @classmethod
    def _parse_tools_delta(
        cls, decoder: K3XtmlDecoder, text: str, flush: bool = False
    ) -> Optional[List[ToolCall]]:
        """Buffer the XTML tools channel and emit complete tool calls.

        Tool call blocks are small, so buffer until the channel closes (or
        the stream flushes) and parse the whole block at once.
        """

        decoder.tools_pending += text
        buf = decoder.tools_pending

        if not decoder.in_tools:
            open_at = buf.find(cls._TOOLS_OPEN)
            if open_at < 0:
                if flush:
                    # Stream ended without a tools channel; drop leftovers
                    # (e.g. stray channel tokens must not surface as content).
                    decoder.tools_pending = ""
                return None
            decoder.in_tools = True
            buf = buf[open_at + len(cls._TOOLS_OPEN) :]

        close_at = buf.find(cls._TOOLS_CLOSE)
        if close_at < 0 and not flush:
            decoder.tools_pending = buf
            return None
        block = buf[:close_at] if close_at >= 0 else buf
        decoder.tools_pending = (
            buf[close_at + len(cls._TOOLS_CLOSE) :] if close_at >= 0 else ""
        )

        calls = cls._parse_tools_block(decoder, block)
        if calls:
            decoder.tool_calls_seen += len(calls)
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
    def _parse_tools_block(cls, decoder: K3XtmlDecoder, block: str) -> List[ToolCall]:
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
                    id=decoder.new_tool_call_id(name),
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
        return [
            _KimiK3StreamStatus(request, thinking=self.in_think_mode(request))
            for _ in range(n)
        ]

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
        if isinstance(status, _KimiK3StreamStatus) and status.request.logprobs:
            token_ids = output.output_ids
            all_probs = output.all_probs
            if token_ids is None or all_probs is None:
                raise ValueError("K3 logprobs require token IDs and target probabilities")
            rows = all_probs.reshape(-1, all_probs.shape[-1])
            ids = token_ids.reshape(-1).tolist()
            if rows.shape[0] != len(ids):
                raise ValueError("K3 target probability rows must match output tokens")
            status.pending_token_probs.extend(zip(ids, rows.unbind(0)))
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
            flush = status.finish_reason is not None
            parsed = status.xtml_decoder.decode(
                delta.output_str,
                delta.logprobs,
                flush=flush,
            )
            # The engine only knows stop/length; report tool_calls when the
            # tools channel produced at least one call.
            if (
                status.finish_reason == FinisheReason.stop
                and status.xtml_decoder.tool_calls_seen > 0
            ):
                status.finish_reason = FinisheReason.tool_calls
            return _K3OutputDelta.from_parsed(delta, parsed)
        return delta

    async def _generate_log_probs(self, status, output):
        if not isinstance(status, _KimiK3StreamStatus):
            return await super()._generate_log_probs(status, output)
        if not status.request.logprobs:
            return None
        return self._drain_token_probabilities(status, status.last_token_length)

    def _drain_token_probabilities(self, status, count):
        if count > len(status.pending_token_probs):
            raise ValueError("K3 target probabilities do not cover emitted tokens")
        rows = status.pending_token_probs[:count]
        del status.pending_token_probs[:count]
        result = []
        for token_id, probs in rows:
            if token_id < 0 or token_id >= probs.numel():
                raise ValueError("K3 output token is outside target vocabulary")
            token = self.tokenizer.decode([token_id])
            selected_prob = probs[token_id].log().item()
            top_count = min(status.request.top_logprobs or 1, int(probs.count_nonzero()))
            top_probs, top_ids = probs.topk(top_count)
            top_logprobs = []
            for other_prob, other_id in zip(top_probs, top_ids):
                other_token = self.tokenizer.decode([int(other_id)])
                top_logprobs.append(
                    TopLogprob(
                        token=other_token,
                        logprob=other_prob.log().item(),
                        bytes=list(other_token.encode("utf-8", errors="replace")),
                    )
                )
            result.append(
                ChatCompletionTokenLogprob(
                    token=token,
                    logprob=selected_prob,
                    bytes=list(token.encode("utf-8", errors="replace")),
                    top_logprobs=top_logprobs,
                )
            )
        return result

    @override
    async def _flush_buffer(
        self,
        buffer_list: List[StreamStatus],
        stop_words_str: List[str],
        is_streaming: bool,
        think_status_list: List[ThinkStatus],
    ):
        if any(not isinstance(buffer, _KimiK3StreamStatus) for buffer in buffer_list):
            return await super()._flush_buffer(
                buffer_list, stop_words_str, is_streaming, think_status_list
            )
        items = []
        for buffer, think_status in zip(buffer_list, think_status_list):
            if buffer.output is None:
                raise ValueError("K3 flush requires a final model output")
            aux = buffer.output.aux_info
            pending_text = buffer.delta_output_string
            if not think_status.decision_made:
                pending_text = think_status.think_buffer + pending_text
                think_status.decision_made = True
                think_status.in_think_mode = False
                think_status.think_buffer = ""
                think_status.decision_token_ids = []
            pending_text = truncate_response_with_stop_words(
                pending_text, stop_words_str, is_streaming
            )
            pending_probs = (
                self._drain_token_probabilities(
                    buffer, len(buffer.pending_token_probs)
                )
                if buffer.request.logprobs
                else None
            )
            parsed = buffer.xtml_decoder.decode(pending_text, pending_probs, flush=True)
            items.append(
                _K3OutputDelta.from_parsed(
                    OutputDelta(
                        pending_text,
                        None,
                        aux.input_len,
                        aux.output_len,
                        aux.reuse_len,
                        multimodal_lengths=aux.multimodal_lengths,
                    ),
                    parsed,
                )
            )
        return await self._generate_stream_response(items, think_status_list)

    @override
    async def _generate_stream_response(
        self, items: List[OutputDelta], think_status_list: List[ThinkStatus]
    ) -> StreamResponseObject:
        response = await super()._generate_stream_response(items, think_status_list)
        channel_logprobs = {}
        for item, choice, think_status in zip(
            items, response.choices, think_status_list
        ):
            if not isinstance(item, _K3OutputDelta):
                continue
            item_logprobs = item.channel_logprobs
            if think_status.is_streaming:
                channel_logprobs[choice.index] = item_logprobs
                choice.logprobs = None
            else:
                content_logprobs = item_logprobs.get("content")
                choice.logprobs = (
                    ChoiceLogprobs(content=content_logprobs)
                    if content_logprobs
                    else None
                )
        return _K3StreamResponseObject(
            choices=response.choices,
            usage=response.usage,
            aux_info=response.aux_info,
            extra_outputs=response.extra_outputs,
            prompt_logits=response.prompt_logits,
            channel_logprobs=channel_logprobs,
        )

    @override
    def should_process_think(self, request: ChatCompletionRequest) -> bool:
        del request
        # K3XtmlDecoder separates the channels; the generic <think> tag parser
        # must not process the output again.
        return False

    @staticmethod
    def _request_dict(request: ChatCompletionRequest) -> Dict[str, Any]:
        return request.model_dump(exclude_none=True, mode="json")

    @staticmethod
    def _collect_and_rewrite(
        messages: List[Dict[str, Any]],
    ) -> tuple[List[Dict[str, Any]], PromptWithMMInput]:
        urls: List[str] = []
        types: List[MMUrlType] = []
        rewritten: List[Dict[str, Any]] = []
        for message in messages:
            role = message.get("role")
            content = message.get("content")
            if not isinstance(content, list):
                rewritten.append(message)
                continue

            new_parts: List[Dict[str, Any]] = []
            for part in content:
                if not isinstance(part, dict):
                    raise ValueError("Kimi K3 message content parts must be objects")
                part_type = part.get("type")
                if part_type == "text":
                    new_parts.append({"type": "text", "text": part.get("text")})
                elif part_type == "image_url":
                    if role != "user":
                        raise ValueError(
                            "Kimi K3 supports image_url content only in user "
                            f"messages; got role {role!r}"
                        )
                    image_url = part.get("image_url")
                    url = image_url.get("url") if isinstance(image_url, dict) else None
                    if not isinstance(url, str) or not url:
                        raise ValueError(
                            "Kimi K3 image_url content requires a non-empty URL"
                        )
                    urls.append(url)
                    types.append(MMUrlType.IMAGE)
                    new_parts.append({"type": "image", "image": url})
                else:
                    raise ValueError(
                        "Kimi K3 supports only text and image_url content parts; "
                        f"got {part_type!r}"
                    )

            new_message = dict(message)
            new_message["content"] = new_parts
            rewritten.append(new_message)

        return rewritten, PromptWithMMInput(prompt="", urls=urls, mm_types=types)

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

        def call_tags(index: int) -> List[Dict[str, Any]]:
            tags = []
            for tool in tools:
                function = tool["function"]
                name = cls._escape_xtml_attr(function["name"])
                tags.append(
                    {
                        "type": "tag",
                        "begin": (
                            f'<|open|>call tool="{name}" index="{index}"<|sep|>'
                            '<|open|>json type="object"<|sep|>'
                        ),
                        "content": {
                            "type": "json_schema",
                            "json_schema": function.get("parameters") or {},
                        },
                        "end": "<|close|>json<|sep|><|close|>call<|sep|>",
                    }
                )
            return tags

        if request.parallel_tool_calls is True:
            # XTML embeds a one-based call index in each opening tag. Build a
            # bounded sequence so the grammar enforces consecutive indices
            # while retaining each selected tool's argument schema.
            def call_choice(index: int) -> Dict[str, Any]:
                tags = call_tags(index)
                return tags[0] if len(tags) == 1 else {"type": "or", "elements": tags}

            content: Dict[str, Any] = call_choice(
                _K3_MAX_CONSTRAINED_PARALLEL_TOOL_CALLS
            )
            for index in range(_K3_MAX_CONSTRAINED_PARALLEL_TOOL_CALLS - 1, 0, -1):
                content = {
                    "type": "sequence",
                    "elements": [
                        call_choice(index),
                        {"type": "optional", "content": content},
                    ],
                }
        else:
            content = {
                "type": "tags_with_separator",
                "tags": call_tags(1),
                "separator": "",
                "at_least_one": True,
                "stop_after_first": True,
            }

        return {
            "type": "structural_tag",
            "format": {
                "type": "tag",
                "begin": cls._RESPONSE_CLOSE + cls._TOOLS_OPEN,
                "content": content,
                "end": cls._TOOLS_CLOSE,
            }
        }

    @staticmethod
    def _response_format_has_grammar(response_format: Any) -> bool:
        if response_format is None:
            return False
        if isinstance(response_format, ResponseFormat):
            return response_format.type in _GRAMMAR_RESPONSE_FORMAT_TYPES
        if isinstance(response_format, str):
            try:
                response_format = json.loads(response_format)
            except ValueError:
                return True
        return (
            not isinstance(response_format, dict)
            or response_format.get("type") in _GRAMMAR_RESPONSE_FORMAT_TYPES
        )

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
        if (response_format is None or response_format.type == "text") and not request.json_format:
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
        options: K3RequestOptions, request_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        kwargs = dict(options.template_kwargs)
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

    def _render_with_image_placeholders(
        self,
        options: K3RequestOptions,
        request_dict: Dict[str, Any],
        messages: List[Dict[str, Any]],
        mm_input: PromptWithMMInput,
    ) -> RenderedInputs:
        # The ViT owns image download, size-dependent prompt construction and
        # the corresponding text embeddings. Keep one stable token per image in
        # the frontend so URL-based ViT cache routing remains possible.
        image_prompts = ["<|media_pad|>"] * len(mm_input.urls)
        tools = self._tools(request_dict)
        template_kwargs = self._template_kwargs(options, request_dict)

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=True,
            add_generation_prompt=True,
            image_prompts=image_prompts,
            **template_kwargs,
        )
        token_ids = self._as_token_ids(input_ids)
        logging.debug("Kimi K3 rendered %d XTML prompt tokens", len(token_ids))
        # Leave rendered_prompt empty: both endpoints decode input_ids on demand
        # for debug output, avoiding a second expensive XTML template pass.
        return RenderedInputs(
            input_ids=token_ids,
            input_urls=mm_input.urls,
            input_urls_type=mm_input.mm_types,
        )

    @override
    def render_chat(self, request: ChatCompletionRequest) -> RenderedInputs:
        validate_kimi_k3_tool_history(request.messages)
        options = K3RequestOptions.from_request(request)
        request_dict = self._request_dict(request)
        messages, mm_input = self._collect_and_rewrite(request_dict["messages"])
        return self._render_with_image_placeholders(
            options, request_dict, messages, mm_input
        )

    @override
    async def render_chat_async(self, request: ChatCompletionRequest) -> RenderedInputs:
        validate_kimi_k3_tool_history(request.messages)
        options = K3RequestOptions.from_request(request)
        request_dict = self._request_dict(request)
        messages, mm_input = self._collect_and_rewrite(request_dict["messages"])
        return self._render_with_image_placeholders(
            options, request_dict, messages, mm_input
        )

    @override
    def apply_chat_completion_constraints(
        self, request: ChatCompletionRequest, generate_config: GenerateConfig
    ) -> None:
        options = K3RequestOptions.from_request(request)
        self._apply_chat_completion_constraints_with_options(
            request, generate_config, options
        )

    def _apply_chat_completion_constraints_with_options(
        self,
        request: ChatCompletionRequest,
        generate_config: GenerateConfig,
        options: K3RequestOptions,
    ) -> None:
        specified_fields = {
            name
            for name in request.model_fields_set
            if getattr(request, name, None) is not None
        }
        for name in (
            "temperature",
            "top_p",
            "presence_penalty",
            "frequency_penalty",
        ):
            if name in specified_fields:
                setattr(generate_config, name, getattr(request, name))
        if "n" in specified_fields:
            generate_config.num_return_sequences = request.n
        apply_kimi_k3_request_contract(
            generate_config,
            specified_fields=specified_fields,
            thinking=options.thinking,
        )

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

            generate_config.structural_tag = structural_tag



register_renderer("kimi_k3", KimiK3Renderer)
