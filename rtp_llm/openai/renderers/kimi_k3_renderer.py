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
from rtp_llm.config.generate_config import GenerateConfig
from rtp_llm.models.kimi_k3.kimi_k3_request_contract import (
    apply_kimi_k3_request_contract,
    kimi_k3_pending_prompt_token_ids,
    validate_kimi_k3_tool_history,
)
from rtp_llm.multimodal.multimodal_mixins.kimi_k3.kimi_k3_image_processor import (
    KimiK3VisionProcessor,
    load_kimi_k3_media_config,
    preflight_kimi_k3_images,
    preflight_kimi_k3_images_async,
)
from rtp_llm.multimodal.multimodal_util import MMUrlType
from rtp_llm.openai.api_datatype import (
    ChatCompletionRequest,
    ChatCompletionResponseStreamChoice,
    ChatCompletionTokenLogprob,
    ChoiceLogprobs,
    DeltaMessage,
    FinisheReason,
    FunctionCall,
    ToolCall,
    UsageInfo,
    get_tool_choice_function_name,
)
from rtp_llm.openai.renderer_factory_register import register_renderer
from rtp_llm.openai.renderers.basic_renderer import PromptWithMMInput
from rtp_llm.openai.renderers.custom_renderer import (
    CustomChatRenderer,
    OutputDelta,
    RenderedInputs,
    RendererRequestContext,
    StreamResponseObject,
    StreamStatus,
    ThinkStatus,
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

_K3_THINKING_EFFORTS = ("low", "high", "max")


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

    if request.thinking is not None:
        return request.thinking.type == "enabled"
    if request.enable_thinking is not None:
        return request.enable_thinking

    if "thinking" in template_kwargs:
        return bool(template_kwargs["thinking"])
    if "enable_thinking" in template_kwargs:
        return bool(template_kwargs["enable_thinking"])

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
    return True


@dataclass(frozen=True)
class K3RequestOptions(RendererRequestContext):
    """Canonical, immutable view of K3 request-level rendering controls."""

    thinking: bool
    thinking_effort: Optional[str]
    template_kwargs: Mapping[str, Any]

    @classmethod
    def from_request(cls, request: ChatCompletionRequest) -> "K3RequestOptions":
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
            thinking_effort=effort,
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
            channel_logprobs=parsed.channel_logprobs,
        )


@dataclass
class _K3StreamResponseObject(StreamResponseObject):
    channel_logprobs: Dict[int, K3ChannelLogprobs] = field(default_factory=dict)


class K3XtmlDecoder:
    """Own all per-choice XTML state and decode one engine delta atomically."""

    def __init__(self, thinking: bool):
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
        self.xtml_decoder = K3XtmlDecoder(thinking)


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
        self._image_processor = KimiK3VisionProcessor(
            load_kimi_k3_media_config(
                self.ckpt_path or self.model_config.checkpoint_path
            )
        )
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
    def _split_logprobs_by_channel(
        cls,
        status: _KimiK3StreamStatus,
        logprobs: Optional[List[ChatCompletionTokenLogprob]],
        *,
        flush: bool = False,
    ) -> K3ChannelLogprobs:
        return cls._decode_logprobs_by_channel(
            status.xtml_decoder, logprobs, flush=flush
        )

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
        if final_usage is not None and include_usage:
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
    def _parse_xtml_delta(
        cls, status: _KimiK3StreamStatus, text: str, flush: bool = False
    ) -> DeltaMessage:
        return cls._decode_xtml_text(status.xtml_decoder, text, flush=flush)

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

        calls = cls._parse_tools_block(block)
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
    async def _create_response_status_list(
        self,
        n: int,
        request: ChatCompletionRequest,
        enable_think_mode: bool,
    ) -> List[StreamStatus]:
        return [
            _KimiK3StreamStatus(request, thinking=enable_think_mode) for _ in range(n)
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
            channel_logprobs=channel_logprobs,
        )

    @override
    def _response_thinking_enabled(
        self,
        request: ChatCompletionRequest,
        generate_config: GenerateConfig,
    ) -> bool:
        del request
        # apply_chat_completion_constraints resolves the request once and stores
        # the canonical mode on GenerateConfig before generation starts.
        return bool(generate_config.in_think_mode)

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

    def _validate_visual_token_budget(
        self,
        token_ids: List[int],
        metadata: List[tuple[int, int]],
    ) -> None:
        if not self.max_seq_len or not metadata:
            return
        visual_tokens = sum(
            self._image_processor.resize_config_for_size(width, height)["num_tokens"]
            for width, height in metadata
        )
        expanded_input_length = len(token_ids) - len(metadata) + visual_tokens
        if expanded_input_length > self.max_seq_len:
            raise ValueError(
                "Kimi K3 expanded multimodal input exceeds max_seq_len: "
                f"{expanded_input_length} > {self.max_seq_len}"
            )

    def _render_preflighted(
        self,
        options: K3RequestOptions,
        request_dict: Dict[str, Any],
        messages: List[Dict[str, Any]],
        mm_input: PromptWithMMInput,
        tensors: List[torch.Tensor],
        metadata: List[tuple[int, int]],
    ) -> RenderedInputs:
        image_prompts = [
            KimiK3VisionProcessor.make_image_prompt(width, height)
            for width, height in metadata
        ]
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
        self._validate_visual_token_budget(token_ids, metadata)
        logging.debug("Kimi K3 rendered %d XTML prompt tokens", len(token_ids))
        # Leave rendered_prompt empty: both endpoints decode input_ids on demand
        # for debug output, avoiding a second expensive XTML template pass.
        return RenderedInputs(
            input_ids=token_ids,
            input_urls=mm_input.urls,
            input_urls_type=mm_input.mm_types,
            input_tensors=tensors,
            renderer_context=options,
        )

    @override
    def render_chat(self, request: ChatCompletionRequest) -> RenderedInputs:
        validate_kimi_k3_tool_history(request.messages)
        options = K3RequestOptions.from_request(request)
        request_dict = self._request_dict(request)
        messages, mm_input = self._collect_and_rewrite(request_dict["messages"])
        tensors, metadata = preflight_kimi_k3_images(mm_input.urls, self.vit_config)
        return self._render_preflighted(
            options,
            request_dict,
            messages,
            mm_input,
            tensors,
            metadata,
        )

    @override
    async def render_chat_async(self, request: ChatCompletionRequest) -> RenderedInputs:
        validate_kimi_k3_tool_history(request.messages)
        options = K3RequestOptions.from_request(request)
        request_dict = self._request_dict(request)
        messages, mm_input = self._collect_and_rewrite(request_dict["messages"])
        tensors, metadata = await preflight_kimi_k3_images_async(
            mm_input.urls, self.vit_config
        )
        return self._render_preflighted(
            options,
            request_dict,
            messages,
            mm_input,
            tensors,
            metadata,
        )

    @override
    def apply_chat_completion_constraints(
        self, request: ChatCompletionRequest, generate_config: GenerateConfig
    ) -> None:
        options = K3RequestOptions.from_request(request)
        self._apply_chat_completion_constraints_with_options(
            request, generate_config, options
        )

    @override
    def apply_rendered_chat_completion_constraints(
        self,
        request: ChatCompletionRequest,
        generate_config: GenerateConfig,
        rendered_inputs: RenderedInputs,
    ) -> None:
        options = rendered_inputs.renderer_context
        if not isinstance(options, K3RequestOptions):
            raise RuntimeError("Kimi K3 rendered inputs are missing request options")
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
