import functools
import json
import logging
import os
from abc import ABC
from typing import List, Optional, Sequence, Tuple

from jinja2 import BaseLoader, Environment
from typing_extensions import override

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.frontend.tokenizer_factory.tokenizers import BaseTokenizer
from rtp_llm.openai.api_datatype import (
    ChatCompletionRequest,
    DeltaMessage,
    FinisheReason,
    RoleEnum,
    ToolCall,
)
from rtp_llm.openai.renderers.custom_renderer import (
    CustomChatRenderer,
    OutputDelta,
    RenderedInputs,
    RendererParams,
    StreamStatus,
    ThinkStatus,
)
from rtp_llm.openai.renderers.sglang_helpers.format_convert_helper import (
    rtp_tools_to_sglang_tools,
    streaming_parse_result_to_tool_calls,
)
from rtp_llm.openai.renderers.sglang_helpers.entrypoints.openai.protocol import Tool
from rtp_llm.openai.renderers.sglang_helpers.function_call.base_format_detector import (
    BaseFormatDetector,
    ToolParseError,
)
from rtp_llm.openai.renderers.sglang_helpers.reasoning_parser import ReasoningParser
from rtp_llm.openai.renderers.sglang_helpers.token_normalizer import (
    TokenNormalizer,
    expand_prev_window,
)
from rtp_llm.utils.base_model_datatypes import GenerateOutput


class ReasoningToolStreamStatus(StreamStatus):
    generating_tool_call: bool = False
    detector: Optional[BaseFormatDetector] = None
    reasoning_parser: Optional[ReasoningParser] = None
    sglang_tools: Tuple[Tool, ...] = ()
    tool_call_keys: set[Tuple[str, object]]

    def __init__(
        self,
        request: ChatCompletionRequest,
        detector: Optional[BaseFormatDetector],
        reasoning_parser: Optional[ReasoningParser],
        sglang_tools: Optional[Tuple[Tool, ...]] = None,
    ):
        super().__init__(request)
        self.generating_tool_call = False
        self.detector = detector
        if self.detector is not None:
            self.detector.strict_tool_validation = (
                request.requires_tool_call() or not request.parallel_tool_calls
            )
        self.reasoning_parser = reasoning_parser
        self.tool_call_keys = set()
        effective_tools = request.effective_tools()
        detector_tools = (
            list(request.tools or [])
            if request.selected_tool_name() is not None
            else effective_tools
        )
        self.sglang_tools = (
            tuple(rtp_tools_to_sglang_tools(detector_tools))
            if sglang_tools is None
            else sglang_tools
        )


class ReasoningToolBaseRenderer(CustomChatRenderer, ABC):
    """
    工具调用渲染器基类
    提供工具调用的通用逻辑，子类需要实现具体的检测器创建逻辑
    """

    def __init__(
        self,
        tokenizer: BaseTokenizer,
        renderer_params: RendererParams,
        generate_env_config,
        render_config=None,
        ckpt_path=None,
        misc_config=None,
        vit_config=None,
    ):
        super().__init__(
            tokenizer,
            renderer_params,
            generate_env_config,
            render_config,
            ckpt_path,
            misc_config,
            vit_config,
        )
        self._setup_stop_words()
        self._setup_chat_template()

    def _setup_stop_words(self):
        """设置额外的停止词，子类可以重写"""

    def _create_detector(
        self, request: ChatCompletionRequest
    ) -> Optional[BaseFormatDetector]:
        """创建Tools解析器，子类可选实现"""
        return None

    def _create_reasoning_parser(
        self, request: ChatCompletionRequest
    ) -> Optional[ReasoningParser]:
        """创建Resoning解析器，子类可选实现"""
        return None

    def _create_reasoning_parser_from_rendered_prompt(
        self, request: ChatCompletionRequest, rendered_prompt: str
    ) -> Optional[ReasoningParser]:
        return self._create_reasoning_parser(request)

    @override
    def should_process_think(self, request: ChatCompletionRequest):
        # 避免在父类中也处理Think
        return False

    @override
    async def _create_status_list(
        self,
        n: int,
        request: ChatCompletionRequest,
        rendered_prompt: Optional[str] = None,
    ) -> List[StreamStatus]:
        """创建状态列表"""
        effective_tools = request.effective_tools()
        if effective_tools or self.in_think_mode(request):
            detector_tools = (
                list(request.tools or [])
                if request.selected_tool_name() is not None
                else effective_tools
            )
            sglang_tools = tuple(rtp_tools_to_sglang_tools(detector_tools))
            return [
                ReasoningToolStreamStatus(
                    request,
                    self._create_detector(request) if effective_tools else None,
                    (
                        self._create_reasoning_parser(request)
                        if rendered_prompt is None
                        else self._create_reasoning_parser_from_rendered_prompt(
                            request, rendered_prompt
                        )
                    ),
                    sglang_tools,
                )
                for _ in range(n)
            ]
        else:
            return [StreamStatus(request) for _ in range(n)]

    def _finalize_tool_detector(self, status: StreamStatus) -> None:
        if not isinstance(status, ReasoningToolStreamStatus) or not status.detector:
            return

        try:
            parse_result = status.detector.finalize_streaming(
                truncated=status.finish_reason == FinisheReason.length
            )
        except ToolParseError as error:
            logging.error("工具调用解析失败: %s", error)
            raise FtRuntimeException(
                ExceptionType.EXECUTION_EXCEPTION, str(error)
            ) from error

        if parse_result.calls:
            raise FtRuntimeException(
                ExceptionType.EXECUTION_EXCEPTION,
                "tool parser committed calls during finalization",
            )
        status.delta_output_string = (
            parse_result.normal_text + status.delta_output_string
        )
        if status.request.requires_tool_call() and not status.tool_call_keys:
            raise FtRuntimeException(
                ExceptionType.EXECUTION_EXCEPTION,
                "model did not produce a required tool call",
            )

    @override
    async def _flush_buffer(
        self,
        buffer_list: List[StreamStatus],
        stop_words_str: List[str],
        is_streaming: bool,
        think_status_list: List[ThinkStatus],
    ):
        for status in buffer_list:
            self._finalize_tool_detector(status)
        return await super()._flush_buffer(
            buffer_list, stop_words_str, is_streaming, think_status_list
        )

    @override
    def render_chat(self, request: ChatCompletionRequest) -> RenderedInputs:
        """渲染聊天请求"""
        prompt: str = self._build_prompt(request)
        input_ids: List[int] = self.tokenizer.encode(prompt)
        self._update_request_from_rendered_prompt(request, prompt)
        return RenderedInputs(input_ids=input_ids, rendered_prompt=prompt)

    @functools.cached_property
    def _compiled_chat_template(self):
        env = Environment(
            loader=BaseLoader(),
            trim_blocks=True,
            lstrip_blocks=True,
            extensions=["jinja2.ext.do", "jinja2.ext.loopcontrols"],
        )
        self._customize_jinja_env(env)
        return env.from_string(self.chat_template)

    def _build_prompt(self, request: ChatCompletionRequest) -> str:
        """
        构建提示文本
        Args:
            request: 聊天完成请求
        Returns:
            str: 格式化后的提示文本
        """
        context = request.model_dump(exclude_none=True, mode="json")

        # 默认添加生成提示
        context["add_generation_prompt"] = True

        messages = self._preprocess_messages(context["messages"])
        context.update({"messages": messages})

        # 合并chat_template_kwargs
        if request.chat_template_kwargs is not None:
            context.update(request.chat_template_kwargs)

        if (
            request.extra_configs is not None
            and request.extra_configs.chat_template_kwargs is not None
            and isinstance(request.extra_configs.chat_template_kwargs, dict)
        ):
            context.update(request.extra_configs.chat_template_kwargs)

        # Request-level tool policy must win over arbitrary template kwargs.
        context["tools"] = [
            tool.model_dump(exclude_none=True, mode="json")
            for tool in request.effective_tools()
        ]
        context["messages"] = self._apply_tool_policy_to_messages(
            request, messages
        )
        try:
            rendered_prompt = self._compiled_chat_template.render(**context)
            return rendered_prompt
        except Exception as e:
            logging.error(f"构建提示文本失败: {str(e)}")
            raise ValueError(f"Error rendering prompt template: {str(e)}")

    def _apply_tool_policy_to_messages(
        self, request: ChatCompletionRequest, messages: List[dict]
    ) -> List[dict]:
        instructions: List[str] = []
        selected_name = request.selected_tool_name()
        if selected_name is not None:
            instructions.append(
                f"You must call the `{selected_name}` tool. "
                "Do not answer without that tool call."
            )
        elif request.tool_choice == "required":
            instructions.append(
                "You must call one or more of the provided tools. "
                "Do not answer without a tool call."
            )
        if request.effective_tools() and not request.parallel_tool_calls:
            instructions.append("Call at most one tool in this response.")
        if not instructions:
            return messages
        return [
            {"role": RoleEnum.system.value, "content": " ".join(instructions)},
            *messages,
        ]

    def _preprocess_messages(self, messages: List[dict]) -> List[dict]:
        """
        预处理消息，子类可以重写
        """
        return messages

    def _customize_jinja_env(self, env: Environment) -> None:
        """
        自定义Jinja2环境，子类可以重写此方法来添加自定义过滤器、函数等

        Args:
            env: Jinja2环境对象
            request: 聊天完成请求
            context: 模板渲染上下文
        """

        # 设置默认的tojson过滤器
        # 透传 json.dumps 支持的关键字参数（如 separators / indent / ensure_ascii），
        # Kimi-K2 等模板会调用 `tojson(separators=(',', ':'))`。
        def _tojson(value, **kwargs):
            if isinstance(value, str):
                return value
            kwargs.setdefault("sort_keys", False)
            kwargs.setdefault("ensure_ascii", False)
            return json.dumps(value, **kwargs)

        env.filters["tojson"] = _tojson

    async def _process_single_token_delta(
        self,
        status: StreamStatus,
        delta_text: str,
        output: GenerateOutput,
        stop_words_str: List[str],
        stop_word_slice_list: List[str],
        is_streaming: bool,
    ) -> Optional[OutputDelta]:
        """
        Process a single token's decoded text delta through stop words and detector.

        Returns OutputDelta if content is ready, None if buffering.
        """
        delta_text = status.delta_output_string + delta_text
        status.delta_output_string = delta_text

        status.delta_output_string, should_buffer = self._process_stop_words(
            status.delta_output_string,
            stop_words_str,
            stop_word_slice_list,
            is_streaming,
            status,
        )

        if should_buffer:
            return None

        if not status.delta_output_string and status.finish_reason:
            return None

        if isinstance(status, ReasoningToolStreamStatus) and (
            status.detector or status.reasoning_parser
        ):
            tool_delta = await self._process_reasoning_and_tool_calls(
                status, output, is_streaming
            )
            if tool_delta is not None:
                # _process_reasoning_and_tool_calls may include remaining normal text
                # in DeltaMessage.content (streaming mode). Clear delta_output_string
                # to avoid re-prepending already-emitted content in subsequent tokens.
                # In non-streaming mode we intentionally keep delta_output_string so
                # that the final flush can emit the remaining normal content.
                if is_streaming:
                    status.delta_output_string = ""
                return tool_delta

        if status.delta_output_string:
            delta = OutputDelta(
                output_str=status.delta_output_string,
                logprobs=await self._generate_log_probs(status, output),
                input_length=output.aux_info.input_len,
                output_length=status.reported_output_length(
                    output.aux_info.output_len
                ),
                reuse_length=output.aux_info.reuse_len,
            )
            status.delta_output_string = ""
            return delta

        return None

    def _merge_deltas(self, deltas: List[OutputDelta]) -> Optional[OutputDelta]:
        """
        Merge multiple OutputDeltas into a single delta.

        Combines tool calls, normal text, and reasoning content from multiple token deltas.
        """
        if not deltas:
            return None

        merged = deltas[0]
        if len(deltas) == 1:
            return merged

        for delta in deltas[1:]:
            self._merge_output_str(merged, delta)
            if delta.logprobs is not None:
                merged.logprobs = delta.logprobs

        return merged

    def _merge_output_str(self, merged: OutputDelta, delta: OutputDelta) -> None:
        """Merge output_str from delta into merged (mutates merged in place)."""
        merged_str = merged.output_str
        delta_str = delta.output_str

        if isinstance(merged_str, DeltaMessage) and isinstance(delta_str, DeltaMessage):
            self._merge_delta_messages(merged_str, delta_str)
        elif isinstance(merged_str, str) and isinstance(delta_str, str):
            # Type checker: both are strings here
            merged.output_str = merged_str + delta_str
        elif isinstance(merged_str, str) and isinstance(delta_str, DeltaMessage):
            merged.output_str = DeltaMessage(
                content=merged_str + (delta_str.content or ""),
                tool_calls=delta_str.tool_calls,
                reasoning_content=delta_str.reasoning_content,
            )
        elif isinstance(merged_str, DeltaMessage) and isinstance(delta_str, str):
            merged_str.content = (merged_str.content or "") + delta_str

    def _merge_delta_messages(self, merged: DeltaMessage, delta: DeltaMessage) -> None:
        """Merge DeltaMessage fields (mutates merged in place)."""
        if delta.content:
            merged.content = (merged.content or "") + delta.content

        if delta.tool_calls:
            if merged.tool_calls:
                for new_tool in delta.tool_calls:
                    existing_tool = next(
                        (t for t in merged.tool_calls if t.index == new_tool.index),
                        None,
                    )
                    if existing_tool:
                        self._merge_tool_calls(existing_tool, new_tool)
                    else:
                        merged.tool_calls.append(new_tool)
            else:
                merged.tool_calls = delta.tool_calls

        if delta.reasoning_content:
            merged.reasoning_content = (
                merged.reasoning_content or ""
            ) + delta.reasoning_content

    def _merge_tool_calls(self, existing: ToolCall, new: ToolCall) -> None:
        """Merge new tool call into existing (mutates existing in place)."""
        if new.id and not existing.id:
            existing.id = new.id
        if new.type and not existing.type:
            existing.type = new.type

        if new.function:
            if not existing.function:
                existing.function = new.function
            else:
                if new.function.name and not existing.function.name:
                    existing.function.name = new.function.name
                if new.function.arguments:
                    existing.function.arguments = (
                        existing.function.arguments or ""
                    ) + new.function.arguments

    @override
    async def _update_single_status(
        self,
        status: StreamStatus,
        output: GenerateOutput,
        max_new_tokens: int,
        stop_words_str: List[str],
        stop_word_slice_list: List[str],
        is_streaming: bool,
    ) -> OutputDelta:
        if status.finish_reason != None:
            return await self._create_empty_delta(status.output.aux_info)
        output_token_limit = self._effective_output_token_limit(
            output.aux_info.input_len, max_new_tokens
        )
        status.update_output(
            output,
            functools.partial(self._check_finish_reason, max_new_tokens=max_new_tokens),
            functools.partial(
                self._remove_stop_word_ids, output_token_limit=output_token_limit
            ),
            output_token_limit,
            self._find_token_stop_end,
        )

        # NOTE: With multi-token stop words (e.g., tokenized from extra_stop_words),
        # `_remove_stop_word_ids()` may truncate `status.output_ids` to an earlier position
        # once a stop-word sequence completes. If we have already advanced `last_output_ids`
        # past a buffered stop-word prefix, `output_ids` can become shorter than
        # `last_output_ids`. In this case we must:
        # 1) realign `last_output_ids` to the truncated output,
        # 2) drop any buffered stop-word prefix from `delta_output_string`,
        # otherwise `_flush_buffer()` may leak partial stop words.
        if status.output_rewound:
            status.finish_reason = FinisheReason.stop
            status.advance_output_cursor(0)
            if stop_word_slice_list and status.delta_output_string:
                longest_suffix = ""
                for slice_candidate in stop_word_slice_list:
                    if not slice_candidate:
                        continue
                    if status.delta_output_string.endswith(slice_candidate) and len(
                        slice_candidate
                    ) > len(longest_suffix):
                        longest_suffix = slice_candidate
                if longest_suffix:
                    status.delta_output_string = status.delta_output_string[
                        : -len(longest_suffix)
                    ]
            flushed = await self._process_single_token_delta(
                status,
                "",
                output,
                stop_words_str,
                stop_word_slice_list,
                is_streaming,
            )
            if flushed is not None:
                return flushed
            status.delta_output_string = ""
            return await self._create_empty_delta(output.aux_info)

        # Extract new token IDs from this iteration
        new_token_ids = status.output_ids[status.processed_token_count :]
        normalizer = TokenNormalizer(self.tokenizer)

        collected_deltas, normalizer_yielded = await self._process_normalized_tokens(
            normalizer,
            status,
            new_token_ids,
            output,
            stop_words_str,
            stop_word_slice_list,
            is_streaming,
        )

        # Update last_output_ids based on what the NORMALIZER yielded, not what we emitted.
        # If normalizer yielded content but detector buffered it (collected_deltas empty),
        # we still consumed the tokens and should update.
        # If normalizer didn't yield anything (buffered for \uFFFD resolution),
        # don't update so next iteration has full context for sliding window.
        if normalizer_yielded and new_token_ids:
            context_length = expand_prev_window(
                self.tokenizer, status.output_ids, len(new_token_ids)
            )
            status.advance_output_cursor(context_length)

        if collected_deltas:
            merged_delta = self._merge_deltas(collected_deltas)
            return merged_delta or await self._create_empty_delta(output.aux_info)

        return await self._create_empty_delta(output.aux_info)

    async def _process_normalized_tokens(
        self,
        normalizer: TokenNormalizer,
        status: StreamStatus,
        new_token_ids: List[int],
        output: GenerateOutput,
        stop_words_str: List[str],
        stop_word_slice_list: List[str],
        is_streaming: bool,
    ) -> Tuple[List[OutputDelta], bool]:
        """Normalize tokens and feed them through the detector pipeline.

        Returns:
            (collected_deltas, normalizer_yielded): the output deltas and whether
            the normalizer emitted any text (used to decide state advancement).
        """
        normalizer_yielded = False

        if is_streaming:
            collected_deltas = []
            policy_state = None
            if isinstance(status, ReasoningToolStreamStatus):
                policy_state = (
                    set(status.tool_call_keys),
                    status.generating_tool_call,
                    status.finish_reason,
                )
            try:
                for delta_text in normalizer.normalize_tokens(
                    status.prev_token_id, new_token_ids
                ):
                    normalizer_yielded = True
                    finish_reason_before_token = status.finish_reason
                    token_delta = await self._process_single_token_delta(
                        status,
                        delta_text,
                        output,
                        stop_words_str,
                        stop_word_slice_list,
                        is_streaming=True,
                    )
                    if token_delta is not None:
                        collected_deltas.append(token_delta)
                    if (
                        status.finish_reason == FinisheReason.stop
                        and finish_reason_before_token != FinisheReason.stop
                    ):
                        break
                self._validate_parallel_tool_boundary(status)
            except Exception:
                if policy_state is not None:
                    (
                        status.tool_call_keys,
                        status.generating_tool_call,
                        status.finish_reason,
                    ) = policy_state
                raise
            return collected_deltas, normalizer_yielded

        # Non-streaming: accumulate all text first, then process once
        all_text = "".join(
            normalizer.normalize_tokens(status.prev_token_id, new_token_ids)
        )
        if not all_text:
            return [], False

        complete_delta = await self._process_single_token_delta(
            status,
            all_text,
            output,
            stop_words_str,
            stop_word_slice_list,
            is_streaming=False,
        )
        return ([complete_delta] if complete_delta is not None else []), True

    def _validate_parallel_tool_boundary(self, status: StreamStatus) -> None:
        if (
            isinstance(status, ReasoningToolStreamStatus)
            and not status.request.parallel_tool_calls
            and status.finish_reason != FinisheReason.length
            and status.tool_call_keys
            and status.detector is not None
            and status.detector.has_pending_tool_call()
        ):
            raise FtRuntimeException(
                ExceptionType.EXECUTION_EXCEPTION,
                "model produced parallel tool calls while parallel tool calls are disabled",
            )

    async def _process_reasoning_and_tool_calls(
        self,
        status: ReasoningToolStreamStatus,
        output: GenerateOutput,
        is_streaming: bool,
    ) -> Optional[OutputDelta]:
        """
        Process reasoning text and tool calls from delta_output_string.

        Extracts reasoning content and tool calls, updates status.delta_output_string
        with remaining text, and returns OutputDelta if anything was parsed.

        Returns None if no reasoning/tool content found (caller uses default logic).
        """
        reasoning_text, remaining_after_reasoning = self._extract_reasoning_content(
            status.reasoning_parser, status.delta_output_string, is_streaming
        )

        tool_calls, remaining_after_tools = await self._extract_tool_calls_content(
            status.detector,
            status.sglang_tools,
            remaining_after_reasoning,
            is_streaming,
            truncated=status.finish_reason == FinisheReason.length,
        )

        status.delta_output_string = remaining_after_tools

        has_reasoning = bool(reasoning_text)
        has_tool_calls = tool_calls and len(tool_calls) > 0

        if not has_reasoning and not has_tool_calls:
            return None

        if has_tool_calls:
            self._record_tool_calls(status, tool_calls)
            status.generating_tool_call = True
            if (
                not status.request.parallel_tool_calls
                and status.detector is not None
                and status.detector.atomic_tool_calls
                and status.finish_reason != FinisheReason.length
            ):
                status.finish_reason = FinisheReason.tool_calls

        remaining_content = (
            remaining_after_tools if is_streaming and remaining_after_tools else None
        )

        return OutputDelta(
            output_str=DeltaMessage(
                content=remaining_content,
                tool_calls=tool_calls if has_tool_calls else None,
                reasoning_content=reasoning_text if has_reasoning else None,
            ),
            logprobs=await self._generate_log_probs(status, output),
            input_length=output.aux_info.input_len,
            output_length=status.reported_output_length(output.aux_info.output_len),
            reuse_length=output.aux_info.reuse_len,
        )

    def _record_tool_calls(
        self, status: ReasoningToolStreamStatus, tool_calls: List[ToolCall]
    ) -> None:
        selected_name = status.request.selected_tool_name()
        next_keys = set(status.tool_call_keys)
        for tool_call in tool_calls:
            call_name = tool_call.function.name
            if selected_name is not None and call_name not in (None, selected_name):
                raise FtRuntimeException(
                    ExceptionType.EXECUTION_EXCEPTION,
                    "model produced a tool call outside tool_choice",
                )
            if tool_call.index is not None:
                call_key: Tuple[str, object] = ("index", tool_call.index)
            elif tool_call.id is not None:
                call_key = ("id", tool_call.id)
            else:
                call_key = ("name", call_name)
            next_keys.add(call_key)

        if not status.request.parallel_tool_calls:
            has_pending_call = (
                status.finish_reason != FinisheReason.length
                and status.detector is not None
                and status.detector.has_pending_tool_call()
            )
            if len(next_keys) > 1 or (next_keys and has_pending_call):
                raise FtRuntimeException(
                    ExceptionType.EXECUTION_EXCEPTION,
                    "model produced parallel tool calls while parallel tool calls are disabled",
                )
        status.tool_call_keys = next_keys

    def _extract_reasoning_content(
        self,
        reasoning_parser: Optional[ReasoningParser],
        text: str,
        is_streaming: bool,
    ) -> Tuple[str, str]:
        """
        Extract reasoning content from text.

        Returns (reasoning_text, remaining_text).
        """
        if not reasoning_parser:
            return "", text

        try:
            if is_streaming:
                return reasoning_parser.parse_stream_chunk(text)
            else:
                return reasoning_parser.parse_non_stream(text)
        except Exception as e:
            logging.error(f"推理文本解析失败: {e}")
            return "", text

    async def _extract_tool_calls_content(
        self,
        detector: Optional[BaseFormatDetector],
        tools: Optional[Sequence[Tool]],
        text: str,
        is_streaming: bool,
        truncated: bool = False,
    ) -> tuple[Optional[List[ToolCall]], str]:
        """
        Extract tool calls from text.

        Returns (tool_calls, remaining_text).
        """
        if not detector or not tools:
            return None, text

        try:
            if is_streaming:
                parse_result = detector.parse_streaming_increment(text, tools)
            else:
                cleaned_text = self._clean_stop_words(text)
                parse_result = (
                    detector.detect_and_parse_truncated(cleaned_text, tools)
                    if truncated
                    else detector.detect_and_parse(cleaned_text, tools)
                )

            tool_calls, remaining_text = streaming_parse_result_to_tool_calls(
                parse_result
            )

            if not is_streaming:
                for i, tool_call in enumerate(tool_calls):
                    tool_call.index = i

            return tool_calls, remaining_text
        except ToolParseError as error:
            logging.error("工具调用解析失败: %s", error)
            raise FtRuntimeException(
                ExceptionType.EXECUTION_EXCEPTION, str(error)
            ) from error
        except Exception as e:
            logging.error("工具调用解析失败: %s", e)
            return None, text

    def _clean_stop_words(self, text: str) -> str:
        """Clean stop words from text (default: no cleaning)."""
        return text
