import functools
import json
import logging
import os
from abc import ABC
from typing import List, Optional, Tuple

from jinja2 import BaseLoader, Environment
from typing_extensions import override

from rtp_llm.frontend.tokenizer_factory.tokenizers import BaseTokenizer
from rtp_llm.openai.api_datatype import (
    ChatCompletionRequest,
    DeltaMessage,
    FinisheReason,
    GPTToolDefinition,
    RoleEnum,
    ToolCall,
)
from rtp_llm.openai.renderers.custom_renderer import (
    CustomChatRenderer,
    OutputDelta,
    RenderedInputs,
    RendererParams,
    StreamStatus,
)
from rtp_llm.openai.renderers.sglang_helpers.format_convert_helper import (
    rtp_tools_to_sglang_tools,
    streaming_parse_result_to_tool_calls,
)
from rtp_llm.openai.renderers.sglang_helpers.function_call.base_format_detector import (
    BaseFormatDetector,
)
from rtp_llm.openai.renderers.sglang_helpers.reasoning_parser import ReasoningParser
from rtp_llm.openai.renderers.sglang_helpers.token_normalizer import TokenNormalizer
from rtp_llm.utils.base_model_datatypes import GenerateOutput


class ReasoningToolStreamStatus(StreamStatus):
    generating_tool_call: bool = False
    detector: Optional[BaseFormatDetector] = None
    reasoning_parser: Optional[ReasoningParser] = None

    def __init__(
        self,
        request: ChatCompletionRequest,
        detector: Optional[BaseFormatDetector],
        reasoning_parser: Optional[ReasoningParser],
    ):
        super().__init__(request)
        self.generating_tool_call = False
        self.detector = detector
        self.reasoning_parser = reasoning_parser


class ReasoningToolBaseRenderer(CustomChatRenderer, ABC):
    """
    工具调用渲染器基类
    提供工具调用的通用逻辑，子类需要实现具体的检测器创建逻辑
    """

    def __init__(
        self,
        tokenizer: BaseTokenizer,
        renderer_params: RendererParams,
    ):
        super().__init__(tokenizer, renderer_params)
        self._setup_stop_words()
        self._setup_chat_template()
        # 避免短期内多次encode prompt的开销
        self._cached_encode = functools.lru_cache()(self.tokenizer.encode)

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

    @override
    def should_process_think(self, request: ChatCompletionRequest):
        # 避免在父类中也处理Think
        return False

    @override
    async def _create_status_list(
        self, n: int, request: ChatCompletionRequest
    ) -> List[StreamStatus]:
        """创建状态列表"""
        if (request.tools or self.in_think_mode(request)) and not request.logprobs:
            return [
                ReasoningToolStreamStatus(
                    request,
                    self._create_detector(request),
                    self._create_reasoning_parser(request),
                )
                for _ in range(n)
            ]
        else:
            # logprobs模式下使用普通StreamStatus
            return [StreamStatus(request) for _ in range(n)]

    @override
    def render_chat(self, request: ChatCompletionRequest) -> RenderedInputs:
        """渲染聊天请求"""
        prompt: str = self._build_prompt(request)
        input_ids: List[int] = self._cached_encode(prompt)
        return RenderedInputs(input_ids=input_ids, rendered_prompt=prompt)

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

        # 创建Jinja2环境
        env = Environment(
            loader=BaseLoader(),
            trim_blocks=True,
            lstrip_blocks=True,
            extensions=["jinja2.ext.do", "jinja2.ext.loopcontrols"],
        )

        # 允许子类自定义环境
        self._customize_jinja_env(env)

        try:
            template = env.from_string(self.chat_template)
            rendered_prompt = template.render(**context)
            return rendered_prompt
        except Exception as e:
            logging.error(f"构建提示文本失败: {str(e)}")
            raise ValueError(f"Error rendering prompt template: {str(e)}")

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
        env.filters["tojson"] = lambda value: (
            value
            if isinstance(value, str)
            else json.dumps(value, sort_keys=False, ensure_ascii=False)
        )

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
            original_delta_string = status.delta_output_string
            tool_delta = await self._process_reasoning_and_tool_calls(
                status, output, is_streaming
            )
            if tool_delta is not None:
                status.update_result()
                return tool_delta
            elif original_delta_string != status.delta_output_string:
                status.update_result()

        if status.delta_output_string:
            status.update_result()
            delta = OutputDelta(
                output_str=status.delta_output_string,
                logprobs=await self._generate_log_probs(status, output),
                input_length=output.aux_info.input_len,
                output_length=output.aux_info.output_len,
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
        status.update_output(
            output,
            functools.partial(self._check_finish_reason, max_new_tokens=max_new_tokens),
            self._remove_stop_word_ids,
        )

        # Extract new token IDs from this iteration
        new_token_ids = status.output_ids[len(status.last_output_ids) :]
        normalizer = TokenNormalizer(self.tokenizer)

        collected_deltas = await self._process_normalized_tokens(
            normalizer,
            status,
            new_token_ids,
            output,
            stop_words_str,
            stop_word_slice_list,
            is_streaming,
        )

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
    ) -> List[OutputDelta]:
        """Process tokens through normalization and detector."""
        if is_streaming:
            return await self._process_streaming_tokens(
                normalizer,
                status,
                new_token_ids,
                output,
                stop_words_str,
                stop_word_slice_list,
            )
        else:
            return await self._process_non_streaming_tokens(
                normalizer,
                status,
                new_token_ids,
                output,
                stop_words_str,
                stop_word_slice_list,
            )

    async def _process_streaming_tokens(
        self,
        normalizer: TokenNormalizer,
        status: StreamStatus,
        new_token_ids: List[int],
        output: GenerateOutput,
        stop_words_str: List[str],
        stop_word_slice_list: List[str],
    ) -> List[OutputDelta]:
        """Process tokens one by one through detector."""
        collected_deltas = []
        for delta_text in normalizer.normalize_tokens(
            status.prev_token_id, new_token_ids
        ):
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
        return collected_deltas

    async def _process_non_streaming_tokens(
        self,
        normalizer: TokenNormalizer,
        status: StreamStatus,
        new_token_ids: List[int],
        output: GenerateOutput,
        stop_words_str: List[str],
        stop_word_slice_list: List[str],
    ) -> List[OutputDelta]:
        """Accumulate all text first, then process once."""
        all_text = "".join(
            normalizer.normalize_tokens(status.prev_token_id, new_token_ids)
        )
        if not all_text:
            return []

        complete_delta = await self._process_single_token_delta(
            status,
            all_text,
            output,
            stop_words_str,
            stop_word_slice_list,
            is_streaming=False,
        )
        return [complete_delta] if complete_delta is not None else []

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
            status.request.tools,
            remaining_after_reasoning,
            is_streaming,
        )

        status.delta_output_string = remaining_after_tools

        has_reasoning = bool(reasoning_text)
        has_tool_calls = tool_calls and len(tool_calls) > 0

        if not has_reasoning and not has_tool_calls:
            return None

        if has_tool_calls:
            status.generating_tool_call = True

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
            output_length=output.aux_info.output_len,
            reuse_length=output.aux_info.reuse_len,
        )

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
        tools: Optional[List[GPTToolDefinition]],
        text: str,
        is_streaming: bool,
    ) -> tuple[Optional[List[ToolCall]], str]:
        """
        Extract tool calls from text.

        Returns (tool_calls, remaining_text).
        """
        if not detector or not tools:
            return None, text

        # 转换工具格式
        sglang_tools = rtp_tools_to_sglang_tools(tools)

        try:
            if is_streaming:
                parse_result = detector.parse_streaming_increment(text, sglang_tools)
            else:
                cleaned_text = self._clean_stop_words(text)
                parse_result = detector.detect_and_parse(cleaned_text, sglang_tools)

            tool_calls, remaining_text = streaming_parse_result_to_tool_calls(
                parse_result
            )

            if not is_streaming:
                for i, tool_call in enumerate(tool_calls):
                    tool_call.index = i

            return tool_calls, remaining_text
        except Exception as e:
            logging.error(f"工具调用解析失败: {e}, 当前文本: {text}")
            return None, text

    def _clean_stop_words(self, text: str) -> str:
        """Clean stop words from text (default: no cleaning)."""
        return text
