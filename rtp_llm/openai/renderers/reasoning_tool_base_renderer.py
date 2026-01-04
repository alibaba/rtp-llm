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
        decoded_prev_token = self.tokenizer.decode(status.prev_token_id)
        decoded_string = self.tokenizer.decode(status.tokens_to_decode)
        logging.info(
            f"[REASONING_DEBUG] prev_token_id={status.prev_token_id}, tokens_to_decode={status.tokens_to_decode}, "
            f"decoded_prev={repr(decoded_prev_token)}, decoded_full={repr(decoded_string)}"
        )
        # For some tokenizers (e.g. ChatGLM), decode a single token differs from decode a list of tokens.
        if is_streaming:
            if len(decoded_string) > 0 and "\uFFFD" == decoded_string[-1]:
                return await self._create_empty_delta(output.aux_info)
        else:
            while (len(decoded_string) > 0) and ("\uFFFD" == decoded_string[-1]):
                decoded_string = decoded_string[:-1]
        status.delta_output_string = decoded_string[len(decoded_prev_token) :]
        logging.info(
            f"[REASONING_DEBUG] delta_output_string={repr(status.delta_output_string)}"
        )

        # Process stop words: truncate complete stop words, detect partial stop words
        status.delta_output_string, should_buffer = self._process_stop_words(
            status.delta_output_string,
            stop_words_str,
            stop_word_slice_list,
            is_streaming,
            status,
        )

        # If detected a partial stop word, we must buffer and not process
        # any content yet, to avoid double-counting when the buffered content arrives next chunk.
        if should_buffer:
            return await self._create_empty_delta(output.aux_info)

        # Process reasoning and tool calls (operates on stop-word-truncated text)
        if isinstance(status, ReasoningToolStreamStatus) and (
            status.detector or status.reasoning_parser
        ):
            original_delta_string = status.delta_output_string
            tool_delta = await self._process_reasoning_and_tool_calls(
                status, output, is_streaming
            )
            if tool_delta is not None:
                logging.info(
                    f"[REASONING_DEBUG] tool_delta returned, calling update_result()"
                )
                status.update_result()
                return tool_delta
            # Parser consumed some text (changed delta_output_string) but didn't return delta yet
            # Still need to mark consumed portion as processed, otherwise next chunk has wrong delta
            # This is safe because we already checked should_buffer=False above.
            elif original_delta_string != status.delta_output_string:
                buffer_content = ""
                if (
                    status.reasoning_parser
                    and hasattr(status.reasoning_parser, "detector")
                    and hasattr(status.reasoning_parser.detector, "_buffer")
                ):
                    buffer_content = status.reasoning_parser.detector._buffer
                logging.info(
                    f"[REASONING_DEBUG] delta changed: original={repr(original_delta_string)} -> new={repr(status.delta_output_string)}, "
                    f"buffer={repr(buffer_content)}, calling update_result()"
                )
                status.update_result()

        # Build delta output for remaining normal content
        if len(status.delta_output_string) > 0:
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
        else:
            return await self._create_empty_delta(output.aux_info)

    async def _process_reasoning_and_tool_calls(
        self,
        status: ReasoningToolStreamStatus,
        output: GenerateOutput,
        is_streaming: bool,
    ) -> Optional[OutputDelta]:
        """
        Process reasoning text and tool calls from delta_output_string.

        This method extracts reasoning content (e.g., <think>...</think>) and tool calls
        from status.delta_output_string. It consumes the parsed portions and updates
        status.delta_output_string with any remaining unparsed text.

        Args:
            status: The stream status object containing delta_output_string to parse
            output: The generation output
            is_streaming: Whether in streaming mode

        Returns:
            OutputDelta if reasoning/tool content was parsed, None otherwise
            - If None: no reasoning/tool content found, caller should use default logic
            - If OutputDelta: contains reasoning_content and/or tool_calls fields

        Side effects:
            - Updates status.delta_output_string with remaining text after parsing
            - Sets status.generating_tool_call = True if tool calls found

        Example flow:
            Input: "<think>思考</think>正常文本<tool_call>...</tool_call>"
            1. Extract reasoning: "思考", remaining: "正常文本<tool_call>...</tool_call>"
            2. Extract tool calls: parsed tools, remaining: "正常文本"
            3. status.delta_output_string = "正常文本"
            4. Return OutputDelta with reasoning_content="思考", tool_calls=[...], content="正常文本"
        """

        original_text = status.delta_output_string

        # 提取推理文本
        reasoning_text, remaining_after_reasoning = self._extract_reasoning_content(
            status.reasoning_parser, original_text, is_streaming
        )

        # 从剩余文本中提取工具调用
        tool_calls, remaining_after_tools = await self._extract_tool_calls_content(
            status.detector,
            status.request.tools,
            remaining_after_reasoning,
            is_streaming,
        )

        status.delta_output_string = remaining_after_tools

        # 检查是否有需要处理的内容
        has_reasoning = bool(reasoning_text)
        has_tool_calls = tool_calls is not None and len(tool_calls) > 0

        # 如果没有推理文本和工具调用，返回None让默认逻辑处理普通文本
        if not has_reasoning and not has_tool_calls:
            return None

        # 更新状态
        if has_tool_calls:
            status.generating_tool_call = True

        # Include any remaining normal text in the delta's content field
        # ONLY in streaming mode for MTP scenarios where reasoning ends mid-chunk
        # In non-streaming mode, all content is processed at once, so don't include it here
        remaining_content = (
            remaining_after_tools if is_streaming and remaining_after_tools else None
        )

        # 创建OutputDelta
        delta = OutputDelta(
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

        return delta

    def _extract_reasoning_content(
        self,
        reasoning_parser: Optional[ReasoningParser],
        text: str,
        is_streaming: bool,
    ) -> Tuple[str, str]:
        """
        从文本中提取推理内容

        Args:
            reasoning_parser: 推理解析器，如果为None则不处理推理文本
            text: 待处理的文本
            is_streaming: 是否为流式处理

        Returns:
            tuple[str, str]: (提取的推理文本, 剩余的文本)
        """

        # 如果没有推理解析器，直接返回
        if reasoning_parser is None:
            return "", text

        try:
            if is_streaming:
                # 流式处理：增量解析
                reasoning_text, normal_text = reasoning_parser.parse_stream_chunk(text)
            else:
                # 非流式处理：一次性解析
                reasoning_text, normal_text = reasoning_parser.parse_non_stream(text)

            return reasoning_text, normal_text

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
        从文本中提取工具调用内容

        Args:
            detector: 格式检测器
            tools: 工具定义列表
            text: 待处理的文本
            is_streaming: 是否为流式处理

        Returns:
            tuple[Optional[List[ToolCall]], str]: (提取的工具调用列表, 剩余的文本)
        """

        # 如果没有工具定义，直接返回
        if not detector:
            return None, text

        # 转换工具格式
        sglang_tools = rtp_tools_to_sglang_tools(tools)

        try:
            if is_streaming:
                parse_result = detector.parse_streaming_increment(text, sglang_tools)
            else:
                # 兼容kimik2在非流式的情况下可能返回结果中有以<|im_end|>的结果
                cleaned_text = self._clean_stop_words(text)
                parse_result = detector.detect_and_parse(cleaned_text, sglang_tools)

            # 有工具调用时，使用格式转换函数
            tool_calls, remaining_text = streaming_parse_result_to_tool_calls(
                parse_result
            )

            # 统一为非流式场景的tool_call重新分配index以兼容sglang中非流式大部分index为-1的情况
            if not is_streaming:
                for i, tool_call in enumerate(tool_calls):
                    tool_call.index = i

            return tool_calls, remaining_text

        except Exception as e:
            logging.error(f"工具调用解析失败: {e}, 当前文本: {text}")
            return None, text

    def _clean_stop_words(self, text: str):
        """
        清理文本中的停止词, 默认不做清理

        Args:
            text: 需要清理的文本

        Returns:
            str: 清理后的文本
        """
        return text
