import functools
import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from jinja2 import BaseLoader, Environment
from transformers import PreTrainedTokenizerBase
from typing_extensions import override

from rtp_llm.models.base_model import GenerateOutput
from rtp_llm.openai.api_datatype import ChatCompletionRequest, FinisheReason, RoleEnum
from rtp_llm.openai.renderers.custom_renderer import (
    CustomChatRenderer,
    OutputDelta,
    RenderedInputs,
    RendererParams,
    StreamStatus,
)
from rtp_llm.openai.renderers.sglang_helpers.format_convert_helper import (
    rtp_tools_to_sglang_tools,
    streaming_parse_result_to_delta_message,
)
from rtp_llm.openai.renderers.sglang_helpers.function_call.base_format_detector import (
    BaseFormatDetector,
)
from rtp_llm.utils.word_util import is_truncated


class ToolStreamStatus(StreamStatus):
    generating_tool_call: bool = False
    detector: BaseFormatDetector

    def __init__(self, request: ChatCompletionRequest, detector: BaseFormatDetector):
        super().__init__(request)
        self.generating_tool_call = False
        self.detector = detector
        self.compatible_pd_seperate_tag = bool(os.environ.get("PD_SEPARATION", 0))
        self.compatible_pd_seperate_first_token = None


class ToolBaseRenderer(CustomChatRenderer, ABC):
    """
    工具调用渲染器基类
    提供工具调用的通用逻辑，子类需要实现具体的检测器创建逻辑
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        renderer_params: RendererParams,
    ):
        super().__init__(tokenizer, renderer_params)
        self._setup_stop_words()

    def _setup_stop_words(self):
        """设置额外的停止词，子类可以重写"""
        pass

    @abstractmethod
    def _create_detector(self) -> BaseFormatDetector:
        """创建格式检测器，子类必须实现"""
        pass

    @override
    async def _create_status_list(
        self, n: int, request: ChatCompletionRequest
    ) -> List[StreamStatus]:
        """创建状态列表"""
        if request.tools and not request.logprobs:
            return [
                ToolStreamStatus(request, self._create_detector()) for _ in range(n)
            ]
        else:
            # logprobs模式下使用普通StreamStatus
            return [StreamStatus(request) for _ in range(n)]

    @override
    def render_chat(self, request: ChatCompletionRequest) -> RenderedInputs:
        """渲染聊天请求"""
        logging.info(f"开始渲染聊天请求，消息数量: {len(request.messages)}")
        prompt: str = self._build_prompt(request)
        input_ids: List[int] = self.tokenizer.encode(prompt)
        return RenderedInputs(input_ids=input_ids, rendered_prompt=prompt)

    def _build_prompt(self, request: ChatCompletionRequest) -> str:
        """
        构建提示文本
        Args:
            request: 聊天完成请求
        Returns:
            str: 格式化后的提示文本
        """
        context = request.model_dump(exclude_none=True)

        # 只要不是已经有assistant消息, 则需要添加生成提示
        if request.messages[-1].role != RoleEnum.assistant:
            context["add_generation_prompt"] = True

        # 合并chat_template_kwargs
        if request.chat_template_kwargs is not None:
            context.update(request.chat_template_kwargs)

        if (
            request.extend_fields is not None
            and "chat_template_kwargs" in request.extend_fields
            and isinstance(request.extend_fields["chat_template_kwargs"], dict)
        ):
            context.update(request.extend_fields["chat_template_kwargs"])

        # 创建Jinja2环境
        env = Environment(loader=BaseLoader(), trim_blocks=True, lstrip_blocks=True)

        # 允许子类自定义环境
        self._customize_jinja_env(env)

        try:
            template = env.from_string(self.tokenizer.chat_template)
            rendered_prompt = template.render(**context)
            logging.debug(f"提示文本构建成功")
            return rendered_prompt
        except Exception as e:
            logging.error(f"构建提示文本失败: {str(e)}")
            raise ValueError(f"Error rendering prompt template: {str(e)}")

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
            self._clean_output_ids,
            functools.partial(self._check_finish_reason, max_new_tokens=max_new_tokens),
            self._remove_stop_word_ids,
        )
        decoded_prev_token = self.tokenizer.decode(status.prev_token_id)
        decoded_string = self.tokenizer.decode(status.tokens_to_decode)
        # For some tokenizers (e.g. ChatGLM), decode a single token differs from decode a list of tokens.
        if is_streaming:
            if len(decoded_string) > 0 and "\uFFFD" == decoded_string[-1]:
                return await self._create_empty_delta(output.aux_info)
        else:
            while (len(decoded_string) > 0) and ("\uFFFD" == decoded_string[-1]):
                decoded_string = decoded_string[:-1]
        status.delta_output_string = decoded_string[len(decoded_prev_token) :]

        # 这里增加劫持逻辑
        if isinstance(status, ToolStreamStatus) and status.request.tools:
            tool_delta = await self._process_tool_calls(status, output, is_streaming)
            if tool_delta is not None:
                status.update_result()
                return tool_delta
        # 结束劫持的位置, 如果没有有关toolcalls的内容, 会将文本还给 delta_output_string，让默认逻辑处理

        if is_truncated(status.delta_output_string, stop_words_str, is_streaming):
            status.finish_reason = FinisheReason.stop
            return await self._create_empty_delta(output.aux_info)
        if not is_truncated(
            status.delta_output_string, stop_word_slice_list, is_streaming
        ):
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

    async def _process_tool_calls(
        self,
        status: ToolStreamStatus,
        output: GenerateOutput,
        is_streaming: bool,
    ) -> Optional[OutputDelta]:
        """处理工具调用的通用逻辑"""
        # 转换工具格式
        sglang_tools = rtp_tools_to_sglang_tools(status.request.tools)
        try:
            if is_streaming:
                parse_result = status.detector.parse_streaming_increment(
                    status.delta_output_string, sglang_tools
                )
            else:
                parse_result = self._handle_non_streaming_parse(status, sglang_tools)
                if parse_result is None:
                    return None

            # 如果没有工具调用，将文本还给 delta_output_string，让默认逻辑处理
            if not parse_result.calls:
                status.delta_output_string = parse_result.normal_text
                return None

            # 有工具调用时，使用格式转换函数
            delta_message = streaming_parse_result_to_delta_message(parse_result)

            # 创建输出delta
            delta = OutputDelta(
                output_str=delta_message,
                logprobs=await self._generate_log_probs(status, output),
                input_length=output.aux_info.input_len,
                output_length=output.aux_info.output_len,
                reuse_length=output.aux_info.reuse_len,
            )

            # 更新状态
            status.generating_tool_call = True
            status.delta_output_string = ""

            return delta

        except Exception as e:
            logging.error(f"工具调用解析失败: {e}")
            return None

    def _handle_non_streaming_parse(self, status: ToolStreamStatus, tools):
        """处理非流式解析的逻辑"""
        # 清理停止词
        status.delta_output_string = self._clean_stop_words(status.delta_output_string)

        # 处理PD分离兼容性
        if status.compatible_pd_seperate_tag:
            status.compatible_pd_seperate_tag = False
            status.compatible_pd_seperate_first_token = status.delta_output_string
            status.delta_output_string = ""
            return None
        else:
            if status.compatible_pd_seperate_first_token:
                status.delta_output_string = (
                    status.compatible_pd_seperate_first_token
                    + status.delta_output_string
                )

            # 执行检测和解析
            return status.detector.detect_and_parse(status.delta_output_string, tools)

    def _clean_stop_words(self, text: str):
        """
        清理文本中的停止词，子类必须实现

        Args:
            text: 需要清理的文本

        Returns:
            str: 清理后的文本
        """
        return text
