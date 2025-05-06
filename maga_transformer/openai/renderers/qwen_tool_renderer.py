from enum import Enum
import json
import logging
from typing import Optional, List, Tuple, Union
import functools
from maga_transformer.models.base_model import GenerateOutput
from maga_transformer.tokenizer.tokenization_qwen import QWenTokenizer
from transformers import Qwen2Tokenizer
from maga_transformer.openai.api_datatype import (
    ChatCompletionRequest,
    DeltaMessage,
    FinisheReason,
    RoleEnum,
    ToolCall,
    FunctionCall,
)
from maga_transformer.openai.renderers.custom_renderer import (
    CustomChatRenderer,
    RendererParams,
    RenderedInputs,
    StreamStatus,
    OutputDelta,
    ThinkStatus
)
from maga_transformer.openai.renderer_factory_register import register_renderer
from maga_transformer.utils.word_util import (
    is_truncated,
    truncate_response_with_stop_words,
)
from jinja2 import Environment, BaseLoader

"""
TODO List
=========

- ChatMessage工具调用ID支持
  • 问题：需要在ChatMessage中增加tool_call_id字段
  • 现状：OpenAI官方SDK中没有该字段
  • 影响：不添加可能导致tool_call和tool_response顺序不对应，引起推理幻觉
  • 相关文档：https://aliyuque.antfin.com/aiplus/aistudio/gurqfferzgx2w9k9#vPzRS

- System Prompt中文支持

- 对某些异常情况的支持:
  • 例如:
        为了给您提供北京、上海和伦敦的当前温度，我将分别查询这三个城市的温度。请稍等。
        <tool_call> 
        {"name": "get_current_temperature", "arguments": {"location": "北京, China", "unit": "celsius"}}
        </tool_call> 
        {"name": "get_current_temperature", "arguments": {"location": "上海, China", "unit": "celsius"}}
        </tool_call> 
        {"name": "get_current_temperature", "arguments": {"location": "London, UK", "unit": "celsius"}}
        </tool_call>
    而非标准的:
        <tool_call> 
        {"name": "get_current_temperature", "arguments": {"location": "北京, China", "unit": "celsius"}}
        </tool_call> 
        <tool_call> 
        {"name": "get_current_temperature", "arguments": {"location": "上海, China", "unit": "celsius"}}
        </tool_call>

- tool_choice的支持        

"""


class ToolCallMessageExtractStrategy(str, Enum):
    DEFAULT = "default"
    SKIP_ON_FAILURE = "skip_on_failure"
    # maybe useful: ERROR_ON_FAILURE

    @classmethod
    def from_extra_configs(cls, request: ChatCompletionRequest):
        """从请求配置中提取工具调用消息提取策略"""
        strategy = cls.DEFAULT
        if extra_configs := request.extra_configs:
            if extra_configs.tool_call_message_extract_strategy == cls.SKIP_ON_FAILURE:
                strategy = cls.SKIP_ON_FAILURE
        return strategy


class QwenToolStreamStatus(StreamStatus):
    generating_tool_call: bool = False
    tool_call_index = 0
    tool_call_responded_string = ""
    tool_call_message_extract_strategy: ToolCallMessageExtractStrategy = (
        ToolCallMessageExtractStrategy.DEFAULT
    )


QwenTokenizerTypes = Union[QWenTokenizer, Qwen2Tokenizer]


JINJA_TEMPLATE = """{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- messages[0]['content'] }}\n    {%- else %}\n        {{- 'You are a helpful assistant.' }}\n    {%- endif %}\n    {{- \"\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}\n    {%- else %}\n        {{- '<|im_start|>system\\nYou are a helpful assistant.<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- for message in messages %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) or (message.role == \"assistant\" and not message.tool_calls) %}\n        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {{- '<|im_start|>' + message.role }}\n        {%- if message.content %}\n            {{- '\\n' + message.content }}\n        {%- endif %}\n        {%- for tool_call in message.tool_calls %}\n            {%- if tool_call.function is defined %}\n                {%- set tool_call = tool_call.function %}\n            {%- endif %}\n            {{- '\\n<tool_call>\\n{\"name\": \"' }}\n            {{- tool_call.name }}\n            {{- '\", \"arguments\": ' }}\n            {{- tool_call.arguments | tojson }}\n            {{- '}\\n</tool_call>' }}\n        {%- endfor %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- message.content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n{%- endif %}\n"""


class QwenToolRenderer(CustomChatRenderer):
    """QwenToolRenderer
    考虑到<|im_start|> ,<tool_call>等token都比较精简, 故不提取成常量, 增强直接可读性
    """

    def __init__(self, tokenizer: QwenTokenizerTypes, renderer_params: RendererParams):
        super().__init__(tokenizer, renderer_params)
        if not tokenizer.chat_template or 'tool' not in tokenizer.chat_template:
            self.chat_template = JINJA_TEMPLATE
        else:
            self.chat_template = tokenizer.chat_template

    # override
    async def _create_status_list(
        self, n: int, request: ChatCompletionRequest
    ) -> List[StreamStatus]:
        if request.logprobs:
            return [StreamStatus(request) for _ in range(n)]
        else:
            return [QwenToolStreamStatus(request) for _ in range(n)]

    # override
    def render_chat(self, request: ChatCompletionRequest) -> RenderedInputs:
        prompt: str = self._build_prompt(request)
        input_ids: List[int] = self.tokenizer.encode(prompt)  # type: ignore
        return RenderedInputs(input_ids=input_ids)

    def _build_prompt(
        self,
        request: ChatCompletionRequest,
    ) -> str:
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

        env = Environment(loader=BaseLoader())
        # 重写tojson过滤器, 这里存在三个注意点
        # 1. tojson过滤器默认会排序, 导致生成的json字符串不符合预期
        # 2. tojson过滤器默认会转义汉字, 导致生成的json字符串不符合预期
        # 3. arguments默认是str, 而官方模板会对json str再次dumps

        env.filters["tojson"] = lambda value: (
            value
            if isinstance(value, str)
            else json.dumps(value, sort_keys=False, ensure_ascii=False)
        )

        try:
            # 使用自定义环境创建模板
            template = env.from_string(self.chat_template)
            rendered_prompt = template.render(**context)
            return rendered_prompt
        except Exception as e:
            raise ValueError(f"Error rendering prompt template: {str(e)}")

    async def _update_single_status(
        self,
        status: StreamStatus,
        output: GenerateOutput,
        max_new_tokens: int,
        stop_words_str: List[str],
        stop_word_slice_list: List[str],
        is_streaming: bool,
    ) -> OutputDelta:
        if status.finish_reason != None:  # type: ignore
            return await self._create_empty_delta(status.output.aux_info)  # type: ignore
        status.update_output(  # type: ignore
            output,
            self._clean_output_ids,
            functools.partial(self._check_finish_reason, max_new_tokens=max_new_tokens),
            self._remove_stop_word_ids,
        )
        decoded_prev_token = self.tokenizer.decode(status.prev_token_id)  # type: ignore
        decoded_string = self.tokenizer.decode(status.tokens_to_decode)  # type: ignore
        # For some tokenizers (e.g. ChatGLM), decode a single token differs from decode a list of tokens.
        if is_streaming:
            if len(decoded_string) > 0 and "\uFFFD" == decoded_string[-1]:
                return await self._create_empty_delta(output.aux_info)
        else:
            while (len(decoded_string) > 0) and ("\uFFFD" == decoded_string[-1]):
                decoded_string = decoded_string[:-1]
        status.delta_output_string = decoded_string[len(decoded_prev_token) :]

        # <qwen_tool_renderer>, 其他部分同父类custom_renderer保持一致
        if isinstance(status, QwenToolStreamStatus) and status.request.tools:
            tool_delta = await self.process_tool_calls(status, output, is_streaming)
            # tool_delta为None代表继续默认逻辑处理
            if tool_delta is not None:
                status.update_result()
                return tool_delta
        # </qwen_tool_renderer>

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

    async def process_tool_calls(
        self,
        status: QwenToolStreamStatus,
        output: GenerateOutput,
        is_streaming: bool,
    ) -> Optional[OutputDelta]:
        status.tool_call_message_extract_strategy = (
            ToolCallMessageExtractStrategy.from_extra_configs(status.request)
        )

        return await (
            self._handle_streaming_case(status, output)
            if is_streaming
            else self._handle_no_streaming_case(status, output)
        )

    async def _handle_no_streaming_case(
        self,
        status: QwenToolStreamStatus,
        output: GenerateOutput,
    ) -> OutputDelta:
        # before: status.delta_output_string = "你好\n<tool_call>\n{"name": "get_current_temperature", "arguments": {"location": "北京, China", "unit": "celsius"}}</tool_call>"
        # after: status.delta_output_string = "你好\n"
        tool_calls, status.delta_output_string = (
            self._extract_tool_calls_from_complete_message(
                status.delta_output_string,
                status.tool_call_message_extract_strategy,
            )
        )
        if tool_calls:
            status.generating_tool_call = True
        return OutputDelta(
            output_str=DeltaMessage(
                tool_calls=tool_calls,
            ),
            logprobs=await self._generate_log_probs(status, output),
            input_length=output.aux_info.input_len,
            output_length=output.aux_info.output_len,
            reuse_length=output.aux_info.reuse_len,
        )

    def _extract_tool_calls_from_complete_message(
        self,
        original_text: str,
        tool_call_message_extract_strategy: ToolCallMessageExtractStrategy = ToolCallMessageExtractStrategy.DEFAULT,
        tool_call_begin_tag: str = "<tool_call>",
        tool_call_end_tag: str = "</tool_call>",
    ) -> Tuple[Optional[List[ToolCall]], str]:
        """
        从文本中提取所有被 <tool_call> </tool_call> 标签包围的内容,并解析成 ToolCall 对象列表,
        同时从原文本中删除这些标签及其内容

        Args:
            status: 包含 tool_call 标签的状态对象

        Returns:
            List[ToolCall]: ToolCall 对象列表
        """
        tool_calls: List[ToolCall] = []

        parts = original_text.split(tool_call_begin_tag)

        # 保存第一部分(标签之前的内容)
        result_text = parts[0]

        # 处理剩余部分

        index = 0
        for part in parts[1:]:
            if tool_call_end_tag not in part:
                # 认为由<tool_call>开始但是没有</tool_call>结束的是一种failure, skip_on_failure情况下, 就不给予解析
                # 例如 <tool_call>{"name": "get_current_temperature", "arguments": {"location": "北京, China", "unit": "celsius"}}<tool_call>{"name": "get_current_temperature", "arguments": {"location": "上海, China", "unit": "celsius"}}</tool_call>的北京就不允许解析
                if (
                    tool_call_message_extract_strategy
                    == ToolCallMessageExtractStrategy.SKIP_ON_FAILURE
                ):
                    continue
                result_text += tool_call_begin_tag + part
                continue

            tool_parts = part.split(tool_call_end_tag)
            content = tool_parts[0].strip()

            if content:
                try:
                    # 解析 JSON 内容
                    function_name, function_args = self._extract_name_args(content)
                    # 创建 FunctionCall 对象
                    function_call = FunctionCall(
                        name=function_name, arguments=function_args
                    )
                    # 创建 ToolCall 对象
                    tool_call = ToolCall(
                        index=index,
                        id=self._generate_random_call_id(),
                        type="function",
                        function=function_call,
                    )
                    index += 1
                    tool_calls.append(tool_call)
                except Exception as e:
                    logging.error(
                        f"Extract function call from complete message error: {e}"
                    )
                    # begin/end tag之中不是一个正确的json, 也认为是一种failure
                    # 例如<tool_call>{"name": "get_current_temperature", "arguments": error_args}</tool_call>
                    if (
                        tool_call_message_extract_strategy
                        == ToolCallMessageExtractStrategy.SKIP_ON_FAILURE
                    ):
                        continue

                    result_text += tool_call_begin_tag + part
                    continue
            # 如果有剩余文本，添加到结果中
            if len(tool_parts) > 1:
                result_text += tool_parts[1]

        if tool_calls:
            # 如果只有\n在result_text中, 则去掉
            result_text = result_text.strip()
            return tool_calls, result_text
        else:
            return None, result_text

    async def _handle_streaming_case(
        self,
        status: QwenToolStreamStatus,
        output: GenerateOutput,
    ) -> Optional[OutputDelta]:
        # 如果是qwen2.5之前的qwen, 可能会提前stream出<tool_call这样的内容
        if (
            "<tool_call>" in status.tool_call_responded_string
            or status.delta_output_string == "<tool_call>"
        ):
            status.tool_call_responded_string += status.delta_output_string
            status.delta_output_string = ""
            if "</tool_call>" not in status.tool_call_responded_string:
                return await self._create_empty_delta(output.aux_info)
            else:
                # 提取和处理工具调用
                function_call, status.tool_call_responded_string = (
                    self._extract_function_call_from_streaming_message(
                        status.tool_call_responded_string,
                        status.tool_call_message_extract_strategy,
                    )
                )
                # 设置工具调用状态
                if not function_call:
                    # 如果没有提取到工具调用，则把完整的原始文本作为delta输出
                    status.delta_output_string = status.tool_call_responded_string
                    status.tool_call_responded_string = ""
                else:
                    status.generating_tool_call = True
                    # 创建工具调用的delta输出
                    delta = OutputDelta(
                        output_str=DeltaMessage(
                            tool_calls=[
                                ToolCall(
                                    index=status.tool_call_index,
                                    id=self._generate_random_call_id(),
                                    type="function",
                                    function=function_call,
                                )
                            ]
                        ),
                        logprobs=await self._generate_log_probs(status, output),
                        input_length=output.aux_info.input_len,
                        output_length=output.aux_info.output_len,
                        reuse_length=output.aux_info.reuse_len,
                    )
                    status.tool_call_index += 1
                    return delta

        """
        处理特殊换行情况, 例如
        <tool_call>
        {"name": "get_current_temperature", "arguments": {"location": "北京, China", "unit": "celsius"}}
        </tool_call>\n<tool_call>
        {"name": "get_current_temperature", "arguments": {"location": "上海, China", "unit": "celsius"}}
        </tool_call>
        """

        if status.generating_tool_call and status.delta_output_string == "\n":
            return await self._create_empty_delta(output.aux_info)

        # 返回None的意义是让_update_single_status按照默认逻辑继续返回status.delta_output_string的delta
        return None

    def _extract_function_call_from_streaming_message(
        self,
        text: str,
        tool_call_message_extract_strategy: ToolCallMessageExtractStrategy = ToolCallMessageExtractStrategy.DEFAULT,
    ) -> Tuple[Optional[FunctionCall], str]:
        tool_call_name_args_str = text[
            text.index("<tool_call>") + len("<tool_call>") : text.index("</tool_call>")
        ]

        # 更新tool_call_responded_string
        extracted_text = (
            text[: text.index("<tool_call>")]
            + text[text.index("</tool_call>") + len("</tool_call>") :]
        )

        # 解析工具调用参数
        try:
            function_name, function_args = self._extract_name_args(
                tool_call_name_args_str
            )
            return (
                FunctionCall(name=function_name, arguments=function_args),
                extracted_text,
            )
        except Exception as e:
            # json提取失败的时候, 返回原始文本
            logging.error(f"qwen tool extract function call error: {str(e)}")
            if (
                tool_call_message_extract_strategy
                == ToolCallMessageExtractStrategy.SKIP_ON_FAILURE
            ):
                return None, ""
            return None, text

    def _generate_random_call_id(self, length: int = 24) -> str:
        """生成随机调用ID"""
        import secrets
        import string

        characters = string.ascii_letters + string.digits
        random_string = "".join(secrets.choice(characters) for _ in range(length))
        return "call_" + random_string

    def _extract_name_args(self, text: str):
        try:
            data = json.loads(text)
            function_name = data["name"]
            # 将 arguments 转换为标准 JSON 字符串格式
            function_args = json.dumps(data["arguments"], ensure_ascii=False)
            return function_name, function_args
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {str(e)}")
        except KeyError as e:
            raise ValueError(f"Missing required field: {str(e)}")
        except Exception as e:
            raise ValueError(f"Unknown error: {str(e)}")

    async def _flush_buffer(
        self,
        buffer_list: List[StreamStatus],
        stop_words_str: List[str],
        is_streaming: bool,
        think_status: ThinkStatus
    ):
        output_items: List[OutputDelta] = []
        for buffer in buffer_list:
            # 解被截断的bad_case
            # "response":"<tool_call>\n{\"name\": \"get_average_month"
            if (
                isinstance(buffer, QwenToolStreamStatus)
                and buffer.tool_call_responded_string
                and "<tool_call>" in buffer.tool_call_responded_string
                and buffer.tool_call_message_extract_strategy
                == ToolCallMessageExtractStrategy.DEFAULT
            ):
                buffer.delta_output_string += buffer.tool_call_responded_string

            if buffer.output is None:
                raise Exception("last output should not be None")
            aux_info = buffer.output.aux_info
            trunc_string = truncate_response_with_stop_words(
                buffer.delta_output_string, stop_words_str, is_streaming
            )
            output_items.append(
                OutputDelta(
                    trunc_string,
                    await self._generate_log_probs(buffer, buffer.output),
                    aux_info.input_len,
                    aux_info.output_len,
                    aux_info.reuse_len,
                )
            )
        return await self._generate_stream_response(output_items, think_status)


register_renderer("qwen_tool", QwenToolRenderer)
