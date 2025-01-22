import copy
import json
import logging
from typing import Optional, List, Dict, Any, Tuple, Union
import functools
from maga_transformer.models.base_model import GenerateOutput
from maga_transformer.tokenizer.tokenization_qwen import QWenTokenizer
from transformers import Qwen2Tokenizer
from maga_transformer.openai.api_datatype import (
    ChatMessage,
    GPTToolDefinition,
    ChatCompletionRequest,
    DeltaMessage,
    FinisheReason,
    ToolCall,
    FunctionCall,
)
from maga_transformer.openai.renderers.custom_renderer import (
    CustomChatRenderer,
    RendererParams,
    RenderedInputs,
    StreamStatus,
    OutputDelta,
)
from maga_transformer.openai.renderer_factory_register import register_renderer
from maga_transformer.utils.word_util import is_truncated

"""
TODO List
=========

- ChatMessage工具调用ID支持
  • 问题：需要在ChatMessage中增加tool_call_id字段
  • 现状：OpenAI官方SDK中没有该字段
  • 影响：不添加可能导致tool_call和tool_response顺序不对应，引起推理幻觉
  • 相关文档：https://aliyuque.antfin.com/aiplus/aistudio/gurqfferzgx2w9k9#vPzRS

- System Prompt中文支持

- System Prompt模板引擎
  • 任务：采用jinja模板引擎处理system_prompt

- System Prompt时间支持
  • 任务：在system_prompt中添加日期时间信息
  • 目的：减少AI推理过程中的时间相关幻觉

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


class QwenToolStreamStatus(StreamStatus):
    generating_tool_call: bool = False
    tool_call_index = 0
    tool_call_responded_string = ""


QwenTokenizerTypes = Union[QWenTokenizer, Qwen2Tokenizer]

# 采用的模板来源
# https://ollama.com/library/qwen2.5:72b/blobs/eb4402837c78
# https://qwen.readthedocs.io/en/latest/framework/function_call.html#qwen2-5-function-calling-templates
TOOL_INSTRUCTION = """
# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
{tool_definitions}

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>"""


class QwenToolRenderer(CustomChatRenderer):
    """QwenToolRenderer
    考虑到<|im_start|> ,<tool_call>等token都比较精简, 故不提取成常量, 增强直接可读性
    """

    def __init__(self, tokenizer: QwenTokenizerTypes, renderer_params: RendererParams):
        super().__init__(tokenizer, renderer_params)

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
        prompt = self._build_prompt(request.messages, request.tools)
        input_ids = self.tokenizer.encode(prompt)
        return RenderedInputs(input_ids=input_ids)

    def _build_prompt(
        self,
        messages: List[ChatMessage],
        tools: Optional[List[GPTToolDefinition]] = None,
    ) -> str:
        messages = copy.deepcopy(messages)
        prompt = ""

        # Handle system message and tools
        if messages and messages[0].role == "system":
            system_msg = messages.pop(0).content.lstrip("\n").rstrip()
        else:
            system_msg = "You are a helpful assistant."

        # Add system message
        prompt += f"<|im_start|>system\n{system_msg}\n"

        # Add tool definitions if present
        if tools:
            tool_definitions = self._format_tool_definitions(tools)
            prompt += TOOL_INSTRUCTION.format(tool_definitions=tool_definitions)
        prompt += f"<|im_end|>\n"

        # Handle conversation messages
        for i, message in enumerate(messages):
            is_last = i == len(messages) - 1
            prev_is_tool = i > 0 and messages[i - 1].role == "tool"
            next_is_tool = i < len(messages) - 1 and messages[i + 1].role == "tool"

            if message.role == "user":
                prompt += f"<|im_start|>user\n{message.content}<|im_end|>\n"

            elif message.role == "assistant":
                prompt += f"<|im_start|>assistant\n"
                if message.content:
                    prompt += message.content
                if message.tool_calls:
                    prompt += self._format_tool_calls(message.tool_calls)
                prompt += f"<|im_end|>\n"

            elif message.role == "tool":
                # 只有当不是连续tool消息中的一个时，才添加im_start和im_end
                if not prev_is_tool:
                    prompt += f"<|im_start|>user\n"

                prompt += f"<tool_response>\n{message.content}\n</tool_response>"

                if not next_is_tool:
                    prompt += f"<|im_end|>\n"
                else:
                    prompt += "\n"

            if not message.role == "assistant" and is_last:
                prompt += f"<|im_start|>assistant\n"

        return prompt

    def _format_tool_definitions(self, tools: List[GPTToolDefinition]) -> str:
        """格式化工具定义"""
        tool_defs = []
        for tool in tools:
            tool_def = {
                "type": "function",
                "function": {
                    "name": tool.function.name,
                    "description": tool.function.description,
                    "parameters": tool.function.parameters,
                },
            }
            tool_defs.append(json.dumps(tool_def, ensure_ascii=False))
        return f"<tools>\n" + "\n".join(tool_defs) + f"\n</tools>"

    def _format_tool_calls(self, tool_calls: List[ToolCall]) -> str:
        """格式化工具调用"""
        formatted_calls = []
        for tool_call in tool_calls:
            try:
                arguments = (
                    json.loads(tool_call.function.arguments)
                    if tool_call.function.arguments
                    else {}
                )
            except json.JSONDecodeError:
                arguments = tool_call.function.arguments

            tool_call_json = {"name": tool_call.function.name, "arguments": arguments}
            formatted_calls.append(
                f"<tool_call>\n{json.dumps(tool_call_json, ensure_ascii=False)}\n</tool_call>"
            )

        return "\n".join(formatted_calls)

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
            self._extract_tool_calls_from_complete_message(status.delta_output_string)
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
    ) -> Tuple[List[ToolCall], str]:
        """
        从文本中提取所有被 <tool_call> </tool_call> 标签包围的内容,并解析成 ToolCall 对象列表,
        同时从原文本中删除这些标签及其内容

        Args:
            status: 包含 tool_call 标签的状态对象

        Returns:
            List[ToolCall]: ToolCall 对象列表
        """
        tool_calls = []

        # 用 <tool_call> 分割
        parts = original_text.split("<tool_call>\n")

        # 保存第一部分(标签之前的内容)
        result_text = parts[0]

        # 处理剩余部分
        for index, part in enumerate(parts[1:]):
            # 用 </tool_call> 分割
            tool_parts = part.split("</tool_call>")
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
                    tool_calls.append(tool_call)
                except (json.JSONDecodeError, KeyError) as e:
                    logging.error(f"json loads error: {e}")

            # 如果有剩余文本，添加到结果中
            if len(tool_parts) > 1 and tool_parts[1] != "\n":
                result_text += tool_parts[1]

        return tool_calls, result_text

    async def _handle_streaming_case(
        self,
        status: QwenToolStreamStatus,
        output: GenerateOutput,
    ) -> Optional[OutputDelta]:
        # 如果是qwen2.5之前的qwen, 可能会提前stream出<tool_call这样的内容
        status.tool_call_responded_string += status.delta_output_string
        if "<tool_call>" in status.tool_call_responded_string:
            status.delta_output_string = ""
            if "</tool_call>" not in status.tool_call_responded_string:
                return await self._create_empty_delta(output.aux_info)
            else:
                # 提取和处理工具调用
                function_call, status.tool_call_responded_string = (
                    self._extract_function_call_from_streaming_message(
                        status.tool_call_responded_string
                    )
                )
                # 设置工具调用状态
                if not function_call:
                    # 如果没有提取到工具调用，则把完整的原始文本作为delta输出
                    status.delta_output_string = status.tool_call_responded_string
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
        self, text: str
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
        except (json.JSONDecodeError, KeyError) as e:
            # json提取失败的时候, 返回原始文本
            logging.error(f"qwen tool extract function call error: {str(e)}")
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


register_renderer("qwen_tool", QwenToolRenderer)
