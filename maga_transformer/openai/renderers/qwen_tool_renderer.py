import copy
import json
import re
import logging
import torch
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union, Callable, Tuple, AsyncGenerator
import functools
import pdb
from maga_transformer.models.base_model import GenerateOutput, GenerateOutputs
from maga_transformer.config.generate_config import GenerateConfig
from maga_transformer.tokenizer.tokenization_qwen import QWenTokenizer
from transformers import Qwen2Tokenizer
from maga_transformer.openai.api_datatype import (
    ChatMessage,
    GPTFunctionDefinition,
    GPTToolDefinition,
    UsageInfo,
    ChatCompletionRequest,
    ChatCompletionResponseStreamChoice,
    DeltaMessage,
    FinisheReason,
    RoleEnum,
    RendererInfo,
    PromptTokensDetails,
    ChatCompletionTokenLogprob,
    TopLogprob,
    ChoiceLogprobs,
    ToolCall,
    FunctionCall,
)
from maga_transformer.openai.renderers.custom_renderer import (
    CustomChatRenderer,
    RendererParams,
    StreamResponseObject,
    RenderedInputs,
    StreamStatus,
    OutputDelta,
)
from maga_transformer.openai.renderers.basic_renderer import BasicRenderer
from maga_transformer.openai.renderer_factory_register import register_renderer
from maga_transformer.utils.word_util import (
    get_stop_word_slices,
    truncate_response_with_stop_words,
    is_truncated,
)

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


"""


class QwenToolStreamStatus(StreamStatus):
    generating_tool_call: bool = False
    tool_call_index = 0


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
        self.add_extra_stop_word_ids([[self.tokenizer.encode("<|im_end|>")[0]]])

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
            tool_defs.append(json.dumps(tool_def))
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
                f"<tool_call>\n{json.dumps(tool_call_json)}\n</tool_call>"
            )

        return "\n".join(formatted_calls)

    async def _process_tool_calls(
        self, status: QwenToolStreamStatus, output: GenerateOutput
    ) -> Optional[OutputDelta]:
        """处理工具调用相关的逻辑"""
        if "<tool_call>" in status.responded_string:
            status.delta_output_string = ""
            if "</tool_call>" in status.responded_string:
                # 提取和处理工具调用
                tool_call_name_args_str = status.responded_string[
                    status.responded_string.index("<tool_call>")
                    + len("<tool_call>") : status.responded_string.index("</tool_call>")
                ]

                # 更新responded_string
                status.responded_string = (
                    status.responded_string[
                        : status.responded_string.index("<tool_call>")
                    ]
                    + status.responded_string[
                        status.responded_string.index("</tool_call>")
                        + len("</tool_call>") :
                    ]
                )

                # 解析工具调用参数
                tool_call_name_args = json.loads(tool_call_name_args_str)
                function_name = str(tool_call_name_args["name"])
                function_args = str(tool_call_name_args["arguments"])

                # 设置工具调用状态
                status.generating_tool_call = True

                # 创建工具调用的delta输出
                delta = OutputDelta(
                    output_str=DeltaMessage(
                        tool_calls=[
                            ToolCall(
                                index=status.tool_call_index,
                                id=self._generate_random_call_id(),
                                type="function",
                                function=FunctionCall(
                                    name=function_name, arguments=function_args
                                ),
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
            else:
                return await self._create_empty_delta(output.aux_info)

        # 处理特殊换行情况
        if status.generating_tool_call and status.delta_output_string == "\n":
            return await self._create_empty_delta(output.aux_info)

        return None

    def _generate_random_call_id(self, length: int = 24) -> str:
        """生成随机调用ID"""
        import secrets
        import string

        characters = string.ascii_letters + string.digits
        random_string = "".join(secrets.choice(characters) for _ in range(length))
        return "call_" + random_string

    # override
    async def _update_single_status(
        self,
        status: QwenToolStreamStatus,
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
        if is_truncated(status.delta_output_string, stop_words_str, is_streaming):
            status.finish_reason = FinisheReason.stop
            return await self._create_empty_delta(output.aux_info)
        if not is_truncated(
            status.delta_output_string, stop_word_slice_list, is_streaming
        ):
            status.update_result()
            # 事实上的修改就下面4行
            if status.request.tools:
                tool_delta = await self._process_tool_calls(status, output)
                if tool_delta is not None:
                    return tool_delta

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


register_renderer("qwen_tool", QwenToolRenderer)
