import copy
import json
import re
import logging
import torch
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union, Callable, Tuple, AsyncGenerator
import functools

from maga_transformer.models.base_model import GenerateOutput, GenerateOutputs
from maga_transformer.config.generate_config import GenerateConfig
from maga_transformer.tokenizer.tokenization_qwen import QWenTokenizer
from transformers import Qwen2Tokenizer
from maga_transformer.openai.api_datatype import (
    ChatMessage,
    GPTFunctionDefinition,
    GPTToolDefinition,
    ChatCompletionRequest,
    RoleEnum,
    FunctionCall,
    ToolCall,
    ChatCompletionResponseStreamChoice,
    DeltaMessage,
    FinisheReason,
    UsageInfo,
    RendererInfo,
    PromptTokensDetails,
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

QwenTokenizerTypes = Union[QWenTokenizer, Qwen2Tokenizer]


class QwenToolRenderer(CustomChatRenderer):
    IM_START = "<|im_start|>"
    IM_END = "<|im_end|>"
    TOOL_START = "<tools>"
    TOOL_END = "</tools>"
    TOOL_CALL_START = "<tool_call>"
    TOOL_CALL_END = "</tool_call>"

    TOOL_INSTRUCTION = """
# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
{tool_definitions}

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>"""

    def __init__(self, tokenizer: QwenTokenizerTypes, renderer_params: RendererParams):
        super().__init__(tokenizer, renderer_params)
        self.add_extra_stop_word_ids([[self.tokenizer.encode(self.IM_END)[0]]])

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
        prompt += f"{self.IM_START}system\n{system_msg}\n"

        # Add tool definitions if present
        if tools:
            tool_definitions = self._format_tool_definitions(tools)
            prompt += self.TOOL_INSTRUCTION.format(tool_definitions=tool_definitions)
        prompt += f"{self.IM_END}\n"

        # Handle conversation messages
        for i, message in enumerate(messages):
            is_last = i == len(messages) - 1
            prev_is_tool = i > 0 and messages[i - 1].role == "tool"
            next_is_tool = i < len(messages) - 1 and messages[i + 1].role == "tool"

            if message.role == "user":
                prompt += f"{self.IM_START}user\n{message.content}{self.IM_END}\n"

            elif message.role == "assistant":
                prompt += f"{self.IM_START}assistant\n"
                if message.content:
                    prompt += message.content
                if message.tool_calls:
                    prompt += self._format_tool_calls(message.tool_calls)
                prompt += f"{self.IM_END}\n"

            elif message.role == "tool":
                # 只有当不是连续tool消息中的一个时，才添加im_start和im_end
                if not prev_is_tool:
                    prompt += f"{self.IM_START}user\n"

                prompt += f"<tool_response>\n{message.content}\n</tool_response>"

                if not next_is_tool:
                    prompt += f"{self.IM_END}\n"
                else:
                    prompt += "\n"

            if not message.role == "assistant" and is_last:
                prompt += f"{self.IM_START}assistant\n"

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
        return f"{self.TOOL_START}\n" + "\n".join(tool_defs) + f"\n{self.TOOL_END}"

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
                f"{self.TOOL_CALL_START}\n{json.dumps(tool_call_json)}\n{self.TOOL_CALL_END}"
            )

        return "\n".join(formatted_calls)


register_renderer("qwen_tool", QwenToolRenderer)
