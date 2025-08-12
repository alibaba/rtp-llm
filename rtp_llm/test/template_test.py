import logging

logging.basicConfig(level=logging.INFO)
import os
from typing import Any
from unittest import TestCase, main

import torch
from transformers import AutoTokenizer, PreTrainedTokenizer
from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer

from rtp_llm.config.py_config_modules import StaticConfig
from rtp_llm.models.llava import LlavaTokenizer
from rtp_llm.models.starcoder import StarcoderTokenizer
from rtp_llm.openai.api_datatype import (
    ChatCompletionRequest,
    ChatMessage,
    ContentPart,
    ContentPartTypeEnum,
    FunctionCall,
    GPTFunctionDefinition,
    GPTToolDefinition,
    RendererInfo,
    RoleEnum,
    ToolCall,
)
from rtp_llm.openai.renderer_factory import (
    ChatRendererFactory,
    CustomChatRenderer,
    FastChatRenderer,
    LlamaTemplateRenderer,
    RendererParams,
)
from rtp_llm.openai.renderers.qwen_agent_renderer import QwenAgentRenderer
from rtp_llm.openai.renderers.qwen_agent_tool_renderer import QwenAgentToolRenderer
from rtp_llm.openai.renderers.qwen_reasoning_tool_renderer import (
    QwenReasoningToolRenderer,
)
from rtp_llm.openai.renderers.qwen_renderer import QwenRenderer
from rtp_llm.pipeline.chatapi_format import encode_chatapi
from rtp_llm.tokenizer.tokenization_qwen import QWenTokenizer


class TemplateTest(TestCase):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.test_data_path = os.path.join(os.getcwd(), "rtp_llm/test")

    def test_qwen_agent(self):
        tokenizer = QWenTokenizer(
            f"{self.test_data_path}/model_test/fake_test/testdata/qwen_7b/tokenizer/qwen.tiktoken"
        )
        render_params = RendererParams(
            model_type="qwen_agent",
            max_seq_len=1024,
            eos_token_id=tokenizer.eos_token_id or 0,
            stop_word_ids_list=[],
        )
        chat_renderer = ChatRendererFactory.get_renderer(tokenizer, render_params)
        logging.info(
            f"------- chat_renderer.get_renderer_info(): {chat_renderer.get_renderer_info()}"
        )
        assert isinstance(chat_renderer, QwenAgentRenderer)

        functions = [
            GPTFunctionDefinition(
                **{
                    "name": "get_current_weather",
                    "description": "Get the current weather in a given location.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA",
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                            },
                        },
                        "required": ["location"],
                    },
                }
            )
        ]

        messages = [
            ChatMessage(
                **{
                    "role": RoleEnum.system,
                    "content": "你是一个有用的助手",
                }
            ),
            ChatMessage(
                **{
                    "role": RoleEnum.user,
                    "content": "波士顿天气如何？",
                }
            ),
        ]

        request = ChatCompletionRequest(
            **{
                "messages": messages,
                "functions": functions,
                "stream": False,
                "extend_fields": {"lang": "zh"},
            }
        )

        ids = chat_renderer.render_chat(request).input_ids
        prompt = tokenizer.decode(ids)
        logging.info(
            f"rendered prompt: \n{prompt}\n-----------------------------------"
        )
        expected_prompt = """<|im_start|>system
你是一个有用的助手

# 工具

## 你拥有如下工具：

### get_current_weather

get_current_weather: Get the current weather in a given location. 输入参数：{"type": "object", "properties": {"location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["location"]} 此工具的输入应为JSON对象。

## 你可以在回复中插入零次、一次或多次以下命令以调用工具：

✿FUNCTION✿: 工具名称，必须是[get_current_weather]之一。
✿ARGS✿: 工具输入
✿RESULT✿: 工具结果
✿RETURN✿: 根据工具结果进行回复，需将图片用![](url)渲染出来<|im_end|>
<|im_start|>user
波士顿天气如何？<|im_end|>
<|im_start|>assistant
"""
        logging.info(
            f"expected prompt: \n{expected_prompt}\n-----------------------------------"
        )
        assert prompt == expected_prompt

        messages.append(
            ChatMessage(
                **{
                    "role": RoleEnum.assistant,
                    "content": None,
                    "function_call": {
                        "name": "get_current_weather",
                        "arguments": '{"location": "Boston, MA"}',
                    },
                }
            )
        )

        messages.append(
            ChatMessage(
                **{
                    "role": RoleEnum.function,
                    "name": "get_current_weather",
                    "content": '{"temperature": "22", "unit": "celsius", "description": "Sunny"}',
                }
            )
        )

        request.messages = messages
        ids = chat_renderer.render_chat(request).input_ids
        prompt = tokenizer.decode(ids)
        logging.info(
            f"rendered prompt: \n{prompt}\n-----------------------------------"
        )
        expected_prompt = """<|im_start|>system
你是一个有用的助手

# 工具

## 你拥有如下工具：

### get_current_weather

get_current_weather: Get the current weather in a given location. 输入参数：{"type": "object", "properties": {"location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["location"]} 此工具的输入应为JSON对象。

## 你可以在回复中插入零次、一次或多次以下命令以调用工具：

✿FUNCTION✿: 工具名称，必须是[get_current_weather]之一。
✿ARGS✿: 工具输入
✿RESULT✿: 工具结果
✿RETURN✿: 根据工具结果进行回复，需将图片用![](url)渲染出来<|im_end|>
<|im_start|>user
波士顿天气如何？<|im_end|>
<|im_start|>assistant
✿FUNCTION✿: get_current_weather
✿ARGS✿: {"location": "Boston, MA"}
✿RESULT✿: {"temperature": "22", "unit": "celsius", "description": "Sunny"}
✿RETURN✿"""
        logging.info(
            f"expected prompt: \n{expected_prompt}\n-----------------------------------"
        )
        logging.info(f"actual prompt: \n{prompt}\n-----------------------------------")
        assert prompt == expected_prompt

    def test_qwen_agent_tool(self):
        tokenizer = QWenTokenizer(
            f"{self.test_data_path}/model_test/fake_test/testdata/qwen_7b/tokenizer/qwen.tiktoken"
        )
        render_params = RendererParams(
            model_type="qwen_agent_tool",
            max_seq_len=1024,
            eos_token_id=tokenizer.eos_token_id or 0,
            stop_word_ids_list=[],
        )
        chat_renderer = ChatRendererFactory.get_renderer(tokenizer, render_params)
        logging.info(
            f"------- chat_renderer.get_renderer_info(): {chat_renderer.get_renderer_info()}"
        )
        assert isinstance(chat_renderer, QwenAgentToolRenderer)

        functions = [
            GPTFunctionDefinition(
                **{
                    "name": "get_current_weather",
                    "description": "Get the current weather in a given location.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA",
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                            },
                        },
                        "required": ["location"],
                    },
                }
            )
        ]

        messages = [
            ChatMessage(
                **{
                    "role": RoleEnum.system,
                    "content": "你是一个有用的助手",
                }
            ),
            ChatMessage(
                **{
                    "role": RoleEnum.user,
                    "content": "波士顿天气如何？",
                }
            ),
        ]

        request = ChatCompletionRequest(
            **{
                "messages": messages,
                "tools": [GPTToolDefinition(function=functions[0])],
                "stream": False,
                "extend_fields": {"lang": "zh"},
            }
        )

        ids = chat_renderer.render_chat(request).input_ids
        prompt = tokenizer.decode(ids)
        logging.info(
            f"rendered prompt: \n{prompt}\n-----------------------------------"
        )
        expected_prompt = """<|im_start|>system
你是一个有用的助手

# 工具

## 你拥有如下工具：

### get_current_weather

get_current_weather: Get the current weather in a given location. 输入参数：{"type": "object", "properties": {"location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["location"]} 此工具的输入应为JSON对象。

## 你可以在回复中插入零次、一次或多次以下命令以调用工具：

✿FUNCTION✿: 工具名称，必须是[get_current_weather]之一。
✿ARGS✿: 工具输入
✿RESULT✿: 工具结果
✿RETURN✿: 根据工具结果进行回复，需将图片用![](url)渲染出来<|im_end|>
<|im_start|>user
波士顿天气如何？<|im_end|>
<|im_start|>assistant
"""
        logging.info(
            f"expected prompt: \n{expected_prompt}\n-----------------------------------"
        )
        assert prompt == expected_prompt

        messages.append(
            ChatMessage(
                **{
                    "role": RoleEnum.assistant,
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_current_weather",
                                "arguments": '{"location": "Boston, MA"}',
                            },
                        }
                    ],
                }
            )
        )

        messages.append(
            ChatMessage(
                **{
                    "role": RoleEnum.tool,
                    "content": '{"temperature": "22", "unit": "celsius", "description": "Sunny"}',
                }
            )
        )

        request.messages = messages
        ids = chat_renderer.render_chat(request).input_ids
        prompt = tokenizer.decode(ids)
        logging.info(
            f"rendered prompt: \n{prompt}\n-----------------------------------"
        )
        expected_prompt = """<|im_start|>system
你是一个有用的助手

# 工具

## 你拥有如下工具：

### get_current_weather

get_current_weather: Get the current weather in a given location. 输入参数：{"type": "object", "properties": {"location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["location"]} 此工具的输入应为JSON对象。

## 你可以在回复中插入零次、一次或多次以下命令以调用工具：

✿FUNCTION✿: 工具名称，必须是[get_current_weather]之一。
✿ARGS✿: 工具输入
✿RESULT✿: 工具结果
✿RETURN✿: 根据工具结果进行回复，需将图片用![](url)渲染出来<|im_end|>
<|im_start|>user
波士顿天气如何？<|im_end|>
<|im_start|>assistant
✿FUNCTION✿: get_current_weather
✿ARGS✿: {"location": "Boston, MA"}
✿RESULT✿: {"temperature": "22", "unit": "celsius", "description": "Sunny"}
✿RETURN✿"""
        logging.info(
            f"expected prompt: \n{expected_prompt}\n-----------------------------------"
        )
        logging.info(f"actual prompt: \n{prompt}\n-----------------------------------")
        assert prompt == expected_prompt

    def test_qwen_with_chat_template(self):
        tokenizer = Qwen2Tokenizer.from_pretrained(
            f"{self.test_data_path}/tokenizer_test/testdata/qwen2_tokenizer"
        )
        render_params = RendererParams(
            model_type="qwen_2",
            max_seq_len=1024,
            eos_token_id=tokenizer.eos_token_id or 0,
            stop_word_ids_list=[],
        )
        chat_renderer = ChatRendererFactory.get_renderer(tokenizer, render_params)
        assert isinstance(chat_renderer, QwenRenderer)
        assert (
            chat_renderer.template_chat_renderer.chat_template
            == "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}template ends here"
        )
        messages = [
            ChatMessage(
                **{
                    "role": RoleEnum.system,
                    "content": "你是小助手",
                }
            ),
            ChatMessage(
                **{
                    "role": RoleEnum.user,
                    "content": "介绍一下自己",
                }
            ),
        ]
        request = ChatCompletionRequest(messages=messages)
        ids = chat_renderer.render_chat(request).input_ids
        prompt = tokenizer.decode(ids)
        logging.info(
            f"rendered prompt: \n{prompt}\n-----------------------------------"
        )
        assert (
            prompt
            == """<|im_start|>system
你是小助手<|im_end|>
<|im_start|>user
介绍一下自己<|im_end|>
<|im_start|>assistant
template ends here"""
        )


if __name__ == "__main__":
    main()
