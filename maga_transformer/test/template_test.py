import os
import torch
import logging
from unittest import TestCase, main
from typing import Any
from transformers import AutoTokenizer, PreTrainedTokenizer

from maga_transformer.openai.renderers.qwen_agent_tool_renderer import (
    QwenAgentToolRenderer,
)
from maga_transformer.openai.renderers.qwen_tool_renderer import QwenToolRenderer
from maga_transformer.pipeline.chatapi_format import encode_chatapi
from maga_transformer.models.starcoder import StarcoderTokenizer
from maga_transformer.models.llava import LlavaTokenizer
from maga_transformer.openai.api_datatype import (
    ChatMessage,
    FunctionCall,
    GPTToolDefinition,
    RoleEnum,
    ChatCompletionRequest,
    GPTFunctionDefinition,
    ContentPart,
    ContentPartTypeEnum,
    RendererInfo,
    ToolCall,
)
from maga_transformer.tokenizer.tokenization_qwen import QWenTokenizer
from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer
from maga_transformer.openai.renderer_factory import ChatRendererFactory, RendererParams, \
    CustomChatRenderer, FastChatRenderer, LlamaTemplateRenderer
from maga_transformer.openai.renderers.qwen_renderer import QwenRenderer
from maga_transformer.openai.renderers.qwen_agent_renderer import QwenAgentRenderer


class TemplateTest(TestCase):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.test_data_path = os.path.join(
            os.getcwd(), 'maga_transformer/test'
        )

    def test_qwen(self):
        tokenizer = QWenTokenizer(f"{self.test_data_path}/model_test/fake_test/testdata/qwen_7b/tokenizer/qwen.tiktoken")
        render_params = RendererParams(
            model_type="qwen",
            max_seq_len=1024,
            eos_token_id=tokenizer.eos_token_id or 0,
            stop_word_ids_list=[],
        )
        chat_renderer = ChatRendererFactory.get_renderer(tokenizer, render_params)
        logging.info(f"------- chat_renderer.get_renderer_info(): {chat_renderer.get_renderer_info()}")
        assert (isinstance(chat_renderer, QwenRenderer))

        functions = [
            GPTFunctionDefinition(**{
                "name": "get_current_weather",
                "description": "Get the current weather in a given location.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            })
        ]

        messages = [
            ChatMessage(**{
                "role": RoleEnum.user,
                "content": "波士顿天气如何？",
            }),
        ]

        request = ChatCompletionRequest(**{
            "messages": messages,
            "functions": functions,
            "stream": False,
        })

        ids = chat_renderer.render_chat(request).input_ids
        prompt = tokenizer.decode(ids)
        logging.info(f"rendered prompt: \n{prompt}\n-----------------------------------")
        expected_prompt = \
"""<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Answer the following questions as best you can. You have access to the following APIs:

get_current_weather: Call this tool to interact with the get_current_weather API. What is the get_current_weather API useful for? Get the current weather in a given location. Parameters: {"type": "object", "properties": {"location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["location"]}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [get_current_weather]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: 波士顿天气如何？<|im_end|>
<|im_start|>assistant
"""
        logging.info(f"expected prompt: \n{expected_prompt}\n-----------------------------------")
        assert (prompt == expected_prompt)

        messages.append(
            ChatMessage(**{
                "role": RoleEnum.assistant,
                "content": "我需要调用get_current_weather API来获取天气",
                "function_call": {
                    "name": "get_current_weather",
                    "arguments": '{"location": "Boston, MA"}',
                },
            })
        )

        messages.append(
            ChatMessage(**{
                "role": RoleEnum.function,
                "name": "get_current_weather",
                "content": '{"temperature": "22", "unit": "celsius", "description": "Sunny"}',
            })
        )

        request.messages = messages
        ids = chat_renderer.render_chat(request).input_ids
        prompt = tokenizer.decode(ids)
        logging.info(f"rendered prompt: \n{prompt}\n-----------------------------------")
        expected_prompt = \
"""<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Answer the following questions as best you can. You have access to the following APIs:

get_current_weather: Call this tool to interact with the get_current_weather API. What is the get_current_weather API useful for? Get the current weather in a given location. Parameters: {"type": "object", "properties": {"location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["location"]}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [get_current_weather]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: 波士顿天气如何？<|im_end|>
<|im_start|>assistant
Thought: 我需要调用get_current_weather API来获取天气
Action: get_current_weather
Action Input: {"location": "Boston, MA"}
Observation: {"temperature": "22", "unit": "celsius", "description": "Sunny"}
Thought:"""
        logging.info(f"expected prompt: \n{expected_prompt}\n-----------------------------------")
        logging.info(f"actual prompt: \n{prompt}\n-----------------------------------")
        assert (prompt == expected_prompt)

    def test_qwen_agent(self):
        tokenizer = QWenTokenizer(f"{self.test_data_path}/model_test/fake_test/testdata/qwen_7b/tokenizer/qwen.tiktoken")
        render_params = RendererParams(
            model_type="qwen_agent",
            max_seq_len=1024,
            eos_token_id=tokenizer.eos_token_id or 0,
            stop_word_ids_list=[],
        )
        chat_renderer = ChatRendererFactory.get_renderer(tokenizer, render_params)
        logging.info(f"------- chat_renderer.get_renderer_info(): {chat_renderer.get_renderer_info()}")
        assert (isinstance(chat_renderer, QwenAgentRenderer))

        functions = [
            GPTFunctionDefinition(**{
                "name": "get_current_weather",
                "description": "Get the current weather in a given location.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            })
        ]

        messages = [
            ChatMessage(**{
                "role": RoleEnum.system,
                "content": "你是一个有用的助手",
            }),
            ChatMessage(**{
                "role": RoleEnum.user,
                "content": "波士顿天气如何？",
            }),
        ]

        request = ChatCompletionRequest(**{
            "messages": messages,
            "functions": functions,
            "stream": False,
            "extend_fields":{"lang":"zh"}
        })

        ids = chat_renderer.render_chat(request).input_ids
        prompt = tokenizer.decode(ids)
        logging.info(f"rendered prompt: \n{prompt}\n-----------------------------------")
        expected_prompt = \
"""<|im_start|>system
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
        logging.info(f"expected prompt: \n{expected_prompt}\n-----------------------------------")
        assert (prompt == expected_prompt)

        messages.append(
            ChatMessage(**{
                "role": RoleEnum.assistant,
                "content": None,
                "function_call": {
                    "name": "get_current_weather",
                    "arguments": '{"location": "Boston, MA"}',
                },
            })
        )

        messages.append(
            ChatMessage(**{
                "role": RoleEnum.function,
                "name": "get_current_weather",
                "content": '{"temperature": "22", "unit": "celsius", "description": "Sunny"}',
            })
        )

        request.messages = messages
        ids = chat_renderer.render_chat(request).input_ids
        prompt = tokenizer.decode(ids)
        logging.info(f"rendered prompt: \n{prompt}\n-----------------------------------")
        expected_prompt = \
"""<|im_start|>system
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
        logging.info(f"expected prompt: \n{expected_prompt}\n-----------------------------------")
        logging.info(f"actual prompt: \n{prompt}\n-----------------------------------")
        assert (prompt == expected_prompt)

    def test_qwen_tool(self):
        logging.info("begin test_qwen_tool")
        tokenizer = QWenTokenizer(
            f"{self.test_data_path}/model_test/fake_test/testdata/qwen_7b/tokenizer/qwen.tiktoken"
        )
        render_params = RendererParams(
            model_type="qwen_tool",
            max_seq_len=1024,
            eos_token_id=tokenizer.eos_token_id or 0,
            stop_word_ids_list=[],
        )
        chat_renderer = ChatRendererFactory.get_renderer(tokenizer, render_params)
        logging.info(
            f"------- chat_renderer.get_renderer_info(): {chat_renderer.get_renderer_info()}"
        )
        assert isinstance(chat_renderer, QwenToolRenderer)

        # 定义两个工具
        tools = [
            GPTToolDefinition(
                type="function",
                function=GPTFunctionDefinition(
                    **{
                        "name": "get_current_temperature",
                        "description": "Get current temperature at a location.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": 'The location to get the temperature for, in the format "City, State, Country".',
                                },
                                "unit": {
                                    "type": "string",
                                    "enum": ["celsius", "fahrenheit"],
                                    "description": 'The unit to return the temperature in. Defaults to "celsius".',
                                },
                            },
                            "required": ["location"],
                        },
                    }
                ),
            ),
            GPTToolDefinition(
                type="function",
                function=GPTFunctionDefinition(
                    **{
                        "name": "get_temperature_date",
                        "description": "Get temperature at a location and date.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": 'The location to get the temperature for, in the format "City, State, Country".',
                                },
                                "date": {
                                    "type": "string",
                                    "description": 'The date to get the temperature for, in the format "Year-Month-Day".',
                                },
                                "unit": {
                                    "type": "string",
                                    "enum": ["celsius", "fahrenheit"],
                                    "description": 'The unit to return the temperature in. Defaults to "celsius".',
                                },
                            },
                            "required": ["location", "date"],
                        },
                    }
                ),
            ),
        ]

        # 初始对话消息
        messages = [
            ChatMessage(
                **{
                    "role": RoleEnum.system,
                    "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\nCurrent Date: 2024-09-30",
                }
            ),
            ChatMessage(
                **{
                    "role": RoleEnum.user,
                    "content": "What's the temperature in San Francisco now? How about tomorrow?",
                }
            ),
        ]

        request = ChatCompletionRequest(
            **{
                "messages": messages,
                "tools": tools,
                "stream": False,
            }
        )

        ids = chat_renderer.render_chat(request).input_ids
        prompt = tokenizer.decode(ids)
        logging.info(
            f"rendered prompt: \n{prompt}\n-----------------------------------"
        )

        # 第一次断言: 检查初始提示
        expected_prompt = """<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.

Current Date: 2024-09-30

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "get_current_temperature", "description": "Get current temperature at a location.", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "The location to get the temperature for, in the format \\"City, State, Country\\"."}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "The unit to return the temperature in. Defaults to \\"celsius\\"."}}, "required": ["location"]}}}
{"type": "function", "function": {"name": "get_temperature_date", "description": "Get temperature at a location and date.", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "The location to get the temperature for, in the format \\"City, State, Country\\"."}, "date": {"type": "string", "description": "The date to get the temperature for, in the format \\"Year-Month-Day\\"."}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "The unit to return the temperature in. Defaults to \\"celsius\\"."}}, "required": ["location", "date"]}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call><|im_end|>
<|im_start|>user
What's the temperature in San Francisco now? How about tomorrow?<|im_end|>
<|im_start|>assistant
"""
        logging.info(
            f"expected prompt: \n{expected_prompt}\n-----------------------------------"
        )
        assert prompt == expected_prompt

        # 添加助手的工具调用
        messages.append(
            ChatMessage(
                **{
                    "role": RoleEnum.assistant,
                    "content": None,
                    "tool_calls": [
                        ToolCall(
                            id="call_1",
                            type="function",
                            index=0,
                            function=FunctionCall(
                                name="get_current_temperature",
                                arguments='{"location": "San Francisco, CA, USA"}',
                            ),
                        ),
                        ToolCall(
                            id="call_2",
                            type="function",
                            index=1,
                            function=FunctionCall(
                                name="get_temperature_date",
                                arguments='{"location": "San Francisco, CA, USA", "date": "2024-10-01"}',
                            ),
                        ),
                    ],
                }
            )
        )

        # 添加工具响应
        messages.append(
            ChatMessage(
                **{
                    "role": RoleEnum.tool,
                    "content": '{"temperature": 26.1, "location": "San Francisco, CA, USA", "unit": "celsius"}',
                    "tool_call_id": "call_1",
                }
            )
        )
        messages.append(
            ChatMessage(
                **{
                    "role": RoleEnum.tool,
                    "content": '{"temperature": 25.9, "location": "San Francisco, CA, USA", "date": "2024-10-01", "unit": "celsius"}',
                    "tool_call_id": "call_2",
                }
            )
        )

        # 添加最终助手响应
        messages.append(
            ChatMessage(
                **{
                    "role": RoleEnum.assistant,
                    "content": "The current temperature in San Francisco is approximately 26.1°C. Tomorrow, on October 1, 2024, the temperature is expected to be around 25.9°C.",
                }
            )
        )

        request.messages = messages
        ids = chat_renderer.render_chat(request).input_ids
        prompt = tokenizer.decode(ids)
        logging.info(f"final prompt: \n{prompt}\n-----------------------------------")

        # 最终断言: 检查完整对话
        expected_final_prompt = """<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.

Current Date: 2024-09-30

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "get_current_temperature", "description": "Get current temperature at a location.", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "The location to get the temperature for, in the format \\"City, State, Country\\"."}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "The unit to return the temperature in. Defaults to \\"celsius\\"."}}, "required": ["location"]}}}
{"type": "function", "function": {"name": "get_temperature_date", "description": "Get temperature at a location and date.", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "The location to get the temperature for, in the format \\"City, State, Country\\"."}, "date": {"type": "string", "description": "The date to get the temperature for, in the format \\"Year-Month-Day\\"."}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "The unit to return the temperature in. Defaults to \\"celsius\\"."}}, "required": ["location", "date"]}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call><|im_end|>
<|im_start|>user
What's the temperature in San Francisco now? How about tomorrow?<|im_end|>
<|im_start|>assistant
<tool_call>
{"name": "get_current_temperature", "arguments": {"location": "San Francisco, CA, USA"}}
</tool_call>
<tool_call>
{"name": "get_temperature_date", "arguments": {"location": "San Francisco, CA, USA", "date": "2024-10-01"}}
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
{"temperature": 26.1, "location": "San Francisco, CA, USA", "unit": "celsius"}
</tool_response>
<tool_response>
{"temperature": 25.9, "location": "San Francisco, CA, USA", "date": "2024-10-01", "unit": "celsius"}
</tool_response><|im_end|>
<|im_start|>assistant
The current temperature in San Francisco is approximately 26.1°C. Tomorrow, on October 1, 2024, the temperature is expected to be around 25.9°C.<|im_end|>
"""
        logging.info(
            f"expected final prompt: \n{expected_final_prompt}\n-----------------------------------"
        )
        assert prompt == expected_final_prompt

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

    def test_qwen_vl(self):
        tokenizer = AutoTokenizer.from_pretrained(f"{self.test_data_path}/model_test/fake_test/testdata/qwen_vl/tokenizer/", trust_remote_code=True)
        assert(isinstance(tokenizer, PreTrainedTokenizer))
        render_params = RendererParams(
            model_type="qwen_vl",
            max_seq_len=1024,
            eos_token_id=tokenizer.eos_token_id or 0,
            stop_word_ids_list=[],
        )
        chat_renderer = ChatRendererFactory.get_renderer(tokenizer, render_params)

        test_messages = [ChatMessage(**{
            "role": RoleEnum.user,
            "content": [
                ContentPart(**{
                    "type": ContentPartTypeEnum.image_url,
                    "image_url": {
                        "url": "https://modelscope.cn/api/v1/models/damo/speech_eres2net_sv_zh-cn_16k-common/repo?Revision=master&FilePath=images/ERes2Net_architecture.png"
                    }
                }),
                ContentPart(**{
                    "type": ContentPartTypeEnum.text,
                    "text": "这是什么"
                }),
            ],
        })]
        request = ChatCompletionRequest(**{
            "messages": test_messages,
            "stream": False,
        })
        ids = chat_renderer.render_chat(request).input_ids
        expected_ids = [151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 24669, 220, 16, 25, 220, 151857, 104, 116, 116, 112, 115, 58, 47, 47, 109, 111, 100, 101, 108, 115, 99, 111, 112, 101, 46, 99, 110, 47, 97, 112, 105, 47, 118, 49, 47, 109, 111, 100, 101, 108, 115, 47, 100, 97, 109, 111, 47, 115, 112, 101, 101, 99, 104, 95, 101, 114, 101, 115, 50, 110, 101, 116, 95, 115, 118, 95, 122, 104, 45, 99, 110, 95, 49, 54, 107, 45, 99, 111, 109, 109, 111, 110, 47, 114, 101, 112, 111, 63, 82, 101, 118, 105, 115, 105, 111, 110, 61, 109, 97, 115, 116, 101, 114, 38, 70, 105, 108, 101, 80, 97, 116, 104, 61, 105, 109, 97, 103, 101, 115, 47, 69, 82, 101, 115, 50, 78, 101, 116, 95, 97, 114, 99, 104, 105, 116, 101, 99, 116, 117, 114, 101, 46, 112, 110, 103, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151858, 198, 100346, 99245, 151645, 198, 151644, 77091, 198]
        assert ids == expected_ids, f"ids: {ids} vs expected ids {expected_ids}"

        request.messages.append(ChatMessage(**{
            "role": RoleEnum.assistant,
            "content": "这是一个深度神经网络模型的结构图。从图中可以看出，这个模型包括全局特征融合和局部特征融合两部分。全局特征融合部分使用了ERes2Net block，而局部特征融合部分则使用了多个ERes2Net block和AFF block。此外，模型的最后一层还使用了一个1×1的卷积层。图中还标注了各个模块之间的连接关系，包括输入、输出以及与其他模块的连接。",
        }))
        request.messages.append(ChatMessage(**{
            "role": RoleEnum.user,
            "content": "输出 embedding 层的检测框",
        }))

        ids = chat_renderer.render_chat(request).input_ids
        expected_ids = [151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 24669, 220, 16, 25, 220, 151857, 104, 116, 116, 112, 115, 58, 47, 47, 109, 111, 100, 101, 108, 115, 99, 111, 112, 101, 46, 99, 110, 47, 97, 112, 105, 47, 118, 49, 47, 109, 111, 100, 101, 108, 115, 47, 100, 97, 109, 111, 47, 115, 112, 101, 101, 99, 104, 95, 101, 114, 101, 115, 50, 110, 101, 116, 95, 115, 118, 95, 122, 104, 45, 99, 110, 95, 49, 54, 107, 45, 99, 111, 109, 109, 111, 110, 47, 114, 101, 112, 111, 63, 82, 101, 118, 105, 115, 105, 111, 110, 61, 109, 97, 115, 116, 101, 114, 38, 70, 105, 108, 101, 80, 97, 116, 104, 61, 105, 109, 97, 103, 101, 115, 47, 69, 82, 101, 115, 50, 78, 101, 116, 95, 97, 114, 99, 104, 105, 116, 101, 99, 116, 117, 114, 101, 46, 112, 110, 103, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151859, 151858, 198, 100346, 99245, 151645, 198, 151644, 77091, 198, 105464, 102217, 102398, 71356, 104949, 9370, 100166, 28029, 1773, 45181, 28029, 15946, 107800, 3837, 99487, 104949, 100630, 109894, 104363, 101164, 33108, 106304, 104363, 101164, 77540, 99659, 1773, 109894, 104363, 101164, 99659, 37029, 34187, 640, 288, 17, 6954, 2504, 3837, 68536, 106304, 104363, 101164, 99659, 46448, 37029, 34187, 101213, 640, 288, 17, 6954, 2504, 33108, 48045, 2504, 1773, 104043, 3837, 104949, 114641, 99371, 97706, 37029, 104059, 16, 17568, 16, 9370, 100199, 99263, 99371, 1773, 28029, 15946, 97706, 111066, 34187, 101284, 106393, 104186, 64064, 100145, 3837, 100630, 31196, 5373, 66017, 101034, 106961, 106393, 9370, 64064, 1773, 151645, 198, 151644, 872, 198, 66017, 39088, 79621, 224, 9370, 101978, 101540, 151645, 198, 151644, 77091, 198]
        assert (ids == expected_ids)

    def test_llava(self):
        os.environ["CHECKPOINT_PATH"] = "llava-v1.5"
        tokenizer = LlavaTokenizer(
            tokenzier_path = f"{self.test_data_path}/model_test/fake_test/testdata/llava/tokenizer/",
            mm_use_im_patch_token = False,
            mm_use_im_start_end = False,
            special_token_ids = {'ignore_token_index': -100, 'image_token_index': -200},
            special_tokens = {
                'default_mm_token': '<image>',
                'default_im_start_token': '<im_start>',
                'default_im_end_token': '<im_end>'
            }
        )

        render_params = RendererParams(
            model_type="llava",
            max_seq_len=1024,
            eos_token_id=tokenizer.tokenizer.eos_token_id or 0,
            stop_word_ids_list=[],
        )
        chat_renderer = ChatRendererFactory.get_renderer(tokenizer, render_params)
        test_messages = [ChatMessage(**{
            "role": RoleEnum.user,
            "content": [
                ContentPart(**{
                    "type": ContentPartTypeEnum.image_url,
                    "image_url": {
                        "url": "https://modelscope.cn/api/v1/models/damo/speech_eres2net_sv_zh-cn_16k-common/repo?Revision=master&FilePath=images/ERes2Net_architecture.png"
                    }
                }),
                ContentPart(**{
                    "type": ContentPartTypeEnum.text,
                    "text": "这是什么"
                }),
            ],
        })]
        test_messages.append(ChatMessage(**{
            "role": RoleEnum.assistant,
            "content": [
                ContentPart(**{
                    "type": ContentPartTypeEnum.text,
                    "text": "这是图"
                }),
            ],
        }))
        request = ChatCompletionRequest(**{
            "messages": test_messages,
            "stream": False,
        })
        ids = chat_renderer.render_chat(request).input_ids
        expected_ids = [1, 319, 13563, 1546, 263, 12758, 5199, 322, 385, 23116, 21082, 20255, 29889, 450, 20255, 4076, 8444, 29892, 13173, 29892, 322, 1248, 568, 6089, 304, 278, 5199, 29915, 29879, 5155, 29889, 3148, 1001, 29901, 29871, -200, 29871, 13, 30810, 30392, 231, 190, 131, 31882, 319, 1799, 9047, 13566, 29901, 29871, 30810, 30392, 30861, 2, 22933, 9047, 13566, 29901]
        assert ids == expected_ids, f"ids: {ids} vs expected ids {expected_ids}"

    def test_baichuan2(self):
        tokenizer = AutoTokenizer.from_pretrained(
            f"{self.test_data_path}/model_test/fake_test/testdata/baichuan/tokenizer/", trust_remote_code=True
        )
        render_params = RendererParams(
            model_type="baichuan2",
            max_seq_len=1024,
            eos_token_id=tokenizer.eos_token_id or 0,
            stop_word_ids_list=[],
        )
        assert (isinstance(tokenizer, PreTrainedTokenizer))
        chat_renderer = ChatRendererFactory.get_renderer(tokenizer, render_params)

        messages = [
            ChatMessage(**{
                "role": RoleEnum.user,
                "content": "你是谁？",
            }),
            ChatMessage(**{
                "role": RoleEnum.assistant,
                "content": "我是百川大模型",
            }),
            ChatMessage(**{
                "role": RoleEnum.assistant,
                "content": "展开讲讲",
            }),
        ]
        request = ChatCompletionRequest(messages=messages)

        ids = chat_renderer.render_chat(request).input_ids
        prompt = tokenizer.decode(ids)
        expected_ids = [195, 92067, 68, 196, 6461, 70335, 92366, 9528, 195, 8277, 57056, 196]
        assert ids == expected_ids, f"ids: {ids} vs expected ids {expected_ids}"

    def test_imported_template(self):
        tokenizer = AutoTokenizer.from_pretrained(
            f"{self.test_data_path}/model_test/fake_test/testdata/baichuan/tokenizer/", trust_remote_code=True
        )
        render_params = RendererParams(
            model_type="mixtral",
            max_seq_len=1024,
            eos_token_id=tokenizer.eos_token_id or 0,
            stop_word_ids_list=[],
        )
        expected_renderer_info = RendererInfo(
            class_name = "LlamaTemplateRenderer",
            renderer_model_type = "mixtral",
            extra_stop_word_ids_list = [[2]],
            extra_stop_words_list = ['</s>'],
            template = "Template(prefix=['<s>'], prompt=['[INST]{{query}}[/INST]'], system='', sep=['</s>'], stop_words=['</s>'], use_history=True, efficient_eos=False, replace_eos=False)"
        )
        assert (isinstance(tokenizer, PreTrainedTokenizer))
        chat_renderer = ChatRendererFactory.get_renderer(tokenizer, render_params)
        self.assertEqual(chat_renderer.get_renderer_info(), expected_renderer_info)

        expected_renderer_info = RendererInfo(
            class_name = "FastChatRenderer",
            renderer_model_type = "mistral",
            extra_stop_word_ids_list = [],
            extra_stop_words_list = [],
            template = "Conversation(name='mistral', system_template='[INST] {system_message}\\n', system_message='', roles=('[INST]', '[/INST]'), messages=[], offset=0, sep_style=<SeparatorStyle.LLAMA2: 7>, sep=' ', sep2='</s>', stop_str=None, stop_token_ids=None)"
        )
        render_params.model_type = "mistral"
        chat_renderer = ChatRendererFactory.get_renderer(tokenizer, render_params)
        self.assertEqual(chat_renderer.get_renderer_info(), expected_renderer_info)

    def test_qwen_with_chat_template(self):
        tokenizer = Qwen2Tokenizer.from_pretrained(f"{self.test_data_path}/tokenizer_test/testdata/qwen2_tokenizer")
        render_params = RendererParams(
            model_type="qwen_2",
            max_seq_len=1024,
            eos_token_id=tokenizer.eos_token_id or 0,
            stop_word_ids_list=[],
        )
        chat_renderer = ChatRendererFactory.get_renderer(tokenizer, render_params)
        assert(isinstance(chat_renderer, QwenRenderer))
        assert(chat_renderer.template_chat_renderer.chat_template == "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}template ends here")
        messages = [
            ChatMessage(**{
                "role": RoleEnum.system,
                "content": "你是小助手",
            }),
            ChatMessage(**{
                "role": RoleEnum.user,
                "content": "介绍一下自己",
            }),
        ]
        request = ChatCompletionRequest(messages=messages)
        ids = chat_renderer.render_chat(request).input_ids
        prompt = tokenizer.decode(ids)
        logging.info(f"rendered prompt: \n{prompt}\n-----------------------------------")
        assert(prompt == """<|im_start|>system
你是小助手<|im_end|>
<|im_start|>user
介绍一下自己<|im_end|>
<|im_start|>assistant
template ends here""")

    def test_qwen_default_system(self):
        tokenizer = Qwen2Tokenizer.from_pretrained(f"{self.test_data_path}/tokenizer_test/testdata/qwen2_tokenizer")
        tokenizer.chat_template = None
        tokenizer.im_start_id = tokenizer.encode('<|im_start|>')[0]
        tokenizer.im_end_id = tokenizer.encode('<|im_end|>')[0]
        render_params = RendererParams(
            model_type="qwen_2",
            max_seq_len=1024,
            eos_token_id=tokenizer.eos_token_id or 0,
            stop_word_ids_list=[],
        )
        chat_renderer = ChatRendererFactory.get_renderer(tokenizer, render_params)
        assert(isinstance(chat_renderer, QwenRenderer))
        assert(chat_renderer.template_chat_renderer == None)
        messages = [
            ChatMessage(**{
                "role": RoleEnum.system,
                "content": "你是小助手",
            }),
            ChatMessage(**{
                "role": RoleEnum.user,
                "content": "介绍一下自己",
            }),
        ]
        request = ChatCompletionRequest(messages=messages)
        ids = chat_renderer.render_chat(request).input_ids
        prompt = tokenizer.decode(ids)
        logging.info(f"rendered prompt: \n{prompt}\n-----------------------------------")
        assert(prompt == """<|im_start|>system
你是小助手<|im_end|>
<|im_start|>user
介绍一下自己<|im_end|>
<|im_start|>assistant
""")

    def test_multi_templates(self):
        tokenizer = AutoTokenizer.from_pretrained(
            f"{self.test_data_path}/model_test/fake_test/testdata/cohere/tokenizer/", trust_remote_code=True
        )
        render_params = RendererParams(
            model_type="cohere",
            max_seq_len=1024,
            eos_token_id=tokenizer.eos_token_id or 0,
            stop_word_ids_list=[],
        )
        chat_renderer = ChatRendererFactory.get_renderer(tokenizer, render_params)

        messages = [
            ChatMessage(**{
                "role": RoleEnum.user,
                "content": "你是谁？",
            }),
        ]
        request = ChatCompletionRequest(messages=messages)
        prompt = tokenizer.decode(chat_renderer.render_chat(request).input_ids)
        self.assertEqual(prompt, "<BOS_TOKEN><BOS_TOKEN><|START_OF_TURN_TOKEN|><|USER_TOKEN|>你是谁？<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>")

        request.template_key = "rag"
        self.maxDiff = None
        prompt = tokenizer.decode(chat_renderer.render_chat(request).input_ids)
        self.assertEqual(prompt,
"""<BOS_TOKEN><BOS_TOKEN><|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|># Safety Preamble
The instructions in this section override those in the task description and style guide sections. Don't answer questions that are harmful or immoral.

# System Preamble
## Basic Rules
You are a powerful conversational AI trained by Cohere to help people. You are augmented by a number of tools, and your job is to use and consume the output of these tools to best help the user. You will see a conversation history between yourself and a user, ending with an utterance from the user. You will then see a specific instruction instructing you what kind of response to generate. When you answer the user's requests, you cite your sources in your answers, according to those instructions.

# User Preamble
## Task and Context
You help people answer their questions and other requests interactively. You will be asked a very wide array of requests on all kinds of topics. You will be equipped with a wide range of search engines or similar tools to help you, which you use to research your answer. You should focus on serving the user's needs as best you can, which will be wide-ranging.

## Style Guide
Unless the user asks for a different style of answer, you should answer in full sentences, using proper grammar and spelling.<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|USER_TOKEN|>你是谁？<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|><results></results><|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>Carefully perform the following instructions, in order, starting each with a new line.
Firstly, Decide which of the retrieved documents are relevant to the user's last input by writing 'Relevant Documents:' followed by comma-separated list of document numbers. If none are relevant, you should instead write 'None'.
Secondly, Decide which of the retrieved documents contain facts that should be cited in a good answer to the user's last input by writing 'Cited Documents:' followed a comma-separated list of document numbers. If you dont want to cite any of them, you should instead write 'None'.
Finally, Write 'Grounded answer:' followed by a response to the user's last input in high quality natural english. Use the symbols <co: doc> and </co: doc> to indicate when a fact comes from a document in the search result, e.g <co: 0>my fact</co: 0> for a fact from document 0.<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>""")
        request.template_key = None
        request.user_template = "{{bos_token}}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ ' [INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' ' + message['content'] + ' ' + eos_token}}{% endif %}{% endfor %}{{123}}"
        prompt = tokenizer.decode(chat_renderer.render_chat(request).input_ids)
        self.assertEqual(prompt, "<BOS_TOKEN><BOS_TOKEN> [INST] 你是谁？ [/INST]123")

        print(prompt)

if __name__ == '__main__':
    main()
