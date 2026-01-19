import logging

logging.basicConfig(level=logging.INFO)
import os
from abc import ABC, abstractmethod
from typing import Any, List
from unittest import TestCase, main

from transformers import AutoTokenizer
from typing_extensions import override

from rtp_llm.config.py_config_modules import GenerateEnvConfig, RenderConfig
from rtp_llm.openai.api_datatype import (
    ChatCompletionRequest,
    ChatMessage,
    GPTFunctionDefinition,
    GPTToolDefinition,
    RoleEnum,
)
from rtp_llm.openai.renderer_factory import ChatRendererFactory, RendererParams
from rtp_llm.openai.renderers.qwen_agent_renderer import QwenAgentRenderer
from rtp_llm.openai.renderers.qwen_agent_tool_renderer import QwenAgentToolRenderer
from rtp_llm.openai.renderers.qwen_renderer import QwenRenderer
from rtp_llm.tokenizer_factory.tokenizers import (
    BaseTokenizer,
    QWenTokenizer,
    QWenV2Tokenizer,
)


class BaseRendererTestMixin(ABC):
    """渲染器测试基类混入"""

    @abstractmethod
    def get_tokenizer(self):
        """获取tokenizer实例"""

    @abstractmethod
    def get_render_params(self) -> RendererParams:
        """获取渲染参数"""

    @abstractmethod
    def get_step1_expected_renderer_prompt(self) -> str:
        """获取步骤1的期望prompt"""

    @abstractmethod
    def get_step2_expected_renderer_prompt(self) -> str:
        """获取步骤2的期望prompt"""

    @abstractmethod
    def get_step3_expected_renderer_prompt(self) -> str:
        """获取步骤3的期望prompt"""

    def get_tools(self) -> List[GPTToolDefinition]:
        """获取工具定义 - 通用实现"""
        return [
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

    def get_initial_messages(self) -> List[ChatMessage]:
        """获取初始消息 - 通用实现"""
        return [
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

    def get_messages_with_tool_response(self) -> List[ChatMessage]:
        """获取包含工具响应的消息 - 通用实现"""
        messages = self.get_initial_messages()
        messages.extend(
            [
                ChatMessage(
                    **{
                        "role": RoleEnum.assistant,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "get_current_temperature",
                                    "arguments": '{"location": "San Francisco, CA, USA"}',
                                },
                            },
                            {
                                "id": "call_2",
                                "type": "function",
                                "function": {
                                    "name": "get_temperature_date",
                                    "arguments": '{"location": "San Francisco, CA, USA", "date": "2024-10-01"}',
                                },
                            },
                        ],
                    }
                ),
                ChatMessage(
                    **{
                        "role": RoleEnum.tool,
                        "content": '{"temperature": 26.1, "location": "San Francisco, CA, USA", "unit": "celsius"}',
                        "tool_call_id": "call_1",
                    }
                ),
                ChatMessage(
                    **{
                        "role": RoleEnum.tool,
                        "content": '{"temperature": 25.9, "location": "San Francisco, CA, USA", "date": "2024-10-01", "unit": "celsius"}',
                        "tool_call_id": "call_2",
                    }
                ),
            ]
        )
        return messages

    def get_messages_with_follow_up(self) -> List[ChatMessage]:
        """获取包含后续对话的消息 - 通用实现"""
        messages = self.get_messages_with_tool_response()
        messages.extend(
            [
                ChatMessage(
                    **{
                        "role": RoleEnum.assistant,
                        "content": "San Francisco今天26.1度, 明天25.9度",
                    }
                ),
                ChatMessage(
                    **{
                        "role": RoleEnum.user,
                        "content": "那北京明天温度如何",
                    }
                ),
            ]
        )
        return messages

    def run_renderer_test(self):
        """运行渲染器测试的通用方法"""
        tokenizer = self.get_tokenizer()
        render_params = self.get_render_params()
        generate_env_config = GenerateEnvConfig()
        render_config = RenderConfig()
        chat_renderer = ChatRendererFactory.get_renderer(
            tokenizer,
            render_params,
            generate_env_config=generate_env_config,
            render_config=render_config,
        )
        tools = self.get_tools()

        # 测试步骤1：初始消息
        initial_messages = self.get_initial_messages()
        request = ChatCompletionRequest(messages=initial_messages, tools=tools)
        renderer_prompt = chat_renderer.render_chat(request).rendered_prompt
        expected_prompt = self.get_step1_expected_renderer_prompt()
        logging.info(
            f"Step 1 prompt: \n{renderer_prompt}\n-----------------------------------"
        )
        assert renderer_prompt == expected_prompt, "Step 1 prompt mismatch"

        # 测试步骤2：包含工具响应的消息
        tool_response_messages = self.get_messages_with_tool_response()
        request = ChatCompletionRequest(messages=tool_response_messages, tools=tools)
        renderer_prompt = chat_renderer.render_chat(request).rendered_prompt
        expected_prompt = self.get_step2_expected_renderer_prompt()

        logging.info(
            f"Step 2 prompt: \n{renderer_prompt}\n-----------------------------------"
        )
        assert renderer_prompt == expected_prompt, "Step 2 prompt mismatch"

        # 测试步骤3：后续对话消息
        follow_up_messages = self.get_messages_with_follow_up()
        request = ChatCompletionRequest(messages=follow_up_messages, tools=tools)
        renderer_prompt = chat_renderer.render_chat(request).rendered_prompt
        expected_prompt = self.get_step3_expected_renderer_prompt()

        logging.info(
            f"Step 3 prompt: \n{renderer_prompt}\n-----------------------------------"
        )
        assert renderer_prompt == expected_prompt, "Step 3 prompt mismatch"


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
        generate_env_config = GenerateEnvConfig()
        render_config = RenderConfig()
        chat_renderer = ChatRendererFactory.get_renderer(
            tokenizer,
            render_params,
            generate_env_config=generate_env_config,
            render_config=render_config,
        )

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
        generate_env_config = GenerateEnvConfig()
        render_config = RenderConfig()
        chat_renderer = ChatRendererFactory.get_renderer(
            tokenizer,
            render_params,
            generate_env_config=generate_env_config,
            render_config=render_config,
        )

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
        tokenizer = QWenV2Tokenizer(
            f"{self.test_data_path}/tokenizer_test/testdata/qwen2_tokenizer"
        )
        render_params = RendererParams(
            model_type="qwen_2",
            max_seq_len=1024,
            eos_token_id=tokenizer.eos_token_id or 0,
            stop_word_ids_list=[],
        )
        generate_env_config = GenerateEnvConfig()
        render_config = RenderConfig()
        chat_renderer = ChatRendererFactory.get_renderer(
            tokenizer,
            render_params,
            generate_env_config=generate_env_config,
            render_config=render_config,
        )

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

    # Qwen3渲染器测试实现
    class Qwen3RendererTestImpl(BaseRendererTestMixin):
        def __init__(self, test_instance):
            self.test_instance = test_instance

        def get_tokenizer(self):
            return BaseTokenizer(
                f"{self.test_instance.test_data_path}/model_test/fake_test/testdata/qwen3_30b/tokenizer/",
            )

        def get_render_params(self) -> RendererParams:
            tokenizer = self.get_tokenizer()
            return RendererParams(
                model_type="qwen_3",
                max_seq_len=1024,
                eos_token_id=tokenizer.eos_token_id or 0,
                stop_word_ids_list=[],
            )

        def get_step1_expected_renderer_prompt(self) -> str:
            return """<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\nCurrent Date: 2024-09-30\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{"type": "function", "function": {"name": "get_current_temperature", "description": "Get current temperature at a location.", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "The location to get the temperature for, in the format \\"City, State, Country\\"."}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "The unit to return the temperature in. Defaults to \\"celsius\\"."}}, "required": ["location"]}}}\n{"type": "function", "function": {"name": "get_temperature_date", "description": "Get temperature at a location and date.", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "The location to get the temperature for, in the format \\"City, State, Country\\"."}, "date": {"type": "string", "description": "The date to get the temperature for, in the format \\"Year-Month-Day\\"."}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "The unit to return the temperature in. Defaults to \\"celsius\\"."}}, "required": ["location", "date"]}}}\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call><|im_end|>\n<|im_start|>user\nWhat\'s the temperature in San Francisco now? How about tomorrow?<|im_end|>\n<|im_start|>assistant\n"""

        def get_step2_expected_renderer_prompt(self) -> str:
            return """<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\nCurrent Date: 2024-09-30\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{"type": "function", "function": {"name": "get_current_temperature", "description": "Get current temperature at a location.", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "The location to get the temperature for, in the format \\"City, State, Country\\"."}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "The unit to return the temperature in. Defaults to \\"celsius\\"."}}, "required": ["location"]}}}\n{"type": "function", "function": {"name": "get_temperature_date", "description": "Get temperature at a location and date.", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "The location to get the temperature for, in the format \\"City, State, Country\\"."}, "date": {"type": "string", "description": "The date to get the temperature for, in the format \\"Year-Month-Day\\"."}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "The unit to return the temperature in. Defaults to \\"celsius\\"."}}, "required": ["location", "date"]}}}\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call><|im_end|>\n<|im_start|>user\nWhat\'s the temperature in San Francisco now? How about tomorrow?<|im_end|>\n<|im_start|>assistant\n<tool_call>\n{"name": "get_current_temperature", "arguments": {"location": "San Francisco, CA, USA"}}\n</tool_call>\n<tool_call>\n{"name": "get_temperature_date", "arguments": {"location": "San Francisco, CA, USA", "date": "2024-10-01"}}\n</tool_call><|im_end|>\n<|im_start|>user\n<tool_response>\n{"temperature": 26.1, "location": "San Francisco, CA, USA", "unit": "celsius"}\n</tool_response>\n<tool_response>\n{"temperature": 25.9, "location": "San Francisco, CA, USA", "date": "2024-10-01", "unit": "celsius"}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"""

        def get_step3_expected_renderer_prompt(self) -> str:
            return """<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\nCurrent Date: 2024-09-30\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{"type": "function", "function": {"name": "get_current_temperature", "description": "Get current temperature at a location.", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "The location to get the temperature for, in the format \\"City, State, Country\\"."}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "The unit to return the temperature in. Defaults to \\"celsius\\"."}}, "required": ["location"]}}}\n{"type": "function", "function": {"name": "get_temperature_date", "description": "Get temperature at a location and date.", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "The location to get the temperature for, in the format \\"City, State, Country\\"."}, "date": {"type": "string", "description": "The date to get the temperature for, in the format \\"Year-Month-Day\\"."}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "The unit to return the temperature in. Defaults to \\"celsius\\"."}}, "required": ["location", "date"]}}}\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call><|im_end|>\n<|im_start|>user\nWhat\'s the temperature in San Francisco now? How about tomorrow?<|im_end|>\n<|im_start|>assistant\n<tool_call>\n{"name": "get_current_temperature", "arguments": {"location": "San Francisco, CA, USA"}}\n</tool_call>\n<tool_call>\n{"name": "get_temperature_date", "arguments": {"location": "San Francisco, CA, USA", "date": "2024-10-01"}}\n</tool_call><|im_end|>\n<|im_start|>user\n<tool_response>\n{"temperature": 26.1, "location": "San Francisco, CA, USA", "unit": "celsius"}\n</tool_response>\n<tool_response>\n{"temperature": 25.9, "location": "San Francisco, CA, USA", "date": "2024-10-01", "unit": "celsius"}\n</tool_response><|im_end|>\n<|im_start|>assistant\nSan Francisco今天26.1度, 明天25.9度<|im_end|>\n<|im_start|>user\n那北京明天温度如何<|im_end|>\n<|im_start|>assistant\n"""

    # Qwen3 Thinking渲染器测试实现
    class Qwen3ThinkingRendererTestImpl(BaseRendererTestMixin):
        def __init__(self, test_instance):
            self.test_instance = test_instance

        def get_tokenizer(self):
            return BaseTokenizer(
                f"{self.test_instance.test_data_path}/model_test/fake_test/testdata/qwen3_30b_thinking_0527/tokenizer/",
            )

        def get_render_params(self) -> RendererParams:
            tokenizer = self.get_tokenizer()
            return RendererParams(
                model_type="qwen_3",
                max_seq_len=1024,
                eos_token_id=tokenizer.eos_token_id or 0,
                stop_word_ids_list=[],
            )

        def get_step1_expected_renderer_prompt(self) -> str:
            return """<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\nCurrent Date: 2024-09-30\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{"type": "function", "function": {"name": "get_current_temperature", "description": "Get current temperature at a location.", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "The location to get the temperature for, in the format \\"City, State, Country\\"."}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "The unit to return the temperature in. Defaults to \\"celsius\\"."}}, "required": ["location"]}}}\n{"type": "function", "function": {"name": "get_temperature_date", "description": "Get temperature at a location and date.", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "The location to get the temperature for, in the format \\"City, State, Country\\"."}, "date": {"type": "string", "description": "The date to get the temperature for, in the format \\"Year-Month-Day\\"."}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "The unit to return the temperature in. Defaults to \\"celsius\\"."}}, "required": ["location", "date"]}}}\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call><|im_end|>\n<|im_start|>user\nWhat\'s the temperature in San Francisco now? How about tomorrow?<|im_end|>\n<|im_start|>assistant\n<think>\n"""

        def get_step2_expected_renderer_prompt(self) -> str:
            return """<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\nCurrent Date: 2024-09-30\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{"type": "function", "function": {"name": "get_current_temperature", "description": "Get current temperature at a location.", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "The location to get the temperature for, in the format \\"City, State, Country\\"."}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "The unit to return the temperature in. Defaults to \\"celsius\\"."}}, "required": ["location"]}}}\n{"type": "function", "function": {"name": "get_temperature_date", "description": "Get temperature at a location and date.", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "The location to get the temperature for, in the format \\"City, State, Country\\"."}, "date": {"type": "string", "description": "The date to get the temperature for, in the format \\"Year-Month-Day\\"."}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "The unit to return the temperature in. Defaults to \\"celsius\\"."}}, "required": ["location", "date"]}}}\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call><|im_end|>\n<|im_start|>user\nWhat\'s the temperature in San Francisco now? How about tomorrow?<|im_end|>\n<|im_start|>assistant\n<tool_call>\n{"name": "get_current_temperature", "arguments": {"location": "San Francisco, CA, USA"}}\n</tool_call>\n<tool_call>\n{"name": "get_temperature_date", "arguments": {"location": "San Francisco, CA, USA", "date": "2024-10-01"}}\n</tool_call><|im_end|>\n<|im_start|>user\n<tool_response>\n{"temperature": 26.1, "location": "San Francisco, CA, USA", "unit": "celsius"}\n</tool_response>\n<tool_response>\n{"temperature": 25.9, "location": "San Francisco, CA, USA", "date": "2024-10-01", "unit": "celsius"}\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n"""

        def get_step3_expected_renderer_prompt(self) -> str:
            return """<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\nCurrent Date: 2024-09-30\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{"type": "function", "function": {"name": "get_current_temperature", "description": "Get current temperature at a location.", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "The location to get the temperature for, in the format \\"City, State, Country\\"."}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "The unit to return the temperature in. Defaults to \\"celsius\\"."}}, "required": ["location"]}}}\n{"type": "function", "function": {"name": "get_temperature_date", "description": "Get temperature at a location and date.", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "The location to get the temperature for, in the format \\"City, State, Country\\"."}, "date": {"type": "string", "description": "The date to get the temperature for, in the format \\"Year-Month-Day\\"."}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "The unit to return the temperature in. Defaults to \\"celsius\\"."}}, "required": ["location", "date"]}}}\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call><|im_end|>\n<|im_start|>user\nWhat\'s the temperature in San Francisco now? How about tomorrow?<|im_end|>\n<|im_start|>assistant\n<tool_call>\n{"name": "get_current_temperature", "arguments": {"location": "San Francisco, CA, USA"}}\n</tool_call>\n<tool_call>\n{"name": "get_temperature_date", "arguments": {"location": "San Francisco, CA, USA", "date": "2024-10-01"}}\n</tool_call><|im_end|>\n<|im_start|>user\n<tool_response>\n{"temperature": 26.1, "location": "San Francisco, CA, USA", "unit": "celsius"}\n</tool_response>\n<tool_response>\n{"temperature": 25.9, "location": "San Francisco, CA, USA", "date": "2024-10-01", "unit": "celsius"}\n</tool_response><|im_end|>\n<|im_start|>assistant\nSan Francisco今天26.1度, 明天25.9度<|im_end|>\n<|im_start|>user\n那北京明天温度如何<|im_end|>\n<|im_start|>assistant\n<think>\n"""

    # Kimik2渲染器测试实现
    class Kimik2RendererTestImpl(BaseRendererTestMixin):
        def __init__(self, test_instance):
            self.test_instance = test_instance

        def get_tokenizer(self):
            return BaseTokenizer(
                f"{self.test_instance.test_data_path}/model_test/fake_test/testdata/kimi_k2/tokenizer/",
            )

        def get_render_params(self) -> RendererParams:
            tokenizer = self.get_tokenizer()
            return RendererParams(
                model_type="kimi_k2",
                max_seq_len=1024,
                eos_token_id=tokenizer.eos_token_id or 0,
                stop_word_ids_list=[],
            )

        @override
        def get_messages_with_tool_response(self) -> List[ChatMessage]:
            """获取包含工具响应的消息 - 通用实现"""
            messages = self.get_initial_messages()
            messages.extend(
                [
                    ChatMessage(
                        **{
                            "role": RoleEnum.assistant,
                            "tool_calls": [
                                {
                                    "id": "get_current_temperature:0",
                                    "type": "function",
                                    "function": {
                                        "name": "get_current_temperature",
                                        "arguments": '{"location": "San Francisco, CA, USA"}',
                                    },
                                },
                                {
                                    "id": "get_temperature_date:1",
                                    "type": "function",
                                    "function": {
                                        "name": "get_temperature_date",
                                        "arguments": '{"location": "San Francisco, CA, USA", "date": "2024-10-01"}',
                                    },
                                },
                            ],
                        }
                    ),
                    ChatMessage(
                        **{
                            "role": RoleEnum.tool,
                            "content": '{"temperature": 26.1, "location": "San Francisco, CA, USA", "unit": "celsius"}',
                            "tool_call_id": "get_current_temperature:0",
                        }
                    ),
                    ChatMessage(
                        **{
                            "role": RoleEnum.tool,
                            "content": '{"temperature": 25.9, "location": "San Francisco, CA, USA", "date": "2024-10-01", "unit": "celsius"}',
                            "tool_call_id": "get_temperature_date:1",
                        }
                    ),
                ]
            )
            return messages

        def get_step1_expected_renderer_prompt(self) -> str:
            return """<|im_system|>tool_declare<|im_middle|>\n  # Tools\n  [{"type": "function", "function": {"name": "get_current_temperature", "description": "Get current temperature at a location.", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "The location to get the temperature for, in the format \\"City, State, Country\\"."}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "The unit to return the temperature in. Defaults to \\"celsius\\"."}}, "required": ["location"]}}}, {"type": "function", "function": {"name": "get_temperature_date", "description": "Get temperature at a location and date.", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "The location to get the temperature for, in the format \\"City, State, Country\\"."}, "date": {"type": "string", "description": "The date to get the temperature for, in the format \\"Year-Month-Day\\"."}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "The unit to return the temperature in. Defaults to \\"celsius\\"."}}, "required": ["location", "date"]}}}]<|im_end|><|im_system|>system<|im_middle|>You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\nCurrent Date: 2024-09-30<|im_end|><|im_user|>user<|im_middle|>What\'s the temperature in San Francisco now? How about tomorrow?<|im_end|><|im_assistant|>assistant<|im_middle|>"""

        def get_step2_expected_renderer_prompt(self) -> str:
            return """<|im_system|>tool_declare<|im_middle|>\n  # Tools\n  [{"type": "function", "function": {"name": "get_current_temperature", "description": "Get current temperature at a location.", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "The location to get the temperature for, in the format \\"City, State, Country\\"."}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "The unit to return the temperature in. Defaults to \\"celsius\\"."}}, "required": ["location"]}}}, {"type": "function", "function": {"name": "get_temperature_date", "description": "Get temperature at a location and date.", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "The location to get the temperature for, in the format \\"City, State, Country\\"."}, "date": {"type": "string", "description": "The date to get the temperature for, in the format \\"Year-Month-Day\\"."}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "The unit to return the temperature in. Defaults to \\"celsius\\"."}}, "required": ["location", "date"]}}}]<|im_end|><|im_system|>system<|im_middle|>You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\nCurrent Date: 2024-09-30<|im_end|><|im_user|>user<|im_middle|>What\'s the temperature in San Francisco now? How about tomorrow?<|im_end|><|im_assistant|>assistant<|im_middle|><|tool_calls_section_begin|><|tool_call_begin|>functions.get_current_temperature:0<|tool_call_argument_begin|>{"location": "San Francisco, CA, USA"}<|tool_call_end|><|tool_call_begin|>functions.get_temperature_date:1<|tool_call_argument_begin|>{"location": "San Francisco, CA, USA", "date": "2024-10-01"}<|tool_call_end|><|tool_calls_section_end|><|im_end|><|im_system|>tool<|im_middle|>## Return of functions.get_current_temperature:0\n    {"temperature": 26.1, "location": "San Francisco, CA, USA", "unit": "celsius"}<|im_end|><|im_system|>tool<|im_middle|>## Return of functions.get_temperature_date:1\n    {"temperature": 25.9, "location": "San Francisco, CA, USA", "date": "2024-10-01", "unit": "celsius"}<|im_end|><|im_assistant|>assistant<|im_middle|>"""

        def get_step3_expected_renderer_prompt(self) -> str:
            return """<|im_system|>tool_declare<|im_middle|>\n  # Tools\n  [{"type": "function", "function": {"name": "get_current_temperature", "description": "Get current temperature at a location.", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "The location to get the temperature for, in the format \\"City, State, Country\\"."}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "The unit to return the temperature in. Defaults to \\"celsius\\"."}}, "required": ["location"]}}}, {"type": "function", "function": {"name": "get_temperature_date", "description": "Get temperature at a location and date.", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "The location to get the temperature for, in the format \\"City, State, Country\\"."}, "date": {"type": "string", "description": "The date to get the temperature for, in the format \\"Year-Month-Day\\"."}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "The unit to return the temperature in. Defaults to \\"celsius\\"."}}, "required": ["location", "date"]}}}]<|im_end|><|im_system|>system<|im_middle|>You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\nCurrent Date: 2024-09-30<|im_end|><|im_user|>user<|im_middle|>What\'s the temperature in San Francisco now? How about tomorrow?<|im_end|><|im_assistant|>assistant<|im_middle|><|tool_calls_section_begin|><|tool_call_begin|>functions.get_current_temperature:0<|tool_call_argument_begin|>{"location": "San Francisco, CA, USA"}<|tool_call_end|><|tool_call_begin|>functions.get_temperature_date:1<|tool_call_argument_begin|>{"location": "San Francisco, CA, USA", "date": "2024-10-01"}<|tool_call_end|><|tool_calls_section_end|><|im_end|><|im_system|>tool<|im_middle|>## Return of functions.get_current_temperature:0\n    {"temperature": 26.1, "location": "San Francisco, CA, USA", "unit": "celsius"}<|im_end|><|im_system|>tool<|im_middle|>## Return of functions.get_temperature_date:1\n    {"temperature": 25.9, "location": "San Francisco, CA, USA", "date": "2024-10-01", "unit": "celsius"}<|im_end|><|im_assistant|>assistant<|im_middle|>San Francisco今天26.1度, 明天25.9度<|im_end|><|im_user|>user<|im_middle|>那北京明天温度如何<|im_end|><|im_assistant|>assistant<|im_middle|>"""

    class Kimik2RendererErrorCheckTestImpl(Kimik2RendererTestImpl):
        @override
        def get_messages_with_tool_response(self) -> List[ChatMessage]:
            """获取包含工具响应的消息 - 通用实现"""
            messages = self.get_initial_messages()
            messages.extend(
                [
                    ChatMessage(
                        **{
                            "role": RoleEnum.assistant,
                            "tool_calls": [
                                {
                                    "id": "get_current_temperature:0",
                                    "type": "function",
                                    "function": {
                                        "name": "get_current_temperature",
                                        "arguments": '{"location": "San Francisco, CA, USA"}',
                                    },
                                },
                                {
                                    "id": "get_temperature_date:1",
                                    "type": "function",
                                    "function": {
                                        "name": "get_temperature_date",
                                        "arguments": '{"location": "San Francisco, CA, USA", "date": "2024-10-01"}',
                                    },
                                },
                            ],
                        }
                    ),
                    ChatMessage(
                        **{
                            "role": RoleEnum.tool,
                            "content": '{"temperature": 26.1, "location": "San Francisco, CA, USA", "unit": "celsius"}',
                            "tool_call_id": "error_get_current_temperature:0",
                        }
                    ),
                    ChatMessage(
                        **{
                            "role": RoleEnum.tool,
                            "content": '{"temperature": 25.9, "location": "San Francisco, CA, USA", "date": "2024-10-01", "unit": "celsius"}',
                            "tool_call_id": "error_get_temperature_date:1",
                        }
                    ),
                ]
            )
            return messages

        @override
        def run_renderer_test(self):
            """运行渲染器测试的通用方法"""
            tokenizer = self.get_tokenizer()
            render_params = self.get_render_params()
            generate_env_config = GenerateEnvConfig()
            render_config = RenderConfig()
            chat_renderer = ChatRendererFactory.get_renderer(
                tokenizer,
                render_params,
                generate_env_config=generate_env_config,
                render_config=render_config,
            )
            tools = self.get_tools()

            # 测试步骤1：初始消息
            initial_messages = self.get_initial_messages()
            request = ChatCompletionRequest(messages=initial_messages, tools=tools)
            renderer_prompt = chat_renderer.render_chat(request).rendered_prompt
            expected_prompt = self.get_step1_expected_renderer_prompt()
            logging.info(
                f"Step 1 prompt: \n{renderer_prompt}\n-----------------------------------"
            )
            assert renderer_prompt == expected_prompt, "Step 1 prompt mismatch"

            # 测试步骤2：包含工具响应的消息
            tool_response_messages = self.get_messages_with_tool_response()
            request = ChatCompletionRequest(
                messages=tool_response_messages, tools=tools
            )

            # 验证是否抛出ValueError
            try:
                renderer_prompt = chat_renderer.render_chat(request).rendered_prompt
                # 如果没有抛出异常，测试失败
                assert False, "Expected ValueError was not raised in step 2"
            except ValueError as e:
                # 预期的异常，测试通过
                logging.info(f"Step 2 correctly raised ValueError: {e}")
            except Exception as e:
                # 抛出了其他异常，测试失败
                assert False, f"Expected ValueError but got {type(e).__name__}: {e}"

    class Kimik2ThinkingRendererTestImpl(Kimik2RendererTestImpl):
        """Kimi-K2-Thinking test inherits from regular Kimik2 test"""

        def get_tools(self) -> List[GPTToolDefinition]:
            return []

        def get_initial_messages(self) -> List[ChatMessage]:
            return [
                ChatMessage(
                    role=RoleEnum.system,
                    content="You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\nCurrent Date: 2024-09-30",
                ),
                ChatMessage(
                    role=RoleEnum.user, content="What is 25 * 4? Think step by step."
                ),
            ]

        def get_messages_with_tool_response(self) -> List[ChatMessage]:
            messages = self.get_initial_messages()
            messages.append(
                ChatMessage(
                    role=RoleEnum.assistant,
                    content="<think>Let me calculate this step by step. 25 * 4 = 25 * 2 * 2 = 50 * 2 = 100</think>The answer is 100.",
                )
            )
            messages.append(
                ChatMessage(role=RoleEnum.user, content="What about 100 / 4?")
            )
            return messages

        def get_messages_with_follow_up(self) -> List[ChatMessage]:
            messages = self.get_messages_with_tool_response()
            messages.append(
                ChatMessage(
                    role=RoleEnum.assistant,
                    content="<think>100 divided by 4 is straightforward: 100 / 4 = 25</think>The result is 25.",
                )
            )
            messages.append(ChatMessage(role=RoleEnum.user, content="Add 10 to that"))
            return messages

        def get_step1_expected_renderer_prompt(self) -> str:
            return """<|im_system|>system<|im_middle|>You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\nCurrent Date: 2024-09-30<|im_end|><|im_user|>user<|im_middle|>What is 25 * 4? Think step by step.<|im_end|><|im_assistant|>assistant<|im_middle|>"""

        def get_step2_expected_renderer_prompt(self) -> str:
            return """<|im_system|>system<|im_middle|>You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\nCurrent Date: 2024-09-30<|im_end|><|im_user|>user<|im_middle|>What is 25 * 4? Think step by step.<|im_end|><|im_assistant|>assistant<|im_middle|><think>Let me calculate this step by step. 25 * 4 = 25 * 2 * 2 = 50 * 2 = 100</think>The answer is 100.<|im_end|><|im_user|>user<|im_middle|>What about 100 / 4?<|im_end|><|im_assistant|>assistant<|im_middle|>"""

        def get_step3_expected_renderer_prompt(self) -> str:
            return """<|im_system|>system<|im_middle|>You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\nCurrent Date: 2024-09-30<|im_end|><|im_user|>user<|im_middle|>What is 25 * 4? Think step by step.<|im_end|><|im_assistant|>assistant<|im_middle|><think>Let me calculate this step by step. 25 * 4 = 25 * 2 * 2 = 50 * 2 = 100</think>The answer is 100.<|im_end|><|im_user|>user<|im_middle|>What about 100 / 4?<|im_end|><|im_assistant|>assistant<|im_middle|><think>100 divided by 4 is straightforward: 100 / 4 = 25</think>The result is 25.<|im_end|><|im_user|>user<|im_middle|>Add 10 to that<|im_end|><|im_assistant|>assistant<|im_middle|>"""

    # Qwen3 Coder渲染器测试实现
    class Qwen3CoderRendererTestImpl(BaseRendererTestMixin):
        def __init__(self, test_instance):
            self.test_instance = test_instance

        def get_tokenizer(self):
            return BaseTokenizer(
                f"{self.test_instance.test_data_path}/model_test/fake_test/testdata/qwen3_coder/tokenizer/",
            )

        def get_render_params(self) -> RendererParams:
            tokenizer = self.get_tokenizer()
            return RendererParams(
                model_type="qwen3_coder_moe",
                max_seq_len=1024,
                eos_token_id=tokenizer.eos_token_id or 0,
                stop_word_ids_list=[],
            )

        def get_step1_expected_renderer_prompt(self) -> str:
            return """<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\nCurrent Date: 2024-09-30\n\nYou have access to the following functions:\n\n<tools>\n<function>\n<name>get_current_temperature</name>\n<description>Get current temperature at a location.</description>\n<parameters>\n<parameter>\n<name>location</name>\n<type>string</type>\n<description>The location to get the temperature for, in the format "City, State, Country".</description>\n</parameter>\n<parameter>\n<name>unit</name>\n<type>string</type>\n<description>The unit to return the temperature in. Defaults to "celsius".</description>\n<enum>[`celsius`, `fahrenheit`]</enum>\n</parameter>\n<required>[`location`]</required>\n</parameters>\n</function>\n<function>\n<name>get_temperature_date</name>\n<description>Get temperature at a location and date.</description>\n<parameters>\n<parameter>\n<name>location</name>\n<type>string</type>\n<description>The location to get the temperature for, in the format "City, State, Country".</description>\n</parameter>\n<parameter>\n<name>date</name>\n<type>string</type>\n<description>The date to get the temperature for, in the format "Year-Month-Day".</description>\n</parameter>\n<parameter>\n<name>unit</name>\n<type>string</type>\n<description>The unit to return the temperature in. Defaults to "celsius".</description>\n<enum>[`celsius`, `fahrenheit`]</enum>\n</parameter>\n<required>[`location`, `date`]</required>\n</parameters>\n</function>\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT><|im_end|>\n<|im_start|>user\nWhat\'s the temperature in San Francisco now? How about tomorrow?<|im_end|>\n<|im_start|>assistant\n"""

        def get_step2_expected_renderer_prompt(self) -> str:
            return """<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\nCurrent Date: 2024-09-30\n\nYou have access to the following functions:\n\n<tools>\n<function>\n<name>get_current_temperature</name>\n<description>Get current temperature at a location.</description>\n<parameters>\n<parameter>\n<name>location</name>\n<type>string</type>\n<description>The location to get the temperature for, in the format "City, State, Country".</description>\n</parameter>\n<parameter>\n<name>unit</name>\n<type>string</type>\n<description>The unit to return the temperature in. Defaults to "celsius".</description>\n<enum>[`celsius`, `fahrenheit`]</enum>\n</parameter>\n<required>[`location`]</required>\n</parameters>\n</function>\n<function>\n<name>get_temperature_date</name>\n<description>Get temperature at a location and date.</description>\n<parameters>\n<parameter>\n<name>location</name>\n<type>string</type>\n<description>The location to get the temperature for, in the format "City, State, Country".</description>\n</parameter>\n<parameter>\n<name>date</name>\n<type>string</type>\n<description>The date to get the temperature for, in the format "Year-Month-Day".</description>\n</parameter>\n<parameter>\n<name>unit</name>\n<type>string</type>\n<description>The unit to return the temperature in. Defaults to "celsius".</description>\n<enum>[`celsius`, `fahrenheit`]</enum>\n</parameter>\n<required>[`location`, `date`]</required>\n</parameters>\n</function>\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT><|im_end|>\n<|im_start|>user\nWhat\'s the temperature in San Francisco now? How about tomorrow?<|im_end|>\n<|im_start|>assistant\n<tool_call>\n<function=get_current_temperature>\n<parameter=location>\nSan Francisco, CA, USA\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=get_temperature_date>\n<parameter=location>\nSan Francisco, CA, USA\n</parameter>\n<parameter=date>\n2024-10-01\n</parameter>\n</function>\n</tool_call><|im_end|>\n<|im_start|>user\n<tool_response>\n{"temperature": 26.1, "location": "San Francisco, CA, USA", "unit": "celsius"}\n</tool_response>\n<tool_response>\n{"temperature": 25.9, "location": "San Francisco, CA, USA", "date": "2024-10-01", "unit": "celsius"}\n</tool_response>\n<|im_end|>\n<|im_start|>assistant\n"""

        def get_step3_expected_renderer_prompt(self) -> str:
            return """<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\nCurrent Date: 2024-09-30\n\nYou have access to the following functions:\n\n<tools>\n<function>\n<name>get_current_temperature</name>\n<description>Get current temperature at a location.</description>\n<parameters>\n<parameter>\n<name>location</name>\n<type>string</type>\n<description>The location to get the temperature for, in the format "City, State, Country".</description>\n</parameter>\n<parameter>\n<name>unit</name>\n<type>string</type>\n<description>The unit to return the temperature in. Defaults to "celsius".</description>\n<enum>[`celsius`, `fahrenheit`]</enum>\n</parameter>\n<required>[`location`]</required>\n</parameters>\n</function>\n<function>\n<name>get_temperature_date</name>\n<description>Get temperature at a location and date.</description>\n<parameters>\n<parameter>\n<name>location</name>\n<type>string</type>\n<description>The location to get the temperature for, in the format "City, State, Country".</description>\n</parameter>\n<parameter>\n<name>date</name>\n<type>string</type>\n<description>The date to get the temperature for, in the format "Year-Month-Day".</description>\n</parameter>\n<parameter>\n<name>unit</name>\n<type>string</type>\n<description>The unit to return the temperature in. Defaults to "celsius".</description>\n<enum>[`celsius`, `fahrenheit`]</enum>\n</parameter>\n<required>[`location`, `date`]</required>\n</parameters>\n</function>\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT><|im_end|>\n<|im_start|>user\nWhat\'s the temperature in San Francisco now? How about tomorrow?<|im_end|>\n<|im_start|>assistant\n<tool_call>\n<function=get_current_temperature>\n<parameter=location>\nSan Francisco, CA, USA\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=get_temperature_date>\n<parameter=location>\nSan Francisco, CA, USA\n</parameter>\n<parameter=date>\n2024-10-01\n</parameter>\n</function>\n</tool_call><|im_end|>\n<|im_start|>user\n<tool_response>\n{"temperature": 26.1, "location": "San Francisco, CA, USA", "unit": "celsius"}\n</tool_response>\n<tool_response>\n{"temperature": 25.9, "location": "San Francisco, CA, USA", "date": "2024-10-01", "unit": "celsius"}\n</tool_response>\n<|im_end|>\n<|im_start|>assistant\nSan Francisco今天26.1度, 明天25.9度<|im_end|>\n<|im_start|>user\n那北京明天温度如何<|im_end|>\n<|im_start|>assistant\n"""

    # GLM45渲染器测试实现
    class Glm45RendererTestImpl(BaseRendererTestMixin):
        def __init__(self, test_instance):
            self.test_instance = test_instance

        def get_tokenizer(self):
            return BaseTokenizer(
                f"{self.test_instance.test_data_path}/model_test/fake_test/testdata/glm45/tokenizer/"
            )

        def get_render_params(self) -> RendererParams:
            tokenizer = self.get_tokenizer()
            return RendererParams(
                model_type="glm4_moe",
                max_seq_len=1024,
                eos_token_id=tokenizer.eos_token_id or 0,
                stop_word_ids_list=[],
            )

        def get_step1_expected_renderer_prompt(self) -> str:
            return """[gMASK]<sop><|system|>\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{"type": "function", "function": {"name": "get_current_temperature", "description": "Get current temperature at a location.", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "The location to get the temperature for, in the format \\"City, State, Country\\"."}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "The unit to return the temperature in. Defaults to \\"celsius\\"."}}, "required": ["location"]}}}\n{"type": "function", "function": {"name": "get_temperature_date", "description": "Get temperature at a location and date.", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "The location to get the temperature for, in the format \\"City, State, Country\\"."}, "date": {"type": "string", "description": "The date to get the temperature for, in the format \\"Year-Month-Day\\"."}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "The unit to return the temperature in. Defaults to \\"celsius\\"."}}, "required": ["location", "date"]}}}\n</tools>\n\nFor each function call, output the function name and arguments within the following XML format:\n<tool_call>{function-name}\n<arg_key>{arg-key-1}</arg_key>\n<arg_value>{arg-value-1}</arg_value>\n<arg_key>{arg-key-2}</arg_key>\n<arg_value>{arg-value-2}</arg_value>\n...\n</tool_call><|system|>\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\nCurrent Date: 2024-09-30<|user|>\nWhat\'s the temperature in San Francisco now? How about tomorrow?<|assistant|>"""

        def get_step2_expected_renderer_prompt(self) -> str:
            return """[gMASK]<sop><|system|>\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{"type": "function", "function": {"name": "get_current_temperature", "description": "Get current temperature at a location.", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "The location to get the temperature for, in the format \\"City, State, Country\\"."}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "The unit to return the temperature in. Defaults to \\"celsius\\"."}}, "required": ["location"]}}}\n{"type": "function", "function": {"name": "get_temperature_date", "description": "Get temperature at a location and date.", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "The location to get the temperature for, in the format \\"City, State, Country\\"."}, "date": {"type": "string", "description": "The date to get the temperature for, in the format \\"Year-Month-Day\\"."}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "The unit to return the temperature in. Defaults to \\"celsius\\"."}}, "required": ["location", "date"]}}}\n</tools>\n\nFor each function call, output the function name and arguments within the following XML format:\n<tool_call>{function-name}\n<arg_key>{arg-key-1}</arg_key>\n<arg_value>{arg-value-1}</arg_value>\n<arg_key>{arg-key-2}</arg_key>\n<arg_value>{arg-value-2}</arg_value>\n...\n</tool_call><|system|>\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\nCurrent Date: 2024-09-30<|user|>\nWhat\'s the temperature in San Francisco now? How about tomorrow?<|assistant|>\n<think></think>\n<tool_call>get_current_temperature\n<arg_key>location</arg_key>\n<arg_value>San Francisco, CA, USA</arg_value>\n</tool_call>\n<tool_call>get_temperature_date\n<arg_key>location</arg_key>\n<arg_value>San Francisco, CA, USA</arg_value>\n<arg_key>date</arg_key>\n<arg_value>2024-10-01</arg_value>\n</tool_call><|observation|>\n<tool_response>\n{"temperature": 26.1, "location": "San Francisco, CA, USA", "unit": "celsius"}\n</tool_response>\n<tool_response>\n{"temperature": 25.9, "location": "San Francisco, CA, USA", "date": "2024-10-01", "unit": "celsius"}\n</tool_response><|assistant|>"""

        def get_step3_expected_renderer_prompt(self) -> str:
            return """[gMASK]<sop><|system|>\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{"type": "function", "function": {"name": "get_current_temperature", "description": "Get current temperature at a location.", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "The location to get the temperature for, in the format \\"City, State, Country\\"."}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "The unit to return the temperature in. Defaults to \\"celsius\\"."}}, "required": ["location"]}}}\n{"type": "function", "function": {"name": "get_temperature_date", "description": "Get temperature at a location and date.", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "The location to get the temperature for, in the format \\"City, State, Country\\"."}, "date": {"type": "string", "description": "The date to get the temperature for, in the format \\"Year-Month-Day\\"."}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "The unit to return the temperature in. Defaults to \\"celsius\\"."}}, "required": ["location", "date"]}}}\n</tools>\n\nFor each function call, output the function name and arguments within the following XML format:\n<tool_call>{function-name}\n<arg_key>{arg-key-1}</arg_key>\n<arg_value>{arg-value-1}</arg_value>\n<arg_key>{arg-key-2}</arg_key>\n<arg_value>{arg-value-2}</arg_value>\n...\n</tool_call><|system|>\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\nCurrent Date: 2024-09-30<|user|>\nWhat\'s the temperature in San Francisco now? How about tomorrow?<|assistant|>\n<think></think>\n<tool_call>get_current_temperature\n<arg_key>location</arg_key>\n<arg_value>San Francisco, CA, USA</arg_value>\n</tool_call>\n<tool_call>get_temperature_date\n<arg_key>location</arg_key>\n<arg_value>San Francisco, CA, USA</arg_value>\n<arg_key>date</arg_key>\n<arg_value>2024-10-01</arg_value>\n</tool_call><|observation|>\n<tool_response>\n{"temperature": 26.1, "location": "San Francisco, CA, USA", "unit": "celsius"}\n</tool_response>\n<tool_response>\n{"temperature": 25.9, "location": "San Francisco, CA, USA", "date": "2024-10-01", "unit": "celsius"}\n</tool_response><|assistant|>\n<think></think>\nSan Francisco今天26.1度, 明天25.9度<|user|>\n那北京明天温度如何<|assistant|>"""

    # dsv31渲染器测试实现
    class DeepseekV31RendererTestImpl(BaseRendererTestMixin):
        def __init__(self, test_instance):
            self.test_instance = test_instance

        def get_tokenizer(self):
            return BaseTokenizer(
                f"{self.test_instance.test_data_path}/model_test/fake_test/testdata/deepseek_v31/tokenizer/",
            )

        def get_render_params(self) -> RendererParams:
            tokenizer = self.get_tokenizer()
            return RendererParams(
                model_type="deepseek_v31",
                max_seq_len=1024,
                eos_token_id=tokenizer.eos_token_id or 0,
                stop_word_ids_list=[],
            )

        def get_step1_expected_renderer_prompt(self) -> str:
            return """<｜begin▁of▁sentence｜>You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\nCurrent Date: 2024-09-30\n\n## Tools\nYou have access to the following tools:\n\n### get_current_temperature\nDescription: Get current temperature at a location.\n\nParameters: {"type": "object", "properties": {"location": {"type": "string", "description": "The location to get the temperature for, in the format \\"City, State, Country\\"."}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "The unit to return the temperature in. Defaults to \\"celsius\\"."}}, "required": ["location"]}\n\n### get_temperature_date\nDescription: Get temperature at a location and date.\n\nParameters: {"type": "object", "properties": {"location": {"type": "string", "description": "The location to get the temperature for, in the format \\"City, State, Country\\"."}, "date": {"type": "string", "description": "The date to get the temperature for, in the format \\"Year-Month-Day\\"."}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "The unit to return the temperature in. Defaults to \\"celsius\\"."}}, "required": ["location", "date"]}\n\nIMPORTANT: ALWAYS adhere to this exact format for tool use:\n<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>tool_call_name<｜tool▁sep｜>tool_call_arguments<｜tool▁call▁end｜>{{additional_tool_calls}}<｜tool▁calls▁end｜>\n\nWhere:\n\n- `tool_call_name` must be an exact match to one of the available tools\n- `tool_call_arguments` must be valid JSON that strictly follows the tool\'s Parameters Schema\n- For multiple tool calls, chain them directly without separators or spaces\n<｜User｜>What\'s the temperature in San Francisco now? How about tomorrow?<｜Assistant｜></think>"""

        def get_step2_expected_renderer_prompt(self) -> str:
            return """<｜begin▁of▁sentence｜>You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\nCurrent Date: 2024-09-30\n\n## Tools\nYou have access to the following tools:\n\n### get_current_temperature\nDescription: Get current temperature at a location.\n\nParameters: {"type": "object", "properties": {"location": {"type": "string", "description": "The location to get the temperature for, in the format \\"City, State, Country\\"."}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "The unit to return the temperature in. Defaults to \\"celsius\\"."}}, "required": ["location"]}\n\n### get_temperature_date\nDescription: Get temperature at a location and date.\n\nParameters: {"type": "object", "properties": {"location": {"type": "string", "description": "The location to get the temperature for, in the format \\"City, State, Country\\"."}, "date": {"type": "string", "description": "The date to get the temperature for, in the format \\"Year-Month-Day\\"."}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "The unit to return the temperature in. Defaults to \\"celsius\\"."}}, "required": ["location", "date"]}\n\nIMPORTANT: ALWAYS adhere to this exact format for tool use:\n<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>tool_call_name<｜tool▁sep｜>tool_call_arguments<｜tool▁call▁end｜>{{additional_tool_calls}}<｜tool▁calls▁end｜>\n\nWhere:\n\n- `tool_call_name` must be an exact match to one of the available tools\n- `tool_call_arguments` must be valid JSON that strictly follows the tool\'s Parameters Schema\n- For multiple tool calls, chain them directly without separators or spaces\n<｜User｜>What\'s the temperature in San Francisco now? How about tomorrow?<｜Assistant｜></think><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_current_temperature<｜tool▁sep｜>{"location": "San Francisco, CA, USA"}<｜tool▁call▁end｜><｜tool▁call▁begin｜>get_temperature_date<｜tool▁sep｜>{"location": "San Francisco, CA, USA", "date": "2024-10-01"}<｜tool▁call▁end｜><｜tool▁calls▁end｜><｜end▁of▁sentence｜><｜tool▁output▁begin｜>{"temperature": 26.1, "location": "San Francisco, CA, USA", "unit": "celsius"}<｜tool▁output▁end｜><｜tool▁output▁begin｜>{"temperature": 25.9, "location": "San Francisco, CA, USA", "date": "2024-10-01", "unit": "celsius"}<｜tool▁output▁end｜>"""

        def get_step3_expected_renderer_prompt(self) -> str:
            return """<｜begin▁of▁sentence｜>You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\nCurrent Date: 2024-09-30\n\n## Tools\nYou have access to the following tools:\n\n### get_current_temperature\nDescription: Get current temperature at a location.\n\nParameters: {"type": "object", "properties": {"location": {"type": "string", "description": "The location to get the temperature for, in the format \\"City, State, Country\\"."}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "The unit to return the temperature in. Defaults to \\"celsius\\"."}}, "required": ["location"]}\n\n### get_temperature_date\nDescription: Get temperature at a location and date.\n\nParameters: {"type": "object", "properties": {"location": {"type": "string", "description": "The location to get the temperature for, in the format \\"City, State, Country\\"."}, "date": {"type": "string", "description": "The date to get the temperature for, in the format \\"Year-Month-Day\\"."}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "The unit to return the temperature in. Defaults to \\"celsius\\"."}}, "required": ["location", "date"]}\n\nIMPORTANT: ALWAYS adhere to this exact format for tool use:\n<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>tool_call_name<｜tool▁sep｜>tool_call_arguments<｜tool▁call▁end｜>{{additional_tool_calls}}<｜tool▁calls▁end｜>\n\nWhere:\n\n- `tool_call_name` must be an exact match to one of the available tools\n- `tool_call_arguments` must be valid JSON that strictly follows the tool\'s Parameters Schema\n- For multiple tool calls, chain them directly without separators or spaces\n<｜User｜>What\'s the temperature in San Francisco now? How about tomorrow?<｜Assistant｜></think><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_current_temperature<｜tool▁sep｜>{"location": "San Francisco, CA, USA"}<｜tool▁call▁end｜><｜tool▁call▁begin｜>get_temperature_date<｜tool▁sep｜>{"location": "San Francisco, CA, USA", "date": "2024-10-01"}<｜tool▁call▁end｜><｜tool▁calls▁end｜><｜end▁of▁sentence｜><｜tool▁output▁begin｜>{"temperature": 26.1, "location": "San Francisco, CA, USA", "unit": "celsius"}<｜tool▁output▁end｜><｜tool▁output▁begin｜>{"temperature": 25.9, "location": "San Francisco, CA, USA", "date": "2024-10-01", "unit": "celsius"}<｜tool▁output▁end｜>San Francisco今天26.1度, 明天25.9度<｜end▁of▁sentence｜><｜User｜>那北京明天温度如何<｜Assistant｜></think>"""

    def test_qwen3(self):
        """Qwen3渲染器测试"""
        qwen3_test = self.Qwen3RendererTestImpl(self)
        qwen3_test.run_renderer_test()

    def test_qwen3_thinking(self):
        """Qwen3 Thinking渲染器测试"""
        qwen3_thinking_test = self.Qwen3ThinkingRendererTestImpl(self)
        qwen3_thinking_test.run_renderer_test()

    def test_kimik2(self):
        """Kimik2渲染器测试"""
        kimik2_test = self.Kimik2RendererTestImpl(self)
        kimik2_test.run_renderer_test()

    def test_kimik2_error_check(self):
        """Kimik2 Error Check渲染器测试"""
        kimik2_error_check_test = self.Kimik2RendererErrorCheckTestImpl(self)
        kimik2_error_check_test.run_renderer_test()

    def test_kimik2_thinking(self):
        """Kimik2 Thinking渲染器测试"""
        kimik2_thinking_test = self.Kimik2ThinkingRendererTestImpl(self)
        kimik2_thinking_test.run_renderer_test()

    def test_qwen3_coder(self):
        """Qwen3 Coder渲染器测试"""
        qwen3_coder_test = self.Qwen3CoderRendererTestImpl(self)
        qwen3_coder_test.run_renderer_test()

    def test_glm45(self):
        """GLM45渲染器测试"""
        glm45_test = self.Glm45RendererTestImpl(self)
        glm45_test.run_renderer_test()

    def test_deepseek_v31(self):
        """DSv31渲染器测试"""
        dsv31_test = self.DeepseekV31RendererTestImpl(self)
        dsv31_test.run_renderer_test()


if __name__ == "__main__":
    main()
