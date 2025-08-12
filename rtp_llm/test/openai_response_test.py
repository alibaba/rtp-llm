import asyncio
import functools
import json
import logging
import os
from contextlib import asynccontextmanager, contextmanager
from typing import Any, AsyncGenerator, List
from unittest import IsolatedAsyncioTestCase, TestCase, main

import torch
from transformers import AutoTokenizer, PreTrainedTokenizer
from typing_extensions import override

from rtp_llm.config.generate_config import GenerateConfig
from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.models.base_model import (
    AuxInfo,
    BaseModel,
    GenerateInput,
    GenerateOutput,
    GenerateOutputs,
)
from rtp_llm.openai.api_datatype import (
    ChatCompletionRequest,
    ChatCompletionStreamResponse,
    ChatMessage,
    FinisheReason,
    GPTFunctionDefinition,
    GPTToolDefinition,
    RoleEnum,
)
from rtp_llm.openai.openai_endpoint import OpenaiEndpoint
from rtp_llm.openai.renderer_factory import ChatRendererFactory, RendererParams
from rtp_llm.openai.renderers import custom_renderer
from rtp_llm.openai.renderers.chatglm45_renderer import ChatGlm45Renderer
from rtp_llm.openai.renderers.kimik2_renderer import KimiK2Renderer
from rtp_llm.openai.renderers.qwen3_code_renderer import Qwen3CoderRenderer
from rtp_llm.openai.renderers.qwen_reasoning_tool_renderer import (
    QwenReasoningToolRenderer,
)
from rtp_llm.openai.renderers.qwen_renderer import QwenRenderer
from rtp_llm.test.utils.stream_util import (
    is_valid_tool_call_chunk,
    merge_stream_responses,
)
from rtp_llm.tokenizer.tokenization_chatglm3 import ChatGLMTokenizer
from rtp_llm.tokenizer.tokenization_qwen import QWenTokenizer


@asynccontextmanager
async def think_mode_context():
    """异步上下文管理器：临时设置 THINK_MODE = 1"""
    original_mode = getattr(custom_renderer, "THINK_MODE", 0)
    try:
        custom_renderer.THINK_MODE = 1
        yield
    finally:
        custom_renderer.THINK_MODE = original_mode


@contextmanager
def think_mode_context_sync():
    """同步上下文管理器：临时设置 THINK_MODE = 1"""
    original_mode = getattr(custom_renderer, "THINK_MODE", 0)
    try:
        custom_renderer.THINK_MODE = 1
        yield
    finally:
        custom_renderer.THINK_MODE = original_mode


def think_mode(func):
    """装饰器：在函数执行期间设置 THINK_MODE = 1"""

    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        async with think_mode_context():
            return await func(*args, **kwargs)

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        with think_mode_context_sync():
            return func(*args, **kwargs)

    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper


async def fake_output_generator(
    output_ids: List[int], max_seq_len: int, eos_id: int, seq_len: int
) -> AsyncGenerator[GenerateOutputs, None]:
    # 流式的返回结果
    for i in range(0, len(output_ids)):
        output_tensor = torch.full((1, max_seq_len), eos_id, dtype=torch.int)

        output_tensor[0, : len(output_ids[i : i + 1])] = torch.tensor(
            output_ids[i : i + 1], dtype=torch.int
        )
        finished = torch.full((1,), (i == (len(output_ids) - 1)), dtype=torch.bool)
        outputs = GenerateOutputs()
        aux = AuxInfo()
        aux.input_len = seq_len
        aux.output_len = i + 1
        outputs.generate_outputs.append(
            GenerateOutput(
                hidden_states=None,
                output_ids=output_tensor,
                finished=finished,
                aux_info=aux,
                loss=None,
                logits=None,
            )
        )
        yield outputs


async def fake_output_generator_once(
    output_ids: List[int], max_seq_len: int, eos_id: int, seq_len: int
) -> AsyncGenerator[GenerateOutputs, None]:
    # 创建包含所有token的完整输出张量, 模拟非流式的返回
    output_tensor = torch.full((1, max_seq_len), eos_id, dtype=torch.int)
    output_tensor[0, : len(output_ids)] = torch.tensor(output_ids, dtype=torch.int)

    # 标记为已完成
    finished = torch.full((1,), True, dtype=torch.bool)

    outputs = GenerateOutputs()
    aux = AuxInfo()
    aux.input_len = seq_len
    aux.output_len = len(output_ids)

    outputs.generate_outputs.append(
        GenerateOutput(
            hidden_states=None,
            output_ids=output_tensor,
            finished=finished,
            aux_info=aux,
            loss=None,
            logits=None,
        )
    )

    yield outputs


MAX_SEQ_LEN = 1024


class FakeModel(BaseModel):
    def load_tokenizer(self):
        pass

    def init_misc(self):
        pass

    def load(self, ckpt_path: str):
        pass


class BaseToolCallTestSuite:
    """工具调用测试的基类，包含通用的测试逻辑"""

    def __init__(self, parent_test: "OpenaiResponseTest"):
        self.parent = parent_test

    # 抽象方法 - 子类必须实现
    def _get_model_type(self):
        """获取模型类型 - 子类必须实现"""
        raise NotImplementedError("子类必须实现 _get_model_type 方法")

    def _get_tokenizer_path(self):
        """获取tokenizer路径 - 子类必须实现"""
        raise NotImplementedError("子类必须实现 _get_tokenizer_path 方法")

    def _get_test_data(self, include_stop_word=False):
        """获取测试数据 - 子类必须实现"""
        raise NotImplementedError("子类必须实现 _get_test_data 方法")

    def _assert_tool_call_response(self, response_delta, expected_content=""):
        """断言工具调用响应的内容 - 子类必须实现"""
        raise NotImplementedError("子类必须实现 _assert_tool_call_response 方法")

    # 可选重写的方法
    def _setup_additional_environment(self):
        """设置额外的环境变量 - 子类可以重写"""
        pass

    def _create_tokenizer(self, tokenizer_path):
        """创建tokenizer - 子类可以重写"""
        return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    def _create_render_params(self, tokenizer):
        """创建渲染参数 - 子类可以重写"""
        return RendererParams(
            model_type=self._get_model_type(),
            max_seq_len=1024,
            eos_token_id=tokenizer.eos_token_id or 0,
            stop_word_ids_list=[],
        )

    def _create_test_functions_and_tools(self):
        """创建测试用的函数和工具定义 - 子类可以重写"""
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
        tools = [GPTToolDefinition(function=functions[0])]
        return functions, tools

    def _validate_renderer(self, chat_renderer):
        """验证renderer类型 - 子类可以重写"""
        pass

    def _validate_stream_chunk(self, chunk, stream):
        """验证流式chunk - 子类可以重写"""
        pass

    def _setup_environment(self):
        """设置测试环境"""
        os.environ["MODEL_TYPE"] = self._get_model_type()

        # 设置额外的环境变量
        self._setup_additional_environment()

        tokenizer_path = f"{self.parent.test_data_path}/{self._get_tokenizer_path()}"
        tokenizer = self._create_tokenizer(tokenizer_path)

        self.parent.model.tokenizer = tokenizer
        self.parent.endpoint = OpenaiEndpoint(
            self.parent.model.config, self.parent.model.tokenizer, None
        )

        return tokenizer

    async def _run_tool_call_test(
        self,
        stream=True,
        include_stop_word=False,
        stop_words_str=None,
    ):
        """运行工具调用测试的通用方法"""
        tokenizer = self._setup_environment()
        test_ids = self._get_test_data(include_stop_word)
        render_params = self._create_render_params(tokenizer)

        chat_renderer = ChatRendererFactory.get_renderer(tokenizer, render_params)

        # 子类特定的renderer验证
        self._validate_renderer(chat_renderer)

        functions, tools = self._create_test_functions_and_tools()
        request = ChatCompletionRequest(
            messages=[ChatMessage(role=RoleEnum.user, content="hello")],
            tools=tools,
            stream=stream,
        )

        seq_len_no_use = 314

        if not stream:
            id_generator = fake_output_generator_once(
                test_ids, MAX_SEQ_LEN, tokenizer.eos_token_id or 0, seq_len_no_use
            )
        else:
            id_generator = fake_output_generator(
                test_ids, MAX_SEQ_LEN, tokenizer.eos_token_id or 0, seq_len_no_use
            )

        generate_config = GenerateConfig(is_streaming=stream)
        if stop_words_str:
            generate_config.stop_words_str = stop_words_str

        stream_generator = chat_renderer.render_response_stream(
            id_generator, request, generate_config=generate_config
        )
        generate = self.parent.endpoint._complete_stream_response(
            stream_generator, None
        )

        chunk_list = []
        async for chunk in generate:
            self._validate_stream_chunk(chunk, stream)
            chunk_list.append(chunk)

        return chunk_list

    def _validate_merged_result(self, merged_result):
        """验证合并后的结果 - 通用验证逻辑"""
        assert merged_result.choices[0].finish_reason == FinisheReason.tool_calls
        delta = merged_result.choices[0].delta
        assert delta.role == RoleEnum.assistant
        self._assert_tool_call_response(delta)

    async def test_streaming_case(self, stop_words_str=None):
        """测试工具调用流式场景"""
        chunk_list = await self._run_tool_call_test(
            stream=True, stop_words_str=stop_words_str
        )

        merged_result: ChatCompletionStreamResponse = merge_stream_responses(chunk_list)
        self._validate_merged_result(merged_result)

    async def test_no_stream(self, stop_words_str=None):
        """测试工具调用非流式场景"""
        chunk_list = await self._run_tool_call_test(
            stream=False,
            stop_words_str=stop_words_str,
        )

        # 验证合并结果
        merged_result: ChatCompletionStreamResponse = merge_stream_responses(chunk_list)
        self._validate_merged_result(merged_result)


class OpenaiResponseTest(IsolatedAsyncioTestCase):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.test_data_path = os.path.join(
            os.getcwd(), "rtp_llm/test/model_test/fake_test/testdata"
        )
        model_params = GptInitModelParameters(
            head_num=1024,
            size_per_head=1024,
            layer_num=1024,
            max_seq_len=1024,
            vocab_size=1024,
        )
        self.model = FakeModel(None)
        self.model.config = model_params

    async def test_parse_qwen_function_call(self):
        os.environ["MODEL_TYPE"] = "qwen"
        tokenizer = QWenTokenizer(
            f"{self.test_data_path}/qwen_7b/tokenizer/qwen.tiktoken"
        )
        self.model.tokenizer = tokenizer
        self.endpoint = OpenaiEndpoint(self.model.config, self.model.tokenizer, None)
        test_ids = [
            198,
            84169,
            25,
            49434,
            239,
            73670,
            37029,
            633,
            11080,
            69364,
            5333,
            8997,
            2512,
            25,
            633,
            11080,
            69364,
            198,
            2512,
            5571,
            25,
            5212,
            2527,
            788,
            330,
            113074,
            11,
            10236,
            122,
            236,
            28404,
            497,
            330,
            3843,
            788,
            330,
            69,
            47910,
            16707,
        ]
        render_params = RendererParams(
            model_type="qwen",
            max_seq_len=1024,
            eos_token_id=tokenizer.eos_token_id or 0,
            stop_word_ids_list=[],
        )
        chat_renderer = ChatRendererFactory.get_renderer(tokenizer, render_params)
        request = ChatCompletionRequest(
            messages=[ChatMessage(role=RoleEnum.user, content="hello")],
            functions=[
                GPTFunctionDefinition(
                    **{
                        "name": "get_current_weather",
                        "description": "Get the current weather in a given location.",
                        "parameters": {},
                    }
                )
            ],
        )
        id_generator = fake_output_generator(
            test_ids, 1024, tokenizer.eos_token_id or 0, 314
        )
        stream_generator = chat_renderer.render_response_stream(
            id_generator, request, GenerateConfig()
        )
        generate = self.endpoint._complete_stream_response(stream_generator, None)
        response = [x async for x in generate][-1]
        response = await generate.gen_complete_response_once()
        print(response.choices[0].model_dump_json())
        self.assertEqual(1, len(response.choices))
        self.assertEqual(
            json.loads(response.choices[0].model_dump_json()),
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Thought: 我可以使用 get_current_weather API。",
                    "reasoning_content": None,
                    "function_call": {
                        "name": "get_current_weather",
                        "arguments": '{"location": "洛杉矶, 美国", "unit": "fahrenheit"}',
                    },
                    "tool_calls": None,
                    "partial": False,
                },
                "finish_reason": "function_call",
                "logprobs": None,
            },
        )

    async def test_finish_reason(self):
        os.environ["MODEL_TYPE"] = "qwen"
        tokenizer = QWenTokenizer(
            f"{self.test_data_path}/qwen_7b/tokenizer/qwen.tiktoken"
        )
        self.model.tokenizer = tokenizer
        self.endpoint = OpenaiEndpoint(self.model.config, self.model.tokenizer, None)
        test_ids = [198, 84169, 25, 49434, 239, 73670, 37029]
        render_params = RendererParams(
            model_type="qwen",
            max_seq_len=MAX_SEQ_LEN,
            eos_token_id=tokenizer.eos_token_id or 0,
            stop_word_ids_list=[],
        )
        chat_renderer = ChatRendererFactory.get_renderer(tokenizer, render_params)
        request = ChatCompletionRequest(
            messages=[ChatMessage(role=RoleEnum.user, content="hello")]
        )
        input_length = 1018
        id_generator = fake_output_generator(
            test_ids, MAX_SEQ_LEN, tokenizer.eos_token_id or 0, input_length
        )
        stream_generator = chat_renderer.render_response_stream(
            id_generator, request, GenerateConfig()
        )
        generate = self.endpoint._complete_stream_response(stream_generator, None)
        response = [x async for x in generate][-1]
        response = await generate.gen_complete_response_once()
        print(response)
        assert response.choices[0].finish_reason
        self.assertEqual(FinisheReason.length, response.choices[0].finish_reason)

    async def test_parse_qwen_agent_function_call(self):
        os.environ["MODEL_TYPE"] = "qwen_agent"
        tokenizer = QWenTokenizer(
            f"{self.test_data_path}/qwen_7b/tokenizer/qwen.tiktoken"
        )
        self.model.tokenizer = tokenizer
        self.endpoint = OpenaiEndpoint(self.model.config, self.model.tokenizer, None)
        test_ids = [
            25,
            220,
            35946,
            85106,
            47872,
            11622,
            455,
            11080,
            69364,
            5333,
            36407,
            45912,
            104307,
            144575,
            18149,
            144575,
            25,
            633,
            11080,
            69364,
            198,
            144575,
            47483,
            144575,
            25,
            5212,
            2527,
            788,
            330,
            113074,
            11,
            10236,
            122,
            236,
            28404,
            497,
            330,
            3843,
            788,
            330,
            69,
            47910,
            16707,
            144575,
            14098,
            144575,
        ]
        # print(f"===test ids decode {tokenizer.decode(test_ids)}")
        # print(tokenizer.encode("你好啊✿FUNCTION✿: get_current_weather\n✿ARGS✿: {\"location\": \"洛杉矶, 美国\", \"unit\": \"fahrenheit\"}\n✿RESULT✿"))

        render_params = RendererParams(
            model_type="qwen_agent",
            max_seq_len=1024,
            eos_token_id=tokenizer.eos_token_id or 0,
            stop_word_ids_list=[],
        )
        chat_renderer = ChatRendererFactory.get_renderer(tokenizer, render_params)
        # function call 格式返回，输入有functions
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
        request = ChatCompletionRequest(
            messages=[ChatMessage(role=RoleEnum.user, content="hello")],
            functions=functions,
        )
        id_generator = fake_output_generator(
            test_ids, 1024, tokenizer.eos_token_id or 0, 314
        )
        stream_generator = chat_renderer.render_response_stream(
            id_generator, request, GenerateConfig()
        )
        generate = self.endpoint._complete_stream_response(stream_generator, None)
        async for x in generate:
            response = x
            response = await generate.gen_complete_response_once()
            print(response.choices[0].model_dump_json())
        self.assertEqual(1, len(response.choices))
        self.assertEqual(
            json.loads(response.choices[0].model_dump_json()),
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "我需要调用get_current_weather API来获取天气",
                    "reasoning_content": None,
                    "function_call": {
                        "name": "get_current_weather",
                        "arguments": '{"location": "洛杉矶, 美国", "unit": "fahrenheit"}',
                    },
                    "tool_calls": None,
                    "partial": False,
                },
                "finish_reason": "function_call",
                "logprobs": None,
            },
        )
        # 非functioncall 格式返回，输入没有functions
        request = ChatCompletionRequest(
            messages=[ChatMessage(role=RoleEnum.user, content="hello")]
        )
        id_generator = fake_output_generator(
            test_ids, 1024, tokenizer.eos_token_id or 0, 314
        )
        gen_config = GenerateConfig()
        stream_generator = chat_renderer.render_response_stream(
            id_generator, request, gen_config
        )
        generate = self.endpoint._complete_stream_response(stream_generator, None)
        async for x in generate:
            response = x
            response = await generate.gen_complete_response_once()
            print(response.choices[0].model_dump_json())
        self.assertEqual(1, len(response.choices))
        self.assertEqual(
            json.loads(response.choices[0].model_dump_json()),
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": ': 我需要调用get_current_weather API来获取天气✿FUNCTION✿: get_current_weather\n✿ARGS✿: {"location": "洛杉矶, 美国", "unit": "fahrenheit"}',
                    "reasoning_content": None,
                    "function_call": None,
                    "tool_calls": None,
                    "partial": False,
                },
                "finish_reason": "stop",
                "logprobs": None,
            },
        )

    async def test_parse_qwen_agent_tool_call(self):
        os.environ["MODEL_TYPE"] = "qwen_agent_tool"
        tokenizer = QWenTokenizer(
            f"{self.test_data_path}/qwen_7b/tokenizer/qwen.tiktoken"
        )
        self.model.tokenizer = tokenizer
        self.endpoint = OpenaiEndpoint(self.model.config, self.model.tokenizer, None)
        test_ids = [
            25,
            220,
            35946,
            85106,
            47872,
            11622,
            455,
            11080,
            69364,
            5333,
            36407,
            45912,
            104307,
            144575,
            18149,
            144575,
            25,
            633,
            11080,
            69364,
            198,
            144575,
            47483,
            144575,
            25,
            5212,
            2527,
            788,
            330,
            113074,
            11,
            10236,
            122,
            236,
            28404,
            497,
            330,
            3843,
            788,
            330,
            69,
            47910,
            16707,
            144575,
            14098,
            144575,
        ]

        render_params = RendererParams(
            model_type="qwen_agent_tool",
            max_seq_len=1024,
            eos_token_id=tokenizer.eos_token_id or 0,
            stop_word_ids_list=[],
        )
        chat_renderer = ChatRendererFactory.get_renderer(tokenizer, render_params)
        # function call 格式返回，输入有functions
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
        tools = [GPTToolDefinition(function=functions[0])]

        request = ChatCompletionRequest(
            messages=[ChatMessage(role=RoleEnum.user, content="hello")], tools=tools
        )
        id_generator = fake_output_generator(
            test_ids, 1024, tokenizer.eos_token_id or 0, 314
        )
        stream_generator = chat_renderer.render_response_stream(
            id_generator, request, GenerateConfig()
        )
        generate = self.endpoint._complete_stream_response(stream_generator, None)
        # response = [x async for x in generate][-1]
        # response = await generate.gen_complete_response_once()
        # print(response.choices[0].model_dump_json())
        async for x in generate:
            response = x
            response = await generate.gen_complete_response_once()
            print(response.choices[0].model_dump_json())
        self.assertEqual(1, len(response.choices))
        target_delta = json.loads(response.choices[0].model_dump_json())
        target_delta["message"]["tool_calls"][0]["id"] = "id"
        self.assertEqual(
            target_delta,
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "我需要调用get_current_weather API来获取天气",
                    "reasoning_content": None,
                    "function_call": None,
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "id",
                            "type": "function",
                            "function": {
                                "name": "get_current_weather",
                                "arguments": '{"location": "洛杉矶, 美国", "unit": "fahrenheit"}',
                            },
                        }
                    ],
                    "partial": False,
                },
                "finish_reason": "tool_calls",
                "logprobs": None,
            },
        )
        # 非functioncall 格式返回，输入没有functions
        request = ChatCompletionRequest(
            messages=[ChatMessage(role=RoleEnum.user, content="hello")]
        )
        id_generator = fake_output_generator(
            test_ids, 1024, tokenizer.eos_token_id or 0, 314
        )
        gen_config = GenerateConfig()
        stream_generator = chat_renderer.render_response_stream(
            id_generator, request, gen_config
        )
        generate = self.endpoint._complete_stream_response(stream_generator, None)
        async for x in generate:
            response = x
            response = await generate.gen_complete_response_once()
            print(response.choices[0].model_dump_json())
        self.assertEqual(1, len(response.choices))
        self.assertEqual(
            json.loads(response.choices[0].model_dump_json()),
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": ': 我需要调用get_current_weather API来获取天气✿FUNCTION✿: get_current_weather\n✿ARGS✿: {"location": "洛杉矶, 美国", "unit": "fahrenheit"}',
                    "reasoning_content": None,
                    "function_call": None,
                    "tool_calls": None,
                    "partial": False,
                },
                "finish_reason": "stop",
                "logprobs": None,
            },
        )

    class QwenToolTestSuite(BaseToolCallTestSuite):
        """QwenTool相关测试的内嵌测试套件"""

        def _get_model_type(self):
            return "qwen_tool"

        def _get_tokenizer_path(self):
            return "qwen3_30b/tokenizer/"

        def _validate_renderer(self, chat_renderer):
            """验证renderer类型"""
            assert isinstance(chat_renderer, QwenReasoningToolRenderer)

        def _get_test_data(self, include_stop_word=False):
            """获取测试数据"""
            # <think>
            # 好的
            # </think>

            # <tool_call>
            # {"name": "get_current_weather", "arguments": {"location": "杭州"}}
            # </tool_call>
            # <tool_call>
            # {"name": "get_current_weather", "arguments": {"location": "北京"}}
            # </tool_call>

            # 文本内容
            token_ids = [
                151667,
                198,
                99692,
                198,
                151668,
                271,
                108704,
                43815,
                198,
                151657,
                198,
                4913,
                606,
                788,
                330,
                455,
                11080,
                69364,
                497,
                330,
                16370,
                788,
                5212,
                2527,
                788,
                330,
                104130,
                95642,
                151658,
                198,
                151657,
                198,
                4913,
                606,
                788,
                330,
                455,
                11080,
                69364,
                497,
                330,
                16370,
                788,
                5212,
                2527,
                788,
                330,
                68990,
                95642,
                151658,
            ]

            return token_ids

        def _assert_tool_call_response(
            self,
            response_delta,
            expected_content="<think>\n好的\n</think>\n\n文本内容\n",
        ):
            """断言工具调用响应的内容"""
            assert response_delta.content.strip() == expected_content.strip()
            assert response_delta.tool_calls is not None
            assert len(response_delta.tool_calls) == 2
            assert response_delta.tool_calls[0].function.name == "get_current_weather"
            assert (
                response_delta.tool_calls[0].function.arguments
                == '{"location": "杭州"}'
            )
            assert response_delta.tool_calls[0].index == 0
            assert response_delta.tool_calls[1].function.name == "get_current_weather"
            assert (
                response_delta.tool_calls[1].function.arguments
                == '{"location": "北京"}'
            )
            assert response_delta.tool_calls[1].index == 1

    class QwenThinkTestSuite(QwenToolTestSuite):
        """Qwen相关测试的内嵌测试套件, 看看QwenThinkTool能否正常工作"""

        def _get_model_type(self):
            return "qwen_3_moe"

        def _validate_renderer(self, chat_renderer):
            """验证renderer类型"""
            assert isinstance(chat_renderer, QwenReasoningToolRenderer)

        def _assert_tool_call_response(
            self,
            response_delta,
            expected_content="文本内容",
        ):
            """断言工具调用响应的内容"""
            assert response_delta.content.strip() == expected_content.strip()
            assert response_delta.reasoning_content.strip() == "好的"
            assert response_delta.tool_calls is not None
            assert len(response_delta.tool_calls) == 2
            assert response_delta.tool_calls[0].function.name == "get_current_weather"
            assert (
                response_delta.tool_calls[0].function.arguments
                == '{"location": "杭州"}'
            )
            assert response_delta.tool_calls[0].index == 0
            assert response_delta.tool_calls[1].function.name == "get_current_weather"
            assert (
                response_delta.tool_calls[1].function.arguments
                == '{"location": "北京"}'
            )
            assert response_delta.tool_calls[1].index == 1

    class QwenForceThinkTestSuite(QwenThinkTestSuite):
        """Qwen相关测试的内嵌测试套件, 看看QwenThinkTool能否正常工作"""

        def _get_test_data(self, include_stop_word=False):
            test_ids = super()._get_test_data(include_stop_word)
            # 把</tool_call>和<tool_call>之间的\n(198)替换为\n\n(271)
            test_ids[29] = 271
            # 移除开头的<think>和\n
            return test_ids[2:]

        @override
        def _get_tokenizer_path(self):
            return "qwen3_30b_thinking_0527/tokenizer/"

    class KimiK2TestSuite(BaseToolCallTestSuite):
        """KimiK2相关测试的内嵌测试套件"""

        def _get_model_type(self):
            return "kimi_k2"

        def _get_tokenizer_path(self):
            return "kimi_k2/tokenizer/"

        def _get_test_data(self, include_stop_word=False):
            """获取测试数据"""
            test_ids = [
                35659,
                19183,
                13021,
                12365,
                488,
                42930,
                8622,
                8597,
                2267,
                292,
                163595,
                163597,
                41937,
                1150,
                20254,
                21055,
                2800,
                25,
                15,
                163598,
                8264,
                5791,
                1289,
                414,
                12365,
                11,
                220,
                10462,
                16934,
                163599,
                163597,
                41937,
                1150,
                20254,
                21055,
                2800,
                25,
                16,
                163598,
                8264,
                5791,
                1289,
                414,
                3372,
                16934,
                163599,
                163596,
            ]

            if include_stop_word:
                test_ids.append(163586)  # 增加一个<|im_end|>

            return test_ids

        def _validate_renderer(self, chat_renderer):
            """验证renderer类型"""
            assert isinstance(chat_renderer, KimiK2Renderer)

        def _validate_stream_chunk(self, chunk, stream):
            """验证流式chunk"""
            if stream:
                assert is_valid_tool_call_chunk(chunk)

        def _assert_tool_call_response(
            self,
            response_delta,
            expected_content="我来为您查询杭州和北京的当前天气情况。",
        ):
            """断言工具调用响应的内容"""
            assert response_delta.content.strip() == expected_content.strip()
            assert response_delta.tool_calls[0].function.name == "get_current_weather"
            assert (
                response_delta.tool_calls[0].function.arguments
                == '{"location": "杭州, 浙江"}'
            )
            assert response_delta.tool_calls[1].function.name == "get_current_weather"
            assert (
                response_delta.tool_calls[1].function.arguments
                == '{"location": "北京"}'
            )
            assert response_delta.tool_calls[0].index == 0
            assert response_delta.tool_calls[1].index == 1

        async def test_no_stream_stop_words(self):
            """测试KimiK2工具调用非流式场景（包含停止词）"""
            chunk_list = await self._run_tool_call_test(
                stream=False,
                include_stop_word=True,
            )

            # 验证合并结果
            merged_result: ChatCompletionStreamResponse = merge_stream_responses(
                chunk_list
            )
            self._validate_merged_result(merged_result)

    class ChatGLM45TestSuite(BaseToolCallTestSuite):
        """GLM45相关测试的内嵌测试套件"""

        def _get_model_type(self):
            return "glm4_moe"

        def _get_tokenizer_path(self):
            return "glm45/tokenizer/"

        def _get_test_data(self, include_stop_word=False):
            """获取测试数据"""
            # <think>用户询问杭州和北京的天气怎么样。</think>
            # 我来帮您查询杭州和北京的天气情况。
            # <tool_call>get_current_weather
            # <arg_key>location</arg_key>
            # <arg_value>杭州</arg_value>
            # <arg_key>unit</arg_key>
            # <arg_value>celsius</arg_value>
            # </tool_call>
            # <tool_call>get_current_weather
            # <arg_key>location</arg_key>
            # <arg_value>北京</arg_value>
            # <arg_key>unit</arg_key>
            # <arg_value>celsius</arg_value>
            # </tool_call>
            test_ids = [
                198,
                151350,
                99833,
                104678,
                102506,
                98327,
                121828,
                101791,
                103753,
                1773,
                151351,
                198,
                110943,
                99215,
                99526,
                102961,
                102506,
                98327,
                121828,
                101791,
                98962,
                8994,
                151352,
                455,
                11075,
                68852,
                198,
                151356,
                2527,
                151357,
                198,
                151358,
                102506,
                151359,
                198,
                151356,
                3843,
                151357,
                198,
                151358,
                66,
                40026,
                151359,
                198,
                151353,
                198,
                151352,
                455,
                11075,
                68852,
                198,
                151356,
                2527,
                151357,
                198,
                151358,
                99334,
                151359,
                198,
                151356,
                3843,
                151357,
                198,
                151358,
                66,
                40026,
                151359,
                198,
                151353,
            ]

            return test_ids

        def _validate_renderer(self, chat_renderer):
            """验证renderer类型"""
            assert isinstance(chat_renderer, ChatGlm45Renderer)

        def _assert_tool_call_response(
            self,
            response_delta,
            expected_content="我来帮您查询杭州和北京的天气情况。",
        ):
            """断言工具调用响应的内容"""
            assert response_delta.content.strip() == expected_content.strip()
            assert (
                response_delta.reasoning_content.strip()
                == "用户询问杭州和北京的天气怎么样。"
            )
            assert response_delta.tool_calls is not None
            assert len(response_delta.tool_calls) == 2
            assert response_delta.tool_calls[0].function.name == "get_current_weather"
            assert (
                response_delta.tool_calls[0].function.arguments
                == '{"location": "杭州", "unit": "celsius"}'
            )
            assert response_delta.tool_calls[0].index == 0
            assert response_delta.tool_calls[1].function.name == "get_current_weather"
            assert (
                response_delta.tool_calls[1].function.arguments
                == '{"location": "北京", "unit": "celsius"}'
            )
            assert response_delta.tool_calls[1].index == 1

    class Qwen3CoderTestSuite(BaseToolCallTestSuite):
        """Qwen3Coder相关测试的内嵌测试套件"""

        def _get_model_type(self):
            return "qwen3_coder_moe"

        def _get_tokenizer_path(self):
            return "qwen3_coder/tokenizer/"

        def _get_test_data(self, include_stop_word=False):
            """获取测试数据"""
            return [
                151657,
                198,
                27,
                1688,
                28280,
                11080,
                69364,
                397,
                27,
                16181,
                28,
                2527,
                397,
                104130,
                198,
                522,
                16181,
                397,
                27,
                16181,
                28,
                3843,
                397,
                66,
                40247,
                198,
                522,
                16181,
                397,
                522,
                1688,
                397,
                151658,
                198,
                151657,
                198,
                27,
                1688,
                28280,
                11080,
                69364,
                397,
                27,
                16181,
                28,
                2527,
                397,
                68990,
                198,
                522,
                16181,
                397,
                27,
                16181,
                28,
                3843,
                397,
                66,
                40247,
                198,
                522,
                16181,
                397,
                522,
                1688,
                397,
                151658,
            ]

        def _validate_renderer(self, chat_renderer):
            """验证renderer类型"""
            assert isinstance(chat_renderer, Qwen3CoderRenderer)

        def _assert_tool_call_response(self, response_delta, expected_content=""):
            """断言工具调用响应的内容"""
            assert response_delta.content.strip() == expected_content.strip()
            assert response_delta.tool_calls is not None
            assert response_delta.tool_calls[0].index == 0
            assert response_delta.tool_calls[0].function.name == "get_current_weather"
            assert (
                response_delta.tool_calls[0].function.arguments
                == '{"location": "杭州", "unit": "celsius"}'
            )
            assert response_delta.tool_calls[1].index == 1
            assert response_delta.tool_calls[1].function.name == "get_current_weather"
            assert (
                response_delta.tool_calls[1].function.arguments
                == '{"location": "北京", "unit": "celsius"}'
            )

    class Qwen3CoderComplexTestSuite(Qwen3CoderTestSuite):
        @override
        def _get_test_data(self, include_stop_word=False):
            """获取测试数据"""
            return [
                151657,
                198,
                27,
                1688,
                28,
                4934,
                2458,
                397,
                27,
                16181,
                28,
                1796,
                397,
                515,
                220,
                330,
                4439,
                675,
                788,
                330,
                266,
                14493,
                58893,
                73016,
                756,
                220,
                330,
                11525,
                788,
                2278,
                262,
                341,
                414,
                330,
                73340,
                788,
                330,
                80233,
                756,
                414,
                330,
                13871,
                788,
                330,
                874,
                72160,
                69131,
                1327,
                30528,
                15734,
                514,
                20201,
                6847,
                14493,
                25762,
                10076,
                16130,
                3332,
                698,
                262,
                1153,
                262,
                341,
                414,
                330,
                73340,
                788,
                330,
                7423,
                3332,
                756,
                414,
                330,
                13871,
                788,
                330,
                874,
                72160,
                69131,
                1327,
                30528,
                15734,
                514,
                20201,
                6847,
                14493,
                25762,
                10076,
                29012,
                3332,
                698,
                262,
                456,
                220,
                3211,
                220,
                330,
                16900,
                788,
                2278,
                262,
                341,
                414,
                330,
                1499,
                788,
                330,
                80233,
                756,
                414,
                330,
                983,
                788,
                330,
                7423,
                3332,
                698,
                262,
                456,
                220,
                5133,
                532,
                522,
                16181,
                397,
                27,
                16181,
                59245,
                2638,
                397,
                97821,
                6324,
                44977,
                93381,
                12697,
                28783,
                14493,
                36464,
                13437,
                15351,
                38900,
                72177,
                59150,
                28989,
                4323,
                198,
                522,
                16181,
                397,
                522,
                1688,
                397,
                151658,
            ]

        @override
        def _create_test_functions_and_tools(self):
            """创建测试用的函数和工具定义 - 子类可以重写"""
            functions = [
                GPTFunctionDefinition(
                    **{
                        "name": "write_file",
                        "description": "Writes content to a specified file in the local filesystem. \n      \n      The user has the ability to modify `content`. If modified, this will be stated in the response.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "file_path": {
                                    "type": "string",
                                    "description": "The absolute path to the file to write to (e.g., '/home/user/project/file.txt'). Relative paths are not supported.",
                                },
                                "content": {
                                    "type": "string",
                                    "description": "The content to write to the file.",
                                },
                            },
                            "required": ["file_path", "content"],
                        },
                    }
                )
            ]
            tools = [GPTToolDefinition(function=functions[0])]
            return functions, tools

        @override
        def _assert_tool_call_response(self, response_delta, expected_content=""):
            """断言工具调用响应的内容"""
            assert response_delta.tool_calls is not None
            assert response_delta.tool_calls[0].index == 0
            assert response_delta.tool_calls[0].function.name == "write_file"
            assert (
                response_delta.tool_calls[0].function.arguments
                == '{"content": "{\\n  \\"graphName\\": \\"atlas-demo-graph\\",\\n  \\"modules\\": [\\n    {\\n      \\"moduleName\\": \\"InputModule\\",\\n      \\"className\\": \\"com.taobao.recommendplatform.solutions.atlasdemo.module.InputModule\\"\\n    },\\n    {\\n      \\"moduleName\\": \\"ProcessModule\\",\\n      \\"className\\": \\"com.taobao.recommendplatform.solutions.atlasdemo.module.ProcessModule\\"\\n    }\\n  ],\\n  \\"edges\\": [\\n    {\\n      \\"from\\": \\"InputModule\\",\\n      \\"to\\": \\"ProcessModule\\"\\n    }\\n  ]\\n}", "file_path": "/Users/wuchen/workspace/test-atlas-gen/src/main/resources/graph_configs/default.json"}'
            )

    # 使用基类的测试方法
    async def test_parse_qwen_tool_call_streaming_case(self):
        """测试QwenTool工具调用流式场景"""
        qwen_suite = self.QwenToolTestSuite(self)
        await qwen_suite.test_streaming_case()

    async def test_parse_qwen_tool_call_no_stream(self):
        """测试QwenTool工具调用非流式场景"""
        qwen_suite = self.QwenToolTestSuite(self)
        await qwen_suite.test_no_stream()

    @think_mode
    async def test_parse_qwen_think_call_streaming_case(self):
        """测试QwenTool工具思考调用流式场景"""
        qwen_suite = self.QwenThinkTestSuite(self)
        await qwen_suite.test_streaming_case(stop_words_str=["<im_end>"])

    @think_mode
    async def test_parse_qwen_think_call_no_stream(self):
        """测试QwenTool工具思考调用非流式场景"""
        qwen_suite = self.QwenThinkTestSuite(self)
        await qwen_suite.test_no_stream(stop_words_str=["<im_end>"])

    @think_mode
    async def test_parse_qwen_force_think_call_streaming_case(self):
        """测试QwenTool工具思考调用流式场景"""
        qwen_suite = self.QwenForceThinkTestSuite(self)
        await qwen_suite.test_streaming_case(stop_words_str=["<im_end>"])

    @think_mode
    async def test_parse_qwen_force_think_call_no_stream(self):
        """测试QwenTool工具思考调用非流式场景"""
        qwen_suite = self.QwenForceThinkTestSuite(self)
        await qwen_suite.test_no_stream(stop_words_str=["<im_end>"])

    async def test_parse_kimik2_tool_call_streaming_case(self):
        """测试KimiK2工具调用流式场景"""
        kimi_suite = self.KimiK2TestSuite(self)
        await kimi_suite.test_streaming_case()

    async def test_parse_kimik2_tool_call_no_stream(self):
        """测试KimiK2工具调用非流式场景"""
        kimi_suite = self.KimiK2TestSuite(self)
        await kimi_suite.test_no_stream()

    async def test_parse_kimik2_tool_call_no_stream_stop_words(self):
        """测试KimiK2工具调用非流式场景（包含停止词）"""
        kimi_suite = self.KimiK2TestSuite(self)
        await kimi_suite.test_no_stream_stop_words()

    @think_mode
    async def test_parse_chatglm45_tool_call_streaming_case(self):
        suite = self.ChatGLM45TestSuite(self)
        await suite.test_streaming_case()

    @think_mode
    async def test_parse_chatglm45_tool_call_no_stream(self):
        suite = self.ChatGLM45TestSuite(self)
        await suite.test_no_stream()

    async def test_parse_qwen3_coder_tool_call_streaming_case(self):
        """测试Qwen3Coder工具调用流式场景"""
        suite = self.Qwen3CoderTestSuite(self)
        await suite.test_streaming_case()

    async def test_parse_qwen3_coder_tool_call_no_stream(self):
        """测试Qwen3Coder工具调用非流式场景"""
        suite = self.Qwen3CoderTestSuite(self)
        await suite.test_no_stream()

    async def test_parse_qwen3_coder_complex_tool_call_streaming_case(self):
        """测试Qwen3Coder工具调用流式场景"""
        suite = self.Qwen3CoderComplexTestSuite(self)
        await suite.test_streaming_case()

    async def test_parse_qwen3_coder_complex_tool_call_no_stream(self):
        """测试Qwen3Coder工具调用非流式场景"""
        suite = self.Qwen3CoderComplexTestSuite(self)
        await suite.test_no_stream()

    class QwenMergeLogicTestSuite(QwenToolTestSuite):
        """基于QwenTestSuite，专门测试generate_choice合并逻辑"""

        def _create_two_step_mock_model_rpc_client(self, tokenizer, test_ids, seq_len):
            """创建模拟的ModelRpcClient - 分两步返回"""

            class MockModelRpcClient:
                def __init__(self, test_ids, max_seq_len, eos_token_id, seq_len):
                    self.test_ids = test_ids
                    self.max_seq_len = max_seq_len
                    self.eos_token_id = eos_token_id
                    self.seq_len = seq_len
                    self.generated_outputs_count = 0

                async def enqueue(
                    self, input: GenerateInput
                ) -> AsyncGenerator[GenerateOutputs, None]:
                    """模拟ModelRpcClient.enqueue - 分两步返回"""

                    # 第一步：返回第一个token
                    self.generated_outputs_count += 1
                    first_token = [self.test_ids[0]]

                    output_tensor1 = torch.full(
                        (1, self.max_seq_len), self.eos_token_id, dtype=torch.int
                    )
                    output_tensor1[0, : len(first_token)] = torch.tensor(
                        first_token, dtype=torch.int
                    )

                    outputs1 = GenerateOutputs()
                    aux1 = AuxInfo()
                    aux1.input_len = self.seq_len
                    aux1.output_len = len(first_token)
                    aux1.reuse_len = 0

                    outputs1.generate_outputs.append(
                        GenerateOutput(
                            hidden_states=None,
                            output_ids=output_tensor1,
                            finished=torch.full((1,), False, dtype=torch.bool),
                            aux_info=aux1,
                            loss=None,
                            logits=None,
                        )
                    )

                    yield outputs1

                    # 第二步：返回剩余的所有token
                    self.generated_outputs_count += 1
                    remaining_tokens = self.test_ids[1:]

                    output_tensor2 = torch.full(
                        (1, self.max_seq_len), self.eos_token_id, dtype=torch.int
                    )
                    output_tensor2[0, : len(remaining_tokens)] = torch.tensor(
                        remaining_tokens, dtype=torch.int
                    )

                    outputs2 = GenerateOutputs()
                    aux2 = AuxInfo()
                    aux2.input_len = self.seq_len
                    aux2.output_len = len(remaining_tokens)
                    aux2.reuse_len = 0

                    outputs2.generate_outputs.append(
                        GenerateOutput(
                            hidden_states=None,
                            output_ids=output_tensor2,
                            finished=torch.full((1,), True, dtype=torch.bool),
                            aux_info=aux2,
                            loss=None,
                            logits=None,
                        )
                    )

                    yield outputs2

            return MockModelRpcClient(
                test_ids, 1024, tokenizer.eos_token_id or 0, seq_len
            )

        def _create_mock_backend_visitor_with_mock_rpc_client(
            self, tokenizer, test_ids, seq_len
        ):
            """创建使用mock ModelRpcClient的BackendRPCServerVisitor"""
            mock_rpc_client = self._create_two_step_mock_model_rpc_client(
                tokenizer, test_ids, seq_len
            )

            class MockBackendRPCServerVisitor:
                def __init__(self, mock_rpc_client):
                    self.model_rpc_client = mock_rpc_client
                    self.enqueue_call_count = 0

                async def enqueue(
                    self, input: GenerateInput
                ) -> AsyncGenerator[GenerateOutputs, None]:
                    """模拟BackendRPCServerVisitor.enqueue的行为 - 直接返回generator"""
                    self.enqueue_call_count += 1
                    return self.model_rpc_client.enqueue(input)

            return MockBackendRPCServerVisitor(mock_rpc_client), mock_rpc_client

        async def test_non_streaming_merge_logic(self):
            """测试非流式场景的合并逻辑"""
            tokenizer = self._setup_environment()
            test_ids = self._get_test_data()
            render_params = self._create_render_params(tokenizer)

            chat_renderer = ChatRendererFactory.get_renderer(tokenizer, render_params)
            self._validate_renderer(chat_renderer)

            functions, tools = self._create_test_functions_and_tools()

            mock_visitor, mock_rpc_client = (
                self._create_mock_backend_visitor_with_mock_rpc_client(
                    tokenizer, test_ids, 314
                )
            )

            request = ChatCompletionRequest(
                messages=[ChatMessage(role=RoleEnum.user, content="hello")],
                tools=tools,
                stream=False,
            )
            generate_config = GenerateConfig(is_streaming=False)

            choice_generator = chat_renderer.generate_choice(
                request_id=123,
                input_ids=[1, 2, 3],
                mm_inputs=[],
                generate_config=generate_config,
                backend_rpc_server_visitor=mock_visitor,
                request=request,
            )

            stream_response_objects = []
            async for stream_response_obj in choice_generator:
                stream_response_objects.append(stream_response_obj)

            # 验证合并逻辑
            self.parent.assertEqual(
                mock_rpc_client.generated_outputs_count,
                2,
                "ModelRpcClient应该返回2个输出",
            )
            # stream_response_objects应该额外包含generate_first, flush_buffer, generate_final 3个chunk, 共4个
            self.parent.assertEqual(
                len(stream_response_objects), 4, "generate_choice应该合并输出"
            )

            # 验证工具调用结果
            async def stream_response_objects_to_generator():
                for stream_obj in stream_response_objects:
                    yield stream_obj

            generate = self.parent.endpoint._complete_stream_response(
                stream_response_objects_to_generator(), None
            )
            chunk_list = [chunk async for chunk in generate]

            from rtp_llm.test.utils.stream_util import merge_stream_responses

            merged_result = merge_stream_responses(chunk_list)

            self._validate_merged_result(merged_result)

    async def test_qwen_non_streaming_merge_logic(self):
        """测试Qwen模型非流式场景的合并逻辑"""
        qwen_suite = self.QwenMergeLogicTestSuite(self)
        await qwen_suite.test_non_streaming_merge_logic()

    def test_chatglm_stop_word(self):
        os.environ["MODEL_TYPE"] = "chatglm3"
        self.model.config.py_env_configs.model_config.model_type = "chatglm3"
        tokenizer = ChatGLMTokenizer.from_pretrained(
            "rtp_llm/test/tokenizer_test/testdata/chatglm3_tokenizer",
            encode_special_tokens=True,
        )
        self.model.tokenizer = tokenizer
        self.endpoint = OpenaiEndpoint(self.model.config, self.model.tokenizer, None)
        self.assertEqual(self.endpoint.stop_words_id_list, [[64795], [64797], [2]])
        self.assertEqual(
            self.endpoint.stop_words_str_list, ["<|user|>", "<|observation|>"]
        )

    @think_mode
    async def test_think_label(self):
        custom_renderer.THINK_END_TAG = "ulaire"  # id = 73675
        os.environ["MODEL_TYPE"] = "qwen"
        tokenizer = QWenTokenizer(
            f"{self.test_data_path}/qwen_7b/tokenizer/qwen.tiktoken"
        )
        self.model.tokenizer = tokenizer
        self.endpoint = OpenaiEndpoint(self.model.config, self.model.tokenizer, None)

        test_ids = [35946, 73670, 73670, 73670, 73675, 35946, 37029, 37029, 37029]
        render_params = RendererParams(
            model_type="qwen",
            max_seq_len=MAX_SEQ_LEN,
            eos_token_id=tokenizer.eos_token_id or 0,
            stop_word_ids_list=[],
        )
        chat_renderer = ChatRendererFactory.get_renderer(tokenizer, render_params)
        request = ChatCompletionRequest(
            messages=[ChatMessage(role=RoleEnum.user, content="hello")]
        )
        input_length = 109
        id_generator = fake_output_generator(
            test_ids, MAX_SEQ_LEN, tokenizer.eos_token_id or 0, input_length
        )
        stream_generator = chat_renderer.render_response_stream(
            id_generator, request, GenerateConfig()
        )
        generate = self.endpoint._complete_stream_response(stream_generator, None)
        # response = [x async for x in generate][-1]
        response = [x async for x in generate][-1]
        response = await generate.gen_complete_response_once()
        print(response.choices[0].model_dump_json())
        self.assertEqual(1, len(response.choices))
        self.assertEqual(
            json.loads(response.choices[0].model_dump_json()),
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "我使用使用使用",
                    "reasoning_content": "我可以可以可以",
                    "function_call": None,
                    "tool_calls": None,
                    "partial": False,
                },
                "finish_reason": "stop",
                "logprobs": None,
            },
        )
        self.assertEqual(
            json.loads(response.usage.completion_tokens_details.model_dump_json()),
            {"audio_tokens": None, "reasoning_tokens": 5},
        )

    @think_mode
    async def test_think_label_more_than_one_token(self):
        custom_renderer.THINK_START_TAG = "我可以"
        custom_renderer.THINK_END_TAG = "可以ulaire"
        os.environ["MODEL_TYPE"] = "qwen"
        tokenizer = QWenTokenizer(
            f"{self.test_data_path}/qwen_7b/tokenizer/qwen.tiktoken"
        )
        self.model.tokenizer = tokenizer
        self.endpoint = OpenaiEndpoint(self.model.config, self.model.tokenizer, None)

        test_ids = [35946, 73670, 73670, 73670, 73675, 35946, 37029, 37029, 37029]
        render_params = RendererParams(
            model_type="qwen",
            max_seq_len=MAX_SEQ_LEN,
            eos_token_id=tokenizer.eos_token_id or 0,
            stop_word_ids_list=[],
        )
        chat_renderer = ChatRendererFactory.get_renderer(tokenizer, render_params)
        request = ChatCompletionRequest(
            messages=[ChatMessage(role=RoleEnum.user, content="hello")]
        )
        input_length = 109
        id_generator = fake_output_generator(
            test_ids, MAX_SEQ_LEN, tokenizer.eos_token_id or 0, input_length
        )
        stream_generator = chat_renderer.render_response_stream(
            id_generator, request, GenerateConfig()
        )
        generate = self.endpoint._complete_stream_response(stream_generator, None)
        # response = [x async for x in generate][-1]
        response = [x async for x in generate][-1]
        response = await generate.gen_complete_response_once()
        print(response.choices[0].model_dump_json())
        self.assertEqual(1, len(response.choices))
        self.assertEqual(
            json.loads(response.choices[0].model_dump_json()),
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "我使用使用使用",
                    "reasoning_content": "可以",
                    "function_call": None,
                    "tool_calls": None,
                    "partial": False,
                },
                "finish_reason": "stop",
                "logprobs": None,
            },
        )
        self.assertEqual(
            json.loads(response.usage.completion_tokens_details.model_dump_json()),
            {"audio_tokens": None, "reasoning_tokens": 5},
        )

    @think_mode
    async def test_think_label_real_situation_union(self):
        custom_renderer.THINK_START_TAG = "<think>\n"
        custom_renderer.THINK_END_TAG = "</think>\n"
        os.environ["MODEL_TYPE"] = "qwen_2"
        tokenizer = AutoTokenizer.from_pretrained(
            f"{self.test_data_path}/deepseek_r1_qwen_14b_tokenizer"
        )
        self.model.tokenizer = tokenizer
        self.endpoint = OpenaiEndpoint(self.model.config, self.model.tokenizer, None)

        test_ids = [151648, 198, 73670, 73670, 73670, 151649, 271, 37029, 37029, 37029]
        render_params = RendererParams(
            model_type="qwen",
            max_seq_len=MAX_SEQ_LEN,
            eos_token_id=tokenizer.eos_token_id or 0,
            stop_word_ids_list=[],
        )
        chat_renderer = ChatRendererFactory.get_renderer(tokenizer, render_params)
        request = ChatCompletionRequest(
            messages=[ChatMessage(role=RoleEnum.user, content="hello")]
        )
        input_length = 109
        id_generator = fake_output_generator(
            test_ids, MAX_SEQ_LEN, tokenizer.eos_token_id or 0, input_length
        )
        stream_generator = chat_renderer.render_response_stream(
            id_generator, request, GenerateConfig()
        )
        generate = self.endpoint._complete_stream_response(stream_generator, None)
        # response = [x async for x in generate][-1]
        async for x in generate:
            pass
        response = await generate.gen_complete_response_once()
        print(response.choices[0].model_dump_json())
        self.assertEqual(1, len(response.choices))
        self.assertEqual(
            json.loads(response.choices[0].model_dump_json()),
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "\n使用使用使用",
                    "reasoning_content": "\n可以可以可以",
                    "function_call": None,
                    "tool_calls": None,
                    "partial": False,
                },
                "finish_reason": "stop",
                "logprobs": None,
            },
        )
        self.assertEqual(
            json.loads(response.usage.completion_tokens_details.model_dump_json()),
            {"audio_tokens": None, "reasoning_tokens": 6},
        )

    async def test_escape(self):
        think_start_tag = "<think>\n"
        self.assertEqual(
            think_start_tag, think_start_tag.encode("utf-8").decode("unicode_escape")
        )
        think_end_tag = "</think>\n\n"
        self.assertEqual(
            think_end_tag, think_end_tag.encode("utf-8").decode("unicode_escape")
        )
        think_start_tag_from_env = "<think>\\n"
        self.assertEqual(
            think_start_tag,
            think_start_tag_from_env.encode("utf-8").decode("unicode_escape"),
        )
        self.assertNotEqual(think_start_tag, think_start_tag_from_env)
        think_end_tag_from_env = "</think>\\n\\n"
        self.assertEqual(
            think_end_tag,
            think_end_tag_from_env.encode("utf-8").decode("unicode_escape"),
        )
        self.assertNotEqual(think_end_tag, think_end_tag_from_env)


if __name__ == "__main__":
    main()
