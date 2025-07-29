import json
import logging
import os
from typing import Any, AsyncGenerator, List
from unittest import IsolatedAsyncioTestCase, TestCase, main

import torch
from transformers import AutoTokenizer, PreTrainedTokenizer

from rtp_llm.config.generate_config import GenerateConfig
from rtp_llm.config.gpt_init_model_parameters import GptInitModelParameters
from rtp_llm.models.base_model import (
    AuxInfo,
    BaseModel,
    GenerateOutput,
    GenerateOutputs,
)
from rtp_llm.openai.api_datatype import (
    ChatCompletionRequest,
    ChatCompletionStreamResponse,
    FinisheReason,
    GPTFunctionDefinition,
    GPTToolDefinition,
    RoleEnum,
)
from rtp_llm.openai.openai_endpoint import OpenaiEndpoint
from rtp_llm.openai.renderer_factory import ChatRendererFactory, RendererParams
from rtp_llm.openai.renderers import custom_renderer
from rtp_llm.openai.renderers.kimik2_renderer import KimiK2Renderer
from rtp_llm.openai.renderers.qwen3_code_renderer import Qwen3CoderRenderer
from rtp_llm.openai.renderers.qwen_renderer import QwenRenderer
from rtp_llm.openai.renderers.qwen_tool_renderer import QwenToolRenderer
from rtp_llm.test.utils.stream_util import (
    is_valid_tool_call_chunk,
    merge_stream_responses,
)
from rtp_llm.tokenizer.tokenization_chatglm3 import ChatGLMTokenizer
from rtp_llm.tokenizer.tokenization_qwen import QWenTokenizer


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


async def fake_output_generator_two_steps(
    output_ids: List[int], max_seq_len: int, eos_id: int, seq_len: int
) -> AsyncGenerator[GenerateOutputs, None]:
    """
    为了兼容目前PD分离场景, 目标token分两次生成的case
    分两次返回结果：
    第一次：返回第一个token
    第二次：返回剩余的所有tokens
    """
    if not output_ids:
        return

    # 第一次返回：只包含第一个token
    first_token = [output_ids[0]]
    output_tensor = torch.full((1, max_seq_len), eos_id, dtype=torch.int)
    output_tensor[0, : len(first_token)] = torch.tensor(first_token, dtype=torch.int)

    finished = torch.full((1,), False, dtype=torch.bool)  # 未完成

    outputs = GenerateOutputs()
    aux = AuxInfo()
    aux.input_len = seq_len
    aux.output_len = 1

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

    # 第二次返回：只包含剩余的tokens（从第二个开始）
    remaining_tokens = output_ids[1:]
    output_tensor = torch.full((1, max_seq_len), eos_id, dtype=torch.int)
    output_tensor[0, : len(remaining_tokens)] = torch.tensor(
        remaining_tokens, dtype=torch.int
    )

    finished = torch.full((1,), True, dtype=torch.bool)  # 第二次返回后完成
    outputs = GenerateOutputs()
    aux = AuxInfo()
    aux.input_len = seq_len
    aux.output_len = len(remaining_tokens)

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

    def _get_target_chunk_index(self, generator_type, pd_separation, chunk_list):
        """获取目标chunk的索引 - 子类可以重写"""
        if generator_type == "once":
            return 1 if len(chunk_list) > 1 else 0
        elif generator_type == "two_steps" and pd_separation:
            return 2 if len(chunk_list) > 2 else -1
        else:
            return 1 if len(chunk_list) > 1 else 0

    def _validate_renderer(self, chat_renderer):
        """验证renderer类型 - 子类可以重写"""
        pass

    def _validate_stream_chunk(self, chunk, stream):
        """验证流式chunk - 子类可以重写"""
        pass

    def _setup_environment(self, pd_separation=None):
        """设置测试环境"""
        os.environ["MODEL_TYPE"] = self._get_model_type()

        # 设置额外的环境变量
        self._setup_additional_environment()

        # 设置PD_SEPARATION环境变量
        if pd_separation:
            os.environ["PD_SEPARATION"] = "1"
        elif "PD_SEPARATION" in os.environ:
            del os.environ["PD_SEPARATION"]

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
        generator_type="normal",
        pd_separation=None,
        include_stop_word=False,
    ):
        """运行工具调用测试的通用方法"""
        tokenizer = self._setup_environment(pd_separation)
        test_ids = self._get_test_data(include_stop_word)
        render_params = self._create_render_params(tokenizer)

        chat_renderer = ChatRendererFactory.get_renderer(tokenizer, render_params)

        # 子类特定的renderer验证
        self._validate_renderer(chat_renderer)

        functions, tools = self._create_test_functions_and_tools()
        request = ChatCompletionRequest(messages=[], tools=tools, stream=stream)

        seq_len_no_use = 314

        # 根据generator_type选择不同的生成器
        if generator_type == "once":
            id_generator = fake_output_generator_once(
                test_ids, MAX_SEQ_LEN, tokenizer.eos_token_id or 0, seq_len_no_use
            )
        elif generator_type == "two_steps":
            id_generator = fake_output_generator_two_steps(
                test_ids, MAX_SEQ_LEN, tokenizer.eos_token_id or 0, seq_len_no_use
            )
        else:  # normal
            id_generator = fake_output_generator(
                test_ids, MAX_SEQ_LEN, tokenizer.eos_token_id or 0, seq_len_no_use
            )

        stream_generator = chat_renderer.render_response_stream(
            id_generator, request, GenerateConfig(is_streaming=stream)
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

    async def test_streaming_case(self):
        """测试工具调用流式场景"""
        chunk_list = await self._run_tool_call_test(stream=True, pd_separation=False)

        merged_result: ChatCompletionStreamResponse = merge_stream_responses(chunk_list)
        self._validate_merged_result(merged_result)

    async def test_no_stream(self):
        """测试工具调用非流式场景"""
        chunk_list = await self._run_tool_call_test(
            stream=False, generator_type="once", pd_separation=False
        )

        # 获取目标chunk并验证
        target_index = self._get_target_chunk_index("once", False, chunk_list)
        target_chunk = (
            chunk_list[target_index]
            if target_index < len(chunk_list)
            else chunk_list[-1]
        )
        target_delta = target_chunk.choices[0].delta
        self._assert_tool_call_response(target_delta)

        # 验证合并结果
        merged_result: ChatCompletionStreamResponse = merge_stream_responses(chunk_list)
        self._validate_merged_result(merged_result)

    async def test_no_stream_pd_separate(self):
        """测试工具调用非流式PD分离场景"""
        chunk_list = await self._run_tool_call_test(
            stream=False, generator_type="two_steps", pd_separation=True
        )

        # 获取目标chunk并验证
        target_index = self._get_target_chunk_index("two_steps", True, chunk_list)
        target_chunk = (
            chunk_list[target_index]
            if target_index < len(chunk_list)
            else chunk_list[-1]
        )
        target_delta = target_chunk.choices[0].delta
        self._assert_tool_call_response(target_delta)

        # 验证合并结果
        merged_result: ChatCompletionStreamResponse = merge_stream_responses(chunk_list)
        self._validate_merged_result(merged_result)

    def cleanup_environment(self):
        """清理测试环境"""
        if "PD_SEPARATION" in os.environ:
            del os.environ["PD_SEPARATION"]


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
            messages=[],
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
        request = ChatCompletionRequest(messages=[])
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
        request = ChatCompletionRequest(messages=[], functions=functions)
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
        request = ChatCompletionRequest(messages=[])
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

        request = ChatCompletionRequest(messages=[], tools=tools)
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
        request = ChatCompletionRequest(messages=[])
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
            assert isinstance(chat_renderer, QwenToolRenderer)

        def _get_test_data(self, include_stop_word=False):
            """获取测试数据"""
            return [
                151667,
                198,
                99692,
                3837,
                20002,
                56007,
                100146,
                104130,
                33108,
                68990,
                9370,
                104307,
                104472,
                8997,
                151668,
                271,
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

        def _assert_tool_call_response(
            self,
            response_delta,
            expected_content="<think>\n好的，用户问的是杭州和北京的天气怎么样。\n</think>",
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

    class QwenTestSuite(QwenToolTestSuite):
        """Qwen相关测试的内嵌测试套件, 看看QwenTool和Qwen是否一样正常工作"""

        def _get_model_type(self):
            return "qwen_3"

        def _validate_renderer(self, chat_renderer):
            """验证renderer类型"""
            assert isinstance(chat_renderer, QwenRenderer)

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

        async def test_no_stream_pd_separate_stop_words(self):
            """测试KimiK2工具调用非流式PD分离场景（包含停止词）"""
            chunk_list = await self._run_tool_call_test(
                stream=False,
                generator_type="two_steps",
                pd_separation=True,
                include_stop_word=True,
            )

            # PD分离场景下，目标chunk是第3个
            target_chunk = chunk_list[2]
            target_delta = target_chunk.choices[0].delta
            self._assert_tool_call_response(target_delta)

            # 验证合并结果
            merged_result: ChatCompletionStreamResponse = merge_stream_responses(
                chunk_list
            )
            self._validate_merged_result(merged_result)

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

    # 使用基类的测试方法
    async def test_parse_qwen_tool_call_streaming_case(self):
        """测试QwenTool工具调用流式场景"""
        qwen_suite = self.QwenToolTestSuite(self)
        await qwen_suite.test_streaming_case()

    async def test_parse_qwen_tool_call_no_stream(self):
        """测试QwenTool工具调用非流式场景"""
        qwen_suite = self.QwenToolTestSuite(self)
        await qwen_suite.test_no_stream()

    async def test_parse_qwen_tool_call_no_stream_PDseperate(self):
        """测试QwenTool工具调用非流式PD分离场景"""
        qwen_suite = self.QwenToolTestSuite(self)
        await qwen_suite.test_no_stream_pd_separate()

    async def test_parse_qwen_call_streaming_case(self):
        """测试QwenTool工具调用流式场景"""
        qwen_suite = self.QwenTestSuite(self)
        await qwen_suite.test_streaming_case()

    async def test_parse_qwen_call_no_stream(self):
        """测试QwenTool工具调用非流式场景"""
        qwen_suite = self.QwenTestSuite(self)
        await qwen_suite.test_no_stream()

    async def test_parse_qwen_call_no_stream_PDseperate(self):
        """测试QwenTool工具调用非流式PD分离场景"""
        qwen_suite = self.QwenTestSuite(self)
        await qwen_suite.test_no_stream_pd_separate()

    async def test_parse_kimik2_tool_call_streaming_case(self):
        """测试KimiK2工具调用流式场景"""
        kimi_suite = self.KimiK2TestSuite(self)
        await kimi_suite.test_streaming_case()

    async def test_parse_kimik2_tool_call_no_stream(self):
        """测试KimiK2工具调用非流式场景"""
        kimi_suite = self.KimiK2TestSuite(self)
        await kimi_suite.test_no_stream()

    async def test_parse_kimik2_tool_call_no_stream_PDseperate(self):
        """测试KimiK2工具调用非流式PD分离场景"""
        kimi_suite = self.KimiK2TestSuite(self)
        await kimi_suite.test_no_stream_pd_separate()

    async def test_parse_kimik2_tool_call_no_stream_PDseperate_stop_words(self):
        """测试KimiK2工具调用非流式PD分离场景（包含停止词）"""
        kimi_suite = self.KimiK2TestSuite(self)
        await kimi_suite.test_no_stream_pd_separate_stop_words()

    async def test_parse_qwen3_coder_tool_call_streaming_case(self):
        """测试Qwen3Coder工具调用流式场景"""
        suite = self.Qwen3CoderTestSuite(self)
        await suite.test_streaming_case()

    async def test_parse_qwen3_coder_tool_call_no_stream(self):
        """测试Qwen3Coder工具调用非流式场景"""
        suite = self.Qwen3CoderTestSuite(self)
        await suite.test_no_stream()

    async def test_parse_qwen3_coder_tool_call_no_stream_PDseperate(self):
        """测试Qwen3Coder工具调用非流式PD分离场景"""
        suite = self.Qwen3CoderTestSuite(self)
        await suite.test_no_stream_pd_separate()

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

    async def test_think_label(self):
        custom_renderer.THINK_MODE = 1
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
        request = ChatCompletionRequest(messages=[])
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

    async def test_think_label_more_than_one_token(self):
        custom_renderer.THINK_MODE = 1
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
        request = ChatCompletionRequest(messages=[])
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

    async def test_think_label_real_situation_union(self):
        custom_renderer.THINK_MODE = 1
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
        request = ChatCompletionRequest(messages=[])
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
                    "reasoning_content": "可以可以可以",
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
