import asyncio
import copy
import functools
import json
import os
from contextlib import asynccontextmanager, contextmanager
from typing import Any, AsyncGenerator, Callable, List
from unittest import IsolatedAsyncioTestCase, main

import torch
from typing_extensions import override

from rtp_llm.config.generate_config import GenerateConfig
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.py_config_modules import (
    GenerateEnvConfig,
    ModelArgs,
    PyEnvConfigs,
    PyMiscellaneousConfig,
    RenderConfig,
    VitConfig,
)
from rtp_llm.frontend.tokenizer_factory.tokenizer_factory import TokenizerFactory
from rtp_llm.frontend.tokenizer_factory.tokenizers import BaseTokenizer
from rtp_llm.frontend.tokenizer_factory.tokenizers.tokenization_qwen import (
    QWenTokenizer,
)
from rtp_llm.models.base_model import BaseModel
from rtp_llm.openai.api_datatype import (
    ChatCompletionExtraOutputs,
    ChatCompletionRequest,
    ChatCompletionStreamResponse,
    ChatMessage,
    DebugInfo,
    FinisheReason,
    GPTFunctionDefinition,
    GPTToolDefinition,
    RoleEnum,
)
from rtp_llm.openai.openai_endpoint import OpenaiEndpoint
from rtp_llm.openai.renderer_factory import ChatRendererFactory, RendererParams
from rtp_llm.openai.renderers import custom_renderer
from rtp_llm.openai.renderers.chatglm45_renderer import ChatGlm45Renderer
from rtp_llm.openai.renderers.deepseekv31_renderer import DeepseekV31Renderer
from rtp_llm.openai.renderers.kimik2_renderer import KimiK2Renderer
from rtp_llm.openai.renderers.qwen3_code_renderer import Qwen3CoderRenderer
from rtp_llm.openai.renderers.qwen_reasoning_tool_renderer import (
    QwenReasoningToolRenderer,
)
from rtp_llm.ops import FfnDisAggregateConfig, PDSepConfig, SpecialTokens
from rtp_llm.server.backend_rpc_server_visitor import BackendRPCServerVisitor
from rtp_llm.test.utils.stream_util import (
    is_valid_tool_call_chunk,
    merge_stream_responses,
)
from rtp_llm.utils.base_model_datatypes import (
    AuxInfo,
    GenerateInput,
    GenerateOutput,
    GenerateOutputs,
)


async def fake_output_generator(
    output_ids: List[int],
    max_seq_len: int,
    eos_id: int,
    seq_len: int,
    output_gen: Callable[..., GenerateOutput] = GenerateOutput,
) -> AsyncGenerator[GenerateOutputs, None]:
    # 流式的返回结果
    for i in range(0, len(output_ids)):
        output_tensor = torch.full((1, 1), eos_id, dtype=torch.int)

        output_tensor[0, 0] = torch.tensor(output_ids[i : i + 1], dtype=torch.int)
        finished = torch.full((1,), (i == (len(output_ids) - 1)), dtype=torch.bool)
        outputs = GenerateOutputs()
        aux = AuxInfo()
        aux.input_len = seq_len
        aux.output_len = i + 1
        outputs.generate_outputs.append(
            output_gen(
                hidden_states=None,
                output_ids=output_tensor,
                finished=finished,
                aux_info=aux,
                loss=None,
                logits=None,
            )
        )
        yield outputs


async def fake_output_generator_mtp(
    output_ids: List[int],
    max_seq_len: int,
    eos_id: int,
    seq_len: int,
    tokens_per_chunk: int = 3,
    output_gen: Callable[..., GenerateOutput] = GenerateOutput,
) -> AsyncGenerator[GenerateOutputs, None]:
    """
    Simulates MTP (Multiple Token Prediction) where multiple tokens are generated at once.

    Args:
        output_ids: List of token IDs to generate
        max_seq_len: Maximum sequence length
        eos_id: End of sequence token ID
        seq_len: Input sequence length
        tokens_per_chunk: Number of tokens to generate per chunk (default 3)
        output_gen: Generator function for creating output objects
    """
    for i in range(0, len(output_ids), tokens_per_chunk):
        # Get chunk of tokens (up to tokens_per_chunk)
        chunk_end = min(i + tokens_per_chunk, len(output_ids))
        chunk_size = chunk_end - i
        chunk_ids = output_ids[i:chunk_end]

        # Create output tensor with multiple tokens
        output_tensor = torch.full((1, chunk_size), eos_id, dtype=torch.int)
        output_tensor[0, :chunk_size] = torch.tensor(chunk_ids, dtype=torch.int)

        finished = torch.full((1,), (chunk_end == len(output_ids)), dtype=torch.bool)
        outputs = GenerateOutputs()
        aux = AuxInfo()
        aux.input_len = seq_len
        aux.output_len = chunk_end
        outputs.generate_outputs.append(
            output_gen(
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
    output_ids: List[int],
    max_seq_len: int,
    eos_id: int,
    seq_len: int,
    output_gen: Callable[..., GenerateOutput] = GenerateOutput,
) -> AsyncGenerator[GenerateOutputs, None]:
    # 创建包含所有token的完整输出张量, 模拟非流式的返回
    output_tensor = torch.full((1, len(output_ids)), eos_id, dtype=torch.int)
    output_tensor[0, : len(output_ids)] = torch.tensor(output_ids, dtype=torch.int)

    # 标记为已完成
    finished = torch.full((1,), True, dtype=torch.bool)

    outputs = GenerateOutputs()
    aux = AuxInfo()
    aux.input_len = seq_len
    aux.output_len = len(output_ids)

    outputs.generate_outputs.append(
        output_gen(
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

    def _create_tokenizer(self, tokenizer_path):
        """创建tokenizer - 子类可以重写"""
        return BaseTokenizer(tokenizer_path)

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

    def _validate_stream_chunk(self, chunk, stream):
        """验证流式chunk - 子类可以重写"""

    def _setup_environment(self):
        tokenizer_path = f"{self.parent.test_data_path}/{self._get_tokenizer_path()}"
        tokenizer = self._create_tokenizer(tokenizer_path)

        self.parent.tokenizer = tokenizer
        # Create minimal configs for test
        generate_env_config = GenerateEnvConfig()
        render_config = RenderConfig()
        misc_config = PyMiscellaneousConfig()
        vit_config = VitConfig()
        special_tokens = self.parent.model_config.special_tokens

        # Create ModelConfig object for OpenaiEndpoint
        model_config = ModelConfig()
        model_config.generate_env_config = generate_env_config
        model_config.render_config = render_config
        model_config.special_tokens = special_tokens
        model_config.max_seq_len = self.parent.model_config.max_seq_len
        model_config.template_type = None
        model_config.model_name = ""
        model_config.ckpt_path = ""

        pd_sep_config = PDSepConfig()
        backend_rpc_server_visitor = BackendRPCServerVisitor(
            max_seq_len=self.parent.model_config.max_seq_len,
            seq_size_per_block=64,
            pd_sep_config=pd_sep_config,
            addresses=["localhost:8080"],  # Default test address
        )
        self.parent.endpoint = OpenaiEndpoint(
            model_config=model_config,
            misc_config=misc_config,
            vit_config=vit_config,
            tokenizer=self.parent.tokenizer,
            backend_rpc_server_visitor=backend_rpc_server_visitor,
        )

        return tokenizer

    def _create_generate_config(self, stream: bool):
        return GenerateConfig(is_streaming=stream)

    def _create_generate_output(self, *args: Any, **kwargs: Any):
        return GenerateOutput(*args, **kwargs)

    async def _run_tool_call_test(
        self,
        stream=True,
        include_stop_word=False,
        stop_words_str=None,
        think_mode=0,
        tokens_per_chunk=None,
    ):
        """运行工具调用测试的通用方法

        Args:
            stream: 是否使用流式模式
            include_stop_word: 是否包含停止词
            stop_words_str: 停止词字符串
            think_mode: 思考模式
            tokens_per_chunk: 如果设置，模拟MTP（多token预测）（每次返回多个token）
        """
        tokenizer = self._setup_environment()
        test_ids = self._get_test_data(include_stop_word)
        render_params = self._create_render_params(tokenizer)

        generate_env_config = GenerateEnvConfig()
        generate_env_config.think_mode = think_mode
        render_config = RenderConfig()
        chat_renderer = ChatRendererFactory.get_renderer(
            tokenizer,
            render_params,
            generate_env_config=generate_env_config,
            render_config=render_config,
        )

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
                test_ids,
                MAX_SEQ_LEN,
                tokenizer.eos_token_id or 0,
                seq_len_no_use,
                self._create_generate_output,
            )
        elif tokens_per_chunk:
            # Use MTP (Multiple Token Prediction) generator
            id_generator = fake_output_generator_mtp(
                test_ids,
                MAX_SEQ_LEN,
                tokenizer.eos_token_id or 0,
                seq_len_no_use,
                tokens_per_chunk=tokens_per_chunk,
                output_gen=self._create_generate_output,
            )
        else:
            id_generator = fake_output_generator(
                test_ids,
                MAX_SEQ_LEN,
                tokenizer.eos_token_id or 0,
                seq_len_no_use,
                self._create_generate_output,
            )

        generate_config = self._create_generate_config(stream)
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
        choice = merged_result.choices[0]
        assert (
            choice.finish_reason == FinisheReason.tool_calls
        ), f"got finish_reason: {choice.finish_reason}"
        delta = choice.delta
        assert delta.role == RoleEnum.assistant
        self._assert_tool_call_response(delta)

    async def test_streaming_case(self, stop_words_str=None, think_mode=0):
        """测试工具调用流式场景"""
        chunk_list = await self._run_tool_call_test(
            stream=True,
            stop_words_str=stop_words_str,
            think_mode=think_mode,
        )

        merged_result: ChatCompletionStreamResponse = merge_stream_responses(chunk_list)
        self._validate_merged_result(merged_result)

    async def test_streaming_mtp(
        self, stop_words_str=None, think_mode=0, tokens_per_chunk=3
    ):
        """测试工具调用流式场景（投机采样，多token预测）

        Args:
            stop_words_str: 停止词
            think_mode: 思考模式
            tokens_per_chunk: 每次返回的token数量，模拟MTP (Multiple Token Prediction)
        """
        chunk_list = await self._run_tool_call_test(
            stream=True,
            stop_words_str=stop_words_str,
            think_mode=think_mode,
            tokens_per_chunk=tokens_per_chunk,
        )

        merged_result: ChatCompletionStreamResponse = merge_stream_responses(chunk_list)
        self._validate_merged_result(merged_result)

    async def test_no_stream(self, stop_words_str=None, think_mode=0):
        """测试工具调用非流式场景"""
        chunk_list = await self._run_tool_call_test(
            stream=False,
            stop_words_str=stop_words_str,
            think_mode=think_mode,
        )

        # 验证合并结果
        merged_result: ChatCompletionStreamResponse = merge_stream_responses(chunk_list)
        self._validate_merged_result(merged_result)


class QwenTestTokenizer(BaseTokenizer):
    def init_tokenizer(self, tokenizer_path: str, config_json={}):
        self.tokenizer = QWenTokenizer(tokenizer_path)
        self.im_start_id = self.tokenizer.im_start_id
        self.im_end_id = self.tokenizer.im_end_id


class OpenaiResponseTest(IsolatedAsyncioTestCase):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.test_data_path = os.path.join(
            os.getcwd(), "rtp_llm/test/model_test/fake_test/testdata"
        )
        self.model_config = ModelConfig()
        self.model_config.attn_config.head_num = 1024
        self.model_config.attn_config.size_per_head = 1024
        self.model_config.num_layers = 1024
        self.model_config.max_seq_len = 1024
        self.model_config.vocab_size = 1024
        self.model_config.special_tokens = SpecialTokens()

    async def test_parse_qwen_function_call(self):
        tokenizer = QwenTestTokenizer(
            f"{self.test_data_path}/qwen_7b/tokenizer/qwen.tiktoken"
        )
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
        generate_env_config = GenerateEnvConfig()
        render_config = RenderConfig()
        chat_renderer = ChatRendererFactory.get_renderer(
            tokenizer,
            render_params,
            generate_env_config=generate_env_config,
            render_config=render_config,
        )
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
        generate = OpenaiEndpoint._complete_stream_response(stream_generator, None)
        response = [x async for x in generate][-1]
        response = await generate.gen_complete_response_once()
        print(response.choices[0].model_dump_json())
        self.assertEqual(1, len(response.choices))
        self.assertEqual(
            json.loads(response.choices[0].model_dump_json(exclude_none=True)),
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Thought: 我可以使用 get_current_weather API。",
                    "function_call": {
                        "name": "get_current_weather",
                        "arguments": '{"location": "洛杉矶, 美国", "unit": "fahrenheit"}',
                    },
                    "partial": False,
                },
                "finish_reason": "function_call",
            },
        )

    async def test_finish_reason(self):
        tokenizer = QwenTestTokenizer(
            f"{self.test_data_path}/qwen_7b/tokenizer/qwen.tiktoken"
        )
        py_env_configs = PyEnvConfigs()
        test_ids = [198, 84169, 25, 49434, 239, 73670, 37029]
        render_params = RendererParams(
            model_type="qwen",
            max_seq_len=MAX_SEQ_LEN,
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
        generate = OpenaiEndpoint._complete_stream_response(stream_generator, None)
        response = [x async for x in generate][-1]
        response = await generate.gen_complete_response_once()
        print(response)
        assert response.choices[0].finish_reason
        self.assertEqual(FinisheReason.length, response.choices[0].finish_reason)

    async def test_parse_qwen_agent_function_call(self):
        tokenizer = QwenTestTokenizer(
            f"{self.test_data_path}/qwen_7b/tokenizer/qwen.tiktoken"
        )
        py_env_configs = PyEnvConfigs()
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
        generate_env_config = GenerateEnvConfig()
        render_config = RenderConfig()
        chat_renderer = ChatRendererFactory.get_renderer(
            tokenizer,
            render_params,
            generate_env_config=generate_env_config,
            render_config=render_config,
        )
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
        generate = OpenaiEndpoint._complete_stream_response(stream_generator, None)
        async for x in generate:
            response = x
            response = await generate.gen_complete_response_once()
            print(response.choices[0].model_dump_json())
        self.assertEqual(1, len(response.choices))
        self.assertEqual(
            json.loads(response.choices[0].model_dump_json(exclude_none=True)),
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "我需要调用get_current_weather API来获取天气",
                    "function_call": {
                        "name": "get_current_weather",
                        "arguments": '{"location": "洛杉矶, 美国", "unit": "fahrenheit"}',
                    },
                    "partial": False,
                },
                "finish_reason": "function_call",
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
        generate = OpenaiEndpoint._complete_stream_response(stream_generator, None)
        async for x in generate:
            response = x
            response = await generate.gen_complete_response_once()
            print(response.choices[0].model_dump_json())
        self.assertEqual(1, len(response.choices))
        self.assertEqual(
            json.loads(response.choices[0].model_dump_json(exclude_none=True)),
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": ': 我需要调用get_current_weather API来获取天气✿FUNCTION✿: get_current_weather\n✿ARGS✿: {"location": "洛杉矶, 美国", "unit": "fahrenheit"}',
                    "partial": False,
                },
                "finish_reason": "stop",
            },
        )

    async def test_parse_qwen_agent_tool_call(self):
        tokenizer = QwenTestTokenizer(
            f"{self.test_data_path}/qwen_7b/tokenizer/qwen.tiktoken"
        )
        py_env_configs = PyEnvConfigs()
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
        generate_env_config = GenerateEnvConfig()
        render_config = RenderConfig()
        chat_renderer = ChatRendererFactory.get_renderer(
            tokenizer,
            render_params,
            generate_env_config=generate_env_config,
            render_config=render_config,
        )
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
        generate = OpenaiEndpoint._complete_stream_response(stream_generator, None)
        # response = [x async for x in generate][-1]
        # response = await generate.gen_complete_response_once()
        # print(response.choices[0].model_dump_json())
        async for x in generate:
            response = x
            response = await generate.gen_complete_response_once()
            print(response.choices[0].model_dump_json())
        self.assertEqual(1, len(response.choices))
        target_delta = json.loads(
            response.choices[0].model_dump_json(exclude_none=True)
        )
        target_delta["message"]["tool_calls"][0]["id"] = "id"
        self.assertEqual(
            target_delta,
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "我需要调用get_current_weather API来获取天气",
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
        generate = OpenaiEndpoint._complete_stream_response(stream_generator, None)
        async for x in generate:
            response = x
            response = await generate.gen_complete_response_once()
            print(response.choices[0].model_dump_json())
        self.assertEqual(1, len(response.choices))
        self.assertEqual(
            json.loads(response.choices[0].model_dump_json(exclude_none=True)),
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": ': 我需要调用get_current_weather API来获取天气✿FUNCTION✿: get_current_weather\n✿ARGS✿: {"location": "洛杉矶, 美国", "unit": "fahrenheit"}',
                    "partial": False,
                },
                "finish_reason": "stop",
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
            """获取测试数据: 注意到, 这里是多轮对话的产物, idx不是从0开始的
            杭州天气很好，所以我再为您查询北京和南京的天气：<|tool_calls_section_begin|><|tool_call_begin|>functions.get_current_weather:1<|tool_call_argument_begin|>{"location": "北京"}<|tool_call_end|><|tool_call_begin|>functions.get_current_weather:2<|tool_call_argument_begin|>{"location": "南京"}<|tool_call_end|><|tool_calls_section_end|>
            """
            test_ids = [
                12365,
                8597,
                6523,
                378,
                24888,
                1295,
                19183,
                13021,
                3372,
                488,
                10525,
                65070,
                2648,
                163595,
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
                163597,
                41937,
                1150,
                20254,
                21055,
                2800,
                25,
                17,
                163598,
                8264,
                5791,
                1289,
                414,
                10525,
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
            expected_content="杭州天气很好，所以我再为您查询北京和南京的天气：",
        ):
            """断言工具调用响应的内容"""
            assert response_delta.content.strip() == expected_content.strip()
            assert response_delta.tool_calls[0].function.name == "get_current_weather"
            assert (
                response_delta.tool_calls[0].function.arguments
                == '{"location": "北京"}'
            )
            assert response_delta.tool_calls[1].function.name == "get_current_weather"
            assert (
                response_delta.tool_calls[1].function.arguments
                == '{"location": "南京"}'
            )
            assert response_delta.tool_calls[0].index == 0
            assert response_delta.tool_calls[1].index == 1
            # kimi需要校验tool_call id的值
            assert response_delta.tool_calls[0].id == "get_current_weather:1"
            assert response_delta.tool_calls[1].id == "get_current_weather:2"

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

    class KimiK2AdvancedTestSuite(KimiK2TestSuite):
        """KimiK2Advanced相关测试的内嵌测试套件, 测试能否支持带有-的functiononname"""

        @override
        def _get_test_data(self, include_stop_word=False):
            token_ids = [
                35659,
                80048,
                13021,
                12365,
                488,
                42930,
                8597,
                2267,
                292,
                163595,
                163597,
                41937,
                1150,
                67651,
                12,
                50171,
                25,
                15,
                163598,
                8264,
                5791,
                1289,
                414,
                12365,
                16934,
                163599,
                163597,
                41937,
                1150,
                67651,
                12,
                50171,
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
                token_ids.append(163586)  # 增加一个<|im_end|>
            return token_ids

        @override
        def _create_test_functions_and_tools(self):
            """创建测试用的函数和工具定义 - 子类可以重写"""
            functions = [
                GPTFunctionDefinition(
                    **{
                        "name": "get-current-weather",
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

        @override
        def _assert_tool_call_response(
            self,
            response_delta,
            expected_content="我来帮您查询杭州和北京的天气情况。",
        ):
            """断言工具调用响应的内容"""
            assert response_delta.content.strip() == expected_content.strip()
            assert response_delta.tool_calls[0].function.name == "get-current-weather"
            assert (
                response_delta.tool_calls[0].function.arguments
                == '{"location": "杭州"}'
            )
            assert response_delta.tool_calls[1].function.name == "get-current-weather"
            assert (
                response_delta.tool_calls[1].function.arguments
                == '{"location": "北京"}'
            )
            assert response_delta.tool_calls[0].index == 0
            assert response_delta.tool_calls[1].index == 1
            assert response_delta.tool_calls[0].id == "get-current-weather:0"
            assert response_delta.tool_calls[1].id == "get-current-weather:1"

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
            assert (
                response_delta.content.strip() == expected_content.strip()
            ), f"内容不匹配: 实际内容: {response_delta.content.strip()}, 期望内容: {expected_content.strip()}"
            assert (
                response_delta.reasoning_content.strip()
                == "用户询问杭州和北京的天气怎么样。"
            ), f"思考内容不匹配: 实际内容: {response_delta.reasoning_content.strip()}"
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

    class DeepseekV31TestSuite(BaseToolCallTestSuite):
        """DeepseekV31相关测试的内嵌测试套件"""

        def _get_model_type(self):
            return "deepseek_v31"

        def _get_tokenizer_path(self):
            return "deepseek_v31/tokenizer/"

        def _validate_renderer(self, chat_renderer):
            """验证renderer类型"""
            assert isinstance(chat_renderer, DeepseekV31Renderer)

        def _get_test_data(self, include_stop_word=False):
            """获取测试数据"""
            # 我来为您查询北京和杭州的天气情况。<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_current_weather<｜tool▁sep｜>{"location": "北京", "unit": "celsius"}<｜tool▁call▁end｜><｜tool▁call▁begin｜>get_current_weather<｜tool▁sep｜>{"location": "杭州", "unit": "celsius"}<｜tool▁call▁end｜><｜tool▁calls▁end｜>

            # 文本内容
            token_ids = [
                38463,
                48289,
                17916,
                6127,
                548,
                24463,
                301,
                16652,
                2782,
                320,
                128806,
                128808,
                1133,
                90605,
                65,
                50219,
                128814,
                24313,
                33182,
                3362,
                582,
                6127,
                1760,
                582,
                15165,
                3362,
                582,
                69,
                33030,
                62773,
                128809,
                128808,
                1133,
                90605,
                65,
                50219,
                128814,
                24313,
                33182,
                3362,
                582,
                24463,
                1760,
                582,
                15165,
                3362,
                582,
                69,
                33030,
                62773,
                128809,
                128807,
                201,
            ]

            return token_ids

        def _assert_tool_call_response(
            self,
            response_delta,
            expected_content="我来为您查询北京和杭州的天气情况。",
        ):
            """断言工具调用响应的内容"""
            assert response_delta.content.strip() == expected_content.strip()
            assert response_delta.tool_calls is not None
            assert len(response_delta.tool_calls) == 2
            assert response_delta.tool_calls[0].function.name == "get_current_weather"
            assert (
                response_delta.tool_calls[0].function.arguments
                == '{"location": "北京", "unit": "celsius"}'
            )
            assert response_delta.tool_calls[0].index == 0
            assert response_delta.tool_calls[1].function.name == "get_current_weather"
            assert (
                response_delta.tool_calls[1].function.arguments
                == '{"location": "杭州", "unit": "celsius"}'
            )
            assert response_delta.tool_calls[1].index == 1

    class DeepseekV31ThinkTestSuite(BaseToolCallTestSuite):
        """DeepseekV31相关测试的内嵌测试套件"""

        def _get_model_type(self):
            return "deepseek_v31"

        def _get_tokenizer_path(self):
            return "deepseek_v31/tokenizer/"

        def _validate_renderer(self, chat_renderer):
            """验证renderer类型"""
            assert isinstance(chat_renderer, DeepseekV31Renderer)

        async def _run_tool_call_test(
            self,
            stream=True,
            include_stop_word=False,
            stop_words_str=None,
            think_mode=0,
        ):
            """运行工具调用测试的通用方法"""
            tokenizer = self._setup_environment()
            test_ids = self._get_test_data(include_stop_word)
            render_params = self._create_render_params(tokenizer)

            generate_env_config = GenerateEnvConfig()
            generate_env_config.think_mode = think_mode
            render_config = RenderConfig()
            chat_renderer = ChatRendererFactory.get_renderer(
                tokenizer,
                render_params,
                generate_env_config=generate_env_config,
                render_config=render_config,
            )

            # 子类特定的renderer验证
            self._validate_renderer(chat_renderer)

            request = ChatCompletionRequest(
                messages=[ChatMessage(role=RoleEnum.user, content="hello")],
                stream=stream,
                chat_template_kwargs={"thinking": True},
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
            generate = OpenaiEndpoint._complete_stream_response(stream_generator, None)

            chunk_list = []
            async for chunk in generate:
                chunk_list.append(chunk)

            return chunk_list

        def _get_test_data(self, include_stop_word=False):
            """获取测试数据"""
            # 唔，用户发来一个简单的问候"你好"。</think>有什么我可以帮你的吗？

            # 文本内容
            token_ids = [
                53217,
                303,
                6640,
                740,
                637,
                1057,
                18341,
                56020,
                4,
                30594,
                75693,
                128799,
                10457,
                34071,
                3950,
                4597,
                3467,
                1148,
            ]

            return token_ids

        def _validate_merged_result(self, merged_result):
            """验证合并后的结果 - 通用验证逻辑"""
            assert merged_result.choices[0].finish_reason == FinisheReason.stop
            delta = merged_result.choices[0].delta
            assert delta.role == RoleEnum.assistant
            self._assert_tool_call_response(delta)

        def _assert_tool_call_response(
            self,
            response_delta,
            expected_content="有什么我可以帮你的吗？",
        ):
            """断言工具调用响应的内容"""
            assert response_delta.content.strip() == expected_content.strip()
            assert (
                response_delta.reasoning_content.strip()
                == '唔，用户发来一个简单的问候"你好"。'
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

    async def test_parse_qwen_think_call_streaming_case(self):
        """测试QwenTool工具思考调用流式场景"""
        qwen_suite = self.QwenThinkTestSuite(self)
        await qwen_suite.test_streaming_case(stop_words_str=["<im_end>"], think_mode=1)

    async def test_parse_qwen_think_call_no_stream(self):
        """测试QwenTool工具思考调用非流式场景"""
        qwen_suite = self.QwenThinkTestSuite(self)
        await qwen_suite.test_no_stream(stop_words_str=["<im_end>"], think_mode=1)

    async def test_parse_qwen_force_think_call_streaming_case(self):
        """测试QwenTool工具思考调用流式场景"""
        qwen_suite = self.QwenForceThinkTestSuite(self)
        await qwen_suite.test_streaming_case(stop_words_str=["<im_end>"], think_mode=1)

    async def test_parse_qwen_force_think_call_no_stream(self):
        """测试QwenTool工具思考调用非流式场景"""
        qwen_suite = self.QwenForceThinkTestSuite(self)
        await qwen_suite.test_no_stream(stop_words_str=["<im_end>"], think_mode=1)

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

    async def test_parse_kimik2_advanced_tool_call_streaming_case(self):
        """测试KimiK2工具调用流式场景"""
        kimi_suite = self.KimiK2AdvancedTestSuite(self)
        await kimi_suite.test_streaming_case()

    async def test_parse_kimik2_advanced_tool_call_no_stream(self):
        """测试KimiK2工具调用非流式场景"""
        kimi_suite = self.KimiK2AdvancedTestSuite(self)
        await kimi_suite.test_no_stream()

    async def test_parse_kimik2_advanced_tool_call_no_stream_stop_words(self):
        """测试KimiK2工具调用非流式场景（包含停止词）"""
        kimi_suite = self.KimiK2AdvancedTestSuite(self)
        await kimi_suite.test_no_stream_stop_words()

    async def test_parse_chatglm45_tool_call_streaming_case(self):
        suite = self.ChatGLM45TestSuite(self)
        await suite.test_streaming_case(think_mode=1)

    async def test_parse_chatglm45_tool_call_no_stream(self):
        suite = self.ChatGLM45TestSuite(self)
        await suite.test_no_stream(think_mode=1)

    async def test_parse_chatglm45_tool_call_streaming_mtp(self):
        """测试ChatGLM45工具调用流式场景（投机采样，多token预测）"""
        suite = self.ChatGLM45TestSuite(self)
        await suite.test_streaming_mtp(think_mode=1, tokens_per_chunk=3)

    async def test_parse_chatglm45_tool_call_streaming_mtp_edge_case(self):
        """测试ChatGLM45工具调用流式场景（MTP边界情况：chunk包含</think>结束标签和后续内容）

        Edge case: 测试当一个chunk包含 ["。", "</think>", "我", "来"] 时的行为
        - "。" 是reasoning的最后一个token
        - "</think>" 是thinking结束标签
        - "我来" 是正常内容
        这个测试确保reasoning parser能正确处理结束标签和后续内容在同一chunk的情况
        """
        suite = self.ChatGLM45TestSuite(self)
        # 使用4个token per chunk来触发edge case:
        # chunk会包含: [1773(。), 151351(</think>), 198(\n), 110943(我来)]
        await suite.test_streaming_mtp(think_mode=1, tokens_per_chunk=4)

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

    async def test_parse_deepseek_v31_tool_call_streaming_case(self):
        """测试deepseek_v31工具调用流式场景, 并且全程打开think_mode"""
        test_suite = self.DeepseekV31TestSuite(self)
        await test_suite.test_streaming_case(think_mode=1)

    async def test_parse_deepseek_v31_tool_call_no_stream(self):
        """测试deepseek_v31工具调用非流式场景, 并且全程打开think_mode"""
        test_suite = self.DeepseekV31TestSuite(self)
        await test_suite.test_no_stream(think_mode=1)

    async def test_parse_deepseek_v31_think_streaming_case(self):
        """测试deepseek_v31打开思考的流式场景"""
        test_suite = self.DeepseekV31ThinkTestSuite(self)
        await test_suite.test_streaming_case(think_mode=1)

    async def test_parse_deepseek_v31_think_no_stream(self):
        """测试deepseek_v31打开思考的非流式场景"""
        test_suite = self.DeepseekV31ThinkTestSuite(self)
        await test_suite.test_no_stream(think_mode=1)

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
                        (1, 1), self.eos_token_id, dtype=torch.int
                    )
                    output_tensor1[0, 0] = torch.tensor(first_token, dtype=torch.int)

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
                        (1, len(remaining_tokens)), self.eos_token_id, dtype=torch.int
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
            generate_env_config = GenerateEnvConfig()
            render_config = RenderConfig()
            chat_renderer = ChatRendererFactory.get_renderer(
                tokenizer,
                render_params,
                generate_env_config=generate_env_config,
                render_config=render_config,
            )
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

            generate = OpenaiEndpoint._complete_stream_response(
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
        tokenizer = TokenizerFactory.create(
            "", "rtp_llm/test/tokenizer_test/testdata/chatglm3_tokenizer", "chatglm3"
        )
        generate_env_config = GenerateEnvConfig()
        render_config = RenderConfig()
        misc_config = PyMiscellaneousConfig()
        vit_config = VitConfig()
        special_tokens = self.model_config.special_tokens
        pd_sep_config = PDSepConfig()
        backend_rpc_server_visitor = BackendRPCServerVisitor(
            max_seq_len=self.model_config.max_seq_len,
            seq_size_per_block=64,
            pd_sep_config=pd_sep_config,
            addresses=["localhost:8080"],
        )
        # Update model_config with test values
        test_model_config = self.model_config
        test_model_config.special_tokens = special_tokens
        test_model_config.generate_env_config = generate_env_config
        test_model_config.render_config = render_config
        test_model_config.model_name = ""
        test_model_config.template_type = None
        test_model_config.ckpt_path = ""
        test_model_config.model_type = "chatglm3"
        self.endpoint = OpenaiEndpoint(
            model_config=test_model_config,
            misc_config=misc_config,
            vit_config=vit_config,
            tokenizer=tokenizer,
            backend_rpc_server_visitor=backend_rpc_server_visitor,
        )
        self.assertEqual(
            sorted(self.endpoint.stop_words_id_list), sorted([[2], [64795], [64797]])
        )
        self.assertEqual(
            sorted(self.endpoint.stop_words_str_list),
            sorted(["<|user|>", "<|observation|>"]),
        )

    async def test_think_label_real_situation_union(self):
        tokenizer = TokenizerFactory.create(
            "", f"{self.test_data_path}/deepseek_r1_qwen_14b_tokenizer", "qwen_2"
        )

        test_ids = [151648, 198, 73670, 73670, 73670, 151649, 271, 37029, 37029, 37029]
        render_params = RendererParams(
            model_type="qwen",
            max_seq_len=MAX_SEQ_LEN,
            eos_token_id=tokenizer.eos_token_id or 0,
            stop_word_ids_list=[],
        )

        generate_env_config = GenerateEnvConfig()
        generate_env_config.think_mode = 1
        generate_env_config.think_start_tag = "<think>\n"
        generate_env_config.think_end_tag = "</think>\n"
        render_config = RenderConfig()

        chat_renderer = ChatRendererFactory.get_renderer(
            tokenizer,
            render_params,
            generate_env_config=generate_env_config,
            render_config=render_config,
        )
        request = ChatCompletionRequest(
            messages=[ChatMessage(role=RoleEnum.user, content="hello")], stream=True
        )
        input_length = 109
        id_generator = fake_output_generator(
            test_ids, MAX_SEQ_LEN, tokenizer.eos_token_id or 0, input_length
        )
        stream_generator = chat_renderer.render_response_stream(
            id_generator, request, GenerateConfig(is_streaming=True)
        )
        generate = OpenaiEndpoint._complete_stream_response(stream_generator, None)
        # response = [x async for x in generate][-1]
        async for x in generate:
            pass
        response = await generate.gen_complete_response_once()
        print(response.choices[0].model_dump_json())
        self.assertEqual(1, len(response.choices))
        self.assertEqual(
            json.loads(response.choices[0].model_dump_json(exclude_none=True)),
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "\n\n使用使用使用",
                    "reasoning_content": "\n可以可以可以",
                    "partial": False,
                },
                "finish_reason": "stop",
            },
        )
        self.assertEqual(
            json.loads(
                response.usage.completion_tokens_details.model_dump_json(
                    exclude_none=True
                )
            ),
            {"reasoning_tokens": 5},
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

    class ExtraOutputsTestSuite(QwenToolTestSuite):
        def __init__(
            self,
            parent_test: "OpenaiResponseTest",
            generate_config: GenerateConfig,
            output_gen: Callable[..., GenerateOutput],
            extra_outputs: ChatCompletionExtraOutputs,
        ):
            super().__init__(parent_test)
            self.generate_config = generate_config
            self.output_gen = output_gen
            self.extra_outputs = extra_outputs

        def _create_generate_config(self, stream: bool):
            generate_config = copy.deepcopy(self.generate_config)
            generate_config.is_streaming = stream
            return generate_config

        def _create_generate_output(self, *args: Any, **kwargs: Any):
            return self.output_gen(*args, **kwargs)

        def _validate_merged_result(self, merged_result):
            self.parent.assertEqual(merged_result.extra_outputs, self.extra_outputs)

    async def test_openai_extra_outputs_no_stream(self):
        """测试 Openai Endpoint 非流式场景的 extra_outputs 字段"""

        no_extra_output_test_suite = self.ExtraOutputsTestSuite(
            self, GenerateConfig(), GenerateOutput, None
        )
        await no_extra_output_test_suite.test_no_stream()

        output_ids = [[1, 2, 3]]
        return_output_ids_test_suite = self.ExtraOutputsTestSuite(
            self,
            GenerateConfig(return_output_ids=True),
            lambda *args, **kwargs: GenerateOutput(
                *args, **{**kwargs, "output_ids": torch.tensor(output_ids)}
            ),
            ChatCompletionExtraOutputs(output_ids=output_ids),
        )

        await return_output_ids_test_suite.test_no_stream()

    async def test_debug_info_with_output_ids_and_raw_output(self):
        """Test that debug_info includes output_ids and raw_output when debug_info=True (non-streaming)"""
        tokenizer = QwenTestTokenizer(
            f"{self.test_data_path}/qwen_7b/tokenizer/qwen.tiktoken"
        )
        test_ids = [198, 84169, 25, 73670, 37029]

        generate_env_config = GenerateEnvConfig()
        render_config = RenderConfig()
        render_params = RendererParams(
            model_type="qwen",
            max_seq_len=MAX_SEQ_LEN,
            eos_token_id=tokenizer.eos_token_id or 0,
            stop_word_ids_list=[],
        )
        chat_renderer = ChatRendererFactory.get_renderer(
            tokenizer,
            render_params,
            generate_env_config=generate_env_config,
            render_config=render_config,
        )
        request = ChatCompletionRequest(
            messages=[ChatMessage(role=RoleEnum.user, content="hello")],
            stream=False,
        )
        input_length = 314
        id_generator = fake_output_generator_once(
            test_ids, MAX_SEQ_LEN, tokenizer.eos_token_id or 0, input_length
        )

        generate_config = GenerateConfig(is_streaming=False, return_output_ids=True)
        stream_generator = chat_renderer.render_response_stream(
            id_generator, request, generate_config
        )

        debug_info = DebugInfo(
            input_prompt="test prompt",
            input_ids=[1, 2, 3],
            input_urls=[],
            tokenizer_info="test tokenizer",
            max_seq_len=MAX_SEQ_LEN,
            eos_token_id=tokenizer.eos_token_id or 0,
            stop_word_ids_list=[],
            stop_words_list=[],
            renderer_info=chat_renderer.get_renderer_info(),
            generate_config=generate_config,
        )
        generate = OpenaiEndpoint._complete_stream_response(
            stream_generator, debug_info, tokenizer
        )
        async for x in generate:
            pass
        response = await generate.gen_complete_response_once()

        self.assertIsNotNone(response.extra_outputs)
        self.assertIsNotNone(response.extra_outputs.output_ids)
        self.assertEqual(len(response.extra_outputs.output_ids), 1)
        self.assertEqual(response.extra_outputs.output_ids[0], test_ids)

        # Verify debug_info has output_ids and raw_output
        self.assertIsNotNone(response.debug_info)
        self.assertIsNotNone(response.debug_info.output_ids)
        self.assertIsNotNone(response.debug_info.raw_output)
        self.assertEqual(response.debug_info.output_ids, [test_ids])
        expected_raw_output = tokenizer.decode(test_ids)
        self.assertEqual(response.debug_info.raw_output, [expected_raw_output])

    async def test_debug_info_with_output_ids_and_raw_output_streaming(self):
        """Test that debug_info includes output_ids and raw_output when debug_info=True (streaming mode)"""
        tokenizer = QwenTestTokenizer(
            f"{self.test_data_path}/qwen_7b/tokenizer/qwen.tiktoken"
        )
        test_ids = [198, 84169, 25, 73670, 37029]
        generate_env_config = GenerateEnvConfig()
        render_config = RenderConfig()
        render_params = RendererParams(
            model_type="qwen",
            max_seq_len=MAX_SEQ_LEN,
            eos_token_id=tokenizer.eos_token_id or 0,
            stop_word_ids_list=[],
        )
        chat_renderer = ChatRendererFactory.get_renderer(
            tokenizer,
            render_params,
            generate_env_config=generate_env_config,
            render_config=render_config,
        )
        request = ChatCompletionRequest(
            messages=[ChatMessage(role=RoleEnum.user, content="hello")],
            stream=True,
        )
        input_length = 314
        id_generator = fake_output_generator(
            test_ids, MAX_SEQ_LEN, tokenizer.eos_token_id or 0, input_length
        )

        generate_config = GenerateConfig(is_streaming=True, return_output_ids=True)
        stream_generator = chat_renderer.render_response_stream(
            id_generator, request, generate_config
        )

        debug_info = DebugInfo(
            input_prompt="test prompt",
            input_ids=[1, 2, 3],
            input_urls=[],
            tokenizer_info="test tokenizer",
            max_seq_len=MAX_SEQ_LEN,
            eos_token_id=tokenizer.eos_token_id or 0,
            stop_word_ids_list=[],
            stop_words_list=[],
            renderer_info=chat_renderer.get_renderer_info(),
            generate_config=generate_config,
        )
        generate = OpenaiEndpoint._complete_stream_response(
            stream_generator, debug_info, tokenizer
        )
        # Collect all streaming chunks
        chunks = []
        async for chunk in generate:
            chunks.append(chunk)

        # Verify that at least 5 chunks exist
        debug_info_chunks = [c for c in chunks if c.debug_info is not None]
        self.assertGreater(
            len(debug_info_chunks),
            5,
            f"recevied chunks' length: {len(debug_info_chunks)}",
        )

        # collect output_ids and raw_output from all debug_info chunks
        # the output may many choices, make sure the return is apended to correct index
        output_ids_collected = []
        raw_output_collected = []
        for chunk in debug_info_chunks:
            if chunk.debug_info.output_ids:
                for i, ids in enumerate(chunk.debug_info.output_ids):
                    if len(output_ids_collected) <= i:
                        output_ids_collected.append(ids)
                    else:
                        output_ids_collected[i].extend(ids)
            if chunk.debug_info.raw_output:
                for i, raw in enumerate(chunk.debug_info.raw_output):
                    if len(raw_output_collected) <= i:
                        raw_output_collected.append(raw)
                    else:
                        raw_output_collected[i] += raw

        self.assertEqual(output_ids_collected, [test_ids])
        expected_raw_output = tokenizer.decode(test_ids)
        self.assertEqual(raw_output_collected, [expected_raw_output])


if __name__ == "__main__":
    main()
