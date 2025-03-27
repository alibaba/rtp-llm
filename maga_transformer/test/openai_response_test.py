import os
import json
import torch
import logging
from unittest import TestCase, main, IsolatedAsyncioTestCase
from typing import Optional, List, Dict, Any, Union, Callable, Tuple, AsyncGenerator
from transformers import AutoTokenizer, PreTrainedTokenizer

from maga_transformer.config.gpt_init_model_parameters import GptInitModelParameters
from maga_transformer.models.base_model import BaseModel, GenerateOutput, GenerateOutputs, AuxInfo
from maga_transformer.pipeline.chatapi_format import encode_chatapi
from maga_transformer.tokenizer.tokenization_qwen import QWenTokenizer
from maga_transformer.tokenizer.tokenization_chatglm3 import ChatGLMTokenizer
from maga_transformer.openai.api_datatype import (
    ChatMessage,
    GPTToolDefinition,
    RoleEnum,
    FinisheReason,
    ChatCompletionRequest,
    GPTFunctionDefinition,
    ContentPart,
    ContentPartTypeEnum,
    RendererInfo,
)
from maga_transformer.openai.openai_endpoint import OpenaiEndopoint
from maga_transformer.config.generate_config import GenerateConfig
from maga_transformer.openai.renderer_factory import ChatRendererFactory, CustomChatRenderer, RendererParams
from maga_transformer.openai.renderers import custom_renderer

async def fake_output_generator(
        output_ids: List[int], max_seq_len: int, eos_id: int, seq_len: int
) -> AsyncGenerator[GenerateOutputs, None]:
    for i in range(0, len(output_ids)):
        output_tensor = torch.full((1, max_seq_len), eos_id, dtype=torch.int)

        output_tensor[0, :len(output_ids[i:i + 1])] = torch.tensor(output_ids[i:i + 1], dtype=torch.int)
        finished = torch.full((1,), (i == (len(output_ids) - 1)), dtype=torch.bool)
        logging.info(f"i={i}, finished={finished}")
        outputs = GenerateOutputs()
        aux = AuxInfo()
        aux.input_len = seq_len
        aux.output_len = i + 1
        outputs.generate_outputs.append(GenerateOutput(
            hidden_states=None,
            output_ids=output_tensor,
            finished=finished,
            aux_info=aux,
            loss=None,
            logits=None
        ))
        yield outputs

MAX_SEQ_LEN=1024

class FakeModel(BaseModel):
    def load_tokenizer(self):
        pass

    def init_misc(self):
        pass

    def load(self, ckpt_path: str):
        pass

class OpenaiResponseTest(IsolatedAsyncioTestCase):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.test_data_path = os.path.join(
            os.getcwd(), 'maga_transformer/test/model_test/fake_test/testdata'
        )
        model_params = GptInitModelParameters(
            head_num = 1024,
            size_per_head = 1024,
            layer_num = 1024,
            max_seq_len = 1024,
            vocab_size = 1024,
        )
        self.model = FakeModel(None)
        self.model.config = model_params

    async def test_parse_qwen_function_call(self):
        os.environ["MODEL_TYPE"] = "qwen"
        tokenizer = QWenTokenizer(f"{self.test_data_path}/qwen_7b/tokenizer/qwen.tiktoken")
        self.model.tokenizer = tokenizer
        self.endpoint = OpenaiEndopoint(self.model.config, self.model.tokenizer, None)
        test_ids = [198, 84169, 25, 49434, 239, 73670, 37029, 633, 11080, 69364, 5333, 8997, 2512, 25, 633, 11080, 69364, 198, 2512, 5571, 25, 5212, 2527, 788, 330, 113074, 11, 10236, 122, 236, 28404, 497, 330, 3843, 788, 330, 69, 47910, 16707]
        render_params = RendererParams(
            model_type="qwen",
            max_seq_len=1024,
            eos_token_id=tokenizer.eos_token_id or 0,
            stop_word_ids_list=[],
        )
        chat_renderer = ChatRendererFactory.get_renderer(tokenizer, render_params)
        request = ChatCompletionRequest(messages=[], functions=[GPTFunctionDefinition(**{
            "name": "get_current_weather",
            "description": "Get the current weather in a given location.",
            "parameters": {}
        })])
        id_generator = fake_output_generator(test_ids, 1024, tokenizer.eos_token_id or 0, 314)
        stream_generator = chat_renderer.render_response_stream(id_generator, request, GenerateConfig())
        generate = self.endpoint._complete_stream_response(stream_generator, None)
        response = [x async for x in generate][-1]
        response = await generate.gen_complete_response_once()
        print(response.choices[0].model_dump_json())
        self.assertEqual(1, len(response.choices))
        self.assertEqual(json.loads(response.choices[0].model_dump_json()), {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Thought: 我可以使用 get_current_weather API。",
                "reasoning_content": None,
                "function_call": {
                    "name": "get_current_weather",
                    "arguments": "{\"location\": \"洛杉矶, 美国\", \"unit\": \"fahrenheit\"}"
                },
                "tool_calls": None,
                "partial": False,
            },
            "finish_reason": "function_call",
            "logprobs": None,
        })

    async def test_finish_reason(self):
        os.environ["MODEL_TYPE"] = "qwen"
        tokenizer = QWenTokenizer(f"{self.test_data_path}/qwen_7b/tokenizer/qwen.tiktoken")
        self.model.tokenizer = tokenizer
        self.endpoint = OpenaiEndopoint(self.model.config, self.model.tokenizer, None)
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
        id_generator = fake_output_generator(test_ids, MAX_SEQ_LEN, tokenizer.eos_token_id or 0, input_length)
        stream_generator = chat_renderer.render_response_stream(id_generator, request, GenerateConfig())
        generate = self.endpoint._complete_stream_response(stream_generator, None)
        response = [x async for x in generate][-1]
        response = await generate.gen_complete_response_once()
        print(response)
        assert(response.choices[0].finish_reason)
        self.assertEqual(FinisheReason.length, response.choices[0].finish_reason)

    async def test_parse_qwen_agent_function_call(self):
        os.environ["MODEL_TYPE"] = "qwen_agent"
        tokenizer = QWenTokenizer(f"{self.test_data_path}/qwen_7b/tokenizer/qwen.tiktoken")
        self.model.tokenizer = tokenizer
        self.endpoint = OpenaiEndopoint(self.model.config, self.model.tokenizer, None)
        test_ids = [25, 220, 35946, 85106, 47872, 11622, 455, 11080, 69364, 5333, 36407, 45912, 104307, 144575, 18149, 144575, 25, 633, 11080, 69364, 198, 144575, 47483, 144575, 25, 5212, 2527, 788, 330, 113074, 11, 10236, 122, 236, 28404, 497, 330, 3843, 788, 330, 69, 47910, 16707, 144575, 14098, 144575]
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
        request = ChatCompletionRequest(messages=[], functions=functions)
        id_generator = fake_output_generator(test_ids, 1024, tokenizer.eos_token_id or 0, 314)
        stream_generator = chat_renderer.render_response_stream(id_generator, request, GenerateConfig())
        generate = self.endpoint._complete_stream_response(stream_generator, None)
        async for x in generate:
            response = x
            response = await generate.gen_complete_response_once()
            print(response.choices[0].model_dump_json())
        self.assertEqual(1, len(response.choices))
        self.assertEqual(json.loads(response.choices[0].model_dump_json()), {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "我需要调用get_current_weather API来获取天气",
                "reasoning_content": None,
                "function_call": {
                    "name": "get_current_weather",
                    "arguments": "{\"location\": \"洛杉矶, 美国\", \"unit\": \"fahrenheit\"}"
                },
                "tool_calls": None,
                "partial": False,
            },
            "finish_reason": "function_call",
            "logprobs": None,
        })
        # 非functioncall 格式返回，输入没有functions
        request = ChatCompletionRequest(messages=[])
        id_generator = fake_output_generator(test_ids, 1024, tokenizer.eos_token_id or 0, 314)
        gen_config = GenerateConfig()
        stream_generator = chat_renderer.render_response_stream(id_generator, request, gen_config)
        generate = self.endpoint._complete_stream_response(stream_generator, None)
        async for x in generate:
            response = x
            response = await generate.gen_complete_response_once()
            print(response.choices[0].model_dump_json())
        self.assertEqual(1, len(response.choices))
        self.assertEqual(json.loads(response.choices[0].model_dump_json()), {
            "index": 0,
            "message":
            {
                "role": "assistant",
                "content": ": 我需要调用get_current_weather API来获取天气✿FUNCTION✿: get_current_weather\n✿ARGS✿: {\"location\": \"洛杉矶, 美国\", \"unit\": \"fahrenheit\"}",
                "reasoning_content": None,
                "function_call": None,
                "tool_calls": None,
                "partial": False,
            },
            "finish_reason": "stop",
            "logprobs": None,
        })

    async def test_parse_qwen_agent_tool_call(self):
        os.environ["MODEL_TYPE"] = "qwen_agent_tool"
        tokenizer = QWenTokenizer(
            f"{self.test_data_path}/qwen_7b/tokenizer/qwen.tiktoken"
        )
        self.model.tokenizer = tokenizer
        self.endpoint = OpenaiEndopoint(self.model.config, self.model.tokenizer, None)
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

    def test_chatglm_stop_word(self):
        os.environ["MODEL_TYPE"] = "chatglm3"
        tokenizer = ChatGLMTokenizer.from_pretrained(
            "maga_transformer/test/tokenizer_test/testdata/chatglm3_tokenizer",
            encode_special_tokens=True)
        self.model.tokenizer = tokenizer
        self.endpoint = OpenaiEndopoint(self.model.config, self.model.tokenizer, None)
        self.assertEqual(self.endpoint.stop_words_id_list, [[64795], [64797], [2]])
        self.assertEqual(self.endpoint.stop_words_str_list, ['<|user|>', '<|observation|>'])

    async def test_think_label(self):
        custom_renderer.think_mode = 1
        custom_renderer.think_end_tag = 'ulaire'    # id = 73675
        custom_renderer.think_end_tag_len = len('ulaire')
        os.environ['MODEL_TYPE'] = 'qwen'
        tokenizer = QWenTokenizer(f"{self.test_data_path}/qwen_7b/tokenizer/qwen.tiktoken")
        self.model.tokenizer = tokenizer
        self.endpoint = OpenaiEndopoint(self.model.config, self.model.tokenizer, None)
        
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
        id_generator = fake_output_generator(test_ids, MAX_SEQ_LEN, tokenizer.eos_token_id or 0, input_length)
        stream_generator = chat_renderer.render_response_stream(id_generator, request, GenerateConfig())
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
                    "content": '我使用使用使用',
                    "reasoning_content": '我可以可以可以',
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
            {
                'audio_tokens': None, 
                'reasoning_tokens': 5
            }
        )

    async def test_think_label_more_than_one_token(self):
        custom_renderer.think_mode = 1
        custom_renderer.think_start_tag = '我可以'
        custom_renderer.think_end_tag = '可以ulaire'
        os.environ['MODEL_TYPE'] = 'qwen'
        tokenizer = QWenTokenizer(f"{self.test_data_path}/qwen_7b/tokenizer/qwen.tiktoken")
        self.model.tokenizer = tokenizer
        self.endpoint = OpenaiEndopoint(self.model.config, self.model.tokenizer, None)
        
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
        id_generator = fake_output_generator(test_ids, MAX_SEQ_LEN, tokenizer.eos_token_id or 0, input_length)
        stream_generator = chat_renderer.render_response_stream(id_generator, request, GenerateConfig())
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
                    "content": '我使用使用使用',
                    "reasoning_content": '可以',
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
            {
                'audio_tokens': None, 
                'reasoning_tokens': 5
            }
        )
        
    async def test_think_label_real_situation_union(self):     
        custom_renderer.think_mode = 1
        custom_renderer.think_start_tag = '<think>\n'
        custom_renderer.think_end_tag = '</think>\n'
        os.environ['MODEL_TYPE'] = 'qwen_2'
        tokenizer = AutoTokenizer.from_pretrained(f"{self.test_data_path}/deepseek_r1_qwen_14b_tokenizer")
        self.model.tokenizer = tokenizer
        self.endpoint = OpenaiEndopoint(self.model.config, self.model.tokenizer, None)
        
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
        id_generator = fake_output_generator(test_ids, MAX_SEQ_LEN, tokenizer.eos_token_id or 0, input_length)
        stream_generator = chat_renderer.render_response_stream(id_generator, request, GenerateConfig())
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
                    "content": '\n使用使用使用',
                    "reasoning_content": '可以可以可以',
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
            {
                'audio_tokens': None, 
                'reasoning_tokens': 6
            }
        )
        
    async def test_escape(self):     
        think_start_tag = '<think>\n'
        self.assertEqual(think_start_tag, think_start_tag.encode('utf-8').decode('unicode_escape'))
        think_end_tag = '</think>\n\n'
        self.assertEqual(think_end_tag, think_end_tag.encode('utf-8').decode('unicode_escape'))
        think_start_tag_from_env = '<think>\\n'
        self.assertEqual(think_start_tag, think_start_tag_from_env.encode('utf-8').decode('unicode_escape'))
        self.assertNotEqual(think_start_tag, think_start_tag_from_env)
        think_end_tag_from_env = '</think>\\n\\n'
        self.assertEqual(think_end_tag, think_end_tag_from_env.encode('utf-8').decode('unicode_escape'))
        self.assertNotEqual(think_end_tag, think_end_tag_from_env)

if __name__ == '__main__':
    main()
