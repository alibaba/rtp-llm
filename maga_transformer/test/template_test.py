import os
import torch
from unittest import TestCase, main
from typing import Any

from maga_transformer.pipeline.chatapi_format import encode_chatapi
from maga_transformer.models.starcoder import StarcoderTokenizer
from maga_transformer.openai.api_datatype import ChatMessage, RoleEnum, \
    ChatCompletionRequest, GPTFunctionDefinition
from maga_transformer.tokenizer.tokenization_qwen import QWenTokenizer
from maga_transformer.openai.renderers.renderer_factory import ChatRendererFactory, RendererParams

class ChatapiTest(TestCase):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.test_data_path = os.path.join(
            os.getcwd(), 'maga_transformer/test/model_test/fake_test/testdata'
        )

    def test_qwen(self):
        tokenizer = QWenTokenizer(f"{self.test_data_path}/qwen_7b/tokenizer/qwen.tiktoken")
        render_params = RendererParams(
            max_seq_len=1024,
            eos_token_id=tokenizer.eos_token_id or 0,
            stop_word_ids_list=[],
        )
        chat_renderer = ChatRendererFactory.get_renderer(tokenizer, render_params)

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
        print(f"rendered prompt: \n{prompt}\n-----------------------------------")
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
        print(f"expected prompt: \n{expected_prompt}\n-----------------------------------")
        assert (prompt == expected_prompt)

        messages.append(
            ChatMessage(**{
                "role": RoleEnum.assistant,
                "content": "",
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
        print(f"rendered prompt: \n{prompt}\n-----------------------------------")
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
Thought: 我可以使用 get_current_weather API。
Action: get_current_weather
Action Input: {"location": "Boston, MA"}
Observation: {"temperature": "22", "unit": "celsius", "description": "Sunny"}
Thought:"""
        print(f"expected prompt: \n{expected_prompt}\n-----------------------------------")
        assert (prompt == expected_prompt)

if __name__ == '__main__':
    main()
