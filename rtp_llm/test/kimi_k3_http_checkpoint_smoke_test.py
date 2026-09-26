"""Checkpoint-backed K3 HTTP requests with deterministic model output.

The tokenizer, renderer, endpoint and FastAPI route are real. Only the text
model forward is replayed because the target K3 modeling code is separate.
Run with --test_env=K3_CKPT_PATH=/ssd/5/kimi-k3.
"""

import base64
import io
import json
import math
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import torch
from fastapi.testclient import TestClient
from PIL import Image
from transformers import AutoTokenizer

from rtp_llm.config.generate_config import ReturnAllProbsMode
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.py_config_modules import (
    GenerateEnvConfig,
    PyMiscellaneousConfig,
    RenderConfig,
    VitConfig,
)
from rtp_llm.frontend.frontend_app import FrontendApp
from rtp_llm.frontend.frontend_server import FrontendServer
from rtp_llm.frontend.shutdown_manager import FrontendShutdownManager
from rtp_llm.multimodal.multimodal_mixins.kimi_k3.kimi_k3_vit import (
    KimiK3ImageEmbedding,
)
from rtp_llm.openai.openai_endpoint import OpenaiEndpoint
from rtp_llm.openai.renderers.kimi_k3_renderer import KimiK3Renderer
from rtp_llm.ops import SpecialTokens
from rtp_llm.utils.base_model_datatypes import AuxInfo, GenerateOutput, GenerateOutputs


class ReplayBackend:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.calls = []
        self.reply = "answer<|close|>response<|sep|><|close|>message<|sep|>"
        self.chunk_size = 3
        self.with_probabilities = False

    async def enqueue(self, generate_input):
        self.calls.append(generate_input)
        ids = self.tokenizer.encode(self.reply)

        async def replay():
            for start in range(0, len(ids), self.chunk_size):
                end = min(start + self.chunk_size, len(ids))
                probabilities = None
                if self.with_probabilities:
                    probabilities = torch.zeros(
                        (1, end - start, len(self.tokenizer)), dtype=torch.float32
                    )
                    for row, token_id in enumerate(ids[start:end]):
                        selected = 0.6 + 0.01 * ((start + row) % 10)
                        probabilities[0, row, token_id] = selected
                        probabilities[0, row, 0 if token_id != 0 else 1] = 1 - selected
                yield GenerateOutputs(
                    [
                        GenerateOutput(
                            output_ids=torch.tensor([ids[start:end]], dtype=torch.int),
                            all_probs=probabilities,
                            finished=torch.tensor([end == len(ids)]),
                            aux_info=AuxInfo(
                                input_len=generate_input.input_length,
                                output_len=end,
                                step_output_len=end - start,
                            ),
                        )
                    ]
                )

        return replay()


class Controller:
    max_concurrency = 4

    def __init__(self):
        self.active = 0

    def increment(self):
        self.active += 1
        return self.active

    def decrement(self):
        self.active -= 1
        return self.active


class KimiK3HttpCheckpointSmokeTest(unittest.TestCase):
    @staticmethod
    def image_url(color):
        image = Image.new("RGB", (28, 28), color)
        data = io.BytesIO()
        image.save(data, format="PNG")
        return "data:image/png;base64," + base64.b64encode(data.getvalue()).decode()

    @classmethod
    def setUpClass(cls):
        checkpoint = os.environ["K3_CKPT_PATH"]
        cls.tokenizer = AutoTokenizer.from_pretrained(
            checkpoint, trust_remote_code=True
        )
        cls.checkpoint = checkpoint

    def setUp(self):
        model_config = ModelConfig()
        model_config.model_type = "kimi_k3"
        model_config.model_name = "kimi_k3"
        model_config.ckpt_path = self.checkpoint
        model_config.max_seq_len = 8192
        model_config.template_type = None
        model_config.special_tokens = SpecialTokens()
        model_config.generate_env_config = GenerateEnvConfig()
        model_config.render_config = RenderConfig()
        self.backend = ReplayBackend(self.tokenizer)
        endpoint = OpenaiEndpoint(
            model_config=model_config,
            misc_config=PyMiscellaneousConfig(),
            vit_config=VitConfig(),
            tokenizer=self.tokenizer,
            backend_rpc_server_visitor=self.backend,
        )
        self.assertIsInstance(endpoint.chat_renderer, KimiK3Renderer)

        frontend = FrontendServer.__new__(FrontendServer)
        frontend._openai_endpoint = endpoint
        frontend._frontend_worker = SimpleNamespace(
            is_streaming=lambda request: bool(request.get("stream", False))
        )
        frontend._global_controller = Controller()
        frontend._access_logger = Mock()
        frontend.rank_id = "0"
        frontend.server_id = "0"
        frontend.is_embedding = False
        frontend.py_env_configs = SimpleNamespace(
            server_config=SimpleNamespace(ip="127.0.0.1", server_port=0),
            model_args=SimpleNamespace(model_type="kimi_k3"),
        )

        owner = FrontendApp.__new__(FrontendApp)
        owner.frontend_server = frontend
        owner.shutdown_manager = FrontendShutdownManager()
        owner.separated_frontend = True
        owner.server_config = SimpleNamespace(http_port=0)
        owner.grpc_client = None
        self.client = TestClient(owner.create_app())
        self.frontend = frontend

    def test_thinking_stream_preserves_channel_order_and_request_defaults(self):
        self.backend.reply = (
            "reasoning<|close|>think<|sep|><|open|>response<|sep|>"
            "answer<|close|>response<|sep|><|close|>message<|sep|>"
        )
        response = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "kimi_k3",
                "messages": [{"role": "user", "content": "Question?"}],
                "enable_thinking": True,
                "stream": True,
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        frames = [
            json.loads(line.removeprefix("data: "))
            for line in response.text.splitlines()
            if line.startswith("data: ")
        ]
        self.assertTrue(frames, response.text)
        deltas = [
            frame["choices"][0]["delta"] for frame in frames if frame.get("choices")
        ]
        self.assertEqual(
            "".join(delta.get("reasoning_content", "") for delta in deltas),
            "reasoning",
        )
        self.assertEqual(
            "".join(delta.get("content", "") for delta in deltas), "answer"
        )
        for delta in deltas:
            self.assertLessEqual(
                sum(
                    key in delta
                    for key in ("reasoning_content", "content", "tool_calls")
                ),
                1,
            )
        config = self.backend.calls[0].generate_config
        self.assertTrue(config.in_think_mode)
        self.assertEqual(config.temperature, 1.0)
        self.assertEqual(config.top_p, 0.95)
        self.assertEqual(self.frontend._global_controller.active, 0)

    def test_multi_token_target_probabilities_follow_visible_channel(self):
        content = "first second third"
        content_ids = self.tokenizer.encode(content)
        self.assertGreater(len(content_ids), 1)
        self.backend.reply = content + "<|close|>response<|sep|><|close|>message<|sep|>"
        self.backend.chunk_size = len(self.tokenizer.encode(self.backend.reply))
        self.backend.with_probabilities = True
        response = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "kimi_k3",
                "messages": [{"role": "user", "content": "Question?"}],
                "enable_thinking": False,
                "logprobs": True,
                "top_logprobs": 2,
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        choice = response.json()["choices"][0]
        self.assertEqual(choice["message"]["content"], content)
        visible = choice["logprobs"]["content"]
        self.assertEqual(len(visible), len(content_ids), choice)
        for i, item in enumerate(visible):
            self.assertEqual(item["token"], self.tokenizer.decode([content_ids[i]]))
            self.assertAlmostEqual(
                item["logprob"], math.log(0.6 + 0.01 * (i % 10)), places=5
            )
            self.assertEqual(len(item["top_logprobs"]), 2)
        self.assertEqual(
            self.backend.calls[0].generate_config.return_all_probs,
            ReturnAllProbsMode.DEFAULT,
        )

    def test_streamed_target_probabilities_cross_think_boundary(self):
        self.backend.reply = (
            "reasoning<|close|>think<|sep|><|open|>response<|sep|>"
            "answer<|close|>response<|sep|><|close|>message<|sep|>"
        )
        self.backend.with_probabilities = True
        response = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "kimi_k3",
                "messages": [{"role": "user", "content": "Question?"}],
                "enable_thinking": True,
                "stream": True,
                "logprobs": True,
                "top_logprobs": 2,
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        frames = [
            json.loads(line.removeprefix("data: "))
            for line in response.text.splitlines()
            if line.startswith("data: ")
        ]
        reasoning_probs = []
        content_probs = []
        for frame in frames:
            for choice in frame.get("choices", []):
                delta = choice["delta"]
                entries = (choice.get("logprobs") or {}).get("content") or []
                if delta.get("reasoning_content"):
                    reasoning_probs.extend(entries)
                elif delta.get("content"):
                    content_probs.extend(entries)
                else:
                    self.assertFalse(entries, choice)
        self.assertEqual(
            [entry["token"] for entry in reasoning_probs],
            [
                self.tokenizer.decode([token])
                for token in self.tokenizer.encode("reasoning")
            ],
        )
        self.assertEqual(
            [entry["token"] for entry in content_probs],
            [
                self.tokenizer.decode([token])
                for token in self.tokenizer.encode("answer")
            ],
        )

    def test_image_http_request_keeps_one_native_placeholder(self):
        url = self.image_url((128, 64, 32))
        response = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "kimi_k3",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe the image"},
                            {"type": "image_url", "image_url": {"url": url}},
                        ],
                    }
                ],
                "enable_thinking": False,
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["choices"][0]["message"]["content"], "answer")
        call = self.backend.calls[0]
        self.assertEqual(len(call.mm_inputs), 1)
        placeholder_ids = self.tokenizer.encode("<|media_pad|>")
        self.assertEqual(len(placeholder_ids), 1)
        self.assertEqual(call.token_ids.tolist()[0].count(placeholder_ids[0]), 1)
        self.assertFalse(call.generate_config.in_think_mode)
        self.assertEqual(call.generate_config.temperature, 0.6)

    def test_two_images_keep_url_and_placeholder_order(self):
        urls = [self.image_url((128, 64, 32)), self.image_url((32, 64, 128))]
        response = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "kimi_k3",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": urls[0]}},
                            {"type": "text", "text": "Compare these images"},
                            {"type": "image_url", "image_url": {"url": urls[1]}},
                        ],
                    }
                ],
                "enable_thinking": False,
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        call = self.backend.calls[0]
        self.assertEqual([mm.url for mm in call.mm_inputs], urls)
        decoded = [
            KimiK3ImageEmbedding.preprocess_input([mm], VitConfig())
            for mm in call.mm_inputs
        ]
        self.assertEqual(
            [image.getpixel((0, 0)) for image in decoded],
            [(128, 64, 32), (32, 64, 128)],
        )
        placeholder_id = self.tokenizer.encode("<|media_pad|>")[0]
        self.assertEqual(call.token_ids.tolist()[0].count(placeholder_id), 2)

    def test_required_tool_call_uses_xtml_constraint_and_http_response(self):
        self.backend.reply = (
            "<|close|>response<|sep|><|open|>tools<|sep|>"
            '<|open|>call tool="get_weather" index="1"<|sep|>'
            '<|open|>json type="object"<|sep|>{"city":"杭州"}'
            "<|close|>json<|sep|><|close|>call<|sep|><|close|>tools<|sep|>"
        )
        response = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "kimi_k3",
                "messages": [{"role": "user", "content": "Weather?"}],
                "enable_thinking": False,
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "Return weather",
                            "parameters": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                                "required": ["city"],
                                "additionalProperties": False,
                            },
                        },
                    }
                ],
                "tool_choice": "required",
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        choice = response.json()["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls", choice)
        calls = choice["message"]["tool_calls"]
        self.assertEqual(len(calls), 1)
        self.assertRegex(calls[0]["id"], r"^get_weather_0_[0-9a-f]{8}$")
        self.assertEqual(calls[0]["function"]["name"], "get_weather")
        self.assertEqual(
            json.loads(calls[0]["function"]["arguments"]), {"city": "杭州"}
        )
        constraint = self.backend.calls[0].generate_config.structural_tag
        self.assertIn(
            'tool="get_weather"',
            constraint["format"]["content"]["tags"][0]["begin"],
        )

    def test_tool_call_id_continues_valid_history_over_http(self):
        history_id = "get_weather_4_aaaaaaaa"
        self.backend.reply = (
            "<|close|>response<|sep|><|open|>tools<|sep|>"
            '<|open|>call tool="get_weather" index="1"<|sep|>'
            '<|open|>json type="object"<|sep|>{"city":"杭州"}'
            "<|close|>json<|sep|><|close|>call<|sep|><|close|>tools<|sep|>"
        )
        response = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "kimi_k3",
                "messages": [
                    {"role": "user", "content": "Weather?"},
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": history_id,
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city":"北京"}',
                                },
                            }
                        ],
                    },
                    {"role": "tool", "tool_call_id": history_id, "content": "sunny"},
                    {"role": "user", "content": "How about Hangzhou?"},
                ],
                "enable_thinking": False,
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "Return weather",
                            "parameters": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                                "required": ["city"],
                                "additionalProperties": False,
                            },
                        },
                    }
                ],
                "tool_choice": "required",
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        call = response.json()["choices"][0]["message"]["tool_calls"][0]
        self.assertRegex(call["id"], r"^get_weather_5_[0-9a-f]{8}$")
        self.assertNotEqual(call["id"], history_id)
        self.assertNotEqual(call["id"][-8:], history_id[-8:])
        self.assertEqual(json.loads(call["function"]["arguments"]), {"city": "杭州"})

    def test_four_parallel_tool_calls_over_http(self):
        self.backend.reply = "<|close|>response<|sep|><|open|>tools<|sep|>"
        for index, city in enumerate(("北京", "上海", "杭州", "深圳"), 1):
            self.backend.reply += (
                f'<|open|>call tool="get_weather" index="{index}"<|sep|>'
                '<|open|>json type="object"<|sep|>'
                f'{{"city":"{city}"}}'
                "<|close|>json<|sep|><|close|>call<|sep|>"
            )
        self.backend.reply += "<|close|>tools<|sep|>"
        response = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "kimi_k3",
                "messages": [{"role": "user", "content": "Four cities?"}],
                "enable_thinking": False,
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "Return weather",
                            "parameters": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                                "required": ["city"],
                                "additionalProperties": False,
                            },
                        },
                    }
                ],
                "tool_choice": "required",
                "parallel_tool_calls": True,
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        calls = response.json()["choices"][0]["message"]["tool_calls"]
        self.assertEqual(len(calls), 4)
        self.assertEqual([call["index"] for call in calls], [0, 1, 2, 3])
        self.assertEqual(
            [json.loads(call["function"]["arguments"])["city"] for call in calls],
            ["北京", "上海", "杭州", "深圳"],
        )
        for index, call in enumerate(calls):
            self.assertRegex(call["id"], rf"^get_weather_{index}_[0-9a-f]{{8}}$")
        self.assertEqual(len({call["id"] for call in calls}), 4)
        content = self.backend.calls[0].generate_config.structural_tag["format"][
            "content"
        ]
        self.assertEqual(content["type"], "sequence")

    def test_invalid_sampling_is_rejected_before_backend_enqueue(self):
        response = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "kimi_k3",
                "messages": [{"role": "user", "content": "Question?"}],
                "top_p": 0.8,
            },
        )
        self.assertEqual(response.status_code, 500, response.text)
        self.assertIn("top_p", response.text)
        self.assertEqual(self.backend.calls, [])


if __name__ == "__main__":
    unittest.main()
