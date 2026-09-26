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
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import torch
from fastapi.testclient import TestClient
from PIL import Image
from safetensors import safe_open
from transformers import AutoTokenizer

from rtp_llm.config.exceptions import ExceptionType
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
from rtp_llm.models.kimi_k3.kimi_k3 import KimiK3
from rtp_llm.multimodal.multimodal_mixins.kimi_k3.kimi_k3_vit import (
    KimiK3ImageEmbedding,
)
from rtp_llm.openai.openai_endpoint import OpenaiEndpoint
from rtp_llm.openai.renderers.kimi_k3_renderer import KimiK3Renderer
from rtp_llm.ops import SpecialTokens
from rtp_llm.utils.base_model_datatypes import (
    AuxInfo,
    GenerateOutput,
    GenerateOutputs,
    MMUrlType,
)


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

    async def batch_enqueue(self, generate_inputs):
        outputs = []
        for generate_input in generate_inputs:
            replay = await self.enqueue(generate_input)
            chunks = [chunk async for chunk in replay]
            if len(chunks) != 1:
                raise AssertionError("batch fixture requires one replay chunk")
            outputs.append(chunks[0])
        return outputs


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
    def image_url(color, size=(28, 28), mode="RGB"):
        image = Image.new(mode, size, color)
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
        model_config = KimiK3._create_config(self.checkpoint)
        model_config.model_type = "kimi_k3"
        model_config.model_name = "kimi_k3"
        model_config.ckpt_path = self.checkpoint
        model_config.max_seq_len = 8192
        model_config.template_type = None
        model_config.special_tokens = SpecialTokens()
        model_config.generate_env_config = GenerateEnvConfig()
        model_config.render_config = RenderConfig()
        self.model_config = model_config
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
        self.assertEqual(config.max_new_tokens, 131072)
        self.assertEqual(config.temperature, 1.0)
        self.assertEqual(config.top_p, 0.95)
        self.assertEqual(self.frontend._global_controller.active, 0)

    def test_single_batch_and_chat_render_share_k3_generation_config(self):
        self.backend.chunk_size = 1_000_000
        request = {
            "model": "kimi_k3",
            "messages": [{"role": "user", "content": "Question?"}],
            "enable_thinking": True,
            "thinking_budget": 64,
            "temperature": 0.7,
            "top_p": 0.95,
            "max_completion_tokens": 128,
        }

        single = self.client.post("/v1/chat/completions", json=request)
        self.assertEqual(single.status_code, 200, single.text)
        rendered = self.client.post("/v1/chat/render", json=request)
        self.assertEqual(rendered.status_code, 200, rendered.text)
        batch = self.client.post(
            "/v1/batch/chat/completions", json={"requests": [request, request]}
        )
        self.assertEqual(batch.status_code, 200, batch.text)
        self.assertEqual(len(batch.json()["responses"]), 2)
        self.assertEqual(len(self.backend.calls), 3)

        expected = {
            "max_new_tokens": 128,
            "max_thinking_tokens": 64,
            "temperature": 0.7,
            "top_p": 0.95,
            "in_think_mode": True,
        }
        rendered_config = rendered.json()["generate_config"]
        for field, value in expected.items():
            self.assertEqual(rendered_config[field], value, field)
            for call in self.backend.calls:
                self.assertEqual(getattr(call.generate_config, field), value, field)
        prompt_ids = rendered.json()["input_ids"]
        for call in self.backend.calls:
            self.assertEqual(call.token_ids.tolist()[0], prompt_ids)
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
        usage = response.json()["usage"]
        expected_prompt_tokens = self.backend.calls[0].input_length - (
            KimiK3Renderer._pending_prompt_token_count(self.tokenizer, False)
        )
        expected_completion_tokens = len(self.tokenizer.encode(self.backend.reply))
        self.assertEqual(usage["prompt_tokens"], expected_prompt_tokens)
        self.assertEqual(usage["completion_tokens"], expected_completion_tokens)
        self.assertEqual(
            usage["total_tokens"],
            expected_prompt_tokens + expected_completion_tokens,
        )
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

    def test_non_image_media_is_rejected_before_backend_enqueue(self):
        for media_type in ("video_url", "audio_url"):
            with self.subTest(media_type=media_type):
                response = self.client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "kimi_k3",
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": media_type,
                                        media_type: {"url": "https://example.com/media"},
                                    }
                                ],
                            }
                        ],
                    },
                )
                self.assertGreaterEqual(response.status_code, 400, response.text)
                self.assertIn("supports only text and image_url", response.text)
                self.assertEqual(self.backend.calls, [])

    def test_image_thinking_stream_has_terminal_frame(self):
        image_url = self.image_url((128, 64, 32))
        self.backend.reply = (
            "reasoning<|close|>think<|sep|><|open|>response<|sep|>"
            "answer<|close|>response<|sep|><|close|>message<|sep|>"
        )
        response = self.client.post(
            "/v1/chat/completions",
            json={
                "model": "kimi_k3",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe the image"},
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ],
                    }
                ],
                "enable_thinking": True,
                "thinking_budget": 100,
                "stream": True,
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        frames = [
            json.loads(line.removeprefix("data: "))
            for line in response.text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        choices = [choice for frame in frames for choice in frame.get("choices", [])]
        self.assertEqual(
            [
                choice.get("finish_reason")
                for choice in choices
                if choice.get("finish_reason")
            ],
            ["stop"],
        )
        self.assertEqual(choices[-1].get("finish_reason"), "stop")
        self.assertEqual(
            "".join(choice["delta"].get("reasoning_content", "") for choice in choices),
            "reasoning",
        )
        self.assertEqual(
            "".join(choice["delta"].get("content", "") for choice in choices),
            "answer",
        )
        call = self.backend.calls[0]
        self.assertEqual([item.url for item in call.mm_inputs], [image_url])
        media_pad = self.tokenizer.encode("<|media_pad|>")
        self.assertEqual(len(media_pad), 1)
        self.assertEqual(call.token_ids.tolist()[0].count(media_pad[0]), 1)

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

    def test_http_images_reach_checkpoint_vision_embeddings(self):
        sizes = ((28, 28), (56, 28), (28, 28))
        urls = [
            self.image_url((128, 64, 32), sizes[0]),
            self.image_url((32, 64, 128), sizes[1]),
            self.image_url((255, 0, 0, 0), sizes[2], mode="RGBA"),
        ]
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
                            {"type": "image_url", "image_url": {"url": urls[2]}},
                        ],
                    }
                ],
                "enable_thinking": False,
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        request = self.backend.calls[0]
        self.assertEqual([item.url for item in request.mm_inputs], urls)

        embedding = KimiK3ImageEmbedding(self.model_config.mm_related_params)
        checkpoint = Path(self.checkpoint)
        weight_map = json.loads(
            (checkpoint / "model.safetensors.index.json").read_text()
        )["weight_map"]
        for prefix, module in (
            ("vision_tower.", embedding.vision_tower),
            ("mm_projector.", embedding.mm_projector),
        ):
            state = {}
            for shard_name in sorted(
                {name for key, name in weight_map.items() if key.startswith(prefix)}
            ):
                with safe_open(
                    checkpoint / shard_name, framework="pt", device="cpu"
                ) as shard:
                    for key in shard.keys():
                        if key.startswith(prefix):
                            state[key.removeprefix(prefix)] = shard.get_tensor(key)
            module.load_state_dict(state, strict=True)
            del state
            module.to(
                device="cuda" if torch.cuda.is_available() else "cpu",
                dtype=torch.bfloat16,
            ).eval()

        images = [
            KimiK3ImageEmbedding.preprocess_input([item], VitConfig())
            for item in request.mm_inputs
        ]
        self.assertEqual([image.size for image in images], list(sizes))
        self.assertEqual(images[2].mode, "RGBA")
        media_config = embedding.image_processor.media_proc_cfg
        self.assertEqual(media_config["transparent_bg_fill_stage"], "after_resize")
        background = media_config["transparent_bg_config"]
        self.assertEqual(background["pattern"], "chessboard")
        transparent_pixels = embedding.image_processor.preprocess(
            {"image": images[2]}, return_tensors="pt"
        ).pixel_values
        patch_size = media_config["patch_size"]
        self.assertEqual(tuple(transparent_pixels.shape), (4, 3, 14, 14))
        for x, y in ((0, 0), (8, 0), (8, 8), (15, 15), (27, 27)):
            square_size = background["chessboard_square_size"]
            gray = (x // square_size + y // square_size) % 2 == int(
                background["chessboard_square_on_top_left"]
            )
            value = (
                background["chessboard_gray_value"]
                if gray
                else background["chessboard_white_value"]
            )
            patch = (y // patch_size) * 2 + x // patch_size
            expected = torch.full((3,), 2 * value / 255 - 1)
            torch.testing.assert_close(
                transparent_pixels[patch, :, y % patch_size, x % patch_size],
                expected,
            )
        assembled = embedding.batched_embedding(images, [MMUrlType.IMAGE] * 3)
        vision = embedding.image_embedding(images)
        self.assertEqual(len(assembled), 3)
        self.assertEqual(
            [tuple(features.shape) for features in vision],
            [(1, 7168), (2, 7168), (1, 7168)],
        )
        media_pad_id = self.tokenizer.encode("<|media_pad|>")[0]
        self.assertEqual(request.token_ids.tolist()[0].count(media_pad_id), 3)
        for image, (full, position), features in zip(images, assembled, vision):
            self.assertIsNone(position)
            width, height = image.size
            prompt_ids = self.tokenizer.encode(
                f"<|media_begin|>image {width}x{height}"
                "<|media_content|><|media_pad|><|media_end|>"
            )
            self.assertEqual(prompt_ids.count(media_pad_id), 1)
            pad_index = prompt_ids.index(media_pad_id)
            self.assertEqual(full.shape[0], len(prompt_ids) - 1 + features.shape[0])
            self.assertEqual(full.shape[1], 7168)
            torch.testing.assert_close(
                full[pad_index : pad_index + features.shape[0]], features
            )
            torch.testing.assert_close(
                full[:pad_index],
                embedding._word_embedding_weight[prompt_ids[:pad_index]].to(
                    device=full.device, dtype=full.dtype
                ),
            )
            torch.testing.assert_close(
                full[pad_index + features.shape[0] :],
                embedding._word_embedding_weight[prompt_ids[pad_index + 1 :]].to(
                    device=full.device, dtype=full.dtype
                ),
            )

        # Use this HTTP request's token IDs and checkpoint-backed image
        # features together in the production C++ placeholder expansion.
        binary = (
            Path(os.environ["TEST_SRCDIR"])
            / os.environ["TEST_WORKSPACE"]
            / "rtp_llm/cpp/multimodal_processor/test/kimi_k3_feature_expand_check"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            token_ids = request.token_ids.reshape(-1).to(torch.int32).cpu()
            token_path = Path(temp_dir) / "token_ids.raw"
            token_path.write_bytes(token_ids.numpy().tobytes())
            command = [
                str(binary),
                str(media_pad_id),
                str(token_path),
                str(token_ids.numel()),
            ]
            for index, (mm_input, (full, _)) in enumerate(
                zip(request.mm_inputs, assembled)
            ):
                self.assertEqual(full.dtype, torch.bfloat16)
                path = Path(temp_dir) / f"image_{index}.raw"
                path.write_bytes(
                    full.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
                )
                command.extend(
                    (mm_input.url, str(int(mm_input.mm_type)), str(path), str(full.shape[0]))
                )
            completed = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_auto_tool_choice_keeps_plain_and_typed_tool_http_paths(self):
        request = {
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
            "tool_choice": "auto",
        }

        self.backend.reply = (
            "<|close|>response<|sep|><|open|>tools<|sep|>"
            '<|open|>call tool="get_weather" index="1"<|sep|>'
            '<|open|>argument key="city" type="string"<|sep|>杭州'
            "<|close|>argument<|sep|><|close|>call<|sep|>"
            "<|close|>tools<|sep|><|close|>message<|sep|>"
        )
        tool_response = self.client.post("/v1/chat/completions", json=request)
        self.assertEqual(tool_response.status_code, 200, tool_response.text)
        tool_choice = tool_response.json()["choices"][0]
        self.assertEqual(tool_choice["finish_reason"], "tool_calls")
        tool_call = tool_choice["message"]["tool_calls"][0]
        self.assertEqual(tool_call["function"]["name"], "get_weather")
        self.assertEqual(
            json.loads(tool_call["function"]["arguments"]), {"city": "杭州"}
        )
        self.assertIsNotNone(self.backend.calls[0].generate_config.structural_tag)

        self.backend.reply = (
            "sunny<|close|>response<|sep|><|close|>message<|sep|>"
        )
        plain_response = self.client.post("/v1/chat/completions", json=request)
        self.assertEqual(plain_response.status_code, 200, plain_response.text)
        plain_choice = plain_response.json()["choices"][0]
        self.assertEqual(plain_choice["message"]["content"], "sunny")
        self.assertFalse(plain_choice["message"].get("tool_calls"))
        self.assertIsNotNone(self.backend.calls[1].generate_config.structural_tag)

        request.pop("tool_choice")
        self.backend.reply = (
            "<|close|>response<|sep|><|open|>tools<|sep|>"
            '<|open|>call tool="get_weather" index="1"<|sep|>'
            '<|open|>argument key="city" type="string"<|sep|>杭州'
            "<|close|>argument<|sep|><|close|>call<|sep|>"
            "<|close|>tools<|sep|><|close|>message<|sep|>"
        )
        default_response = self.client.post("/v1/chat/completions", json=request)
        self.assertEqual(default_response.status_code, 200, default_response.text)
        default_choice = default_response.json()["choices"][0]
        self.assertEqual(default_choice["finish_reason"], "tool_calls")
        self.assertEqual(
            json.loads(
                default_choice["message"]["tool_calls"][0]["function"]["arguments"]
            ),
            {"city": "杭州"},
        )
        self.assertIsNotNone(self.backend.calls[2].generate_config.structural_tag)

    def test_default_tool_choice_preserves_json_shaped_string_in_http_responses(self):
        request = {
            "model": "kimi_k3",
            "messages": [
                {
                    "role": "user",
                    "content": "Require output-format unless quiet or verbose",
                }
            ],
            "enable_thinking": False,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "OptionSpecBuilder_requiredUnless",
                        "description": (
                            "Require an option unless a dependent is present"
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "dependent": {"type": "string"},
                                "otherDependents": {"type": "string"},
                            },
                            "required": ["dependent"],
                        },
                    },
                }
            ],
        }
        self.backend.reply = (
            "<|close|>response<|sep|><|open|>tools<|sep|>"
            '<|open|>call tool="OptionSpecBuilder_requiredUnless" index="1"<|sep|>'
            '<|open|>argument key="dependent" type="string"<|sep|>output-format'
            "<|close|>argument<|sep|>"
            '<|open|>argument key="otherDependents" type="string"<|sep|>'
            '["quiet","verbose"]<|close|>argument<|sep|>'
            "<|close|>call<|sep|><|close|>tools<|sep|><|close|>message<|sep|>"
        )
        expected = {
            "dependent": "output-format",
            "otherDependents": '["quiet","verbose"]',
        }

        for stream in (False, True):
            with self.subTest(stream=stream):
                response = self.client.post(
                    "/v1/chat/completions", json={**request, "stream": stream}
                )
                self.assertEqual(response.status_code, 200, response.text)
                if not stream:
                    choice = response.json()["choices"][0]
                    self.assertEqual(choice["finish_reason"], "tool_calls")
                    call = choice["message"]["tool_calls"][0]
                else:
                    frames = [
                        json.loads(line.removeprefix("data: "))
                        for line in response.text.splitlines()
                        if line.startswith("data: ")
                    ]
                    choices = [
                        choice
                        for frame in frames
                        for choice in frame.get("choices", [])
                    ]
                    self.assertIn(
                        "tool_calls", [c.get("finish_reason") for c in choices]
                    )
                    deltas = [
                        item
                        for choice in choices
                        for item in choice["delta"].get("tool_calls") or []
                    ]
                    self.assertTrue(deltas, response.text)
                    self.assertEqual({item["index"] for item in deltas}, {0})
                    call = {
                        "id": next(item["id"] for item in deltas if item.get("id")),
                        "function": {
                            "name": next(
                                item["function"]["name"]
                                for item in deltas
                                if (item.get("function") or {}).get("name")
                            ),
                            "arguments": "".join(
                                (item.get("function") or {}).get("arguments") or ""
                                for item in deltas
                            ),
                        },
                    }
                self.assertRegex(
                    call["id"],
                    r"^OptionSpecBuilder_requiredUnless_0_[0-9a-f]{8}$",
                )
                self.assertEqual(
                    call["function"]["name"], "OptionSpecBuilder_requiredUnless"
                )
                self.assertEqual(
                    json.loads(call["function"]["arguments"]), expected
                )
                self.assertIsNotNone(
                    self.backend.calls[-1].generate_config.structural_tag
                )

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

    def test_image_request_preserves_tool_id_history(self):
        history_id = "get_weather_4_aaaaaaaa"
        image_url = self.image_url((128, 64, 32))
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
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": image_url}},
                            {"type": "text", "text": "How about Hangzhou?"},
                        ],
                    },
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
                            },
                        },
                    }
                ],
                "tool_choice": "required",
            },
        )

        self.assertEqual(response.status_code, 200, response.text)
        backend_input = self.backend.calls[0]
        self.assertEqual([item.url for item in backend_input.mm_inputs], [image_url])
        media_pad = self.tokenizer.encode("<|media_pad|>")[0]
        self.assertEqual(backend_input.token_ids.tolist()[0].count(media_pad), 1)
        call = response.json()["choices"][0]["message"]["tool_calls"][0]
        self.assertRegex(call["id"], r"^get_weather_5_[0-9a-f]{8}$")
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
        for name, value in (("top_p", 0.8), ("top_logprobs", -1)):
            with self.subTest(name=name):
                response = self.client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "kimi_k3",
                        "messages": [{"role": "user", "content": "Question?"}],
                        "logprobs": True,
                        name: value,
                    },
                )
                self.assertEqual(response.status_code, 500, response.text)
                self.assertIn(name, response.text)
                self.assertEqual(self.backend.calls, [])

    def test_non_positive_completion_limit_is_rejected_before_backend_enqueue(self):
        for name in ("max_completion_tokens", "max_tokens"):
            for value in (0, -1):
                with self.subTest(name=name, value=value):
                    response = self.client.post(
                        "/v1/chat/completions",
                        json={
                            "model": "kimi_k3",
                            "messages": [{"role": "user", "content": "Question?"}],
                            name: value,
                        },
                    )
                    self.assertGreaterEqual(response.status_code, 400, response.text)
                    self.assertEqual(
                        response.json()["error_code"], int(ExceptionType.INVALID_PARAMS)
                    )
                    self.assertIn(name, response.text)
                    self.assertEqual(self.backend.calls, [])


if __name__ == "__main__":
    unittest.main()
