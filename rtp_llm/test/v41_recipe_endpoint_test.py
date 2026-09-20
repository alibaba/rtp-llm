"""Exercise the actual RTP response/config path with CPU token-source fixtures."""

import asyncio
import io
import json
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from PIL import Image

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.config.generate_config import GenerateConfig
from rtp_llm.config.py_config_modules import (
    GenerateEnvConfig,
    PyEnvConfigs,
    RenderConfig,
)
from rtp_llm.frontend.frontend_server import FrontendServer
from rtp_llm.models.multimodal.deepseek_v41_processor import (
    IMAGE_PLACEHOLDER,
    V41ImageProcessorConfig,
    preprocess_image,
)
from rtp_llm.openai.api_datatype import ChatCompletionRequest
from rtp_llm.openai.openai_endpoint import OpenaiEndpoint
from rtp_llm.openai.renderers.basic_renderer import BasicRenderer
from rtp_llm.openai.renderers.custom_renderer import RendererParams
from rtp_llm.openai.renderers.deepseekv4_renderer import DeepseekV4Renderer
from rtp_llm.openai.renderers.deepseekv41_renderer import DeepseekV41Renderer
from rtp_llm.utils.base_model_datatypes import AuxInfo, GenerateOutput, GenerateOutputs


class CharacterTokenizer:
    def encode(self, text, **kwargs):
        return [ord(char) for char in text]

    def decode(self, ids, **kwargs):
        return "".join(chr(value) for value in ids)

    def tokenize(self, text):
        return list(text)

    def convert_tokens_to_ids(self, text):
        return ord(text) if len(text) == 1 else None

    def __len__(self):
        return 1114112


def make_renderer():
    renderer = DeepseekV41Renderer.__new__(DeepseekV41Renderer)
    renderer.tokenizer = CharacterTokenizer()
    renderer.think_mode = False
    renderer.think_start_tag = "<think>"
    renderer.think_end_tag = "</think>"
    renderer.eos_token_id = 0
    renderer.max_seq_len = 1048576
    renderer.extra_stop_words = []
    renderer.extra_stop_word_ids_list = []
    renderer.stop_words_id_list = []
    renderer.image_processor_config = V41ImageProcessorConfig()
    return renderer


def make_endpoint(renderer):
    endpoint = OpenaiEndpoint.__new__(OpenaiEndpoint)
    endpoint.chat_renderer = renderer
    endpoint.tokenizer = renderer.tokenizer
    endpoint.stop_words_str_list = []
    endpoint.stop_words_id_list = [[0]]
    endpoint.generate_env_config = GenerateEnvConfig()
    endpoint.template_renderer = BasicRenderer(
        renderer.tokenizer,
        RendererParams("deepseek_v41", 1048576, 0, []),
        endpoint.generate_env_config,
        RenderConfig(),
    )
    return endpoint


def request(**kwargs):
    return ChatCompletionRequest.model_validate(
        {"messages": [{"role": "user", "content": "hello"}], **kwargs}
    )


class EndpointConfigTest(unittest.TestCase):
    def setUp(self):
        self.renderer = make_renderer()
        self.endpoint = make_endpoint(self.renderer)
        self.endpoint.stop_words_str_list = ["engine_stop"]

    def test_partial_text_preserves_v41_ids_and_masks(self):
        suffix = " answer prefix"
        prepared = self.renderer.render_chat(request()).v41_inputs
        rendered = self.endpoint.render_chat(
            request(
                messages=[
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": suffix, "partial": True},
                ]
            )
        )
        suffix_ids = tuple(self.renderer.tokenizer.encode(suffix))
        self.assertEqual(rendered.v41_inputs.token_ids, prepared.token_ids + suffix_ids)
        self.assertEqual(rendered.input_ids, list(rendered.v41_inputs.token_ids))
        self.assertEqual(rendered.rendered_prompt, prepared.prompt + suffix)
        self.assertEqual(
            rendered.v41_inputs.token_types,
            prepared.token_types + (-1,) * len(suffix_ids),
        )
        self.assertFalse(rendered.v41_inputs.image_mask.any().item())

    def test_literal_image_placeholder_rejected_in_text_and_partial(self):
        for partial in (False, True):
            with self.subTest(partial=partial):
                req = request(
                    messages=[
                        {"role": "user", "content": "hello"},
                        {
                            "role": "assistant",
                            "content": "prefix " + IMAGE_PLACEHOLDER + " suffix",
                            "partial": partial,
                        },
                    ]
                )
                with self.assertRaisesRegex(
                    FtRuntimeException, "literal V4.1 image placeholders"
                ) as raised:
                    self.endpoint.render_chat(req)
                self.assertEqual(
                    raised.exception.exception_type, ExceptionType.INVALID_PARAMS
                )

    def test_partial_placeholder_keeps_basic_renderer_behavior(self):
        rendered = self.endpoint.render_chat(
            request(
                messages=[
                    {"role": "user", "content": "hello"},
                    {
                        "role": "assistant",
                        "content": IMAGE_PLACEHOLDER,
                        "partial": True,
                    },
                ],
                user_template="{{ messages[0].content }}",
            )
        )
        self.assertIsNone(rendered.v41_inputs)
        self.assertEqual(rendered.rendered_prompt, "hello" + IMAGE_PLACEHOLDER)
        self.assertEqual(
            rendered.input_ids,
            self.renderer.tokenizer.encode("hello" + IMAGE_PLACEHOLDER),
        )

    def test_user_template_uses_basic_renderer_stop_and_thinking_rules(self):
        for stop in ("STOP", ["STOP"]):
            for streaming in (False, True):
                with self.subTest(stop=stop, streaming=streaming):
                    req = request(
                        user_template="{{ messages[0].content }}",
                        stop=stop,
                        stream=streaming,
                        thinking={"type": "enabled"},
                    )
                    config = self.endpoint._extract_generation_config(req)
                    self.assertIn("STOP", config.stop_words_str)
                    self.assertIn(
                        self.renderer.tokenizer.encode("STOP"), config.stop_words_list
                    )
                    self.assertEqual(config.is_streaming, streaming)
                    self.assertFalse(config.in_think_mode)

    def test_explicit_response_format_schema_survives_renderer_constraints(self):
        schema = {
            "type": "object",
            "properties": {"answer": {"type": "integer"}},
            "required": ["answer"],
            "additionalProperties": False,
        }
        req = request(
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "answer", "strict": True, "schema": schema},
            }
        )
        config = self.endpoint._extract_generation_config(req)
        self.renderer.apply_chat_completion_constraints(req, config)
        self.assertEqual(json.loads(config.json_schema), schema)
        self.assertIsNone(config.structural_tag)

    def test_user_stops_never_reach_engine_and_config_is_not_mutated(self):
        req = request(
            stop=["STOP"], extra_configs={"stop_words_str": ["explicit_engine_stop"]}
        )
        before = req.model_dump()
        config = self.endpoint._extract_generation_config(req)
        self.assertNotIn("STOP", config.stop_words_str)
        self.assertNotIn(self.renderer.tokenizer.encode("STOP"), config.stop_words_list)
        self.assertIn("engine_stop", config.stop_words_str)
        self.assertIn("explicit_engine_stop", config.stop_words_str)
        self.assertIn([0], config.stop_words_list)
        self.assertEqual(req.model_dump(), before)

    def test_engine_thinking_matches_prompt_and_parser(self):
        for fields, expected in (
            ({"reasoning_effort": "low"}, True),
            ({"reasoning_effort": "none"}, False),
            ({"reasoning_effort": "none", "thinking": {"type": "enabled"}}, True),
            ({"enable_thinking": False, "thinking": {"type": "enabled"}}, True),
            ({"thinking": {"type": "disabled"}, "reasoning_effort": "max"}, False),
        ):
            req = request(**fields)
            config = self.endpoint._extract_generation_config(req)
            self.assertEqual(config.in_think_mode, expected)
            self.assertEqual(self.renderer.in_think_mode(req), expected)
            if expected:
                self.assertTrue(config.begin_think_token_ids)
                self.assertTrue(config.end_think_token_ids)

    def test_forced_prefix_preserves_explicit_grammar(self):
        req = request(
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "description": "",
                        "parameters": {"type": "object"},
                    },
                }
            ],
            tool_choice="required",
        )
        config = GenerateConfig(regex="existing grammar")
        self.renderer.apply_chat_completion_constraints(req, config)
        self.assertEqual(config.regex, "existing grammar")
        self.assertIsNone(config.structural_tag)
        legacy = DeepseekV4Renderer.__new__(DeepseekV4Renderer)
        self.assertEqual(legacy._normalize_reasoning_effort("xhigh"), "max")
        self.assertIsNone(legacy._normalize_reasoning_effort("low"))


class EndpointAsyncPreparationTest(unittest.IsolatedAsyncioTestCase):
    async def test_partial_image_placeholder_rejected_before_backend(self):
        endpoint = make_endpoint(make_renderer())
        for streaming in (False, True):
            with self.subTest(stream=streaming), patch.object(
                endpoint, "_chat_completion_from_inputs"
            ) as start_backend:
                req = request(
                    messages=[
                        {"role": "user", "content": "hello"},
                        {
                            "role": "assistant",
                            "content": IMAGE_PLACEHOLDER,
                            "partial": True,
                        },
                    ],
                    stream=streaming,
                )
                with self.assertRaisesRegex(
                    FtRuntimeException, "literal V4.1 image placeholders"
                ) as raised:
                    await endpoint.chat_completion_async(
                        1, req, SimpleNamespace(headers={})
                    )
                self.assertEqual(
                    raised.exception.exception_type, ExceptionType.INVALID_PARAMS
                )
                start_backend.assert_not_called()

    async def test_cancel_during_preparation_does_not_start_backend(self):
        endpoint = make_endpoint(make_renderer())
        loop = asyncio.get_running_loop()
        started = asyncio.Event()
        finished = asyncio.Event()
        release = threading.Event()
        render_chat = endpoint.render_chat

        def slow_render(req):
            loop.call_soon_threadsafe(started.set)
            try:
                if not release.wait(10):
                    raise TimeoutError("test did not release preparation")
                return render_chat(req)
            finally:
                loop.call_soon_threadsafe(finished.set)

        with patch.object(
            endpoint, "render_chat", side_effect=slow_render
        ), patch.object(endpoint, "_chat_completion_from_inputs") as start_backend:
            task = asyncio.create_task(
                endpoint.chat_completion_async(
                    1, request(), SimpleNamespace(headers={})
                )
            )
            try:
                await asyncio.wait_for(started.wait(), timeout=5)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
            finally:
                release.set()
                await asyncio.wait_for(finished.wait(), timeout=5)
                await asyncio.gather(task, return_exceptions=True)
            start_backend.assert_not_called()

    async def test_slow_image_keeps_text_and_existing_sse_responsive(self):
        class ImageTokenizer(CharacterTokenizer):
            def encode(self, text, **kwargs):
                tokens = []
                for index, part in enumerate(text.split(IMAGE_PLACEHOLDER)):
                    if index:
                        tokens.append(129264)
                    tokens.extend(super().encode(part, **kwargs))
                return tokens

            def convert_tokens_to_ids(self, text):
                if text == IMAGE_PLACEHOLDER:
                    return 129264
                return super().convert_tokens_to_ids(text)

        renderer = make_renderer()
        renderer.tokenizer = ImageTokenizer()
        endpoint = make_endpoint(renderer)
        server = FrontendServer.__new__(FrontendServer)
        server._openai_endpoint = endpoint
        server._frontend_worker = SimpleNamespace(
            is_streaming=lambda req: req.get("stream", False)
        )
        server._access_logger = MagicMock()
        server._global_controller = MagicMock()
        server._global_controller.increment.return_value = 1
        server.py_env_configs = PyEnvConfigs()
        server.rank_id = server.server_id = "0"
        raw_request = SimpleNamespace(headers={})

        async def connected():
            return False

        raw_request.is_disconnected = connected
        loop = asyncio.get_running_loop()
        loop_thread = threading.get_ident()
        download_started = asyncio.Event()
        release_download = threading.Event()
        continue_sse = asyncio.Event()
        received = []
        preparation_threads = []
        image_bytes = io.BytesIO()
        Image.new("RGB", (42, 84), color="red").save(image_bytes, format="PNG")

        def slow_download(url):
            preparation_threads.append(threading.get_ident())
            loop.call_soon_threadsafe(download_started.set)
            if not release_download.wait(10):
                raise TimeoutError("test did not release image download")
            return io.BytesIO(image_bytes.getvalue())

        def prepare_image(*args, **kwargs):
            preparation_threads.append(threading.get_ident())
            return preprocess_image(*args, **kwargs)

        def check_loop():
            self.assertIs(asyncio.get_running_loop(), loop)
            self.assertEqual(threading.get_ident(), loop_thread)

        generate_choice = renderer.generate_choice

        def create_choices(*args, **kwargs):
            check_loop()
            return generate_choice(*args, **kwargs)

        class Visitor:
            async def enqueue(self, inputs):
                check_loop()
                received.append(inputs)
                is_sse = "sse" in inputs.v41_inputs.prompt

                async def source():
                    chunks = ("first", " second") if is_sse else ("answer STOP hidden",)
                    count = 0
                    for index, chunk in enumerate(chunks):
                        if is_sse and index:
                            await continue_sse.wait()
                        check_loop()
                        ids = [ord(char) for char in chunk]
                        finished = index == len(chunks) - 1
                        if finished:
                            ids.append(0)
                        count += len(ids)
                        yield GenerateOutputs(
                            generate_outputs=[
                                GenerateOutput(
                                    output_ids=torch.tensor([ids]),
                                    finished=finished,
                                    aux_info=AuxInfo(
                                        input_len=inputs.token_ids.numel(),
                                        output_len=count,
                                    ),
                                )
                            ]
                        )

                return source()

        endpoint.backend_rpc_server_visitor = Visitor()
        image_request = request(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "image"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://image.invalid/slow.png"},
                        },
                    ],
                }
            ],
            stop="STOP",
        )
        image_before = image_request.model_dump()
        with patch.object(
            renderer, "generate_choice", side_effect=create_choices
        ), patch(
            "rtp_llm.utils.multimodal_util.get_bytes_io_from_url",
            side_effect=slow_download,
        ), patch(
            "rtp_llm.models.multimodal.deepseek_v41_processor.preprocess_image",
            side_effect=prepare_image,
        ):
            sse = await server.chat_completion(
                request(messages=[{"role": "user", "content": "sse"}], stream=True),
                raw_request,
            )
            chunks = [await anext(sse.body_iterator), await anext(sse.body_iterator)]
            image_task = asyncio.create_task(
                server.chat_completion(image_request, raw_request)
            )
            try:
                await asyncio.wait_for(download_started.wait(), timeout=5)
                text = await asyncio.wait_for(
                    server.chat_completion(request(stop="STOP"), raw_request), timeout=5
                )
                self.assertEqual(text.status_code, 200)
                self.assertEqual(
                    json.loads(text.body)["choices"][0]["message"]["content"], "answer "
                )
                continue_sse.set()

                async def drain_sse():
                    return [chunk async for chunk in sse.body_iterator]

                chunks.extend(await asyncio.wait_for(drain_sse(), timeout=5))
                self.assertEqual(chunks[-1], "data: [DONE]\r\n\r\n")
                content = "".join(
                    json.loads(chunk.removeprefix("data: "))["choices"][0]["delta"].get(
                        "content", ""
                    )
                    for chunk in chunks[:-1]
                )
                self.assertEqual(content, "first second")
                self.assertFalse(image_task.done())
                self.assertEqual(len(received), 2)
                self.assertTrue(all(not item.v41_inputs.images for item in received))
            finally:
                release_download.set()
                continue_sse.set()
                image_response = await asyncio.wait_for(image_task, timeout=10)
                await sse.body_iterator.aclose()

        self.assertEqual(image_response.status_code, 200)
        self.assertEqual(
            json.loads(image_response.body)["choices"][0]["message"]["content"],
            "answer ",
        )
        self.assertEqual(len(received[-1].v41_inputs.images), 1)
        self.assertEqual(len(preparation_threads), 2)
        self.assertNotIn(loop_thread, preparation_threads)
        self.assertEqual(image_request.model_dump(), image_before)
        self.assertEqual(renderer.stop_words_id_list, [])
        self.assertEqual(renderer.extra_stop_words, [])


class EndpointStreamTest(unittest.IsolatedAsyncioTestCase):
    async def test_user_template_entrypoint_stops_stream_and_complete_response(self):
        endpoint = make_endpoint(make_renderer())
        received = []

        class Visitor:
            async def enqueue(self, inputs):
                received.append(inputs)

                async def source():
                    text = "answer STOP hidden"
                    yield GenerateOutputs(
                        generate_outputs=[
                            GenerateOutput(
                                output_ids=torch.tensor([[ord(char) for char in text]]),
                                finished=True,
                                aux_info=AuxInfo(input_len=5, output_len=len(text)),
                            )
                        ]
                    )

                return source()

        endpoint.backend_rpc_server_visitor = Visitor()
        for streaming in (False, True):
            for async_entrypoint in (False, True):
                with self.subTest(
                    streaming=streaming, async_entrypoint=async_entrypoint
                ):
                    req = request(
                        stream=streaming,
                        user_template="{{ messages[0].content }}",
                        stop="STOP",
                    )
                    args = (1, req, SimpleNamespace(headers={}))
                    response = (
                        await endpoint.chat_completion_async(*args)
                        if async_entrypoint
                        else endpoint.chat_completion(*args)
                    )
                    frames = [frame async for frame in response]
                    complete = await response.gen_complete_response_once()
                    self.assertEqual(complete.choices[0].message.content, "answer ")
                    self.assertEqual(complete.choices[0].finish_reason, "stop")
                    self.assertEqual(
                        "".join(
                            frame.choices[0].delta.content or "" for frame in frames
                        ),
                        "answer ",
                    )
                    self.assertIn("STOP", received[-1].generate_config.stop_words_str)
                    self.assertIsNone(received[-1].v41_inputs)

    async def test_nonstream_entrypoint_stops_before_consuming_later_backend_chunks(
        self,
    ):
        renderer = make_renderer()
        endpoint = OpenaiEndpoint.__new__(OpenaiEndpoint)
        endpoint.chat_renderer = renderer
        endpoint.tokenizer = renderer.tokenizer
        endpoint.stop_words_str_list = []
        endpoint.stop_words_id_list = [[0]]
        endpoint.generate_env_config = GenerateEnvConfig()
        req = request(stream=False, stop="STOP", extra_configs={"is_streaming": False})
        before = req.model_dump()
        config = endpoint._extract_generation_config(req)
        received, consumed, closed = [], [], []
        chunks = ("answer ST", "OP", "must never be generated")

        class Visitor:
            async def enqueue(self, inputs):
                received.append(inputs)

                async def source():
                    count = 0
                    try:
                        for index, chunk in enumerate(chunks):
                            consumed.append(index)
                            count += len(chunk)
                            yield GenerateOutputs(
                                generate_outputs=[
                                    GenerateOutput(
                                        output_ids=torch.tensor(
                                            [[ord(char) for char in chunk]]
                                        ),
                                        finished=False,
                                        aux_info=AuxInfo(
                                            input_len=5, output_len=count, reuse_len=0
                                        ),
                                    )
                                ]
                            )
                    finally:
                        closed.append(True)

                return source()

        choices = renderer.generate_choice(1, [1, 2], [], config, Visitor(), req)
        complete = await endpoint._collect_complete_response(choices, None)
        self.assertTrue(received[0].generate_config.is_streaming)
        self.assertEqual(consumed, [0, 1])
        self.assertEqual(closed, [True])
        self.assertEqual(complete.choices[0].message.content, "answer ")
        self.assertEqual(complete.choices[0].finish_reason, "stop")
        self.assertEqual(complete.usage.completion_tokens, len("answer STOP"))
        self.assertEqual(req.model_dump(), before)

    async def _render(self, text, req, *, chunk_size=1, max_tokens=10000, eos=True):
        closed = []
        ids = [ord(char) for char in text] + ([0] if eos else [])

        async def source():
            try:
                for offset in range(0, len(ids), chunk_size):
                    end = min(offset + chunk_size, len(ids))
                    yield GenerateOutputs(
                        generate_outputs=[
                            GenerateOutput(
                                output_ids=torch.tensor([ids[offset:end]]),
                                finished=end == len(ids),
                                aux_info=AuxInfo(
                                    input_len=5, output_len=end, reuse_len=2
                                ),
                            )
                        ]
                    )
            finally:
                closed.append(True)

        frames = []
        async for frame in make_renderer().render_response_stream(
            source(),
            req,
            GenerateConfig(is_streaming=req.stream, max_new_tokens=max_tokens),
        ):
            frames.append(frame)
        self.assertEqual(closed, [True])
        content, reasoning, calls = "", "", {}
        for frame in frames:
            delta = frame.choices[0].delta
            content += delta.content or ""
            reasoning += delta.reasoning_content or ""
            for call in delta.tool_calls or []:
                item = calls.setdefault(
                    call.index, {"id": None, "name": None, "arguments": ""}
                )
                if call.id:
                    self.assertIsNone(item["id"])
                    item["id"] = call.id
                if call.function.name:
                    item["name"] = call.function.name
                item["arguments"] += call.function.arguments or ""
        return frames[-1], content, reasoning, calls

    async def test_stream_nonstream_stops_leave_reasoning_intact(self):
        for is_stream in (True, False):
            for size in (1, 7, 1000):
                req = request(
                    stream=is_stream, thinking={"type": "enabled"}, stop="STOP"
                )
                final, content, reasoning, calls = await self._render(
                    "reason STOP stays</think>answer STOP hidden", req, chunk_size=size
                )
                self.assertEqual(
                    (content, reasoning, calls), ("answer ", "reason STOP stays", {})
                )
                self.assertEqual(final.choices[0].finish_reason, "stop")
                self.assertEqual(final.usage.prompt_tokens, 5)
                self.assertEqual(final.usage.prompt_tokens_details.cached_tokens, 2)

    async def test_forced_tool_arguments_and_stable_ids(self):
        text = '<\uff5cDSML\uff5c invoke name="lookup"><\uff5cDSML\uff5c parameter name="value" string="true">STOP \u4e2d\u6587</\uff5cDSML\uff5c parameter></\uff5cDSML\uff5c invoke></\uff5cDSML\uff5c calls>'
        for is_stream in (True, False):
            req = request(
                stream=is_stream,
                stop="STOP",
                tool_choice="required",
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "description": "",
                            "parameters": {"type": "object"},
                        },
                    }
                ],
            )
            for size in (1, 23, 1000):
                final, content, reasoning, calls = await self._render(
                    text, req, chunk_size=size
                )
                self.assertEqual((content, reasoning), ("", ""))
                self.assertEqual(final.choices[0].finish_reason, "tool_calls")
                self.assertEqual(calls[0]["name"], "lookup")
                self.assertEqual(
                    json.loads(calls[0]["arguments"]), {"value": "STOP \u4e2d\u6587"}
                )
                self.assertRegex(calls[0]["id"], r"^call_[a-f0-9]{32}_0$")

    async def test_usage_matches_recipe_without_invented_reasoning_token_counts(self):
        # Fixed recipe 8cadfede CompletionUsage reports total completion IDs;
        # response/schema.rs leaves completion_tokens_details unset.
        for text in ("reason</think>answer", "unfinished reasoning", "</think>answer"):
            for streaming in (True, False):
                for size in (1, 7, 1000):
                    with self.subTest(text=text, streaming=streaming, size=size):
                        final, _, _, _ = await self._render(
                            text,
                            request(stream=streaming, thinking={"type": "enabled"}),
                            chunk_size=size,
                        )
                        self.assertIsNone(final.usage.completion_tokens_details)
                        self.assertEqual(final.usage.completion_tokens, len(text) + 1)
                        self.assertEqual(final.usage.total_tokens, len(text) + 6)
                        self.assertEqual(final.usage.prompt_tokens, 5)
                        self.assertEqual(
                            final.usage.prompt_tokens_details.cached_tokens, 2
                        )

    async def test_length_keeps_partial_content_and_usage(self):
        final, content, _, _ = await self._render(
            "short", request(stream=True), max_tokens=5, eos=False
        )
        self.assertEqual(content, "short")
        self.assertEqual(final.choices[0].finish_reason, "length")
        self.assertEqual(final.usage.completion_tokens, 5)
        self.assertEqual(final.usage.total_tokens, 10)

    async def test_errors_and_cancellation_release_source(self):
        for exception in (RuntimeError("backend failed"), asyncio.CancelledError()):
            closed = []

            async def source():
                try:
                    yield GenerateOutputs(
                        generate_outputs=[
                            GenerateOutput(
                                output_ids=torch.tensor([[ord("a")]]),
                                aux_info=AuxInfo(input_len=5, output_len=1),
                            )
                        ]
                    )
                    raise exception
                finally:
                    closed.append(True)

            with self.assertRaises(type(exception)):
                async for _ in make_renderer().render_response_stream(
                    source(), request(stream=True), GenerateConfig(is_streaming=True)
                ):
                    pass
            self.assertEqual(closed, [True])


if __name__ == "__main__":
    unittest.main()
