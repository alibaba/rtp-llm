"""Exercise the actual RTP response/config path with CPU token-source fixtures."""

import asyncio
import json
import unittest

import torch

from rtp_llm.config.generate_config import GenerateConfig
from rtp_llm.config.py_config_modules import GenerateEnvConfig
from rtp_llm.openai.api_datatype import ChatCompletionRequest
from rtp_llm.openai.openai_endpoint import OpenaiEndpoint
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
    return renderer


def request(**kwargs):
    return ChatCompletionRequest.model_validate(
        {"messages": [{"role": "user", "content": "hello"}], **kwargs}
    )


class EndpointConfigTest(unittest.TestCase):
    def setUp(self):
        self.renderer = make_renderer()
        self.endpoint = OpenaiEndpoint.__new__(OpenaiEndpoint)
        self.endpoint.chat_renderer = self.renderer
        self.endpoint.tokenizer = self.renderer.tokenizer
        self.endpoint.stop_words_str_list = ["engine_stop"]
        self.endpoint.stop_words_id_list = [[0]]
        self.endpoint.generate_env_config = GenerateEnvConfig()

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


class EndpointStreamTest(unittest.IsolatedAsyncioTestCase):
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
