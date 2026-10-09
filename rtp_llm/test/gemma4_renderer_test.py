"""Gemma4 tool, reasoning-channel and stop-token contracts."""

import json
import os
import unittest

from rtp_llm.config.generate_config import ThinkingMode
from rtp_llm.openai.api_datatype import (
    AudioURL,
    ChatCompletionRequest,
    ChatMessage,
    ContentPart,
    ContentPartTypeEnum,
    FunctionCall,
    GPTFunctionDefinition,
    GPTToolDefinition,
    ImageURL,
    RoleEnum,
    ToolCall,
)
from rtp_llm.openai.renderers.gemma4_renderer import (
    THINK_OPEN,
    Gemma4ReasoningParser,
    Gemma4Renderer,
    Gemma4ToolCallDetector,
    _ArgsParser,
)
from rtp_llm.openai.renderers.sglang_helpers.entrypoints.openai.protocol import (
    Function,
    Tool,
)
from rtp_llm.utils.base_model_datatypes import MMUrlType


class FakeTokenizer:
    eos_token_id = 1
    bos_token = "<bos>"

    def encode(self, prompt: str, **kwargs):
        # ord-based ids; deliberately adds NO BOS (the real gemma4 tokenizer
        # post_processor is empty, so the template is the sole BOS source).
        return [ord(ch) for ch in prompt]

    def decode(self, token_ids, **kwargs):
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        return "".join(chr(t) for t in token_ids)

    def convert_tokens_to_ids(self, word: str):
        return {"<|tool_response>": 50, "<turn|>": 106, "<eos>": 1}[word]


def _make_renderer(stop_words_id_list=None) -> Gemma4Renderer:
    renderer = Gemma4Renderer.__new__(Gemma4Renderer)
    renderer.tokenizer = FakeTokenizer()
    renderer.chat_template = ""
    renderer.think_mode = False
    renderer.default_thinking_mode = ThinkingMode.DISABLED
    renderer.stop_words_id_list = stop_words_id_list or []
    renderer.stop_words_str_list = []
    renderer.extra_stop_words = []
    renderer.extra_stop_word_ids_list = []
    return renderer


def _request(messages, tools=None, chat_template_kwargs=None):
    return ChatCompletionRequest(
        messages=messages, tools=tools, chat_template_kwargs=chat_template_kwargs
    )


WEATHER_TOOL = GPTToolDefinition(
    type="function",
    function=GPTFunctionDefinition(
        name="get_weather",
        description="Get weather for a city",
        parameters={
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "days": {"type": "integer"},
            },
            "required": ["city"],
        },
    ),
)

SGLANG_TOOLS = [
    Tool(
        type="function",
        function=Function(
            name="get_weather", description="Get weather for a city", parameters={}
        ),
    )
]


class TestArgsParser(unittest.TestCase):
    def test_flat(self):
        args = _ArgsParser('{city:<|"|>Paris<|"|>,days:3}').parse()
        self.assertEqual(args, {"city": "Paris", "days": 3})

    def test_nested(self):
        args = _ArgsParser(
            '{loc:{city:<|"|>A<|"|>,geo:[1.5,2]},flag:true,note:null,s:<|"|><|"|>}'
        ).parse()
        self.assertEqual(
            args,
            {
                "loc": {"city": "A", "geo": [1.5, 2]},
                "flag": True,
                "note": None,
                "s": "",
            },
        )

    def test_string_with_specials(self):
        args = _ArgsParser('{q:<|"|>a,b:c}d<|"|>}').parse()
        self.assertEqual(args, {"q": "a,b:c}d"})


class TestToolCallDetector(unittest.TestCase):
    def test_single_call(self):
        det = Gemma4ToolCallDetector()
        result = det.detect_and_parse(
            'Sure.<|tool_call>call:get_weather{city:<|"|>Paris<|"|>,days:3}<tool_call|>',
            SGLANG_TOOLS,
        )
        self.assertEqual(result.normal_text, "Sure.")
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(result.calls[0].tool_index, 0)
        self.assertEqual(
            json.loads(result.calls[0].parameters), {"city": "Paris", "days": 3}
        )

    def test_multiple_calls(self):
        det = Gemma4ToolCallDetector()
        result = det.detect_and_parse(
            '<|tool_call>call:get_weather{city:<|"|>A<|"|>}<tool_call|>'
            "and "
            '<|tool_call>call:get_weather{city:<|"|>B<|"|>}<tool_call|>',
            SGLANG_TOOLS,
        )
        self.assertEqual(result.normal_text, "and ")
        self.assertEqual(len(result.calls), 2)
        self.assertEqual([c.tool_index for c in result.calls], [0, 1])
        self.assertEqual(json.loads(result.calls[1].parameters), {"city": "B"})

    def test_unknown_tool_dropped(self):
        det = Gemma4ToolCallDetector()
        result = det.detect_and_parse(
            "text<|tool_call>call:not_a_tool{x:1}<tool_call|>", SGLANG_TOOLS
        )
        self.assertEqual(result.normal_text, "text")
        self.assertEqual(result.calls, [])

    def test_streaming_chunked(self):
        full = (
            'Let me check.<|tool_call>call:get_weather{city:<|"|>Paris<|"|>,days:3}'
            "<tool_call|> done"
        )
        expected = Gemma4ToolCallDetector().detect_and_parse(full, SGLANG_TOOLS)
        for chunk_size in (1, 3, 7, 64):
            det = Gemma4ToolCallDetector()
            texts, calls = "", []
            for i in range(0, len(full), chunk_size):
                result = det.parse_streaming_increment(
                    full[i : i + chunk_size], SGLANG_TOOLS
                )
                texts += result.normal_text
                calls.extend(result.calls)
            self.assertEqual(texts, expected.normal_text, f"chunk={chunk_size}")
            self.assertEqual(len(calls), 1, f"chunk={chunk_size}")
            self.assertEqual(calls[0].name, "get_weather")
            self.assertEqual(
                json.loads(calls[0].parameters),
                json.loads(expected.calls[0].parameters),
            )


class TestReasoningParser(unittest.TestCase):
    def test_nonstream(self):
        parser = Gemma4ReasoningParser()
        reasoning, normal = parser.parse_non_stream(
            "<|channel>thought\nthink text\n<channel|>answer"
        )
        self.assertEqual(reasoning.strip(), "think text")
        self.assertEqual(normal, "answer")

    def test_forced_after_open_channel(self):
        # After a tool response the template leaves the channel open; the model
        # continues inside it without re-emitting the open marker.
        parser = Gemma4ReasoningParser(force_reasoning=True)
        reasoning, normal = parser.parse_non_stream("think text\n<channel|>answer")
        self.assertEqual(reasoning.strip(), "think text")
        self.assertEqual(normal, "answer")

    def test_streaming(self):
        full = "<|channel>thought\nthink text\n<channel|>answer"
        parser = Gemma4ReasoningParser()
        reasoning, normal = "", ""
        for i in range(0, len(full), 5):
            r, n = parser.parse_stream_chunk(full[i : i + 5])
            reasoning += r
            normal += n
        self.assertEqual(reasoning.strip(), "think text")
        self.assertEqual(normal, "answer")

    def test_empty_thought_block_stripped(self):
        # Non-thinking tool-continuation: the model emits the empty thought
        # block itself; it is protocol framing, not content.
        parser = Gemma4ReasoningParser()
        reasoning, normal = parser.parse_non_stream(
            "<|channel>thought\n<channel|>The answer."
        )
        self.assertEqual(reasoning, "")
        self.assertEqual(normal, "The answer.")


class TestReasoningParserPolicy(unittest.TestCase):
    def test_tools_request_gets_parser_when_not_thinking(self):
        renderer = _make_renderer()
        req = _request(
            [ChatMessage(role=RoleEnum.user, content="weather?")],
            tools=[WEATHER_TOOL],
        )
        parser = renderer._create_reasoning_parser(req)
        self.assertIsNotNone(parser)
        self.assertFalse(parser.detector._in_reasoning)

    def test_plain_request_no_parser(self):
        renderer = _make_renderer()
        req = _request([ChatMessage(role=RoleEnum.user, content="hi")])
        self.assertIsNone(renderer._create_reasoning_parser(req))

    def test_force_after_tool_response_with_thinking(self):
        renderer = _make_renderer()
        req = _request(
            [
                ChatMessage(role=RoleEnum.user, content="weather?"),
                ChatMessage(
                    role=RoleEnum.assistant,
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="c1",
                            type="function",
                            function=FunctionCall(
                                name="get_weather", arguments='{"city": "Paris"}'
                            ),
                        )
                    ],
                ),
                ChatMessage(role=RoleEnum.tool, tool_call_id="c1", content="sunny"),
            ],
            tools=[WEATHER_TOOL],
            chat_template_kwargs={"enable_thinking": True},
        )
        renderer._build_prompt = lambda _request: THINK_OPEN
        parser = renderer._create_reasoning_parser(req)
        self.assertIsNotNone(parser)
        self.assertTrue(parser.detector._in_reasoning)


class TestStopWords(unittest.TestCase):
    def test_adds_tool_response_and_turn_close(self):
        renderer = _make_renderer(stop_words_id_list=[[1]])
        renderer._setup_stop_words()
        all_ids = [tuple(w) for w in renderer.stop_words_id_list] + [
            tuple(w) for w in renderer.extra_stop_word_ids_list
        ]
        self.assertIn((50,), all_ids)
        self.assertIn((106,), all_ids)

    def test_no_duplicates_when_present(self):
        renderer = _make_renderer(stop_words_id_list=[[1], [50], [106]])
        renderer._setup_stop_words()
        self.assertEqual(renderer.extra_stop_word_ids_list, [])


class TestRegistration(unittest.TestCase):
    def test_lazy_registered(self):
        from rtp_llm.openai.renderer_factory_register import (
            _renderer_factory,
            ensure_renderer_registered,
        )

        self.assertTrue(ensure_renderer_registered("gemma4"))
        self.assertIs(_renderer_factory["gemma4"], Gemma4Renderer)


if __name__ == "__main__":
    unittest.main()
