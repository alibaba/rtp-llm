import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List
from unittest import TestCase, main, skipUnless

from jinja2 import BaseLoader, Environment

from rtp_llm.config.py_config_modules import GenerateEnvConfig, RenderConfig
from rtp_llm.openai.api_datatype import ChatCompletionRequest
from rtp_llm.openai.renderers.custom_renderer import RendererParams
from rtp_llm.openai.renderers.minimax_m3_renderer import MiniMaxM3Renderer
from rtp_llm.openai.renderers.minimax_m3_vl_renderer import MiniMaxM3VLRenderer
from rtp_llm.openai.renderers.sglang_helpers.entrypoints.openai.protocol import (
    Function,
    Tool,
)
from rtp_llm.openai.renderers.sglang_helpers.function_call.minimax_m3_detector import (
    NS_TOKEN,
    MiniMaxM3Detector,
)
from rtp_llm.openai.renderers.sglang_helpers.reasoning_parser import ReasoningParser
from rtp_llm.utils.base_model_datatypes import MMUrlType

MINIMAX_M3_PATH = Path(
    os.environ.get("MINIMAX_M3_PATH", "/data7/brucelee.ly/models/MiniMax-M3-MXFP8")
)
_HAS_MODEL = (MINIMAX_M3_PATH / "tokenizer_config.json").exists()

PLAN_FUNCTION: Dict[str, Any] = {
    "name": "make_plan",
    "description": "Build a study plan.",
    "parameters": {
        "type": "object",
        "properties": {
            "start_date": {"type": "string"},
            "session_minutes": {"type": "integer"},
            "score": {"type": "number"},
            "include_buffer": {"type": "boolean"},
            "priority": {"enum": [1, 2, 3]},
            "note": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            "rest_days": {"type": "array", "items": {"type": "string"}},
            "preferences": {
                "type": "object",
                "properties": {
                    "algorithm": {"type": "string"},
                    "retries": {"type": "integer"},
                },
            },
            "subjects": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "hours": {"type": "number"},
                    },
                },
            },
        },
        "required": ["start_date"],
    },
}

PLAN_TOOL_DEFINITION: Dict[str, Any] = {"type": "function", "function": PLAN_FUNCTION}

PLAN_TOOL = Tool(type="function", function=Function(**PLAN_FUNCTION))


def tag(name: str, inner: str) -> str:
    return f"{NS_TOKEN}<{name}>{inner}{NS_TOKEN}</{name}>"


def wrap_tool_call(*invokes: str) -> str:
    body = "".join(invokes)
    return f"{NS_TOKEN}<tool_call>\n{body}{NS_TOKEN}</tool_call>"


def invoke(name: str, inner: str) -> str:
    return f'{NS_TOKEN}<invoke name="{name}">{inner}{NS_TOKEN}</invoke>\n'


FULL_ARGS_XML = (
    tag("start_date", "2025-11-10")
    + tag("session_minutes", "45")
    + tag("score", "18.5")
    + tag("include_buffer", "true")
    + tag("priority", "2")
    + tag("note", "null")
    + tag("rest_days", tag("item", "Sun") + tag("item", "Sat"))
    + tag("preferences", tag("algorithm", "SM2") + tag("retries", "3"))
    + tag(
        "subjects",
        tag("item", tag("name", "Calculus") + tag("hours", "30"))
        + tag("item", tag("name", "History") + tag("hours", "18")),
    )
)

EXPECTED_ARGS: Dict[str, Any] = {
    "start_date": "2025-11-10",
    "session_minutes": 45,
    "score": 18.5,
    "include_buffer": True,
    "priority": 2,
    "note": None,
    "rest_days": ["Sun", "Sat"],
    "preferences": {"algorithm": "SM2", "retries": 3},
    "subjects": [
        {"name": "Calculus", "hours": 30},
        {"name": "History", "hours": 18},
    ],
}


def collapse_streaming_calls(calls) -> Dict[int, Dict[str, Any]]:
    merged: Dict[int, Dict[str, Any]] = {}
    for call in calls:
        entry = merged.setdefault(call.tool_index, {"name": None, "arguments": ""})
        if call.name:
            entry["name"] = call.name
        entry["arguments"] += call.parameters
    return merged


def feed_in_chunks(text: str, tools: List[Tool], size_fn):
    detector = MiniMaxM3Detector()
    normal, calls = "", []
    pos = 0
    while pos < len(text):
        step = size_fn()
        result = detector.parse_streaming_increment(text[pos : pos + step], tools)
        normal += result.normal_text
        calls.extend(result.calls)
        pos += step
    return normal, calls


class MiniMaxM3DetectorTest(TestCase):
    def test_nested_types_recovered_from_schema(self):
        text = "Here you go." + wrap_tool_call(invoke("make_plan", FULL_ARGS_XML))
        result = MiniMaxM3Detector().detect_and_parse(text, [PLAN_TOOL])

        self.assertEqual(result.normal_text, "Here you go.")
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "make_plan")
        args = json.loads(result.calls[0].parameters)
        self.assertEqual(args, EXPECTED_ARGS)
        # Equality alone would accept 45 == 45.0 and True == 1.
        self.assertIsInstance(args["session_minutes"], int)
        self.assertIsInstance(args["score"], float)
        self.assertIsInstance(args["include_buffer"], bool)
        self.assertIsInstance(args["priority"], int)

    def test_parallel_tool_calls_are_indexed(self):
        text = wrap_tool_call(
            invoke("make_plan", tag("start_date", "2025-01-01")),
            invoke("make_plan", tag("start_date", "2025-02-02")),
        )
        result = MiniMaxM3Detector().detect_and_parse(text, [PLAN_TOOL])

        self.assertEqual([c.tool_index for c in result.calls], [0, 1])
        self.assertEqual(
            [json.loads(c.parameters)["start_date"] for c in result.calls],
            ["2025-01-01", "2025-02-02"],
        )

    def test_value_containing_angle_brackets(self):
        text = wrap_tool_call(invoke("make_plan", tag("start_date", "a < b > c")))
        result = MiniMaxM3Detector().detect_and_parse(text, [PLAN_TOOL])

        self.assertEqual(
            json.loads(result.calls[0].parameters), {"start_date": "a < b > c"}
        )

    def test_unknown_tool_is_dropped(self):
        text = wrap_tool_call(invoke("no_such_tool", tag("start_date", "2025-11-10")))
        result = MiniMaxM3Detector().detect_and_parse(text, [PLAN_TOOL])

        self.assertEqual(result.calls, [])

    def test_text_without_tool_call_passes_through(self):
        result = MiniMaxM3Detector().detect_and_parse("just prose", [PLAN_TOOL])

        self.assertEqual(result.normal_text, "just prose")
        self.assertEqual(result.calls, [])

    def test_streaming_matches_non_streaming(self):
        text = "Here you go." + wrap_tool_call(invoke("make_plan", FULL_ARGS_XML))
        rng = random.Random(1234)

        for size_fn in [lambda: 1, lambda: 3, lambda: rng.randint(1, 40)]:
            for _ in range(50):
                normal, calls = feed_in_chunks(text, [PLAN_TOOL], size_fn)
                merged = collapse_streaming_calls(calls)
                self.assertEqual(list(merged), [0])
                self.assertEqual(merged[0]["name"], "make_plan")
                self.assertEqual(json.loads(merged[0]["arguments"]), EXPECTED_ARGS)
                self.assertEqual(normal, "Here you go.")

    def test_streaming_does_not_leak_partial_begin_token(self):
        # `]<]minimax[>[<tool_call>` repeats its leading `]` at index 2, so a naive
        # shortest-prefix hold-back flushes `]<` as content and never matches.
        text = "done]" + wrap_tool_call(
            invoke("make_plan", tag("start_date", "2025-11-10"))
        )
        split = text.index("<tool_call>") - 2

        detector = MiniMaxM3Detector()
        first = detector.parse_streaming_increment(text[:split], [PLAN_TOOL])
        second = detector.parse_streaming_increment(text[split:], [PLAN_TOOL])

        self.assertNotIn(NS_TOKEN, first.normal_text + second.normal_text)
        merged = collapse_streaming_calls(first.calls + second.calls)
        self.assertEqual(
            json.loads(merged[0]["arguments"]), {"start_date": "2025-11-10"}
        )

    def _parse_args(self, args_xml: str, tools=None):
        text = wrap_tool_call(invoke("make_plan", args_xml))
        result = MiniMaxM3Detector().detect_and_parse(text, tools or [PLAN_TOOL])
        self.assertEqual(len(result.calls), 1)
        return json.loads(result.calls[0].parameters)

    def test_recovers_from_dropped_array_closing_tag(self):
        # A dropped close tag otherwise swallows every following sibling into the
        # unclosed element; observed live when the model emits a long argument tree.
        broken = FULL_ARGS_XML.replace(f"{NS_TOKEN}</rest_days>", "", 1)

        self.assertEqual(self._parse_args(broken), EXPECTED_ARGS)

    def test_recovers_from_dropped_object_closing_tag(self):
        broken = FULL_ARGS_XML.replace(f"{NS_TOKEN}</preferences>", "", 1)

        self.assertEqual(self._parse_args(broken), EXPECTED_ARGS)

    def test_ignores_stray_closing_tag(self):
        broken = FULL_ARGS_XML.replace(
            f"{NS_TOKEN}<score>", f"{NS_TOKEN}</bogus>{NS_TOKEN}<score>", 1
        )

        self.assertEqual(self._parse_args(broken), EXPECTED_ARGS)

    def test_keeps_unknown_keys_of_schemaless_object(self):
        # Repair must not hoist keys out of an object that declares no properties.
        tool = Tool(
            type="function",
            function=Function(
                name="make_plan",
                description="d",
                parameters={
                    "type": "object",
                    "properties": {"extras": {"type": "object"}},
                },
            ),
        )
        xml = tag("extras", tag("whatever", "1") + tag("other", "2"))

        self.assertEqual(
            self._parse_args(xml, [tool]),
            {"extras": {"whatever": "1", "other": "2"}},
        )

    def test_does_not_hoist_key_shadowing_an_outer_property(self):
        # `name` is declared both at the top level and inside the nested object, so
        # the nested one is legitimate and must stay put.
        tool = Tool(
            type="function",
            function=Function(
                name="make_plan",
                description="d",
                parameters={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "inner": {
                            "type": "object",
                            "properties": {"name": {"type": "string"}},
                        },
                    },
                },
            ),
        )
        xml = tag("name", "outer") + tag("inner", tag("name", "nested"))

        self.assertEqual(
            self._parse_args(xml, [tool]),
            {"name": "outer", "inner": {"name": "nested"}},
        )

    def test_streaming_emits_normal_text_before_tool_call(self):
        text = "thinking out loud" + wrap_tool_call(
            invoke("make_plan", tag("start_date", "2025-11-10"))
        )
        normal, calls = feed_in_chunks(text, [PLAN_TOOL], lambda: 7)

        self.assertEqual(normal, "thinking out loud")
        self.assertEqual(len(collapse_streaming_calls(calls)), 1)


class MiniMaxM3ReasoningParserTest(TestCase):
    def test_non_stream_split(self):
        parser = ReasoningParser(model_type="minimax_m3")

        reasoning, normal = parser.parse_non_stream(
            "<mm:think>weighing options</mm:think>the answer"
        )

        self.assertEqual(reasoning, "weighing options")
        self.assertEqual(normal, "the answer")

    def test_non_thinking_reply_strips_bare_end_tag(self):
        # When M3 skips thinking it opens with a bare `</mm:think>`; that tag must
        # not survive into the content.
        parser = ReasoningParser(model_type="minimax_m3")

        reasoning, normal = parser.parse_non_stream("</mm:think>the answer")

        self.assertEqual(reasoning, "")
        self.assertEqual(normal, "the answer")

    def test_unterminated_thinking_stays_reasoning(self):
        # No end tag means the reply was cut off mid-thought (finish_reason=length),
        # so everything is reasoning rather than content.
        parser = ReasoningParser(model_type="minimax_m3")

        reasoning, normal = parser.parse_non_stream("<mm:think>still working on it")

        self.assertEqual(reasoning, "still working on it")
        self.assertEqual(normal, "")

    def test_non_thinking_reply_strips_bare_end_tag_streaming(self):
        parser = ReasoningParser(model_type="minimax_m3")
        reasoning, normal = "", ""

        for chunk in ["</mm:think>", "the ", "answer"]:
            chunk_reasoning, chunk_normal = parser.parse_stream_chunk(chunk)
            reasoning += chunk_reasoning
            normal += chunk_normal

        self.assertEqual(reasoning, "")
        self.assertEqual(normal, "the answer")

    def test_streaming_split(self):
        # Chunks mirror per-token deltas: `<mm:think>` and `</mm:think>` are single
        # entries in M3's added vocab, so the tags always arrive intact.
        parser = ReasoningParser(model_type="minimax_m3")
        chunks = [
            "<mm:think>",
            "step ",
            "one ",
            "step ",
            "two",
            "</mm:think>",
            "final ",
            "words",
        ]
        reasoning, normal = "", ""

        for chunk in chunks:
            chunk_reasoning, chunk_normal = parser.parse_stream_chunk(chunk)
            reasoning += chunk_reasoning
            normal += chunk_normal

        self.assertEqual(reasoning, "step one step two")
        self.assertEqual(normal, "final words")


class FakeTokenizer:
    eos_token_id = 200001
    chat_template = "{{ payload | tojson(ensure_ascii=False) }}|{{ flag | tojson }}"
    special_tokens_map = {"eos_token": "[e~[", "bos_token": "]~b]"}

    def encode(self, text: str, **kwargs) -> List[int]:
        return [1] * len(text)

    def decode(self, ids, **kwargs) -> str:
        return ""

    def convert_tokens_to_ids(self, token: str) -> int:
        return 7

    def apply_chat_template(self, messages, **kwargs) -> str:
        self.applied_messages = messages
        self.applied_kwargs = kwargs
        return "rendered multimodal prompt"


def build_renderer(tokenizer, ckpt_path: str = "") -> MiniMaxM3Renderer:
    params = RendererParams(
        model_type="minimax_m3",
        max_seq_len=8192,
        eos_token_id=tokenizer.eos_token_id,
        stop_word_ids_list=[],
        ckpt_path=ckpt_path,
    )
    return MiniMaxM3Renderer(
        tokenizer, params, GenerateEnvConfig(), RenderConfig(), ckpt_path
    )


def build_vl_renderer(tokenizer, ckpt_path: str = "") -> MiniMaxM3VLRenderer:
    params = RendererParams(
        model_type="minimax_m3_vl",
        max_seq_len=8192,
        eos_token_id=tokenizer.eos_token_id,
        stop_word_ids_list=[],
        ckpt_path=ckpt_path,
    )
    return MiniMaxM3VLRenderer(
        tokenizer, params, GenerateEnvConfig(), RenderConfig(), ckpt_path
    )


class MiniMaxM3RendererTest(TestCase):
    def setUp(self):
        self.renderer = build_renderer(FakeTokenizer())

    def test_eos_registered_as_stop_word(self):
        # Without this the engine never stops before max_new_tokens.
        self.assertIn("[e~[", self.renderer.extra_stop_words)

    def test_tool_call_markers_not_stop_words(self):
        self.assertNotIn(NS_TOKEN, self.renderer.extra_stop_words)
        self.assertNotIn("<mm:think>", self.renderer.extra_stop_words)

    def test_reasoning_always_enabled(self):
        self.assertTrue(self.renderer.in_think_mode(ChatCompletionRequest(messages=[])))

    def test_tojson_filter_accepts_ensure_ascii(self):
        env = Environment(loader=BaseLoader())
        self.renderer._customize_jinja_env(env)

        rendered = env.from_string(FakeTokenizer.chat_template).render(
            payload={"city": "杭州"}, flag=True
        )

        self.assertEqual(rendered, '{"city": "杭州"}|true')

    def test_preprocess_messages_decodes_arguments(self):
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "make_plan",
                            "arguments": '{"start_date": "2025-11-10", "retries": 3}',
                        }
                    }
                ],
            }
        ]

        processed = self.renderer._preprocess_messages(messages)

        self.assertEqual(
            processed[0]["tool_calls"][0]["function"]["arguments"],
            {"start_date": "2025-11-10", "retries": 3},
        )

    def test_preprocess_messages_tolerates_malformed_arguments(self):
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"function": {"name": "make_plan", "arguments": "not json"}}
                ],
            }
        ]

        processed = self.renderer._preprocess_messages(messages)

        self.assertEqual(processed[0]["tool_calls"][0]["function"]["arguments"], {})

    def test_detector_only_created_when_tools_present(self):
        with_tools = ChatCompletionRequest(
            messages=[{"role": "user", "content": "hi"}],
            tools=[PLAN_TOOL_DEFINITION],
        )
        without_tools = ChatCompletionRequest(
            messages=[{"role": "user", "content": "hi"}]
        )

        self.assertIsInstance(
            self.renderer._create_detector(with_tools), MiniMaxM3Detector
        )
        self.assertIsNone(self.renderer._create_detector(without_tools))


class MiniMaxM3VLRendererTest(TestCase):
    def setUp(self):
        self.tokenizer = FakeTokenizer()
        self.renderer = build_vl_renderer(self.tokenizer)

    def test_reuses_m3_reasoning_and_tool_detectors(self):
        request = ChatCompletionRequest(
            messages=[{"role": "user", "content": "plan my finals"}],
            tools=[PLAN_TOOL_DEFINITION],
        )

        self.assertIsInstance(self.renderer, MiniMaxM3Renderer)
        self.assertIsInstance(
            self.renderer._create_detector(request), MiniMaxM3Detector
        )
        reasoning, content = self.renderer._create_reasoning_parser(
            request
        ).parse_non_stream("<mm:think>reasoning</mm:think>answer")
        self.assertEqual(reasoning, "reasoning")
        self.assertEqual(content, "answer")

    def test_multimodal_render_keeps_inputs_and_preprocesses_tool_calls(self):
        request = ChatCompletionRequest(
            messages=[
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "make_plan",
                                "arguments": '{"start_date": "2025-11-10"}',
                            },
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://example.com/image.jpg"},
                        },
                        {"type": "text", "text": "plan my finals"},
                    ],
                },
            ],
            tools=[PLAN_TOOL_DEFINITION],
        )

        rendered = self.renderer.render_chat(request)

        self.assertEqual(len(rendered.multimodal_inputs), 1)
        self.assertEqual(
            rendered.multimodal_inputs[0].url, "https://example.com/image.jpg"
        )
        self.assertEqual(rendered.multimodal_inputs[0].mm_type, MMUrlType.IMAGE)
        self.assertEqual(
            self.tokenizer.applied_messages[0]["tool_calls"][0]["function"][
                "arguments"
            ],
            {"start_date": "2025-11-10"},
        )
        self.assertEqual(
            self.tokenizer.applied_kwargs["tools"][0]["function"]["name"],
            "make_plan",
        )
        self.assertEqual(self.tokenizer.applied_kwargs["thinking_mode"], "adaptive")

    def test_multimodal_preprocess_config_is_aligned_and_supports_inline_fields(self):
        data_url = "data:image/png;base64,AA=="
        request = ChatCompletionRequest(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url},
                            "preprocess_config": {"max_long_side_pixel": 1008},
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://example.com/image.webp"},
                        },
                        {
                            "type": "video_url",
                            "video_url": {
                                "url": "https://example.com/video.mp4",
                                "max_long_side_pixel": 896,
                                "fps": 0.2,
                            },
                            "max_long_side_pixel": 784,
                            "fps": 0.5,
                        },
                    ],
                }
            ]
        )

        rendered = self.renderer.render_chat(request)

        self.assertEqual(
            [item.url for item in rendered.multimodal_inputs],
            [
                data_url,
                "https://example.com/image.webp",
                "https://example.com/video.mp4",
            ],
        )
        configs = [item.mm_preprocess_config for item in rendered.multimodal_inputs]
        self.assertEqual(configs[0].max_long_side_pixel, 1008)
        self.assertEqual(configs[1].max_long_side_pixel, -1)
        self.assertEqual(configs[2].max_long_side_pixel, 784)
        self.assertAlmostEqual(configs[2].fps, 0.5)


@skipUnless(_HAS_MODEL, f"MiniMax-M3 tokenizer not found at {MINIMAX_M3_PATH}")
class MiniMaxM3RealTokenizerTest(TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer

        cls.tokenizer = AutoTokenizer.from_pretrained(
            str(MINIMAX_M3_PATH), trust_remote_code=True
        )
        cls.renderer = build_renderer(cls.tokenizer, str(MINIMAX_M3_PATH))
        cls.vl_renderer = build_vl_renderer(cls.tokenizer, str(MINIMAX_M3_PATH))
        cls.tools = [PLAN_TOOL_DEFINITION]
        cls.messages = [{"role": "user", "content": "plan my finals"}]

    def test_tools_reach_the_prompt(self):
        request = ChatCompletionRequest(messages=self.messages, tools=self.tools)

        prompt = self.renderer.render_chat(request).rendered_prompt

        self.assertIn("# Tools", prompt)
        self.assertIn("<tools>", prompt)
        self.assertIn("make_plan", prompt)

    def test_tools_grow_the_prompt(self):
        # The original bug silently dropped `tools`, leaving both prompts identical.
        with_tools = self.renderer.render_chat(
            ChatCompletionRequest(messages=self.messages, tools=self.tools)
        )
        without_tools = self.renderer.render_chat(
            ChatCompletionRequest(messages=self.messages)
        )

        self.assertGreater(
            len(with_tools.input_ids), len(without_tools.input_ids) + 100
        )

    def test_prompt_token_count_matches_hf_template(self):
        request = ChatCompletionRequest(messages=self.messages, tools=self.tools)
        reference = self.tokenizer.apply_chat_template(
            self.messages, tools=self.tools, tokenize=False, add_generation_prompt=True
        )

        rendered = self.renderer.render_chat(request)

        self.assertEqual(len(rendered.input_ids), len(self.tokenizer.encode(reference)))

    def test_vl_prompt_keeps_image_input_and_tools(self):
        request = ChatCompletionRequest(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://example.com/image.jpg"},
                        },
                        {"type": "text", "text": "describe the image"},
                    ],
                }
            ],
            tools=self.tools,
        )

        rendered = self.vl_renderer.render_chat(request)

        self.assertIn("]<]image[>[", rendered.rendered_prompt)
        self.assertIn("make_plan", rendered.rendered_prompt)
        self.assertEqual(len(rendered.multimodal_inputs), 1)
        self.assertEqual(
            rendered.multimodal_inputs[0].url, "https://example.com/image.jpg"
        )
        self.assertIsInstance(
            self.vl_renderer._create_detector(request), MiniMaxM3Detector
        )

    def test_rendered_tool_call_round_trips_through_detector(self):
        # What the template emits for an assistant turn is exactly what the model
        # is trained to produce, so it must parse back to the original arguments.
        arguments = {
            "start_date": "2025-11-10",
            "session_minutes": 45,
            "include_buffer": True,
            "rest_days": ["Sun"],
            "preferences": {"algorithm": "SM2", "retries": 3},
            "subjects": [{"name": "Calculus", "hours": 30}],
        }
        request = ChatCompletionRequest(
            messages=self.messages
            + [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "make_plan",
                                "arguments": json.dumps(arguments),
                            },
                        }
                    ],
                }
            ],
            tools=self.tools,
        )

        prompt = self.renderer.render_chat(request).rendered_prompt
        start = prompt.rindex(f"{NS_TOKEN}<tool_call>")
        result = MiniMaxM3Detector().detect_and_parse(prompt[start:], [PLAN_TOOL])

        self.assertEqual(len(result.calls), 1)
        self.assertEqual(json.loads(result.calls[0].parameters), arguments)


if __name__ == "__main__":
    main()
