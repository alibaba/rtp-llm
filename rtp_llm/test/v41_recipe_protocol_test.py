"""Independent literal protocol fixtures; no Torch or recipe native runtime."""

import copy
import importlib.util
import json
import random
import sys
import unittest
from pathlib import Path


def load_source(name, relative):
    path = Path(__file__).resolve().parents[1] / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


effort_module = load_source(
    "rtp_llm.openai.reasoning_effort", "openai/reasoning_effort.py"
)
recipe = load_source("rtp_v41_recipe_fixture", "openai/renderers/v41_recipe.py")
stream = load_source("rtp_v41_stream_fixture", "openai/renderers/v41_stream.py")

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Find an item",
            "parameters": {"type": "object"},
        },
    }
]
CALL = '<\uff5cDSML\uff5c calls>\n<\uff5cDSML\uff5c invoke name="lookup">\n<\uff5cDSML\uff5c parameter name="value" string="true">STOP \u4e2d\u6587 "\\\n</\uff5cDSML\uff5c parameter>\n<\uff5cDSML\uff5c parameter name="object" string="false">{"x": [1, true, null]}</\uff5cDSML\uff5c parameter>\n</\uff5cDSML\uff5c invoke>\n<\uff5cDSML\uff5c invoke name="done">\n</\uff5cDSML\uff5c invoke>\n</\uff5cDSML\uff5c calls>'


def parse(chunks, **options):
    parser = stream.V41StreamParser(**options)
    events = []
    for chunk in chunks:
        events.extend(parser.feed(chunk))
    events.extend(parser.finish())
    result = {"content": "", "reasoning": "", "tools": []}
    for event in events:
        if event.kind in ("content", "reasoning"):
            result[event.kind] += event.text
        elif event.kind == "tool":
            result["tools"].append({"name": event.text, "arguments": ""})
        elif event.kind == "arguments":
            result["tools"][event.index]["arguments"] += event.text
    return result, parser


class RequestFixtures(unittest.TestCase):
    def test_tool_schema_order_spacing_and_strict_metadata(self):
        request = {
            "messages": [{"role": "user", "content": "go"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "f",
                        "strict": True,
                        "description": "\u4e2d\u6587",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "z": {
                                    "type": "number",
                                    "enum": [
                                        0.0,
                                        -0.0,
                                        0.0001,
                                        1e-5,
                                        1e-6,
                                        1e-9,
                                        1e16,
                                        1e20,
                                    ],
                                },
                                "a": {"description": 'quote: "', "type": "string"},
                            },
                        },
                    },
                }
            ],
        }
        converted = recipe.convert_request(request)
        self.assertIs(converted.tools[0]["strict"], True)
        prompt, _ = recipe.render_request(converted)
        expected = '{"name": "f", "description": "\u4e2d\u6587", "parameters": {"type": "object", "properties": {"z": {"type": "number", "enum": [0.0, -0.0, 0.0001, 1e-05, 1e-06, 1e-09, 1e+16, 1e+20]}, "a": {"description": "quote: \\"", "type": "string"}}}}'
        self.assertIn("### Available Tool Schemas\n\n" + expected + "\n\n", prompt)
        self.assertNotIn('"strict":', prompt)

    def test_historical_argument_order_and_nested_json_spacing(self):
        request = {
            "messages": [
                {"role": "user", "content": "go"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "a",
                            "function": {
                                "name": "f",
                                "arguments": '{"z":{"b":1,"a":2},"a":true}',
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "a", "content": "done"},
            ]
        }
        prompt, _ = recipe.render_request(recipe.convert_request(request))
        expected = '<\uff5cDSML\uff5c parameter name="z" string="false">{"b": 1, "a": 2}</\uff5cDSML\uff5c parameter>\n<\uff5cDSML\uff5c parameter name="a" string="false">true</\uff5cDSML\uff5c parameter>'
        self.assertIn(expected, prompt)

    def test_nonfinite_schema_values_are_rejected(self):
        for value in (float("nan"), float("inf"), float("-inf")):
            request = {
                "messages": [{"role": "user", "content": "go"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "f",
                            "parameters": {"type": "object", "default": value},
                        },
                    }
                ],
            }
            with self.assertRaises(ValueError):
                recipe.convert_request(request)

    def test_effort_recipe_table_and_exact_integer_extension(self):
        aliases = {
            None: 75,
            "none": 75,
            "minimal": 50,
            "low": 50,
            "medium": 75,
            "high": 75,
            "xhigh": 75,
            "max": 100,
        }
        for value, expected in aliases.items():
            self.assertEqual(
                effort_module.normalize_v41_reasoning_effort(value), expected
            )
        for value in range(1, 101):
            self.assertEqual(effort_module.normalize_v41_reasoning_effort(value), value)
        for invalid in (True, False, 1.0, 0, 101, "1", "75", "HIGH", [], {}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                effort_module.normalize_v41_reasoning_effort(invalid)
        with self.assertRaises(ValueError):
            effort_module.validate_reasoning_effort_for_model(75, "deepseek_v4")
        effort_module.validate_reasoning_effort_for_model("medium", "deepseek_v4")

    def test_thinking_priority(self):
        cases = [
            ({}, False, False, 75),
            ({}, True, True, 75),
            ({"reasoning_effort": "none"}, True, False, 75),
            ({"reasoning_effort": "low"}, False, True, 50),
            ({"reasoning_effort": 37}, False, True, 37),
            (
                {"reasoning_effort": "none", "thinking": {"type": "enabled"}},
                False,
                True,
                75,
            ),
            (
                {"reasoning_effort": "max", "thinking": {"type": "disabled"}},
                True,
                False,
                100,
            ),
            (
                {"enable_thinking": False, "thinking": {"type": "enabled"}},
                False,
                True,
                75,
            ),
            ({"enable_thinking": True, "reasoning_effort": "none"}, False, True, 75),
            ({"thinking_budget": 0, "thinking": {"type": "enabled"}}, True, False, 75),
        ]
        for request, default, thinking, effort in cases:
            with self.subTest(request=request):
                self.assertEqual(
                    recipe.resolve_thinking(request, default), (thinking, effort)
                )

    def test_literal_request_prompt_golden(self):
        request = {"messages": [{"role": "user", "content": "hello"}]}
        prompt, images = recipe.render_request(recipe.convert_request(request))
        self.assertEqual(
            prompt,
            "<\uff5cbegin\u2581of\u2581sentence\uff5c><\uff5cUser\uff5c>hello<\uff5cAssistant\uff5c></think>",
        )
        self.assertEqual(images, [])

    def test_history_and_consecutive_users(self):
        request = {
            "reasoning_effort": "minimal",
            "messages": [
                {"role": "user", "content": "first"},
                {"role": "user", "content": "second"},
                {
                    "role": "assistant",
                    "content": "answer",
                    "reasoning_content": "prior reasoning",
                },
                {"role": "system", "content": "new instruction"},
            ],
        }
        prompt, _ = recipe.render_request(recipe.convert_request(request))
        expected = "<\uff5cbegin\u2581of\u2581sentence\uff5c><\uff5cSystem\uff5c>Reasoning Effort: 50 (range 1-100, the higher the value, the more thorough the reasoning)\n\n<\uff5cUser\uff5c>first\n\nsecond<\uff5cAssistant\uff5c><think>prior reasoning</think>answer<\uff5cend\u2581of\u2581sentence\uff5c><\uff5cSystem\uff5c>new instruction<\uff5cAssistant\uff5c><think>"
        self.assertEqual(prompt, expected)

    def test_forced_tools_and_thinking_rejection(self):
        for choice in (
            "required",
            {"type": "function", "function": {"name": "lookup"}},
        ):
            request = {
                "tools": TOOLS,
                "tool_choice": choice,
                "messages": [{"role": "user", "content": "Find it"}],
            }
            converted = recipe.convert_request(request)
            prompt, _ = recipe.render_request(converted)
            self.assertTrue(converted.force_tools)
            self.assertTrue(
                prompt.endswith(
                    "<\uff5cAssistant\uff5c></think>\n\n<\uff5cDSML\uff5c calls>\n"
                )
            )
            request["thinking"] = {"type": "enabled"}
            with self.assertRaisesRegex(ValueError, "Thinking mode"):
                recipe.convert_request(request)

    def test_named_tool_selection(self):
        request = {
            "messages": [{"role": "user", "content": "go"}],
            "tools": TOOLS
            + [
                {
                    "type": "function",
                    "function": {
                        "name": "other",
                        "description": "",
                        "parameters": {"type": "object"},
                    },
                }
            ],
            "tool_choice": {"type": "function", "function": {"name": "lookup"}},
        }
        self.assertEqual(
            [tool["name"] for tool in recipe.convert_request(request).tools], ["lookup"]
        )
        request["tool_choice"] = "none"
        self.assertEqual(recipe.convert_request(request).tools, [])

    def test_tool_ids_reorder_results_and_images_together(self):
        calls = [
            {"id": value, "function": {"name": "lookup", "arguments": "{}"}}
            for value in ("a", "b")
        ]
        request = {
            "messages": [
                {"role": "user", "content": "go"},
                {
                    "role": "assistant",
                    "tool_calls": calls,
                    "reasoning_content": "keep me",
                },
                {
                    "role": "tool",
                    "tool_call_id": "b",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://example.invalid/b.png"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "a",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://example.invalid/a.png"},
                        }
                    ],
                },
            ],
            "enable_thinking": True,
        }
        original = copy.deepcopy(request)
        prompt, images = recipe.render_request(recipe.convert_request(request))
        self.assertEqual(
            [image["url"] for image in images],
            ["https://example.invalid/a.png", "https://example.invalid/b.png"],
        )
        self.assertIn("<think>keep me</think>", prompt)
        self.assertEqual(request, original)
        for broken in (
            [*request["messages"][:-1]],
            [*request["messages"][:-1], request["messages"][-2]],
        ):
            with self.assertRaises(ValueError):
                recipe.convert_request({"messages": broken})

    def test_content_blocks_keep_order(self):
        request = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "before"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://example.invalid/image.png"},
                        },
                        {"type": "text", "text": "after"},
                    ],
                }
            ]
        }
        prompt, images = recipe.render_request(recipe.convert_request(request))
        self.assertEqual(
            prompt,
            "<\uff5cbegin\u2581of\u2581sentence\uff5c><\uff5cUser\uff5c>before\n\n<\uff5cdeepseek_image\uff5c>\n\nafter<\uff5cAssistant\uff5c></think>",
        )
        self.assertEqual(len(images), 1)

    def test_invalid_input_and_json_requirement(self):
        for request in (
            {"messages": []},
            {"messages": [{"role": "tool", "content": "orphan", "tool_call_id": "x"}]},
            {"messages": [{"role": "user", "content": "<\uff5cdeepseek_image\uff5c>"}]},
            {"messages": [{"role": "assistant", "content": None}]},
            {"messages": [{"role": "user", "content": "go"}], "stop": ["x"] * 17},
            {
                "messages": [{"role": "user", "content": "go"}],
                "response_format": {"type": "json_object"},
            },
        ):
            with self.subTest(request=request), self.assertRaises(ValueError):
                recipe.convert_request(request)
        converted = recipe.convert_request(
            {
                "messages": [{"role": "user", "content": "Return JSON"}],
                "response_format": {"type": "json_object"},
            }
        )
        self.assertTrue(converted.json_output)
        self.assertIn(
            'schema to reply:\n{"type": "json_object"}',
            recipe.render_request(converted)[0],
        )


class StreamFixtures(unittest.TestCase):
    def test_dsml_every_character_boundary_and_random_partitions(self):
        expected = {
            "content": "",
            "reasoning": "",
            "tools": [
                {
                    "name": "lookup",
                    "arguments": '{"value": "STOP \u4e2d\u6587 \\"\\\\\\n", "object": {"x": [1, true, null]}}',
                },
                {"name": "done", "arguments": "{}"},
            ],
        }
        partitions = [[CALL], list(CALL)]
        partitions.extend([CALL[:split], CALL[split:]] for split in range(1, len(CALL)))
        rng = random.Random(20260911)
        for _ in range(30):
            boundaries = [0, *sorted(rng.sample(range(1, len(CALL)), 25)), len(CALL)]
            partitions.append([CALL[a:b] for a, b in zip(boundaries, boundaries[1:])])
        for chunks in partitions:
            actual, parser = parse(chunks, stop=("STOP",))
            self.assertEqual(actual, expected)
            self.assertEqual(parser.finish_reason("stop"), "tool_calls")
            self.assertFalse(parser.stopped)

    def test_stop_matches_content_after_reasoning_only(self):
        value = "\n\nreason STOP stays\n\n</think>\nanswer ST" + "OP hidden"
        for split in range(1, len(value)):
            actual, parser = parse(
                [value[:split], value[split:]], thinking=True, stop=("STOP",)
            )
            self.assertEqual(
                actual,
                {"content": "answer ", "reasoning": "reason STOP stays", "tools": []},
            )
            self.assertTrue(parser.stopped)
            self.assertEqual(parser.finish_reason("length"), "stop")

    def test_forced_tools_begin_inside_calls(self):
        raw = CALL.removeprefix("<\uff5cDSML\uff5c calls>\n")
        actual, parser = parse(list(raw), force_tools=True, stop=("STOP",))
        self.assertEqual(actual["tools"][0]["name"], "lookup")
        self.assertEqual(
            json.loads(actual["tools"][0]["arguments"]),
            {"value": 'STOP \u4e2d\u6587 "\\\n', "object": {"x": [1, True, None]}},
        )
        self.assertEqual(parser.finish_reason("length"), "length")

    def test_json_fences_ignore_stops_in_surrounding_text(self):
        text = 'STOP intro ```json\n{"value": "ok"}\n``` STOP suffix'
        one, parser = parse([text], json_output=True, stop=("STOP",))
        many, _ = parse(list(text), json_output=True, stop=("STOP",))
        self.assertEqual(one, many)
        self.assertEqual(json.loads(one["content"]), {"value": "ok"})
        self.assertFalse(parser.stopped)
        for value in ('prefix {"a": "STOP"}', '```json\n{"a": "STOP"}'):
            output, stopped = parse(list(value), json_output=True, stop=("STOP",))
            self.assertTrue(stopped.stopped)
            self.assertEqual(output["content"].strip(), '{"a": "')

    def test_eof_keeps_unmatched_prefix_without_inventing_tool_json(self):
        for text in ("hello <", "hello </thi", "hello ST", "hello\n"):
            actual, parser = parse(list(text), stop=("STOP",))
            self.assertEqual(actual["content"], text)
            self.assertEqual(parser.finish_reason(), "stop")
        partial = '<\uff5cDSML\uff5c calls><\uff5cDSML\uff5c invoke name="lookup"><\uff5cDSML\uff5c parameter name="a" string="true">unfinished'
        actual, parser = parse(list(partial))
        self.assertEqual(actual["tools"][0]["arguments"], '{"a": "unfinished')
        self.assertEqual(parser.finish_reason(), "stop")
        self.assertEqual(parser.finish_reason("length"), "length")
        self.assertEqual(parser.finish_reason("content_filter"), "content_filter")

    def test_tools_disabled_and_repeated_think_end(self):
        actual, _ = parse(list(CALL), parse_tools=False)
        self.assertEqual(actual["content"], CALL)
        actual, _ = parse(list("a</think>b</think>c"), parse_tools=False)
        self.assertEqual(actual["content"], "abc")

    def test_json_scalar_and_parameter_types(self):
        for value in (
            "0",
            "-1.25e4",
            "true",
            "false",
            "null",
            "[]",
            "{}",
            '["a", 1, null]',
            '{"quoted": "x\\ny"}',
        ):
            text = (
                '<\uff5cDSML\uff5c calls><\uff5cDSML\uff5c invoke name="f"><\uff5cDSML\uff5c parameter name="value" string="false">'
                + value
                + "</\uff5cDSML\uff5c parameter></\uff5cDSML\uff5c invoke></\uff5cDSML\uff5c calls>"
            )
            actual, _ = parse(list(text))
            self.assertEqual(
                json.loads(actual["tools"][0]["arguments"]),
                {"value": json.loads(value)},
            )

    def test_finish_is_idempotent_and_feed_after_finish_fails(self):
        parser = stream.V41StreamParser()
        parser.feed("a<")
        self.assertEqual(parser.finish(), [stream.ProtocolDelta("content", "<")])
        self.assertEqual(parser.finish(), [])
        with self.assertRaises(ValueError):
            parser.feed("later")


if __name__ == "__main__":
    unittest.main()
