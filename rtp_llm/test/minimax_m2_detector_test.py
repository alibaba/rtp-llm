"""Unit tests for MinimaxM2Detector."""

from __future__ import annotations

import json
import unittest

from rtp_llm.openai.renderers.sglang_helpers.entrypoints.openai.protocol import (
    Function,
    Tool,
)
from rtp_llm.openai.renderers.sglang_helpers.function_call.minimax_m2_detector import (
    MinimaxM2Detector,
)


def _tool(name: str, properties: dict) -> Tool:
    return Tool(
        type="function",
        function=Function(
            name=name,
            description=f"{name} tool",
            parameters={
                "type": "object",
                "properties": properties,
                "required": list(properties),
            },
        ),
    )


WEATHER = _tool(
    "get_weather",
    {
        "location": {"type": "string"},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
    },
)
SEARCH = _tool(
    "search_web",
    {
        "query_list": {"type": "array", "items": {"type": "string"}},
        "query_tag": {"type": "array", "items": {"type": "string"}},
    },
)
CALC = _tool(
    "calc",
    {
        "expr": {"type": "string"},
        "precision": {"type": "integer"},
        "scientific": {"type": "boolean"},
    },
)


class TestHasAndDetect(unittest.TestCase):
    def test_no_tool_call_passthrough(self):
        d = MinimaxM2Detector()
        text = "Plain assistant reply with no tool call."
        self.assertFalse(d.has_tool_call(text))
        res = d.detect_and_parse(text, [WEATHER])
        self.assertEqual(res.normal_text, text)
        self.assertEqual(res.calls, [])

    def test_single_call_with_prose_prefix(self):
        d = MinimaxM2Detector()
        body = (
            "I'll check the weather.\n"
            "<minimax:tool_call>\n"
            '<invoke name="get_weather">\n'
            '<parameter name="location">San Francisco</parameter>\n'
            '<parameter name="unit">celsius</parameter>\n'
            "</invoke>\n"
            "</minimax:tool_call>"
        )
        self.assertTrue(d.has_tool_call(body))
        res = d.detect_and_parse(body, [WEATHER])
        self.assertEqual(res.normal_text, "I'll check the weather.\n")
        self.assertEqual(len(res.calls), 1)
        self.assertEqual(res.calls[0].name, "get_weather")
        params = json.loads(res.calls[0].parameters)
        self.assertEqual(params, {"location": "San Francisco", "unit": "celsius"})

    def test_multi_invoke_in_one_block(self):
        d = MinimaxM2Detector()
        body = (
            "<minimax:tool_call>\n"
            '<invoke name="search_web">\n'
            '<parameter name="query_tag">["technology"]</parameter>\n'
            '<parameter name="query_list">["latest"]</parameter>\n'
            "</invoke>\n"
            '<invoke name="get_weather">\n'
            '<parameter name="location">SF</parameter>\n'
            '<parameter name="unit">celsius</parameter>\n'
            "</invoke>\n"
            "</minimax:tool_call>"
        )
        res = d.detect_and_parse(body, [WEATHER, SEARCH])
        self.assertEqual(len(res.calls), 2)
        names = [c.name for c in res.calls]
        self.assertEqual(names, ["search_web", "get_weather"])
        params0 = json.loads(res.calls[0].parameters)
        self.assertEqual(params0["query_tag"], ["technology"])
        self.assertEqual(params0["query_list"], ["latest"])

    def test_type_coercion_integer_and_boolean(self):
        d = MinimaxM2Detector()
        body = (
            "<minimax:tool_call>\n"
            '<invoke name="calc">\n'
            '<parameter name="expr">1+1</parameter>\n'
            '<parameter name="precision">4</parameter>\n'
            '<parameter name="scientific">true</parameter>\n'
            "</invoke>\n"
            "</minimax:tool_call>"
        )
        res = d.detect_and_parse(body, [CALC])
        self.assertEqual(len(res.calls), 1)
        params = json.loads(res.calls[0].parameters)
        self.assertEqual(params["expr"], "1+1")
        self.assertEqual(params["precision"], 4)
        self.assertEqual(params["scientific"], True)

    def test_type_coercion_boolean_false_and_invalid(self):
        d = MinimaxM2Detector()
        body = (
            "<minimax:tool_call>\n"
            '<invoke name="calc">\n'
            '<parameter name="expr">1+1</parameter>\n'
            '<parameter name="precision">4</parameter>\n'
            '<parameter name="scientific">false</parameter>\n'
            "</invoke>\n"
            "</minimax:tool_call>"
        )
        res = d.detect_and_parse(body, [CALC])
        params = json.loads(res.calls[0].parameters)
        self.assertEqual(params["scientific"], False)

        # Invalid boolean value falls back to raw string
        body_invalid = (
            "<minimax:tool_call>\n"
            '<invoke name="calc">\n'
            '<parameter name="expr">2+2</parameter>\n'
            '<parameter name="precision">2</parameter>\n'
            '<parameter name="scientific">maybe</parameter>\n'
            "</invoke>\n"
            "</minimax:tool_call>"
        )
        res2 = d.detect_and_parse(body_invalid, [CALC])
        params2 = json.loads(res2.calls[0].parameters)
        self.assertEqual(params2["scientific"], "maybe")

    def test_string_param_value_with_internal_newlines_preserved(self):
        d = MinimaxM2Detector()
        body = (
            "<minimax:tool_call>\n"
            '<invoke name="get_weather">\n'
            '<parameter name="location">\nMulti\nLine\nLocation\n</parameter>\n'
            '<parameter name="unit">celsius</parameter>\n'
            "</invoke>\n"
            "</minimax:tool_call>"
        )
        res = d.detect_and_parse(body, [WEATHER])
        params = json.loads(res.calls[0].parameters)
        self.assertEqual(params["location"], "Multi\nLine\nLocation")

    def test_unknown_tool_falls_back_to_string(self):
        d = MinimaxM2Detector()
        body = (
            "<minimax:tool_call>\n"
            '<invoke name="totally_unknown">\n'
            '<parameter name="x">42</parameter>\n'
            "</invoke>\n"
            "</minimax:tool_call>"
        )
        res = d.detect_and_parse(body, [WEATHER])
        self.assertIsInstance(res.calls, list)

    def test_prose_suffix_after_tool_block(self):
        d = MinimaxM2Detector()
        body = (
            "<minimax:tool_call>\n"
            '<invoke name="get_weather">\n'
            '<parameter name="location">SF</parameter>\n'
            '<parameter name="unit">celsius</parameter>\n'
            "</invoke>\n"
            "</minimax:tool_call>\n"
            "Here is some trailing text."
        )
        res = d.detect_and_parse(body, [WEATHER])
        self.assertEqual(len(res.calls), 1)
        self.assertEqual(res.normal_text, "")


class TestStreaming(unittest.TestCase):
    """The streaming detector emits multiple `ToolCallItem` *deltas* per
    logical tool call:
      - one with `name=<func>, parameters=""` when `<invoke>` is seen
      - one per `</parameter>` close, carrying a JSON fragment
      - one with `parameters="}"` when `</invoke>` closes
    Concatenating all `parameters` deltas for a given `tool_index` yields
    the full args JSON object. Tests assert that invariant rather than a
    single-delta count."""

    def _stream(self, full: str, chunks: list[int], tools: list[Tool]):
        d = MinimaxM2Detector()
        outputs = []
        cursor = 0
        for n in chunks:
            piece = full[cursor : cursor + n]
            cursor += n
            outputs.append(d.parse_streaming_increment(piece, tools))
        if cursor < len(full):
            outputs.append(d.parse_streaming_increment(full[cursor:], tools))
        return outputs

    def _aggregate(self, outs):
        """Return (names_in_order, args_by_index_dict) reconstructed from
        delta calls. `args_by_index_dict[i]` is the parsed args JSON for
        tool with `tool_index=i`."""
        names: list[str] = []
        frags: dict[int, str] = {}
        for out in outs:
            for c in out.calls:
                if c.name is not None:
                    names.append(c.name)
                if c.parameters:
                    frags.setdefault(c.tool_index, "")
                    frags[c.tool_index] += c.parameters
        args = {idx: json.loads(frag) for idx, frag in frags.items()}
        return names, args

    def test_streaming_name_emitted_before_parameters(self):
        """Sanity check on the streaming emit ordering."""
        body = (
            "<minimax:tool_call>\n"
            '<invoke name="get_weather">\n'
            '<parameter name="location">SF</parameter>\n'
            "</invoke>\n"
            "</minimax:tool_call>"
        )
        outs = self._stream(body, [40, 60], [WEATHER])
        all_calls = [c for out in outs for c in out.calls]
        # First emitted item must carry the name with empty parameters.
        self.assertEqual(all_calls[0].name, "get_weather")
        self.assertEqual(all_calls[0].parameters, "")
        # All later items must have name=None (parameter / close-brace deltas).
        for c in all_calls[1:]:
            self.assertIsNone(c.name)

    def test_streaming_emits_only_after_eot(self):
        """Pre-tool prose must surface as normal_text, and the tool's args
        must reconstruct exactly once the eot arrives."""
        body = (
            "Let me think.\n"
            "<minimax:tool_call>\n"
            '<invoke name="get_weather">\n'
            '<parameter name="location">SF</parameter>\n'
            '<parameter name="unit">celsius</parameter>\n'
            "</invoke>\n"
            "</minimax:tool_call>"
        )
        outs = self._stream(body, [25, 60], [WEATHER])
        names, args = self._aggregate(outs)
        self.assertEqual(names, ["get_weather"])
        self.assertEqual(args, {0: {"location": "SF", "unit": "celsius"}})
        joined_normal = "".join(o.normal_text for o in outs)
        self.assertIn("Let me think.", joined_normal)

    def test_streaming_two_invokes_emit_together(self):
        body = (
            "<minimax:tool_call>\n"
            '<invoke name="search_web">\n'
            '<parameter name="query_tag">["a"]</parameter>\n'
            '<parameter name="query_list">["x"]</parameter>\n'
            "</invoke>\n"
            '<invoke name="get_weather">\n'
            '<parameter name="location">SF</parameter>\n'
            '<parameter name="unit">celsius</parameter>\n'
            "</invoke>\n"
            "</minimax:tool_call>"
        )
        outs = self._stream(body, [50, 80, 80], [WEATHER, SEARCH])
        names, args = self._aggregate(outs)
        self.assertEqual(names, ["search_web", "get_weather"])
        self.assertEqual(
            args,
            {
                0: {"query_tag": ["a"], "query_list": ["x"]},
                1: {"location": "SF", "unit": "celsius"},
            },
        )

    def test_streaming_partial_bot_token_buffered(self):
        """When a chunk ends mid-way through '<minimax:tool_call>', the
        partial tag must be held in the buffer — not emitted as normal text."""
        body = (
            "Hello<minimax:tool_call>\n"
            '<invoke name="get_weather">\n'
            '<parameter name="location">SF</parameter>\n'
            '<parameter name="unit">celsius</parameter>\n'
            "</invoke>\n"
            "</minimax:tool_call>"
        )
        d = MinimaxM2Detector()
        # First chunk: "Hello<minimax:tool_" — partial bot token
        r1 = d.parse_streaming_increment("Hello<minimax:tool_", [WEATHER])
        self.assertNotIn("<minimax:", r1.normal_text)
        self.assertEqual(r1.calls, [])
        # Second chunk delivers the rest; aggregate args must reconstruct.
        r2 = d.parse_streaming_increment(body[len("Hello<minimax:tool_") :], [WEATHER])
        names, args = self._aggregate([r1, r2])
        self.assertEqual(names, ["get_weather"])
        self.assertEqual(args, {0: {"location": "SF", "unit": "celsius"}})

    def test_streaming_partial_eot_token_buffered(self):
        """When a chunk ends mid-way through '</minimax:tool_call>', the
        in-flight invoke must still close cleanly once the eot arrives."""
        body = (
            "<minimax:tool_call>\n"
            '<invoke name="get_weather">\n'
            '<parameter name="location">SF</parameter>\n'
            '<parameter name="unit">celsius</parameter>\n'
            "</invoke>\n"
            "</minimax:tool_call>"
        )
        d = MinimaxM2Detector()
        # Split right in the middle of </minimax:tool_call>
        split_at = body.index("</minimax:tool_call>") + 10  # mid-eot
        r1 = d.parse_streaming_increment(body[:split_at], [WEATHER])
        r2 = d.parse_streaming_increment(body[split_at:], [WEATHER])
        names, args = self._aggregate([r1, r2])
        self.assertEqual(names, ["get_weather"])
        self.assertEqual(args, {0: {"location": "SF", "unit": "celsius"}})

    def test_streaming_plain_text_no_tool(self):
        """Pure text without any tool tokens streams through as normal_text."""
        d = MinimaxM2Detector()
        r1 = d.parse_streaming_increment("Hello ", [WEATHER])
        r2 = d.parse_streaming_increment("world!", [WEATHER])
        joined = r1.normal_text + r2.normal_text
        self.assertEqual(joined, "Hello world!")
        self.assertEqual(r1.calls, [])
        self.assertEqual(r2.calls, [])

    def test_streaming_drains_buffer_after_eot_in_same_chunk(self):
        """If the closing eot_token and trailing content arrive in the SAME
        chunk and no further chunk follows, the call must still be emitted —
        the buffer must NOT retain the parsed block as leftover."""
        d = MinimaxM2Detector()
        body = (
            "<minimax:tool_call>\n"
            '<invoke name="get_weather">\n'
            '<parameter name="location">SF</parameter>\n'
            '<parameter name="unit">celsius</parameter>\n'
            "</invoke>\n"
            "</minimax:tool_call>tail"
        )
        r = d.parse_streaming_increment(body, [WEATHER])
        names, args = self._aggregate([r])
        self.assertEqual(names, ["get_weather"])
        self.assertEqual(args, {0: {"location": "SF", "unit": "celsius"}})
        self.assertNotIn("<minimax:tool_call>", d._buffer)

    def test_streaming_back_to_back_blocks_in_one_chunk(self):
        """Two complete tool_call blocks delivered in a single chunk must both
        be parsed in this call (no waiting for the next chunk)."""
        d = MinimaxM2Detector()
        body = (
            "<minimax:tool_call>\n"
            '<invoke name="get_weather">\n'
            '<parameter name="location">SF</parameter>\n'
            "</invoke>\n"
            "</minimax:tool_call>"
            "<minimax:tool_call>\n"
            '<invoke name="search_web">\n'
            '<parameter name="query_tag">["x"]</parameter>\n'
            '<parameter name="query_list">["y"]</parameter>\n'
            "</invoke>\n"
            "</minimax:tool_call>"
        )
        r = d.parse_streaming_increment(body, [WEATHER, SEARCH])
        names, args = self._aggregate([r])
        self.assertEqual(names, ["get_weather", "search_web"])
        self.assertEqual(
            args,
            {
                0: {"location": "SF"},
                1: {"query_tag": ["x"], "query_list": ["y"]},
            },
        )
        self.assertEqual(d._buffer, "")

    def test_streaming_param_value_split_across_chunks(self):
        """A parameter value that crosses a chunk boundary must not be
        emitted until `</parameter>` arrives, then forms a single delta."""
        body = (
            "<minimax:tool_call>\n"
            '<invoke name="get_weather">\n'
            '<parameter name="location">San Francisco</parameter>\n'
            "</invoke>\n"
            "</minimax:tool_call>"
        )
        # Split mid-value, just after "San "
        split_at = body.index("San ") + 4
        d = MinimaxM2Detector()
        r1 = d.parse_streaming_increment(body[:split_at], [WEATHER])
        r2 = d.parse_streaming_increment(body[split_at:], [WEATHER])
        # r1 should have the name delta but no parameter delta yet.
        r1_param_deltas = [c for c in r1.calls if c.parameters and c.name is None]
        self.assertEqual(r1_param_deltas, [])
        names, args = self._aggregate([r1, r2])
        self.assertEqual(names, ["get_weather"])
        self.assertEqual(args, {0: {"location": "San Francisco"}})

    def test_streaming_empty_invoke_emits_curly_braces(self):
        """An invoke with zero parameters must emit `{}` so the args JSON
        is well-formed."""
        no_args_tool = _tool("now", {})
        body = (
            "<minimax:tool_call>\n"
            '<invoke name="now"></invoke>\n'
            "</minimax:tool_call>"
        )
        d = MinimaxM2Detector()
        r = d.parse_streaming_increment(body, [no_args_tool])
        names, args = self._aggregate([r])
        self.assertEqual(names, ["now"])
        self.assertEqual(args, {0: {}})

    def test_streaming_param_deltas_concat_to_valid_json_per_param(self):
        """Each post-name delta must itself stitch into the running args
        prefix without breaking JSON syntax (i.e. `{...`, `, ...`, `}`)."""
        body = (
            "<minimax:tool_call>\n"
            '<invoke name="calc">\n'
            '<parameter name="expr">1+2</parameter>\n'
            '<parameter name="precision">3</parameter>\n'
            '<parameter name="scientific">true</parameter>\n'
            "</invoke>\n"
            "</minimax:tool_call>"
        )
        d = MinimaxM2Detector()
        r = d.parse_streaming_increment(body, [CALC])
        # Inspect the per-param deltas.
        param_deltas = [c.parameters for c in r.calls if c.name is None]
        # Expect: 3 parameter fragments + 1 closing brace
        self.assertEqual(len(param_deltas), 4)
        self.assertTrue(param_deltas[0].startswith("{"))
        self.assertTrue(param_deltas[1].startswith(", "))
        self.assertTrue(param_deltas[2].startswith(", "))
        self.assertEqual(param_deltas[-1], "}")
        # Concat reconstructs the expected types (integer, boolean coercions).
        full = "".join(param_deltas)
        self.assertEqual(
            json.loads(full), {"expr": "1+2", "precision": 3, "scientific": True}
        )

    def test_streaming_unknown_tool_dropped(self):
        """An invoke whose name is not in the tools list must be skipped
        without emitting partial garbage to the client."""
        body = (
            "<minimax:tool_call>\n"
            '<invoke name="never_declared">\n'
            '<parameter name="x">1</parameter>\n'
            "</invoke>\n"
            "</minimax:tool_call>"
        )
        d = MinimaxM2Detector()
        r = d.parse_streaming_increment(body, [WEATHER])
        self.assertEqual(r.calls, [])
        self.assertEqual(d._buffer, "")

    def test_streaming_mixed_quote_style_preserved(self):
        """Single-quoted invoke / parameter names must be parsed too."""
        body = (
            "<minimax:tool_call>\n"
            "<invoke name='get_weather'>\n"
            "<parameter name='location'>SF</parameter>\n"
            "</invoke>\n"
            "</minimax:tool_call>"
        )
        d = MinimaxM2Detector()
        r = d.parse_streaming_increment(body, [WEATHER])
        names, args = self._aggregate([r])
        self.assertEqual(names, ["get_weather"])
        self.assertEqual(args, {0: {"location": "SF"}})


if __name__ == "__main__":
    unittest.main()
