import unittest

from rtp_llm.openai.renderers.sglang_helpers.format_convert_helper import (
    streaming_parse_result_to_tool_calls,
)
from rtp_llm.openai.renderers.sglang_helpers.function_call.core_types import (
    StreamingParseResult,
    ToolCallItem,
)


class TestStreamingParseResultToToolCalls(unittest.TestCase):
    def test_squash_same_index_fragments(self):
        result = StreamingParseResult(
            normal_text="",
            calls=[
                ToolCallItem(tool_index=0, name="get_current_weather", parameters=""),
                ToolCallItem(tool_index=0, parameters="{"),
                ToolCallItem(tool_index=0, parameters='"location": '),
                ToolCallItem(tool_index=0, parameters='"'),
                ToolCallItem(tool_index=0, parameters="杭州"),
            ],
        )

        tool_calls, remaining = streaming_parse_result_to_tool_calls(result)
        self.assertEqual(remaining, "")
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0].index, 0)
        self.assertEqual(tool_calls[0].function.name, "get_current_weather")
        self.assertEqual(tool_calls[0].function.arguments, '{"location": "杭州')
        self.assertTrue(
            tool_calls[0].id is not None and tool_calls[0].id.startswith("call_")
        )

    def test_output_sorted_by_tool_index(self):
        """tool_index reflects the model's intended order; output must be
        sorted by tool_index regardless of stream arrival order."""
        result = StreamingParseResult(
            normal_text="",
            calls=[
                ToolCallItem(tool_index=1, name="b", parameters=""),
                ToolCallItem(tool_index=0, name="a", parameters=""),
                ToolCallItem(tool_index=1, parameters='{"x":1}'),
                ToolCallItem(tool_index=0, parameters='{"y":2}'),
            ],
        )

        tool_calls, _ = streaming_parse_result_to_tool_calls(result)
        self.assertEqual([tc.index for tc in tool_calls], [0, 1])
        self.assertEqual(tool_calls[0].function.name, "a")
        self.assertEqual(tool_calls[0].function.arguments, '{"y":2}')
        self.assertEqual(tool_calls[1].function.name, "b")
        self.assertEqual(tool_calls[1].function.arguments, '{"x":1}')

    def test_name_can_arrive_after_arguments(self):
        result = StreamingParseResult(
            normal_text="",
            calls=[
                ToolCallItem(tool_index=0, parameters="{"),
                ToolCallItem(tool_index=0, parameters='"location": "'),
                ToolCallItem(tool_index=0, name="get_current_weather", parameters=""),
            ],
        )

        tool_calls, _ = streaming_parse_result_to_tool_calls(result)
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0].function.name, "get_current_weather")
        self.assertEqual(tool_calls[0].function.arguments, '{"location": "')
        self.assertTrue(
            tool_calls[0].id is not None and tool_calls[0].id.startswith("call_")
        )


if __name__ == "__main__":
    unittest.main()
