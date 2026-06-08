import unittest

from smoke.common_def import QueryStatus, SmokeException
from smoke.openai_comparer import OpenaiComparer

from rtp_llm.openai.api_datatype import ChatCompletionResponse


def _response(content: str) -> ChatCompletionResponse:
    return ChatCompletionResponse(
        choices=[
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                    "reasoning_content": None,
                    "function_call": None,
                    "tool_calls": None,
                    "partial": False,
                    "tool_call_id": None,
                },
                "finish_reason": "length",
                "logprobs": None,
            }
        ],
        usage={
            "prompt_tokens": 1,
            "total_tokens": 2,
            "completion_tokens": 1,
        },
        aux_info=None,
        extra_outputs=None,
    )


class OpenaiComparerTest(unittest.TestCase):
    def _comparer(self, result):
        comparer = OpenaiComparer.__new__(OpenaiComparer)
        comparer.qr_info = {"result": result}
        return comparer

    def test_content_alternatives_accept_only_message_content_diff(self):
        comparer = self._comparer({"content_alternatives": [["alternate"]]})
        comparer.compare_result(_response("primary"), _response("alternate"))

    def test_content_alternatives_reject_unlisted_message_content(self):
        comparer = self._comparer({"content_alternatives": [["alternate"]]})

        with self.assertRaises(SmokeException) as ctx:
            comparer.compare_result(_response("primary"), _response("unexpected"))

        self.assertEqual(QueryStatus.COMPARE_FAILED, ctx.exception.error_status)


if __name__ == "__main__":
    unittest.main()
