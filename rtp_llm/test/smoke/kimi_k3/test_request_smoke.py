"""Adversarial checks for the request smoke's independent protocol assertions."""

import io
import json
import pathlib
import tempfile
import unittest
from types import SimpleNamespace
from urllib.error import HTTPError

from request_smoke import SmokeRequests, merge_sse, strict_json, validate_answer


class RequestSmokeChecksTest(unittest.TestCase):

    def test_invalid_params_uses_public_error_code(self):
        with tempfile.TemporaryDirectory() as directory:
            client = SmokeRequests(
                SimpleNamespace(
                    output=pathlib.Path(directory) / "accuracy.json", timeout=30
                )
            )

            def reject(code, message):
                body = json.dumps({"error_code": code, "message": message}).encode()
                raise HTTPError(
                    "http://example.test", 500, message, {}, io.BytesIO(body)
                )

            client.post = lambda *args: reject(605, "INVALID_PARAMS")
            client.openai(
                "request_invalid_temperature",
                {"temperature": 1.1},
                expect_invalid_params=True,
            )
            self.assertTrue(client.records[-1]["passed"])

            client.post = lambda *args: reject(514, "UNKNOWN_ERROR")
            with self.assertRaisesRegex(
                AssertionError, "expected OpenAI INVALID_PARAMS"
            ):
                client.openai(
                    "request_invalid_temperature",
                    {"temperature": 1.1},
                    expect_invalid_params=True,
                )
            self.assertFalse(client.records[-1]["passed"])

    def test_protocol_failure_reports_case_and_retains_response(self):
        with tempfile.TemporaryDirectory() as directory:
            client = SmokeRequests(
                SimpleNamespace(
                    output=pathlib.Path(directory) / "accuracy.json", timeout=30
                )
            )
            response = io.BytesIO(b'{"error":{"message":"backend failed"}}')
            response.status = 500
            client.post = lambda *args: response
            with self.assertRaisesRegex(
                AssertionError,
                "case=tool-stream transport=openai.*summary.json.*backend failed",
            ):
                client.openai("tool-stream", {})
            records = json.loads((client.root / "summary.json").read_text())
            self.assertEqual(
                records[0]["response"]["error"]["message"], "backend failed"
            )
            self.assertFalse(records[0]["passed"])

    def test_fragmented_tool_arguments_and_usage_only_event(self):
        events = [
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call-1",
                                    "function": {"name": "add", "arguments": '{"x":2,'},
                                }
                            ]
                        }
                    }
                ]
            },
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {"index": 0, "function": {"arguments": '"y":3}'}}
                            ]
                        }
                    }
                ]
            },
            {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 17,
                    "completion_tokens": 9,
                    "total_tokens": 26,
                },
            },
        ]
        result = merge_sse(events, include_usage=True)
        message = validate_answer(result, expected=None, finish="tool_calls")
        self.assertEqual(
            message["tool_calls"],
            [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "add", "arguments": '{"x":2,"y":3}'},
                }
            ],
        )

    def test_missing_terminal_and_data_after_terminal_fail(self):
        with self.assertRaisesRegex(AssertionError, "terminal"):
            merge_sse([{"choices": [{"delta": {"content": "5"}}]}])
        with self.assertRaisesRegex(AssertionError, "after terminal"):
            merge_sse(
                [
                    {"choices": [{"delta": {}, "finish_reason": "stop"}]},
                    {"choices": [{"delta": {"content": "wrong"}}]},
                ]
            )

    def test_protocol_errors_and_nonfinite_json_fail(self):
        with self.assertRaisesRegex(AssertionError, "SSE error"):
            merge_sse([{"error": {"message": "backend failed"}}])
        with self.assertRaisesRegex(AssertionError, "SSE error"):
            merge_sse([{"error_code": 514, "message": "collector failed"}])
        for value in ("NaN", "Infinity", "-Infinity"):
            with self.assertRaises(ValueError):
                strict_json('{"logprob":' + value + "}")

    def test_requested_usage_requires_one_final_usage_frame(self):
        terminal = {"choices": [{"delta": {}, "finish_reason": "stop"}]}
        usage = {
            "choices": [],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        for events in ([terminal], [terminal, usage, usage], [usage, terminal]):
            with self.subTest(events=events), self.assertRaises(AssertionError):
                merge_sse(events, include_usage=True)

    def test_http_eof_and_done_both_require_complete_sse(self):
        events = [
            {"choices": [{"index": 0, "delta": {"content": "5"}}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 17,
                    "completion_tokens": 9,
                    "total_tokens": 26,
                },
            },
        ]
        for done in (False, True):
            with self.subTest(done=done), tempfile.TemporaryDirectory() as directory:
                client = SmokeRequests(
                    SimpleNamespace(
                        output=pathlib.Path(directory) / "accuracy.json", timeout=30
                    )
                )
                wire = "".join(f"data: {json.dumps(event)}\r\n\r\n" for event in events)
                if done:
                    wire += "data: [DONE]\r\n\r\n"
                response = io.BytesIO(wire.encode())
                response.status = 200
                client.post = lambda *args: response
                result = client.openai(
                    "sse",
                    {"stream": True, "stream_options": {"include_usage": True}},
                    validate_answer,
                )
                self.assertEqual(result["choices"][0]["message"]["content"], "5")
                self.assertEqual(client.records[0]["sse_done"], done)


if __name__ == "__main__":
    unittest.main()
