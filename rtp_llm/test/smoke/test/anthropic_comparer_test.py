import json
import unittest

from smoke.anthropic_comparer import AnthropicComparer
from smoke.common_def import QueryStatus, SmokeException, Tracer


def _sse_bytes(events):
    lines = []
    for name, payload in events:
        lines.append(f"event: {name}".encode("utf-8"))
        lines.append(f"data: {json.dumps(payload)}".encode("utf-8"))
        lines.append(b"")
    return lines


def _basic_golden(**overrides):
    data = {
        "id": "msg_golden_123",
        "type": "message",
        "role": "assistant",
        "model": "qwen2.5",
        "content": [{"type": "text", "text": "Paris"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": 14,
            "output_tokens": 2,
            "cache_read_input_tokens": 0,
        },
    }
    data.update(overrides)
    return data


class AnthropicComparerTest(unittest.TestCase):
    def test_non_streaming_compare_model_stop_sequence_and_save_dump(self):
        golden = _basic_golden()
        qr_info = {
            "query": {
                "model": "qwen2.5",
                "max_tokens": 32,
                "messages": [{"role": "user", "content": "Capital of France?"}],
            },
            "result": golden,
        }
        comparer = AnthropicComparer(None, "/v1/messages", qr_info, Tracer(), False)
        query = comparer.format_query(qr_info["query"])

        actual_raw = json.dumps({**golden, "id": "msg_other_runtime_id"})
        actual = comparer.format_result(
            comparer.curl_response_to_json(query, actual_raw)
        )
        expect = comparer.format_result(qr_info["result"])
        comparer.compare_result(expect, actual)

        # Save-for-rewrite must preserve "type" discriminator and "stop_sequence": null
        saved = comparer._dump_result_for_save(actual)
        self.assertEqual(saved["content"][0]["type"], "text")
        self.assertIn("stop_sequence", saved)
        self.assertIsNone(saved["stop_sequence"])

        for bad_override in (
            {"content": [{"type": "text", "text": "Lyon"}]},
            {"stop_sequence": "<STOP>"},
            {"model": "other-model"},
            {"stop_reason": "max_tokens"},
            {
                "usage": {
                    "input_tokens": 14,
                    "output_tokens": 99,
                    "cache_read_input_tokens": 0,
                }
            },
        ):
            with self.subTest(bad_override=bad_override):
                mismatch = comparer.format_result(
                    comparer.curl_response_to_json(
                        query, json.dumps(_basic_golden(**bad_override))
                    )
                )
                with self.assertRaises(SmokeException) as ctx:
                    comparer.compare_result(expect, mismatch)
                self.assertEqual(ctx.exception.error_status, QueryStatus.COMPARE_FAILED)

    def test_compare_config_options_and_error_responses(self):
        qr_info = {
            "query": {
                "model": "qwen2.5",
                "max_tokens": 32,
                "messages": [{"role": "user", "content": "JSON"}],
            },
            "compare_config": {
                "skip_choices": True,
                "skip_usage": True,
                "expected_stop_reason": "stop_sequence",
                "expected_stop_sequence": "<STOP>",
                "required_substrings": ['"ok": true'],
                "forbidden_substrings": ["FORBIDDEN"],
                "require_json_object_content": True,
                "expected_json": {"ok": True},
            },
            "result": {},
        }
        comparer = AnthropicComparer(None, "/v1/messages", qr_info, Tracer(), False)
        query = comparer.format_query(qr_info["query"])
        expect = comparer.format_result(qr_info["result"])

        good_actual = comparer.format_result(
            comparer.curl_response_to_json(
                query,
                json.dumps(
                    _basic_golden(
                        content=[{"type": "text", "text": '{"ok": true}'}],
                        stop_reason="stop_sequence",
                        stop_sequence="<STOP>",
                        usage={"input_tokens": 99, "output_tokens": 99},
                    )
                ),
            )
        )
        comparer.compare_result(expect, good_actual)

        # Empty content fails lightweight output health check when skip_content/skip_choices=True
        empty_actual = comparer.format_result(
            comparer.curl_response_to_json(
                query,
                json.dumps(
                    _basic_golden(
                        content=[{"type": "text", "text": "   "}],
                        stop_reason="stop_sequence",
                        stop_sequence="<STOP>",
                    )
                ),
            )
        )
        with self.assertRaises(SmokeException):
            comparer.compare_result(expect, empty_actual)

        # Stream=True + expected_status_code=400 expects error JSON, NOT SSE
        err_qr = {
            "expected_status_code": 400,
            "query": {
                "model": "qwen2.5",
                "max_tokens": 8,
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
            "result": {
                "type": "error",
                "error": {"type": "invalid_request_error", "message": "bad input"},
            },
        }
        err_cmp = AnthropicComparer(None, "/v1/messages", err_qr, Tracer(), False)
        err_query = err_cmp.format_query(err_qr["query"])
        err_actual = err_cmp.format_result(
            err_cmp.curl_response_to_json(
                err_query,
                json.dumps(err_qr["result"]),
            )
        )
        err_cmp.compare_result(err_cmp.format_result(err_qr["result"]), err_actual)

    def test_streaming_sse_reconstruction_and_strict_validation(self):
        events = [
            (
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_live",
                        "type": "message",
                        "role": "assistant",
                        "model": "qwen3",
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {
                            "input_tokens": 20,
                            "output_tokens": 0,
                            "cache_read_input_tokens": 4,
                        },
                    },
                },
            ),
            ("ping", {"type": "ping"}),
            (
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "thinking", "thinking": ""},
                },
            ),
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "thinking_delta", "thinking": "Step 1"},
                },
            ),
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "signature_delta", "signature": ""},
                },
            ),
            ("content_block_stop", {"type": "content_block_stop", "index": 0}),
            (
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 1,
                    "content_block": {"type": "text", "text": ""},
                },
            ),
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 1,
                    "delta": {"type": "text_delta", "text": "Checking"},
                },
            ),
            ("content_block_stop", {"type": "content_block_stop", "index": 1}),
            (
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 2,
                    "content_block": {
                        "type": "tool_use",
                        "id": "toolu_999",
                        "name": "weather",
                        "input": {},
                    },
                },
            ),
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 2,
                    "delta": {"type": "input_json_delta", "partial_json": '{"city": '},
                },
            ),
            (
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 2,
                    "delta": {"type": "input_json_delta", "partial_json": '"Paris"}'},
                },
            ),
            ("content_block_stop", {"type": "content_block_stop", "index": 2}),
            (
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "tool_use", "stop_sequence": None},
                    "usage": {"output_tokens": 11},
                },
            ),
            ("message_stop", {"type": "message_stop"}),
        ]

        golden = {
            "id": "msg_smoke",
            "type": "message",
            "role": "assistant",
            "model": "qwen3",
            "content": [
                {"type": "thinking", "thinking": "Step 1", "signature": ""},
                {"type": "text", "text": "Checking"},
                {
                    "type": "tool_use",
                    "id": "toolu_other",
                    "name": "weather",
                    "input": {"city": "Paris"},
                },
            ],
            "stop_reason": "tool_use",
            "stop_sequence": None,
            "usage": {
                "input_tokens": 20,
                "output_tokens": 11,
                "cache_read_input_tokens": 4,
            },
        }
        qr_info = {
            "query": {
                "model": "qwen3",
                "max_tokens": 64,
                "stream": True,
                "messages": [{"role": "user", "content": "Weather?"}],
            },
            "result": golden,
        }
        comparer = AnthropicComparer(None, "/v1/messages", qr_info, Tracer(), False)
        query = comparer.format_query(qr_info["query"])
        actual = comparer.format_result(
            comparer.curl_response_to_json(query, _sse_bytes(events))
        )
        expect = comparer.format_result(qr_info["result"])
        comparer.compare_result(expect, actual)

    def test_streaming_rejects_malformed_lines_and_protocol_violations(self):
        comparer = AnthropicComparer(
            None,
            "/v1/messages",
            {
                "query": {
                    "max_tokens": 8,
                    "stream": True,
                    "messages": [{"role": "user", "content": "hi"}],
                },
                "result": {},
            },
            Tracer(),
            False,
        )
        query = comparer.format_query(comparer.qr_info["query"])

        # Raw invalid SSE line / duplicate event without blank line
        for raw_bad in (
            [b"not-an-sse-line"],
            [b"event: message_start", b"event: ping", b"data: {}"],
        ):
            with self.assertRaises(ValueError):
                comparer.curl_response_to_json(query, raw_bad)

        start_ev = (
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "m",
                    "type": "message",
                    "role": "assistant",
                    "model": "q",
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 5, "output_tokens": 0},
                },
            },
        )
        # Missing message_start, non-zero initial output_tokens, missing delta key, duplicate message_delta, SSE error event
        bad_streams = [
            (
                ValueError,
                [
                    (
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": 0,
                            "content_block": {"type": "text", "text": ""},
                        },
                    )
                ],
            ),
            (
                ValueError,
                [
                    (
                        "message_start",
                        {
                            "type": "message_start",
                            "message": {
                                "id": "m",
                                "type": "message",
                                "role": "assistant",
                                "model": "q",
                                "content": [],
                                "usage": {"input_tokens": 5, "output_tokens": 3},
                            },
                        },
                    ),
                    (
                        "message_delta",
                        {
                            "type": "message_delta",
                            "delta": {},
                            "usage": {"output_tokens": 1},
                        },
                    ),
                    ("message_stop", {"type": "message_stop"}),
                ],
            ),
            (
                ValueError,
                [
                    start_ev,
                    (
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": 0,
                            "content_block": {"type": "text", "text": ""},
                        },
                    ),
                    (
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": 0,
                            "delta": {"type": "text_delta"},
                        },
                    ),
                    ("content_block_stop", {"type": "content_block_stop", "index": 0}),
                    (
                        "message_delta",
                        {
                            "type": "message_delta",
                            "delta": {},
                            "usage": {"output_tokens": 1},
                        },
                    ),
                    ("message_stop", {"type": "message_stop"}),
                ],
            ),
            (
                ValueError,
                [
                    start_ev,
                    (
                        "message_delta",
                        {
                            "type": "message_delta",
                            "delta": {},
                            "usage": {"output_tokens": 1},
                        },
                    ),
                    (
                        "message_delta",
                        {
                            "type": "message_delta",
                            "delta": {},
                            "usage": {"output_tokens": 2},
                        },
                    ),
                    ("message_stop", {"type": "message_stop"}),
                ],
            ),
            (
                SmokeException,
                [
                    start_ev,
                    (
                        "error",
                        {
                            "type": "error",
                            "error": {"type": "api_error", "message": "failed"},
                        },
                    ),
                ],
            ),
        ]
        for exc_cls, bad_events in bad_streams:
            with self.subTest(bad_events=bad_events[0][0]):
                with self.assertRaises(exc_cls):
                    comparer.curl_response_to_json(query, _sse_bytes(bad_events))

    def test_count_tokens_compare_with_query_params(self):
        qr_info = {
            "query": {
                "model": "qwen2.5",
                "messages": [{"role": "user", "content": "hello"}],
            },
            "result": {"input_tokens": 15},
        }
        comparer = AnthropicComparer(
            None, "/v1/messages/count_tokens?beta=1", qr_info, Tracer(), False
        )
        self.assertTrue(comparer.is_count_tokens)
        query = comparer.format_query(qr_info["query"])
        actual = comparer.format_result(
            comparer.curl_response_to_json(query, '{"input_tokens": 15}')
        )
        expect = comparer.format_result(qr_info["result"])
        comparer.compare_result(expect, actual)


if __name__ == "__main__":
    unittest.main()
