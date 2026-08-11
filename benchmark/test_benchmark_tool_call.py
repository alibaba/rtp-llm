#!/usr/bin/env python3

import asyncio
import contextlib
import io
import json
import re
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from aiohttp import ClientSession, ClientTimeout, TCPConnector, web

from benchmark_tool_call import (
    RequestResult,
    ToolCallState,
    benchmark_failed,
    build_payload,
    get_recovery_error,
    get_warmup_error,
    parse_args,
    run_benchmark,
    run_request,
    select_cancel_flags,
    summarize_results,
    validate_completed_request,
)


class FakeContent:
    def __init__(self, lines: list[bytes], line_delay_s: float = 0.0):
        self.lines = lines
        self.line_delay_s = line_delay_s

    def __aiter__(self):
        return self.iter_lines()

    async def iter_lines(self):
        for line in self.lines:
            if self.line_delay_s:
                await asyncio.sleep(self.line_delay_s)
            yield line


class FakeResponse:
    status = 200

    def __init__(
        self,
        chunks: list[object],
        include_done: bool = True,
        line_delay_s: float = 0.0,
        enter_delay_s: float = 0.0,
    ):
        self.closed = False
        self.enter_delay_s = enter_delay_s
        lines = [f"data: {json.dumps(chunk)}\n\n".encode() for chunk in chunks]
        if include_done:
            lines.append(b"data: [DONE]\n\n")
        self.content = FakeContent(lines, line_delay_s)

    async def __aenter__(self):
        if self.enter_delay_s:
            await asyncio.sleep(self.enter_delay_s)
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        return False

    def close(self):
        self.closed = True


class FakeSession:
    def __init__(
        self,
        chunks: list[object],
        include_done: bool = True,
        line_delay_s: float = 0.0,
        enter_delay_s: float = 0.0,
    ):
        self.response = FakeResponse(
            chunks, include_done, line_delay_s, enter_delay_s
        )
        self.payload = None

    def post(self, endpoint: str, json: dict):
        self.payload = json
        return self.response


class AdvancingSemaphore:
    def __init__(self, clock: list[float], queue_wait_s: float):
        self.clock = clock
        self.queue_wait_s = queue_wait_s

    async def __aenter__(self):
        self.clock[0] += self.queue_wait_s

    async def __aexit__(self, exc_type, exc_value, traceback):
        return False


class BuildPayloadTest(unittest.TestCase):
    def test_exactly_once_policy_is_sent_to_server(self):
        payload = build_payload("fake-model", 7, 32)

        self.assertEqual(
            payload["tool_choice"],
            {"type": "function", "function": {"name": "echo_request"}},
        )
        self.assertFalse(payload["parallel_tool_calls"])

    def test_auto_parallel_policy_can_be_requested_explicitly(self):
        payload = build_payload(
            "fake-model", 7, 32, tool_choice="auto", parallel_tool_calls=True
        )

        self.assertEqual(payload["tool_choice"], "auto")
        self.assertTrue(payload["parallel_tool_calls"])
        self.assertEqual(
            [tool["function"]["name"] for tool in payload["tools"]],
            ["echo_request", "echo_payload"],
        )
        self.assertIn("Call echo_payload exactly once", payload["messages"][0]["content"])

    def test_named_parallel_policy_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "require tool_choice"):
            build_payload(
                "fake-model", 7, 32, tool_choice="named", parallel_tool_calls=True
            )


class BenchmarkGateTest(unittest.TestCase):
    def test_semantic_errors_fail_by_default(self):
        summary = {"structural_error_count": 0, "semantic_error_count": 1}

        self.assertTrue(benchmark_failed(summary))
        self.assertFalse(benchmark_failed(summary, fail_on_semantic_errors=False))

    def test_warmup_rejects_semantic_errors(self):
        result = RequestResult(request_index=-1, expected_cancel=False)
        result.semantic_errors.append("response contains no tool call")

        error = get_warmup_error(result)

        self.assertIn("warmup request -1 failed", error)
        self.assertIn("response contains no tool call", error)

    def test_allow_semantic_errors_does_not_abort_on_semantic_warmup(self):
        result = RequestResult(request_index=-1, expected_cancel=False)
        result.semantic_errors.append("response contains no tool call")

        self.assertIsNone(
            get_warmup_error(result, fail_on_semantic_errors=False)
        )

    def test_cli_defaults_to_named_single_tool_and_strict_semantics(self):
        args = parse_args(["--model", "fake-model"])

        self.assertEqual(args.tool_choice, "named")
        self.assertFalse(args.parallel_tool_calls)
        self.assertEqual(args.cancel_after, 0.05)
        self.assertEqual(args.cancel_dispatch_timeout, 10.0)
        self.assertEqual(args.worker_status_timeout, 5.0)
        self.assertTrue(args.fail_on_semantic_errors)

    def test_cli_can_opt_into_auto_parallel_and_semantic_reporting(self):
        args = parse_args(
            [
                "--model",
                "fake-model",
                "--tool-choice",
                "auto",
                "--parallel-tool-calls",
                "--allow-semantic-errors",
            ]
        )

        self.assertEqual(args.tool_choice, "auto")
        self.assertTrue(args.parallel_tool_calls)
        self.assertFalse(args.fail_on_semantic_errors)

    def test_cli_rejects_named_parallel_policy(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args(["--model", "fake-model", "--parallel-tool-calls"])

    def test_cancellation_selection_is_exact_and_keeps_a_normal_request(self):
        first = select_cancel_flags(10, 0.2, seed=7)
        second = select_cancel_flags(10, 0.2, seed=7)

        self.assertEqual(first, second)
        self.assertEqual(sum(first), 2)
        self.assertEqual(sum(select_cancel_flags(2, 0.99, seed=7)), 1)
        self.assertEqual(sum(select_cancel_flags(1, 0.99, seed=7)), 0)

    def test_cli_rejects_zero_cancel_delay(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args(
                    ["--model", "fake-model", "--cancel-after", "0"]
                )


class ValidateCompletedRequestTest(unittest.TestCase):
    REQUEST_INDEX = 7

    def make_tool_call(self, index: int) -> ToolCallState:
        return ToolCallState(
            index=index,
            call_id=f"call-{self.REQUEST_INDEX}-{index}",
            call_type="function",
            name="echo_request",
            arguments=json.dumps(
                {
                    "request_id": f"request-{self.REQUEST_INDEX:08d}",
                    "payload": f"payload-{self.REQUEST_INDEX:08d}",
                }
            ),
            name_emissions=1,
        )

    def make_result(self, tool_calls: list[ToolCallState]) -> RequestResult:
        return RequestResult(
            request_index=self.REQUEST_INDEX,
            expected_cancel=False,
            http_status=200,
            finish_reason="tool_calls",
            saw_done=True,
            done_count=1,
            tool_calls=tool_calls,
        )

    def test_exactly_one_tool_call_passes(self):
        result = self.make_result([self.make_tool_call(0)])

        validate_completed_request(result)

        self.assertEqual(result.structural_errors, [])
        self.assertEqual(result.semantic_errors, [])

    def test_missing_tool_type_fails(self):
        tool_call = self.make_tool_call(0)
        tool_call.call_type = None
        result = self.make_result([tool_call])

        validate_completed_request(result)

        self.assertIn(
            "tool 0 type is None, expected 'function'",
            result.structural_errors,
        )

    def test_deeply_nested_tool_arguments_are_reported_without_crashing(self):
        tool_call = self.make_tool_call(0)
        tool_call.arguments = "[" * 2000 + "]" * 2000
        result = self.make_result([tool_call])

        validate_completed_request(result)

        self.assertTrue(
            any(
                "invalid JSON arguments: RecursionError" in error
                for error in result.structural_errors
            )
        )

    def test_no_tool_call_fails(self):
        result = self.make_result([])

        validate_completed_request(result)

        self.assertEqual(result.structural_errors, [])
        self.assertEqual(result.semantic_errors, ["response contains no tool call"])

    def test_normal_text_is_recorded_without_a_tool_call(self):
        result = self.make_result([])
        result.normal_text = "normal text"

        validate_completed_request(result)

        self.assertEqual(result.structural_errors, [])
        self.assertEqual(
            result.semantic_errors,
            [
                "response contains normal text: 'normal text'",
                "response contains no tool call",
            ],
        )

    def test_two_valid_unique_tool_calls_still_fail(self):
        result = self.make_result([self.make_tool_call(0), self.make_tool_call(1)])

        validate_completed_request(result)

        self.assertEqual(result.structural_errors, [])
        self.assertIn(
            "response contains 2 tool calls, expected exactly 1",
            result.semantic_errors,
        )
        self.assertIn(
            "function 'echo_request' was called 2 times, expected once",
            result.semantic_errors,
        )

    def test_parallel_profile_requires_both_request_scoped_tools(self):
        result = self.make_result(
            [
                ToolCallState(
                    index=0,
                    call_id="call-7-0",
                    call_type="function",
                    name="echo_payload",
                    arguments=json.dumps({"payload": "payload-00000007"}),
                    name_emissions=1,
                ),
                ToolCallState(
                    index=1,
                    call_id="call-7-1",
                    call_type="function",
                    name="echo_request",
                    arguments=json.dumps({"request_id": "request-00000007"}),
                    name_emissions=1,
                ),
            ]
        )
        result.parallel_tool_calls = True

        validate_completed_request(result)

        self.assertEqual(result.structural_errors, [])
        self.assertEqual(result.semantic_errors, [])

    def test_parallel_profile_rejects_a_single_tool(self):
        result = self.make_result(
            [
                ToolCallState(
                    index=0,
                    call_id="call-7-0",
                    call_type="function",
                    name="echo_request",
                    arguments=json.dumps({"request_id": "request-00000007"}),
                    name_emissions=1,
                )
            ]
        )
        result.parallel_tool_calls = True

        validate_completed_request(result)

        self.assertIn(
            "response contains 1 tool calls, expected exactly 2",
            result.semantic_errors,
        )
        self.assertIn(
            "function 'echo_payload' was called 0 times, expected once",
            result.semantic_errors,
        )


class RecoveryValidationTest(unittest.TestCase):
    def test_missing_status_fails_when_cancellation_was_selected(self):
        self.assertEqual(
            get_recovery_error(None, None, expected_cancel_count=1),
            "cannot verify frontend concurrency recovery: before=None, after=None",
        )

    def test_invalid_status_value_is_bounded_in_error(self):
        error = get_recovery_error(
            {"frontend_available_concurrency": "x" * 1000},
            {"frontend_available_concurrency": False},
            expected_cancel_count=1,
        )

        self.assertIsNotNone(error)
        self.assertLessEqual(len(error), 400)
        self.assertIn("before='xxx", error)
        self.assertIn("...", error)
        self.assertIn("after=False", error)

    def test_missing_status_is_ignored_without_selected_cancellation(self):
        self.assertIsNone(get_recovery_error(None, None, expected_cancel_count=0))

    def test_same_or_increased_availability_passes(self):
        for after in (32, 33):
            with self.subTest(after=after):
                self.assertIsNone(
                    get_recovery_error(
                        {"frontend_available_concurrency": 32},
                        {"frontend_available_concurrency": after},
                        expected_cancel_count=1,
                    )
                )

    def test_decreased_availability_fails(self):
        self.assertEqual(
            get_recovery_error(
                {"frontend_available_concurrency": 32},
                {"frontend_available_concurrency": 31},
                expected_cancel_count=1,
            ),
            "frontend concurrency did not recover: before=32, after=31",
        )


class SummaryTest(unittest.TestCase):
    def make_args(self, requests: int = 4):
        return SimpleNamespace(
            model="fake-model",
            requests=requests,
            concurrency=4,
            cancel_rate=0.25,
            cancel_after=0.01,
            cancel_dispatch_timeout=1.0,
            warmup=2,
            max_tokens=32,
            timeout=10.0,
            worker_status_timeout=1.0,
            recovery_wait=0.0,
            seed=7,
            tool_choice="named",
            parallel_tool_calls=False,
            max_reported_errors=100,
        )

    def make_success(self, request_index: int, elapsed_s: float = 1.0):
        return RequestResult(
            request_index=request_index,
            expected_cancel=False,
            http_status=200,
            saw_done=True,
            done_count=1,
            elapsed_s=elapsed_s,
            ttft_s=elapsed_s / 10,
            tool_calls=[
                ToolCallState(
                    index=0,
                    call_id=f"call-{request_index}",
                    call_type="function",
                )
            ],
        )

    def test_failed_and_cancelled_requests_do_not_pollute_success_metrics(self):
        success = self.make_success(0)
        semantic_failure = self.make_success(1, elapsed_s=100.0)
        semantic_failure.semantic_errors.append("wrong tool arguments")
        http_failure = RequestResult(
            request_index=2,
            expected_cancel=False,
            http_status=503,
            elapsed_s=100.0,
            structural_errors=["HTTP 503"],
        )
        cancellation = RequestResult(
            request_index=3,
            expected_cancel=True,
            cancelled=True,
            elapsed_s=100.0,
        )

        summary = summarize_results(
            self.make_args(),
            "http://unused/chat/completions",
            [success, semantic_failure, http_failure, cancellation],
            wall_time_s=2.0,
            status_before={"frontend_available_concurrency": 4},
            status_after={"frontend_available_concurrency": 4},
        )

        self.assertEqual(summary["completed_requests"], 1)
        self.assertEqual(summary["failed_requests"], 2)
        self.assertEqual(summary["selected_cancellation_requests"], 1)
        self.assertEqual(summary["cancelled_requests"], 1)
        self.assertEqual(summary["throughput_rps"], 0.5)
        self.assertEqual(summary["attempted_rps"], 2.0)
        self.assertEqual(summary["latency_s"]["mean"], 1.0)

    def test_duplicate_call_id_disqualifies_the_later_request(self):
        first = self.make_success(0)
        second = self.make_success(1)
        second.tool_calls[0].call_id = first.tool_calls[0].call_id

        summary = summarize_results(
            self.make_args(requests=2),
            "http://unused/chat/completions",
            [first, second],
            wall_time_s=1.0,
            status_before=None,
            status_after=None,
        )

        self.assertEqual(summary["duplicate_call_id_count"], 1)
        self.assertEqual(summary["structural_error_count"], 1)
        self.assertEqual(summary["completed_requests"], 1)
        self.assertEqual(summary["failed_requests"], 1)


class RunRequestTextValidationTest(unittest.IsolatedAsyncioTestCase):
    REQUEST_INDEX = 7

    def make_chunk(self, **delta_fields) -> dict:
        return {
            "choices": [
                {
                    "index": 0,
                    "delta": delta_fields,
                    "finish_reason": None,
                }
            ]
        }

    def make_tool_chunk(self) -> dict:
        return {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call-7-0",
                                "type": "function",
                                "function": {
                                    "name": "echo_request",
                                    "arguments": json.dumps(
                                        {
                                            "request_id": "request-00000007",
                                            "payload": "payload-00000007",
                                        }
                                    ),
                                },
                            }
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }

    async def run_chunks(
        self,
        chunks: list[object],
        include_done: bool = True,
        parallel_tool_calls: bool = False,
    ) -> RequestResult:
        return await run_request(
            FakeSession(chunks, include_done=include_done),
            "http://unused/chat/completions",
            "fake-model",
            self.REQUEST_INDEX,
            32,
            False,
            asyncio.Semaphore(1),
            "required" if parallel_tool_calls else "named",
            parallel_tool_calls,
        )

    async def run_raw_lines(self, lines: list[bytes]) -> RequestResult:
        session = FakeSession([])
        session.response.content = FakeContent(lines)
        return await run_request(
            session,
            "http://unused/chat/completions",
            "fake-model",
            self.REQUEST_INDEX,
            32,
            False,
            asyncio.Semaphore(1),
        )

    async def run_cancel_chunks(
        self,
        chunks: list[object],
        line_delay_s: float = 0.0,
        cancel_after_s: float = 0.05,
        enter_delay_s: float = 0.0,
        cancel_dispatch_timeout_s: float = 1.0,
    ) -> tuple[RequestResult, FakeResponse]:
        session = FakeSession(
            chunks,
            line_delay_s=line_delay_s,
            enter_delay_s=enter_delay_s,
        )
        result = await run_request(
            session,
            "http://unused/chat/completions",
            "fake-model",
            self.REQUEST_INDEX,
            32,
            True,
            asyncio.Semaphore(1),
            cancel_after_s=cancel_after_s,
            cancel_dispatch_timeout_s=cancel_dispatch_timeout_s,
        )
        return result, session.response

    async def test_nonempty_content_fails(self):
        result = await self.run_chunks(
            [
                self.make_chunk(content="normal "),
                self.make_chunk(content="text"),
                self.make_tool_chunk(),
            ]
        )

        self.assertEqual(result.structural_errors, [])
        self.assertEqual(
            result.semantic_errors,
            ["response contains normal text: 'normal text'"],
        )

    async def test_reasoning_content_with_tool_call_passes(self):
        result = await self.run_chunks(
            [
                self.make_chunk(reasoning_content="legitimate reasoning"),
                self.make_tool_chunk(),
            ]
        )

        self.assertEqual(result.structural_errors, [])
        self.assertEqual(result.semantic_errors, [])

    async def test_parallel_tool_chunks_are_merged_and_isolated_by_index(self):
        chunks = [
            self.make_chunk(
                tool_calls=[
                    {
                        "index": 0,
                        "id": "call-7-0",
                        "type": "function",
                        "function": {
                            "name": "echo_request",
                            "arguments": '{"request_id":"request-',
                        },
                    },
                    {
                        "index": 1,
                        "id": "call-7-1",
                        "type": "function",
                        "function": {
                            "name": "echo_payload",
                            "arguments": '{"payload":"payload-',
                        },
                    },
                ]
            ),
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 1,
                                    "function": {"arguments": '00000007"}'},
                                },
                                {
                                    "index": 0,
                                    "function": {"arguments": '00000007"}'},
                                },
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            },
        ]

        result = await self.run_chunks(chunks, parallel_tool_calls=True)

        self.assertEqual(result.structural_errors, [])
        self.assertEqual(result.semantic_errors, [])

    async def test_falsy_non_string_content_fails_structural_validation(self):
        for content in (0, False):
            with self.subTest(content=content, content_type=type(content).__name__):
                result = await self.run_chunks(
                    [self.make_chunk(content=content), self.make_tool_chunk()]
                )

                self.assertEqual(
                    result.structural_errors,
                    ["content delta is not a string"],
                )
                self.assertEqual(result.semantic_errors, [])

    async def test_falsy_non_string_arguments_fail_structural_validation(self):
        for arguments in (0, False):
            with self.subTest(
                arguments=arguments, arguments_type=type(arguments).__name__
            ):
                malformed = self.make_chunk(
                    tool_calls=[
                        {
                            "index": 0,
                            "function": {"arguments": arguments},
                        }
                    ]
                )
                result = await self.run_chunks([malformed, self.make_tool_chunk()])

                self.assertIn(
                    "tool 0 arguments delta is not a string",
                    result.structural_errors,
                )
                self.assertEqual(result.semantic_errors, [])

    async def test_bool_tool_index_is_not_accepted_as_zero(self):
        malformed = self.make_chunk(
            tool_calls=[
                {
                    "index": False,
                    "function": {"name": "echo_request", "arguments": "{}"},
                }
            ]
        )

        result = await self.run_chunks([malformed, self.make_tool_chunk()])

        self.assertIn("invalid tool index: False", result.structural_errors)
        self.assertEqual(result.semantic_errors, [])

    async def test_non_string_reasoning_content_is_structural_error(self):
        result = await self.run_chunks(
            [self.make_chunk(reasoning_content=1), self.make_tool_chunk()]
        )

        self.assertIn(
            "reasoning_content delta is not a string", result.structural_errors
        )
        self.assertEqual(result.semantic_errors, [])

    async def test_success_requires_done_sentinel(self):
        result = await self.run_chunks([self.make_tool_chunk()], include_done=False)

        self.assertIn("stream ended before [DONE]", result.structural_errors)
        self.assertEqual(result.semantic_errors, [])

    async def test_exactly_one_done_sentinel_passes(self):
        result = await self.run_chunks([self.make_tool_chunk()])

        self.assertTrue(result.saw_done)
        self.assertEqual(result.done_count, 1)
        self.assertEqual(result.structural_errors, [])
        self.assertEqual(result.semantic_errors, [])

    async def test_latency_excludes_client_semaphore_queue_wait(self):
        clock = [100.0]
        with patch(
            "benchmark_tool_call.time.perf_counter",
            side_effect=lambda: clock[0],
        ):
            result = await run_request(
                FakeSession([self.make_tool_chunk()]),
                "http://unused/chat/completions",
                "fake-model",
                self.REQUEST_INDEX,
                32,
                False,
                AdvancingSemaphore(clock, queue_wait_s=30.0),
            )

        self.assertEqual(result.ttft_s, 0.0)
        self.assertEqual(result.elapsed_s, 0.0)

    async def test_duplicate_done_sentinel_fails(self):
        tool_chunk = json.dumps(self.make_tool_chunk()).encode("utf-8")
        result = await self.run_raw_lines(
            [
                b"data: " + tool_chunk + b"\n\n",
                b"data: [DONE]\n\n",
                b"data: [DONE]\n\n",
            ]
        )

        self.assertEqual(result.done_count, 2)
        self.assertIn(
            "stream contains 2 [DONE] sentinels, expected exactly one",
            result.structural_errors,
        )

    async def test_data_after_done_sentinel_fails(self):
        tool_chunk = json.dumps(self.make_tool_chunk()).encode("utf-8")
        trailing_chunk = json.dumps({"choices": []}).encode("utf-8")
        result = await self.run_raw_lines(
            [
                b"data: " + tool_chunk + b"\n\n",
                b"data: [DONE]\n\n",
                b"data: " + trailing_chunk + b"\n\n",
            ]
        )

        self.assertEqual(result.done_count, 1)
        self.assertTrue(
            any(
                error.startswith("SSE data received after [DONE]:")
                for error in result.structural_errors
            )
        )

    async def test_error_after_done_sentinel_fails(self):
        tool_chunk = json.dumps(self.make_tool_chunk()).encode("utf-8")
        error_chunk = json.dumps({"error": {"message": "late failure"}}).encode(
            "utf-8"
        )
        result = await self.run_raw_lines(
            [
                b"data: " + tool_chunk + b"\n\n",
                b"data: [DONE]\n\n",
                b"data: " + error_chunk + b"\n\n",
            ]
        )

        self.assertEqual(result.done_count, 1)
        self.assertTrue(
            any("late failure" in error for error in result.structural_errors)
        )

    async def test_falsy_choice_delta_and_function_shapes_are_rejected(self):
        malformed_chunks = [
            {"choices": False},
            {"choices": None},
            {
                "choices": [
                    {"index": 0, "delta": False, "finish_reason": None}
                ]
            },
            {
                "choices": [
                    {"index": 0, "delta": None, "finish_reason": None}
                ]
            },
            self.make_chunk(tool_calls=[{"index": 0, "function": False}]),
            self.make_chunk(tool_calls=[{"index": 0, "function": None}]),
        ]
        expected_errors = [
            "SSE choices is not a list",
            "SSE choices is not a list",
            "choice delta is not an object",
            "choice delta is not an object",
            "tool 0 function delta is not an object",
            "tool 0 function delta is not an object",
        ]

        for malformed, expected_error in zip(malformed_chunks, expected_errors):
            with self.subTest(expected_error=expected_error):
                result = await self.run_chunks([malformed, self.make_tool_chunk()])

                self.assertIn(expected_error, result.structural_errors)
                self.assertEqual(result.semantic_errors, [])

    async def test_request_is_actively_cancelled_before_the_first_chunk(self):
        result, response = await self.run_cancel_chunks(
            [self.make_chunk(content="first token")],
            line_delay_s=0.05,
            cancel_after_s=0.001,
        )

        self.assertTrue(result.cancelled)
        self.assertTrue(response.closed)
        self.assertFalse(result.saw_done)
        self.assertEqual(result.done_count, 0)
        self.assertEqual(result.structural_errors, [])

    async def test_cancel_before_response_headers_is_not_a_clean_success(self):
        result, response = await self.run_cancel_chunks(
            [self.make_chunk(content="first token")],
            enter_delay_s=0.05,
            cancel_dispatch_timeout_s=0.001,
        )

        self.assertTrue(result.cancelled)
        self.assertFalse(response.closed)
        self.assertEqual(len(result.structural_errors), 1)
        self.assertIn(
            "response headers were not received",
            result.structural_errors[0],
        )

    async def test_terminal_first_choice_is_not_counted_as_cancelled(self):
        result, response = await self.run_cancel_chunks([self.make_tool_chunk()])

        self.assertFalse(result.cancelled)
        self.assertFalse(response.closed)
        self.assertEqual(
            result.structural_errors,
            ["request selected for cancellation did not cancel"],
        )

    async def test_error_chunk_is_not_counted_as_cancelled(self):
        result, response = await self.run_cancel_chunks(
            [
                {"error": {"message": "request failed"}},
                {"usage": {"completion_tokens": 1}},
                {"choices": []},
            ]
        )

        self.assertFalse(result.cancelled)
        self.assertFalse(response.closed)
        self.assertEqual(
            result.structural_errors,
            [
                "SSE error: {'message': 'request failed'}",
                "request selected for cancellation did not cancel",
            ],
        )

    async def test_typed_sse_error_is_terminal_structural_failure(self):
        result = await self.run_chunks(
            [
                {
                    "error_code": 606,
                    "error_code_str": "EXECUTION_EXCEPTION",
                    "message": "tool policy failed",
                }
            ]
        )

        self.assertEqual(len(result.structural_errors), 1)
        self.assertIn("606", result.structural_errors[0])
        self.assertIn("tool policy failed", result.structural_errors[0])
        self.assertEqual(result.semantic_errors, [])

    async def test_openai_sse_error_is_terminal_structural_failure(self):
        result = await self.run_chunks(
            [{"error": {"message": "request failed", "type": "server_error"}}]
        )

        self.assertEqual(len(result.structural_errors), 1)
        self.assertIn("request failed", result.structural_errors[0])
        self.assertEqual(result.semantic_errors, [])

    async def test_non_object_sse_json_is_reported_without_crashing(self):
        result = await self.run_chunks([["not", "an", "object"]])

        self.assertEqual(len(result.structural_errors), 1)
        self.assertIn("SSE payload is not an object", result.structural_errors[0])
        self.assertEqual(result.semantic_errors, [])

    async def test_deeply_nested_sse_json_is_reported_without_crashing(self):
        nested_json = "[" * 2000 + "]" * 2000
        session = FakeSession([])
        session.response.content = FakeContent(
            [
                f"data: {nested_json}\n\n".encode("utf-8"),
                b"data: [DONE]\n\n",
            ]
        )

        result = await run_request(
            session,
            "http://unused/chat/completions",
            "fake-model",
            self.REQUEST_INDEX,
            32,
            False,
            asyncio.Semaphore(1),
        )

        self.assertTrue(
            any(
                "invalid SSE JSON: RecursionError" in error
                for error in result.structural_errors
            )
        )

    async def test_malformed_choice_shapes_are_reported_without_crashing(self):
        malformed_chunks = [
            {"choices": "not-a-list"},
            {"choices": ["not-an-object"]},
            {
                "choices": [
                    {"index": 0, "delta": "not-an-object", "finish_reason": None}
                ]
            },
            self.make_chunk(tool_calls="not-a-list"),
        ]
        expected_errors = [
            "SSE choices is not a list",
            "SSE choice is not an object",
            "choice delta is not an object",
            "tool_calls delta is not a list",
        ]

        for malformed, expected_error in zip(malformed_chunks, expected_errors):
            with self.subTest(expected_error=expected_error):
                result = await self.run_chunks([malformed, self.make_tool_chunk()])

                self.assertIn(expected_error, result.structural_errors)
                self.assertEqual(result.semantic_errors, [])

    async def test_multiple_choices_in_one_chunk_are_rejected(self):
        malformed = self.make_tool_chunk()
        malformed["choices"].append(
            {"index": 0, "delta": {}, "finish_reason": None}
        )

        result = await self.run_chunks([malformed])

        self.assertIn(
            "SSE chunk contains 2 choices, expected at most one",
            result.structural_errors,
        )

    async def test_concurrent_requests_keep_markers_and_call_ids_isolated(self):
        semaphore = asyncio.Semaphore(64)

        async def run_one(request_index: int) -> RequestResult:
            arguments = json.dumps(
                {
                    "request_id": f"request-{request_index:08d}",
                    "payload": f"payload-{request_index:08d}",
                }
            )
            chunk = {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": f"call-{request_index}",
                                    "type": "function",
                                    "function": {
                                        "name": "echo_request",
                                        "arguments": arguments,
                                    },
                                }
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            }
            return await run_request(
                FakeSession([chunk]),
                "http://unused/chat/completions",
                "fake-model",
                request_index,
                32,
                False,
                semaphore,
            )

        results = await asyncio.gather(*(run_one(i) for i in range(256)))

        self.assertTrue(
            all(
                not result.structural_errors and not result.semantic_errors
                for result in results
            )
        )
        self.assertEqual(
            len({result.tool_calls[0].call_id for result in results}), 256
        )


class WireIntegrationTest(unittest.IsolatedAsyncioTestCase):
    async def start_server(self, handler):
        app = web.Application()
        app.router.add_post("/chat/completions", handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        return runner, f"http://127.0.0.1:{port}/chat/completions"

    def make_wire_chunk(self, request_index: int) -> dict:
        return {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": f"call-{request_index}",
                                "type": "function",
                                "function": {
                                    "name": "echo_request",
                                    "arguments": json.dumps(
                                        {
                                            "request_id": (
                                                f"request-{request_index:08d}"
                                            ),
                                            "payload": (
                                                f"payload-{request_index:08d}"
                                            ),
                                        }
                                    ),
                                },
                            }
                        ]
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        }

    async def test_cancelled_stream_releases_the_connection_slot(self):
        accepted = asyncio.Event()
        disconnected = asyncio.Event()
        request_count = 0

        async def handler(request):
            nonlocal request_count
            await request.json()
            request_count += 1
            current_request = request_count
            response = web.StreamResponse(
                status=200, headers={"Content-Type": "text/event-stream"}
            )
            await response.prepare(request)
            if current_request == 1:
                accepted.set()
                for _ in range(100):
                    transport = request.transport
                    if transport is None or transport.is_closing():
                        disconnected.set()
                        break
                    await asyncio.sleep(0.01)
                try:
                    await response.write_eof()
                except (ConnectionResetError, RuntimeError):
                    pass
                return response

            chunk = self.make_wire_chunk(8)
            await response.write(
                f"data: {json.dumps(chunk)}\n\n".encode("utf-8")
            )
            await response.write(b"data: [DONE]\n\n")
            await response.write_eof()
            return response

        runner, endpoint = await self.start_server(handler)
        connector = TCPConnector(limit=1)
        try:
            async with ClientSession(
                connector=connector, timeout=ClientTimeout(total=2.0)
            ) as session:
                cancelled_result = await run_request(
                    session,
                    endpoint,
                    "fake-model",
                    7,
                    32,
                    True,
                    asyncio.Semaphore(1),
                    cancel_after_s=0.01,
                    cancel_dispatch_timeout_s=0.5,
                )
                self.assertTrue(accepted.is_set())
                self.assertTrue(cancelled_result.cancelled)
                self.assertEqual(cancelled_result.structural_errors, [])

                completed_result = await asyncio.wait_for(
                    run_request(
                        session,
                        endpoint,
                        "fake-model",
                        8,
                        32,
                        False,
                        asyncio.Semaphore(1),
                    ),
                    timeout=1.0,
                )

            await asyncio.wait_for(disconnected.wait(), timeout=1.0)
            self.assertEqual(completed_result.structural_errors, [])
            self.assertEqual(completed_result.semantic_errors, [])
        finally:
            await runner.cleanup()

    async def test_done_does_not_bypass_the_existing_total_timeout(self):
        async def handler(request):
            await request.json()
            response = web.StreamResponse(
                status=200, headers={"Content-Type": "text/event-stream"}
            )
            await response.prepare(request)
            chunk = self.make_wire_chunk(7)
            await response.write(
                f"data: {json.dumps(chunk)}\n\n".encode("utf-8")
            )
            await response.write(b"data: [DONE]\n\n")
            await asyncio.sleep(0.2)
            try:
                await response.write_eof()
            except (ConnectionResetError, RuntimeError):
                pass
            return response

        runner, endpoint = await self.start_server(handler)
        try:
            async with ClientSession(timeout=ClientTimeout(total=0.05)) as session:
                result = await run_request(
                    session,
                    endpoint,
                    "fake-model",
                    7,
                    32,
                    False,
                    asyncio.Semaphore(1),
                )

            self.assertEqual(result.done_count, 1)
            self.assertTrue(
                any(error.startswith("TimeoutError:") for error in result.structural_errors)
            )
        finally:
            await runner.cleanup()

    async def test_done_before_cancel_deadline_is_not_counted_as_cancelled(self):
        async def handler(request):
            await request.json()
            response = web.StreamResponse(
                status=200, headers={"Content-Type": "text/event-stream"}
            )
            await response.prepare(request)
            chunk = self.make_wire_chunk(7)
            await response.write(
                f"data: {json.dumps(chunk)}\n\n".encode("utf-8")
            )
            await response.write(b"data: [DONE]\n\n")
            await asyncio.sleep(0.2)
            try:
                await response.write_eof()
            except (ConnectionResetError, RuntimeError):
                pass
            return response

        runner, endpoint = await self.start_server(handler)
        try:
            async with ClientSession(timeout=ClientTimeout(total=0.05)) as session:
                result = await run_request(
                    session,
                    endpoint,
                    "fake-model",
                    7,
                    32,
                    True,
                    asyncio.Semaphore(1),
                    cancel_after_s=0.02,
                )

            self.assertTrue(result.saw_done)
            self.assertEqual(result.done_count, 1)
            self.assertFalse(result.cancelled)
            self.assertTrue(
                any(error.startswith("TimeoutError:") for error in result.structural_errors)
            )
            self.assertIn(
                "request selected for cancellation did not cancel",
                result.structural_errors,
            )
        finally:
            await runner.cleanup()

    async def test_split_wire_frames_preserve_parallel_tool_calls(self):
        async def handler(request):
            payload = await request.json()
            prompt = payload["messages"][0]["content"]
            request_marker = re.search(r"request-\d{8}", prompt).group(0)
            payload_marker = re.search(r"payload-\d{8}", prompt).group(0)
            chunks = [
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "wire-call-0",
                                        "type": "function",
                                        "function": {
                                            "name": "echo_request",
                                            "arguments": '{"request_id":"',
                                        },
                                    },
                                    {
                                        "index": 1,
                                        "id": "wire-call-1",
                                        "type": "function",
                                        "function": {
                                            "name": "echo_payload",
                                            "arguments": '{"payload":"',
                                        },
                                    },
                                ]
                            },
                            "finish_reason": None,
                        }
                    ]
                },
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 1,
                                        "function": {
                                            "arguments": payload_marker + '"}'
                                        },
                                    },
                                    {
                                        "index": 0,
                                        "function": {
                                            "arguments": request_marker + '"}'
                                        },
                                    },
                                ]
                            },
                            "finish_reason": "tool_calls",
                        }
                    ]
                },
            ]
            wire = "".join(
                f"data: {json.dumps(chunk)}\n\n" for chunk in chunks
            ) + "data: [DONE]\n\n"
            encoded = wire.encode("utf-8")
            response = web.StreamResponse(
                status=200, headers={"Content-Type": "text/event-stream"}
            )
            await response.prepare(request)
            offsets = (1, 3, 8, 21, 55, len(encoded))
            start = 0
            for end in offsets:
                await response.write(encoded[start:end])
                await asyncio.sleep(0)
                start = end
            await response.write_eof()
            return response

        runner, endpoint = await self.start_server(handler)
        try:
            async with ClientSession(timeout=ClientTimeout(total=2.0)) as session:
                result = await run_request(
                    session,
                    endpoint,
                    "fake-model",
                    42,
                    32,
                    False,
                    asyncio.Semaphore(1),
                    tool_choice="required",
                    parallel_tool_calls=True,
                )

            self.assertEqual(result.structural_errors, [])
            self.assertEqual(result.semantic_errors, [])
        finally:
            await runner.cleanup()

    async def test_full_benchmark_counts_only_successful_completions(self):
        async def chat_handler(request):
            payload = await request.json()
            prompt = payload["messages"][0]["content"]
            marker = re.search(r"request-(\d{8})", prompt)
            request_index = int(marker.group(1))
            response = web.StreamResponse(
                status=200, headers={"Content-Type": "text/event-stream"}
            )
            await response.prepare(request)
            await asyncio.sleep(0.05)
            chunk = self.make_wire_chunk(request_index)
            try:
                await response.write(
                    f"data: {json.dumps(chunk)}\n\n".encode("utf-8")
                )
                await response.write(b"data: [DONE]\n\n")
                await response.write_eof()
            except (ConnectionResetError, RuntimeError):
                pass
            return response

        async def status_handler(request):
            return web.json_response({"frontend_available_concurrency": 4})

        app = web.Application()
        app.router.add_post("/v1/chat/completions", chat_handler)
        app.router.add_get("/worker_status", status_handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        args = parse_args(
            [
                "--base-url",
                f"http://127.0.0.1:{port}/v1",
                "--model",
                "fake-model",
                "--requests",
                "8",
                "--concurrency",
                "4",
                "--cancel-rate",
                "0.25",
                "--cancel-after",
                "0.01",
                "--recovery-wait",
                "0",
                "--warmup",
                "0",
                "--timeout",
                "2",
                "--worker-status-timeout",
                "1",
            ]
        )
        try:
            summary = await run_benchmark(args)

            self.assertEqual(summary["completed_requests"], 6)
            self.assertEqual(summary["failed_requests"], 0)
            self.assertEqual(summary["selected_cancellation_requests"], 2)
            self.assertEqual(summary["cancelled_requests"], 2)
            self.assertEqual(summary["failed_cancellation_requests"], 0)
            self.assertEqual(summary["structural_error_count"], 0)
            self.assertEqual(summary["semantic_error_count"], 0)
        finally:
            await runner.cleanup()


if __name__ == "__main__":
    unittest.main()
