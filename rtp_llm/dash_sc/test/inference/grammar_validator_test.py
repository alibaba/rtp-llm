from __future__ import annotations

import io
import queue
import signal
import threading
import unittest
from collections import OrderedDict
from unittest.mock import MagicMock

from rtp_llm.dash_sc.inference.grammar_validator import (
    GrammarCheckUnavailable,
    GrammarValidator,
    _compile_exception_reply,
    _format_worker_exitcode,
    _GrammarCheckResult,
    _is_resource_exhaustion,
    _WorkerStatus,
)


class CompileExceptionReplyTest(unittest.TestCase):
    def test_resource_exhaustion_is_unavailable_and_retires_worker(self) -> None:
        for error in (MemoryError("out of memory"), RuntimeError("std::bad_alloc")):
            self.assertTrue(_is_resource_exhaustion(error))
            status, retire_after_reply, message = _compile_exception_reply(error)
            self.assertIs(status, _WorkerStatus.UNAVAILABLE)
            self.assertTrue(retire_after_reply)
            self.assertTrue(message)

    def test_deterministic_error_is_invalid_and_keeps_worker(self) -> None:
        for error in (
            ValueError("invalid json schema"),
            RuntimeError("grammar parser error at byte 4"),
        ):
            self.assertFalse(_is_resource_exhaustion(error))
            status, retire_after_reply, message = _compile_exception_reply(error)
            self.assertIs(status, _WorkerStatus.INVALID)
            self.assertFalse(retire_after_reply)
            self.assertTrue(message)

    def test_generic_runtime_error_is_unavailable_and_retires_worker(self) -> None:
        status, retire_after_reply, message = _compile_exception_reply(
            RuntimeError("thread creation failed")
        )
        self.assertIs(status, _WorkerStatus.UNAVAILABLE)
        self.assertTrue(retire_after_reply)
        self.assertEqual(message, "thread creation failed")

    def test_exit_code_format_includes_signal_name(self) -> None:
        self.assertEqual(
            _format_worker_exitcode(-signal.SIGSEGV),
            "terminated by SIGSEGV (signal 11)",
        )
        self.assertEqual(_format_worker_exitcode(7), "exited with code 7")
        self.assertEqual(_format_worker_exitcode(None), "exit status unavailable")


class GrammarValidatorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.validator = GrammarValidator.__new__(GrammarValidator)
        self.validator._check_grammar = MagicMock(return_value=True)

    def test_json_object_allows_object_or_array(self) -> None:
        self.assertTrue(
            self.validator.validate_response_format({"type": "json_object"})
        )
        self.validator._check_grammar.assert_called_once_with(
            "json",
            {"anyOf": [{"type": "object"}, {"type": "array"}]},
        )

    def test_json_schema_preserves_request_schema(self) -> None:
        schema = {"type": "array", "items": {"type": "string"}}
        response_format = {
            "type": "json_schema",
            "json_schema": {"schema": schema},
        }

        self.assertTrue(self.validator.validate_response_format(response_format))
        self.validator._check_grammar.assert_called_once_with("json", schema)

    def test_worker_crash_is_transient_and_reports_signal_and_trace(self) -> None:
        self.validator._queue_timeout_s = 1.0
        self.validator._compile_timeout_s = 1.0
        self.validator._idle = queue.Queue()
        self.validator._ensure_pool = MagicMock()
        self.validator._retire = MagicMock()

        process = MagicMock()
        process.is_alive.return_value = True
        process.exitcode = -signal.SIGSEGV
        connection = MagicMock()
        connection.poll.return_value = True
        connection.recv.side_effect = EOFError
        fault_trace = io.BytesIO(b"xgrammar native stack trace")
        self.validator._idle.put((process, connection, fault_trace))

        with self.assertLogs(
            "rtp_llm.dash_sc.inference.grammar_validator", level="WARNING"
        ) as logs:
            with self.assertRaises(GrammarCheckUnavailable) as context:
                self.validator._compile_in_worker("ebnf", 'root ::= "x"')

        self.assertIn("SIGSEGV (signal 11)", str(context.exception))
        crash_logs = "\n".join(logs.output)
        self.assertIn("worker fatal traceback", crash_logs)
        self.assertIn("xgrammar native stack trace", crash_logs)

    def test_only_worker_verdicts_enter_result_cache(self) -> None:
        self.validator._result_cache_max_entries = 4
        self.validator._result_cache_lock = threading.Lock()
        self.validator._result_cache = OrderedDict()
        self.validator._inflight_lock = threading.Lock()
        self.validator._inflight = {}
        self.validator._check_grammar_uncached = MagicMock(
            side_effect=[
                _GrammarCheckResult(False),
                _GrammarCheckResult(False),
                _GrammarCheckResult(False, "invalid grammar", cacheable=True),
            ]
        )

        key = ("ebnf", 'root ::= "x"')
        self.validator._check_grammar_singleflight(*key)
        self.validator._check_grammar_singleflight(*key)
        self.assertEqual(self.validator._check_grammar_uncached.call_count, 2)

        deterministic_key = ("ebnf", 'root ::= "y"')
        first = self.validator._check_grammar_singleflight(*deterministic_key)
        second = self.validator._check_grammar_singleflight(*deterministic_key)
        self.assertEqual(first, second)
        self.assertEqual(self.validator._check_grammar_uncached.call_count, 3)


if __name__ == "__main__":
    unittest.main()
