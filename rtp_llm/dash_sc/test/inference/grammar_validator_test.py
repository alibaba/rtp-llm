from __future__ import annotations

import io
import multiprocessing
import os
import queue
import signal
import tempfile
import threading
import unittest
from collections import OrderedDict
from unittest.mock import MagicMock, patch

from rtp_llm.dash_sc.inference import grammar_validator as grammar_validator_module
from rtp_llm.dash_sc.inference.core_dump_control import (
    _XGRAMMAR_SANDBOX_CORE_DUMP_ENV,
    _configure_xgrammar_sandbox_core_dump_for_current_process,
)
from rtp_llm.dash_sc.inference.grammar_validator import (
    GrammarCheckOverloaded,
    GrammarCheckTimeout,
    GrammarCheckUnavailable,
    GrammarValidator,
    _compile_exception_reply,
    _CRASH_CIRCUIT_MAX_ENTRIES,
    _DeferredDupFd,
    _enable_worker_faulthandler,
    _format_worker_exitcode,
    _GrammarCheckResult,
    _is_resource_exhaustion,
    _WorkerStatus,
)


def _crash_with_faulthandler(fault_trace_fd) -> None:
    os.environ[_XGRAMMAR_SANDBOX_CORE_DUMP_ENV] = "0"
    _configure_xgrammar_sandbox_core_dump_for_current_process()
    with os.fdopen(fault_trace_fd.detach(), "wb", buffering=0) as fault_file:
        os.dup2(fault_file.fileno(), 2)
        _enable_worker_faulthandler(fault_file)
        os.kill(os.getpid(), signal.SIGSEGV)


class CompileExceptionReplyTest(unittest.TestCase):
    def test_resource_exhaustion_is_overloaded_and_retires_worker(self) -> None:
        for error in (MemoryError("out of memory"), RuntimeError("std::bad_alloc")):
            self.assertTrue(_is_resource_exhaustion(error))
            status, retire_after_reply, message = _compile_exception_reply(error)
            self.assertIs(status, _WorkerStatus.OVERLOADED)
            self.assertTrue(retire_after_reply)
            self.assertTrue(message)

    def test_typed_deterministic_error_is_invalid_and_keeps_worker(self) -> None:
        error = ValueError("invalid json schema")
        self.assertFalse(_is_resource_exhaustion(error))
        status, retire_after_reply, message = _compile_exception_reply(error)
        self.assertIs(status, _WorkerStatus.INVALID)
        self.assertFalse(retire_after_reply)
        self.assertTrue(message)

    def test_cpp_runtime_error_markers_are_invalid_and_keep_worker(self) -> None:
        messages = (
            "invalid json document",
            "json parse failed",
            "json parsing error at byte 4",
            "failed to parse json schema",
            "invalid regex pattern",
            "regex parse failed",
            "regex parsing error at byte 4",
            "failed to parse regex",
            "invalid ebnf grammar",
            "ebnf parse failed",
            "ebnf parsing error at byte 4",
            "ebnf lexer error at byte 4",
            "failed to parse ebnf",
            "invalid grammar",
            "grammar parse failed",
            "grammar parsing error at byte 4",
            "grammar lexer error at byte 4",
            "invalid structural tag",
            "structural tag parse failed",
            "structural tag parsing error at byte 4",
            "structural tag syntax error at byte 4",
            "grammar parser error at byte 4",
            "regex lexer error at byte 4",
        )
        for message in messages:
            with self.subTest(message=message):
                status, retire_after_reply, reply = _compile_exception_reply(
                    RuntimeError(message)
                )
                self.assertIs(status, _WorkerStatus.INVALID)
                self.assertFalse(retire_after_reply)
                self.assertEqual(reply, message)

    def test_generic_runtime_error_is_unavailable_and_retires_worker(self) -> None:
        messages = (
            "thread creation failed",
            "unexpected token: syntax error",
            "parser error",
            "token lexer error at byte 4",
        )
        for message in messages:
            with self.subTest(message=message):
                status, retire_after_reply, reply = _compile_exception_reply(
                    RuntimeError(message)
                )
                self.assertIs(status, _WorkerStatus.UNAVAILABLE)
                self.assertTrue(retire_after_reply)
                self.assertEqual(reply, message)

    def test_exit_code_format_includes_signal_name(self) -> None:
        self.assertEqual(
            _format_worker_exitcode(-signal.SIGSEGV),
            "terminated by SIGSEGV (signal 11)",
        )
        self.assertEqual(_format_worker_exitcode(7), "exited with code 7")
        self.assertEqual(_format_worker_exitcode(None), "exit status unavailable")

    def test_fault_trace_dupfd_is_created_only_during_serialization(self) -> None:
        deferred_fd = _DeferredDupFd(17)
        transferred_fd = object()
        with patch.object(
            grammar_validator_module, "DupFd", return_value=transferred_fd
        ) as dup_fd:
            dup_fd.assert_not_called()
            rebuild, args = deferred_fd.__reduce__()

        dup_fd.assert_called_once_with(17)
        self.assertIs(rebuild(*args), transferred_fd)

    def test_spawned_fatal_signal_writes_real_python_traceback(self) -> None:
        context = multiprocessing.get_context("spawn")
        with tempfile.TemporaryFile(mode="w+b") as fault_file:
            process = context.Process(
                target=_crash_with_faulthandler,
                args=(_DeferredDupFd(fault_file.fileno()),),
            )
            process.start()
            try:
                process.join(timeout=10)
                self.assertFalse(process.is_alive())
                self.assertEqual(process.exitcode, -signal.SIGSEGV)
                fault_file.seek(0)
                fault_trace = fault_file.read().decode("utf-8", errors="replace")
            finally:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=10)

        self.assertIn("Fatal Python error: Segmentation fault", fault_trace)
        self.assertIn("_crash_with_faulthandler", fault_trace)


class GrammarValidatorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.validator = GrammarValidator.__new__(GrammarValidator)
        self.validator._check_grammar = MagicMock(return_value=True)
        self.validator._closed = False
        self.validator._crash_circuit_lock = threading.Lock()
        self.validator._crash_circuit = OrderedDict()
        self.validator._pool_lock = threading.Lock()
        self.validator._live = 1
        self.validator._last_spawn_error = ""

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
        self.assertIn(("ebnf", 'root ::= "x"'), self.validator._crash_circuit)

    def test_crash_circuit_is_keyed_bounded_and_expires(self) -> None:
        with patch.object(grammar_validator_module.time, "monotonic", return_value=100.0):
            for index in range(_CRASH_CIRCUIT_MAX_ENTRIES + 1):
                self.validator._record_native_crash(("ebnf", f"root ::= {index}"))

        self.assertEqual(len(self.validator._crash_circuit), _CRASH_CIRCUIT_MAX_ENTRIES)
        self.assertNotIn(("ebnf", "root ::= 0"), self.validator._crash_circuit)
        active_key = ("ebnf", "root ::= 1")
        with patch.object(grammar_validator_module.time, "monotonic", return_value=109.0):
            with self.assertRaises(GrammarCheckUnavailable):
                self.validator._raise_if_crash_circuit_open(active_key)
            self.validator._raise_if_crash_circuit_open(("ebnf", "different"))
        with patch.object(grammar_validator_module.time, "monotonic", return_value=110.0):
            self.validator._raise_if_crash_circuit_open(active_key)
        self.assertNotIn(active_key, self.validator._crash_circuit)

    def test_retryable_worker_outcomes_have_distinct_types(self) -> None:
        process = MagicMock()
        process.is_alive.return_value = True
        connection = MagicMock()
        connection.poll.return_value = True
        fault_file = io.BytesIO()
        self.validator._queue_timeout_s = 1.0
        self.validator._compile_timeout_s = 1.0
        self.validator._idle = queue.Queue()
        self.validator._ensure_pool = MagicMock()
        self.validator._retire = MagicMock()

        connection.recv.return_value = (
            _WorkerStatus.OVERLOADED,
            True,
            "out of memory",
        )
        self.validator._idle.put((process, connection, fault_file))
        with self.assertRaises(GrammarCheckOverloaded):
            self.validator._compile_in_worker("ebnf", 'root ::= "overload"')

        timeout_connection = MagicMock()
        timeout_connection.poll.return_value = False
        self.validator._idle.put((process, timeout_connection, io.BytesIO()))
        with self.assertRaises(GrammarCheckTimeout):
            self.validator._compile_in_worker("ebnf", 'root ::= "timeout"')

        self.validator._queue_timeout_s = 0.001
        with self.assertRaises(GrammarCheckOverloaded):
            self.validator._compile_in_worker("ebnf", 'root ::= "queue-full"')

        self.validator._live = 0
        self.validator._last_spawn_error = "cannot spawn"
        with self.assertRaises(GrammarCheckUnavailable) as unavailable:
            self.validator._compile_in_worker("ebnf", 'root ::= "pool-down"')
        self.assertNotIsInstance(unavailable.exception, GrammarCheckOverloaded)

    def test_spawn_failure_does_not_block_a_healthy_worker(self) -> None:
        self.validator._queue_timeout_s = 1.0
        self.validator._compile_timeout_s = 1.0
        self.validator._idle = queue.Queue()
        self.validator._ensure_pool = MagicMock()
        self.validator._last_spawn_error = "replacement failed"

        process = MagicMock()
        process.is_alive.return_value = True
        connection = MagicMock()
        connection.poll.return_value = True
        connection.recv.return_value = (_WorkerStatus.VALID, False, "")
        worker = (process, connection, io.BytesIO())
        self.validator._idle.put(worker)

        self.assertTrue(self.validator._compile_in_worker("ebnf", 'root ::= "ok"'))
        self.assertIs(self.validator._idle.get_nowait(), worker)

    def test_worker_startup_timeout_terminates_kills_and_joins(self) -> None:
        self.validator._mp = MagicMock()
        self.validator._worker_tokenizer_info_json = "{}"
        self.validator._worker_grammar_config = MagicMock()
        self.validator._worker_admission_config = MagicMock()
        self.validator._worker_memory_limit_bytes = 0
        self.validator._compile_timeout_s = 0
        self.validator._spawning = 1
        self.validator._replacement_failures = 0
        self.validator._replacement_not_before = 0.0
        self.validator._last_spawn_error = ""

        parent_conn = MagicMock()
        parent_conn.poll.return_value = False
        child_conn = MagicMock()
        process = MagicMock()
        process.is_alive.side_effect = (True, True)
        process.exitcode = None
        self.validator._mp.Pipe.return_value = (parent_conn, child_conn)
        self.validator._mp.Process.return_value = process

        with self.assertLogs(
            "rtp_llm.dash_sc.inference.grammar_validator", level="WARNING"
        ):
            self.assertFalse(self.validator._spawn_one())

        process.start.assert_called_once_with()
        process.terminate.assert_called_once_with()
        process.kill.assert_called_once_with()
        self.assertGreaterEqual(process.join.call_count, 2)
        parent_conn.close.assert_called_once_with()
        child_conn.close.assert_called_once_with()
        self.assertEqual(self.validator._spawning, 0)
        self.assertEqual(self.validator._replacement_failures, 1)
        self.assertGreater(self.validator._replacement_not_before, 0.0)
        self.assertIn("readiness handshake timed out", self.validator._last_spawn_error)

    def test_replacement_backoff_uses_one_timer(self) -> None:
        self.validator._pool_target = 2
        self.validator._live = 1
        self.validator._spawning = 0
        self.validator._coordinator_running = False
        self.validator._replacement_timer = None
        self.validator._replacement_not_before = 105.0

        timer = MagicMock()
        with patch.object(
            grammar_validator_module.time, "monotonic", return_value=100.0
        ), patch.object(
            grammar_validator_module.threading, "Timer", return_value=timer
        ) as timer_factory:
            self.validator._ensure_pool()
            self.validator._ensure_pool()

        timer_factory.assert_called_once_with(
            5.0, self.validator._replacement_timer_fired
        )
        timer.start.assert_called_once_with()
        self.assertEqual(self.validator._live, 1)
        self.assertEqual(self.validator._spawning, 0)

    def test_close_is_idempotent_and_stops_idle_workers(self) -> None:
        self.validator._close_lock = threading.Lock()
        self.validator._idle = queue.Queue()
        self.validator._pool_target = 1
        self.validator._spawning = 0
        self.validator._coordinator_running = False
        self.validator._coordinator_thread = None
        self.validator._compile_timeout_s = 1.0
        self.validator._replacement_timer = MagicMock()

        process = MagicMock()
        process.is_alive.return_value = False
        connection = MagicMock()
        fault_file = MagicMock()
        self.validator._idle.put((process, connection, fault_file))

        self.validator.close()
        self.validator.close()

        connection.close.assert_called_once_with()
        fault_file.close.assert_called_once_with()
        self.assertTrue(self.validator._closed)
        self.assertEqual(self.validator._pool_target, 0)
        self.assertEqual(self.validator._live, 0)

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
