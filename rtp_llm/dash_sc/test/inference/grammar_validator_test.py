from __future__ import annotations

import copy
import io
import json
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
    GrammarCompilationError,
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
        self.validator._check_grammar = MagicMock(return_value=(True, None))
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

    def _use_real_grammar_check(self) -> None:
        del self.validator._check_grammar
        self.validator._result_cache_max_entries = 8
        self.validator._result_cache_lock = threading.Lock()
        self.validator._result_cache = OrderedDict()
        self.validator._inflight_lock = threading.Lock()
        self.validator._inflight = {}
        self.validator._compile_in_worker = MagicMock(return_value=True)

    def test_normalization_preserves_schema_literal_values(self) -> None:
        literal = {"type": "string", "minLength": 1, "maxLength": 2}
        annotations = {
            "const": literal,
            "enum": [literal],
            "default": literal,
            "examples": [literal],
        }
        for with_schema in (False, True):
            with self.subTest(with_schema=with_schema):
                schema = copy.deepcopy(annotations)
                expected = copy.deepcopy(annotations)
                if with_schema:
                    schema["properties"] = {
                        "text": {"type": "string", "minLength": 129}
                    }
                    expected["properties"] = {"text": {"type": "string"}}
                original = copy.deepcopy(schema)
                normalized, error = GrammarValidator.normalize_json_for_xgrammar(schema)
                self.assertIsNone(error)
                self.assertEqual(schema, original)
                if with_schema:
                    self.assertEqual(json.loads(normalized), expected)
                else:
                    self.assertIsNone(normalized)

    def test_normalization_visits_only_schema_keyword_children(self) -> None:
        string_schema = {"type": ["string", "null"], "minLength": 129, "maxLength": 256}
        normalized_schema = {"type": ["string", "null"]}
        for keyword in (
            "properties",
            "patternProperties",
            "definitions",
            "$defs",
            "dependencies",
            "dependentSchemas",
            "items",
            "additionalItems",
            "contains",
            "additionalProperties",
            "propertyNames",
            "not",
            "if",
            "then",
            "else",
            "allOf",
            "anyOf",
            "oneOf",
            "prefixItems",
            "unevaluatedProperties",
            "unevaluatedItems",
        ):
            with self.subTest(keyword=keyword):
                child = copy.deepcopy(string_schema)
                expected_child = copy.deepcopy(normalized_schema)
                if keyword in (
                    "properties",
                    "patternProperties",
                    "definitions",
                    "$defs",
                    "dependencies",
                    "dependentSchemas",
                ):
                    child, expected_child = {"text": child}, {"text": expected_child}
                elif keyword in ("allOf", "anyOf", "oneOf", "prefixItems"):
                    child, expected_child = [child], [expected_child]
                normalized, error = GrammarValidator.normalize_json_for_xgrammar(
                    {keyword: child}
                )
                self.assertIsNone(error)
                self.assertEqual(json.loads(normalized), {keyword: expected_child})

        schema = {"items": [string_schema, False], "dependencies": {"text": ["other"]}}
        normalized, error = GrammarValidator.normalize_json_for_xgrammar(schema)
        self.assertIsNone(error)
        self.assertEqual(
            json.loads(normalized),
            {
                "items": [normalized_schema, False],
                "dependencies": {"text": ["other"]},
            },
        )

    def test_newer_schema_keywords_do_not_hide_invalid_length_bounds(self) -> None:
        self._use_real_grammar_check()
        for keyword in ("$defs", "prefixItems"):
            for bound in ("minLength", "maxLength"):
                for value in ("bad", True, -1, 1.5):
                    with self.subTest(keyword=keyword, bound=bound, value=value):
                        child = {"type": "string", bound: value}
                        schema = {
                            keyword: {"text": child} if keyword == "$defs" else [child]
                        }
                        tag = {
                            "type": "structural_tag",
                            "format": {
                                "type": "json_schema",
                                "json_schema": schema,
                            },
                        }
                        for normalize, validate, payload in (
                            (
                                GrammarValidator.normalize_json_for_xgrammar,
                                self.validator.validate_and_norm_json,
                                schema,
                            ),
                            (
                                GrammarValidator.normalize_structural_tag_for_xgrammar,
                                self.validator.validate_and_norm_structural_tag,
                                tag,
                            ),
                        ):
                            original = copy.deepcopy(payload)
                            self.assertEqual(
                                normalize(payload),
                                (None, "invalid Draft 7 JSON schema"),
                            )
                            self.assertEqual(validate(payload), (False, None))
                            self.assertEqual(payload, original)
        self.validator._compile_in_worker.assert_not_called()

    def test_normalizers_preserve_dict_inputs_and_repeat_results(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 128,
                }
            },
        }
        tag = {
            "type": "structural_tag",
            "format": {
                "type": "tag",
                "begin": "<tool>",
                "end": "</tool>",
                "content": {"type": "json_schema", "json_schema": schema},
                "maxLength": 5,
            },
        }
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "schema": schema,
                "strict": True,
            },
        }
        cases = (
            (GrammarValidator.normalize_json_for_xgrammar, schema),
            (GrammarValidator.normalize_structural_tag_for_xgrammar, tag),
            (GrammarValidator.normalize_response_format_for_xgrammar, response_format),
        )
        for normalize, payload in cases:
            with self.subTest(normalizer=normalize.__name__):
                original = copy.deepcopy(payload)
                first = normalize(payload)
                self.assertIsNotNone(first[0])
                self.assertIsNone(first[1])
                self.assertEqual(payload, original)
                self.assertEqual(normalize(payload), first)
                self.assertEqual(normalize(json.dumps(payload)), first)
                self.assertEqual(normalize(first[0]), (None, None))
        normalized_tag = json.loads(cases[1][0](tag)[0])
        self.assertEqual(normalized_tag["format"]["maxLength"], 5)
        self.assertEqual(
            normalized_tag["format"]["content"]["json_schema"]["properties"]["text"],
            {"type": "string"},
        )

    def test_response_format_preserves_bare_and_wrapped_json_schemas(self) -> None:
        self._use_real_grammar_check()
        for wrapped in (False, True):
            for has_bounds in (False, True):
                with self.subTest(wrapped=wrapped, has_bounds=has_bounds):
                    schema = {"type": "string"}
                    if has_bounds:
                        schema["minLength"] = 129
                    inner = (
                        {"name": "answer", "schema": schema, "strict": True}
                        if wrapped
                        else schema
                    )
                    payload = {"type": "json_schema", "json_schema": inner}
                    original = copy.deepcopy(payload)
                    ok, normalized = self.validator.validate_and_norm_response_format(
                        payload
                    )
                    self.assertTrue(ok)
                    self.assertEqual(payload, original)
                    self.assertEqual(
                        self.validator.validate_and_norm_response_format(payload),
                        (ok, normalized),
                    )
                    self.assertEqual(
                        GrammarValidator.normalize_response_format_for_xgrammar(
                            payload
                        ),
                        (normalized, None),
                    )
                    if has_bounds:
                        expected_inner = (
                            {
                                "name": "answer",
                                "schema": {"type": "string"},
                                "strict": True,
                            }
                            if wrapped
                            else {"type": "string"}
                        )
                        self.assertEqual(
                            json.loads(normalized),
                            {"type": "json_schema", "json_schema": expected_inner},
                        )
                    else:
                        self.assertIsNone(normalized)
                    self.assertIs(
                        self.validator.validate_response_format(payload), True
                    )

    def test_structural_tag_response_format_retains_outer_envelope(self) -> None:
        self._use_real_grammar_check()
        tag = {
            "type": "structural_tag",
            "format": {
                "type": "json_schema",
                "json_schema": {"type": "string", "minLength": 129},
            },
        }
        payload = {"type": "structural_tag", "structural_tag": tag}
        original = copy.deepcopy(payload)
        ok, normalized = self.validator.validate_and_norm_response_format(payload)
        self.assertTrue(ok)
        self.assertEqual(payload, original)
        self.assertEqual(
            self.validator.validate_and_norm_response_format(payload), (ok, normalized)
        )
        expected_tag = {
            "type": "structural_tag",
            "format": {
                "type": "json_schema",
                "json_schema": {"type": "string"},
            },
        }
        self.assertEqual(
            json.loads(normalized),
            {"type": "structural_tag", "structural_tag": expected_tag},
        )
        self.validator._compile_in_worker.assert_called_once_with(
            "structural_tag", json.dumps(expected_tag)
        )

    def test_structural_tag_validates_original_bounds_before_normalizing(self) -> None:
        self._use_real_grammar_check()
        for legacy in (False, True):
            for bound in ("minLength", "maxLength"):
                for value in (-1, 1.5, True, "1", 129):
                    with self.subTest(legacy=legacy, bound=bound, value=value):
                        schema = {"type": "string", bound: value}
                        if legacy:
                            tag = {
                                "structures": [
                                    {
                                        "begin": "<tool>",
                                        "end": "</tool>",
                                        "schema": schema,
                                    }
                                ],
                                "triggers": ["<tool>"],
                            }
                        else:
                            tag = {
                                "type": "structural_tag",
                                "format": {
                                    "type": "json_schema",
                                    "json_schema": schema,
                                },
                            }
                        original = copy.deepcopy(tag)
                        self.validator._compile_in_worker.reset_mock()
                        normalized, error = (
                            GrammarValidator.normalize_structural_tag_for_xgrammar(tag)
                        )
                        ok, admitted = self.validator.validate_and_norm_structural_tag(
                            tag
                        )
                        self.assertEqual(tag, original)
                        if value == 129:
                            self.assertIsNone(error)
                            self.assertTrue(ok)
                            self.assertEqual(admitted, normalized)
                            self.assertNotIn(
                                bound,
                                (
                                    json.loads(normalized)["structures"][0]["schema"]
                                    if legacy
                                    else json.loads(normalized)["format"]["json_schema"]
                                ),
                            )
                            self.validator._compile_in_worker.assert_called_once_with(
                                "structural_tag", normalized
                            )
                        else:
                            self.assertEqual(error, "invalid Draft 7 JSON schema")
                            self.assertIsNone(normalized)
                            self.assertEqual((ok, admitted), (False, None))
                            self.validator._compile_in_worker.assert_not_called()

    def test_json_schema_rejects_invalid_bound_before_normalizing(self) -> None:
        self._use_real_grammar_check()
        for value in (-1, 1.5):
            with self.subTest(value=value):
                schema = {"type": "string", "minLength": value}
                payload = {"type": "json_schema", "json_schema": {"schema": schema}}
                self.assertEqual(
                    GrammarValidator.normalize_response_format_for_xgrammar(payload),
                    (None, "invalid Draft 7 JSON schema"),
                )
                self.assertEqual(
                    self.validator.validate_and_norm_response_format(payload),
                    (False, None),
                )
        self.validator._compile_in_worker.assert_not_called()

    def test_normalized_success_is_cached_by_original_schema(self) -> None:
        self._use_real_grammar_check()
        schema = json.dumps({"type": "string", "minLength": 129})
        first = self.validator.validate_and_norm_json(schema)
        second = self.validator.validate_and_norm_json(schema)
        self.assertEqual(first, second)
        self.assertTrue(first[0])
        self.assertEqual(json.loads(first[1]), {"type": "string"})
        self.validator._compile_in_worker.assert_called_once_with(
            "json", json.dumps({"type": "string"})
        )
        cached = self.validator._get_cached_result(("json", schema))
        self.assertTrue(cached.cacheable)
        self.assertEqual(cached.normalized_spec, first[1])

    def test_normalized_rejection_is_cached_with_original_diagnostic(self) -> None:
        self._use_real_grammar_check()
        self.validator._compile_in_worker.side_effect = GrammarCompilationError(
            "invalid grammar: test rejection"
        )
        schema = json.dumps({"type": "string", "minLength": 129})
        for _ in range(2):
            with self.assertRaisesRegex(
                GrammarCompilationError, "invalid grammar: test rejection"
            ):
                self.validator.validate_and_norm_json(schema)
        self.validator._compile_in_worker.assert_called_once_with(
            "json", json.dumps({"type": "string"})
        )
        cached = self.validator._get_cached_result(("json", schema))
        self.assertFalse(cached.ok)
        self.assertTrue(cached.cacheable)
        self.assertEqual(json.loads(cached.normalized_spec), {"type": "string"})

    def test_normalization_does_not_cache_transient_failures(self) -> None:
        self._use_real_grammar_check()
        schema = json.dumps({"type": "string", "minLength": 129})
        for error_type in (
            GrammarCheckOverloaded,
            GrammarCheckTimeout,
            GrammarCheckUnavailable,
        ):
            with self.subTest(error_type=error_type):
                self.validator._compile_in_worker.reset_mock()
                self.validator._compile_in_worker.side_effect = error_type(
                    "temporary failure"
                )
                for _ in range(2):
                    with self.assertRaises(error_type):
                        self.validator.validate_and_norm_json(schema)
                self.assertEqual(self.validator._compile_in_worker.call_count, 2)
                self.assertIsNone(self.validator._get_cached_result(("json", schema)))

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
