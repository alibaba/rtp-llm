from __future__ import annotations

import asyncio
import io
import json
import queue
import signal
import threading
import unittest
from collections import OrderedDict
from unittest.mock import MagicMock

from rtp_llm.dash_sc.codec import SamplingParams
from rtp_llm.dash_sc.inference.grammar_validator import (
    GrammarCompilationError,
    GrammarValidator,
    _compile_exception_reply,
    _WorkerStatus,
)
from rtp_llm.dash_sc.inference.servicer import DashScInferenceServicer


class GrammarValidatorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.validator = GrammarValidator.__new__(GrammarValidator)
        self.validator._check_grammar = MagicMock(return_value=(True, None))

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

    def test_structural_tag_normalization_strips_only_string_length_bounds(
        self,
    ) -> None:
        structural_tag = {
            "format": {
                "type": "tag",
                "begin": "<tool>",
                "content": {
                    "type": "json_schema",
                    "json_schema": {
                        "type": "object",
                        "properties": {
                            "reason": {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": 128,
                            },
                            "count": {"type": "integer", "minimum": 0},
                        },
                        "required": ["reason"],
                    },
                },
                "end": "</tool>",
                # This belongs to the structural-tag DSL, not its JSON Schema.
                "maxLength": 5,
            }
        }

        normalized, error = GrammarValidator.normalize_structural_tag_for_xgrammar(
            json.dumps(structural_tag)
        )

        self.assertIsNone(error)
        self.assertIsNotNone(normalized)
        parsed = json.loads(normalized)
        self.assertEqual(
            parsed["format"]["content"]["json_schema"]["properties"]["reason"],
            {"type": "string"},
        )
        self.assertEqual(parsed["format"]["maxLength"], 5)

    def test_normalized_structural_tag_is_compiled_and_forwarded(self) -> None:
        structural_tag = json.dumps(
            {
                "format": {
                    "type": "json_schema",
                    "json_schema": {
                        "type": "object",
                        "properties": {"reason": {"type": "string", "minLength": 1}},
                    },
                }
            }
        )

        class _Validator:
            compiled = ""

            def validate_and_norm_structural_tag(self, payload, request_id=""):
                normalized, error = (
                    GrammarValidator.normalize_structural_tag_for_xgrammar(payload)
                )
                self.compiled = normalized or payload
                return error is None, normalized

        validator = _Validator()
        servicer = DashScInferenceServicer.__new__(DashScInferenceServicer)
        servicer._grammar_validator = validator

        error, normalized_sampling = asyncio.run(
            servicer._validate_request_grammar(
                SamplingParams(structural_tag=structural_tag), "req-normalize"
            )
        )

        self.assertIsNone(error)
        self.assertEqual(validator.compiled, normalized_sampling.structural_tag)
        self.assertNotIn("minLength", validator.compiled)

    def test_response_format_normalization_matches_dashllm(self) -> None:
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "schema": {
                    "type": "object",
                    "properties": {
                        "reason": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 128,
                        }
                    },
                }
            },
        }

        normalized, error = GrammarValidator.normalize_response_format_for_xgrammar(
            response_format
        )

        self.assertIsNone(error)
        parsed = json.loads(normalized)
        self.assertEqual(
            parsed["json_schema"]["schema"]["properties"]["reason"],
            {"type": "string"},
        )

    def test_response_format_rejects_invalid_length_before_normalization(self) -> None:
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "schema": {"type": "string", "minLength": -1},
            },
        }

        normalized, error = GrammarValidator.normalize_response_format_for_xgrammar(
            response_format
        )

        self.assertIsNone(normalized)
        self.assertEqual(error, "invalid Draft 7 JSON schema")

    def test_normalization_result_is_cached_by_original_schema(self) -> None:
        validator = GrammarValidator.__new__(GrammarValidator)
        validator._result_cache_max_entries = 8
        validator._result_cache_lock = threading.Lock()
        validator._result_cache = OrderedDict()
        validator._inflight_lock = threading.Lock()
        validator._inflight = {}
        validator._compile_in_worker = MagicMock(return_value=True)
        schema = json.dumps({"type": "string", "minLength": 1})

        first = validator.validate_and_norm_json(schema)
        second = validator.validate_and_norm_json(schema)

        self.assertEqual(first, second)
        self.assertTrue(first[0])
        self.assertEqual(json.loads(first[1]), {"type": "string"})
        validator._compile_in_worker.assert_called_once_with(
            "json", json.dumps({"type": "string"})
        )

    def test_allocation_failure_retires_worker_without_relabeling(self) -> None:
        status, retire_worker, message = _compile_exception_reply(
            MemoryError("std::bad_alloc")
        )

        self.assertIs(status, _WorkerStatus.INVALID)
        self.assertTrue(retire_worker)
        self.assertEqual(message, "std::bad_alloc")

    def test_reproducible_worker_crash_returns_signal_details(self) -> None:
        self.validator._queue_timeout_s = 1.0
        self.validator._compile_timeout_s = 1.0
        self.validator._idle = queue.Queue()
        self.validator._ensure_pool = MagicMock()
        self.validator._retire = MagicMock()

        for attempt in range(2):
            process = MagicMock()
            process.is_alive.return_value = True
            process.exitcode = -signal.SIGSEGV
            connection = MagicMock()
            connection.poll.return_value = True
            connection.recv.side_effect = EOFError
            fault_trace = io.BytesIO(
                f"xgrammar native stack trace\ncompile attempt {attempt + 1}".encode()
            )
            self.validator._idle.put((process, connection, fault_trace))

        with self.assertLogs(
            "rtp_llm.dash_sc.inference.grammar_validator", level="WARNING"
        ) as logs:
            with self.assertRaises(GrammarCompilationError) as context:
                self.validator._compile_in_worker("ebnf", 'root ::= "x"')

        message = str(context.exception)
        self.assertIn(
            "xgrammar sandbox workers crashed while compiling the grammar", message
        )
        self.assertEqual(message.count("SIGSEGV (signal 11)"), 2)
        crash_logs = "\n".join(logs.output)
        self.assertEqual(crash_logs.count("worker fatal traceback"), 2)
        self.assertIn("compile attempt 1", crash_logs)
        self.assertIn("compile attempt 2", crash_logs)


if __name__ == "__main__":
    unittest.main()
