from __future__ import annotations

import io
import os
import queue
import signal
import unittest
from unittest.mock import MagicMock, patch

from rtp_llm.config.py_config_modules import GrammarAdmissionConfig
from rtp_llm.dash_sc.inference.grammar_validator import (
    GrammarCompilationError,
    GrammarValidator,
    _compile_exception_reply,
    _WorkerStatus,
)
from rtp_llm.ops import GrammarConfig


class GrammarValidatorTest(unittest.TestCase):

    def test_template_leaves_grammar_workers_for_the_first_validation(self):
        def initialize(validator, *args):
            validator._compile_threads = 1
            validator._cache_limit_bytes = 0

        for phase, warm in (
            ("normal", True),
            ("checkpoint", False),
            ("restore", False),
        ):
            with self.subTest(phase=phase), patch.dict(
                os.environ, {"RTPLLM_ENABLE_SCR": "1", "SCR_PHASE": phase}
            ), patch.object(
                GrammarValidator, "_initialize_compiler", initialize
            ), patch.object(
                GrammarValidator, "_ensure_pool"
            ) as ensure:
                validator = GrammarValidator(
                    "{}", GrammarConfig(), GrammarAdmissionConfig()
                )
                self.assertEqual(ensure.call_count, int(warm))
                self.assertEqual(validator._live, 0)
                self.assertTrue(validator._idle.empty())
                # The real validation path initializes workers on demand.
                ensure.side_effect = RuntimeError(
                    "first validation reached worker startup"
                )
                with self.assertRaisesRegex(RuntimeError, "first validation"):
                    validator._compile_in_worker("ebnf", 'root ::= "x"')

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
