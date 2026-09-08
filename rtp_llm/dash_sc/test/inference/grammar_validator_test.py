from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from rtp_llm.dash_sc.inference.grammar_validator import (
    GrammarValidator,
    _compile_exception_reply,
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
        error = ValueError("invalid json schema")
        self.assertFalse(_is_resource_exhaustion(error))
        status, retire_after_reply, message = _compile_exception_reply(error)
        self.assertIs(status, _WorkerStatus.INVALID)
        self.assertFalse(retire_after_reply)
        self.assertEqual(message, "invalid json schema")


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


if __name__ == "__main__":
    unittest.main()
