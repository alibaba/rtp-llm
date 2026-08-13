from __future__ import annotations

import json
import os
import queue
import unittest
from unittest.mock import MagicMock, call, patch

import xgrammar as xgr

from rtp_llm.config.grammar_constraint import GrammarConstraint
from rtp_llm.config.py_config_modules import GrammarAdmissionConfig
from rtp_llm.dash_sc.inference.grammar_validator import (
    GrammarCompilationError,
    GrammarValidator,
    _compile_exception_reply,
    _WorkerStatus,
)
from rtp_llm.ops import GrammarConfig


class GrammarValidatorResponseFormatTest(unittest.TestCase):
    def setUp(self) -> None:
        # Fast shape/routing tests stop at the sandbox boundary. The integration test
        # below covers that boundary with a real spawned xgrammar worker.
        self.validator = GrammarValidator.__new__(GrammarValidator)
        self.validator._compile_in_worker = MagicMock(return_value=True)
        self.validator._check_grammar_cached.cache_clear()

    def tearDown(self) -> None:
        self.validator._check_grammar_cached.cache_clear()

    def test_json_schema_envelope_compiles_inner_schema(self) -> None:
        schema = {"type": "object", "properties": {"answer": {"type": "string"}}}

        self.assertTrue(
            self.validator.validate_response_format(
                {"type": "json_schema", "json_schema": {"schema": schema}},
                request_id="req-json",
            )
        )

        compiled_schema = self.validator._compile_in_worker.call_args.args[1]
        self.assertEqual(json.loads(compiled_schema), schema)

    def test_empty_and_boolean_json_schemas_reach_compiler(self) -> None:
        schemas = ({}, True, False)

        for schema in schemas:
            with self.subTest(schema=schema):
                self.assertTrue(
                    self.validator.validate_constraint(
                        GrammarConstraint("json_schema", schema)
                    )
                )

        compiled_schemas = [
            json.loads(mock_call.args[1])
            for mock_call in self.validator._compile_in_worker.call_args_list
        ]
        self.assertEqual(compiled_schemas, list(schemas))

    def test_regex_and_ebnf_use_matching_compiler_entry_points(self) -> None:
        self.assertTrue(
            self.validator.validate_response_format(
                {"type": "regex", "pattern": "[0-9]+"}
            )
        )
        self.assertTrue(
            self.validator.validate_response_format(
                {"type": "ebnf", "grammar": 'root ::= "yes" | "no"'}
            )
        )

        self.validator._compile_in_worker.assert_has_calls(
            [
                call("regex", "[0-9]+"),
                call("ebnf", 'root ::= "yes" | "no"'),
            ]
        )

    def test_structural_tag_envelope_compiles_payload(self) -> None:
        structural_tag = {"format": {"type": "any_text"}}

        self.assertTrue(
            self.validator.validate_response_format(
                {"type": "structural_tag", "structural_tag": structural_tag}
            )
        )

        compiled_tag = self.validator._compile_in_worker.call_args.args[1]
        self.assertEqual(json.loads(compiled_tag), structural_tag)

    def test_xgrammar_compile_error_is_preserved_on_cache_hit(self) -> None:
        error_message = "failed to compile grammar: Cannot find field defs/filter"
        self.validator._compile_in_worker.side_effect = GrammarCompilationError(
            error_message
        )

        for _ in range(2):
            with self.assertRaisesRegex(
                GrammarCompilationError, "Cannot find field defs/filter"
            ):
                self.validator.validate_structural_tag(
                    {"type": "structural_tag", "format": {"type": "any_text"}}
                )

        self.validator._compile_in_worker.assert_called_once()

    def test_missing_grammar_payload_is_rejected_without_compile(self) -> None:
        for response_format in (
            {"type": "json_schema"},
            {"type": "regex"},
            {"type": "ebnf", "grammar": ""},
            {"type": "unknown"},
        ):
            with self.subTest(response_format=response_format):
                self.assertFalse(
                    self.validator.validate_response_format(response_format)
                )

        self.validator._compile_in_worker.assert_not_called()

    def test_oversized_nested_min_length_is_rejected_without_compile(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "answer": {
                    "allOf": [
                        {"type": "string", "minLength": 129},
                    ]
                }
            },
        }

        self.assertFalse(self.validator.validate_json(schema))

        self.validator._compile_in_worker.assert_not_called()

    def test_min_length_at_limit_is_compiled(self) -> None:
        schema = {"type": "string", "minLength": 128}

        self.assertTrue(self.validator.validate_json(schema))

        self.validator._compile_in_worker.assert_called_once()

    def test_structural_tag_with_oversized_min_length_is_rejected_without_compile(
        self,
    ) -> None:
        structural_tag = {
            "format": {
                "type": "json_schema",
                "json_schema": {"type": "string", "minLength": 129},
            }
        }

        self.assertFalse(self.validator.validate_structural_tag(structural_tag))

        self.validator._compile_in_worker.assert_not_called()


class GrammarValidatorBackendTest(unittest.TestCase):
    def test_worker_build_backend_deserializes_python_tokenizer_info(self) -> None:
        xgrammar = MagicMock()
        tokenizer_info = object()
        backend = object()
        xgrammar.TokenizerInfo.deserialize_json.return_value = tokenizer_info
        xgrammar.GrammarCompiler.return_value = backend
        validator = GrammarValidator.__new__(GrammarValidator)
        validator._tokenizer_info_json = "tokenizer-info"
        validator._compile_threads = 4
        validator._cache_limit_bytes = 1024

        with patch.object(GrammarValidator, "_xgrammar", return_value=xgrammar):
            result = validator._build_backend()

        self.assertIs(result, backend)
        xgrammar.TokenizerInfo.deserialize_json.assert_called_once_with(
            "tokenizer-info"
        )
        xgrammar.GrammarCompiler.assert_called_once_with(
            tokenizer_info,
            max_threads=4,
            cache_enabled=True,
            cache_limit_bytes=1024,
        )


class GrammarValidatorCompileExceptionTest(unittest.TestCase):
    def test_all_catchable_xgrammar_errors_are_input_rejections(self) -> None:
        errors = (
            xgr.InvalidJSONError("invalid JSON"),
            xgr.InvalidStructuralTagError("invalid structural tag"),
            xgr.DeserializeFormatError("invalid serialization format"),
            xgr.DeserializeVersionError("unsupported serialization version"),
            TypeError("invalid compiler arguments"),
            ValueError("invalid grammar value"),
            RuntimeError("invalid JSON schema"),
            json.JSONDecodeError("invalid JSON", "{", 1),
        )

        for error in errors:
            with self.subTest(error_type=type(error).__name__):
                status, retire_worker, message = _compile_exception_reply(error)

                self.assertIs(status, _WorkerStatus.INVALID)
                self.assertFalse(retire_worker)
                self.assertEqual(message, str(error))

    def test_memory_error_is_rejected_and_worker_is_retired(self) -> None:
        error = MemoryError("out of memory")

        status, retire_worker, message = _compile_exception_reply(error)

        self.assertIs(status, _WorkerStatus.INVALID)
        self.assertTrue(retire_worker)
        self.assertEqual(message, str(error))

    def test_user_controlled_bad_alloc_text_does_not_retire_worker(self) -> None:
        error = RuntimeError("Cannot find field bad_alloc")

        status, retire_worker, message = _compile_exception_reply(error)

        self.assertIs(status, _WorkerStatus.INVALID)
        self.assertFalse(retire_worker)
        self.assertEqual(message, str(error))

    def test_compile_error_message_is_bounded(self) -> None:
        status, retire_worker, message = _compile_exception_reply(
            ValueError("x" * 2048)
        )

        self.assertIs(status, _WorkerStatus.INVALID)
        self.assertFalse(retire_worker)
        self.assertEqual(len(message), 1024)


class GrammarValidatorSandboxTest(unittest.TestCase):
    def test_real_xgrammar_compiles_in_spawned_worker(self) -> None:
        tokenizer_info = xgr.TokenizerInfo(
            [chr(token_id) for token_id in range(128)],
            xgr.VocabType.RAW,
            vocab_size=128,
            stop_token_ids=[0],
        )
        grammar_config = GrammarConfig()
        grammar_config.tokenizer_info_json = tokenizer_info.serialize_json()
        grammar_config.num_workers = 1
        grammar_config.compiler_cache_bytes = 1024
        admission_config = GrammarAdmissionConfig(
            queue_timeout_s=10.0,
            compile_timeout_s=10.0,
            sandbox_pool_size=1,
            sandbox_process_memory_limit_mb=0,
        )
        validator = GrammarValidator(grammar_config, admission_config)

        try:
            self.assertTrue(
                validator.validate_json(
                    {
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                    }
                )
            )
            proc, conn = validator._idle.get(timeout=1.0)
            self.assertTrue(proc.is_alive())
            self.assertNotEqual(proc.pid, os.getpid())
            validator._idle.put((proc, conn))
        finally:
            # This validator is test-local; stop its daemon worker rather than leaving
            # multiprocessing's interpreter-exit hook to reap it.
            validator._pool_target = 0
            while True:
                try:
                    proc, conn = validator._idle.get_nowait()
                except queue.Empty:
                    break
                conn.close()
                if proc.is_alive():
                    proc.terminate()
                proc.join(timeout=1.0)


if __name__ == "__main__":
    unittest.main()
