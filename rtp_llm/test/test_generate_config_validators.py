"""Prefill-only validation and atomic updates on the base GenerateConfig API."""

import unittest

from pydantic import ValidationError

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.config.generate_config import GenerateConfig


class TestPrefillOnlyGenerateConfig(unittest.TestCase):
    def test_capture_intent_is_independent_and_preserves_request_options(self):
        self.assertFalse(GenerateConfig().capture_hidden_states)
        for tokens in (0, 4):
            for reuse in (False, True):
                for capture in (False, True):
                    with self.subTest(tokens=tokens, reuse=reuse, capture=capture):
                        config = GenerateConfig(
                            max_new_tokens=tokens,
                            reuse_cache=reuse,
                            capture_hidden_states=capture,
                        )
                        config.validate()
                        clone = GenerateConfig.model_validate_json(
                            config.model_dump_json()
                        )
                        self.assertEqual(clone.capture_hidden_states, capture)
                        self.assertEqual(clone.reuse_cache, reuse)
                        self.assertEqual(clone.max_new_tokens, tokens)
                        config.update({"capture_hidden_states": not capture})
                        self.assertEqual(config.capture_hidden_states, not capture)
                        self.assertEqual(config.reuse_cache, reuse)
                        self.assertEqual(
                            config.update_and_pop({"capture_hidden_states": capture}),
                            {},
                        )
                        self.assertEqual(config.capture_hidden_states, capture)

    def test_prefill_only_preserves_cache_setting(self):
        for reuse_cache in (False, True):
            with self.subTest(reuse_cache=reuse_cache):
                config = GenerateConfig(max_new_tokens=0, reuse_cache=reuse_cache)
                self.assertTrue(config.is_prefill_only())
                config.validate()
                self.assertEqual(config.reuse_cache, reuse_cache)

    def test_negative_or_null_token_limits_are_rejected(self):
        for field, value in (
            ("max_new_tokens", -1),
            ("max_new_tokens", None),
            ("min_new_tokens", None),
        ):
            with self.subTest(field=field, value=value):
                with self.assertRaisesRegex(ValidationError, field):
                    GenerateConfig(**{field: value})

    def test_is_prefill_only_is_a_pure_query(self):
        config = GenerateConfig(max_new_tokens=1)
        config.max_new_tokens = -1
        before = config.model_copy(deep=True)
        self.assertFalse(config.is_prefill_only())
        with self.assertRaisesRegex(FtRuntimeException, "max_new_tokens") as ctx:
            config.validate()
        self.assertEqual(
            ctx.exception.exception_type, ExceptionType.ERROR_INPUT_FORMAT_ERROR
        )
        self.assertEqual(config, before)

    def test_unsupported_prefill_options_on_every_entry_path(self):
        cases = {
            "min_new_tokens": 1,
            "num_beams": 2,
            "variable_num_beams": [1],
            "num_return_sequences": 2,
            "return_logits": True,
            "calculate_loss": 1,
            "return_softmax_probs": True,
            "return_all_probs": True,
            "return_cum_log_probs": True,
            "return_logprobs": True,
            "return_hidden_states": True,
            "return_all_hidden_states": True,
        }
        for field, value in cases.items():
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValidationError, field):
                    GenerateConfig(max_new_tokens=0, **{field: value})
                for method in ("update", "update_and_pop"):
                    for already_prefill in (False, True):
                        config = (
                            GenerateConfig(max_new_tokens=0)
                            if already_prefill
                            else GenerateConfig(max_new_tokens=1, **{field: value})
                        )
                        updates = (
                            {field: value} if already_prefill else {"max_new_tokens": 0}
                        )
                        before = config.model_copy(deep=True)
                        with self.assertRaisesRegex(FtRuntimeException, field) as ctx:
                            getattr(config, method)({"reuse_cache": False, **updates})
                        self.assertEqual(
                            ctx.exception.exception_type,
                            ExceptionType.ERROR_INPUT_FORMAT_ERROR,
                        )
                        self.assertEqual(config, before)
                        self.assertEqual(
                            config.model_fields_set, before.model_fields_set
                        )
                config = GenerateConfig(max_new_tokens=1, **{field: value})
                config.max_new_tokens = 0
                with self.assertRaisesRegex(FtRuntimeException, field):
                    config.validate()

    def test_positive_logprobs_and_atomic_transition_to_prefill_only(self):
        for top_logprobs in (0, 20):
            for method in ("update", "update_and_pop"):
                with self.subTest(top_logprobs=top_logprobs, method=method):
                    config = GenerateConfig(
                        max_new_tokens=1,
                        return_logprobs=True,
                        top_logprobs=top_logprobs,
                    )
                    config.validate()
                    getattr(config, method)(
                        {
                            "max_new_tokens": 0,
                            "return_logprobs": False,
                            "top_logprobs": 0,
                        }
                    )
                    config.validate()
                    self.assertTrue(config.is_prefill_only())
                    self.assertFalse(config.return_logprobs)

    def test_base_list_min_tokens_remains_supported(self):
        GenerateConfig(max_new_tokens=1, min_new_tokens=[1, 2]).validate()
        GenerateConfig(max_new_tokens=0, min_new_tokens=[0, 0]).validate()
        with self.assertRaisesRegex(ValidationError, "min_new_tokens"):
            GenerateConfig(max_new_tokens=0, min_new_tokens=[0, 1])

    def test_positive_generation_preserves_base_features(self):
        config = GenerateConfig(
            max_new_tokens=1,
            min_new_tokens=1,
            num_beams=2,
            variable_num_beams=[1],
            num_return_sequences=2,
            return_logits=True,
            calculate_loss=1,
            return_softmax_probs=True,
            return_all_probs=True,
            return_cum_log_probs=True,
            return_hidden_states=True,
            return_all_hidden_states=True,
        )
        config.validate()
        self.assertFalse(config.is_prefill_only())
        self.assertTrue(config.reuse_cache)


class TestGenerateConfigUpdates(unittest.TestCase):
    def test_updates_transition_into_and_out_of_prefill_only(self):
        for method in ("update", "update_and_pop"):
            config = GenerateConfig(max_new_tokens=1)
            for limit in (0, 1):
                values = {
                    "max_new_tokens": limit,
                    "unknown": "keep",
                    "validate": "keep",
                }
                original = values.copy()
                remaining = getattr(config, method)(values)
                self.assertEqual(config.is_prefill_only(), limit == 0)
                self.assertTrue(config.reuse_cache)
                self.assertTrue(callable(config.validate))
                self.assertEqual(values, original)
                self.assertEqual(
                    remaining,
                    (
                        {"unknown": "keep", "validate": "keep"}
                        if method == "update_and_pop"
                        else None
                    ),
                )

    def test_failed_update_is_atomic(self):
        for method in ("update", "update_and_pop"):
            for field, value in (
                ("max_new_tokens", -1),
                ("max_new_tokens", None),
                ("aux_info", object()),
                ("min_new_tokens", None),
            ):
                with self.subTest(method=method, field=field, value=value):
                    config = GenerateConfig(max_new_tokens=3, min_new_tokens=1)
                    before = config.model_copy(deep=True)
                    with self.assertRaisesRegex(FtRuntimeException, field) as ctx:
                        getattr(config, method)({"reuse_cache": False, field: value})
                    self.assertEqual(
                        ctx.exception.exception_type,
                        ExceptionType.ERROR_INPUT_FORMAT_ERROR,
                    )
                    self.assertEqual(config, before)
                    self.assertEqual(config.model_fields_set, before.model_fields_set)

    def test_strict_field_updates_reject_constructor_invalid_types_atomically(self):
        for value in (True, False, "0", "2", 0.0, 2.0, 1.5, None, [], {}):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValidationError, "top_logprobs"):
                    GenerateConfig(return_logprobs=True, top_logprobs=value)
                for method in ("update", "update_and_pop"):
                    config = GenerateConfig(return_logprobs=True, top_logprobs=2)
                    original = config
                    before = config.model_copy(deep=True)
                    updates = {
                        "reuse_cache": False,
                        "json_schema": {"type": "object"},
                        "top_logprobs": value,
                        "unknown": "keep",
                    }
                    original_updates = updates.copy()
                    with self.assertRaisesRegex(
                        FtRuntimeException, "top_logprobs"
                    ) as ctx:
                        getattr(config, method)(updates)
                    self.assertEqual(
                        ctx.exception.exception_type,
                        ExceptionType.ERROR_INPUT_FORMAT_ERROR,
                    )
                    self.assertIs(config, original)
                    self.assertEqual(config, before)
                    self.assertEqual(config.model_fields_set, before.model_fields_set)
                    self.assertEqual(updates, original_updates)

    def test_strict_field_updates_accept_integer_boundaries(self):
        for value in (0, 20):
            for method in ("update", "update_and_pop"):
                with self.subTest(value=value, method=method):
                    config = GenerateConfig(return_logprobs=True)
                    original = config
                    updates = {"top_logprobs": value, "unknown": "keep"}
                    remaining = getattr(config, method)(updates)
                    self.assertIs(config, original)
                    self.assertEqual(config.top_logprobs, value)
                    self.assertIs(type(config.top_logprobs), int)
                    self.assertIn("top_logprobs", config.model_fields_set)
                    self.assertEqual(
                        updates, {"top_logprobs": value, "unknown": "keep"}
                    )
                    self.assertEqual(
                        remaining,
                        {"unknown": "keep"} if method == "update_and_pop" else None,
                    )
                    config.validate()

    def test_updates_preserve_constructor_coercion(self):
        values = {"max_new_tokens": "4", "reuse_cache": "false", "top_p": "0.75"}
        expected = GenerateConfig(**values)
        for method in ("update", "update_and_pop"):
            config = GenerateConfig()
            getattr(config, method)(values)
            for field in values:
                self.assertEqual(getattr(config, field), getattr(expected, field))

    def test_update_defers_full_request_validation_and_grammar_normalization(self):
        for method in ("update", "update_and_pop"):
            with self.subTest(method=method):
                config = GenerateConfig()
                getattr(config, method)(
                    {"json_schema": {"type": "object"}, "top_logprobs": 2}
                )
                self.assertEqual(config.json_schema, {"type": "object"})
                with self.assertRaisesRegex(FtRuntimeException, "top_logprobs"):
                    config.validate()
                getattr(config, method)({"return_logprobs": True})
                config.validate()
                self.assertEqual(config.json_schema, '{"type":"object"}')


if __name__ == "__main__":
    unittest.main()
