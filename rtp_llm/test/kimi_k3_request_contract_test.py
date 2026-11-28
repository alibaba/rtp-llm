import unittest
from enum import Enum
from types import SimpleNamespace

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.models.kimi_k3.kimi_k3_request_contract import (
    apply_kimi_k3_request_contract,
    kimi_k3_pending_prompt_token_count,
    kimi_k3_pending_prompt_token_ids,
    kimi_k3_prompt_channel,
    validate_kimi_k3_tool_history,
)


class KimiK3ToolHistoryTest(unittest.TestCase):
    @staticmethod
    def call(call_id="call_a", **function_overrides):
        return {
            "id": call_id,
            "type": "function",
            "function": {
                "name": "lookup",
                "arguments": '{"q":"x"}',
                **function_overrides,
            },
        }

    def test_plain_histories_and_empty_assistant_calls(self):
        for messages in (
            [],
            [{"role": "user", "content": "question"}],
            [SimpleNamespace(role="assistant", content="answer")],
            [{"role": "assistant", "tool_calls": []}],
        ):
            with self.subTest(messages=messages):
                validate_kimi_k3_tool_history(messages)

    def test_attribute_records_enum_roles_and_tuple_calls(self):
        class Role(Enum):
            assistant = "assistant"
            tool = "tool"

        call = self.call()
        call["function"] = SimpleNamespace(**call["function"])
        messages = [
            SimpleNamespace(role=Role.assistant, tool_calls=(SimpleNamespace(**call),)),
            SimpleNamespace(role=Role.tool, tool_call_id="call_a"),
        ]
        validate_kimi_k3_tool_history(messages)

    def test_incomplete_history_is_rejected(self):
        cases = (
            [{"role": "tool", "tool_call_id": "unseen"}],
            [{"role": "assistant", "tool_calls": [self.call()]}],
            [
                {"role": "assistant", "tool_calls": [self.call()]},
                {"role": "user", "content": "continue"},
            ],
            [
                {"role": "tool", "tool_call_id": "unseen"},
                {"role": "tool", "tool_call_id": "unseen"},
            ],
        )
        for messages in cases:
            with self.subTest(messages=messages), self.assertRaises(FtRuntimeException):
                validate_kimi_k3_tool_history(messages)

    def test_rejects_malformed_records(self):
        cases = [
            ([{"role": "user", "tool_calls": []}], "only valid"),
            ([{"role": "assistant", "tool_calls": "bad"}], "must be an array"),
            ([{"role": "tool"}], "non-empty string"),
            ([{"role": "tool", "tool_call_id": " "}], "non-empty string"),
        ]
        malformed_calls = (
            (self.call(" "), "id must"),
            ({**self.call(), "type": "other"}, "type must"),
            ({**self.call(), "function": None}, "function must"),
            (self.call(name=" "), "name must"),
            (self.call(arguments=None), "non-empty JSON"),
            (self.call(arguments=" "), "non-empty JSON"),
            (self.call(arguments="{"), "valid JSON"),
            (self.call(arguments="[]"), "decode to a JSON object"),
        )
        cases.extend(
            ([{"role": "assistant", "tool_calls": [call]}], detail)
            for call, detail in malformed_calls
        )
        for messages, detail in cases:
            with self.subTest(messages=messages), self.assertRaisesRegex(
                FtRuntimeException, detail
            ) as caught:
                validate_kimi_k3_tool_history(messages)
            self.assertEqual(
                caught.exception.exception_type, ExceptionType.INVALID_PARAMS
            )

    def test_call_ids_stay_unique_after_results_complete(self):
        messages = [
            {"role": "assistant", "tool_calls": [self.call()]},
            {"role": "tool", "tool_call_id": "call_a"},
            {"role": "assistant", "tool_calls": [self.call()]},
        ]
        with self.assertRaisesRegex(
            FtRuntimeException, r"messages\[2\].tool_calls\[0\].id duplicates"
        ):
            validate_kimi_k3_tool_history(messages)

    def test_strict_history_pairs_parallel_results_and_rejects_repeated_results(self):
        messages = [
            {
                "role": "assistant",
                "tool_calls": [self.call(), self.call("call_b")],
            },
            {"role": "tool", "tool_call_id": "call_b"},
            {"role": "tool", "tool_call_id": "call_a"},
        ]
        validate_kimi_k3_tool_history(messages)
        with self.assertRaisesRegex(FtRuntimeException, "missing tool results"):
            validate_kimi_k3_tool_history(messages[:-1])
        with self.assertRaisesRegex(FtRuntimeException, "does not match"):
            validate_kimi_k3_tool_history(messages + [messages[-1]])
        with self.assertRaisesRegex(FtRuntimeException, "advances the conversation"):
            validate_kimi_k3_tool_history(
                messages[:-1] + [{"role": "user", "content": "continue"}]
            )


class KimiK3RequestContractTest(unittest.TestCase):
    @staticmethod
    def config(**overrides):
        values = {
            "max_new_tokens": 32000,
            "temperature": 1.0,
            "top_p": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "num_return_sequences": 0,
            "in_think_mode": False,
            "max_thinking_tokens": 32000,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_applies_mode_specific_defaults(self) -> None:
        for thinking, expected_temperature in ((False, 0.6), (True, 1.0)):
            with self.subTest(thinking=thinking):
                config = self.config()

                apply_kimi_k3_request_contract(
                    config, specified_fields=set(), thinking=thinking
                )

                self.assertEqual(config.temperature, expected_temperature)
                self.assertEqual(config.max_new_tokens, 131072)
                self.assertEqual(config.top_p, 0.95)
                self.assertEqual(config.num_return_sequences, 0)
                self.assertEqual(config.presence_penalty, 0.0)
                self.assertEqual(config.frequency_penalty, 0.0)
                self.assertEqual(config.in_think_mode, thinking)
                if not thinking:
                    self.assertEqual(config.max_thinking_tokens, 0)

    def test_completion_limit_preserves_explicit_small_value_and_clamps_large_value(
        self,
    ) -> None:
        for field in ("max_completion_tokens", "max_new_tokens", "max_tokens"):
            limits = ((128, 128), (131072, 131072), (1048576, 131072))
            for requested, expected in limits:
                with self.subTest(field=field, requested=requested):
                    config = self.config(max_new_tokens=requested)
                    apply_kimi_k3_request_contract(
                        config, specified_fields={field}, thinking=True
                    )
                    self.assertEqual(config.max_new_tokens, expected)

    def test_accepts_and_canonicalizes_supported_values(self) -> None:
        config = self.config(
            temperature=0.7,
            top_p=0.949999988079071,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            num_return_sequences=1,
        )

        apply_kimi_k3_request_contract(
            config,
            specified_fields={
                "temperature",
                "top_p",
                "presence_penalty",
                "frequency_penalty",
                "n",
            },
            thinking=False,
        )

        self.assertEqual(config.temperature, 0.7)
        self.assertEqual(config.top_p, 0.95)
        self.assertEqual(config.num_return_sequences, 1)

    def test_rejects_unsupported_values(self) -> None:
        cases = (
            ("temperature", self.config(temperature=1.1)),
            ("top_p", self.config(top_p=0.8)),
            ("presence_penalty", self.config(presence_penalty=0.5)),
            ("frequency_penalty", self.config(frequency_penalty=0.5)),
            ("n", self.config(num_return_sequences=0)),
            ("n", self.config(num_return_sequences=2)),
        )
        for field, config in cases:
            with self.subTest(field=field), self.assertRaisesRegex(
                FtRuntimeException, field
            ):
                apply_kimi_k3_request_contract(
                    config, specified_fields={field}, thinking=True
                )

    def test_rejects_multiple_sequences_with_public_range_message(self) -> None:
        for thinking in (False, True):
            with self.subTest(thinking=thinking), self.assertRaisesRegex(
                FtRuntimeException, r"Range of n should be \[1, 1\]"
            ):
                apply_kimi_k3_request_contract(
                    self.config(num_return_sequences=3),
                    specified_fields={"n"},
                    thinking=thinking,
                )

    def test_temperature_inclusive_boundaries_and_non_finite_values(self):
        for value in (0, 0.0, 1, 1.0):
            with self.subTest(value=value):
                config = self.config(temperature=value)
                apply_kimi_k3_request_contract(
                    config, specified_fields={"temperature"}, thinking=True
                )
                self.assertEqual(config.temperature, float(value))
        for value in (
            -1e-9,
            1.000000001,
            float("nan"),
            float("inf"),
            -float("inf"),
            "0.6",
            [],
        ):
            with self.subTest(value=value), self.assertRaisesRegex(
                FtRuntimeException, "temperature"
            ):
                apply_kimi_k3_request_contract(
                    self.config(temperature=value),
                    specified_fields={"temperature"},
                    thinking=False,
                )

    def test_top_p_roundoff_tolerance_and_non_finite_values(self):
        for base in (0.95, 1.0):
            for offset in (-0.5e-6, 0, 0.5e-6):
                with self.subTest(base=base, offset=offset):
                    config = self.config(top_p=base + offset)
                    apply_kimi_k3_request_contract(
                        config, specified_fields={"top_p"}, thinking=True
                    )
                    self.assertEqual(config.top_p, base)
        for value in (
            0.95 - 2e-6,
            0.95 + 2e-6,
            1 - 2e-6,
            1 + 2e-6,
            float("nan"),
            float("inf"),
            "0.95",
        ):
            with self.subTest(value=value), self.assertRaisesRegex(
                FtRuntimeException, "top_p"
            ):
                apply_kimi_k3_request_contract(
                    self.config(top_p=value), specified_fields={"top_p"}, thinking=True
                )

    def test_penalties_reject_non_scalar_and_non_finite_values(self):
        for field in ("presence_penalty", "frequency_penalty"):
            for value in (-1e-9, float("nan"), float("inf"), "0", None):
                with self.subTest(field=field, value=value), self.assertRaisesRegex(
                    FtRuntimeException, field
                ):
                    apply_kimi_k3_request_contract(
                        self.config(**{field: value}),
                        specified_fields={field},
                        thinking=True,
                    )

    def test_thinking_budget_is_preserved_only_for_enabled_mode(self):
        for thinking, expected in ((True, 73), (False, 0)):
            with self.subTest(thinking=thinking):
                config = self.config(max_thinking_tokens=73)
                apply_kimi_k3_request_contract(
                    config, specified_fields=set(), thinking=thinking
                )
                self.assertEqual(config.max_thinking_tokens, expected)
                self.assertEqual(config.in_think_mode, thinking)


class KimiK3PendingPromptTest(unittest.TestCase):
    @staticmethod
    def tokenizer():
        return SimpleNamespace(
            encode=lambda text: {
                "<|open|>response<|sep|>": [10, 11, 12],
                "<|open|>think<|sep|>": [10, 13, 12],
                "<|open|>": [10],
                "<|close|>": [14],
            }[text]
        )

    def test_partial_assistant_keeps_only_the_last_open_channel(self):
        tokenizer = self.tokenizer()
        for ids, expected in (
            ([99, 10, 11, 12], ("response", 3)),
            ([99, 10, 13, 12], ("think", 3)),
            ([99, 10, 11, 12, 90, 91], ("response", 3)),
            ([99, 10, 13, 12, 90], ("think", 3)),
            ([10, 13, 12, 90, 14, 13, 12], (None, 0)),
            ([10, 11, 12, 90, 10, 77, 12], (None, 0)),
            ([10, 11, 12, 14, 11, 12, 10, 13, 12, 90], ("think", 3)),
            ([10, 11], (None, 0)),
            ([], (None, 0)),
        ):
            with self.subTest(ids=ids):
                self.assertEqual(
                    kimi_k3_prompt_channel(tokenizer, ids, allow_partial=True), expected
                )
        # Usage subtraction for a normal generation prompt must not consume
        # a history channel or a partial assistant's already-written body.
        self.assertEqual(kimi_k3_prompt_channel(tokenizer, [10, 11, 12, 90]), (None, 0))
        self.assertEqual(
            kimi_k3_prompt_channel(None, [10, 11, 12], allow_partial=True), (None, 0)
        )

    def test_matches_only_complete_structural_suffix(self):
        class Tokenizer:
            def encode(self, text):
                return {
                    "<|open|>response<|sep|>": [10, 11, 12],
                    "<|open|>think<|sep|>": [10, 13, 14, 12],
                }[text]

        tokenizer = Tokenizer()
        for ids, expected in (
            ([99, 10, 11, 12], 3),
            ([99, 10, 13, 14, 12], 4),
            ([10, 11, 12, 99], 0),
            ([99, 11, 12], 0),
            ([99, 10, 11], 0),
            ([99, 900, 901], 0),
            ([], 0),
        ):
            with self.subTest(ids=ids):
                self.assertEqual(
                    kimi_k3_pending_prompt_token_count(tokenizer, ids), expected
                )
        self.assertEqual(kimi_k3_pending_prompt_token_count(None, [99]), 0)

    def test_rejects_invalid_tokenizer_results(self):
        for result in (None, [], "123", ["10"], (10, 11, 12)):
            with self.subTest(result=result), self.assertRaises(TypeError):
                kimi_k3_pending_prompt_token_ids(
                    SimpleNamespace(encode=lambda text: result), False
                )


if __name__ == "__main__":
    unittest.main()
