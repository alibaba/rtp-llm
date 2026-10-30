import unittest
from enum import Enum
from types import SimpleNamespace

from rtp_llm.config.exceptions import ExceptionType, FtRuntimeException
from rtp_llm.models.kimi_k3.kimi_k3_request_contract import (
    apply_kimi_k3_request_contract,
    kimi_k3_pending_prompt_token_count,
    kimi_k3_pending_prompt_token_ids,
    validate_kimi_k3_tool_history,
    validate_kimi_k3_top_logprobs,
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
        for partial in (False, True):
            for messages in (
                [],
                [{"role": "user", "content": "question"}],
                [SimpleNamespace(role="assistant", content="answer")],
                [{"role": "assistant", "tool_calls": []}],
            ):
                with self.subTest(partial=partial, messages=messages):
                    validate_kimi_k3_tool_history(messages, allow_partial=partial)

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
        for partial in (False, True):
            with self.subTest(partial=partial):
                validate_kimi_k3_tool_history(messages, allow_partial=partial)

    def test_partial_history_does_not_require_pairing(self):
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
            with self.subTest(messages=messages):
                validate_kimi_k3_tool_history(messages, allow_partial=True)
                with self.assertRaises(FtRuntimeException):
                    validate_kimi_k3_tool_history(messages)

    def test_partial_history_still_rejects_malformed_records(self):
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
                validate_kimi_k3_tool_history(messages, allow_partial=True)
            self.assertEqual(
                caught.exception.exception_type, ExceptionType.INVALID_PARAMS
            )

    def test_call_ids_stay_unique_after_results_complete(self):
        messages = [
            {"role": "assistant", "tool_calls": [self.call()]},
            {"role": "tool", "tool_call_id": "call_a"},
            {"role": "assistant", "tool_calls": [self.call()]},
        ]
        for partial in (False, True):
            with self.subTest(partial=partial), self.assertRaisesRegex(
                FtRuntimeException, r"messages\[2\].tool_calls\[0\].id duplicates"
            ):
                validate_kimi_k3_tool_history(messages, allow_partial=partial)

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

    def test_top_logprobs_candidate_count_boundary(self) -> None:
        for value in (None, 0, 2):
            with self.subTest(value=value):
                validate_kimi_k3_top_logprobs(value)
        with self.assertRaisesRegex(FtRuntimeException, "top_logprobs"):
            validate_kimi_k3_top_logprobs(-1)

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


class KimiK3PendingPromptTest(unittest.TestCase):
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
