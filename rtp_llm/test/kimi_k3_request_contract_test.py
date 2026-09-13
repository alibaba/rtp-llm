import unittest
from types import SimpleNamespace

from rtp_llm.config.exceptions import FtRuntimeException
from rtp_llm.models.kimi_k3.kimi_k3_request_contract import (
    apply_kimi_k3_request_contract,
    kimi_k3_pending_prompt_token_count,
    kimi_k3_pending_prompt_token_ids,
)


class KimiK3RequestContractTest(unittest.TestCase):
    @staticmethod
    def config(**overrides):
        values = {
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
                self.assertEqual(config.top_p, 0.95)
                self.assertEqual(config.num_return_sequences, 0)
                self.assertEqual(config.presence_penalty, 0.0)
                self.assertEqual(config.frequency_penalty, 0.0)
                self.assertEqual(config.in_think_mode, thinking)
                if not thinking:
                    self.assertEqual(config.max_thinking_tokens, 0)

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
