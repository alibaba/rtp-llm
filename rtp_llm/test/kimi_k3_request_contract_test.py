import unittest
from types import SimpleNamespace

from rtp_llm.config.exceptions import FtRuntimeException
from rtp_llm.models.kimi_k3.kimi_k3_request_contract import (
    apply_kimi_k3_request_contract,
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


if __name__ == "__main__":
    unittest.main()
