import unittest

from rtp_llm.test.utils.semantic_response_validator import validate_semantic_response


K3_TERMINAL = (
    "<|close|>response<|sep|>"
    "<|close|>message<|sep|>"
    "<|end_of_msg|>"
)
K3_RAW_DECODED_TERMINAL = (
    "<|close|> response <|sep|> "
    "<|close|> message <|sep|> "
    "<|end_of_msg|>"
)

CONFIG = {
    "minimum_chars": 20,
    "required_concept_groups": [
        ["Kimi"],
        ["月之暗面", "Moonshot"],
        ["助手", "人工智能", "AI"],
    ],
    "reject_repetition": True,
    "terminal_sequences": [K3_TERMINAL, K3_RAW_DECODED_TERMINAL],
}


class SemanticResponseTest(unittest.TestCase):
    def test_accepts_coherent_identity_response(self) -> None:
        failures = validate_semantic_response(
            "你好，我是 Kimi，由月之暗面开发的 AI 助手，可以协助回答问题。"
            + K3_TERMINAL,
            CONFIG,
        )
        self.assertEqual(failures, [])

    def test_accepts_raw_decoded_k3_terminal_sequence(self) -> None:
        failures = validate_semantic_response(
            "你好，我是 Kimi，由月之暗面开发的 AI 助手，可以协助回答问题。"
            + K3_RAW_DECODED_TERMINAL,
            CONFIG,
        )
        self.assertEqual(failures, [])

    def test_rejects_missing_identity_concept(self) -> None:
        failures = validate_semantic_response(
            "你好，我是一个可以帮助你处理日常问题的通用助手。" + K3_TERMINAL,
            CONFIG,
        )
        self.assertTrue(any("Kimi" in failure for failure in failures))

    def test_rejects_repeated_sentence(self) -> None:
        failures = validate_semantic_response(
            "我是 Kimi，由月之暗面开发的 AI 助手。"
            "我可以帮助你完成任务。我可以帮助你完成任务。"
            + K3_TERMINAL,
            CONFIG,
        )
        self.assertTrue(any("repeat" in failure for failure in failures))

    def test_rejects_content_after_terminal_sequence(self) -> None:
        failures = validate_semantic_response(
            "我是 Kimi，由月之暗面开发的 AI 助手。"
            + K3_TERMINAL
            + "又开始生成无关内容",
            CONFIG,
        )
        self.assertTrue(
            any("after terminal sequence" in failure for failure in failures)
        )

    def test_rejects_missing_terminal_sequence(self) -> None:
        failures = validate_semantic_response(
            "我是 Kimi，由月之暗面开发的 AI 助手。",
            CONFIG,
        )
        self.assertTrue(
            any("missing required terminal sequence" in failure for failure in failures)
        )

    def test_rejects_incomplete_terminal_sequence(self) -> None:
        failures = validate_semantic_response(
            "我是 Kimi，由月之暗面开发的 AI 助手。"
            "<|close|>response<|sep|><|close|>message<|sep|>",
            CONFIG,
        )
        self.assertTrue(
            any("missing required terminal sequence" in failure for failure in failures)
        )

    def test_stop_marker_mode_remains_supported(self) -> None:
        config = dict(CONFIG)
        config.pop("terminal_sequences")
        config["stop_markers"] = ["[EOS]"]
        failures = validate_semantic_response(
            "你好，我是 Kimi，由月之暗面开发的 AI 助手。[EOS]",
            config,
        )
        self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main()
