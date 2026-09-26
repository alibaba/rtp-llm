"""The grammar smoke oracle rejects plausible but schema-invalid answers."""

import unittest

from grammar_smoke import check_json_answer


class GrammarSmokeChecksTest(unittest.TestCase):
    def test_wrong_types_extra_fields_and_truncation_fail(self):
        for text in (
            '{"answer":"5"}',
            '{"answer":true}',
            '{"answer":5,"extra":0}',
            '{"answer":6}',
            '{"answer":',
        ):
            body = {
                "choices": [{"finish_reason": "stop", "message": {"content": text}}],
                "usage": {
                    "prompt_tokens": 8,
                    "completion_tokens": 5,
                    "total_tokens": 13,
                },
            }
            with self.subTest(text=text), self.assertRaises(
                (AssertionError, ValueError)
            ):
                check_json_answer(body)


if __name__ == "__main__":
    unittest.main()
