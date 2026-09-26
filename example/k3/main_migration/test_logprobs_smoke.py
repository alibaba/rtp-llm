"""Probability smoke checks reject broken token alignment and mode semantics."""

import unittest

from logprobs_smoke import check_logprobs


class LogprobsSmokeChecksTest(unittest.TestCase):
    def body(self, rows):
        return {
            "choices": [
                {
                    "message": {"content": "AB"},
                    "finish_reason": "stop",
                    "logprobs": {"content": rows},
                }
            ],
            "usage": {"prompt_tokens": 8, "completion_tokens": 2, "total_tokens": 10},
        }

    def test_distinct_rows_and_corrupt_bytes(self):
        rows = [
            {"token": "A", "bytes": [65], "logprob": -0.2, "top_logprobs": []},
            {"token": "B", "bytes": [66], "logprob": -1.4, "top_logprobs": []},
        ]
        check_logprobs(self.body(rows), mode="original", top_count=0, expected="AB")
        rows[1]["bytes"] = [65]
        with self.assertRaisesRegex(AssertionError, "reconstruct"):
            check_logprobs(self.body(rows), mode="original", top_count=0, expected="AB")

    def test_filtered_and_original_modes_are_not_interchangeable(self):
        rows = [{"token": "AB", "bytes": [65, 66], "logprob": -0.5, "top_logprobs": []}]
        with self.assertRaisesRegex(AssertionError, "not one"):
            check_logprobs(self.body(rows), mode="default", top_count=0, expected="AB")
        rows[0]["logprob"] = 0
        with self.assertRaisesRegex(AssertionError, "point mass"):
            check_logprobs(self.body(rows), mode="original", top_count=0, expected="AB")


if __name__ == "__main__":
    unittest.main()
