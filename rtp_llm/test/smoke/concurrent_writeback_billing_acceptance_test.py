import unittest

from concurrent_writeback_billing_acceptance import (
    cached_usage,
    max_overlap,
    validate_billing,
    validate_stream_usage,
)


class ConcurrentBillingAcceptanceTest(unittest.TestCase):
    def response(self, first_use=512):
        return {
            "aux_info": {
                "pd_sep": True, "input_len": 672, "output_len": 4,
                "reuse_len": 512 - first_use, "actual_reuse_len": 512,
                "writeback_first_use_len": first_use,
                "prefill_total_reuse_len": 512, "prefill_local_reuse_len": 512,
                "decode_total_reuse_len": 512,
            },
            "usage": {
                "prompt_tokens": 672, "completion_tokens": 4, "total_tokens": 676,
                "prompt_tokens_details": {"cached_tokens": 512 - first_use},
            },
        }

    def test_fresh_and_accounted_usage(self):
        for first_use in (0, 512):
            validate_billing(self.response(first_use), list(range(672)), 4, 512, {0, 512})

    def test_usage_must_follow_billable_not_physical_reuse(self):
        response = self.response()
        response["usage"]["prompt_tokens_details"]["cached_tokens"] = 512
        with self.assertRaisesRegex(AssertionError, "Usage differs"):
            validate_billing(response, list(range(672)), 4, 512, {512})

    def test_distinct_fresh_request_must_keep_first_use(self):
        with self.assertRaisesRegex(AssertionError, "first-use snapshot"):
            validate_billing(self.response(0), list(range(672)), 4, 512, {512})

    def test_incorrect_usage_total_rejected(self):
        response = self.response()
        response["usage"]["total_tokens"] += 1
        with self.assertRaisesRegex(AssertionError, "total tokens"):
            validate_billing(response, list(range(672)), 4, 512, {512})

    def test_absent_zero_cached_tokens(self):
        self.assertEqual(cached_usage({"usage": {}}), 0)
        self.assertEqual(cached_usage({"usage": {"prompt_tokens_details": None}}), 0)

    def test_real_overlap_not_only_configured_concurrency(self):
        self.assertEqual(max_overlap([
            {"start": 0, "end": 2}, {"start": 1, "end": 3},
        ]), 2)
        self.assertEqual(max_overlap([
            {"start": 0, "end": 1}, {"start": 1, "end": 2},
        ]), 1)

    def stream_frames(self):
        return [
            {"usage": None},
            *[
                {"usage": {
                    "prompt_tokens": 672,
                    "completion_tokens": completed,
                    "total_tokens": 672 + completed,
                    "prompt_tokens_details": {"cached_tokens": 512},
                }}
                for completed in (1, 2, 3, 4, 4)
            ],
        ]

    def test_stream_reports_snapshots_not_summed_chunks(self):
        validate_stream_usage(self.stream_frames(), 672, 4, 512)

    def test_stream_repeated_prompt_tokens_are_rejected(self):
        frames = self.stream_frames()
        frames[-1]["usage"]["prompt_tokens"] *= 2
        with self.assertRaisesRegex(AssertionError, "accumulated prompt"):
            validate_stream_usage(frames, 672, 4, 512)

    def test_stream_missing_final_usage_is_rejected(self):
        with self.assertRaisesRegex(AssertionError, "final completion"):
            validate_stream_usage(self.stream_frames()[:3], 672, 4, 512)


if __name__ == "__main__":
    unittest.main()
