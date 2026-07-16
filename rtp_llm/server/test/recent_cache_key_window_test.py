import os
import unittest
from unittest.mock import patch

from rtp_llm.server.recent_cache_key_window import (
    CACHE_HIT_TIME_WINDOW_MS_ENV,
    DEFAULT_TIME_WINDOW_MS,
    RecentCacheKeyWindow,
)


class TestRecentCacheKeyWindow(unittest.TestCase):
    def test_counts_request_hits_against_prior_window(self):
        now_ms = 0
        window = RecentCacheKeyWindow(1000, lambda: now_ms)

        first = window.record([1, 2, 3])
        self.assertEqual(first.request_occurrences, 3)
        self.assertEqual(first.request_hit_occurrences, 0)
        self.assertEqual(first.request_hit_ratio, 0.0)

        second = window.record([2, 3, 4])
        self.assertEqual(second.request_occurrences, 3)
        self.assertEqual(second.request_hit_occurrences, 2)
        self.assertAlmostEqual(second.request_hit_ratio, 2 / 3)
        self.assertEqual(second.retained_occurrences, 6)
        self.assertEqual(second.retained_unique_cache_keys, 4)

    def test_expires_old_entries_before_matching(self):
        now_ms = 0

        def get_now_ms():
            return now_ms

        window = RecentCacheKeyWindow(1000, get_now_ms)
        window.record([1, 2, 3])
        now_ms = 1001

        snapshot = window.record([1, 4])
        self.assertEqual(snapshot.request_occurrences, 2)
        self.assertEqual(snapshot.request_hit_occurrences, 0)
        self.assertEqual(snapshot.retained_occurrences, 2)
        self.assertEqual(snapshot.retained_unique_cache_keys, 2)

    def test_repeated_keys_do_not_self_hit_within_one_request(self):
        now_ms = 0
        window = RecentCacheKeyWindow(1000, lambda: now_ms)

        first = window.record([7, 7, 7])
        self.assertEqual(first.request_occurrences, 3)
        self.assertEqual(first.request_hit_occurrences, 0)

        second = window.record([7, 7])
        self.assertEqual(second.request_occurrences, 2)
        self.assertEqual(second.request_hit_occurrences, 2)
        self.assertEqual(second.request_hit_ratio, 1.0)

    def test_uses_environment_time_window(self):
        with patch.dict(os.environ, {CACHE_HIT_TIME_WINDOW_MS_ENV: "1234"}):
            self.assertEqual(RecentCacheKeyWindow().time_window_ms, 1234)

    def test_invalid_environment_time_window_falls_back(self):
        with patch.dict(os.environ, {CACHE_HIT_TIME_WINDOW_MS_ENV: "bad"}):
            self.assertEqual(RecentCacheKeyWindow().time_window_ms, DEFAULT_TIME_WINDOW_MS)


if __name__ == "__main__":
    unittest.main()
