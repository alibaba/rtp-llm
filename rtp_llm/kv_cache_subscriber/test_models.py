from __future__ import annotations

import unittest

from rtp_llm.kv_cache_subscriber.models import CacheDiffTracker


class CacheDiffTrackerTest(unittest.TestCase):
    def test_initial_snapshot_is_fully_added(self) -> None:
        tracker = CacheDiffTracker(deletion_confirmations=2)

        diff = tracker.plan(frozenset({3, 1, 2}))

        self.assertEqual(diff.added, (1, 2, 3))
        self.assertEqual(diff.removed, ())
        tracker.commit(diff)
        self.assertEqual(tracker.acknowledged_keys, frozenset({1, 2, 3}))

    def test_removal_requires_consecutive_full_snapshots(self) -> None:
        tracker = CacheDiffTracker(deletion_confirmations=2)
        initial = tracker.plan(frozenset({1, 2}))
        tracker.commit(initial)

        first_missing = tracker.plan(frozenset({1}))
        second_missing = tracker.plan(frozenset({1}))

        self.assertEqual(first_missing.removed, ())
        self.assertEqual(second_missing.removed, (2,))

    def test_reappearing_key_cancels_pending_removal(self) -> None:
        tracker = CacheDiffTracker(deletion_confirmations=2)
        initial = tracker.plan(frozenset({1, 2}))
        tracker.commit(initial)

        tracker.plan(frozenset({1}))
        tracker.plan(frozenset({1, 2}))
        diff = tracker.plan(frozenset({1}))

        self.assertEqual(diff.removed, ())

    def test_uncommitted_add_is_retried(self) -> None:
        tracker = CacheDiffTracker(deletion_confirmations=2)

        first = tracker.plan(frozenset({7, 8}))
        retry = tracker.plan(frozenset({7, 8}))

        self.assertEqual(first, retry)
        self.assertEqual(tracker.acknowledged_keys, frozenset())

    def test_force_full_add_replays_acknowledged_keys(self) -> None:
        tracker = CacheDiffTracker(deletion_confirmations=2)
        initial = tracker.plan(frozenset({1, 2}))
        tracker.commit(initial)

        refresh = tracker.plan(frozenset({1, 2}), force_full_add=True)

        self.assertEqual(refresh.added, (1, 2))

    def test_uncertain_add_is_deleted_if_it_disappears(self) -> None:
        tracker = CacheDiffTracker(deletion_confirmations=2)
        failed = tracker.plan(frozenset({9}))
        tracker.mark_uncertain(failed)

        first_missing = tracker.plan(frozenset())
        second_missing = tracker.plan(frozenset())

        self.assertEqual(first_missing.removed, ())
        self.assertEqual(second_missing.removed, (9,))


if __name__ == "__main__":
    unittest.main()
