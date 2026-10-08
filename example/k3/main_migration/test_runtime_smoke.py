"""Replay evidence must reject stale generations and cross-request slot aliasing."""

import unittest

from runtime_smoke import check_replay_events, check_replay_owners


class RuntimeSmokeChecksTest(unittest.TestCase):
    def event(self, time, generation, epoch, request):
        return {
            "event": "linear_replay_commit",
            "time_ns": time,
            "slot": 0,
            "generation": generation,
            "verify_epoch": epoch,
            "request_id": request,
            "produced_steps": 4,
        }

    def test_slot_reuse_with_new_generation(self):
        self.assertEqual(
            check_replay_events(
                [
                    self.event(1, 1, 1, 10),
                    self.event(2, 1, 2, 10),
                    self.event(3, 2, 1, 20),
                ]
            ),
            {"commits": 3, "slot_reuses": 1},
        )

    def test_cross_request_stale_epoch_and_generation_fail(self):
        for bad in (
            self.event(2, 1, 2, 20),
            self.event(2, 1, 1, 10),
            self.event(2, 0, 2, 10),
        ):
            with self.subTest(bad=bad), self.assertRaises(AssertionError):
                check_replay_events([self.event(1, 1, 1, 10), bad])

    def test_dp_owners_commit_while_tp_followers_have_no_stream_leases(self):
        events = [self.event(1, 1, 1, 10), self.event(2, 2, 1, 20)]
        ranks = {rank: [] for rank in range(8)}
        ranks[0] = events
        ranks[4] = events
        result = check_replay_owners(ranks, 2)
        self.assertEqual(set(result), {"0", "4"})
        self.assertEqual(result["4"], {"commits": 2, "slot_reuses": 1})

    def test_missing_dp_owner_commits_fail(self):
        ranks = {rank: [] for rank in range(8)}
        ranks[0] = [self.event(1, 1, 1, 10), self.event(2, 2, 1, 20)]
        with self.assertRaisesRegex(AssertionError, "not exercised"):
            check_replay_owners(ranks, 2)


if __name__ == "__main__":
    unittest.main()
