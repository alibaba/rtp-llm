import unittest

from rtp_llm.utils.speculative_acceptance import AcceptanceMonitor, checked_counters


def counts(accepted, valid, hist, generated):
    rounds = sum(hist)
    return dict(version=1, proposal_rounds=rounds, proposal_candidates=rounds * 3,
                verify_rounds=rounds, verified_candidates=valid, accepted_draft_tokens=accepted,
                generated_tokens=generated, **{f"accept_length_{n}": c for n, c in enumerate(hist)})


class AcceptanceMonitorTest(unittest.TestCase):
    def test_weighted_total_deduplication_and_owner(self):
        monitor = AcceptanceMonitor()
        a, b = counts(3, 3, [0, 0, 0, 1], 4), counts(0, 27, [9, 0, 0, 0], 9)
        monitor.observe("a", "owner0", a, 5)
        monitor.observe("a", "owner0", a, 5)  # repeated snapshot, not another TP root
        monitor.observe("b", "owner1", b, 10)
        report = monitor.report()
        self.assertEqual(report["overall"]["acceptance_rate"], 0.1)
        self.assertEqual(report["overall"]["mean_accept_length"], 0.3)
        self.assertEqual(report["actual_generated_tokens"], 15)
        with self.assertRaisesRegex(ValueError, "two owners"):
            monitor.observe("a", "owner1", a, 5)

    def test_na_and_forced_fixture_are_not_real_rounds(self):
        monitor = AcceptanceMonitor()
        monitor.observe("a", "owner0", dict(version=1, excluded_forced_rounds=3), 10)
        self.assertIsNone(monitor.report()["overall"]["acceptance_rate"])
        self.assertIsNone(monitor.report()["overall"]["mean_accept_length"])
        self.assertEqual(monitor.report()["overall"]["excluded_forced_rounds"], 3)

    def test_window_and_observed_complete_cycle(self):
        clock = [0.0]
        monitor = AcceptanceMonitor(lambda: clock[0])
        monitor.observe("a", "owner0", {}, 1, first_token=True)
        clock[0] = 5.0
        monitor.observe("a", "owner0", counts(1, 3, [0, 1, 0, 0], 2), 3)
        self.assertEqual(monitor.report(reset_window=True)["window"]["verified_candidates"], 3)
        clock[0] = 10.0
        monitor.observe("a", "owner0", counts(3, 6, [0, 1, 1, 0], 5), 6, finished=True)
        report = monitor.report()
        self.assertEqual(report["window"]["verified_candidates"], 3)
        self.assertEqual(report["window"]["accepted_draft_tokens"], 2)
        self.assertEqual(report["by_request"]["a"]["amortized_complete_cycle_wall_seconds"], 5.0)
        self.assertEqual(report["by_request"]["a"]["generation_tokens_per_second"], 0.5)

    def test_truncated_snapshot_is_not_assumed_to_start_at_prefill(self):
        monitor = AcceptanceMonitor()
        monitor.observe("a", "owner0", counts(0, 3, [1, 0, 0, 0], 1), 2, finished=True)
        self.assertIsNone(monitor.report()["by_request"]["a"]["decode_wall_seconds"])

    def test_invalid_and_regressing_counts_fail(self):
        with self.assertRaises(ValueError):
            checked_counters(counts(4, 3, [0, 0, 0, 1], 4))
        monitor = AcceptanceMonitor()
        monitor.observe("a", "owner0", counts(3, 3, [0, 0, 0, 1], 4), 5)
        with self.assertRaisesRegex(ValueError, "regressed"):
            monitor.observe("a", "owner0", {}, 5)


if __name__ == "__main__":
    unittest.main()
