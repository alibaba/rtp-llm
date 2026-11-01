"""CPU contracts for statistics and metadata. GPU capture validated separately."""
import unittest
from bench_consumer import (_epoch_bootstrap_p99, _validate_metadata, balanced_order,
                            _wall_parts, calibrate_timing, summarize, summarize_records)


class BenchmarkContracts(unittest.TestCase):
    def test_nearest_rank_and_negative_delta_preserved(self):
        result = summarize([-3., -2., -1., 0., 101.])
        self.assertEqual(result["p50_us"], -1.)
        self.assertEqual(result["p99_us"], 101.)
        self.assertEqual(result["over_threshold_count"], 1)
        self.assertEqual(result["min_us"], -3.)

    def test_nonfinite_rejected(self):
        for values in ([], [float("nan")], [float("inf")]):
            with self.assertRaises(ValueError):
                summarize(values)

    def test_order_balances_positions_and_is_deterministic(self):
        names = ["a", "b", "c", "d"]
        orders = [balanced_order(names, i) for i in range(8)]
        for position in range(4):
            self.assertEqual(sorted(order[position] for order in orders), sorted(names * 2))
        self.assertEqual(orders, [balanced_order(names, i) for i in range(8)])
        self.assertEqual(names, ["a", "b", "c", "d"])

    def test_pairwise_difference_is_not_difference_of_quantiles(self):
        records = []
        for pair, (local, remote) in enumerate([(1., 99.), (99., 100.)]):
            for path, duration in [("local", local), ("remote", remote)]:
                records.append(dict(distribution="random", clock="gpu", epoch=pair,
                                    pair_id=pair, path=path, duration_us=duration))
        result = summarize_records(records, {"delta": ("remote", "local")}, bootstrap_iterations=20)
        delta = result["paired_deltas"][0]
        self.assertEqual(delta["paired_delta_stats"]["p99_us"], 98.)
        self.assertEqual(delta["difference_of_p99_us"], 1.)
        self.assertEqual(result["model_slo"], "NOT_RUN")

    def test_duplicate_or_missing_pair_rejected(self):
        row = dict(distribution="hot", clock="gpu", epoch=0, pair_id=0, path="a", duration_us=1.)
        with self.assertRaises(ValueError):
            summarize_records([row, row], {})
        with self.assertRaises(ValueError):
            summarize_records([row], {"delta": ("b", "a")})

    def test_bootstrap_reproducible_and_not_production_guarantee(self):
        a = _epoch_bootstrap_p99([[1., 2.], [3., 4.]], 20, 41)
        self.assertEqual(a, _epoch_bootstrap_p99([[1., 2.], [3., 4.]], 20, 41))
        self.assertEqual(a["status"], "EXPLORATORY")
        self.assertEqual(_epoch_bootstrap_p99([[1.]], 20, 41)["status"], "NOT_COMPUTED")

    def test_metadata_capabilities_and_nonfinite_are_rejected(self):
        _validate_metadata({"M": 1, "cuda_capability": [10, 0]})
        for value in ({"nested": {"fabric_handle": "secret"}}, {"x": float("nan")}, {"x": b"bytes"}):
            with self.assertRaises(ValueError):
                _validate_metadata(value)

    def test_calibration_rejects_invalid_sizes_before_cuda_import(self):
        for value in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                calibrate_timing(value)
            with self.assertRaises(ValueError):
                calibrate_timing(1, warmup=value)

    def test_wall_partitions_preserve_full_interval_and_thread_deltas(self):
        result = _wall_parts(1000, 4500, 12000, (3, 5), (5, 6))
        self.assertEqual(result["cpu_graph_replay_call_us"], 3.5)
        self.assertEqual(result["cpu_completion_sync_call_us"], 7.5)
        self.assertEqual(result["thread_nvcsw_delta"], 2)
        self.assertEqual(result["thread_nivcsw_delta"], 1)
        self.assertEqual(result["cpu_graph_replay_call_us"] + result["cpu_completion_sync_call_us"], 11.)
        self.assertIsNone(_wall_parts(1, 2, 3)["thread_nvcsw_delta"])

    def test_wall_partitions_reject_invalid_clocks_and_counters(self):
        with self.assertRaises(ValueError):
            _wall_parts(100, 90, 110)
        with self.assertRaises(ValueError):
            _wall_parts(100, 110, 120, (3, 0), (2, 0))


if __name__ == "__main__":
    unittest.main()
