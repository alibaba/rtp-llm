import gzip
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from CompareLatencyPressureBuckets import (
    attach_fit_inflight, augment, bucket, compare, parse_fit_inflight_line,
    make_metadata, predictions, pressure, run_replay, summarize,
)


def sample(sequence, running, waiting=0):
    features = [0] * 13
    features[1] = running
    features[6] = waiting
    return {"type": "sample", "sequence": sequence, "history_sequence": sequence,
            "features": features}


class CompareLatencyPressureBucketsTest(unittest.TestCase):
    def test_production_mode_is_forwarded_to_replay(self):
        with patch("CompareLatencyPressureBuckets.subprocess.run") as run:
            run_replay(Path("replay"), Path("events"), Path("result"), 4, 6, 0.03,
                       "production")
        self.assertEqual("production", run.call_args[0][0][-1])
        with patch("CompareLatencyPressureBuckets.subprocess.run") as run:
            run_replay(Path("replay"), Path("events"), Path("result"), 6, 16, 0.8,
                       "production_cap:96,1.0")
        self.assertEqual("production_cap:96,1.0", run.call_args[0][0][-1])

    def test_numeric_node_names_preserve_hardware_group(self):
        rows = {("11_225_196_216", 1): (10.0, 12.0, 12.0, 1),
                ("215_2_192_71", 1): (10.0, 11.0, 11.0, 1)}
        metadata = {key: (1, 1, 2) for key in rows}
        result = compare(rows, rows, metadata, 0)
        self.assertEqual(1, result["MI/all"]["count"])
        self.assertEqual(1, result["RTX/all"]["count"])

    def test_manifest_architecture_overrides_ip_prefix(self):
        rows = {("215_77_72_161", 1): (10.0, 12.0, 12.0, 1)}
        metadata = {key: (1, 1, 2) for key in rows}
        result = compare(rows, rows, metadata, 0, {"215_77_72_161": "MI"})
        self.assertEqual(1, result["MI/all"]["count"])
        self.assertEqual(1, result["MI/host_215_77_72_161"]["count"])
        self.assertNotIn("RTX/all", result)

    def test_ready_time_window_matches_prediction_and_sample_metadata(self):
        current = sample(1, 4)
        current.update(event_time_us=12_000, target_ms=2.0)
        files = {"host.jsonl": [current]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.tsv"
            path.write_text("P\thost\t4\t20261008\t1\t12000\t2\t1\t0\t3\t1\t3\t3\n",
                            encoding="utf-8")
            self.assertEqual(1, len(predictions(path, 9_000, 11_000, "ready")))
            self.assertEqual(0, len(predictions(path, 9_000, 11_000, "event")))
            self.assertEqual(0, len(predictions(path, 9_000, 11_000, "ready",
                                                {"host"})))
        self.assertEqual((4, 4, 4), make_metadata(files, 9_000, 11_000,
                                                   "ready")[("host", 1)])

    def test_bucket_uses_running_and_waiting(self):
        self.assertEqual(pressure(sample(1, 4, 3)), 7)
        self.assertEqual(bucket(7), 2)
        self.assertEqual(bucket(8), 3)
        self.assertEqual(bucket(0), -1)

    def test_current_sample_pressure_and_no_future_sample(self):
        events = [sample(1, 3), sample(2, 4), sample(3, 5), sample(4, 4),
                  {"type": "update", "fit_batches": [[1, 3], [2, 4]]}]
        update = list(augment(events, 2, "current_sample"))[-1]
        self.assertEqual(update["fit_batches"], [[1, 2, 3], [2, 3, 2, 4]])
        self.assertEqual(events[-1]["fit_batches"], [[1, 3], [2, 4]])

    def test_latest_pressure_proxy_is_separate(self):
        events = [sample(1, 4), sample(2, 8), sample(3, 4), sample(4, 8),
                  {"type": "update", "fit_batches": [[1, 3], [2, 4]]}]
        current = list(augment(events, 1, "current_sample"))[-1]
        latest = list(augment(events, 1, "update_latest"))[-1]
        self.assertEqual(current["fit_batches"][0], [1, 1, 3])
        self.assertEqual(latest["fit_batches"][0], [1, 2, 3])

    def test_replace_covers_other_pressure_buckets_without_future_samples(self):
        events = [sample(1, 2), sample(2, 4), sample(3, 8), sample(4, 4),
                  {"type": "update", "fit_batches": [[1, 4]]}]
        update = list(augment(events, 1, "current_sample", replace=True))[-1]
        self.assertEqual(set(update["fit_batches"][0][:-1]), {1, 2, 3})
        self.assertEqual(update["fit_batches"][0][-1], 4)

    def test_fit_inflight_selects_live_pressure_not_sample_pressure(self):
        events = [sample(1, 4), sample(2, 8), sample(3, 4),
                  {"type": "update", "fit_batches": [[1, 3]],
                   "fit_inflight_before_valid": 1,
                   "fit_inflight_before_requests": 8,
                   "fit_inflight_stable": True}]
        update = list(augment(events, 1, "fit_inflight"))[-1]
        self.assertEqual(update["fit_batches"], [[1, 2, 3]])

    def test_changing_inflight_uses_fit_start_snapshot(self):
        events = [sample(1, 4), sample(2, 8), sample(3, 4),
                  {"type": "update", "fit_batches": [[1, 3]],
                   "fit_inflight_before_valid": 1,
                   "fit_inflight_before_requests": 8,
                   "fit_inflight_stable": False}]
        update = list(augment(events, 1, "fit_inflight"))[-1]
        self.assertEqual(update["fit_batches"], [[1, 2, 3]])

    def test_missing_fit_start_snapshot_keeps_original_training_batch(self):
        events = [sample(1, 4), sample(2, 8), sample(3, 4),
                  {"type": "update", "fit_batches": [[1, 3]],
                   "fit_inflight_before_valid": 0,
                   "fit_inflight_before_requests": 8}]
        update = list(augment(events, 1, "fit_inflight"))[-1]
        self.assertEqual(update["fit_batches"], [[1, 3]])

    def test_pressure_selectors_share_exact_fit_window(self):
        events = [sample(1, 4), sample(2, 8), sample(3, 4),
                  {"type": "update", "fit_batches": [[1, 3]]}]
        self.assertEqual([[1, 3]], list(augment(
            events, 1, "current_sample", shared_fit_window=True))[-1]["fit_batches"])
        events[-1]["fit_inflight_before_valid"] = 1
        events[-1]["fit_inflight_before_requests"] = 8
        self.assertEqual([[1, 1, 3]], list(augment(
            events, 1, "current_sample", shared_fit_window=True))[-1]["fit_batches"])
        self.assertEqual([[1, 2, 3]], list(augment(
            events, 1, "fit_inflight", shared_fit_window=True))[-1]["fit_batches"])

    def test_overshoot_p95_uses_positive_errors_only(self):
        metrics = summarize([(10.0, 10.0), (10.0, 9.0), (10.0, 12.0)])
        self.assertEqual(2.0, metrics["over_p95_ms"])
        self.assertEqual(1.0, metrics["under_p95_ms"])

    def test_parse_live_fit_inflight(self):
        line = ("[navi-latency-training-update] sequence=17 event_time_us=42 "
                "fit_inflight_before_valid=1 fit_inflight_before_requests=8 "
                "fit_inflight_before_revision=12 fit_inflight_before_time_us=40 "
                "fit_inflight_after_valid=1 fit_inflight_after_requests=7 "
                "fit_inflight_after_revision=13 fit_inflight_after_time_us=43")
        sequence, values = parse_fit_inflight_line(line)
        self.assertEqual(sequence, 17)
        self.assertEqual(values["fit_inflight_before_requests"], 8)
        self.assertEqual(values["fit_inflight_before_time_us"], 40)
        self.assertEqual(values["fit_inflight_after_revision"], 13)
        self.assertEqual(values["fit_inflight_after_time_us"], 43)
        self.assertIsNone(parse_fit_inflight_line(line.replace(
            "fit_inflight_before_time_us=40 ", "")))

    def test_missing_live_pressure_fails_only_inside_window(self):
        old = sample(1, 4)
        old["event_time_us"] = 10
        current = sample(2, 8)
        current["event_time_us"] = 20
        files = {"host.jsonl": [old, {"type": "update", "sequence": 1,
                                     "fit_batches": [[1]]}, current,
                                {"type": "update", "sequence": 2,
                                 "fit_batches": [[2]]}]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "host.log.gz"
            with gzip.open(path, "wt") as stream:
                stream.write("[navi-latency-training-update] sequence=2 "
                             "fit_inflight_before_valid=1 fit_inflight_before_requests=8 "
                             "fit_inflight_before_revision=12 fit_inflight_before_time_us=21 "
                             "fit_inflight_after_valid=1 fit_inflight_after_requests=8 "
                             "fit_inflight_after_revision=12 fit_inflight_after_time_us=22\n")
            attach_fit_inflight(files, Path(directory), 15, 25)
            self.assertNotIn("fit_inflight_before_requests", files["host.jsonl"][1])
            self.assertEqual(files["host.jsonl"][3]["fit_inflight_before_requests"], 8)
            self.assertTrue(files["host.jsonl"][3]["fit_inflight_stable"])
            with self.assertRaisesRegex(ValueError, "missing live fit inflight"):
                attach_fit_inflight(files, Path(directory), 5, 25)

    def test_changed_revision_retains_fit_start_pressure(self):
        samples = [sample(1, 4), sample(2, 8), sample(3, 4)]
        for current in samples:
            current["event_time_us"] = 20
        files = {"host.jsonl": [*samples, {"type": "update", "sequence": 1,
                                              "fit_batches": [[1, 3]]}]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "host.log.gz"
            with gzip.open(path, "wt") as stream:
                stream.write("[navi-latency-training-update] sequence=1 "
                             "fit_inflight_before_valid=1 fit_inflight_before_requests=8 "
                             "fit_inflight_before_revision=12 fit_inflight_before_time_us=21 "
                             "fit_inflight_after_valid=1 fit_inflight_after_requests=7 "
                             "fit_inflight_after_revision=13 fit_inflight_after_time_us=22\n")
            coverage = attach_fit_inflight(files, Path(directory), 15, 25)
            self.assertEqual(0, coverage["stable"])
            self.assertEqual(1, coverage["unstable"])
            self.assertFalse(files["host.jsonl"][-1]["fit_inflight_stable"])
            self.assertEqual([[1, 2, 3]], list(augment(files["host.jsonl"], 1,
                                                       "fit_inflight"))[-1]["fit_batches"])

    def test_fit_time_not_sample_time_selects_window(self):
        current = sample(1, 8)
        current["event_time_us"] = 10
        files = {"host.jsonl": [current, {"type": "update", "sequence": 1,
                                          "fit_batches": [[1]]}]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "host.log.gz"
            with gzip.open(path, "wt") as stream:
                stream.write("[navi-latency-training-update] sequence=1 "
                             "fit_inflight_before_valid=1 fit_inflight_before_requests=8 "
                             "fit_inflight_before_revision=12 fit_inflight_before_time_us=21 "
                             "fit_inflight_after_valid=1 fit_inflight_after_requests=8 "
                             "fit_inflight_after_revision=12 fit_inflight_after_time_us=22\n")
            coverage = attach_fit_inflight(files, Path(directory), 15, 25)
            self.assertEqual(1, coverage["stable"])
            self.assertEqual(21, files["host.jsonl"][1]["fit_inflight_before_time_us"])

            inside_sample = sample(1, 8)
            inside_sample["event_time_us"] = 20
            outside_files = {"host.jsonl": [inside_sample, {"type": "update",
                          "sequence": 1, "fit_batches": [[1]]}]}
            outside_coverage = attach_fit_inflight(outside_files, Path(directory), 15, 21)
            self.assertEqual(0, outside_coverage["stable"])
            self.assertNotIn("fit_inflight_before_requests", outside_files["host.jsonl"][1])

    def test_reversed_fit_times_are_rejected(self):
        current = sample(1, 8)
        current["event_time_us"] = 20
        files = {"host.jsonl": [current, {"type": "update", "sequence": 1,
                                          "fit_batches": [[1]]}]}
        with tempfile.TemporaryDirectory() as directory:
            with gzip.open(Path(directory) / "host.log.gz", "wt") as stream:
                stream.write("[navi-latency-training-update] sequence=1 "
                             "fit_inflight_before_valid=1 fit_inflight_before_requests=8 "
                             "fit_inflight_before_revision=12 fit_inflight_before_time_us=22 "
                             "fit_inflight_after_valid=1 fit_inflight_after_requests=8 "
                             "fit_inflight_after_revision=12 fit_inflight_after_time_us=21\n")
            with self.assertRaisesRegex(ValueError, "invalid fit inflight time"):
                attach_fit_inflight(files, Path(directory), 15, 25)


if __name__ == "__main__":
    unittest.main()
