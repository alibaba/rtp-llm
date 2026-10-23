import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import summarize_prefill_traffic as summary


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.rows = []
        for mode in ("stable", "independent"):
            run = self.root / mode
            (run / "environment").mkdir(parents=True)
            planned = dict(
                request_id=1,
                stage="steady",
                input_tokens=100,
                keys=[1],
                session="",
                offset_s=0.5,
            )
            request = dict(
                planned,
                success=True,
                planned_s=10.5,
                started_s=10.5,
                ended_s=11,
                ttft_s=0.2,
                pacing_lag_s=0,
            )
            stage = dict(
                stage="steady",
                name="steady",
                seconds=2,
                start_s=10,
                drained_s=12,
                start_epoch_ms=10000,
                end_epoch_ms=12000,
            )
            event = dict(
                event="prefill_done",
                rid=1,
                input_len=100,
                cache_hit_tokens=20,
                cancelled=False,
                engine_name="p0",
                engine_incarnation="i0",
                prefill_done_ms=11000,
            )
            self.write(run / "trace.json", {"traffic": [planned]})
            manifest = dict(
                case={"name": "fixture", "phases": [stage]},
                dispatch="NON_BATCH",
                jitter=mode,
                seed=1,
                master_sha256="master",
                mock_sha256="mock",
                source_sha256={"script": "same"},
                schedule_proto_sha256="proto",
                trace_sha256=hashlib.sha256(
                    (run / "trace.json").read_bytes()
                ).hexdigest(),
            )
            self.write(run / "manifest.json", manifest)
            self.write(run / "requests.json", [request])
            self.write(run / "stages.json", [stage])
            (run / "environment" / "engine_events.jsonl").write_text(
                json.dumps(event) + "\n"
            )
            self.write(
                run / "report.json",
                {
                    "status": "PASS",
                    "stages": [summary.describe_phase(stage, [request], [event])],
                },
            )
            self.rows.append(
                dict(
                    directory=mode,
                    case="fixture",
                    dispatch="NON_BATCH",
                    jitter=mode,
                    seed=1,
                    exit_code=0,
                )
            )
        self.write(self.root / "results.json", self.rows)

    def tearDown(self):
        self.tmp.cleanup()

    @staticmethod
    def write(path, value):
        path.write_text(json.dumps(value))

    def test_complete_matched_inputs(self):
        matrix, runs, phases = summary.summarize_matrix(self.root)
        self.assertEqual(matrix["status"], "COMPLETE")
        self.assertTrue(
            all(
                run["status"] == "PASS" and run["complete_request_trace"]
                for run in runs
            )
        )
        diff = summary.comparisons(runs, phases)[0]
        self.assertEqual(diff["comparison_status"], "MATCHED")
        self.assertEqual(diff["delta_success_token_hit_rate"], 0)

    def test_failure_never_overwritten_by_old_pass_report(self):
        self.write(
            self.root / "independent" / "failure.json",
            {"status": "FAIL", "error": "failed gate"},
        )
        _, runs, phases = summary.summarize_matrix(self.root)
        self.assertEqual(runs[1]["status"], "FAIL")
        self.assertEqual(
            summary.comparisons(runs, phases)[0]["comparison_status"],
            "MATCHED_UNVALIDATED",
        )

    def test_failed_run_raw_metrics(self):
        (self.root / "independent" / "report.json").unlink()
        self.rows[1]["exit_code"] = 1
        self.write(self.root / "results.json", self.rows)
        _, runs, phases = summary.summarize_matrix(self.root)
        self.assertEqual(runs[1]["status"], "FAIL")
        self.assertEqual(phases[1]["metrics_source"], "RAW_UNVALIDATED")
        self.assertEqual(phases[1]["success_token_hit_rate"], 0.2)

    def test_ambiguous_engine_evidence_does_not_invent_cache_metrics(self):
        run = self.root / "independent"
        (run / "report.json").unlink()
        events = run / "environment" / "engine_events.jsonl"
        events.write_text(events.read_text() * 2)
        _, _, phases = summary.summarize_matrix(self.root)
        self.assertIsNone(phases[1]["success_token_hit_rate"])
        self.assertIsNone(phases[1]["compute_tpm_including_drain"])

    def test_missing_actual_request_blocks_pairing(self):
        self.write(self.root / "independent" / "requests.json", [])
        _, runs, phases = summary.summarize_matrix(self.root)
        self.assertFalse(runs[1]["complete_request_trace"])
        self.assertIsNone(summary.comparisons(runs, phases)[0]["delta_ttft_p99_s"])

    def test_different_code_blocks_pairing(self):
        path = self.root / "independent" / "manifest.json"
        manifest = json.loads(path.read_text())
        manifest["source_sha256"]["script"] = "different"
        self.write(path, manifest)
        _, runs, phases = summary.summarize_matrix(self.root)
        self.assertEqual(
            summary.comparisons(runs, phases)[0]["comparison_status"],
            "NO_COMPLETE_MATCHED_CONTROL",
        )

    def test_slot_snapshot_is_incomplete_even_if_present_runs_pass(self):
        (self.root / "results.json").rename(self.root / "results-slot-0.json")
        matrix, runs, _ = summary.summarize_matrix(self.root)
        self.assertEqual(matrix["status"], "INCOMPLETE")
        self.assertTrue(all(run["matrix_status"] == "INCOMPLETE" for run in runs))

    def test_early_failure_without_manifest_is_visible(self):
        early = self.root / "early"
        early.mkdir()
        self.write(
            early / "failure.json", {"status": "FAIL", "error": "startup failed"}
        )
        matrix, runs, _ = summary.summarize_matrix(self.root)
        self.assertEqual(matrix["status"], "INCOMPLETE")
        self.assertEqual(runs[-1]["status"], "FAIL")
        self.assertEqual(runs[-1]["failure"], "startup failed")

    def test_report_fail_wins_over_zero_exit_code(self):
        self.write(self.root / "independent" / "report.json", {"status": "FAIL"})
        _, runs, _ = summary.summarize_matrix(self.root)
        self.assertEqual(runs[1]["status"], "FAIL")

    def test_corrupted_results_cannot_be_complete(self):
        (self.root / "results.json").write_text("[{")
        matrix, _, _ = summary.summarize_matrix(self.root)
        self.assertEqual(matrix["status"], "INCOMPLETE")
        self.assertTrue(matrix["warnings"])


if __name__ == "__main__":
    unittest.main()
