import json
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from analyze_slo_batch import analyze  # noqa: E402


class AnalyzeSloBatchTest(unittest.TestCase):
    def test_summarizes_decisions_and_detects_invariant_violations(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            (run / "load_client").mkdir()
            (run / "flexlb_logs").mkdir()
            (run / "flexlb_logs" / "flexlb.log.2026-07-17.0.log").write_text(
                "\n".join(
                    [
                        "flexlb_batch_dispatch batch_id=1 reason=predicted_execution_cap "
                        "batch_size=8 wait_ms=40 predicted_ms=510 threshold_ms=500 "
                        "fixed_wait_ms=160 batch_size_max=32 queue_after=2 worker=127.0.0.1:61000",
                        "flexlb_batch_dispatch batch_id=2 reason=batch_full "
                        "batch_size=31 wait_ms=20 predicted_ms=480 threshold_ms=500 "
                        "fixed_wait_ms=160 batch_size_max=32 queue_after=0 worker=127.0.0.1:61000",
                    ]
                ),
                encoding="utf-8",
            )
            (run / "flexlb_logs" / "flexlb.log").write_text(
                "flexlb_batch_complete batch_id=1 predicted_ms=510 actual_ms=520 "
                "gap_ms=10 batch_size=8 engine=127.0.0.1\n",
                encoding="utf-8",
            )
            (run / "mock_engine.log").write_text(
                "java_mock_stats enqueue_rpcs=2 enqueued_requests=39 status_rpcs=4 cache_rpcs=4 "
                "prefill_batches=2 avg_batch_size=19.50 max_batch_size=31 "
                "avg_batch_ms=500.00 max_batch_ms=520 prefill_waiting=3 prefill_running=1 "
                "max_prefill_waiting=2 decode_waiting=0 decode_running=0\n",
                encoding="utf-8",
            )
            (run / "load_client" / "summary.json").write_text(
                json.dumps({"actual_send_qps": 100.0, "error_count": 0}),
                encoding="utf-8",
            )
            (run / "load_client" / "server_latency.json").write_text(
                json.dumps({"arrival_qps": 99.0, "completion_qps": 99.0}),
                encoding="utf-8",
            )
            config = run / "master.json"
            config.write_text(
                json.dumps(
                    {
                        "zone_process_setting": {
                            "process_info": {
                                "envs": [
                                    [
                                        "FLEXLB_CONFIG",
                                        json.dumps(
                                            {
                                                "schemaVersion": 2,
                                                "scheduler": {
                                                    "type": "QUEUE",
                                                    "ordering": {"type": "PRIORITY"},
                                                },
                                                "dispatcher": {"type": "BATCH"},
                                            }
                                        ),
                                    ],
                                ]
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            result = analyze(run, config)

            self.assertEqual(2, result["decisions"]["count"])
            self.assertEqual(
                {"batch_full": 1, "predicted_execution_cap": 1},
                result["decisions"]["reasons"],
            )
            # batch_full below batch_size_max, plus a capped group of 8 whose
            # own prediction already exceeded the budget.
            self.assertEqual(2, result["decisions"]["invariant_violation_count"])
            self.assertEqual(1, result["completions"]["matched_decision_count"])
            self.assertEqual(500, result["config"]["predict_threshold_ms"])
            self.assertEqual(160, result["config"]["fixed_wait_ms"])
            self.assertEqual("QUEUE", result["config"]["scheduler_type"])
            self.assertEqual("PRIORITY", result["config"]["ordering_type"])
            self.assertEqual("BATCH", result["config"]["dispatcher_type"])
            self.assertEqual(31, result["mock"]["last"]["max_batch_size"])
            self.assertEqual(3, result["mock"]["max_observed_prefill_waiting"])

    def test_falls_back_to_legacy_mock_stats_field_names(self) -> None:
        # Older mock engines log prefill_pending / max_prefill_pending instead
        # of prefill_waiting / max_prefill_waiting; analysis must still work.
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            (run / "load_client").mkdir()
            (run / "flexlb_logs").mkdir()
            (run / "mock_engine.log").write_text(
                "java_mock_stats enqueue_rpcs=2 enqueued_requests=39 status_rpcs=4 cache_rpcs=4 "
                "prefill_batches=2 avg_batch_size=19.50 max_batch_size=31 "
                "avg_batch_ms=500.00 max_batch_ms=520 prefill_pending=3 "
                "max_prefill_pending=2 decode_running=0\n",
                encoding="utf-8",
            )

            result = analyze(run, None)

            self.assertEqual(1, result["mock"]["stats_samples"])
            self.assertEqual(3, result["mock"]["max_observed_prefill_waiting"])
            self.assertEqual(2, result["mock"]["max_observed_engine_prefill_waiting"])

    def test_uses_prometheus_dispatch_counter_as_authoritative_count(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            (run / "load_client").mkdir()
            (run / "flexlb_logs").mkdir()
            (run / "flexlb_logs" / "flexlb.log").write_text(
                "flexlb_batch_dispatch batch_id=1 reason=predicted_execution_cap "
                "batch_size=8 wait_ms=40 predicted_ms=510 threshold_ms=500 "
                "fixed_wait_ms=160 batch_size_max=32 queue_after=2 worker=127.0.0.1:61000\n",
                encoding="utf-8",
            )
            (run / "master_prometheus_after.prom").write_text(
                "flexlb_app_engine_balancing_master_dispatch_reason_total{"
                'engineIp="127.0.0.1",'
                'reason="predicted_execution_cap",role="PREFILL"} 70.0\n'
                "flexlb_app_engine_balancing_master_dispatch_reason_total{"
                'engineIp="127.0.0.1",'
                'reason="predicted_execution_cap",role="PREFILL"} 30.0\n'
                "flexlb_app_engine_balancing_master_dispatch_reason_total{"
                'engineIp="127.0.0.1",'
                'reason="fixed_window_timeout",role="PREFILL"} 25.0\n',
                encoding="utf-8",
            )

            result = analyze(run, None)

            self.assertEqual(125, result["decisions"]["count"])
            self.assertEqual("prometheus_counter", result["decisions"]["source"])
            self.assertEqual(
                {"fixed_window_timeout": 25, "predicted_execution_cap": 100},
                result["decisions"]["reasons"],
            )
            self.assertEqual(1, result["decisions"]["log_count"])
            self.assertEqual(0.008, result["decisions"]["log_coverage_ratio"])


if __name__ == "__main__":
    unittest.main()
