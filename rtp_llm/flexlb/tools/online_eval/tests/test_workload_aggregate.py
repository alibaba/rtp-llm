import json
import tempfile
import unittest
from pathlib import Path

from flexlb_test_framework.workload.aggregate import aggregate_workload


class WorkloadAggregateTest(unittest.TestCase):
    def test_real_stress_aggregator_consumes_workload_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            env = root / "env"
            env.mkdir()
            logs = root / "owned-master-logs"
            logs.mkdir()
            (env / "flexlb_master.log").write_text(
                "console has no periodic statistics\n"
            )
            (logs / "application.log").write_text(
                "1970-01-01 00:00:01.000 flexlb_server_schedule_latency count=1 arrival_qps=1 completion_qps=1 server_p50_ms=1 server_p95_ms=1 server_p99_ms=1 grpc_queue_p95_ms=0 route_submit_p95_ms=0 batch_wait_p95_ms=0 dispatch_ack_p95_ms=0 ack_response_p95_ms=0\n"
            )
            event = dict(
                event="decode_done",
                rid=1,
                engine_name="D",
                batch_id="b",
                decode_done_ms=1200,
                exec_ms=100,
                engine_arrival_ms=1100,
                decode_start_ms=1100,
            )
            (env / "engine_events.jsonl").write_text(json.dumps(event) + "\n")
            joined = dict(
                requests=[
                    dict(
                        env_epoch="1",
                        original=dict(
                            rid=1,
                            request_id=1,
                            status="ok",
                            error="",
                            send_start_epoch_ms=1000,
                            total_ms=200,
                            wall_clock_ts=1.2,
                            input_len=10,
                            output_len=2,
                        ),
                    )
                ]
            )
            second = dict(
                joined["requests"][0]["original"],
                rid=2,
                request_id=2,
                send_start_epoch_ms=3000,
                wall_clock_ts=3.2,
            )
            joined["requests"].append(dict(env_epoch="1", original=second))
            joined["requests"].append(
                dict(
                    env_epoch="1",
                    original=dict(
                        second,
                        rid=3,
                        request_id=3,
                        purpose="preconditioning",
                        route_path="direct",
                        send_start_epoch_ms=500,
                        wall_clock_ts=0.7,
                    ),
                )
            )
            with (env / "engine_events.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(
                        dict(
                            event,
                            rid=2,
                            decode_done_ms=3200,
                            engine_arrival_ms=3100,
                            decode_start_ms=3100,
                        )
                    )
                    + "\n"
                )
            telemetry = root / "telemetry/1"
            telemetry.mkdir(parents=True)
            (telemetry / "master-single.prom").write_text("# ts=1000\nflexlb_test 1\n")
            (telemetry / "server-latency-single.json").write_text(
                json.dumps(dict(arrival_count=2, completion_count=2))
            )
            result = aggregate_workload(
                root,
                joined,
                {"1": str(env)},
                dict(epoch_s=0, monotonic_s=0),
                10,
                master_log_directories={"1": {"single": str(logs)}},
                environment_metadata={
                    "1": dict(
                        n_prefill=4,
                        n_decode=8,
                        load_client_workers=1,
                        send_mode="case program",
                    )
                },
            )
            self.assertEqual(result[0]["status"], "GENERATED", result)
            self.assertTrue(Path(result[0]["path"]).is_file())
            self.assertTrue(Path(result[0]["report"]).is_file())
            summary = json.loads(Path(result[0]["path"]).read_text())["summary"]
            self.assertIsNotNone(summary["full_e2e_latency_ms"])
            self.assertTrue(
                summary["validity_checks"]["master_arrival_matches_success"]
            )
            self.assertTrue(
                summary["validity_checks"]["master_completion_matches_success"]
            )
            self.assertEqual(summary["full_e2e_latency_ms"]["count"], 2)
            scope = json.loads(
                (Path(result[0]["path"]).parent / "request-scope.json").read_text()
            )
            self.assertEqual(scope["preconditioning_requests"], 1)
            self.assertEqual(scope["measured_requests"], 2)
            html = Path(result[0]["report"]).read_text()
            self.assertIn("4P + 8D mock", html)
            self.assertIn("case program", html)
            self.assertNotIn("500D", html)
            self.assertIn('"timeOriginLabel": "t=0 = 首个请求发出"', html)

    def test_actual_engine_address_is_joined_only_when_unambiguous(self):
        from flexlb_test_framework.workload.aggregate import client_row

        record = dict(
            wire_request_id=1,
            issued_s=1,
            business_finished=True,
            business_error_code=None,
            schedule={"status": "OK"},
            stream={"status": "OK"},
            cancel={"requested_s": None},
        )
        event = dict(event="decode_done", engine_address="D:10", cancelled=False)
        self.assertEqual(
            client_row(record, dict(epoch_s=0, monotonic_s=0), [event])["decode"],
            "D:10",
        )
        ambiguous = [event, dict(event, engine_address="D:20")]
        self.assertNotIn(
            "decode", client_row(record, dict(epoch_s=0, monotonic_s=0), ambiguous)
        )
