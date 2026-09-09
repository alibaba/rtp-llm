"""Elastic adapter contracts without JVMs, sockets or production changes."""

import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flexlb_test_framework.scenario.actions import elastic as e


class Deadline:
    def __init__(self, remaining=100):
        self.end = time.monotonic() + remaining

    def remaining(self):
        return self.end - time.monotonic()

    def check(self):
        if self.remaining() <= 0:
            raise TimeoutError()

    def sleep(self, seconds):
        self.check()


class Call:
    def __init__(self, value=None, error=None, frames=()):
        self.value, self.error, self.frames = value, error, frames
        self.cancelled = False

    def result(self):
        if self.error:
            raise self.error
        return self.value

    def __iter__(self):
        yield from self.frames
        if self.error:
            raise self.error

    def cancel(self):
        self.cancelled = True
        return True


def frame(finished=False, code=0):
    return NS(
        HasField=lambda name: bool(code),
        error_info=NS(error_code=code, error_message="retired"),
        flatten_output=NS(finished=[finished]),
    )


def ops(schedule_error=None, stream_error=None, frames=(), enqueued=True):
    response = NS(code=200, success=True, enqueued_by_master=enqueued)
    scheduled = Call(response, schedule_error)
    streamed = Call(error=stream_error, frames=frames)
    timeouts = []

    def future(req, timeout):
        timeouts.append(("Schedule", timeout))
        return scheduled

    def fetch(req, timeout):
        timeouts.append(("FetchResponse", timeout))
        return streamed

    result = NS(
        _channel=lambda target: None,
        master_target=lambda: "master",
        build_schedule_request=lambda rid, **shape: rid,
        prefill_addr=lambda resp: "p:1",
        next_request_id=lambda: 1,
        schedule_pb2_grpc=NS(
            FlexlbServiceStub=lambda channel: NS(Schedule=NS(future=future))
        ),
        pb2_grpc=NS(RpcServiceStub=lambda channel: NS(FetchResponse=fetch)),
        pb2=NS(FetchRequestPB=lambda **kw: kw),
    )
    return result, timeouts, scheduled, streamed


class ElasticEvidenceTests(unittest.TestCase):
    def test_lifecycle_request_keeps_separate_30s_and_10s_rpc_caps(self):
        operation, timeouts, _, _ = ops(frames=[frame(True)])
        records = e.RecordedRequests(operation, 1)
        record = records.issue(1, time.monotonic)
        records.run(record, dict(output_len=2), timeout_s=40, stream_timeout_s=10)
        self.assertEqual(timeouts[0], ("Schedule", 30))
        self.assertEqual(timeouts[1], ("FetchResponse", 10))

    def test_cold_flow_preserves_unique_key_and_independent_completion(self):
        import threading

        flow = e.ColdFlow(NS(next_request_id=lambda: 7), 1)
        called = []
        entered = threading.Event()

        def run(record, shape, **kwargs):
            called.append((shape, kwargs))
            flow.update(
                record,
                consumer_exit_s=time.monotonic(),
                transport_terminal_s=time.monotonic(),
            )
            flow._stop.set()
            entered.set()

        flow.run = run
        flow.start()
        self.assertTrue(entered.wait(2))
        result = flow.stop(Deadline(2))
        self.assertTrue(flow.done.is_set())
        self.assertEqual(called[0][0], dict(output_len=2, block_keys=[701]))
        self.assertEqual(called[0][1]["stream_timeout_s"], 10)
        self.assertTrue(result["result_complete"])
        self.assertFalse(result["zero_errors"])

    def test_flow_assert_keeps_zero_error_and_90_percent_contracts_distinct(self):
        ctx = NS(
            resource=lambda *args: dict(issued=10, completed=9, result_complete=True)
        )
        for floor, expected in [(0.9, "PASS"), (1, "FAIL")]:
            result = e._flow_assert(
                ctx, dict(result={}, min_success_rate=floor), Deadline()
            )
            self.assertEqual(result.checks[-1].status, expected)

    def test_empty_flow_never_passes_even_with_zero_success_floor(self):
        ctx = NS(
            resource=lambda *args: dict(issued=0, completed=0, result_complete=True)
        )
        result = e._flow_assert(ctx, dict(result={}, min_success_rate=0), Deadline())
        self.assertEqual(result.checks[0].status, "FAIL")
        self.assertEqual(result.checks[-1].status, "FAIL")

    def run_record(self, **kwargs):
        operation, timeouts, scheduled, streamed = ops(**kwargs)
        records = e.RecordedRequests(operation, 7)
        record = records.issue(123, time.monotonic)
        records.run(record, {}, timeout_s=5)
        return records, records.snapshot_records()[0], timeouts

    def test_finished_and_transport_success_required(self):
        records, record, timeouts = self.run_record(frames=[frame(True)])
        self.assertTrue(e.request_success(record))
        self.assertTrue(e.completeness(records.snapshot_records())["zero_errors"])
        self.assertTrue(all(0 < timeout <= 5 for _, timeout in timeouts))
        self.assertEqual(record["stream"]["method"], "FetchResponse")
        self.assertIsNone(record["endpoint_generation"])

    def test_empty_transport_is_not_business_success(self):
        records, record, _ = self.run_record()
        self.assertIsNotNone(record["transport_terminal_s"])
        self.assertFalse(e.request_success(record))
        self.assertEqual(
            e.completeness(records.snapshot_records())["failed_request_ids"], [123]
        )

    def test_typed_retirement_preserved_even_with_finished(self):
        _, record, _ = self.run_record(frames=[frame(True, 8510)])
        self.assertEqual(record["business_error_code"], 8510)
        self.assertFalse(e.request_success(record))

    def test_schedule_failure_not_misattributed_to_fetch(self):
        _, record, timeouts = self.run_record(
            schedule_error=TimeoutError("Schedule expired")
        )
        self.assertEqual(record["schedule"]["status"], "ERROR")
        self.assertIsNone(record["stream"]["method"])
        self.assertEqual(len(timeouts), 1)

    def test_fetch_failure_after_finished_remains_failure(self):
        _, record, _ = self.run_record(
            frames=[frame(True)], stream_error=TimeoutError("Fetch expired")
        )
        self.assertTrue(record["business_finished"])
        self.assertFalse(e.request_success(record))
        self.assertEqual(record["stream"]["status"], "ERROR")

    def test_records_are_deep_copies_and_nonterminal_kept(self):
        records = e.ClientRecords(1)
        record = records.issue(4, lambda: 10)
        snapshot = records.snapshot_records()
        snapshot[0]["schedule"]["status"] = "corrupt"
        self.assertIsNone(records.snapshot_records()[0]["schedule"]["status"])
        self.assertEqual(records.snapshot_cohort(10, 11)["record_count"], 1)
        self.assertEqual(records.snapshot_cohort(10, 11, "terminal")["record_count"], 0)
        self.assertFalse(e.completeness(records.snapshot_records())["result_complete"])
        self.assertFalse(e.completeness([])["zero_errors"])

    def test_cancellation_is_explicit_and_idempotent(self):
        operation, *_ = ops()
        records = e.RecordedRequests(operation, 2)
        record = records.issue(9, time.monotonic)
        call = Call()
        records._activate(record, call)
        records.cancel_active("test")
        stamp = record["cancel"]["requested_s"]
        records.cancel_active("second")
        self.assertTrue(call.cancelled)
        self.assertEqual(record["cancel"]["requested_s"], stamp)
        self.assertEqual(record["cancel"]["reason"], "test")
        self.assertFalse(e.request_success(record))

    def test_cancel_before_call_activation_is_not_lost(self):
        operation, *_ = ops()
        records = e.RecordedRequests(operation, 1)
        record = records.issue(1, time.monotonic)
        records.cancel_active()
        call = Call()
        records._activate(record, call)
        self.assertTrue(call.cancelled)

    def test_http_success_does_not_imply_drained(self):
        families = dict(hot="p0", cold="p1")
        ctx = NS(
            ops=object(),
            clock=time.monotonic,
            resource=lambda *args: families,
            register_resource=lambda *args, **kwargs: {
                "kind": "snapshot",
                "id": "1",
                "env_epoch": 1,
            },
        )
        snap = {
            "p0": {"cache_key_set": list(range(90))},
            "p1": {"cache_key_set": list(range(10))},
        }
        with patch.object(e, "_snapshot", return_value=snap), patch.object(
            e, "_http", return_value={"drained": False}
        ):
            output = e._scale(ctx, {"families": {}, "victim": "hot"}, Deadline())
        self.assertEqual(output.checks[-1].id, "pre_scale_skew")
        self.assertFalse(output.output["drained"])

    def test_scale_refuses_short_client_budget_before_side_effect(self):
        ctx = NS(resource=lambda *args: dict(hot="p0", cold="p1"))
        snap = {
            "p0": {"cache_key_set": list(range(90))},
            "p1": {"cache_key_set": list(range(10))},
        }
        with patch.object(e, "_snapshot", return_value=snap), patch.object(
            e, "_http"
        ) as http:
            with self.assertRaises(TimeoutError):
                e._scale(ctx, {"families": {}, "victim": "hot"}, Deadline(5))
            http.assert_not_called()


if __name__ == "__main__":
    unittest.main()


class ElasticMetricTests(unittest.TestCase):
    @staticmethod
    def data():
        samples = []
        for second in range(21):
            samples.append(
                dict(
                    time_s=second,
                    engines={
                        "p1": dict(
                            role="prefill",
                            mock_engine_cache_key_hits_total=second * 9,
                            mock_engine_cache_keys_requested_total=second * 10,
                            mock_engine_waiting=1,
                            mock_engine_cache_blocks=100,
                            mock_engine_available_blocks=20,
                        )
                    },
                )
            )
        return dict(samples=samples, errors=[], env_epoch=1)

    def test_hit_rate_uses_real_counter_delta(self):
        value = e.metric_window(self.data(), 0, 20, "p1")
        self.assertEqual(value["hit_rate"], 0.9)
        self.assertEqual(value["requested"], 200)
        self.assertAlmostEqual(value["occupancy_peak"], 0.8)

    def test_zero_traffic_is_error(self):
        data = self.data()
        for sample in data["samples"]:
            sample["engines"]["p1"]["mock_engine_cache_keys_requested_total"] = 0
        with self.assertRaises(ValueError):
            e.metric_window(data, 0, 20)

    def test_counter_reset_is_error(self):
        data = self.data()
        data["samples"][-1]["engines"]["p1"]["mock_engine_cache_key_hits_total"] = 0
        with self.assertRaisesRegex(ValueError, "epoch reset"):
            e.metric_window(data, 0, 20)

    def test_sample_gap_is_error(self):
        data = self.data()
        del data["samples"][5:10]
        with self.assertRaisesRegex(ValueError, "uncovered gap"):
            e.metric_window(data, 0, 20)

    def test_failed_scrape_not_silently_skipped(self):
        data = self.data()
        data["errors"].append(dict(time_s=2, error="timeout"))
        with self.assertRaisesRegex(ValueError, "acquisition failed"):
            e.metric_window(data, 0, 20)

    def test_missing_survivor_is_error(self):
        with self.assertRaisesRegex(ValueError, "survivor missing"):
            e.metric_window(self.data(), 0, 20, "p2")

    def test_parse_rejects_nan_and_duplicate_series(self):
        line = 'mock_engine_cache_key_hits_total{role="prefill",engine_name="p1"} '
        with self.assertRaises(ValueError):
            e.parse_metrics(line + "NaN")
        with self.assertRaises(ValueError):
            e.parse_metrics(line + "1\n" + line + "2")
        self.assertEqual(
            e.parse_metrics(line + "2")["p1"]["mock_engine_cache_key_hits_total"], 2
        )


class ElasticVerdictTests(unittest.TestCase):
    def test_all_independent_failures_are_reported_together(self):
        import tempfile

        flow = dict(
            issued=2,
            completed=1,
            result_complete=True,
            zero_errors=False,
            failed_request_ids=[7],
        )
        objects = dict(
            baseline=dict(hit_rate=1),
            transient=dict(hit_rate=0.1),
            steady=dict(hit_rate=0.9, waiting_peak=3, occupancy_peak=0.99),
            scale=dict(response=dict(drained=False)),
            flow_result=flow,
            recovery=e.ClientRecords(1),
        )
        with tempfile.TemporaryDirectory() as root:
            ctx = NS(
                resource=lambda value, kind: objects[value], artifact_dir=Path(root)
            )
            params = {k: k for k in objects}
            params["victim"] = "cold"
            result = e._verdict(ctx, params, Deadline())
        self.assertEqual([c.id for c in result.checks], ["PC", "PQ", "PK", "P6", "P2"])
        self.assertTrue(all(c.status == "FAIL" for c in result.checks))


class ElasticFlowTests(unittest.TestCase):
    def test_flow_stays_bounded_and_stop_accounts_for_all_issued(self):
        import itertools
        import threading

        ids = itertools.count(1)
        flow = e.BoundedFlow(
            NS(next_request_id=lambda: next(ids)), 1, [[1], [2]], interval_s=0.005
        )
        release = threading.Event()
        saturated = threading.Event()
        active = 0
        maximum = 0
        lock = threading.Lock()

        def run(record, shape):
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(maximum, active)
                if active == 2:
                    saturated.set()
            release.wait(2)
            flow.update(
                record,
                business_finished=True,
                schedule=dict(status="OK"),
                stream=dict(status="OK"),
                consumer_exit_s=time.monotonic(),
                transport_terminal_s=time.monotonic(),
            )
            with lock:
                active -= 1

        flow.run = run
        flow.start()
        try:
            self.assertTrue(saturated.wait(2))
            self.assertEqual(len(flow.snapshot_records()), 2)
            flow._stop.set()
            release.set()
            summary = flow.stop(Deadline(2))
            self.assertEqual(maximum, 2)
            self.assertTrue(summary["result_complete"])
            self.assertTrue(summary["zero_errors"])
            self.assertFalse(flow.thread.is_alive())
        finally:
            release.set()
            flow.stop(Deadline(2), cancel=True)


class ElasticBaselineTests(unittest.TestCase):
    def test_low_nonempty_hit_is_observed_without_new_threshold(self):
        from flexlb_test_framework.scenario.runtime import Deadline as RuntimeDeadline
        from test_scenario_elastic_runtime import Clock

        data = ElasticMetricTests.data()
        for sample in data["samples"]:
            sample["engines"]["p1"]["mock_engine_cache_key_hits_total"] = (
                sample["time_s"] * 5
            )
        clock = Clock()
        metrics = NS(snapshot=lambda: data, skew_started_s=0)
        ctx = NS(
            clock=clock,
            resource=lambda *args: metrics,
            register_resource=lambda *args, **kw: {},
        )
        result = e._window(
            ctx,
            dict(observation={}, phase="baseline"),
            RuntimeDeadline(30, clock, clock.sleep),
        )
        self.assertEqual(result.checks[0].id, "traffic")
        self.assertEqual(result.checks[0].status, "PASS")
        self.assertEqual(result.checks[0].evidence["hit_rate"], 0.5)


class ElasticCompletionTests(unittest.TestCase):
    def test_flow_false_is_alive_is_not_completion_evidence(self):
        flow = e.BoundedFlow(NS(), 1, [[1]])
        flow.thread = NS(ident=1, is_alive=lambda: False)
        with self.assertRaises(TimeoutError):
            flow.stop(Deadline(0.001))

    def test_flow_done_without_record_terminal_is_error(self):
        flow = e.BoundedFlow(NS(), 1, [[1]])
        flow.thread = NS(ident=1, is_alive=lambda: False)
        flow.issue(1, time.monotonic)
        flow.done.set()
        with self.assertRaisesRegex(RuntimeError, "without final consumer"):
            flow.stop(Deadline(1))

    def test_metrics_requires_done_and_persists_incomplete_evidence(self):
        import tempfile

        with tempfile.TemporaryDirectory() as root:
            ctx = NS(clock=time.monotonic, env_epoch=1, artifact_dir=Path(root))
            metrics = e.ElasticMetrics(ctx)
            metrics.thread = NS(ident=1, is_alive=lambda: False)
            with self.assertRaises(TimeoutError):
                metrics.stop(Deadline(0.001))
            import json

            evidence = json.loads((Path(root) / "elastic-metrics.json").read_text())
            self.assertFalse(evidence["complete"])
            metrics.done.set()
            metrics.stop(Deadline(1))
            evidence = json.loads((Path(root) / "elastic-metrics.json").read_text())
            self.assertTrue(evidence["complete"])


class ElasticMutationTests(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.saved = []

        def register(kind, value, **kwargs):
            self.saved.append(value)
            return dict(kind=kind, id=str(len(self.saved)), env_epoch=1)

        self.ctx = NS(
            clock=time.monotonic,
            ops=NS(),
            artifact_dir=Path(self.temp.name),
            resolve=lambda value: "p2" if isinstance(value, dict) else value,
            register_resource=register,
        )
        self.engine = dict(role="prefill", grpc_addr="127.0.0.1:12345")

    def test_add_budget_is_explicit_and_role_is_validated(self):
        handler = next(h for h in e.HANDLERS if h.name == "elastic_add")
        self.assertEqual(handler.max_dynamic_additions, 1)
        for role in ["worker", "PREFILL", None]:
            with self.assertRaises(ValueError):
                handler.validate(dict(role=role), NS(path="add"))
        with self.assertRaises(ValueError):
            handler.validate(dict(role="prefill", port=9999), NS(path="add"))

    def test_add_matches_ack_to_new_snapshot_identity(self):
        response = dict(
            status="ok", action="added", engine="p2", port=12345, http_port=12344
        )
        with patch.object(
            e, "_snapshot", side_effect=[{}, {"p2": self.engine}]
        ), patch.object(e, "_http", return_value=response) as http:
            result = e._add(self.ctx, dict(role="prefill"), Deadline())
        self.assertEqual(result.output["engine"], "p2")
        self.assertEqual(result.checks[0].status, "PASS")
        self.assertEqual(http.call_args.args[1], "add_engine")
        self.assertTrue(self.saved[0]["complete"])

    def test_wrong_add_identity_preserves_response_and_errors(self):
        import json

        response = dict(
            status="ok", action="added", engine="p2", port=12346, http_port=12345
        )
        with patch.object(
            e, "_snapshot", side_effect=[{}, {"p2": self.engine}]
        ), patch.object(e, "_http", return_value=response):
            with self.assertRaisesRegex(ValueError, "role or address mismatch"):
                e._add(self.ctx, dict(role="prefill"), Deadline())
        evidence = json.loads(
            next(Path(self.temp.name).glob("elastic-add-*.json")).read_text()
        )
        self.assertEqual(evidence["response"], response)
        self.assertFalse(evidence["complete"])

    def test_remove_resolves_target_and_keeps_false_drained(self):
        response = dict(
            status="ok",
            action="removed",
            engine="p2",
            port=12345,
            mode="graceful",
            drained=False,
        )
        with patch.object(
            e, "_snapshot", side_effect=[{"p2": self.engine}, {}]
        ), patch.object(e, "_http", return_value=response) as http:
            result = e._remove(
                self.ctx,
                dict(
                    engine={"$ref": "stages.add.output.engine"}, drain_timeout_ms=5000
                ),
                Deadline(),
            )
        self.assertEqual(
            http.call_args.args[3],
            dict(engine="p2", mode="graceful", drain_timeout_ms=5000),
        )
        self.assertEqual(result.checks[0].id, "membership")
        self.assertIs(self.saved[0]["response"]["drained"], False)

    def test_remove_checks_remaining_budget_before_mutation(self):
        with patch.object(
            e, "_snapshot", return_value={"p2": self.engine}
        ), patch.object(e, "_http") as http:
            with self.assertRaises(TimeoutError):
                e._remove(
                    self.ctx, dict(engine="p2", drain_timeout_ms=60000), Deadline(5)
                )
            http.assert_not_called()

    def test_missing_drain_evidence_is_error(self):
        response = dict(
            status="ok", action="removed", engine="p2", port=12345, mode="graceful"
        )
        with patch.object(
            e, "_snapshot", side_effect=[{"p2": self.engine}, {}]
        ), patch.object(e, "_http", return_value=response):
            with self.assertRaisesRegex(ValueError, "drain evidence"):
                e._remove(
                    self.ctx, dict(engine="p2", drain_timeout_ms=5000), Deadline()
                )

    def test_remove_timeout_must_preserve_named_contract(self):
        for cap in [True, 0, 1000, 5000.0, 600000]:
            with self.assertRaises(ValueError):
                e._remove_validate(
                    dict(engine="p2", drain_timeout_ms=cap), NS(path="remove")
                )

    def topology(self, summary):
        import json

        now = [0.0]
        self.ctx.clock = lambda: now[0]

        def sleep(seconds):
            now[0] += seconds

        deadline = NS(check=lambda: None, remaining=lambda: 35 - now[0], sleep=sleep)
        self.ctx.ops = NS(master_http_port=1)
        file = Path(self.temp.name) / "discovery.json"
        file.write_text(
            json.dumps(
                {
                    "mock.prefill.hosts.address": [
                        "127.0.0.1:12344",
                        "127.0.0.1:12346",
                        "127.0.0.1:12348",
                    ]
                }
            )
        )
        self.ctx.env = NS(discovery_file=file)
        with patch(
            "flexlb_test_framework.harness.http_post_json",
            return_value=(200, {"worker_summary": {"PREFILL": summary}}),
        ):
            return e._topology(
                self.ctx, dict(role="PREFILL", discovered=3, alive=2), deadline
            )

    def test_topology_stopped_worker_remains_discovered(self):
        result = self.topology(dict(discovered=3, alive=2))
        self.assertEqual([c.status for c in result.checks], ["PASS", "PASS"])

    def test_topology_alive_does_not_substitute_discovered(self):
        result = self.topology(dict(discovered=4, alive=2))
        self.assertEqual([c.status for c in result.checks], ["PASS", "FAIL"])

    def test_topology_missing_fields_are_not_zero(self):
        with self.assertRaisesRegex(ValueError, "lacks discovered/alive"):
            self.topology(dict(alive=2))
