import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from tests.ha_fixtures import client_resource
from workload.evidence_analysis import analyze_report as read_analysis

def analyze_report(directory, result, evidence):
    from monitoring.metric_store import export_metrics
    export_metrics(directory)
    return read_analysis(directory, result, evidence)


def executed_result(status="PASS", **values):
    check_status = "FAIL" if status in {"FAIL", "FINDING-CONFIRMED"} else "PASS"
    result = dict(id="test", status=status, error=None, cleanup=[], stages=[dict(
        id="check", status="TIMEOUT" if status == "TIMEOUT" else check_status,
        error="stage deadline exceeded" if status == "TIMEOUT" else None,
        checks=[dict(id="criterion", status=check_status)],
    )])
    if status == "FINDING-CONFIRMED":
        result["finding_confirmed"] = ["check.criterion"]
    elif status == "FINDING-RESOLVED":
        result["finding_resolved"] = ["check.criterion"]
    result.update(values)
    return result


class EvidenceIntegrityTest(unittest.TestCase):
    def test_optional_missing_curve_is_diagnostic_but_required_curve_invalidates(self):
        with tempfile.TemporaryDirectory() as d:
            telemetry = Path(d) / "telemetry" / "1"
            telemetry.mkdir(parents=True)
            queries = dict(start=1, end=2, step=1, targets={}, queries={
                "mock/running_avg": dict(promql="running", result=[dict(
                    metric={"role": "prefill"}, values=[[1, "1"]]
                )])
            }, missing_queries=["master-single/optional"], errors=[])
            path = telemetry / "queries.json"
            path.write_text(json.dumps(queries))
            def result():
                return executed_result(workload=dict(capture_metrics=True,
                                          runtime_validity="VALID"))
            evidence = dict(clock_anchor={"epoch_s": 0}, phases=[],
                            expected_telemetry=["1/mock"])
            report = result()
            analyze_report(d, report, evidence)
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["workload"]["runtime_validity"], "VALID")
            self.assertEqual(report["workload"]["telemetry_completeness"], "PARTIAL")
            self.assertEqual(len(report["workload"]["telemetry_diagnostics"]), 1)
            queries["errors"] = [dict(query="mock/running_avg",
                                     error="required monitor series absent")]
            path.write_text(json.dumps(queries))
            report = result()
            analyze_report(d, report, evidence)
            self.assertEqual(report["status"], "ERROR")
            self.assertEqual(report["workload"]["runtime_validity"], "INVALID")

    def test_failed_scrape_is_attributed_at_observed_completion(self):
        from workload.evidence_analysis import (
            classify_gaps,
            collection_gaps,
            journal_rows,
        )

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "telemetry/1/master-A.prom.samples.jsonl"
            p.parent.mkdir(parents=True)
            rows = [
                dict(epoch_s=9.995, ended_epoch_s=10.01, error="reset"),
                dict(epoch_s=9.8, ended_epoch_s=9.9, error="refused"),
            ]
            p.write_text("".join(json.dumps(r) + "\n" for r in rows))
            gaps = collection_gaps(d, 0, 5)
            expected, unexpected = classify_gaps(
                gaps,
                dict(
                    clock_anchor=dict(epoch_s=0),
                    expected_outages=[
                        dict(source="1/master-A", started_epoch_s=10, ended_epoch_s=20)
                    ],
                ),
            )
            self.assertEqual(expected, {"1/master-A/collection": [10.01]})
            self.assertEqual(unexpected, {"1/master-A/collection": [9.9]})
            p.write_text(json.dumps(dict(epoch_s=10, ended_epoch_s=9)))
            self.assertTrue(journal_rows(p)[1])

    def test_sparse_metric_does_not_claim_source_outage(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            telemetry = root / "telemetry/1"
            telemetry.mkdir(parents=True)
            raw, journal = [], []
            for t in range(1, 13):
                raw.append(f"# ts={t * 1000}\nalways 1\n")
                if t in (1, 10):
                    raw.append("dynamic 1\n")
                journal.append(
                    json.dumps(dict(sequence=t, epoch_s=t, monotonic_s=t, error=None))
                )
            (telemetry / "mock.prom").write_text("".join(raw))
            (telemetry / "mock-samples.jsonl").write_text("\n".join(journal) + "\n")
            result = executed_result(
                id="sparse",
                workload=dict(
                    capture_metrics=True,
                    runtime_validity="VALID",
                    runtime_configuration=dict(max_sample_gap_s=5),
                ),
            )
            analyze_report(
                root,
                result,
                dict(
                    clock_anchor={"epoch_s": 0},
                    phases=[],
                    expected_telemetry=["1/mock"],
                ),
            )
            self.assertEqual(result["status"], "ERROR")
            self.assertEqual(result["workload"]["runtime_validity"], "INVALID")
            self.assertEqual(result["workload"]["collection_gaps"], {})
            self.assertIn("1/mock", result["workload"]["missing_telemetry"])

    def test_invalid_evidence_cannot_pass_or_confirm_probe(self):
        for status in (
            "PASS",
            "FINDING-CONFIRMED",
            "FINDING-RESOLVED",
            "FAIL",
            "TIMEOUT",
        ):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as d:
                result = executed_result(
                    status=status,
                    workload=dict(capture_metrics=False, runtime_validity="INVALID"),
                )
                analyze_report(d, result, dict(clock_anchor={"epoch_s": 0}, phases=[]))
                self.assertEqual(result["status"], "TIMEOUT" if status == "TIMEOUT" else "ERROR")
                self.assertEqual(result["outcome"]["validity"], "INVALID")
                self.assertEqual(result["outcome"]["execution"], "TIMEOUT" if status == "TIMEOUT" else "PASS")
                self.assertEqual(result["outcome"]["gate"], "PASS" if status == "TIMEOUT" else status)

    def test_interrupted_ha_keeps_partial_rows_and_rejects_completeness(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "client_events.jsonl").write_text('{"rid":1}\n{"rid":')
            client = client_resource(out_dir=p)
            evidence = client.evidence_snapshot()
            self.assertEqual(evidence["records"], [{"rid": 1}])
            self.assertFalse(evidence["complete"])
            self.assertEqual(len(evidence["errors"]), 2)

    def test_interrupted_ha_keeps_live_terminal_and_unfinished_requests(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            rows = [
                dict(
                    rid="a",
                    sequence=1,
                    event="issued",
                    send_start_epoch_ms=1000,
                    recorded_epoch_ms=1000,
                ),
                dict(
                    rid="a",
                    sequence=2,
                    event="terminal",
                    send_start_epoch_ms=1000,
                    recorded_epoch_ms=1001,
                    status="ok",
                    route_path="master",
                    failover=False,
                ),
                dict(
                    rid="b",
                    sequence=3,
                    event="issued",
                    send_start_epoch_ms=1002,
                    recorded_epoch_ms=1002,
                ),
            ]
            (p / "client_lifecycle.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows)
            )
            evidence = client_resource(out_dir=p).evidence_snapshot()
            self.assertEqual(evidence["records"], [rows[1], rows[2]])
            self.assertFalse(evidence["complete"])
            self.assertTrue(evidence["path"].endswith("client_lifecycle.jsonl"))

    def test_missing_second_master_cannot_be_valid(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            t = p / "telemetry/1"
            t.mkdir(parents=True)
            (t / "queries.json").write_text(
                json.dumps(
                    dict(
                        backend="prometheus",
                        start=1,
                        end=2,
                        step=1,
                        targets={"mock": "http://mock", "master-A": "http://master"},
                        errors=[],
                        queries={
                            name
                            + "/up": dict(
                                promql="up",
                                result=[
                                    dict(
                                        metric={"job": name, "__name__": "up"},
                                        values=[[1, "1"], [2, "1"]],
                                    )
                                ],
                            )
                            for name in ("mock", "master-A")
                        },
                    )
                )
            )
            r = executed_result(
                workload=dict(capture_metrics=True, runtime_validity="VALID"),
            )
            e = dict(
                clock_anchor={"epoch_s": 0},
                phases=[],
                expected_telemetry=["1/mock", "1/master-A", "1/master-B"],
            )
            analyze_report(p, r, e)
            self.assertEqual(r["status"], "ERROR")
            self.assertEqual(r["workload"]["missing_telemetry"], ["1/master-B"])
            self.assertEqual(r["workload"]["runtime_validity"], "INVALID")

    def test_outage_exemption_is_source_and_window_specific(self):
        from workload.evidence_analysis import classify_gaps

        evidence = dict(
            clock_anchor={"epoch_s": 100},
            phases=[],
            expected_outages=[
                dict(source="1/master-A", started_epoch_s=102, ended_epoch_s=104)
            ],
        )
        gaps = {"1/master-A/x": [1, 3, 5], "1/master-B/x": [3]}
        expected, unexpected = classify_gaps(gaps, evidence)
        self.assertEqual(expected, {"1/master-A/x": [3]})
        self.assertEqual(unexpected, {"1/master-A/x": [1, 5], "1/master-B/x": [3]})
