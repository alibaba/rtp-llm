import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from flexlb_test_framework.scenario.actions.master import OwnedHaClient
from flexlb_test_framework.workload.compare import main
from flexlb_test_framework.workload.report import write_report


class EvidenceIntegrityTest(unittest.TestCase):
    def test_failed_scrape_is_attributed_at_observed_completion(self):
        from flexlb_test_framework.workload.report import (
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

    def test_collector_lifetime_requires_start_and_tail_coverage(self):
        from flexlb_test_framework.workload.report import audit_journals

        with tempfile.TemporaryDirectory() as d:
            directory = Path(d) / "telemetry/1"
            directory.mkdir(parents=True)
            (directory / "mock.prom").write_text("# ts=10000\nx 1\n")
            (directory / "mock-samples.jsonl").write_text(
                json.dumps(dict(sequence=1, epoch_s=10, monotonic_s=10, error=None))
                + "\n"
            )
            errors = audit_journals(
                d, ["1/mock"], 5, {"1/mock": dict(started_epoch_s=0, ended_epoch_s=20)}
            )
            self.assertTrue(any("began too late" in e["error"] for e in errors))
            self.assertTrue(any("ended too early" in e["error"] for e in errors))
            self.assertEqual(
                audit_journals(
                    d,
                    ["1/mock"],
                    5,
                    {"1/mock": dict(started_epoch_s=9, ended_epoch_s=11)},
                ),
                [],
            )

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
            result = dict(
                id="sparse",
                status="PASS",
                error=None,
                stages=[],
                workload=dict(
                    capture_metrics=True,
                    runtime_validity="VALID",
                    runtime_configuration=dict(max_sample_gap_s=5),
                ),
            )
            write_report(
                root,
                result,
                dict(
                    clock_anchor={"epoch_s": 0},
                    phases=[],
                    expected_telemetry=["1/mock"],
                ),
            )
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["workload"]["runtime_validity"], "VALID")
            self.assertEqual(result["workload"]["collection_gaps"], {})
            self.assertTrue(result["workload"]["telemetry_gaps"])

    def test_invalid_evidence_cannot_pass_or_confirm_probe(self):
        for status in (
            "PASS",
            "FINDING-CONFIRMED",
            "FINDING-RESOLVED",
            "FAIL",
            "TIMEOUT",
        ):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as d:
                result = dict(
                    id="test",
                    status=status,
                    error=None,
                    stages=[],
                    workload=dict(capture_metrics=False, runtime_validity="INVALID"),
                )
                write_report(d, result, dict(clock_anchor={"epoch_s": 0}, phases=[]))
                if status in ("FAIL", "TIMEOUT"):
                    self.assertEqual(result["status"], status)
                else:
                    self.assertEqual(result["status"], "ERROR")
                    self.assertEqual(result["workload"]["prior_status"], status)

    def test_interrupted_ha_keeps_partial_rows_and_rejects_completeness(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "client_events.jsonl").write_text('{"rid":1}\n{"rid":')
            client = OwnedHaClient(SimpleNamespace(out_dir=p))
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
            evidence = OwnedHaClient(SimpleNamespace(out_dir=p)).evidence_snapshot()
            self.assertEqual(evidence["records"], [rows[1], rows[2]])
            self.assertFalse(evidence["complete"])
            self.assertTrue(evidence["path"].endswith("client_lifecycle.jsonl"))

    def test_missing_second_master_cannot_be_valid(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            t = p / "telemetry/1"
            t.mkdir(parents=True)
            for name in ["mock", "master-A"]:
                (t / (name + ".prom")).write_text("# ts=1000\nx 1\n")
            r = dict(
                id="test",
                status="PASS",
                stages=[],
                error=None,
                workload=dict(capture_metrics=True, runtime_validity="VALID"),
            )
            e = dict(
                clock_anchor={"epoch_s": 0},
                phases=[],
                expected_telemetry=["1/mock", "1/master-A", "1/master-B"],
            )
            write_report(p, r, e)
            self.assertEqual(r["status"], "ERROR")
            self.assertEqual(r["workload"]["missing_telemetry"], ["1/master-B"])
            self.assertEqual(r["workload"]["runtime_validity"], "INVALID")

    def test_async_curves_keep_own_points_and_numeric_axis(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            for name, times in [("a", [1, 2, 12]), ("b", [1.2, 2.2, 12.2])]:
                r = dict(
                    id="test",
                    configuration_sha256="same",
                    workload={"runtime_validity": "VALID"},
                    clock_anchor={"monotonic_s": 0},
                    stages=[dict(id="load", started_s=0, finished_s=15, status="PASS")],
                    series={"metric": [[t, 3] for t in times]},
                )
                (p / (name + ".json")).write_text(json.dumps(r))
            main(
                [
                    "--baseline",
                    str(p / "a.json"),
                    "--candidate",
                    str(p / "b.json"),
                    "--out",
                    str(p / "out"),
                ]
            )
            html = (p / "out/comparison.html").read_text()
            spec = json.loads(html.split("const SPEC = ", 1)[1].split(";\n", 1)[0])
            self.assertEqual(spec["timeAxis"], {"min": 0, "max": 12.2})
            self.assertEqual(spec["timeOriginLabel"], "t=0 = 当前阶段开始")
            series = spec["panels"][0]["series"]
            self.assertEqual([x["x"] for x in series[0]["points"]], [1, 2, 12])
            self.assertEqual([x["x"] for x in series[1]["points"]], [1.2, 2.2, 12.2])

    def test_failed_sample_breaks_curve_and_disables_window_comparison(self):
        from flexlb_test_framework.workload.compare import compare
        from flexlb_test_framework.workload.report import read_series

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "telemetry/1"
            p.mkdir(parents=True)
            (p / "master-A.prom").write_text("# ts=1000\nx 3\n# ts=3000\nx 4\n")
            (p / "master-A.prom.samples.jsonl").write_text(
                json.dumps(dict(epoch_s=2, error="offline")) + "\n"
            )
            series = read_series(d, 0)
            self.assertEqual(next(iter(series.values())), [[1, 3], [2, None], [3, 4]])
            report = dict(
                id="same",
                configuration_sha256="same",
                workload={"runtime_validity": "INVALID"},
                clock_anchor={"monotonic_s": 0},
                stages=[dict(id="load", status="PASS", started_s=0, finished_s=4)],
                series=series,
            )
            row = compare(report, report)["changes"][0]
            self.assertEqual(row["status"], "MISSING_DATA")
            self.assertIsNone(row["rank_score"])

    def test_outage_exemption_is_source_and_window_specific(self):
        from flexlb_test_framework.workload.report import classify_gaps

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

    def test_stress_aggregator_never_promotes_unknown_terminal_status(self):
        import ast

        source = Path(__file__).resolve().parents[1] / "stress/aggregate_canvas_run.py"
        tree = ast.parse(source.read_text())
        fn = next(
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "is_ok"
        )
        scope = {}
        exec(
            compile(ast.Module(body=[fn], type_ignores=[]), str(source), "exec"), scope
        )
        predicate = scope["is_ok"]
        self.assertTrue(predicate(dict(status="ok", error="")))
        for status in [
            None,
            "unknown",
            "cancelled",
            "incomplete_response",
            "empty_response",
            "engine_error",
            "schedule_error",
        ]:
            self.assertFalse(predicate(dict(status=status, error="")), status)
        self.assertFalse(predicate(dict(status="ok", error="business failure")))

    def test_raw_samples_without_matching_rounds_are_incomplete(self):
        from flexlb_test_framework.workload.report import audit_journals

        with tempfile.TemporaryDirectory() as d:
            directory = Path(d) / "telemetry/1"
            directory.mkdir(parents=True)
            raw = directory / "mock.prom"
            raw.write_text("# ts=1000\nx 1\n# ts=3000\nx 3\n")
            journal = directory / "mock-samples.jsonl"
            journal.write_text(
                json.dumps(dict(sequence=1, epoch_s=1, error=None)) + "\n"
            )
            issues = audit_journals(d, ["1/mock"])
            self.assertTrue(any("do not match" in issue["error"] for issue in issues))
            with journal.open("a") as stream:
                stream.write(json.dumps(dict(sequence=2, epoch_s=3, error=None)) + "\n")
            self.assertEqual(audit_journals(d, ["1/mock"]), [])
            journal.write_text("{broken\n")
            self.assertTrue(audit_journals(d, ["1/mock"]))

    def test_invalid_evidence_is_not_ranked_even_with_visible_samples(self):
        from flexlb_test_framework.workload.compare import compare

        report = dict(
            id="same",
            configuration_sha256="same",
            workload={"runtime_validity": "INVALID"},
            clock_anchor={"monotonic_s": 0},
            stages=[dict(id="load", started_s=0, finished_s=4, status="PASS")],
            series={"metric": [[1, 3], [2, 4]]},
        )
        row = compare(report, report)["changes"][0]
        self.assertEqual(row["status"], "INVALID_EVIDENCE")
        self.assertIsNone(row["rank_score"])

    def test_silent_sampling_pause_breaks_curve_without_fabricating_zero(self):
        from flexlb_test_framework.workload.report import audit_journals, read_series

        with tempfile.TemporaryDirectory() as d:
            directory = Path(d) / "telemetry/1"
            directory.mkdir(parents=True)
            (directory / "mock.prom").write_text("# ts=1000\nx 1\n# ts=11000\nx 2\n")
            (directory / "mock-samples.jsonl").write_text(
                "".join(
                    json.dumps(dict(sequence=index, epoch_s=timestamp, error=None))
                    + "\n"
                    for index, timestamp in enumerate([1, 11], 1)
                )
            )
            points = next(iter(read_series(d, 0, 5).values()))
            self.assertEqual(points, [[1, 1], [6, None], [11, 2]])
            self.assertTrue(
                any(
                    "sampling gap" in issue["error"]
                    for issue in audit_journals(d, ["1/mock"], 5)
                )
            )
