import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from workload.performance_gate import analyze, report, validate, trace_workload_sha
from workload.performance_compare import compare
from scenario import compile_scenarios, load_scenarios
from scenario.catalog import handlers

ROOT = Path(__file__).resolve().parents[1]


def evidence():
    # Small arithmetic fixture only; never a runnable benchmark.
    c = {
        "benchmark_id": "synthetic-100ms-10qps-v1",
        "warmup_s": 5,
        "measure_s": 30,
        "sample_s": 1,
        "max_gap_s": 3,
        "qps": 10,
        "qps_tolerance": 0.1,
        "min_requests": 250,
        "max_pacing_lag_ms": 100,
        "min_input_tps": 9000,
        "min_output_tps": 70,
        "min_goodput_rps": 9,
        "min_slo_fraction": 0.95,
        "max_error_rate": 0,
        "max_ttft_p99_ms": 1000,
        "max_e2e_p99_ms": 2000,
        "max_tpot_p99_ms": 150,
        "slo_ttft_ms": 1000,
        "slo_e2e_ms": 2000,
        "slo_tpot_ms": 150,
        "max_inflight_growth_rps": 0.2,
    }
    c.update(measure_s=10, min_requests=80, warmup_s=0)
    issued = []
    records = []
    for i in range(100):
        r = dict(
            rid=str(i),
            send_start_epoch_ms=100000 + i * 100,
            input_len=1024,
            output_len=8,
            pacing_lag_ms=0,
        )
        issued.append(r)
        records.append(
            dict(r, status="ok", total_ms=100, ttft_ms=50, observed_output_tokens=8)
        )
    return dict(
        schema_version=1,
        criteria=c,
        errors=[],
        window=dict(start_epoch_ms=100000, end_epoch_ms=110000),
        samples=[dict(epoch_ms=100000 + i * 1000) for i in range(11)],
        flow=dict(complete=True, errors=[], issued=issued, records=records),
        provenance=dict(
            benchmark_id=c["benchmark_id"],
            master_artifact=dict(jar_sha256="a" * 64),
            mock_jar_sha256="b" * 64,
            actual_master_config=dict(dispatcher=dict(type="BATCH")),
            performance=dict(prefill=dict(fixed_ms=100)),
            topology=dict(prefill=2, decode=4),
            capacity=dict(prefill_cache_blocks=1000, decode_cache_blocks=1000),
            trace=dict(sha256="c" * 64, workload_sha256="d" * 64),
            client_environment=dict(FETCH_OUTPUT_STREAM="true"),
            analyzer_sha256="e" * 64,
        ),
    )


class PerformanceGateTest(unittest.TestCase):
    def test_absolute_success_and_renderer(self):
        e = evidence()
        r = analyze(e)
        self.assertEqual(r["verdict"], "PASS", r)
        self.assertEqual(r["metrics"]["goodput_rps"], 10)
        with tempfile.TemporaryDirectory() as d:
            path = report(d, e)
            self.assertTrue((path / "report.html").is_file())
            spec = json.loads((path / "report-spec.json").read_text())
            self.assertEqual(len(spec["panels"]), 1)
            for panel in spec["panels"]:
                self.assertTrue(panel["overlay"])
                self.assertTrue(panel["series"][0]["points"])
                self.assertIn(panel["series"][0]["axis"], panel["axes"])
            self.assertEqual(
                analyze(
                    json.loads((Path(d) / "performance-gate-evidence.json").read_text())
                ),
                r,
            )

    def test_multiview_ab_preserves_decode_monitoring_and_request_buckets(self):
        from workload.performance_views import panel

        e = evidence()
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            archive = root / "telemetry/1/queries.json"
            archive.parent.mkdir(parents=True)
            archive.write_text(
                json.dumps(
                    dict(
                        start=100,
                        end=110,
                        step=1,
                        targets={},
                        queries={
                            "mock/engine_count": dict(
                                promql="sum by(role)(engines)",
                                result=[
                                    dict(
                                        metric=dict(role=role),
                                        values=[[100, count], [101, count]],
                                    )
                                    for role, count in [("prefill", 2), ("decode", 4)]
                                ],
                            )
                        },
                    )
                )
            )
            chart, audit = panel(root, e, analyze(e))
            self.assertTrue(audit["available"])
            self.assertEqual(len(chart["presets"]["规模"]), 2)
            by_name = {c["name"]: c for c in chart["series"]}
            self.assertEqual(
                by_name["mock · D engine count"]["points"][0], dict(x=0, y=4)
            )
            self.assertEqual(by_name["TTFT p99"]["points"][0]["y"], 50)
            self.assertEqual(by_name["到达 cohort 成功率"]["points"][0]["y"], 1)
            self.assertEqual(
                sum(p["y"] for p in by_name["完成输出 TPS"]["points"]), 792
            )
            compare(e, e, root / "ab", left_directory=root, right_directory=root)
            spec = json.loads(
                (
                    root / "ab/reports/comparison/master-performance/report-spec.json"
                ).read_text()
            )
            self.assertEqual([p["id"] for p in spec["panels"]], ["ab", "A", "B"])
            overlay = spec["panels"][0]
            self.assertEqual(len(overlay["presets"]["规模"]), 4)
            self.assertEqual(overlay["series"][0]["dash"], [6, 4])
            self.assertTrue((root / "ab/left/telemetry/1/queries.json").is_file())

    def test_tail_cohort_and_actual_tokens_not_requested_budget(self):
        e = evidence()
        for r in e["flow"]["records"]:
            r["observed_output_tokens"] = 2
        r = analyze(e)
        self.assertEqual(r["verdict"], "FAIL")
        self.assertLess(r["metrics"]["output_tps"], 20)
        # Last request finishes beyond measurement, remains in cohort SLO denominator.
        e = evidence()
        e["flow"]["records"][-1]["total_ms"] = 5000
        r = analyze(e)
        self.assertEqual(r["metrics"]["cohort_requests"], 100)
        self.assertEqual(r["metrics"]["goodput_rps"], 9.9)
        self.assertEqual(r["metrics"]["slo_fraction"], 0.99)

    def test_any_failure_including_warmup_blocks_gate(self):
        for index in (0, 99):
            e = evidence()
            e["flow"]["records"][index]["status"] = "timeout"
            self.assertEqual(analyze(e)["verdict"], "FAIL")
        e = evidence()
        for r in (e["flow"]["issued"][0], e["flow"]["records"][0]):
            r["send_start_epoch_ms"] = 99900
        e["flow"]["records"][0]["status"] = "timeout"
        self.assertEqual(analyze(e)["verdict"], "FAIL")

    def test_zero_success_is_failure_not_missing_percentile_invalid(self):
        e = evidence()
        for r in e["flow"]["records"]:
            r["status"] = "engine_error"
        self.assertEqual(analyze(e)["verdict"], "FAIL")

    def test_missing_or_corrupt_evidence_never_passes(self):
        mutations = [
            lambda e: e["flow"]["records"][0].pop("observed_output_tokens"),
            lambda e: e["provenance"].pop("mock_jar_sha256"),
            lambda e: e["provenance"].pop("analyzer_sha256"),
            lambda e: e.update(provenance=[]),
            lambda e: e.update(errors=None),
            lambda e: e["flow"].update(complete=False),
            lambda e: e["flow"]["records"].pop(),
            lambda e: e["flow"]["records"].append(e["flow"]["records"][0]),
            lambda e: e["flow"]["records"][0].update(ttft_ms=float("nan")),
            lambda e: e["flow"]["records"][0].update(status="scheduled"),
            lambda e: e.update(samples=e["samples"][:2] + e["samples"][8:]),
            lambda e: e["provenance"]["client_environment"].update(
                FETCH_OUTPUT_STREAM="false"
            ),
        ]
        for mutate in mutations:
            e = evidence()
            mutate(e)
            self.assertEqual(analyze(e)["verdict"], "INVALID")

    def test_sender_invalid_and_delayed_completion_backlog_fails(self):
        e = evidence()
        e["flow"]["issued"][1]["pacing_lag_ms"] = 1000
        self.assertEqual(analyze(e)["verdict"], "INVALID")
        e = evidence()
        for r in e["flow"]["records"][50:]:
            r["total_ms"] = 20000
        r = analyze(e)
        self.assertEqual(r["verdict"], "FAIL")
        self.assertGreater(r["metrics"]["inflight_growth_rps"], 4)

    def test_one_token_tpot_not_applicable(self):
        e = evidence()
        for r in e["flow"]["records"]:
            r["observed_output_tokens"] = 1
        e["criteria"]["min_output_tps"] = 1
        r = analyze(e)
        self.assertEqual(r["verdict"], "PASS")
        self.assertEqual(
            next(x for x in r["checks"] if x["metric"] == "tpot_p99_ms")["status"],
            "NOT_APPLICABLE",
        )

    def test_config_ab_same_binary_no_bad_version_required(self):
        left = evidence()
        right = copy.deepcopy(left)
        right["provenance"]["actual_master_config"]["dispatcher"]["type"] = "NON_BATCH"
        with tempfile.TemporaryDirectory() as d:
            r = compare(
                left, right, d, allowed=["/actual_master_config/dispatcher/type"]
            )
            self.assertTrue(r["controls"]["aligned"])
            self.assertEqual(r["verdicts"], dict(left="PASS", right="PASS"))
            right["criteria"]["min_output_tps"] = 10000
            r = compare(
                left, right, d, allowed=["/actual_master_config/dispatcher/type"]
            )
            self.assertFalse(r["controls"]["aligned"])
            self.assertEqual(r["verdicts"]["right"], "FAIL")
        with self.assertRaises(ValueError):
            compare(left, right, "unused", allowed=["/performance"])

    def test_workload_checksum_preserves_everything_except_rid(self):
        with tempfile.TemporaryDirectory() as d:
            a = Path(d) / "a"
            b = Path(d) / "b"
            a.write_text('{"rid":"left:1","il":1,"ol":2,"input_ids":[3]}\n')
            b.write_text('{"rid":"right:1","il":1,"ol":2,"input_ids":[3]}\n')
            self.assertEqual(trace_workload_sha(a), trace_workload_sha(b))
            b.write_text('{"rid":"right:1","il":1,"ol":3,"input_ids":[3]}\n')
            self.assertNotEqual(trace_workload_sha(a), trace_workload_sha(b))

    def test_contract_rejects_relaxed_success_and_compiles_both_profiles(self):
        c = evidence()["criteria"]
        c["max_error_rate"] = 0.01
        with self.assertRaises(ValueError):
            validate(c)
        with mock.patch("scenario.compiler.VICTIM_OFFSETS", (300, 301, 302)):
            plans = compile_scenarios(
                load_scenarios(ROOT / "config/scenarios/master_performance.yaml"),
                handlers=handlers(),
            )
        self.assertEqual(len(plans), 2)
        for p in plans:
            self.assertIn("performance_finish", [s["action"] for s in p["stages"]])
