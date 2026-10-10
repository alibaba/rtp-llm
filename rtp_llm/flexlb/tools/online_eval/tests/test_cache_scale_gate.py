import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from traffic.traffic_source import materialize
from traffic.workload_profile import profile
from cases.cache_scale_in.analysis import align_send_counters, analyze
from cases.cache_scale_in.report import prepare_report as read_report
from cases.cache_scale_in.publication import publish_cache as write_report
from reporting.view_config import view
from scenario import compile_scenarios, load_scenarios
from scenario.catalog import handlers

ROOT = Path(__file__).resolve().parents[1]




def prepare_report(directory, evidence):
    from monitoring.metric_store import export_metrics
    from monitoring.query_plan import load_plan
    export_metrics(directory, load_plan("cache_scale_in.yaml"))
    return read_report(directory, evidence)


def metric_spec(directory, evidence, result, prepared):
    from cases.cache_scale_in.metrics import produce
    from cases.cache_scale_in.report import build_spec
    produce(directory, evidence, result)
    return build_spec(directory, evidence, result, prepared)


class CacheGateTest(unittest.TestCase):
    def test_derived_curve_label_comes_from_yaml_presentation(self):
        template = copy.deepcopy(view("cache_scale_in_overview.yaml"))
        template["charts"]["curves"]["derived/survivor_hit_ratio"]["name"] = "YAML survivor"
        prepared = dict(curves=[], audit=[], sources={}, gaps={}, errors=[],
                        monitoring_status="OK", monitor_warnings=[])
        evidence = self.evidence()
        with tempfile.TemporaryDirectory() as d, mock.patch(
            "cases.cache_scale_in.report.view", return_value=template
        ):
            spec = metric_spec(d, evidence, analyze(evidence), prepared)
        self.assertEqual(spec["panels"][0]["series"][0]["name"], "YAML survivor")

    def test_dispatch_query_has_explicit_yaml_presentation(self):
        with tempfile.TemporaryDirectory() as d:
            archive = Path(d) / "telemetry/1/queries.json"
            archive.parent.mkdir(parents=True)
            archive.write_text(json.dumps({
                "missing_queries": [], "start": 1000, "end": 1001, "step": 1,
                "targets": {}, "errors": [],
                "queries": {"master-a/dispatch_qps": {
                    "promql": "rate(dispatch_total[10s])",
                    "result": [{"metric": {"reason": "cache"},
                                "values": [[1000, "3"]]}],
                }},
            }))
            result = prepare_report(d, self.evidence())
        dispatch = [curve for curve in result["curves"]
                    if curve["metric_id"] == "master/dispatch_qps"]
        self.assertEqual(len(dispatch), 1)
        self.assertEqual(dispatch[0]["name"], "Master dispatch QPS · cache")
        self.assertEqual(dispatch[0]["points"][0]["y"], 3)

    def test_report_layout_follows_yaml_view(self):
        template = copy.deepcopy(view("cache_scale_in_overview.yaml"))
        template["charts"]["panels"][0]["title"] = "YAML panel"
        template["charts"]["panels"][0]["curve_ids"] = ["mock/engine_count"]
        prepared = dict(
            curves=[dict(curve_id="mock/engine_count", metric_id="mock/engine_count", name="Renamed engine count", group="规模", axis="count",
                         points=[dict(x=0, y=2)])],
            audit=[], sources={}, gaps={}, errors=[], monitoring_status="OK",
            monitor_warnings=[],
        )
        with tempfile.TemporaryDirectory() as d, mock.patch(
            "cases.cache_scale_in.report.view", return_value=template
        ):
            evidence = self.evidence()
            spec = metric_spec(d, evidence, analyze(evidence), prepared)
        self.assertEqual(spec["title"], "cache_scale_in : default : batch-window")
        self.assertEqual(spec["panels"][0]["title"], "YAML panel")
        self.assertEqual([s["name"] for s in spec["panels"][0]["series"]], ["Renamed engine count"])
        self.assertEqual([p["id"] for p in spec["panels"]], [
            "cache-hit", "client-qps", "prefill-queue", "prefill-forward",
            "prefill-tps", "prefill-qps", "client-latency",
        ])
        self.assertTrue(all("presets" not in p for p in spec["panels"]))

    def test_split_panels_keep_metric_values_and_missing_annotations(self):
        from cases.cache_scale_in.report import report_panels
        names = ["P cache hit ratio", "P engine count", "Client sent QPS",
                 "Client success QPS", "Client error QPS", "P Waiting / engine"]
        metric_ids = ["mock/cache_hit_ratio", "mock/engine_count", "client/actual_send_qps",
                      "client/success_qps", "client/error_qps", "mock/waiting_avg"]
        curves = [dict(curve_id=metric_id, metric_id=metric_id, name=name,
                       points=[dict(x=5, y=None), dict(x=6, y=2)])
                  for metric_id, name in zip(metric_ids, names)]
        panels = report_panels(curves, view("cache_scale_in_overview.yaml"))
        self.assertEqual([s["name"] for s in panels[0]["series"]], names[:2])
        self.assertEqual([s["name"] for s in panels[1]["series"]], names[2:5] + names[1:2])
        self.assertEqual([s["name"] for s in panels[2]["series"]],
                         ["P Waiting / engine", "P engine count"])
        self.assertEqual(panels[0]["series"][1]["points"], panels[1]["series"][-1]["points"])
        self.assertEqual(panels[2]["series"][0]["points"], curves[-1]["points"])
        self.assertIsNone(panels[0]["series"][0]["points"][0]["y"])
        for panel, axis in zip(panels, ("ratio", "qps")):
            self.assertEqual("left", panel["axes"][axis]["position"])
            self.assertEqual("right", panel["axes"]["count"]["position"])
        sparse = report_panels(curves[:2], view("cache_scale_in_overview.yaml"))
        self.assertIn("Client error QPS", sparse[1]["caption"])

    def evidence(self, hit=0.8):
        rows = []
        for t in range(81):
            engines = {}
            for name in ["p0", "p1"] if t <= 20 else ["p0"]:
                engines[name] = dict(
                    grpc_addr=name,
                    hit_tokens_total=800 * min(t, 20) + hit * 1000 * max(t - 20, 0),
                    context_tokens_total=1000 * t,
                    context_requests_total=10 * t,
                    cache_evictions=t,
                    waiting=100 if t > 20 else 0,
                    running=1,
                    prefill_ms_avg=200,
                )
            rows.append(
                dict(
                    t=t,
                    epoch_s=t + 1000,
                    engines=engines,
                    started=t * 10,
                    terminal=t * 8,
                    master_p=2 if t < 20 else 1,
                    waiting=100,
                    running=1,
                )
            )
        return dict(
            provenance=dict(instance="cache_scale_in::default::batch-window"),
            samples=rows,
            criteria=dict(
                max_gap_s=2,
                baseline_min_hit=0.7,
                min_completed=10,
                qps=10,
                qps_tolerance=0.1,
                absolute_min_hit=0.4,
                max_drop=0.25,
                observe_s=60,
                window_s=10,
                step_s=1,
                sustain_s=15,
            ),
            baseline_start=0,
            baseline_end=20,
            initial_engines=["p0", "p1"],
            survivors=["p0"],
            post_start=20,
            post_end=80,
            max_pacing_lag_ms=0,
            events=[],
        )

    def test_healthy_overload_and_removed_counters(self):
        result = analyze(self.evidence())
        self.assertEqual(result["verdict"], "PASS")
        self.assertAlmostEqual(result["windows"][-1]["hit"], 0.8)
        self.assertEqual(result["windows"][-1]["evictions"], 10)

    def test_buffered_journal_does_not_distort_actual_send_rate(self):
        evidence = self.evidence()
        evidence["samples"][40]["started"] -= 120
        self.assertEqual(analyze(evidence)["verdict"], "INVALID")
        issued = [
            dict(send_start_epoch_ms=1000000 + (i + 0.5) * 100) for i in range(800)
        ]
        align_send_counters(evidence, issued)
        self.assertEqual(evidence["samples"][40]["journal_observed_started"], 280)
        self.assertEqual(analyze(evidence)["verdict"], "PASS")
        # A real sender gap must still invalidate the load, unlike a read delay.
        align_send_counters(evidence, issued[:300] + issued[380:])
        self.assertEqual(analyze(evidence)["verdict"], "INVALID")

    def test_persistent_collapse_fails(self):
        result = analyze(self.evidence(0.1))
        self.assertEqual(result["verdict"], "FAIL")
        self.assertEqual(result["first_collapse_s"], 45)

    def test_short_dip_is_not_sustained_collapse(self):
        e = self.evidence()
        for r in e["samples"]:
            t = r["t"]
            for engine in r["engines"].values():
                engine["hit_tokens_total"] -= 700 * min(max(t - 30, 0), 3)
        self.assertEqual(analyze(e)["verdict"], "PASS")

    def test_no_completions_is_invalid_not_healthy(self):
        e = self.evidence()
        for r in e["samples"][21:]:
            for engine in r["engines"].values():
                engine.update(
                    hit_tokens_total=16000,
                    context_tokens_total=20000,
                    context_requests_total=200,
                )
        self.assertEqual(analyze(e)["verdict"], "INVALID")

    def test_missing_cache_data_and_off_target_load_are_invalid(self):
        for fault in ("reset", "gap", "missing-survivor", "rate"):
            e = self.evidence()
            if fault == "reset":
                e["samples"][40]["engines"]["p0"]["hit_tokens_total"] = 0
            if fault == "gap":
                del e["samples"][40:46]
            if fault == "missing-survivor":
                del e["samples"][40]["engines"]["p0"]
            if fault == "rate":
                for r in e["samples"]:
                    r["started"] //= 2
            self.assertEqual(analyze(e)["verdict"], "INVALID", fault)

    def test_missing_diagnostic_counters_do_not_hide_valid_hit_window(self):
        evidence = self.evidence(0.1)
        for row in evidence["samples"]:
            row["terminal"] = None
            row["engines"]["p0"].pop("cache_evictions")
            row["engines"]["p0"].pop("waiting")
            row["engines"]["p0"].pop("prefill_ms_avg")
        result = analyze(evidence)
        self.assertEqual(result["verdict"], "FAIL")
        self.assertIsNone(result["windows"][0]["evictions"])
        self.assertIsNone(result["windows"][0]["terminal_qps"])
        self.assertIsNone(result["windows"][0]["waiting"])
        self.assertTrue(result["windows"][0]["diagnostic_gaps"])

    def test_report_uses_same_verdict_and_data(self):
        e = self.evidence(0.1)
        with tempfile.TemporaryDirectory() as d:
            write_report(d, e, analyze(e))
            from reporting import discover_reports

            self.assertEqual(discover_reports(d, role="gate"), [
                str((Path(d) / "reports/run/cache-scale-in/report.html").resolve())
            ])
            self.assertEqual(
                json.loads(
                    (Path(d) / "reports/run/cache-scale-in/analysis.json").read_text()
                )["result"]["verdict"],
                "FAIL",
            )
            self.assertIn(
                "Prefill 缓存命中率 · P 数量",
                (Path(d) / "reports/run/cache-scale-in/report.html").read_text(),
            )

    def test_report_includes_warmup_and_survivor_transition(self):
        e = self.evidence(0.1)
        result = analyze(e)
        with tempfile.TemporaryDirectory() as d:
            write_report(d, e, result)
            html = (Path(d) / "reports/run/cache-scale-in/report.html").read_text()
            spec, _ = json.JSONDecoder().raw_decode(html.split("const SPEC = ", 1)[1])
            curves = {s["name"]: s["points"] for s in spec["panels"][0]["series"]}
            self.assertEqual(curves, {"Survivor window hit ratio":
                                     [dict(x=w["end"], y=w["hit"]) for w in result["windows"]]})
            self.assertIn("缺少监控序列", spec["panels"][0]["caption"])
            self.assertNotIn("P cache hit ratio", curves)
            self.assertEqual(spec["summary"]["kpis"][1]["value"], "WARN")
            self.assertIn("缺少 Prometheus queries.json 归档",
                          html)
            self.assertEqual(analyze(e), result)

    def test_report_uses_friendly_monitor_labels_and_audits_missing_series(self):
        e = self.evidence(0.8)
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            telemetry = root / "telemetry" / "1"
            telemetry.mkdir(parents=True)
            epoch = e["samples"][0]["epoch_s"]
            (telemetry / "queries.json").write_text(json.dumps({
                "missing_queries": ["master-single/completions_qps"],
                "start": epoch, "end": epoch + 1, "step": 1,
                "targets": {}, "target_bounds": {}, "errors": [],
                "queries": {"mock/waiting_avg": {
                    "promql": "avg by (role) (rtp_llm_wait_stream_size)",
                    "result": [{"metric": {"role": "prefill"},
                                "values": [[epoch, "12"]]}],
                }},
            }))
            spec = write_report(root, e, analyze(e))
            self.assertEqual([s["name"] for s in spec["panels"][0]["series"]],
                             ["Survivor window hit ratio"])
            self.assertEqual(spec["panels"][2]["id"], "prefill-queue")
            self.assertEqual([s["name"] for s in spec["panels"][2]["series"]],
                             ["P Waiting / engine"])
            self.assertIn("P Waiting / engine", json.dumps(spec["sections"]))
            self.assertNotIn("1/mock/", json.dumps(spec["panels"][0]["series"]))
            self.assertEqual(spec["kpis"][1]["value"], "WARN")
            audit = spec["sections"][0]["rows"]
            self.assertIn(["Master schedule response QPS", "0%", "MISSING", "本次归档没有该监控序列"], audit)

    def test_formula_shared_by_master_and_mock_envelope(self):
        from flexlb_cfg import ConfigOverride, render_env, render_process_config

        expression = "12 + 0.025 * sum(computeTokens) + 0.001 * sum(hitCacheTokens)"
        override = ConfigOverride(prefill_expression=expression)
        config = json.loads(render_env("single-nonbatch", override))
        envelope = json.loads(render_process_config("single-nonbatch", override))
        envs = dict(envelope["zone_process_setting"]["process_info"]["envs"])
        self.assertEqual(json.loads(envs["FLEXLB_CONFIG"]), config)
        self.assertEqual(
            config["router"]["roles"]["prefill"]["executionTimeEstimator"][
                "expression"
            ],
            expression,
        )

    def test_real_scenario_compiles_and_preserves_cache_policy(self):
        with mock.patch("scenario.compiler.VICTIM_OFFSETS", (700, 701, 702)):
            plans = compile_scenarios(
                load_scenarios(ROOT / "config/scenarios/cache_scale_in.yaml"),
                handlers=handlers(),
            )
        self.assertEqual(len(plans), 2)
        self.assertEqual(
            plans[0]["environment"]["prefill_cache_policy"]["memory_tree"], False
        )
        self.assertIn("cache_scale_in_check", [s["action"] for s in plans[0]["stages"]])

    def test_topology_budget_is_independent_of_drain(self):
        import yaml
        from cases.config import configure_program
        case = yaml.safe_load((ROOT / "config/scenarios/cache_scale_in.yaml").read_text())
        for mode in ("graceful",):
            changed = copy.deepcopy(case)
            changed["parameters"]["gate"].update(topology_timeout_s=31, removal_mode=mode)
            with mock.patch("scenario.compiler.VICTIM_OFFSETS", (700, 701, 702)):
                plans = compile_scenarios([("test", configure_program(changed, "test"))], handlers=handlers())
            self.assertEqual(len(plans), 2)
        changed["parameters"]["gate"]["removal_mode"] = "silent"
        with self.assertRaisesRegex(Exception, "separate case"):
            compile_scenarios([("test", configure_program(changed, "test"))], handlers=handlers())

    def test_drain_outcome_does_not_change_survivor_windows(self):
        evidence = self.evidence()
        before = analyze(evidence)
        evidence.setdefault("errors", []).append("graceful drain timed out; removal introduced request loss")
        evidence["removals"] = [dict(drained=False, remaining_work={"owners": 12})]
        after = analyze(evidence)
        self.assertEqual(after["verdict"], before["verdict"])
        self.assertEqual(after["windows"], before["windows"])
        self.assertEqual(len(after["excluded_drain_diagnostics"]), 1)
        evidence["errors"].append("real collection failure")
        self.assertEqual(analyze(evidence)["verdict"], "INVALID")

    def test_failures_and_removal_diagnostics_do_not_decide_cache_gate(self):
        from cases.cache_scale_in.analysis import MEASUREMENT_POLICY, attribute_client, topology_ready
        evidence = self.evidence()
        evidence["measurement_policy"] = MEASUREMENT_POLICY
        for row in evidence["samples"]:
            for engine in row["engines"].values():
                engine["admission_open"] = 1
            if row["t"] > 20:
                row["engines"]["p0"]["waiting"] = 0
                row["engines"]["p1"] = dict(row["engines"]["p0"], grpc_addr="p1",
                                                waiting=9999, admission_open=0)
        self.assertTrue(topology_ready(evidence["samples"][-1], ["p0"], ["p0", "p1"]))
        evidence["events"] = [dict(id="withdraw_start", epoch_s=1020)]
        records = [dict(rid=1, prefill="p1", status="exception", error="removed",
                        send_start_epoch_ms=1021000, total_ms=200)]
        attribute_client(evidence, dict(complete=True, records=records))
        evidence["removals"] = [dict(engine="p1", admission=dict(admission_open=0, admission_closed_epoch_ms=1020000,
                admitted_rpcs_total=42, admitted_rpcs_at_close=42), drained=False)]
        result = analyze(evidence)
        self.assertEqual(result["verdict"], "PASS")
        self.assertEqual(result["windows"][0]["waiting"], 0)
        self.assertGreater(result["windows"][0]["client_cohorts"]["removed"]["failed_qps"], 0)
        evidence["removals"][0]["admission"]["admitted_rpcs_total"] += 1
        self.assertEqual(analyze(evidence)["verdict"], "PASS")
        evidence["removals"][0]["admission"]["admitted_rpcs_total"] -= 1
        records[0]["send_start_epoch_ms"] = 1019000
        attribute_client(evidence, dict(complete=True, records=records))
        self.assertTrue(evidence["client_attribution"]["removed_failures_without_intervention"])
        self.assertEqual(analyze(evidence)["verdict"], "PASS")
        records[0]["send_start_epoch_ms"] = 1021000
        records[0]["prefill"] = ""
        attribute_client(evidence, dict(complete=True, records=records))
        self.assertEqual(analyze(evidence)["verdict"], "PASS")
        records[0]["prefill"] = "p0"
        attribute_client(evidence, dict(complete=True, records=records))
        self.assertEqual(analyze(evidence)["verdict"], "PASS")
        attribute_client(evidence, dict(complete=False, records=records))
        self.assertEqual(analyze(evidence)["verdict"], "PASS")
        evidence["samples"][-1]["engines"]["p1"]["admission_open"] = 1
        self.assertFalse(topology_ready(evidence["samples"][-1], ["p0"], ["p0", "p1"]))
        self.assertEqual(analyze(evidence)["verdict"], "PASS")
        evidence["max_pacing_lag_ms"] = 1000
        self.assertEqual(analyze(evidence)["verdict"], "PASS")

    def test_reinterpretation_preserves_frozen_input_and_requires_new_output(self):
        import sys
        from cases.cache_scale_in.replay import cache_main as main
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "evidence.json"
            source.write_text(json.dumps(self.evidence()))
            frozen = source.read_bytes()
            destination = Path(directory) / "new"
            argv = ["cache_gate", str(source), "--reinterpret", "--output", str(destination)]
            with mock.patch.object(sys, "argv", argv):
                self.assertEqual(main(), 0)
            self.assertEqual(source.read_bytes(), frozen)
            result = json.loads((destination / "reports/run/cache-scale-in/analysis.json").read_text())
            self.assertEqual(result["result"]["verdict"], "PASS")
            with mock.patch.object(sys, "argv", argv), self.assertRaises(SystemExit):
                main()

    def test_staircase_requires_a_separate_case_identity(self):
        import yaml

        case = yaml.safe_load(
            (ROOT / "config/scenarios/cache_scale_in.yaml").read_text()
        )
        gate = case["parameters"]["gate"]
        gate.update(intermediate_p=72, intermediate_hold_s=60)
        flow = case["parameters"]["flow"]
        flow["source"]["parameters"]["count"] = 170000
        flow["client"]["DURATION_S"] = "700"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "staircase.yaml"

            def compile_case(value):
                path.write_text(yaml.safe_dump(value))
                with mock.patch("scenario.compiler.VICTIM_OFFSETS", (700, 701, 702)):
                    return compile_scenarios(
                        load_scenarios(path), handlers=handlers()
                    )

            with self.assertRaisesRegex(ValueError, "separate case"):
                compile_case(case)



class WorkloadTest(unittest.TestCase):
    def spec(self):
        return dict(
            kind="synthetic",
            model="realistic",
            version="1",
            parameters=dict(
                seed=42,
                count=80,
                block_size=1024,
                families=4,
                shared_blocks=1,
                prefix_blocks=2,
                suffix_blocks=1,
                zipf_alpha=0.7,
                cold_fraction=0.1,
                session_requests=3,
                session_growth_blocks=1,
                output_tokens=8,
                priority=50,
            ),
        )

    def test_determinism_prefix_sharing_and_unique_tails(self):
        with tempfile.TemporaryDirectory() as d:
            a = materialize(Path(d) / "a.jsonl", self.spec(), "g", d)
            b = materialize(Path(d) / "b.jsonl", self.spec(), "g", d)
            self.assertEqual(a.read_bytes(), b.read_bytes())
            rows = [json.loads(l) for l in a.read_text().splitlines()]
            self.assertEqual(
                len({tuple(r["input_token_blocks"][-1:]) for r in rows}), len(rows)
            )
            hot = [r for r in rows if r["family"] != "cold"]
            self.assertEqual(len({tuple(r["input_token_blocks"][:1]) for r in hot}), 1)
            self.assertGreater(len({r["family"] for r in hot}), 1)
            self.assertEqual(rows[-1]["ts"], 79)

    def test_workload_profile_exposes_capacity_loss_and_potential(self):
        with tempfile.TemporaryDirectory() as d:
            path = materialize(Path(d) / "input.jsonl", self.spec(), "g", d)
            result = profile(path, (1, 10000))
            self.assertLess(
                result["lru_hit_by_capacity"]["1"],
                result["lru_hit_by_capacity"]["10000"],
            )
            self.assertEqual(
                result["lru_hit_by_capacity"]["10000"],
                result["infinite_cache_potential_hit"],
            )
            self.assertEqual(result["requests"], 80)

    def test_production_512_block_source_is_published_and_profiled(self):
        with tempfile.TemporaryDirectory() as d:
            spec = self.spec()
            spec["parameters"]["block_size"] = 512
            path = materialize(Path(d) / "512.jsonl", spec, "g", d)
            row = json.loads(path.read_text().splitlines()[0])
            self.assertEqual(row["cache_key_block_size"], 512)
            self.assertEqual(row["il"], len(row["input_token_blocks"]) * 512)
            self.assertEqual(profile(path)["requests"], 80)

    def test_reject_nan_and_wrong_block_before_publication(self):
        with tempfile.TemporaryDirectory() as d:
            for key, value in [
                ("qps", float("nan")),
                ("block_size", 513),
                ("families", 0),
            ]:
                spec = self.spec()
                spec["parameters"][key] = value
                with self.assertRaises(ValueError):
                    materialize(Path(d) / "out.jsonl", spec, "g", d)
                self.assertFalse((Path(d) / "out.jsonl").exists())
