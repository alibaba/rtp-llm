import copy
import json
import tempfile
import unittest
from pathlib import Path

from traffic.traffic_source import materialize
from traffic.workload_profile import profile
from workload.cache_gate import (
    align_send_counters,
    analyze,
    write_report,
)
from scenario import compile_scenarios, load_scenarios
from scenario.catalog import handlers

ROOT = Path(__file__).resolve().parents[1]


class CacheGateTest(unittest.TestCase):
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
                min_waiting=1,
                max_pacing_lag_ms=100,
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

    def test_reset_gap_topology_and_pacing_fail_closed(self):
        for fault in ("reset", "gap", "topology", "pacing", "rate"):
            e = self.evidence()
            if fault == "reset":
                e["samples"][40]["engines"]["p0"]["hit_tokens_total"] = 0
            if fault == "gap":
                del e["samples"][40:46]
            if fault == "topology":
                e["samples"][40]["engines"]["p1"] = copy.deepcopy(
                    e["samples"][20]["engines"]["p1"]
                )
            if fault == "pacing":
                e["max_pacing_lag_ms"] = 1000
            if fault == "rate":
                for r in e["samples"]:
                    r["started"] //= 2
            self.assertEqual(analyze(e)["verdict"], "INVALID", fault)

    def test_report_uses_same_verdict_and_data(self):
        e = self.evidence(0.1)
        with tempfile.TemporaryDirectory() as d:
            write_report(d, e, analyze(e))
            self.assertEqual(
                json.loads(
                    (Path(d) / "reports/run/cache-scale-in/analysis.json").read_text()
                )["result"]["verdict"],
                "FAIL",
            )
            self.assertIn(
                "监控聚合曲线",
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
            self.assertEqual(curves, {})
            self.assertIn("缺少监控数据", spec["panels"][0]["caption"])
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
            self.assertEqual(spec["panels"][0]["series"][0]["name"], "P Waiting / engine")
            self.assertNotIn("1/mock/", json.dumps(spec["panels"][0]["series"]))
            self.assertEqual(spec["kpis"][1]["value"], "INVALID")
            audit = spec["sections"][0]["rows"]
            self.assertIn(["Master completion QPS", "0%", "MISSING", "本次归档没有该监控序列"], audit)

    def test_ab_requires_aligned_controls(self):
        from workload.cache_gate_ab import compare

        old, new = self.evidence(0.1), self.evidence(0.8)
        for i, e in enumerate((old, new)):
            e["events"] = [{"name": "withdraw_start", "t": 20}]
            e["provenance"] = dict(
                topology={"prefill": 2, "decode": 2},
                performance={},
                master_config={},
                mock_formula_config={},
                client_environment={},
                trace={"sha256": "trace"},
                files={"/mock/flexlb-mock-engine-test.jar": "mock"},
                historical_master={"source_commit": str(i) * 40},
            )
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d) / "a.json", Path(d) / "b.json"
            a.write_text(json.dumps(old))
            b.write_text(json.dumps(new))
            result = compare(a, b, Path(d) / "report")
            self.assertTrue(result["expected_control_observed"])
            new["provenance"]["trace"]["sha256"] = "other"
            b.write_text(json.dumps(new))
            result = compare(a, b, Path(d) / "unaligned")
            self.assertFalse(result["expected_control_observed"])
            self.assertIn(
                "/trace_sha256",
                [d["path"] for d in result["control_comparison"]["differences"]],
            )
            new["provenance"]["trace"]["sha256"] = "trace"
            new.update(
                events=[],
                baseline_start=0,
                baseline_end=0,
                post_start=0,
                post_end=0,
                samples=new["samples"][:20],
                errors=["warmup timeout"],
            )
            b.write_text(json.dumps(new))
            result = compare(a, b, Path(d) / "no-withdrawal")
            self.assertEqual(result["new"]["verdict"], "INVALID")
            self.assertFalse(result["expected_control_observed"])
            self.assertIn(
                "缩容事件不完整",
                (
                    Path(d)
                    / "no-withdrawal/reports/comparison/cache-scale-in-ab/report.html"
                ).read_text(),
            )

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
        plans = compile_scenarios(
            load_scenarios(ROOT / "config/scenarios/workload/cache_scale_in.yaml"),
            handlers=handlers(),
        )
        self.assertEqual(len(plans), 1)
        self.assertEqual(
            plans[0]["environment"]["prefill_cache_policy"]["memory_tree"], False
        )
        self.assertIn("cache_scale_in_check", [s["action"] for s in plans[0]["stages"]])

    def test_staircase_validates_order_hold_and_traffic_budget(self):
        import yaml

        case = yaml.safe_load(
            (ROOT / "config/scenarios/workload/cache_scale_in.yaml").read_text()
        )
        gate = case["parameters"]["gate"]
        gate.update(intermediate_p=6, intermediate_hold_s=30)
        flow = case["parameters"]["flow"]
        flow["source"]["parameters"]["count"] = 30000
        flow["client"]["DURATION_S"] = "360"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "staircase.yaml"

            def compile_case(value):
                path.write_text(yaml.safe_dump(value))
                return compile_scenarios(
                    load_scenarios(path), handlers=handlers()
                )

            self.assertEqual(len(compile_case(case)), 1)
            for key, value in (
                ("intermediate_p", 4),
                ("intermediate_p", 8),
                ("intermediate_hold_s", 29),
            ):
                invalid = copy.deepcopy(case)
                invalid["parameters"]["gate"][key] = value
                with self.assertRaises(Exception):
                    compile_case(invalid)
            insufficient = copy.deepcopy(case)
            insufficient["parameters"]["flow"]["client"]["DURATION_S"] = "310"
            with self.assertRaisesRegex(Exception, "traffic plan must cover"):
                compile_case(insufficient)


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
