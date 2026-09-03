"""compare_twin.py 单元与 fixture 测试。

覆盖：
  * 数学单元：Wasserstein-1 / KS / gini / nearest-rank percentile 手算对照
  * 加载层：prom 长表/宽表/混合行解析、counter 嗅探与差分、指标名匹配
  * 自洽断言：mock 侧数据经 synthesize_real_inputs 喂成 real 侧输入，
    全部可比指标距离≈0、ALIGNED（twin 工具链的往返无损性）
  * 偏差 fixture：e2e / TPS / 命中率 / dispatch-reason 偏移触发
    DEVIATED/DIVERGED 与对应归因提示
  * 判定门：exit code 0/1/2；1P1D gini N/A；多 run 实测噪声地板
"""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

import compare_twin as ct  # noqa: E402

# ---------------------------------------------------------------------------
# fixture builder — a self-contained mini aggregate (2P2D, 100s)
# ---------------------------------------------------------------------------


def make_mini_aggregate(**overrides):
    agg = {
        "meta": {
            "run_dir": "mini",
            "duration_s": 100,
            "trace_file_sha256": "deadbeef" * 8,
            "send_mode": "replay",
        },
        "summary": {
            "total_requests": 1000,
            "success_count": 980,
            "error_count": 20,
            "error_rate": 2.0,
            "e2e_latency_ms": {
                "count": 980,
                "p50": 100.0,
                "p90": 200.0,
                "p95": 250.0,
                "p99": 400.0,
                "max": 600.0,
                "mean": 130.0,
            },
            "schedule_latency_ms": {
                "count": 980,
                "p50": 50.0,
                "p90": 80.0,
                "p95": 90.0,
                "p99": 120.0,
                "max": 150.0,
                "mean": 55.0,
            },
            "ttft_latency_ms": {
                "count": 980,
                "p50": 40.0,
                "p90": 70.0,
                "p95": 85.0,
                "p99": 110.0,
                "max": 140.0,
                "mean": 45.0,
            },
            "cache_hit_summary": {
                "engine_token_hit_pct": 80.0,
                "engine_key_hit_pct": 90.0,
            },
        },
        "per_second": [
            {"t": t, "arrivals": 10, "success": 10, "errors": 0} for t in range(100)
        ],
        "mock_tps_ts": [
            {
                "t": float(t),
                "context_tps": 4000.0,
                "context_tps_with_cache": 5000.0,
                "generate_tps": 500.0,
            }
            for t in range(100)
        ],
        "kv_ts": [
            {
                "t": float(t),
                "used_tokens": 1000 * t,
                "capacity_tokens": 100000,
                "used_pct": 20.0 + 0.1 * t,
            }
            for t in range(100)
        ],
        "cache_hit_ts": [
            {"t": float(t), "engine_key": 90.0, "engine_token": 80.0}
            for t in range(100)
        ],
        "dispatch_reason_ts": [
            {
                "t": float(t),
                "predicted_execution_cap": 30.0,
                "batch_full": 60.0,
                "fixed_window_timeout": 10.0,
            }
            for t in range(100)
        ],
        "dispatch_batch_size_ts": [
            {
                "t": float(t),
                "predicted_execution_cap": 15.0,
                "batch_full": 32.0,
                "fixed_window_timeout": 20.0,
            }
            for t in range(100)
        ],
        "engine_dist": {
            "prefill": {
                "engine_count": 2,
                "requests_per_engine": [600, 380],
                "total": 980,
                "gini_cum": ct.gini_coef([600, 380]),
            },
            "decode": {
                "engine_count": 2,
                "requests_per_engine": [500, 480],
                "total": 980,
                "gini_cum": ct.gini_coef([500, 480]),
            },
        },
    }
    for k, v in overrides.items():
        agg[k] = v
    return agg


def write_run(root, name, aggregate, n_prefill=2, n_decode=2):
    run = Path(root) / name
    run.mkdir(parents=True, exist_ok=True)
    (run / "aggregate.json").write_text(json.dumps(aggregate), encoding="utf-8")
    (run / "run_meta.json").write_text(
        json.dumps({"params": {"n_prefill": n_prefill, "n_decode": n_decode}}),
        encoding="utf-8",
    )
    return run


def run_cli(*argv):
    return subprocess.run(
        [sys.executable, str(SCRIPT_DIR / "compare_twin.py")] + list(argv),
        capture_output=True,
        text=True,
    )


class MathTests(unittest.TestCase):
    def test_wasserstein_hand_computed(self):
        self.assertEqual(ct.wasserstein_1d([0], [1]), 1.0)
        self.assertEqual(ct.wasserstein_1d([0, 0], [1, 1]), 1.0)
        self.assertEqual(ct.wasserstein_1d([1, 2, 3], [1, 2, 3]), 0.0)
        # shift-by-one: every unit of mass moves 1 -> W = 1
        self.assertAlmostEqual(ct.wasserstein_1d([1, 2, 3], [2, 3, 4]), 1.0)
        self.assertAlmostEqual(ct.wasserstein_1d([1, 2, 3], [1.5, 2.5, 3.5]), 0.5)
        self.assertIsNone(ct.wasserstein_1d([], [1.0]))

    def test_ks_hand_computed(self):
        self.assertAlmostEqual(ct.ks_statistic([1, 2, 3, 4], [3, 4, 5, 6]), 0.5)
        self.assertEqual(ct.ks_statistic([1, 2], [1, 2]), 0.0)
        self.assertEqual(ct.ks_statistic([1, 2, 3], [4, 5, 6]), 1.0)
        self.assertIsNone(ct.ks_statistic([], [1.0]))

    def test_gini_hand_computed(self):
        self.assertAlmostEqual(ct.gini_coef([1, 1, 1, 1]), 0.0)
        self.assertAlmostEqual(ct.gini_coef([0, 0, 0, 4]), 0.75)
        self.assertAlmostEqual(ct.gini_coef([1, 2, 3]), 2.0 / 9.0, places=6)

    def test_percentile_nearest_rank(self):
        self.assertEqual(ct.percentile_nr(list(range(1, 101)), 0.50), 50)
        self.assertEqual(ct.percentile_nr(list(range(1, 11)), 0.90), 9)
        self.assertEqual(ct.percentile_nr([], 0.5), 0.0)

    def test_quantile_expand_respects_control_points(self):
        samples = ct.quantile_expand(
            {
                "count": 100,
                "p50": 100.0,
                "p90": 200.0,
                "p95": 250.0,
                "p99": 400.0,
                "max": 600.0,
                "mean": 130.0,
            },
            n_samples=1000,
        )
        self.assertEqual(len(samples), 1000)
        self.assertAlmostEqual(ct.percentile_nr(samples, 0.50, nd=3), 100.0, delta=2.0)
        self.assertAlmostEqual(ct.percentile_nr(samples, 0.90, nd=3), 200.0, delta=3.0)
        self.assertLessEqual(max(samples), 600.0)
        self.assertEqual(ct.quantile_expand({"count": 0}), [])


class PromParsingTests(unittest.TestCase):
    def test_long_wide_and_mixed_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "prom_export.jsonl"
            p.write_text(
                "\n".join(
                    [
                        json.dumps({"t": 0, "metric": "counter_a", "value": 10}),
                        json.dumps({"t": 1, "metric": "counter_a", "value": 25}),
                        json.dumps(
                            {"t": 0, "gauge_x": 1.5, "t2": "x"}
                        ),  # t2 non-numeric col skipped
                        json.dumps({"t": 1, "gauge_x": 2.5}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            parsed = ct._parse_prom_export(str(p))
            self.assertEqual(parsed["counter_a"], [(0.0, 10.0), (1.0, 25.0)])
            self.assertEqual(parsed["gauge_x"], [(0.0, 1.5), (1.0, 2.5)])
            self.assertNotIn("t2", parsed)

    def test_long_same_name_same_t_sums(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "prom_export.jsonl"
            p.write_text(
                "\n".join(
                    [
                        json.dumps({"t": 0, "metric": "c", "value": 3}),
                        json.dumps({"t": 0, "metric": "c", "value": 4}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            parsed = ct._parse_prom_export(str(p))
            self.assertEqual(parsed["c"], [(0.0, 7.0)])

    def test_gauge_same_name_same_t_averages(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "prom_export.jsonl"
            p.write_text(
                "\n".join(
                    [
                        json.dumps({"t": 0, "metric": "batch_size_x", "value": 10}),
                        json.dumps({"t": 0, "metric": "batch_size_x", "value": 20}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            parsed = ct._parse_prom_export(str(p))
            self.assertEqual(parsed["batch_size_x"], [(0.0, 15.0)])

    def test_counter_sniff_and_diff(self):
        self.assertTrue(ct._looks_like_counter([0, 5, 12, 20]))
        # a reset mid-series breaks ups >= len-1? [0,5,3,20]: ups pairs (0,5),(5,3),(3,20) -> 2 of 3 -> False
        self.assertFalse(ct._looks_like_counter([0, 5, 3, 20]))
        self.assertFalse(ct._looks_like_counter([10, 9, 11, 8, 12]))  # rate series
        self.assertEqual(ct._counter_to_rate([(0.0, 10.0), (2.0, 30.0)]), [(2.0, 10.0)])
        # reset drops the interval
        self.assertEqual(
            ct._counter_to_rate([(0.0, 100.0), (1.0, 5.0), (2.0, 15.0)]), [(2.0, 10.0)]
        )

    def test_name_matching(self):
        self.assertTrue(
            ct._name_matches("rtp_llm_context_tps_with_cache", "context_tps_with_cache")
        )
        self.assertTrue(
            ct._name_matches(
                "context_wall_tps_with_cache_seconds_count",
                "context_wall_tps_with_cache",
            )
        )
        self.assertFalse(ct._name_matches("context_tps", "generate_tps"))


class SelfConsistencyTests(unittest.TestCase):
    """同数据自洽断言：mock 侧数据喂成 real 侧输入 → 可比指标全 ALIGNED。"""

    def test_round_trip_all_comparable_aligned(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = write_run(tmp, "mini", make_mini_aggregate())
            mock = ct.load_mock_side(str(run))
            real_dir = Path(tmp) / "real"
            ct.synthesize_real_inputs(mock, str(real_dir))
            real = ct.load_real_side(
                str(real_dir / "client_events.jsonl"),
                str(real_dir / "prom_export.jsonl"),
            )
            lo, hi = mock.steady_window()
            results = ct.evaluate_metrics(mock, real, lo, hi)
            floors = ct.compute_floors(mock, [], lo, hi)
            by_name = {}
            for r in results:
                verdict, _ = ct.verdict_for(r, floors[r["name"]]["floor"])
                by_name[r["name"]] = (verdict, r)
            for name, (verdict, r) in by_name.items():
                if verdict == ct.VERDICT_SKIP:
                    self.assertIn(
                        name,
                        ("ttft_dist",),  # mini fixture HAS ttft
                        f"{name} unexpectedly SKIP: {r.get('skip_reason')}",
                    )
                    continue
                self.assertEqual(
                    verdict,
                    ct.VERDICT_ALIGNED,
                    f"{name}: distance={r.get('distance')} "
                    f"floor={floors[name]['floor']}",
                )
                if r.get("distance") is not None:
                    self.assertLess(
                        r["distance"],
                        floors[name]["floor"],
                        f"{name} round-trip must be ~0, got {r['distance']}",
                    )

    def test_cli_round_trip_exit_zero_and_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = write_run(tmp, "mini", make_mini_aggregate())
            mock = ct.load_mock_side(str(run))
            real_dir = Path(tmp) / "real"
            ct.synthesize_real_inputs(mock, str(real_dir))
            out = Path(tmp) / "twin"
            r = run_cli(
                "--mock-aggregate",
                str(run),
                "--real-client-events",
                str(real_dir / "client_events.jsonl"),
                "--real-prom",
                str(real_dir / "prom_export.jsonl"),
                "--out",
                str(out),
            )
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertIn("ALIGNED 12", r.stdout)
            self.assertIn("SKIP 0", r.stdout)
            summary = json.loads((Path(str(out) + "_summary.json")).read_text())
            self.assertEqual(summary["gate"]["exit_code"], 0)
            self.assertEqual(len(summary["metrics"]), 12)
            html = Path(str(out) + "_report.html").read_text()
            self.assertIn("v-ALIGNED", html)


class DeviationFixtureTests(unittest.TestCase):
    """偏差注入：对应指标 DEVIATED/DIVERGED + 归因提示触发。"""

    def _prepare(self, tmp):
        run = write_run(tmp, "mini", make_mini_aggregate())
        mock = ct.load_mock_side(str(run))
        real_dir = Path(tmp) / "real"
        ct.synthesize_real_inputs(mock, str(real_dir))
        return run, real_dir

    def _run(self, tmp):
        run, real_dir = self._prepare(tmp)
        r = run_cli(
            "--mock-aggregate",
            str(run),
            "--real-client-events",
            str(real_dir / "client_events.jsonl"),
            "--real-prom",
            str(real_dir / "prom_export.jsonl"),
            "--out",
            "-",
        )
        return r

    @staticmethod
    def _scale_client_events(path, field, factor):
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        for row in rows:
            if row.get(field):
                row[field] = row[field] * factor
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    @staticmethod
    def _scale_prom_wide_col(path, column, factor):
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        for row in rows:
            if column in row:
                row[column] = row[column] * factor
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    @staticmethod
    def _set_prom_wide_col(path, column, value):
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        for row in rows:
            if column in row:
                row[column] = value
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    def test_e2e_inflation_triggers_decode_tail_hint(self):
        with tempfile.TemporaryDirectory() as tmp:
            run, real_dir = self._prepare(tmp)
            # ttft kept, e2e doubled: R2 (TTFT aligned, e2e exceeded).
            self._scale_client_events(real_dir / "client_events.jsonl", "total_ms", 2.0)
            r = run_cli(
                "--mock-aggregate",
                str(run),
                "--real-client-events",
                str(real_dir / "client_events.jsonl"),
                "--real-prom",
                str(real_dir / "prom_export.jsonl"),
                "--out",
                "-",
            )
            self.assertEqual(r.returncode, 1)
            self.assertIn("decode 尾部问题", r.stdout)
            lines = [
                l for l in r.stdout.splitlines() if l.strip().startswith("e2e_dist")
            ]
            self.assertTrue(
                any("DIVERGED" in l or "DEVIATED" in l for l in lines), r.stdout
            )

    def test_ttft_and_e2e_inflation_triggers_time_model_hint(self):
        with tempfile.TemporaryDirectory() as tmp:
            run, real_dir = self._prepare(tmp)
            ce = real_dir / "client_events.jsonl"
            self._scale_client_events(ce, "total_ms", 2.0)
            self._scale_client_events(ce, "ttft_ms", 3.0)
            r = run_cli(
                "--mock-aggregate",
                str(run),
                "--real-client-events",
                str(ce),
                "--real-prom",
                str(real_dir / "prom_export.jsonl"),
                "--out",
                "-",
            )
            self.assertEqual(r.returncode, 1)
            self.assertIn("时间模型偏差", r.stdout)
            self.assertNotIn("decode 尾部问题", r.stdout)

    def test_tps_gap_with_quiet_latencies_triggers_accounting_hint(self):
        with tempfile.TemporaryDirectory() as tmp:
            run, real_dir = self._prepare(tmp)
            self._scale_prom_wide_col(
                real_dir / "prom_export.jsonl", "context_wall_tps_with_cache", 0.5
            )
            r = run_cli(
                "--mock-aggregate",
                str(run),
                "--real-client-events",
                str(real_dir / "client_events.jsonl"),
                "--real-prom",
                str(real_dir / "prom_export.jsonl"),
                "--out",
                "-",
            )
            self.assertEqual(r.returncode, 1)
            self.assertIn("记账口径问题非行为问题", r.stdout)
            lines = [
                l
                for l in r.stdout.splitlines()
                if l.strip().startswith("tps_context_with_cache")
            ]
            self.assertTrue(any("DIVERGED" in l for l in lines), r.stdout)

    def test_cache_hit_gap_triggers_kv_hint(self):
        with tempfile.TemporaryDirectory() as tmp:
            run, real_dir = self._prepare(tmp)
            self._set_prom_wide_col(
                real_dir / "prom_export.jsonl", "engine_token_hit_pct", 60.0
            )
            r = run_cli(
                "--mock-aggregate",
                str(run),
                "--real-client-events",
                str(real_dir / "client_events.jsonl"),
                "--real-prom",
                str(real_dir / "prom_export.jsonl"),
                "--out",
                "-",
            )
            self.assertEqual(r.returncode, 1)
            self.assertIn("前缀/KV 机制差", r.stdout)

    def test_dispatch_reason_shift_triggers_batch_hint(self):
        with tempfile.TemporaryDirectory() as tmp:
            run, real_dir = self._prepare(tmp)
            # halve the batch_full counter growth -> share shifts to the other reasons
            prom = real_dir / "prom_export.jsonl"
            rows = [json.loads(l) for l in prom.read_text().splitlines() if l.strip()]
            for row in rows:
                if row.get("metric", "").endswith('reason="batch_full"}'):
                    row["value"] = row["value"] * 0.5
            prom.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            r = run_cli(
                "--mock-aggregate",
                str(run),
                "--real-client-events",
                str(real_dir / "client_events.jsonl"),
                "--real-prom",
                str(prom),
                "--out",
                "-",
            )
            self.assertEqual(r.returncode, 1)
            self.assertIn("切批行为偏", r.stdout)


class GateAndFloorTests(unittest.TestCase):
    def test_exit_2_on_missing_input(self):
        r = run_cli(
            "--mock-aggregate",
            "/nonexistent/run",
            "--real-client-events",
            "/nonexistent/ce.jsonl",
        )
        self.assertEqual(r.returncode, 2)
        self.assertIn("ERROR", r.stderr)

    def test_exit_2_on_empty_client_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            ce = Path(tmp) / "client_events.jsonl"
            ce.write_text("\n", encoding="utf-8")
            r = run_cli(
                "--mock-aggregate", "/nonexistent", "--real-client-events", str(ce)
            )
            self.assertEqual(r.returncode, 2)

    def test_measured_floor_with_multiple_mock_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            run1 = write_run(tmp, "r1", make_mini_aggregate())
            # a second run with slightly different TPS: pairwise distance
            # becomes the measured floor for the TPS metrics.
            agg2 = make_mini_aggregate()
            agg2["mock_tps_ts"] = [
                {
                    "t": float(t),
                    "context_tps": 4020.0,
                    "context_tps_with_cache": 5050.0,
                    "generate_tps": 505.0,
                }
                for t in range(100)
            ]
            run2 = write_run(tmp, "r2", agg2)
            mock = ct.load_mock_side(str(run1))
            real_dir = Path(tmp) / "real"
            ct.synthesize_real_inputs(mock, str(real_dir))
            out = Path(tmp) / "twin"
            r = run_cli(
                "--mock-aggregate",
                str(run1),
                "--mock-runs",
                f"{run1},{run2}",
                "--real-client-events",
                str(real_dir / "client_events.jsonl"),
                "--real-prom",
                str(real_dir / "prom_export.jsonl"),
                "--out",
                str(out),
            )
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertIn("实测噪声地板", r.stdout)
            summary = json.loads((Path(str(out) + "_summary.json")).read_text())
            tps = next(
                m for m in summary["metrics"] if m["name"] == "tps_context_with_cache"
            )
            self.assertIn("measured", tps["floor_source"])
            # measured floor ~= 1% (5050 vs 5000); real side unchanged -> aligned
            self.assertEqual(tps["verdict"], "ALIGNED")

    def test_gini_na_on_1p1d(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = write_run(tmp, "p1", make_mini_aggregate(), n_prefill=1, n_decode=1)
            mock = ct.load_mock_side(str(run))
            real_dir = Path(tmp) / "real"
            ct.synthesize_real_inputs(mock, str(real_dir))
            r = run_cli(
                "--mock-aggregate",
                str(run),
                "--real-client-events",
                str(real_dir / "client_events.jsonl"),
                "--real-prom",
                str(real_dir / "prom_export.jsonl"),
                "--out",
                "-",
            )
            self.assertEqual(r.returncode, 0)
            self.assertIn("N/A", r.stdout)
            lines = [l for l in r.stdout.splitlines() if l.strip().startswith("gini")]
            self.assertTrue(any("1P1D" in l for l in lines), r.stdout)

    def test_dropped_prom_skips_prom_only_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = write_run(tmp, "mini", make_mini_aggregate())
            mock = ct.load_mock_side(str(run))
            real_dir = Path(tmp) / "real"
            ct.synthesize_real_inputs(mock, str(real_dir))
            r = run_cli(
                "--mock-aggregate",
                str(run),
                "--real-client-events",
                str(real_dir / "client_events.jsonl"),
                "--out",
                "-",
            )
            self.assertEqual(r.returncode, 0)
            # TPS / KV / dispatch / batch-size need the prom export -> SKIP
            for name in (
                "tps_context_with_cache",
                "tps_generate",
                "kv_used_pct",
                "dispatch_reason_share",
                "batch_size_dist",
                "cache_hit_token_pct",
            ):
                lines = [l for l in r.stdout.splitlines() if l.strip().startswith(name)]
                self.assertTrue(
                    any("SKIP" in l for l in lines),
                    f"{name} should SKIP without --real-prom:\n{r.stdout}",
                )


if __name__ == "__main__":
    unittest.main()
