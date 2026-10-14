"""CPU-only unit tests for perf_analyzer (ABBA paired OFF/ON analysis).

Run directly:
    /opt/conda310/bin/python -I \\
        rtp_llm/test/eplb_monitoring/perf_analyzer_test.py
"""

import copy
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from perf_analyzer import analyze_abba  # noqa: E402
from perf_analyzer import main as analyzer_main  # noqa: E402


def _sample(prefill, decode, throughput):
    return {
        "prefill_ms": prefill,
        "decode_ms_per_token": decode,
        "throughput_tokens_per_s": throughput,
    }


def _run(cycle, mode, samples, window_count=2):
    return {
        "cycle": cycle,
        "mode": mode,
        "samples": samples,
        "window_count": window_count,
    }


def _abba_records(cycles, on_factor=1.0, window_count=2):
    """cycles: list of per-cycle base samples (off-mode).  ON runs scale
    latency metrics by on_factor and divide throughput by it."""
    records = []
    for cycle, base in enumerate(cycles):
        on = [
            _sample(
                s["prefill_ms"] * on_factor,
                s["decode_ms_per_token"] * on_factor,
                s["throughput_tokens_per_s"] / on_factor,
            )
            for s in base
        ]
        records.extend(
            [
                _run(cycle, "off", base, 0),
                _run(cycle, "on", on, window_count),
                _run(cycle, "on", list(on), window_count),
                _run(cycle, "off", list(base), 0),
            ]
        )
    return records


class AnalyzeAbbaTest(unittest.TestCase):
    BASE = [_sample(100.0, 10.0, 1000.0), _sample(102.0, 10.2, 990.0)]

    def test_clean_runs_conclude_ok(self):
        records = _abba_records([self.BASE, self.BASE, self.BASE], on_factor=1.0)
        res = analyze_abba(records)
        self.assertEqual(res["conclusion"], "ok")
        self.assertEqual(res["cycles_complete"], 3)
        for metric, info in res["metrics"].items():
            for pc in info["per_cycle"]:
                self.assertAlmostEqual(pc["regression"], 0.0)
        # raw samples preserved
        self.assertEqual(res["raw_records"], records)

    def test_abba_merge_two_runs_per_mode(self):
        """Each cycle merges its two OFF and two ON runs before the median."""
        records = [
            _run(0, "off", [_sample(100.0, 10.0, 1000.0)], 0),
            _run(0, "on", [_sample(101.0, 10.1, 995.0)], 2),
            _run(0, "on", [_sample(101.0, 10.1, 995.0)], 2),
            _run(0, "off", [_sample(112.0, 11.2, 900.0)], 0),
        ]
        for cycle in (1, 2):
            records.extend(
                [
                    _run(cycle, "off", list(self.BASE), 0),
                    _run(cycle, "on", list(self.BASE), 2),
                    _run(cycle, "on", list(self.BASE), 2),
                    _run(cycle, "off", list(self.BASE), 0),
                ]
            )
        res = analyze_abba(records)
        pc0 = res["metrics"]["prefill_ms"]["per_cycle"][0]
        # merged OFF {100,112} median 106; ON {101,101} median 101
        self.assertEqual(pc0["off_median"], 106.0)
        self.assertEqual(pc0["on_median"], 101.0)

    def test_majority_repeated_regression_flags_regression(self):
        """2 of 3 cycles degrade prefill by 5% -> regression."""
        records = []
        for cycle in range(3):
            on = (
                [
                    _sample(
                        s["prefill_ms"] * 1.05,
                        s["decode_ms_per_token"],
                        s["throughput_tokens_per_s"],
                    )
                    for s in self.BASE
                ]
                if cycle < 2
                else [
                    _sample(
                        s["prefill_ms"],
                        s["decode_ms_per_token"],
                        s["throughput_tokens_per_s"],
                    )
                    for s in self.BASE
                ]
            )
            records.extend(
                [
                    _run(cycle, "off", self.BASE, 0),
                    _run(cycle, "on", on, 3),
                    _run(cycle, "on", list(on), 3),
                    _run(cycle, "off", list(self.BASE), 0),
                ]
            )
        res = analyze_abba(records)
        self.assertEqual(res["conclusion"], "regression")
        self.assertEqual(res["metrics"]["prefill_ms"]["verdict"], "regression")
        self.assertEqual(res["metrics"]["decode_ms_per_token"]["verdict"], "ok")
        self.assertEqual(res["metrics"]["prefill_ms"]["cycles_exceeding_threshold"], 2)

    def test_minority_exceedance_is_noisy_not_pass(self):
        records = _abba_records([self.BASE] * 3, on_factor=1.0)
        # Make only cycle 1 degrade decode latency.
        for r in records:
            if r["cycle"] == 1 and r["mode"] == "on":
                r["samples"] = [
                    _sample(
                        s["prefill_ms"],
                        s["decode_ms_per_token"] * 1.10,
                        s["throughput_tokens_per_s"],
                    )
                    for s in r["samples"]
                ]
        res = analyze_abba(records)
        self.assertEqual(res["conclusion"], "noisy")
        self.assertEqual(res["metrics"]["decode_ms_per_token"]["verdict"], "noisy")
        self.assertEqual(res["metrics"]["prefill_ms"]["verdict"], "ok")

    def test_throughput_direction(self):
        """Throughput degradation means on < off: off/on - 1 > threshold."""
        records = _abba_records([self.BASE] * 3, on_factor=1.05)
        res = analyze_abba(records)
        self.assertEqual(res["conclusion"], "regression")
        for info in res["metrics"].values():
            self.assertEqual(info["verdict"], "regression")
            for pc in info["per_cycle"]:
                self.assertGreater(pc["regression"], 0.03)

    def test_rare_flush_latency_cost_cannot_hide_behind_median(self):
        # One slow request in ten leaves the median unchanged but adds 10%
        # amortized latency. Repeat in two cycles to establish a regression.
        base = [_sample(100.0, 10.0, 1000.0) for _ in range(10)]
        records = _abba_records([base] * 3)
        for run in records:
            if run["mode"] == "on" and run["cycle"] < 2:
                run["samples"] = copy.deepcopy(run["samples"])
                run["samples"][-1]["prefill_ms"] = 200.0
                run["samples"][-1]["decode_ms_per_token"] = 20.0
        res = analyze_abba(records)
        self.assertEqual(res["conclusion"], "regression")
        self.assertFalse(res["ok"])
        for metric in ("prefill_ms", "decode_ms_per_token"):
            info = res["metrics"][metric]
            self.assertEqual(info["median_cycles_exceeding_threshold"], 0)
            self.assertEqual(info["amortized_cycles_exceeding_threshold"], 2)
            self.assertEqual(info["amortized_estimator"], "arithmetic_mean")
            self.assertAlmostEqual(info["per_cycle"][0]["regression"], 0.0)
            self.assertAlmostEqual(info["per_cycle"][0]["amortized_regression"], 0.1)
            self.assertGreater(info["on_variance"], 0.0)

    def test_rare_flush_throughput_cost_uses_elapsed_time(self):
        # Nine 1-second requests and one 2-second request serve 10,000 tokens
        # in 11 seconds: 10% cost increase, despite an unchanged median rate.
        for with_totals in (False, True):
            with self.subTest(with_totals=with_totals):
                base = [_sample(100.0, 10.0, 1000.0) for _ in range(10)]
                records = copy.deepcopy(_abba_records([base] * 3))
                for run in records:
                    if run["mode"] == "on":
                        run["samples"][-1]["throughput_tokens_per_s"] = 500.0
                    if with_totals:
                        for sample in run["samples"]:
                            sample["output_len"] = 1000
                            sample["wall_s"] = 1000 / sample["throughput_tokens_per_s"]
                res = analyze_abba(records)
                info = res["metrics"]["throughput_tokens_per_s"]
                self.assertEqual(res["conclusion"], "regression")
                self.assertEqual(info["median_cycles_exceeding_threshold"], 0)
                self.assertEqual(info["amortized_cycles_exceeding_threshold"], 3)
                self.assertAlmostEqual(info["off_amortized"], 1000.0)
                self.assertAlmostEqual(info["on_amortized"], 10000 / 11)
                self.assertAlmostEqual(
                    info["per_cycle"][0]["amortized_regression"], 0.1
                )
                self.assertEqual(
                    info["amortized_estimator"],
                    (
                        "aggregate_tokens_over_wall_time"
                        if with_totals
                        else "harmonic_mean_equal_tokens"
                    ),
                )

    def test_variable_output_lengths_use_token_weighted_aggregate(self):
        records = []
        for cycle in range(3):
            for mode in ("off", "on", "on", "off"):
                samples = []
                walls = (1.0, 10.0) if mode == "off" else (2.0, 5.0)
                for tokens, wall in zip((100, 1000), walls):
                    sample = _sample(100.0, 10.0, tokens / wall)
                    sample.update(output_len=tokens, wall_s=wall)
                    samples.append(sample)
                records.append(_run(cycle, mode, samples, 2))
        res = analyze_abba(records)
        info = res["metrics"]["throughput_tokens_per_s"]
        self.assertEqual(res["conclusion"], "ok")
        self.assertAlmostEqual(info["off_amortized"], 100.0)
        self.assertAlmostEqual(info["on_amortized"], 1100 / 7)
        # An unweighted harmonic mean of ON rates 50 and 200 would be 80,
        # incorrectly reporting a regression instead of this improvement.
        self.assertLess(info["per_cycle"][0]["amortized_regression"], 0.0)

    def test_one_cycle_amortized_outlier_is_noisy_not_pass(self):
        base = [_sample(100.0, 10.0, 1000.0) for _ in range(10)]
        records = _abba_records([base] * 3)
        for run in records:
            if run["mode"] == "on" and run["cycle"] == 0:
                run["samples"] = copy.deepcopy(run["samples"])
                run["samples"][-1]["decode_ms_per_token"] = 20.0
        res = analyze_abba(records)
        self.assertEqual(res["conclusion"], "noisy")
        self.assertFalse(res["ok"])

    def test_aggregate_metadata_must_be_complete_and_valid(self):
        records = copy.deepcopy(_abba_records([self.BASE] * 3))
        for run in records:
            for sample in run["samples"]:
                sample.update(
                    output_len=32, wall_s=32 / sample["throughput_tokens_per_s"]
                )
        for field, value in (
            ("output_len", True),
            ("output_len", 0),
            ("wall_s", 0),
            ("wall_s", float("nan")),
        ):
            with self.subTest(field=field, value=value):
                bad = copy.deepcopy(records)
                bad[0]["samples"][0][field] = value
                self.assertEqual(analyze_abba(bad)["conclusion"], "invalid")
        for remove_fields in (("wall_s",), ("output_len", "wall_s")):
            bad = copy.deepcopy(records)
            for field in remove_fields:
                del bad[0]["samples"][0][field]
            self.assertEqual(analyze_abba(bad)["conclusion"], "invalid")

    def test_fewer_than_three_cycles_inconclusive(self):
        records = _abba_records([self.BASE, self.BASE], on_factor=1.0)
        res = analyze_abba(records)
        self.assertEqual(res["conclusion"], "inconclusive")
        self.assertIn("3 complete ABBA cycles", res["reason"])

    def test_insufficient_window_coverage_inconclusive(self):
        records = _abba_records([self.BASE] * 3, on_factor=1.0, window_count=1)
        res = analyze_abba(records)
        self.assertEqual(res["conclusion"], "inconclusive")
        self.assertIn("flush windows", res["reason"])

    def test_incomplete_cycle_excluded(self):
        records = _abba_records([self.BASE] * 3, on_factor=1.0)
        # Add a 4th cycle missing its ON runs -> still 3 complete cycles.
        records.append(_run(3, "off", self.BASE, 0))
        res = analyze_abba(records)
        self.assertEqual(res["cycles_total"], 4)
        self.assertEqual(res["cycles_complete"], 3)
        self.assertEqual(res["conclusion"], "ok")

    def test_ab_pairs_are_not_complete_cycles(self):
        """3 cycles of (off,on) = 6 runs must NOT pass as 3 ABBA cycles."""
        records = []
        for cycle in range(3):
            records.extend(
                [
                    _run(cycle, "off", list(self.BASE), 0),
                    _run(cycle, "on", list(self.BASE), 2),
                ]
            )
        res = analyze_abba(records)
        self.assertEqual(res["conclusion"], "inconclusive")
        self.assertEqual(res["cycles_complete"], 0)
        self.assertFalse(res["ok"])

    def test_wrong_abba_order_invalid(self):
        """off,on,off,on inside a cycle is invalid, not 'complete'."""
        records = []
        for cycle in range(3):
            records.extend(
                [
                    _run(cycle, "off", list(self.BASE), 0),
                    _run(cycle, "on", list(self.BASE), 2),
                    _run(cycle, "off", list(self.BASE), 0),
                    _run(cycle, "on", list(self.BASE), 2),
                ]
            )
        res = analyze_abba(records)
        self.assertEqual(res["conclusion"], "invalid")
        self.assertTrue(any("off,on,on,off" in e for e in res["errors"]))

    def test_interleaved_cycles_invalid(self):
        """Revisiting a closed cycle (cross-cycle interleave) is invalid."""
        records = _abba_records([self.BASE, self.BASE], on_factor=1.0)
        records.extend(
            [
                _run(0, "off", list(self.BASE), 0),
                _run(0, "on", list(self.BASE), 2),
                _run(0, "on", list(self.BASE), 2),
                _run(0, "off", list(self.BASE), 0),
            ]
        )
        res = analyze_abba(records)
        self.assertEqual(res["conclusion"], "invalid")
        self.assertTrue(any("contiguous" in e for e in res["errors"]))

    def test_overlong_cycle_invalid(self):
        records = _abba_records([self.BASE, self.BASE], on_factor=1.0)
        extra = [
            _run(2, "off", list(self.BASE), 0),
            _run(2, "on", list(self.BASE), 2),
            _run(2, "on", list(self.BASE), 2),
            _run(2, "off", list(self.BASE), 0),
            _run(2, "off", list(self.BASE), 0),  # 5th run in one cycle
        ]
        res = analyze_abba(records + extra)
        self.assertEqual(res["conclusion"], "invalid")

    def test_trailing_unfinished_cycle_ignored(self):
        """3 full cycles + a trailing unfinished cycle still conclude ok."""
        records = _abba_records([self.BASE] * 3, on_factor=1.0)
        records.extend(
            [
                _run(3, "off", list(self.BASE), 0),
                _run(3, "on", list(self.BASE), 2),
            ]
        )
        res = analyze_abba(records)
        self.assertEqual(res["conclusion"], "ok")
        self.assertEqual(res["cycles_complete"], 3)
        self.assertTrue(res["ok"])

    def test_ok_flag_follows_conclusion(self):
        """Callers gate on conclusion=='ok'; ok mirrors exactly that."""
        ok_res = analyze_abba(_abba_records([self.BASE] * 3, on_factor=1.0))
        self.assertTrue(ok_res["ok"])
        reg_res = analyze_abba(_abba_records([self.BASE] * 3, on_factor=1.10))
        self.assertEqual(reg_res["conclusion"], "regression")
        self.assertFalse(reg_res["ok"])
        short_res = analyze_abba(_abba_records([self.BASE] * 2))
        self.assertEqual(short_res["conclusion"], "inconclusive")
        self.assertFalse(short_res["ok"])

    def test_invalid_inputs_fail_explicitly(self):
        good = _abba_records([self.BASE] * 3)
        # zero value
        bad = [dict(r) for r in good]
        bad[0]["samples"] = [_sample(0.0, 10.0, 1000.0)]
        self.assertEqual(analyze_abba(bad)["conclusion"], "invalid")
        # NaN
        bad = [dict(r) for r in good]
        bad[1]["samples"] = [_sample(float("nan"), 10.0, 1000.0)]
        self.assertEqual(analyze_abba(bad)["conclusion"], "invalid")
        # missing metric
        bad = [dict(r) for r in good]
        bad[2]["samples"] = [{"prefill_ms": 1.0, "decode_ms_per_token": 1.0}]
        self.assertEqual(analyze_abba(bad)["conclusion"], "invalid")
        # bad mode
        bad = [dict(r) for r in good]
        bad[0] = dict(bad[0], mode="warmup")
        self.assertEqual(analyze_abba(bad)["conclusion"], "invalid")
        # empty samples
        bad = [dict(r) for r in good]
        bad[0] = dict(bad[0], samples=[])
        self.assertEqual(analyze_abba(bad)["conclusion"], "invalid")
        # not a list
        self.assertEqual(analyze_abba({"runs": []})["conclusion"], "invalid")
        # bad threshold
        self.assertEqual(
            analyze_abba(good, regression_threshold=0.0)["conclusion"], "invalid"
        )

    def test_variance_reported(self):
        records = _abba_records([self.BASE] * 3, on_factor=1.0)
        res = analyze_abba(records)
        info = res["metrics"]["prefill_ms"]
        self.assertGreater(info["off_variance"], 0.0)  # {100,102} spread
        self.assertEqual(info["off_sample_count"], 12)  # 3 cycles x 2 runs x 2

    def test_cli_exit_codes(self):
        records = _abba_records([self.BASE] * 3, on_factor=1.0)
        regressing = _abba_records([self.BASE] * 3, on_factor=1.10)
        short = _abba_records([self.BASE] * 2, on_factor=1.0)
        with tempfile.TemporaryDirectory() as d:
            ok_path = Path(d) / "ok.json"
            ok_path.write_text(json.dumps(records))
            reg_path = Path(d) / "reg.json"
            reg_path.write_text(json.dumps(regressing))
            short_path = Path(d) / "short.json"
            short_path.write_text(json.dumps(short))
            bad_path = Path(d) / "bad.json"
            bad_path.write_text("{not json")
            self.assertEqual(analyzer_main([str(ok_path)]), 0)
            self.assertEqual(analyzer_main([str(reg_path)]), 3)
            self.assertEqual(analyzer_main([str(short_path)]), 2)
            self.assertEqual(analyzer_main([str(bad_path)]), 1)


if __name__ == "__main__":
    unittest.main()
