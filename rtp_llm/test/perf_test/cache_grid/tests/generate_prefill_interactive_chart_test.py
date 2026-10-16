import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rtp_llm.test.perf_test.cache_grid.config.topology import profile_cards
from rtp_llm.test.perf_test.cache_grid.plot.generate_prefill_interactive_chart import (
    apply_z_metric,
    detect_cards,
    load_rows,
    main,
    representative_levels,
    representative_slice,
)
from rtp_llm.test.perf_test.cache_grid.runner.result_schema import (
    finite_number as number,
)
from rtp_llm.test.perf_test.cache_grid.runner.result_schema import (
    observed_cache_len,
    prefill_rt,
)

PLOTLY_AVAILABLE = importlib.util.find_spec("plotly") is not None


def _sample_metrics(statuses=None):
    if statuses is None:
        statuses = ["ok"]
    metrics = []
    for idx, status in enumerate(statuses):
        metrics.append(
            {
                "batch_size": 1,
                "input_len": 2048 + idx * 1024,
                "cache_len_requested": 512,
                "cache_len_observed": [512, 512, 512],
                "measure_runs": 3,
                "success_runs": 3,
                "status": status,
                "runs": [
                    {
                        "prefill_time_ms": 10.0 + idx,
                        "success": True,
                        "reuse_len": 512,
                        "ttft_ms": 10.0 + idx,
                    }
                    for _ in range(3)
                ],
            }
        )
    return {"schema_version": 2, "mode": "prefix_cache_grid", "metrics": metrics}


class LoadRowsTest(unittest.TestCase):
    def test_accepts_invalid_reuse_status(self):
        payload = _sample_metrics(["invalid_reuse"])
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "results.json"
            source.write_text(json.dumps(payload), encoding="utf-8")
            rows = load_rows(source, 1)
        self.assertEqual(len(rows), 1)

    def test_rejects_failed_status(self):
        payload = _sample_metrics(["failed"])
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "results.json"
            source.write_text(json.dumps(payload), encoding="utf-8")
            rows = load_rows(source, 1)
        self.assertEqual(len(rows), 0)

    def test_deduplicates_by_geometry_median(self):
        payload = {
            "schema_version": 2,
            "mode": "prefix_cache_grid",
            "metrics": [
                {
                    "batch_size": 1,
                    "input_len": 2048,
                    "cache_len_requested": 512,
                    "cache_len_observed": [512, 512, 512],
                    "measure_runs": 3,
                    "success_runs": 3,
                    "status": "ok",
                    "runs": [
                        {
                            "prefill_time_ms": rt,
                            "success": True,
                            "reuse_len": 512,
                            "ttft_ms": rt,
                        }
                        for rt in (9.0, 10.0, 11.0)
                    ],
                },
                {
                    "batch_size": 1,
                    "input_len": 2048,
                    "cache_len_requested": 512,
                    "cache_len_observed": [512, 512, 512],
                    "measure_runs": 3,
                    "success_runs": 3,
                    "status": "ok",
                    "runs": [
                        {
                            "prefill_time_ms": rt,
                            "success": True,
                            "reuse_len": 512,
                            "ttft_ms": rt,
                        }
                        for rt in (19.0, 20.0, 21.0)
                    ],
                },
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "results.json"
            source.write_text(json.dumps(payload), encoding="utf-8")
            rows = load_rows(source, 1)
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["prefill_rt"], 15.0)
        self.assertEqual(rows[0]["compute_len"], 1536)
        self.assertEqual(rows[0]["cache_len"], 512)

    def test_axes_are_compute_cache_rt(self):
        payload = _sample_metrics(["ok"])
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "results.json"
            source.write_text(json.dumps(payload), encoding="utf-8")
            rows = load_rows(source, 1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["input_len"], 2048)
        self.assertEqual(rows[0]["cache_len"], 512)
        self.assertEqual(rows[0]["compute_len"], 1536)
        self.assertAlmostEqual(rows[0]["prefill_rt"], 10.0)


class LatencyColorTest(unittest.TestCase):
    @unittest.skipUnless(PLOTLY_AVAILABLE, "plotly is an optional chart dependency")
    def test_higher_latency_is_darker_without_changing_values(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "results.json"
            source.write_text(json.dumps(_sample_metrics(["ok", "ok"])))
            with patch("sys.argv", ["chart", "--input", str(source)]), patch(
                "plotly.graph_objects.Figure.write_html", autospec=True
            ) as write_html:
                main()
            figure = write_html.call_args.args[0]
            points = next(
                trace for trace in figure.data if trace.name == "measurements"
            )
            self.assertTrue(points.marker.reversescale)
            self.assertEqual(list(points.marker.color), [10.0, 11.0])
            self.assertEqual(list(points.z), [10.0, 11.0])
            self.assertIn("darker = slower", points.marker.colorbar.title.text)


class AllRunsTest(unittest.TestCase):
    @staticmethod
    def _write(directory, payload):
        source = Path(directory) / "results.json"
        source.write_text(json.dumps(payload), encoding="utf-8")
        return source

    def test_plots_every_run_without_median_collapse(self):
        def metric(rts):
            return {
                "batch_size": 1,
                "input_len": 2048,
                "cache_len_requested": 512,
                "cache_len_observed": [512, 512, 512],
                "measure_runs": 3,
                "success_runs": 3,
                "status": "ok",
                "runs": [
                    {
                        "prefill_time_ms": rt,
                        "success": True,
                        "reuse_len": 512,
                        "ttft_ms": rt,
                    }
                    for rt in rts
                ],
            }

        payload = {
            "schema_version": 2,
            "mode": "prefix_cache_grid",
            "metrics": [metric((9.0, 10.0, 11.0)), metric((19.0, 20.0, 21.0))],
        }
        with tempfile.TemporaryDirectory() as directory:
            source = self._write(directory, payload)
            rows = load_rows(source, 1, all_runs=True)
        self.assertEqual(len(rows), 6)
        self.assertEqual(
            [row["prefill_rt"] for row in rows],
            [9.0, 10.0, 11.0, 19.0, 20.0, 21.0],
        )
        self.assertEqual([row["run_index"] for row in rows], [1, 2, 3, 1, 2, 3])
        self.assertTrue(all(row["compute_len"] == 1536 for row in rows))

    def test_uses_run_level_reuse_len(self):
        payload = {
            "schema_version": 2,
            "mode": "prefix_cache_grid",
            "metrics": [
                {
                    "batch_size": 1,
                    "input_len": 2048,
                    "cache_len_requested": 512,
                    "cache_len_observed": [512],
                    "measure_runs": 1,
                    "success_runs": 1,
                    "status": "ok",
                    "runs": [
                        {
                            "prefill_time_ms": 10.0,
                            "success": True,
                            "reuse_len": 256,
                            "ttft_ms": 10.0,
                        }
                    ],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            source = self._write(directory, payload)
            rows = load_rows(source, 1, all_runs=True)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["cache_len"], 256)
        self.assertEqual(rows[0]["compute_len"], 1792)

    def test_skips_failed_runs(self):
        payload = {
            "schema_version": 2,
            "mode": "prefix_cache_grid",
            "metrics": [
                {
                    "batch_size": 1,
                    "input_len": 2048,
                    "cache_len_requested": 512,
                    "cache_len_observed": [512],
                    "measure_runs": 2,
                    "success_runs": 1,
                    "status": "ok",
                    "runs": [
                        {
                            "prefill_time_ms": 10.0,
                            "success": True,
                            "reuse_len": 512,
                            "ttft_ms": 10.0,
                        },
                        {
                            "prefill_time_ms": 99.0,
                            "success": False,
                            "reuse_len": 512,
                            "ttft_ms": 99.0,
                        },
                    ],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            source = self._write(directory, payload)
            rows = load_rows(source, 1, all_runs=True)
        self.assertEqual([row["prefill_rt"] for row in rows], [10.0])

    @unittest.skipUnless(PLOTLY_AVAILABLE, "plotly is an optional chart dependency")
    def test_main_all_runs_hover_shows_run_index(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self._write(directory, _sample_metrics(["ok"]))
            with patch(
                "sys.argv", ["chart", "--input", str(source), "--all-runs"]
            ), patch(
                "plotly.graph_objects.Figure.write_html", autospec=True
            ) as write_html:
                main()
        figure = write_html.call_args.args[0]
        points = next(trace for trace in figure.data if trace.name == "measurements")
        self.assertEqual(list(points.z), [10.0, 10.0, 10.0])
        self.assertIn("Run: %{customdata[3]}", points.hovertemplate)
        self.assertEqual([entry[3] for entry in points.customdata], [1, 2, 3])
        self.assertEqual([entry[4] for entry in points.customdata], [10.0] * 3)

    @unittest.skipUnless(PLOTLY_AVAILABLE, "plotly is an optional chart dependency")
    def test_main_median_hover_uses_fixed_customdata_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self._write(directory, _sample_metrics(["ok"]))
            with patch("sys.argv", ["chart", "--input", str(source)]), patch(
                "plotly.graph_objects.Figure.write_html", autospec=True
            ) as write_html:
                main()
        figure = write_html.call_args.args[0]
        points = next(trace for trace in figure.data if trace.name == "measurements")
        self.assertNotIn("Run: %{customdata[3]}", points.hovertemplate)
        self.assertIn(
            "Server first-token latency: %{customdata[4]", points.hovertemplate
        )
        self.assertTrue(all(len(entry) == 5 for entry in points.customdata))
        self.assertTrue(all(entry[3] is None for entry in points.customdata))


class ZMetricTest(unittest.TestCase):
    ROWS = [
        {
            "input_len": 2048.0,
            "cache_len": 512.0,
            "compute_len": 1536.0,
            "prefill_rt": 30.0,
        },
        {
            "input_len": 4096.0,
            "cache_len": 0.0,
            "compute_len": 4096.0,
            "prefill_rt": 60.0,
        },
        {
            "input_len": 1024.0,
            "cache_len": 1024.0,
            "compute_len": 0.0,
            "prefill_rt": 0.0,
        },
    ]

    def test_tpm_effective_uses_input_len(self):
        rows = apply_z_metric(self.ROWS, "tpm-effective")
        self.assertEqual(len(rows), 2)
        self.assertAlmostEqual(rows[0]["tpm_effective"], 2048 * 60000 / 30.0)
        self.assertAlmostEqual(rows[1]["tpm_effective"], 4096 * 60000 / 60.0)

    def test_tpm_compute_uses_compute_len(self):
        rows = apply_z_metric(self.ROWS, "tpm-compute")
        self.assertEqual(len(rows), 2)
        self.assertAlmostEqual(rows[0]["tpm_compute"], 1536 * 60000 / 30.0)
        self.assertAlmostEqual(rows[1]["tpm_compute"], 4096 * 60000 / 60.0)

    def test_rt_metric_is_identity(self):
        rows = apply_z_metric(self.ROWS, "rt")
        self.assertEqual(len(rows), 3)
        self.assertEqual([row["prefill_rt"] for row in rows], [30.0, 60.0, 0.0])

    def test_zero_rt_is_dropped_for_tpm(self):
        rows = apply_z_metric(self.ROWS, "tpm-effective")
        self.assertNotIn(1024.0, [row["input_len"] for row in rows])

    @unittest.skipUnless(PLOTLY_AVAILABLE, "plotly is an optional chart dependency")
    def test_main_tpm_effective_axes_and_hover(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "results.json"
            source.write_text(json.dumps(_sample_metrics(["ok"])), encoding="utf-8")
            with patch(
                "sys.argv",
                ["chart", "--input", str(source), "--z-metric", "tpm-effective"],
            ), patch(
                "plotly.graph_objects.Figure.write_html", autospec=True
            ) as write_html:
                main()
        figure = write_html.call_args.args[0]
        points = next(trace for trace in figure.data if trace.name == "measurements")
        input_len = 2048.0
        rt = 10.0
        expected = input_len * 60000 / rt
        self.assertAlmostEqual(list(points.z)[0], expected)
        self.assertIn(
            "Effective TPM (Z, tokens/min)", figure.layout.scene.zaxis.title.text
        )
        self.assertIn(
            "Server first-token latency: %{customdata[4]", points.hovertemplate
        )
        self.assertIn("darker = lower throughput", points.marker.colorbar.title.text)
        self.assertTrue(points.marker.reversescale)

    def test_cards_divides_tpm(self):
        rows = apply_z_metric(self.ROWS, "tpm-effective", cards=8.0)
        self.assertEqual(len(rows), 2)
        self.assertAlmostEqual(rows[0]["tpm_effective"], 2048 * 60000 / 30.0 / 8.0)

    def test_cards_does_not_apply_to_rt(self):
        rows = apply_z_metric(self.ROWS, "rt", cards=8.0, rt_cards=1.0)
        self.assertEqual([row["prefill_rt"] for row in rows], [30.0, 60.0, 0.0])

    def test_detect_cards_missing_config(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "results.json"
            source.write_text(
                json.dumps(
                    {"schema_version": 2, "mode": "prefix_cache_grid", "metrics": []}
                ),
                encoding="utf-8",
            )
            self.assertIsNone(detect_cards(source))

    @unittest.skipUnless(PLOTLY_AVAILABLE, "plotly is an optional chart dependency")
    def test_main_per_card_tpm_uses_embedded_profile(self):
        payload = {
            "schema_version": 2,
            "mode": "prefix_cache_grid",
            "profile": {"schema_version": 1, "engine": {"tp_size": 8}},
            "metrics": [
                {
                    "batch_size": 1,
                    "input_len": 2048,
                    "cache_len_requested": 512,
                    "cache_len_observed": [512, 512, 512],
                    "measure_runs": 3,
                    "success_runs": 3,
                    "status": "ok",
                    "runs": [
                        {
                            "prefill_time_ms": 10.0,
                            "success": True,
                            "reuse_len": 512,
                            "ttft_ms": 10.0,
                        }
                        for _ in range(3)
                    ],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "results.json"
            source.write_text(json.dumps(payload), encoding="utf-8")
            with patch(
                "sys.argv",
                ["chart", "--input", str(source), "--z-metric", "tpm-effective"],
            ), patch(
                "plotly.graph_objects.Figure.write_html", autospec=True
            ) as write_html:
                main()
        figure = write_html.call_args.args[0]
        points = next(trace for trace in figure.data if trace.name == "measurements")
        expected = 2048 * 60000 / 10.0 / 8.0
        self.assertAlmostEqual(list(points.z)[0], expected)
        self.assertIn(
            "Effective TPM per card (Z, tokens/min)",
            figure.layout.scene.zaxis.title.text,
        )

    @unittest.skipUnless(PLOTLY_AVAILABLE, "plotly is an optional chart dependency")
    def test_main_all_run_tpm_uses_explicit_profile_world_size(self):
        payload = {
            "schema_version": 2,
            "mode": "prefix_cache_grid",
            "metrics": [
                {
                    "batch_size": 1,
                    "input_len": 2048,
                    "cache_len_requested": 512,
                    "cache_len_observed": [512, 512, 512],
                    "measure_runs": 3,
                    "success_runs": 3,
                    "status": "ok",
                    "runs": [
                        {
                            "prefill_time_ms": rt,
                            "success": True,
                            "reuse_len": 512,
                            "ttft_ms": rt,
                        }
                        for rt in (10.0, 20.0, 30.0)
                    ],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "results.json"
            source.write_text(json.dumps(payload), encoding="utf-8")
            profile = Path(directory) / "profile.json"
            profile.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "engine": {"world_size": 16, "tp_size": 8, "ep_size": 8},
                    }
                )
            )
            with patch(
                "sys.argv",
                [
                    "chart",
                    "--input",
                    str(source),
                    "--z-metric",
                    "tpm-effective",
                    "--profile",
                    str(profile),
                    "--all-runs",
                ],
            ), patch(
                "plotly.graph_objects.Figure.write_html", autospec=True
            ) as write_html:
                main()
        figure = write_html.call_args.args[0]
        points = next(trace for trace in figure.data if trace.name == "measurements")
        self.assertEqual(len(points.z), 3)
        for value, rt in zip(points.z, (10.0, 20.0, 30.0)):
            self.assertAlmostEqual(value, 2048 * 60000 / rt / 16.0)
        self.assertIn(
            "Effective TPM per card (Z, tokens/min)",
            figure.layout.scene.zaxis.title.text,
        )

    def test_profile_card_topology(self):
        self.assertEqual(
            profile_cards({"engine": {"world_size": 8, "tp_size": 8, "ep_size": 8}}), 8
        )
        self.assertEqual(
            profile_cards(
                {"engine": {"tp_size": 2, "dp_size": 4, "pp_size": 2, "ep_size": 8}}
            ),
            16,
        )
        self.assertIsNone(profile_cards({}))
        for value in (0, -1, 1.5, "nan", True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                profile_cards({"engine": {"world_size": value}})

    def test_embedded_profile_sets_card_count(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "results.json"
            source.write_text(
                json.dumps(
                    {
                        "profile": {"schema_version": 1, "engine": {"world_size": 16}},
                    }
                )
            )
            self.assertEqual(detect_cards(source), 16)

    @unittest.skipUnless(PLOTLY_AVAILABLE, "plotly is an optional chart dependency")
    def test_main_explicit_cards_one_keeps_system_tpm(self):
        payload = {
            "schema_version": 2,
            "mode": "prefix_cache_grid",
            "metrics": [
                {
                    "batch_size": 1,
                    "input_len": 2048,
                    "cache_len_requested": 0,
                    "cache_len_observed": [0],
                    "measure_runs": 1,
                    "success_runs": 1,
                    "status": "ok",
                    "runs": [
                        {
                            "prefill_time_ms": 10.0,
                            "success": True,
                            "reuse_len": 0,
                            "ttft_ms": 10.0,
                        }
                    ],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "results.json"
            source.write_text(json.dumps(payload), encoding="utf-8")
            with patch(
                "sys.argv",
                [
                    "chart",
                    "--input",
                    str(source),
                    "--z-metric",
                    "tpm-effective",
                    "--cards",
                    "1",
                ],
            ), patch(
                "plotly.graph_objects.Figure.write_html", autospec=True
            ) as write_html:
                main()
        figure = write_html.call_args.args[0]
        points = next(trace for trace in figure.data if trace.name == "measurements")
        expected = 2048 * 60000 / 10.0
        self.assertAlmostEqual(list(points.z)[0], expected)
        self.assertIn(
            "Effective TPM (Z, tokens/min)", figure.layout.scene.zaxis.title.text
        )


class TrendGuideTest(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {
                "input_len": float(value * 2),
                "cache_len": float(value),
                "compute_len": float(value),
                "prefill_rt": float(value),
            }
            for value in range(0, 101, 10)
        ]

    def test_representative_levels_span_axis(self):
        self.assertEqual(
            representative_levels(self.rows, "cache_len"),
            [0.0, 30.0, 70.0, 100.0],
        )

    def test_representative_slice_stays_on_selected_plane(self):
        points = representative_slice(self.rows, "cache_len", 70.0, "compute_len")
        self.assertGreaterEqual(len(points), 2)
        self.assertLessEqual(len(points), 12)
        self.assertTrue(all(point["cache_len"] == 70.0 for point in points))
        self.assertEqual(
            [point["compute_len"] for point in points],
            sorted(point["compute_len"] for point in points),
        )

    def test_representative_slice_recomputes_tpm_for_adjusted_geometry(self):
        enriched = apply_z_metric(self.rows, "tpm-effective", cards=2.0)
        points = representative_slice(enriched, "cache_len", 70.0, "compute_len")
        for point in points:
            self.assertAlmostEqual(
                point["tpm_effective"],
                (point["cache_len"] + point["compute_len"])
                * 60000.0
                / point["prefill_rt"]
                / 2.0,
            )


class NumberTest(unittest.TestCase):
    def test_finite(self):
        self.assertEqual(number(42), 42.0)
        self.assertEqual(number("3.14"), 3.14)

    def test_non_finite(self):
        self.assertIsNone(number(float("nan")))
        self.assertIsNone(number(float("inf")))
        self.assertIsNone(number("abc"))
        self.assertIsNone(number(None))


class ObservedCacheLenTest(unittest.TestCase):
    def test_aggregate_reuse_is_not_a_fallback(self):
        item = {"cache_len_observed": [512, 512, 512]}
        self.assertIsNone(observed_cache_len(item))

    def test_inconsistent_list(self):
        item = {"cache_len_observed": [512, 256, 512]}
        self.assertIsNone(observed_cache_len(item))

    def test_from_runs(self):
        item = {
            "runs": [
                {"success": True, "reuse_len": 512},
                {"success": True, "reuse_len": 512},
                {"success": True, "reuse_len": 512},
            ]
        }
        self.assertEqual(observed_cache_len(item), 512.0)


class PrefillRtTest(unittest.TestCase):
    def test_aggregate_latency_is_not_a_fallback(self):
        item = {"ttft_ms": 42.0}
        self.assertIsNone(prefill_rt(item))

    def test_from_runs(self):
        item = {
            "runs": [
                {"prefill_time_ms": 9.0, "success": True, "ttft_ms": 9.0},
                {"prefill_time_ms": 11.0, "success": True, "ttft_ms": 11.0},
                {"prefill_time_ms": 10.0, "success": True, "ttft_ms": 10.0},
            ]
        }
        self.assertEqual(prefill_rt(item), 10.0)


if __name__ == "__main__":
    unittest.main()
