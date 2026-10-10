import json
import tempfile
import unittest
from pathlib import Path

from rtp_llm.test.perf_test.cache_grid.plot.generate_prefill_3d_chart import (
    load_rows,
    render_clean,
    render_cold_miss_2d,
)


def _sample_payload():
    return {
        "schema_version": 2,
        "mode": "prefix_cache_grid",
        "metrics": [
            {
                "batch_size": 1,
                "input_len": 1024,
                "cache_len_requested": 256,
                "cache_len_observed": [256, 256, 256],
                "measure_runs": 3,
                "success_runs": 3,
                "status": "ok",
                "runs": [
                    {
                        "prefill_time_ms": value,
                        "success": True,
                        "reuse_len": 256,
                        "ttft_ms": value,
                    }
                    for value in (9.0, 10.0, 11.0)
                ],
            }
        ],
    }


class GeneratePrefill3dChartTest(unittest.TestCase):
    def test_render_uses_compute_cache_ttft_axis_order(self):
        payload = _sample_payload()
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "results.json"
            source.write_text(json.dumps(payload), encoding="utf-8")
            rows = load_rows(source, 1)
            self.assertEqual(rows[0]["compute"], 768)
            svg = render_clean(rows, source, 1)
        self.assertIn("compute tokens (X)", svg)
        self.assertIn("cached tokens (Y)", svg)
        self.assertIn("Server first-token latency (Z, ms)", svg)

    def test_cold_chart_uses_escaped_model_label(self):
        rows = [{"input": 1024.0, "compute": 1024.0, "cache": 0.0, "rt": 10.0}]
        svg = render_cold_miss_2d(rows, Path("results.json"), 1, "Qwen <test>")
        self.assertIn("Qwen &lt;test&gt;", svg)
        self.assertNotIn("DeepSeek", svg)

    def test_cold_chart_handles_long_sequences_without_inset_rows(self):
        rows = [
            {
                "input": 262144.0,
                "compute": 262144.0,
                "cache": 0.0,
                "rt": 10.0,
            }
        ]

        svg = render_cold_miss_2d(rows, Path("results.json"), 1)

        self.assertIn("短序列区间无样本，未绘制插图", svg)

    def test_default_title_is_generic(self):
        payload = _sample_payload()
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "results.json"
            source.write_text(json.dumps(payload), encoding="utf-8")
            rows = load_rows(source, 1)
            svg = render_clean(rows, source, 1)
        self.assertIn("Model Prefill", svg)
        self.assertNotIn("DeepSeek", svg)

    def test_custom_title(self):
        payload = _sample_payload()
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "results.json"
            source.write_text(json.dumps(payload), encoding="utf-8")
            rows = load_rows(source, 1)
            svg = render_clean(rows, source, 1, title="Custom Model Chart")
        self.assertIn("Custom Model Chart", svg)
        self.assertNotIn("DeepSeek-V4-Pro", svg)

    def test_dot_colour_depth_represents_local_query_density(self):
        rows = [
            {"compute": 10.0, "cache": 10.0, "rt": 1.0, "input": 20.0},
            {"compute": 11.0, "cache": 11.0, "rt": 2.0, "input": 22.0},
            {"compute": 12.0, "cache": 12.0, "rt": 3.0, "input": 24.0},
            {"compute": 1000.0, "cache": 1000.0, "rt": 4.0, "input": 2000.0},
        ]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "results.json"
            source.write_text(json.dumps(_sample_payload()), encoding="utf-8")
            svg = render_clean(rows, source, 1)
        self.assertEqual(svg.count('data-query-density="3"'), 3)
        self.assertEqual(svg.count('data-query-density="1"'), 1)
        self.assertIn('fill="#1e3a8a"', svg)
        self.assertIn('fill="#dbeafe"', svg)
        self.assertIn("Query distribution density", svg)
        self.assertIn("sparse · local 16×16 compute/cache bins · dense (max 3)", svg)

    def test_annotate_cold_threshold(self):
        payload = {
            "schema_version": 2,
            "mode": "prefix_cache_grid",
            "metrics": [
                {
                    "batch_size": 1,
                    "input_len": 2048,
                    "cache_len_requested": 0,
                    "cache_len_observed": [0, 0, 0],
                    "measure_runs": 3,
                    "success_runs": 3,
                    "status": "ok",
                    "runs": [
                        {
                            "prefill_time_ms": value,
                            "success": True,
                            "reuse_len": 0,
                            "ttft_ms": value,
                        }
                        for value in (50.0, 51.0, 52.0)
                    ],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "results.json"
            source.write_text(json.dumps(payload), encoding="utf-8")
            rows = load_rows(source, 1)
            svg_high = render_clean(rows, source, 1, annotate_cold_threshold=1024)
            svg_low = render_clean(rows, source, 1, annotate_cold_threshold=4096)
        self.assertIn("cold", svg_high)
        self.assertNotIn("cold", svg_low)


if __name__ == "__main__":
    unittest.main()
