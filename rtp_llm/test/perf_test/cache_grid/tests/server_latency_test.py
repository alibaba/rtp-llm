"""All benchmark consumers select server first_token_cost_time from raw results."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rtp_llm.test.perf_test.cache_grid.formula.prefill_formula_fit import (
    load_observations,
)
from rtp_llm.test.perf_test.cache_grid.plot.generate_batch_interactive_chart import (
    analyze_results,
)
from rtp_llm.test.perf_test.cache_grid.plot.generate_batch_interactive_chart import (
    main as batch_main,
)
from rtp_llm.test.perf_test.cache_grid.plot.generate_prefill_3d_chart import (
    load_rows as static_rows,
)
from rtp_llm.test.perf_test.cache_grid.plot.generate_prefill_interactive_chart import (
    apply_z_metric,
)
from rtp_llm.test.perf_test.cache_grid.plot.generate_prefill_interactive_chart import (
    load_rows as interactive_rows,
)


def result(grouped=False):
    requests = [
        dict(
            success=True,
            input_len=16,
            output_len=1,
            reuse_len=8,
            prefill_time_ms=server,
            wait_time_ms=2,
            ttft_ms=1000 + server,
            client_wall_time_ms=1000 + server,
        )
        for server in (10, 20, 30)
    ]
    metric = dict(
        case_id=1,
        batch_size=1,
        input_len=16,
        cache_len_requested=8,
        status="ok",
        measure_runs=3,
        success_runs=3,
        runs=requests,
    )
    if grouped:
        metric.update(
            batch_size=2,
            request_groups=[dict(count=2, input_len=16, cache_len=8)],
            runs=[
                dict(
                    valid=True,
                    completed_requests=2,
                    batch_wall_time_ms=9000,
                    requests=[r, dict(r, prefill_time_ms=r["prefill_time_ms"] + 5)],
                )
                for r in requests
            ],
        )
    return dict(
        schema_version=2, mode="prefix_cache_grid", complete=True, metrics=[metric]
    )


class ServerLatencyTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "cache_grid_results.json"

    def test_fit_charts_and_tpm_use_server_time_without_subtracting_wait(self):
        self.path.write_text(json.dumps(result()))
        rows, audit = load_observations([self.path])
        self.assertEqual(rows[0].target_ms, 20)
        self.assertEqual(
            audit["measurement_contracts"],
            ["batch_max_server_first_token_cost_time_ms"],
        )
        self.assertEqual(static_rows(self.path, 1)[0]["rt"], 20)
        points = interactive_rows(self.path, 1, all_runs=True)
        self.assertEqual([p["prefill_rt"] for p in points], [10, 20, 30])
        self.assertEqual(
            apply_z_metric(points, "tpm-effective")[0]["tpm_effective"], 16 * 60000 / 10
        )

    def test_server_only_results_do_not_require_client_diagnostics(self):
        data = result()
        for run in data["metrics"][0]["runs"]:
            del run["ttft_ms"], run["client_wall_time_ms"]
        self.path.write_text(json.dumps(data))
        self.assertEqual(load_observations([self.path])[0][0].target_ms, 20)

    def test_missing_server_value_never_falls_back_to_client(self):
        data = result()
        del data["metrics"][0]["runs"][0]["prefill_time_ms"]
        self.path.write_text(json.dumps(data))
        for reader in (
            lambda: load_observations([self.path]),
            lambda: static_rows(self.path, 1),
            lambda: interactive_rows(self.path, 1),
        ):
            with self.assertRaisesRegex(ValueError, "prefill_time_ms"):
                reader()

    def test_invalid_server_value_is_rejected_despite_valid_client_time(self):
        for value in (0, -1, float("nan"), float("inf")):
            data = result()
            data["metrics"][0]["runs"][0]["prefill_time_ms"] = value
            self.path.write_text(json.dumps(data))
            with self.subTest(value=value):
                self.assertEqual(load_observations([self.path])[0], [])
                self.assertEqual(static_rows(self.path, 1), [])

    def test_batch_aggregates_max_server_time_then_median_rounds(self):
        data = result(grouped=True)
        original = copy.deepcopy(data)
        rows = analyze_results(data)
        self.assertEqual(rows[0]["formal_rounds_ms"], [15, 25, 35])
        self.assertEqual(rows[0]["median_batch_first_token_cost_time_ms"], 25)
        self.assertEqual(rows[0]["cached_tokens"], 16)
        self.assertEqual(data, original)
        data["metrics"][0]["runs"][0]["requests"][0]["prefill_time_ms"] = 0
        self.assertEqual(analyze_results(data), [])

    def test_batch_cli_renders_raw_json_and_rejects_old_intermediate(self):
        output = self.path.with_suffix(".html")
        self.path.write_text(json.dumps(result(grouped=True)))
        with patch(
            "sys.argv", ["chart", "--input", str(self.path), "--output", str(output)]
        ):
            batch_main()
        page = output.read_text()
        self.assertIn('"formal_rounds_ms": [15.0, 25.0, 35.0]', page)
        self.assertIn("每轮取批内请求的服务端 first_token_cost_time 最大值", page)
        with self.assertRaises(ValueError):
            analyze_results(
                {
                    "measurement_contract": "batch_input_ids_wall_barrier_to_last_response_ms",
                    "rows": [],
                }
            )

    def test_analysis_accepts_only_raw_json_not_target_csv(self):
        path = self.path.with_suffix(".csv")
        path.write_text("batch_size,input_len,cache_len,target_ms\n1,16,8,999\n")
        with self.assertRaises(ValueError):
            load_observations([path])
        with self.assertRaises(ValueError):
            static_rows(path, 1)


if __name__ == "__main__":
    unittest.main()
