import copy
import json
import tempfile
import unittest
from pathlib import Path
from statistics import median

from rtp_llm.test.perf_test.cache_grid.formula.prefill_formula_fit import (
    load_observations,
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
from rtp_llm.test.perf_test.cache_grid.runner.result_schema import single_request_metric

LATENCIES = (1452.7469000022393, 1432.8472790002706, 1425.4715130009572)


def grouped_metric(input_len=255744, cache_len=0, latencies=LATENCIES):
    return {
        "case_id": 0,
        "batch_size": 1,
        "prefix_policy": "independent",
        "request_groups": [
            {"count": 1, "input_len": input_len, "cache_len": cache_len}
        ],
        "input_len": input_len,
        "cache_len": cache_len,
        "execution_mode": "scheduler_fixed_batch",
        "measure_runs": 3,
        "success_runs": 3,
        "status": "ok",
        "shape_exact": True,
        "reuse_exact": True,
        "timing_valid": True,
        "cache_len_observed": [cache_len] * 3,
        "runs": [
            {
                "run_index": i,
                "valid": True,
                "completed_requests": 1,
                "batch_wall_time_ms": latency + 0.03,
                "requests": [
                    {
                        "success": True,
                        "input_len": input_len,
                        "output_len": 1,
                        "reuse_len": cache_len,
                        "prefill_time_ms": latency,
                        "ttft_ms": latency + 25,
                        "client_wall_time_ms": latency,
                        "ttft_source": "client_dashsc_grpc_input_ids_wall_max_new_tokens_1",
                        "shape_exact": True,
                        "reuse_exact": True,
                        "timing_valid": True,
                    }
                ],
            }
            for i, latency in enumerate(latencies)
        ],
    }


class GroupedSingleRequestTest(unittest.TestCase):
    def test_fit_svg_and_full_run_tpm_use_same_request_ttft(self):
        for cached in (0, 4096):
            item = grouped_metric(cache_len=cached)
            original = copy.deepcopy(item)
            with self.subTest(cached=cached), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "results.json"
                path.write_text(
                    json.dumps(
                        {
                            "schema_version": 2,
                            "mode": "prefix_cache_grid",
                            "metrics": [item],
                        }
                    )
                )
                observations, audit = load_observations([path])
                self.assertEqual(audit["rejected_counts"], {})
                self.assertEqual(audit["valid_observation_count"], 1)
                self.assertEqual(
                    audit["measurement_contracts"],
                    ["batch_max_server_first_token_cost_time_ms"],
                )
                self.assertEqual(observations[0].target_ms, median(LATENCIES))
                self.assertEqual(observations[0].cache_len, cached)
                self.assertEqual(static_rows(path, 1)[0]["rt"], median(LATENCIES))
                self.assertEqual(
                    interactive_rows(path, 1)[0]["prefill_rt"], median(LATENCIES)
                )
                runs = interactive_rows(path, 1, all_runs=True)
                self.assertEqual([row["prefill_rt"] for row in runs], list(LATENCIES))
                tpm = apply_z_metric(runs, "tpm-effective", cards=8)
                for row, latency in zip(tpm, LATENCIES):
                    self.assertAlmostEqual(
                        row["tpm_effective"], 255744 * 60000 / latency / 8
                    )
                self.assertEqual(
                    load_observations([path], estimator="min")[0][0].target_ms,
                    min(LATENCIES),
                )
            self.assertEqual(item, original)

    def test_invalid_nested_data_is_not_repaired_into_valid_measurements(self):
        mutations = [
            lambda m: m["runs"][0].update(valid=False),
            lambda m: m["runs"][0].update(requests=[]),
            lambda m: m["runs"][0]["requests"].append(
                dict(m["runs"][0]["requests"][0])
            ),
            lambda m: m["runs"][0]["requests"][0].update(success=False),
            lambda m: m["runs"][0]["requests"][0].update(input_len=2),
            lambda m: m["runs"][0]["requests"][0].update(output_len=2),
            lambda m: m["runs"][0]["requests"][0].update(reuse_len=4096),
            lambda m: m["runs"][0]["requests"][0].update(prefill_time_ms=float("nan")),
            lambda m: m.update(success_runs=2),
            lambda m: m["request_groups"][0].update(count=2),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.json"
            for index, mutate in enumerate(mutations):
                with self.subTest(index=index):
                    item = grouped_metric()
                    mutate(item)
                    path.write_text(
                        json.dumps(
                            {
                                "schema_version": 2,
                                "mode": "prefix_cache_grid",
                                "metrics": [item],
                            }
                        )
                    )
                    observations, audit = load_observations([path])
                    self.assertEqual(observations, [])
                    self.assertEqual(sum(audit["rejected_counts"].values()), 1)
                    self.assertEqual(static_rows(path, 1), [])
                    self.assertEqual(interactive_rows(path, 1, all_runs=True), [])

    def test_multiple_requests_are_not_collapsed_to_one_geometry(self):
        item = grouped_metric()
        item["batch_size"] = 2
        item["request_groups"][0]["count"] = 2
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "mode": "prefix_cache_grid",
                        "metrics": [item],
                    }
                )
            )
            observations, audit = load_observations([path], batch_size=2)
            self.assertEqual(observations, [])
            self.assertEqual(audit["rejected_counts"], {"invalid_round": 1})
            self.assertEqual(interactive_rows(path, 2, all_runs=True), [])

    def test_scalar_rows_are_unchanged(self):
        item = {"batch_size": 1, "input_len": 100, "cache_len_requested": 0, "runs": []}
        self.assertIs(single_request_metric(item), item)

    def test_client_timing_sources_do_not_change_server_contract(self):
        item = grouped_metric()
        item["runs"][0]["requests"][0]["ttft_source"] = "different_transport"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "mode": "prefix_cache_grid",
                        "metrics": [item],
                    }
                )
            )
            observations, audit = load_observations([path])
            self.assertEqual(len(observations), 1)
            self.assertEqual(
                audit["measurement_contracts"],
                ["batch_max_server_first_token_cost_time_ms"],
            )


class ReaderPolicyTest(unittest.TestCase):
    def test_diagnostic_reuse_mismatch_is_excluded_from_fit_and_static_chart(self):
        item = single_request_metric(grouped_metric(cache_len=4096))
        item.pop("request_groups")
        item.pop("execution_mode")
        item.update(
            status="invalid_reuse", reuse_exact=False, cache_len_observed=[2048] * 3
        )
        for run in item["runs"]:
            run.update(reuse_len=2048, reuse_exact=False)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "mode": "prefix_cache_grid",
                        "metrics": [item],
                    }
                )
            )
            self.assertEqual(load_observations([path])[0], [])
            self.assertEqual(static_rows(path, 1), [])
            rows = interactive_rows(path, 1)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["cache_len"], 2048)
            self.assertEqual(len(interactive_rows(path, 1, all_runs=True)), 3)

    def test_strict_chart_does_not_default_invalid_batch_to_one(self):
        item = {
            "input_len": 8192,
            "cache_len_requested": 0,
            "cache_len_observed": [0],
            "status": "ok",
            "measure_runs": 1,
            "success_runs": 1,
            "runs": [
                {"prefill_time_ms": 10, "success": True, "reuse_len": 0, "ttft_ms": 10}
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.json"
            for batch in (0, None, "invalid"):
                with self.subTest(batch=batch):
                    item["batch_size"] = batch
                    path.write_text(
                        json.dumps(
                            {
                                "schema_version": 2,
                                "mode": "prefix_cache_grid",
                                "metrics": [item],
                            }
                        )
                    )
                    self.assertEqual(static_rows(path, 1), [])


if __name__ == "__main__":
    unittest.main()
