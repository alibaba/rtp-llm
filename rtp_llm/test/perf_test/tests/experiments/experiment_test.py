"""Performance framework behavior and failure regressions."""

import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rtp_llm.test.perf_test.entry import main
from rtp_llm.test.perf_test.errors import ConfigurationError
from rtp_llm.test.perf_test.experiment import expand_experiment, run_experiment
from rtp_llm.test.perf_test.state import RunState
from rtp_llm.test.perf_test.tests.support.prefill import prefill_resume_config


class ExperimentTest(unittest.TestCase):

    def document(self):
        return {
            "experiment_schema_version": 1,
            "name": "compare",
            "base_config": prefill_resume_config(),
            "benchmarks": [{"name": "one", "overrides": {}}],
        }

    def test_zip_product_and_independent_copies(self):
        source = self.document()
        source["benchmarks"][0].update(
            zip_axes={"model.label": ["first", "second"]},
            product_axes={"workload.seed": [1, 2]},
        )
        tasks = expand_experiment(source)
        self.assertEqual(4, len(tasks))
        self.assertEqual([1, 2, 1, 2], [x["config"]["workload"]["seed"] for x in tasks])
        tasks[0]["config"]["model"]["label"] = "changed"
        self.assertEqual("first", tasks[1]["config"]["model"]["label"])
        self.assertNotIn("seed", source["base_config"]["workload"])

    def test_overlapping_paths_and_unsafe_axes_are_rejected(self):
        for overrides, axes in (
            ({"model.topology": {}}, {"model.topology.tp": [1]}),
            ({"execution.source.ref": "main"}, {}),
            ({}, {"engine.args.not_registered": []}),
        ):
            with self.subTest(overrides=overrides, axes=axes):
                source = self.document()
                source["benchmarks"][0].update(overrides=overrides, product_axes=axes)
                with self.assertRaises(ConfigurationError):
                    expand_experiment(source)

    def test_all_tasks_are_planned_before_any_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.document()
            source["base_config"]["output"] = {"result_root": str(root / "results")}
            source["benchmarks"].append(
                {"name": "bad", "overrides": {"workload.batch_sizes": [0]}}
            )
            path = root / "experiment.json"
            path.write_text(json.dumps(source))
            with patch("rtp_llm.test.perf_test.entry.execute_plan") as execute:
                with self.assertRaises(ConfigurationError):
                    run_experiment(path)
                execute.assert_not_called()
            self.assertFalse((root / "results").exists())

    def test_dry_run_does_not_write_and_freezes_each_task(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.document()
            source["base_config"]["output"] = {"result_root": str(root / "results")}
            source["benchmarks"][0]["name"] = "model.v2"
            source["benchmarks"][0]["product_axes"] = {"workload.seed": [3, 7]}
            path = root / "experiment.json"
            path.write_text(json.dumps(source))
            result = run_experiment(path, dry_run=True)
            repeated = run_experiment(path, dry_run=True)
            self.assertEqual(
                [run["name"] for run in result["runs"]],
                [run["name"] for run in repeated["runs"]],
            )
            first_ids = {Path(run["directory"]).name for run in result["runs"]}
            repeated_ids = {Path(run["directory"]).name for run in repeated["runs"]}
            self.assertEqual(2, len(first_ids))
            self.assertTrue(first_ids.isdisjoint(repeated_ids))
            for first, second in zip(result["runs"], repeated["runs"]):
                self.assertEqual(Path(first["directory"]).name, first["run_id"])
                self.assertRegex(first["run_id"], r"^[A-Za-z0-9_-]+$")
                self.assertEqual(
                    first["config_fingerprint"], second["config_fingerprint"]
                )
                self.assertEqual(
                    first["workload_fingerprint"], second["workload_fingerprint"]
                )
            self.assertEqual(2, len(result["runs"]))
            self.assertNotEqual(
                result["runs"][0]["workload_fingerprint"],
                result["runs"][1]["workload_fingerprint"],
            )
            self.assertFalse((root / "results").exists())

        with tempfile.TemporaryDirectory() as directory:
            source = self.document()
            base = source["base_config"]
            base["schema_version"] = 2
            base["execution"] = {
                "backend": "aone",
                "platform": "cuda13",
                "aone": {
                    "resource": "daily",
                    "gpu_type": "H20_CU13",
                    "max_parallel": 1,
                    "shard_count": 1,
                },
            }
            source["base_config"]["output"] = {
                "result_root": str(Path(directory) / "results")
            }
            source["benchmarks"][0]["product_axes"] = {
                "execution.gpus_per_machine": [1]
            }
            path = Path(directory) / "experiment.json"
            path.write_text(json.dumps(source))
            ordinary = run_experiment(path, dry_run=True)
            overridden = run_experiment(
                path, dry_run=True, overrides=("base_config.model.label=other",)
            )
            self.assertEqual(len(ordinary["runs"]), len(overridden["runs"]))

    def test_failures_do_not_hide_later_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.document()
            source["base_config"]["output"] = {"result_root": str(root / "results")}
            source["benchmarks"].append({"name": "two"})
            path = root / "experiment.json"
            path.write_text(json.dumps(source))
            with patch(
                "rtp_llm.test.perf_test.entry.execute_plan",
                side_effect=[RuntimeError("first failed"), {"state": "completed"}],
            ) as execute:
                result = run_experiment(path)
            self.assertEqual(2, execute.call_count)
            persisted = json.loads(
                (Path(result["directory"]) / "experiment_plan.json").read_text()
            )
            for task, record in zip(persisted["tasks"], result["runs"]):
                self.assertEqual(task["run_id"], record["run_id"])
                self.assertEqual(Path(record["directory"]).name, task["run_id"])
            self.assertEqual("failed", result["status"])
            self.assertEqual(
                ["failed", "completed"], [r["status"] for r in result["runs"]]
            )

    def test_returned_interruption_stops_the_whole_experiment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self.document()
            source["base_config"]["output"] = {"result_root": str(root / "results")}
            source["benchmarks"].append({"name": "must-not-start"})
            path = root / "experiment.json"
            path.write_text(json.dumps(source))
            with patch(
                "rtp_llm.test.perf_test.entry.execute_plan",
                return_value=SimpleNamespace(state=RunState.INTERRUPTED, exit_code=130),
            ) as execute:
                result = run_experiment(path)
            self.assertEqual(1, execute.call_count)
            self.assertEqual("interrupted", result["status"])
            self.assertEqual(130, result["exit_code"])
            self.assertEqual("interrupted", result["runs"][0]["status"])
            persisted = json.loads(
                (Path(result["directory"]) / "experiment_summary.json").read_text()
            )
            self.assertEqual(result, persisted)

    def test_cli_exit_code_preserves_first_failure_and_normalizes_signals(self):
        for codes, expected, run_codes, status in (
            ((-15,), 143, [143], "failed"),
            ((-2, 0), 130, [130], "interrupted"),
            ((0, -9, 0), 137, [0, 137, 0], "failed"),
            ((5, -15, 0), 5, [5, 143, 0], "failed"),
            ((-15, 5, 0), 143, [143, 5, 0], "failed"),
            ((0, 0), 0, [0, 0], "completed"),
        ):
            with self.subTest(codes=codes), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = self.document()
                source["base_config"]["output"] = {"result_root": str(root / "results")}
                source["benchmarks"] = [
                    {"name": f"task-{index}"} for index in range(len(codes))
                ]
                path = root / "experiment.json"
                path.write_text(json.dumps(source))
                output = io.StringIO()
                with patch(
                    "rtp_llm.test.perf_test.entry.execute_plan",
                    side_effect=[{"exit_code": code} for code in codes],
                ):
                    exit_code = main(["--config=" + str(path)], stdout=output)
                result = json.loads(output.getvalue())
                self.assertEqual(expected, exit_code)
                self.assertEqual(expected, result["exit_code"])
                self.assertEqual(status, result["status"])
                self.assertEqual(
                    run_codes, [run["exit_code"] for run in result["runs"]]
                )
                persisted = json.loads(
                    (Path(result["directory"]) / "experiment_summary.json").read_text()
                )
                self.assertEqual(result, persisted)

    def test_invalid_launch_arguments_are_rejected_before_any_experiment_output(self):
        for dry_run in (False, True):
            with self.subTest(
                dry_run=dry_run
            ), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = self.document()
                source["base_config"]["output"] = {"result_root": str(root / "results")}
                path = root / "experiment.json"
                path.write_text(json.dumps(source))
                with self.assertRaises(ConfigurationError):
                    run_experiment(
                        path, dry_run=dry_run, bazel_args=("--test_timeout=1",)
                    )
                self.assertFalse((root / "results").exists())


if __name__ == "__main__":
    unittest.main()
