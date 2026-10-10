import argparse
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from rtp_llm.test.perf_test.cache_grid.runner import cache_perf_ci as ci


class CachePerfCiTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.repo = Path(temporary.name)
        self.profile = self.repo / "profile.json"
        self.profile.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "engine": {"model_type": "qwen_2"},
                    "engine_env": {"PERF_PROFILE_RUNS": "0"},
                    "cache_grid": {"measure_runs": 2},
                }
            )
        )
        self.grid = self.repo / "grid.json"
        self.grid.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "generator": {"cache_alignment": 1024},
                    "cases": [
                        {
                            "case_id": i,
                            "batch_size": 1,
                            "input_len": 4096 + i * 1024,
                            "cache_len": i * 512,
                        }
                        for i in range(6)
                    ],
                }
            )
        )

    def args(self, **updates):
        values = dict(
            profile=self.profile,
            grid=self.grid,
            output_dir=self.repo / "generated",
            shards=3,
            gpu_type="SM100_ARM_CU13",
            gpu_count=4,
            estimated_total_seconds=18000,
            startup_seconds=0,
            target_seconds=6900,
            runs=None,
        )
        values.update(updates)
        return argparse.Namespace(**values)

    def prepare_args(self, **updates):
        values = dict(
            inputs_dir=self.repo / "inputs",
            model_label="Test model",
            model_type="qwen_2",
            checkpoint_path="/models/test",
            tokenizer_path=None,
            tp_size=4,
            dp_size=1,
            ep_size=None,
            pp_size=None,
            world_size=4,
            engine_setting=["max_seq_len=4096", "concurrency_limit=1"],
            engine_arg=["--reuse_cache=1", "act_type=BF16"],
            engine_env_setting=["WORLD_SIZE=4", "PERF_PROFILE_RUNS=0"],
            runtime_env_setting=["CC=/usr/bin/gcc"],
            config=["cuda13_arm"],
            bazel=None,
            runs=2,
            grid_mode="stratified",
            min_input_len=256,
            max_input_len=1024,
            alignment=128,
            cache_alignment=128,
            commit_tail_tokens=128,
            input_mode="stratified",
            input_points=3,
            cache_points_per_input=4,
            cache_ratio_points=1,
            fixed_cache_len=None,
            random_cache_count=None,
            min_cache_len=0,
            max_cache_len=None,
            compute_step=None,
            min_compute_len=None,
            batch_size=None,
            seed=104729,
            max_cases=100,
            allow_large_grid=False,
        )
        values.update(updates)
        return argparse.Namespace(**values)

    def test_prepare_generates_deterministic_valid_profile_and_grid(self):
        first = ci.create_inputs(self.prepare_args(), repo=self.repo)
        second = ci.create_inputs(
            self.prepare_args(inputs_dir=self.repo / "other-inputs"), repo=self.repo
        )
        self.assertEqual(first["artifacts"], second["artifacts"])
        ci.write_inputs(first)
        profile = json.loads((self.repo / "inputs" / ci.PROFILE_NAME).read_text())
        grid = json.loads((self.repo / "inputs" / ci.GRID_NAME).read_text())
        self.assertEqual(profile["engine"]["tokenizer_path"], "/models/test")
        self.assertEqual(profile["engine_args"]["act_type"], "BF16")
        self.assertEqual(profile["runtime_env"]["CC"], "/usr/bin/gcc")
        self.assertEqual(grid["profile_sha256"], first["profile_fingerprint"])
        self.assertEqual(grid["profile"], profile)
        self.assertEqual(grid["summary"]["estimated_measure_requests"], 18)

    def test_prepare_rejects_unsafe_or_ambiguous_settings(self):
        with self.assertRaisesRegex(ValueError, "repeats or overrides tp_size"):
            ci.create_inputs(
                self.prepare_args(engine_setting=["tp_size=8"]), repo=self.repo
            )
        with self.assertRaisesRegex(ValueError, "credentials"):
            ci.create_inputs(
                self.prepare_args(runtime_env_setting=["API_TOKEN=secret"]),
                repo=self.repo,
            )
        with self.assertRaisesRegex(ValueError, "TP x DP x PP"):
            ci.create_inputs(self.prepare_args(world_size=8), repo=self.repo)

    def test_plan_is_deterministic_and_covers_every_case_once(self):
        first = ci.create_plan(self.args(), repo=self.repo)
        second = ci.create_plan(self.args(), repo=self.repo)
        self.assertEqual(ci.plan_summary(first), ci.plan_summary(second))
        ids = [case for shard in first["shards"] for case in shard["case_ids"]]
        self.assertEqual(sorted(ids), list(range(6)))
        self.assertEqual(len(ids), len(set(ids)))
        build = first["artifacts"][ci.BUILD_NAME].decode()
        self.assertIn('name = "all"', build)
        self.assertIn('gpu_type = "SM100_ARM_CU13"', build)
        self.assertIn('"PERF_PROFILE_RUNS": "0"', build)
        environment = first["artifacts"][ci.ENV_NAME].decode()
        self.assertIn("export PERF_PROFILE_RUNS=0", environment)

    def test_plan_rejects_shared_seed_and_too_many_shards(self):
        profile = json.loads(self.profile.read_text())
        profile["engine_args"] = {"cache_shared_seed": 1}
        self.profile.write_text(json.dumps(profile))
        with self.assertRaisesRegex(ValueError, "cache_shared_seed"):
            ci.create_plan(self.args(), repo=self.repo)
        self.profile.write_text(json.dumps({"schema_version": 1}))
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            ci.create_plan(self.args(shards=7), repo=self.repo)

    def test_plan_rejects_topology_and_action_timeout_mismatches(self):
        profile = json.loads(self.profile.read_text())
        profile["engine"]["world_size"] = 2
        self.profile.write_text(json.dumps(profile))
        with self.assertRaisesRegex(ValueError, "profile topology"):
            ci.create_plan(self.args(), repo=self.repo)
        profile["engine"]["world_size"] = 4
        self.profile.write_text(json.dumps(profile))
        with self.assertRaisesRegex(ValueError, "configured 7000s"):
            ci.create_plan(self.args(target_seconds=7000), repo=self.repo)

    def test_collect_validates_and_materializes_batch_manifest(self):
        plan = ci.create_plan(self.args(), repo=self.repo)
        ci.write_plan(plan)
        testlogs = self.repo / "testlogs"
        for shard in plan["shards"]:
            output = (
                testlogs
                / plan["package"]
                / shard["name"]
                / "test.outputs"
                / "outputs.zip"
            )
            output.parent.mkdir(parents=True)
            payload = {
                "schema_version": 2,
                "complete": True,
                "grid_sha256": shard["grid_sha256"],
                "profile_sha256": plan["profile_fingerprint"],
                "measure_runs": plan["measure_runs"],
                "metrics": [{"case_id": case_id} for case_id in shard["case_ids"]],
            }
            with zipfile.ZipFile(output, "w") as archive:
                archive.writestr("cache_grid_results.json", json.dumps(payload))
        args = argparse.Namespace(
            plan=Path(plan["output_dir"]) / ci.PLAN_NAME,
            result_root=self.repo / "results",
            bazel_testlogs=testlogs,
            partial=False,
        )
        manifest, code = ci.collect_results(args, repo=self.repo)
        self.assertEqual(code, 0)
        self.assertEqual(manifest["collected_case_ids"], list(range(6)))
        saved = json.loads((args.result_root / "batch_runs.json").read_text())
        self.assertEqual(len(saved["runs"]), 3)

    def test_collect_rejects_result_from_wrong_grid(self):
        plan = ci.create_plan(self.args(shards=1, target_seconds=None), repo=self.repo)
        ci.write_plan(plan)
        shard = plan["shards"][0]
        output = (
            self.repo
            / "testlogs"
            / plan["package"]
            / shard["name"]
            / "test.outputs"
            / "outputs.zip"
        )
        output.parent.mkdir(parents=True)
        with zipfile.ZipFile(output, "w") as archive:
            archive.writestr(
                "cache_grid_results.json",
                json.dumps(
                    {
                        "complete": True,
                        "grid_sha256": "wrong",
                        "profile_sha256": plan["profile_fingerprint"],
                        "measure_runs": plan["measure_runs"],
                        "metrics": [],
                    }
                ),
            )
        args = argparse.Namespace(
            plan=Path(plan["output_dir"]) / ci.PLAN_NAME,
            result_root=self.repo / "results",
            bazel_testlogs=self.repo / "testlogs",
            partial=False,
        )
        with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
            ci.collect_results(args, repo=self.repo)

    def test_partial_collect_treats_empty_outputs_as_missing(self):
        plan = ci.create_plan(self.args(shards=2, target_seconds=None), repo=self.repo)
        ci.write_plan(plan)
        testlogs = self.repo / "testlogs"
        first = plan["shards"][0]
        output = (
            testlogs / plan["package"] / first["name"] / "test.outputs" / "outputs.zip"
        )
        output.parent.mkdir(parents=True)
        with zipfile.ZipFile(output, "w") as archive:
            archive.writestr(
                "cache_grid_results.json",
                json.dumps(
                    {
                        "complete": False,
                        "grid_sha256": first["grid_sha256"],
                        "profile_sha256": plan["profile_fingerprint"],
                        "measure_runs": plan["measure_runs"],
                        "metrics": [{"case_id": first["case_ids"][0]}],
                    }
                ),
            )
        failed = plan["shards"][1]
        output = (
            testlogs / plan["package"] / failed["name"] / "test.outputs" / "outputs.zip"
        )
        output.parent.mkdir(parents=True)
        with zipfile.ZipFile(output, "w") as archive:
            archive.writestr("test.log", "model failed before first checkpoint")
        args = argparse.Namespace(
            plan=Path(plan["output_dir"]) / ci.PLAN_NAME,
            result_root=self.repo / "results",
            bazel_testlogs=testlogs,
            partial=True,
        )
        manifest, _ = ci.collect_results(args, repo=self.repo)
        self.assertEqual(manifest["missing_shards"], [failed["name"]])
        self.assertEqual(manifest["collected_case_ids"], [first["case_ids"][0]])


if __name__ == "__main__":
    unittest.main()
