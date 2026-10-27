import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from rtp_llm.test.perf_test.cache_grid.config import run_request as request
from rtp_llm.test.perf_test.cache_grid.runner import cache_perf_ci as ci

MODEL = '{"type":"qwen_2","checkpoint":"Qwen/Qwen2.5-0.5B-Instruct"}'
ENGINE = '{"max_seq_len":4096,"seq_size_per_block":64,' '"args":{"act_type":"BF16"}}'
GRID = '{"cache_alignment":64,"compute_step":[1024,256]}'


class RunRequestTest(unittest.TestCase):
    def resolve(self, **updates):
        values = dict(
            model_json=MODEL,
            engine_json=ENGINE,
            grid_json=GRID,
            engine_env_json="{}",
            runtime_env_json="{}",
        )
        values.update(updates)
        return request.resolve_run_request(**values)

    def test_minimum_request_derives_defaults(self):
        resolved = request.resolve_run_request(MODEL, ENGINE, GRID, "{}", "{}")
        self.assertEqual(resolved.model["label"], "Qwen2.5-0.5B-Instruct")
        self.assertEqual(resolved.model["tokenizer"], resolved.model["checkpoint"])
        self.assertEqual(
            resolved.model["topology"], {"tp": 1, "dp": 1, "pp": 1, "ep": 1}
        )
        self.assertEqual(resolved.world_size, 1)
        self.assertEqual(resolved.gpu_count, 1)
        self.assertEqual(resolved.grid["max_input_len"], 4095)
        self.assertEqual(resolved.grid["measure_runs"], 3)
        self.assertEqual(resolved.grid["random_cache_count"], 8)
        self.assertEqual(resolved.grid["max_cases"], 200)
        self.assertEqual(resolved.grid["commit_tail_tokens"], 64)
        self.assertEqual(resolved.grid["min_compute_len"], 256)
        self.assertEqual(resolved.grid["seed"], 104729)
        self.assertEqual(resolved.engine["max_batch_tokens_size"], 4096)
        self.assertEqual(resolved.engine["max_context_batch_size"], 1)
        self.assertEqual(resolved.engine["concurrency_limit"], 1)
        self.assertEqual(
            resolved.engine["args"],
            {
                "act_type": "BF16",
                "reuse_cache": 1,
                "enable_device_cache": 1,
                "enable_cuda_graph": 0,
            },
        )
        self.assertEqual(
            resolved.engine_env,
            {
                "WORLD_SIZE": "1",
                "GPU_COUNT": "1",
                "LOCAL_WORLD_SIZE": "1",
                "WORLD_RANK": "0",
                "TP_SIZE": "1",
                "DP_SIZE": "1",
                "EP_SIZE": "1",
                "TOKENIZERS_PARALLELISM": "false",
            },
        )
        with self.assertRaises(FrozenInstanceError):
            resolved.world_size = 2

    def test_tp_dp_pp_derive_gpu_count_without_multiplying_ep(self):
        model = json.dumps(
            {
                "type": "deepseek_v4",
                "checkpoint": "/models/dsv4",
                "topology": {"tp": 2, "dp": 4, "pp": 1, "ep": 4},
            }
        )
        resolved = request.resolve_run_request(model, ENGINE, GRID, "{}", "{}")
        self.assertEqual((resolved.world_size, resolved.gpu_count), (8, 8))
        self.assertEqual(resolved.engine_env["WORLD_SIZE"], "8")
        resolved = self.resolve(
            model_json=json.dumps(
                {
                    "type": "qwen_2",
                    "checkpoint": "/models/qwen/",
                    "topology": {"tp": 2, "dp": 1, "pp": 3, "ep": 6},
                }
            )
        )
        self.assertEqual((resolved.world_size, resolved.gpu_count), (6, 6))
        self.assertEqual(resolved.model["label"], "qwen")

    def test_rejects_reserved_unknown_and_unsafe_values(self):
        with self.assertRaisesRegex(ValueError, "model.world_size"):
            request.resolve_run_request(
                '{"type":"qwen_2","checkpoint":"x","world_size":2}',
                ENGINE,
                GRID,
                "{}",
                "{}",
            )
        with self.assertRaisesRegex(ValueError, "engine.unknown"):
            request.resolve_run_request(
                MODEL, ENGINE[:-1] + ',"unknown":1}', GRID, "{}", "{}"
            )
        with self.assertRaisesRegex(ValueError, "engine_env.API_TOKEN"):
            request.resolve_run_request(MODEL, ENGINE, GRID, '{"API_TOKEN":"x"}', "{}")
        with self.assertRaisesRegex(ValueError, "runtime_env.PATH"):
            request.resolve_run_request(
                MODEL, ENGINE, GRID, "{}", '{"PATH":"$PATH:/x"}'
            )

    def test_rejects_managed_gpu_and_topology_environment_in_both_categories(self):
        for category in ("engine_env", "runtime_env"):
            for name in (
                "GPU_COUNT",
                "LOCAL_WORLD_SIZE",
                "WORLD_RANK",
                "TP_SIZE",
                "DP_SIZE",
                "EP_SIZE",
                "CUDA_VISIBLE_DEVICES",
                "HIP_VISIBLE_DEVICES",
            ):
                with self.subTest(category=category, name=name):
                    with self.assertRaisesRegex(ValueError, rf"{category}\.{name}"):
                        self.resolve(**{f"{category}_json": json.dumps({name: "0"})})

    def test_rejects_cross_category_and_grid_conflicts(self):
        with self.assertRaisesRegex(ValueError, "max_input_len.*max_seq_len"):
            request.resolve_run_request(
                MODEL,
                ENGINE,
                '{"max_input_len":4096,"cache_alignment":64,'
                '"compute_step":[1024,256]}',
                "{}",
                "{}",
            )
        with self.assertRaisesRegex(ValueError, "coarse-to-fine"):
            request.resolve_run_request(
                MODEL,
                ENGINE,
                '{"cache_alignment":64,"compute_step":[256,1024]}',
                "{}",
                "{}",
            )
        with self.assertRaisesRegex(ValueError, "fixed_cache_len.*random_cache_count"):
            request.resolve_run_request(
                MODEL,
                ENGINE,
                '{"cache_alignment":64,"compute_step":[1024,256],'
                '"fixed_cache_len":512,"random_cache_count":2}',
                "{}",
                "{}",
            )
        with self.assertRaisesRegex(ValueError, "both sections"):
            request.resolve_run_request(
                MODEL, ENGINE, GRID, '{"CC":"gcc"}', '{"CC":"clang"}'
            )

    def test_requires_json_objects_in_each_category(self):
        for category in ("model", "engine", "grid", "engine_env", "runtime_env"):
            for raw in ("[]", "null", "42", "{"):
                with self.subTest(category=category, raw=raw):
                    with self.assertRaisesRegex(ValueError, category):
                        self.resolve(**{f"{category}_json": raw})

    def test_engine_args_reject_protected_fields_and_cli_aliases(self):
        for name in (
            "world_size",
            "WORLD_SIZE",
            "gpu_count",
            "TOKENIZERS_PARALLELISM",
            "tp",
            "dp",
            "pp",
            "ep",
            "tp_size",
            "dp_size",
            "pp_size",
            "ep_size",
            "TP_SIZE",
            "--tp_size",
            "--world_size",
            "local_world_size",
            "world_rank",
            "type",
            "label",
            "model_type",
            "model_label",
            "checkpoint",
            "checkpoint_path",
            "tokenizer",
            "tokenizer_path",
            "max_seq_len",
            "--max_seq_len",
            "seq_size_per_block",
            "kernel_seq_size_per_block",
            "max_batch_tokens_size",
            "max_context_batch_size",
            "concurrency_limit",
            "max_input_len",
            "cache_alignment",
            "commit_tail_tokens",
            "compute_step",
            "min_compute_len",
            "min_cache_len",
            "max_cache_len",
            "fixed_cache_len",
            "random_cache_count",
            "measure_runs",
            "batch_size",
            "seed",
            "max_cases",
        ):
            with self.subTest(name=name):
                engine = json.loads(ENGINE)
                engine["args"][name] = 1
                with self.assertRaisesRegex(ValueError, f"engine.args.{name}"):
                    self.resolve(engine_json=json.dumps(engine))

    def test_rejects_duplicate_json_keys_at_every_object_depth(self):
        cases = (
            ("model", MODEL[:-1] + ',"type":"qwen_2"}', "model.type"),
            ("model", MODEL[:-1] + ',"topology":{"tp":2,"tp":1}}', "model.topology.tp"),
            ("engine", ENGINE[:-1] + ',"max_seq_len":4096}', "engine.max_seq_len"),
            (
                "engine",
                ENGINE.replace(
                    '"act_type":"BF16"', '"act_type":"BF16","act_type":"BF16"'
                ),
                "engine.args.act_type",
            ),
            ("grid", GRID[:-1] + ',"cache_alignment":64}', "grid.cache_alignment"),
            (
                "engine_env",
                '{"BUILD_TAG":"old","BUILD_TAG":"new"}',
                "engine_env.BUILD_TAG",
            ),
            ("runtime_env", '{"CC":"gcc","CC":"clang"}', "runtime_env.CC"),
            (
                "engine",
                ENGINE.replace(
                    '"act_type":"BF16"',
                    '"act_type":"BF16","nested":{"items":[{"key":1,"key":2}]}',
                ),
                r"engine.args.nested.items\[0\].key",
            ),
        )
        for category, raw, path in cases:
            with self.subTest(category=category, path=path):
                with self.assertRaisesRegex(ValueError, path + ".*duplicate"):
                    self.resolve(**{f"{category}_json": raw})

    def test_adapter_preserves_numeric_looking_environment_strings(self):
        engine_env = {"BUILD_TAG": "00123", "ENGINE_VERSION": "1.10", "STEP": "+01"}
        runtime_env = {"PACKAGE_VERSION": "1.10", "BUILD_NUMBER": "00123", "EMPTY": ""}
        resolved = self.resolve(
            engine_env_json=json.dumps(engine_env),
            runtime_env_json=json.dumps(runtime_env),
        )
        bundle = ci.build_input_artifacts(
            request.to_prepare_namespace(
                resolved, inputs_dir=Path("inputs"), configs=("cuda13",)
            )
        )
        profile = json.loads(bundle["artifacts"][ci.PROFILE_NAME])
        for section, expected in (
            ("engine_env", engine_env),
            ("runtime_env", runtime_env),
        ):
            for name, value in expected.items():
                with self.subTest(section=section, name=name):
                    self.assertEqual(profile[section][name], value)

    def test_rejects_invalid_fields_with_full_paths(self):
        cases = (
            ("model", {"type": ""}, "model.type"),
            ("model", {"checkpoint": 1}, "model.checkpoint"),
            ("model", {"label": ["bad"]}, "model.label"),
            ("model", {"tokenizer": 3}, "model.tokenizer"),
            ("model", {"topology": []}, "model.topology"),
            ("model", {"topology": {"tp_size": 2}}, "model.topology.tp_size"),
            ("model", {"topology": {"tp": True}}, "model.topology.tp"),
            ("model", {"topology": {"dp": 0}}, "model.topology.dp"),
            ("engine", {"max_seq_len": False}, "engine.max_seq_len"),
            ("engine", {"seq_size_per_block": 0}, "engine.seq_size_per_block"),
            ("engine", {"max_batch_tokens_size": -1}, "engine.max_batch_tokens_size"),
            (
                "engine",
                {"max_context_batch_size": 1.5},
                "engine.max_context_batch_size",
            ),
            ("engine", {"concurrency_limit": "1"}, "engine.concurrency_limit"),
            (
                "engine",
                {"kernel_seq_size_per_block": 0},
                "engine.kernel_seq_size_per_block",
            ),
            ("engine", {"args": []}, "engine.args"),
            ("engine", {"args": {}}, "engine.args.act_type"),
            (
                "engine",
                {"args": {"act_type": "BF16", "flag": True}},
                "engine.args.flag",
            ),
            (
                "engine",
                {"args": {"act_type": "BF16", "scale": float("inf")}},
                "engine.args.scale",
            ),
            ("grid", {"unknown": 1}, "grid.unknown"),
            ("grid", {"cache_alignment": False}, "grid.cache_alignment"),
            ("grid", {"measure_runs": 0}, "grid.measure_runs"),
            ("grid", {"min_cache_len": -1}, "grid.min_cache_len"),
            ("grid", {"seed": True}, "grid.seed"),
            ("grid", {"compute_step": [1024, 1024]}, "grid.compute_step"),
            ("grid", {"compute_step": [True]}, "grid.compute_step"),
            ("grid", {"compute_step": []}, "grid.compute_step"),
        )
        for category, updates, path in cases:
            with self.subTest(path=path, updates=updates):
                raw = json.loads(
                    {"model": MODEL, "engine": ENGINE, "grid": GRID}[category]
                )
                raw.update(updates)
                with self.assertRaisesRegex(ValueError, path.replace(".", r"\.")):
                    self.resolve(**{f"{category}_json": json.dumps(raw)})

    def test_environment_values_are_literal_finite_scalars(self):
        for category in ("engine_env", "runtime_env"):
            for name, value in (
                ("BAD-NAME", "x"),
                ("API_TOKEN", "secret"),
                ("FEATURE", True),
                ("FEATURE", None),
                ("FEATURE", []),
                ("FEATURE", float("nan")),
                ("FEATURE", float("inf")),
                ("PATH", "$PATH:/x"),
                ("CC", "`which gcc`"),
                ("CC", "a\x00b"),
                ("WORLD_SIZE", 1),
                ("TOKENIZERS_PARALLELISM", "true"),
            ):
                with self.subTest(category=category, name=name, value=value):
                    with self.assertRaisesRegex(ValueError, f"{category}.{name}"):
                        self.resolve(**{f"{category}_json": json.dumps({name: value})})
        resolved = self.resolve(
            engine_env_json='{"PERF_PROFILE_RUNS":0,"SCALE":0.5}',
            runtime_env_json='{"CC":"/opt/gcc/bin/gcc","CXX":"/opt/gcc/bin/g++"}',
        )
        self.assertEqual(resolved.engine_env["PERF_PROFILE_RUNS"], "0")
        self.assertEqual(resolved.engine_env["SCALE"], "0.5")
        self.assertEqual(resolved.runtime_env["CC"], "/opt/gcc/bin/gcc")

    def test_explicit_null_numeric_fields_do_not_activate_defaults(self):
        for category, fields in (
            (
                "engine",
                (
                    "max_seq_len",
                    "seq_size_per_block",
                    "kernel_seq_size_per_block",
                    "max_batch_tokens_size",
                    "max_context_batch_size",
                    "concurrency_limit",
                ),
            ),
            (
                "grid",
                (
                    "max_input_len",
                    "cache_alignment",
                    "commit_tail_tokens",
                    "compute_step",
                    "min_compute_len",
                    "min_cache_len",
                    "max_cache_len",
                    "fixed_cache_len",
                    "random_cache_count",
                    "measure_runs",
                    "batch_size",
                    "seed",
                    "max_cases",
                ),
            ),
        ):
            for field in fields:
                with self.subTest(category=category, field=field):
                    raw = json.loads({"engine": ENGINE, "grid": GRID}[category])
                    raw[field] = None
                    with self.assertRaisesRegex(ValueError, f"{category}.{field}"):
                        self.resolve(**{f"{category}_json": json.dumps(raw)})

    def test_explicit_model_label_and_tokenizer_require_nonempty_strings(self):
        for field in ("label", "tokenizer"):
            for value in (None, "", False, 0, [], {}):
                with self.subTest(field=field, value=value):
                    raw = json.loads(MODEL)
                    raw[field] = value
                    with self.assertRaisesRegex(ValueError, f"model.{field}"):
                        self.resolve(model_json=json.dumps(raw))

    def test_adapter_omits_ep_size_when_ep_is_not_requested(self):
        resolved = self.resolve()
        bundle = ci.build_input_artifacts(
            request.to_prepare_namespace(
                resolved, inputs_dir=Path("inputs"), configs=("cuda13",)
            )
        )

        self.assertNotIn("ep_size", bundle["profile_payload"]["engine"])
        self.assertEqual(resolved.engine_env["EP_SIZE"], "1")

    def test_adapter_preserves_explicit_ep_size_one(self):
        model = json.loads(MODEL)
        model["topology"] = {"ep": 1}
        resolved = self.resolve(model_json=json.dumps(model))
        bundle = ci.build_input_artifacts(
            request.to_prepare_namespace(
                resolved, inputs_dir=Path("inputs"), configs=("cuda13",)
            )
        )

        self.assertEqual(bundle["profile_payload"]["engine"]["ep_size"], 1)

    def test_adapter_generates_default_random_cache_slices(self):
        resolved = self.resolve()
        bundle = ci.build_input_artifacts(
            request.to_prepare_namespace(
                resolved, inputs_dir=Path("inputs"), configs=("cuda13",)
            )
        )
        sampling = bundle["grid_payload"]["generator"]["cache_sampling"]
        self.assertEqual(sampling["mode"], "random")
        self.assertEqual(len(sampling["values"]), 8)
        self.assertEqual(bundle["profile_payload"]["bazel"]["executable"], "bazelisk")
        self.assertTrue(
            all(case["input_len"] <= 4095 for case in bundle["grid_payload"]["cases"])
        )

    def test_adapter_builds_profile_and_bounded_grid_without_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            inputs_dir = Path(directory) / "inputs"
            resolved = self.resolve(
                model_json='{"label":"DSV4","type":"deepseek_v4","checkpoint":"/models/dsv4",'
                '"tokenizer":"/models/tokenizer","topology":{"tp":2,"dp":2,"pp":2,"ep":4}}',
                engine_json='{"max_seq_len":4096,"seq_size_per_block":64,'
                '"kernel_seq_size_per_block":128,"max_batch_tokens_size":8192,'
                '"max_context_batch_size":2,"concurrency_limit":2,'
                '"args":{"act_type":"BF16","enable_cuda_graph":1,"scale":0.5}}',
                grid_json='{"cache_alignment":64,"compute_step":[1024,256],'
                '"fixed_cache_len":512,"measure_runs":2,"batch_size":2}',
                engine_env_json='{"PERF_PROFILE_RUNS":0}',
                runtime_env_json='{"CC":"/usr/bin/gcc"}',
            )
            self.assertEqual(resolved.grid["fixed_cache_len"], 512)
            self.assertIsNone(resolved.grid["random_cache_count"])
            args = request.to_prepare_namespace(
                resolved, inputs_dir=inputs_dir, configs=("cuda13",), bazel="bazelisk"
            )
            bundle = ci.build_input_artifacts(args)
            profile = bundle["profile_payload"]
            self.assertEqual(profile["model_label"], "DSV4")
            self.assertEqual(profile["engine"]["tokenizer_path"], "/models/tokenizer")
            self.assertEqual(
                {
                    key: profile["engine"][key]
                    for key in (
                        "world_size",
                        "tp_size",
                        "dp_size",
                        "ep_size",
                    )
                },
                {
                    "world_size": 8,
                    "tp_size": 2,
                    "dp_size": 2,
                    "ep_size": 4,
                },
            )
            self.assertNotIn("pp_size", profile["engine"])
            self.assertEqual(profile["engine"]["kernel_seq_size_per_block"], 128)
            self.assertEqual(profile["engine"]["max_batch_tokens_size"], 8192)
            self.assertEqual(profile["engine"]["max_context_batch_size"], 2)
            self.assertEqual(profile["engine"]["concurrency_limit"], 2)
            self.assertEqual(
                profile["engine_args"],
                {
                    "act_type": "BF16",
                    "reuse_cache": 1,
                    "enable_device_cache": 1,
                    "enable_cuda_graph": 1,
                    "scale": 0.5,
                },
            )
            self.assertEqual(profile["engine_env"]["WORLD_SIZE"], "8")
            self.assertEqual(profile["engine_env"]["TOKENIZERS_PARALLELISM"], "false")
            self.assertEqual(profile["engine_env"]["PERF_PROFILE_RUNS"], "0")
            self.assertEqual(profile["runtime_env"], {"CC": "/usr/bin/gcc"})
            self.assertEqual(profile["bazel"]["configs"], ["cuda13"])
            grid = bundle["grid_payload"]
            self.assertGreater(len(grid["cases"]), 0)
            self.assertLessEqual(len(grid["cases"]), 200)
            self.assertTrue(all(case["input_len"] <= 4095 for case in grid["cases"]))
            self.assertTrue(all(case["cache_len"] == 512 for case in grid["cases"]))
            self.assertTrue(all(case["batch_size"] == 2 for case in grid["cases"]))
            self.assertEqual(
                grid["summary"]["estimated_measure_requests"], 2 * len(grid["cases"])
            )
            self.assertFalse(inputs_dir.exists())
            self.assertEqual(resolved.grid["fixed_cache_len"], 512)


if __name__ == "__main__":
    unittest.main()
