#!/usr/bin/env python3

import json
import os
import re
import signal
import stat
import sys
import tempfile
import time
import unittest
from unittest import mock

import run_block_tree_cache_benchmark as driver
from benchmark_cases import ALL_CASES, PROFILE_CASES, SMOKE_CASES


class BenchmarkRegistryDocumentationTest(unittest.TestCase):
    def test_canonical_case_counts_and_unique_names(self):
        self.assertEqual(len(SMOKE_CASES), 2)
        self.assertEqual(len(PROFILE_CASES), 109)
        names = [case.name for case in ALL_CASES]
        self.assertEqual(len(names), len(set(names)))
        families = {
            "tree_": 1,
            "transfer_": 20,
            "matrix_": 24,
            "e2e_": 64,
        }
        for prefix, expected in families.items():
            with self.subTest(prefix=prefix):
                self.assertEqual(
                    sum(case.name.startswith(prefix) for case in PROFILE_CASES),
                    expected,
                )
        path = driver.resolve_runfile_path("docs/benchmark_cases.md")
        self.assertIsNotNone(path)
        with open(path) as source:
            documentation = source.read()
        self.assertIn(
            "当前 profile 共 109 个 case：1 个 Tree、20 个 mixed Transfer"
            "（含 6 个并发变体）、24 个 batch API matrix 和 64 个 e2e case。",
            documentation,
        )

    def test_nsys_documented_profiles_exist(self):
        path = driver.resolve_runfile_path("docs/nsys_collection_guide.md")
        self.assertIsNotNone(path)
        with open(path) as source:
            references = re.findall(
                r"\$BENCH_DIR/(profiles/[\w.-]+\.json)", source.read()
            )
        self.assertEqual(len(references), 2)
        for reference in references:
            with self.subTest(profile=reference):
                profile_path = driver.resolve_runfile_path(reference)
                self.assertIsNotNone(profile_path)
                with open(profile_path) as source:
                    profile = json.load(source)
                self.assertTrue(profile["groups"])


class BenchmarkCopyCapabilityTest(unittest.TestCase):
    def _binary(self, directory, response):
        path = os.path.join(directory, "capability_binary")
        with open(path, "w") as script:
            script.write(
                f"#!{sys.executable}\n"
                "import sys\n"
                "assert sys.argv[1:] == ['capabilities', '--cuda-device', '2']\n"
                f"print({response!r})\n"
            )
        os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR)
        return path

    def test_registry_marks_only_explicit_3d_cases(self):
        marked = [case for case in ALL_CASES if case.requires_cuda_3d_batch]
        self.assertEqual(len(marked), 28)
        for case in ALL_CASES:
            self.assertEqual(
                case.requires_cuda_3d_batch,
                case.params.get("--copy-strategy") == "cuda-3d-batch",
                case.name,
            )

    def test_unsupported_cases_are_preserved_and_suite_remains_strict(self):
        cases = [
            case
            for case in PROFILE_CASES
            if case.params.get("--copy-strategy") == "cuda-3d-batch"
        ][:2]
        profile = driver.resolve_runfile_path(
            "profiles/deepseek_v4_pro_fp8_tp1_cp1.json"
        )
        with tempfile.TemporaryDirectory() as tmp:
            binary = self._binary(tmp, '{"cuda_3d_batch": false}')
            for allow_incomplete in (False, True):
                with self.subTest(allow_incomplete=allow_incomplete):
                    output = os.path.join(tmp, str(allow_incomplete))
                    with mock.patch.object(
                        driver, "collect_environment", return_value={}
                    ):
                        result = driver.run_suite(
                            suite="profile",
                            case_name=",".join(c.name for c in cases),
                            binary=binary,
                            model_profile=profile,
                            output_dir=output,
                            cuda_device=2,
                            max_device_memory_fraction=0.8,
                            base_seed=42,
                            task_pool_sizes=None,
                            repetitions=1,
                            case_timeout=5,
                            grace_seconds=1,
                            perf_mode="off",
                            perf_frequency=99,
                            flamegraph_tools=None,
                            disk_root=None,
                            allow_incomplete=allow_incomplete,
                        )
                    self.assertEqual(result, 0 if allow_incomplete else 1)
                    with open(
                        os.path.join(output, "profile", "suite_manifest.json")
                    ) as source:
                        manifest = json.load(source)
                    self.assertEqual(manifest["skipped"], 2)
                    self.assertEqual(manifest["failed"], 0)
                    self.assertEqual(manifest["completed"], 0)
                    for case in manifest["cases"]:
                        self.assertEqual(case["status"], "skipped_unsupported_platform")
                        self.assertIn("CUDA 13", case["skip_reason"])
                        self.assertEqual(case["repetitions"], [])

    def test_capability_probe_validates_native_response(self):
        with tempfile.TemporaryDirectory() as tmp:
            for value in (False, True):
                binary = self._binary(tmp, json.dumps({"cuda_3d_batch": value}))
                self.assertEqual(
                    driver.query_copy_capabilities(binary, 2), {"cuda_3d_batch": value}
                )
            for response in ("not json", '{"cuda_3d_batch": "false"}', "{}"):
                binary = self._binary(tmp, response)
                with self.subTest(response=response), self.assertRaises(RuntimeError):
                    driver.query_copy_capabilities(binary, 2)

    def test_supported_case_still_runs_and_validates_measurement(self):
        case = next(
            c
            for c in PROFILE_CASES
            if c.params.get("--copy-strategy") == "cuda-3d-batch"
        )
        with tempfile.TemporaryDirectory() as tmp:
            binary = os.path.join(tmp, "supported_binary")
            with open(binary, "w") as script:
                script.write(
                    f"#!{sys.executable}\n"
                    "import json,sys\n"
                    "if sys.argv[1] == 'capabilities':\n"
                    " print(json.dumps({'cuda_3d_batch':True}))\n"
                    "else:\n"
                    " assert sys.argv[1] == 'transfer'\n"
                    " p=sys.argv[sys.argv.index('--output-json')+1]\n"
                    " with open(p,'w') as f: json.dump({'schema_version':1,'component':'BlockTreeCache',"
                    "'binary':'block_tree_cache_gpu_benchmark','runner':'transfer','status':'completed',"
                    "'phases_ns':{'measured':1}},f)\n"
                )
            os.chmod(binary, os.stat(binary).st_mode | stat.S_IXUSR)
            manifest = driver.run_case(
                case=case,
                binary=binary,
                model_profile="unused.json",
                output_dir=tmp,
                cuda_device=2,
                max_device_memory_fraction=0.8,
                base_seed=42,
                repetitions=1,
                case_timeout=5,
                grace_seconds=1,
                perf_mode="off",
                perf_frequency=99,
                flamegraph_tools=None,
                disk_root=None,
            )
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(len(manifest["repetitions"]), 1)
            self.assertTrue(manifest["repetitions"][0]["valid"])


class BenchmarkDriverProfileTest(unittest.TestCase):
    def test_loads_dsv4_flash_full_and_swa_layout(self):
        path = driver.resolve_runfile_path(
            "profiles/deepseek_v4_flash_fp8_tp1_cp1_tpb1024.json"
        )
        self.assertIsNotNone(path)
        with open(path) as source:
            profile = json.load(source)

        self.assertEqual(profile["model"], "deepseek_v4_flash")
        self.assertEqual(profile["num_layers"], 43)
        self.assertEqual(profile["tokens_per_block"], 1024)
        self.assertEqual(profile["kernel_tokens_block"], 128)
        self.assertEqual(
            driver.load_profile_group_set_payloads(path),
            {"full_context": 4087296, "swa": 4940160},
        )

        groups = {group["tag"]: group for group in profile["groups"]}
        self.assertEqual(
            [
                (tag, groups[tag]["layer_count"])
                for tag in ("csa_kv", "hca_kv", "indexer_kv")
            ],
            [("csa_kv", 21), ("hca_kv", 20), ("indexer_kv", 21)],
        )
        self.assertEqual(
            [
                (tag, groups[tag]["layer_count"])
                for tag in ("csa_state", "indexer_state", "swa_kv")
            ],
            [("csa_state", 21), ("indexer_state", 21), ("swa_kv", 43)],
        )
        self.assertEqual(profile["device_only_groups"], ["hca_state"])
        self.assertEqual(profile["swa_config"]["gen_num_per_cycle"], 0)
        self.assertFalse(profile["swa_config"]["include_mtp"])


class BenchmarkDriverProcessTest(unittest.TestCase):
    def _write_executable(self, directory, name, body):
        path = os.path.join(directory, name)
        with open(path, "w") as script:
            script.write(f"#!{sys.executable}\n{body}")
        os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR)
        return path

    def _assert_process_gone(self, pid_path):
        with open(pid_path) as source:
            pid = int(source.read())
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)

    def test_perf_record_marker_timeout_drains_output_and_reaps_native(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            pid_path = os.path.join(temp_dir, "native.pid")
            native = self._write_executable(
                temp_dir,
                "native",
                "import os, sys, time\n"
                "open(os.environ['FAKE_NATIVE_PID_PATH'], 'w').write(str(os.getpid()))\n"
                "sys.stderr.write('x' * 1048576)\n"
                "sys.stderr.flush()\n"
                "time.sleep(30)\n",
            )
            started = time.monotonic()
            with mock.patch.dict(
                os.environ, {"FAKE_NATIVE_PID_PATH": pid_path}, clear=False
            ):
                ok, message = driver.run_perf_record(
                    native,
                    "tree",
                    {},
                    temp_dir,
                    0,
                    0.8,
                    42,
                    "unused.json",
                    99,
                    ("unused", "unused"),
                    process_timeout_seconds=1,
                    attach_timeout_seconds=0.1,
                )
            self.assertFalse(ok)
            self.assertIn("profiler attach marker", message)
            self.assertLess(time.monotonic() - started, 2)
            self._assert_process_gone(pid_path)

    def test_perf_stat_nonzero_is_not_success_with_nonempty_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            fake_perf = self._write_executable(
                temp_dir,
                "perf",
                "import pathlib, sys\n"
                "path = pathlib.Path(sys.argv[sys.argv.index('--output') + 1])\n"
                "path.write_text('partial stats')\n"
                "sys.exit(9)\n",
            )
            self.assertTrue(os.path.exists(fake_perf))
            with mock.patch.dict(
                os.environ,
                {"PATH": temp_dir + os.pathsep + os.environ.get("PATH", "")},
                clear=False,
            ):
                ok, message = driver.run_perf_stat(
                    "unused", "tree", {}, temp_dir, 0, 0.8, 42, "unused.json"
                )
            self.assertFalse(ok)
            self.assertIn("rc=9", message)

    def test_perf_stat_timeout_reaps_process_group(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            pid_path = os.path.join(temp_dir, "perf.pid")
            self._write_executable(
                temp_dir,
                "perf",
                "import os, time\n"
                "open(os.environ['FAKE_PERF_PID_PATH'], 'w').write(str(os.getpid()))\n"
                "time.sleep(30)\n",
            )
            with mock.patch.dict(
                os.environ,
                {
                    "PATH": temp_dir + os.pathsep + os.environ.get("PATH", ""),
                    "FAKE_PERF_PID_PATH": pid_path,
                },
                clear=False,
            ):
                ok, message = driver.run_perf_stat(
                    "unused",
                    "tree",
                    {},
                    temp_dir,
                    0,
                    0.8,
                    42,
                    "unused.json",
                    process_timeout_seconds=0.1,
                )
            self.assertFalse(ok)
            self.assertIn("timed out", message)
            self._assert_process_gone(pid_path)


if __name__ == "__main__":
    unittest.main()
