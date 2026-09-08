#!/usr/bin/env python3

import json
import os
import signal
import stat
import sys
import tempfile
import time
import unittest
from unittest import mock

import run_block_tree_cache_benchmark as driver


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
            [(tag, groups[tag]["layer_count"]) for tag in ("csa_kv", "hca_kv", "indexer_kv")],
            [("csa_kv", 21), ("hca_kv", 20), ("indexer_kv", 21)],
        )
        self.assertEqual(
            [(tag, groups[tag]["layer_count"]) for tag in ("csa_state", "indexer_state", "swa_kv")],
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
