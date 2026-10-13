#!/usr/bin/env python3
"""End-to-end GPU smoke for the BlockTreeCache benchmark harness."""

import json
import os
import subprocess
import sys
import tempfile
import unittest


def _runfiles_path(*parts):
    root = os.environ.get("RUNFILES_DIR") or os.environ.get("TEST_SRCDIR")
    if not root:
        return None
    for workspace in ("rtp_llm", "github-opensource"):
        candidate = os.path.join(root, workspace, *parts)
        if os.path.exists(candidate):
            return candidate
    return None


def _libpython_dir():
    root = os.environ.get("RUNFILES_DIR") or os.environ.get("TEST_SRCDIR")
    if not root:
        return None
    for base, _, files in os.walk(root):
        if "libpython3.10.so" in files:
            return base
    return None


def _load_json(path):
    with open(path) as source:
        return json.load(source)


class BenchmarkSmokeTest(unittest.TestCase):
    def test_smoke_suite(self):
        driver = _runfiles_path(
            "rtp_llm/cpp/cache/block_tree_cache/benchmark/run_block_tree_cache_benchmark.py"
        )
        self.assertIsNotNone(driver, "driver script not found in runfiles")

        env = dict(os.environ)
        lib_dir = _libpython_dir()
        if lib_dir:
            env["LD_LIBRARY_PATH"] = (
                lib_dir + os.pathsep + env.get("LD_LIBRARY_PATH", "")
            )
        env["BLOCK_TREE_CACHE_BENCHMARK_TEST_CONFIG"] = "1"

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = os.path.join(tmp, "out")
            cmd = [
                sys.executable,
                driver,
                "--suite",
                "smoke",
                "--output-dir",
                output_dir,
                "--perf",
                "off",
            ]
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
                env=env,
            )
            self.assertEqual(
                proc.returncode,
                0,
                f"benchmark smoke suite failed rc={proc.returncode}\n"
                f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}",
            )

            results = {}
            for case in ("smoke_tree_online_mini", "smoke_transfer_d2h_mini"):
                case_dir = os.path.join(output_dir, "smoke", case)
                manifest_path = os.path.join(case_dir, "manifest.json")
                result_path = os.path.join(case_dir, "rep_0000", "result.json")
                self.assertTrue(
                    os.path.exists(manifest_path), f"missing {case} manifest"
                )
                self.assertTrue(os.path.exists(result_path), f"missing {case} result")
                self.assertEqual(_load_json(manifest_path).get("status"), "completed")
                results[case] = _load_json(result_path)
                self.assertEqual(results[case].get("status"), "completed")

            tree = results["smoke_tree_online_mini"]
            self.assertGreater(tree.get("metrics", {}).get("loads_committed", 0), 0)

            transfer = results["smoke_transfer_d2h_mini"]
            self.assertGreater(
                transfer.get("workload", {}).get("succeeded_operations", 0), 0
            )


if __name__ == "__main__":
    unittest.main()
