#!/usr/bin/env python3
"""End-to-end GPU smoke for the BlockTreeCache benchmark harness."""

import json
import os
from pathlib import Path
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
    libpython = "libpython%d.%d.so" % sys.version_info[:2]
    for base, _, files in os.walk(root):
        if libpython in files:
            return base
    return None


def _load_json(path):
    with open(path) as source:
        return json.load(source)


def _failed_repetitions(output_dir):
    details = []
    root = Path(output_dir)
    # The driver writes a repetition manifest when the native benchmark fails.
    for manifest in sorted(root.glob("smoke/*/rep_*/manifest.json")):
        for name in ("manifest.json", "stdout.txt", "stderr.txt"):
            path = manifest.parent / name
            if path.is_file():
                with path.open("rb") as stream:
                    stream.seek(max(0, path.stat().st_size - 8192))
                    tail = stream.read().decode(errors="replace")
                details.append(f"{path.relative_to(root)}\n{tail}")
    return "\n".join(details)


class BenchmarkSmokeTest(unittest.TestCase):
    def test_three_tier_transfer_and_route_batch_limit(self):
        binary = _runfiles_path(
            "rtp_llm/cpp/cache/block_tree_cache/benchmark/block_tree_cache_gpu_benchmark"
        )
        profile = _runfiles_path(
            "rtp_llm/cpp/cache/block_tree_cache/benchmark/profiles/"
            "deepseek_v4_flash_fp8_tp1_cp1_tpb1024.json"
        )
        self.assertIsNotNone(binary)
        self.assertIsNotNone(profile)
        env = dict(os.environ)
        lib_dir = _libpython_dir()
        if lib_dir:
            env["LD_LIBRARY_PATH"] = (
                lib_dir + os.pathsep + env.get("LD_LIBRARY_PATH", "")
            )
        with tempfile.TemporaryDirectory() as tmp:
            result_path = os.path.join(tmp, "result.json")
            cmd = [
                binary,
                "transfer",
                f"--model-profile={profile}",
                f"--output-json={result_path}",
                "--group-set=full_context",
                "--transfer-directions=d2h,h2disk,disk2d",
                "--transfer-operation-count=12",
                "--transfer-concurrency=4",
                "--transfer-descriptor-batch-size=4",
                "--working-set-blocks=4",
                "--device-disk-staging-block-count=4",
                "--btc-crc-slot-count=1",
                "--min-measured-seconds=1",
                f"--disk-path={tmp}",
                "--disk-io-mode=buffered",
            ]
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=120, env=env
            )
            self.assertEqual(proc.returncode, 0, f"{proc.stdout}\n{proc.stderr}")
            self.assertEqual(_load_json(result_path).get("status"), "completed")

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
            if proc.returncode:
                self.fail(
                    f"benchmark smoke suite failed rc={proc.returncode}\n"
                    f"{proc.stdout}\n{proc.stderr}\n{_failed_repetitions(output_dir)}"
                )

            results = {}
            for case, expected_owners in (
                ("smoke_tree_online_mini", 2),
                ("smoke_transfer_d2h_mini", 1),
            ):
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
                mode = results[case].get("resolved_config", {})
                self.assertEqual(mode.get("transfer_implementation"), "workloop")
                self.assertEqual(mode.get("enable_crc"), 1)
                self.assertEqual(mode.get("owners_per_direction"), expected_owners)

            tree = results["smoke_tree_online_mini"]
            self.assertGreater(tree.get("metrics", {}).get("loads_committed", 0), 0)

            transfer = results["smoke_transfer_d2h_mini"]
            self.assertGreater(
                transfer.get("workload", {}).get("succeeded_operations", 0), 0
            )


if __name__ == "__main__":
    unittest.main()
