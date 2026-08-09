#!/usr/bin/env python3
"""Smoke test for the BlockTreeCache GPU benchmark harness.

Runs the driver with --suite smoke to guard the benchmark itself:
- binary and driver start successfully
- case registry parameters stay compatible with the binary
- model profile loads
- minimal tree and transfer paths run end-to-end

Requires a GPU.
"""

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
    """Return the runfiles dir containing libpython3.10.so, if any."""
    root = os.environ.get("RUNFILES_DIR") or os.environ.get("TEST_SRCDIR")
    if not root:
        return None
    for base, _, files in os.walk(root):
        if "libpython3.10.so" in files:
            return base
    return None


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

        with tempfile.TemporaryDirectory() as tmp:
            cmd = [
                sys.executable,
                driver,
                "--suite",
                "smoke",
                "--output-dir",
                os.path.join(tmp, "out"),
                "--perf",
                "off",
            ]
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=600, env=env
            )
            self.assertEqual(
                proc.returncode,
                0,
                f"benchmark smoke suite failed rc={proc.returncode}\n"
                f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}",
            )


if __name__ == "__main__":
    unittest.main()
