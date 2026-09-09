"""Child public protocol and real process ownership cleanup."""

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
from flexlb_test_framework.scenario.backend import JavaMockBackend
from flexlb_test_framework.scenario.loader import ScenarioError
from flexlb_test_framework.scenario.runtime import Deadline, RuntimeContext
from scenario_runner import instance_directory, select, summarize


class ChildTest(unittest.TestCase):
    def test_storage_key_cannot_be_parsed_as_jvm_log_options(self):
        ids = [
            "request_completion::immediate::batch-window",
            "request_completion::deferred_fetch::batch-window",
            "../odd:id",
        ]
        keys = [instance_directory(value) for value in ids]
        self.assertEqual(len(set(keys)), len(ids))
        for key in keys:
            self.assertRegex(key, r"^instance-[0-9a-f]{64}$")
            self.assertNotIn(":", key)
            self.assertNotIn("/", key)
        self.assertEqual(instance_directory(ids[0]), keys[0])

    def test_backend_rejects_unsafe_root_before_process_harness_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = RuntimeContext(
                {}, None, Path(tmp) / "unsafe:root", time.monotonic, time.sleep
            )
            backend = JavaMockBackend({})
            with self.assertRaisesRegex(ValueError, "JVM -Xlog"):
                backend.setup(ctx, {}, Deadline(time.monotonic() + 1))
            self.assertEqual(backend.environments, [])

    def test_list_contract_no_runtime_and_exact_selection(self):
        proc = subprocess.run(
            [
                sys.executable,
                "scenario_runner.py",
                "--source",
                "scenarios/core/request_completion.yaml",
                "--list-json",
                "--profile",
                "batch-window",
                "--grade",
                "loose",
            ],
            cwd=TOOLS,
            capture_output=True,
            text=True,
            timeout=5,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        rows = json.loads(proc.stdout)["instances"]
        self.assertEqual(
            {r["variant_id"] for r in rows},
            {"immediate", "deferred_fetch", "client_no_fetch"},
        )
        self.assertTrue(all(row["grade"] == "loose" for row in rows))
        self.assertNotIn("stages", rows[0])
        self.assertNotIn("environment", rows[0])
        self.assertGreater(rows[0]["execution"]["cleanup_timeout_s"], 0)
        self.assertEqual(select(rows, rows[1]["id"]), [rows[1]])
        for ids in ("", "missing", rows[0]["id"] + "," + rows[0]["id"]):
            with self.assertRaises(ScenarioError):
                select(rows, ids)

    def test_execution_without_lease_fails_before_importing_or_starting_backend(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = subprocess.run(
                [
                    sys.executable,
                    "scenario_runner.py",
                    "--source",
                    "scenarios/core/request_completion.yaml",
                    "--out-dir",
                    tmp,
                ],
                cwd=TOOLS,
                capture_output=True,
                text=True,
                timeout=5,
            )
            self.assertEqual(proc.returncode, 2)
            self.assertIn("--lease-json", proc.stderr)
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_cleanup_errors_override_findings_in_summary(self):
        rows = [dict(status="FINDING-CONFIRMED", cleanup=[dict(status="ERROR")])]
        self.assertEqual(summarize(rows)["exit_code"], 1)
        self.assertEqual(summarize(rows)["failed_count"], 1)

    def test_backend_reaps_only_its_owned_process_and_preserves_unrelated_process(self):
        owned = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        unrelated = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"]
        )
        try:
            managed = NS(pid=owned.pid, proc=owned, alive=lambda: owned.poll() is None)
            backend = JavaMockBackend({})
            backend.environments = [
                NS(
                    load_clients=[],
                    victims={},
                    masters={},
                    master=managed,
                    zk_helper=None,
                    mock=None,
                )
            ]
            backend.remember_processes(backend.environments[0])
            # A restart/kill helper can clear this slot before final teardown.
            backend.environments[0].master = None
            with tempfile.TemporaryDirectory() as tmp:
                ctx = RuntimeContext({}, backend, tmp, time.monotonic, time.sleep)
                backend.teardown(ctx, Deadline(time.monotonic() + 3))
                evidence = json.loads((Path(tmp) / "process-cleanup.json").read_text())
            self.assertIsNotNone(owned.poll())
            self.assertIsNone(unrelated.poll())
            self.assertEqual(evidence, dict(owned_pids=[owned.pid], remaining_pids=[]))
        finally:
            for proc in (owned, unrelated):
                if proc.poll() is None:
                    proc.kill()
                proc.wait(timeout=3)


if __name__ == "__main__":
    unittest.main()
