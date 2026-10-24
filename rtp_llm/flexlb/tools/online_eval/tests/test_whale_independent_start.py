import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from mode_profiles import render_shell_defaults

START = Path(__file__).resolve().parents[3] / "flexlb-mock-engine" / "container" / "start.sh"


class WhaleIndependentStartTest(unittest.TestCase):
    def launch(self, extra=None, arguments=()):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_java = root / "java"
            fake_java.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
            fake_java.chmod(0o755)
            env = {
                **os.environ,
                "PATH": f"{root}:{os.environ['PATH']}",
                "FLEXLB_MOCK_WHALE": "1",
                "POD_IP": "10.0.0.5",
                "START_PORT": "7001",
                "ROLE_TYPE": "PREFILL",
                "MOCK_CALIBRATION_CONFIG_JSON": "{}",
                "MOCK_ENGINE_CONFIG_JSON": (START.parents[2] / "tools/mock_bundle/engine-config.json").read_text(),
                "MOCK_RUN_DIR": str(root / "run"),
                "MOCK_KMONITOR_ENABLED": "false",
            }
            env.pop("MOCK_ENGINE_CONFIG", None)
            env.pop("MOCK_CALIBRATION_CONFIG", None)
            env.pop("MOCK_EVENT_LOG_ENABLED", None)
            env.update(extra or {})
            return subprocess.run(["sh", str(START), *arguments], env=env, text=True, capture_output=True)

    def test_whale_default_has_no_jsonl_and_pod_address(self):
        result = self.launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        argv = result.stdout.splitlines()
        self.assertNotIn("--events-file", argv)
        self.assertEqual(argv[argv.index("--host") + 1], "10.0.0.5")
        self.assertEqual(argv[argv.index("--kmonitor") + 1], "false")
        self.assertEqual(argv[argv.index("--unique-engine-ips") + 1], "false")

    def test_cli_cannot_override_placement_contract(self):
        for arguments in (("--whale-bundle", "true"), ("--n-prefill", "2"), ("--discovery-file", "/tmp/unused")):
            with self.subTest(arguments=arguments):
                result = self.launch(arguments=arguments)
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn("-jar", result.stdout)

    def test_explicit_per_engine_contract(self):
        result = self.launch({"MOCK_DEPLOYMENT_MODE": "per_engine", "ROLE_TYPE": "DECODE"})
        self.assertEqual(result.returncode, 0, result.stderr)
        argv = result.stdout.splitlines()
        self.assertEqual(argv[argv.index("--n-prefill") + 1], "0")
        self.assertEqual(argv[argv.index("--n-decode") + 1], "1")
        self.assertEqual(argv[argv.index("--base-grpc-port") + 1], "7002")

    def test_other_placements_and_conflicts_cannot_reach_java(self):
        for extra in (
            {"MOCK_DEPLOYMENT_MODE": "embedded"},
            {"MOCK_DEPLOYMENT_MODE": "single_pod"},
            {"MOCK_BUNDLE_ROLE": "bundle"},
            {"MOCK_REMOTE_DECODE": "0"},
            {"MOCK_BUNDLE_FILE_DISCOVERY": "1"},
            {"MOCK_PREFILL_COUNT": "2"},
            {"MOCK_BUNDLE_OVERRIDES_YAML": "{}"},
        ):
            with self.subTest(extra=extra):
                result = self.launch(extra)
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn("-jar", result.stdout)

    def test_packaged_shell_defaults_match_shared_mode_table(self):
        self.assertEqual((START.parent / "mode_defaults.sh").read_text(), render_shell_defaults("whale_independent"))

    def test_jsonl_can_be_opted_in(self):
        result = self.launch({"MOCK_EVENT_LOG_ENABLED": "1"})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--events-file", result.stdout.splitlines())


if __name__ == "__main__":
    unittest.main()
