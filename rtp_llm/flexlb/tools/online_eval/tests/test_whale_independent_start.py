import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mode_profiles import render_shell_defaults


START = Path(__file__).resolve().parents[3] / "flexlb-mock-engine" / "whale" / "start.sh"


class WhaleIndependentStartTest(unittest.TestCase):
    def launch(self, extra=None):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_java = root / "java"
            fake_java.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
            fake_java.chmod(0o755)
            env = {
                **os.environ, "PATH": f"{root}:{os.environ['PATH']}",
                "FLEXLB_MOCK_WHALE": "1", "POD_IP": "10.0.0.5",
                "START_PORT": "7001", "ROLE_TYPE": "PREFILL",
                "MOCK_PERFORMANCE_CONFIG_JSON": "{}", "MOCK_MASTER_CONFIG_JSON": "{}",
                "MOCK_RUN_DIR": str(root / "run"), "MOCK_KMONITOR_ENABLED": "false",
            }
            env.pop("MOCK_PERFORMANCE_CONFIG", None)
            env.pop("MOCK_MASTER_CONFIG", None)
            env.pop("MOCK_EVENT_LOG_ENABLED", None)
            env.update(extra or {})
            return subprocess.run(["sh", str(START)], env=env, text=True,
                                  capture_output=True)

    def test_whale_default_has_no_jsonl_and_pod_address(self):
        result = self.launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        argv = result.stdout.splitlines()
        self.assertNotIn("--events-file", argv)
        self.assertEqual(argv[argv.index("--host") + 1], "10.0.0.5")
        self.assertEqual(argv[argv.index("--kmonitor") + 1], "false")
        self.assertEqual(argv[argv.index("--unique-engine-ips") + 1], "false")

    def test_packaged_shell_defaults_match_shared_mode_table(self):
        self.assertEqual(
            (START.parent / "mode_defaults.sh").read_text(),
            render_shell_defaults("whale_independent"),
        )

    def test_jsonl_can_be_opted_in(self):
        result = self.launch({"MOCK_EVENT_LOG_ENABLED": "1"})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--events-file", result.stdout.splitlines())


if __name__ == "__main__":
    unittest.main()
