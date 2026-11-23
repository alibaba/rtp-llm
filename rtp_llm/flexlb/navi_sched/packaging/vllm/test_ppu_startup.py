import os
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

SOURCE = (
    Path(os.environ.get("VLLM_SOURCE_ROOT", Path.cwd())).resolve()
    / "vllm/whale_vllm/start.sh"
)


class PPUStartupTest(unittest.TestCase):
    def startup_env(self, **overrides):
        script = Path(os.environ.get("PPU_STARTUP_ENTRYPOINT", str(SOURCE))).read_text()
        start = script.index("export VLLM_NAVI_SCHED_ENABLE=")
        end = script.index("# setting vllm config", start)
        env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("VLLM_NAVI_SCHED_") and key != "PYTHONHASHSEED"
        }
        env.update(START_PORT="28520", HIPPO_APP="ppu_test", VLLM_NAVI_SCHED_ENABLE="1")
        env.update(overrides)
        result = subprocess.run(
            [
                *shlex.split(os.environ.get("PPU_TEST_SHELL", "sh")),
                "-c",
                script[start:end] + "\nenv",
            ],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertNotIn("not found", result.stderr)
        return dict(
            line.split("=", 1) for line in result.stdout.splitlines() if "=" in line
        )

    def test_sh_initializes_agent_configuration(self):
        env = self.startup_env()
        self.assertEqual("28520", env["VLLM_NAVI_SCHED_SERVER_PORT"])
        self.assertEqual("28524", env["VLLM_NAVI_SCHED_AGENT_GIG_PORT"])
        self.assertEqual("ppu_test", env["VLLM_NAVI_SCHED_BIZ_NAME"])
        self.assertEqual("ppu_test", env["VLLM_NAVI_SCHED_TARGET_BIZ_NAME"])
        self.assertEqual(
            "vllm_navi_sched_master", env["VLLM_NAVI_SCHED_MASTER_BIZ_NAME"]
        )
        self.assertTrue(env["VLLM_NAVI_SCHED_SERVER_IP"])
        self.assertEqual("0", env["PYTHONHASHSEED"])

    def test_explicit_configuration_is_preserved(self):
        env = self.startup_env(
            VLLM_NAVI_SCHED_SERVER_IP="127.0.0.2",
            VLLM_NAVI_SCHED_SERVER_PORT="12345",
            VLLM_NAVI_SCHED_AGENT_GIG_PORT="12349",
            VLLM_NAVI_SCHED_BIZ_NAME="agent",
            VLLM_NAVI_SCHED_TARGET_BIZ_NAME="target",
            VLLM_NAVI_SCHED_MASTER_BIZ_NAME="master",
        )
        for name, value in (
            ("SERVER_IP", "127.0.0.2"),
            ("SERVER_PORT", "12345"),
            ("AGENT_GIG_PORT", "12349"),
            ("BIZ_NAME", "agent"),
            ("TARGET_BIZ_NAME", "target"),
            ("MASTER_BIZ_NAME", "master"),
        ):
            self.assertEqual(value, env["VLLM_NAVI_SCHED_" + name])

    def test_unconfigured_entrypoint_defaults_to_native(self):
        self.assertIn("VLLM_NAVI_SCHED_ENABLE:-0", SOURCE.read_text())

    def test_disabled_mode_does_not_initialize_agent(self):
        env = self.startup_env(VLLM_NAVI_SCHED_ENABLE="0")
        self.assertNotIn("VLLM_NAVI_SCHED_SERVER_PORT", env)
        self.assertNotIn("VLLM_NAVI_SCHED_BIZ_NAME", env)
        self.assertNotIn("PYTHONHASHSEED", env)

    def test_failure_records_bounded_logs_and_preserves_exit_status(self):
        script = SOURCE.read_text()
        footer = script[script.index("\nstartup_status=$?") :]
        with tempfile.TemporaryDirectory() as work:
            root = Path(work)
            stderr = root / "stderr"
            stdout = root / "stdout"
            message = root / "termination-log"
            stderr.write_text("e" * 10000 + "FAILED_ENGINE")
            stdout.write_text("o" * 10000 + "ENGINE_CONTEXT")
            message.touch()
            command = footer.replace("/dev/termination-log", str(message))
            result = subprocess.run(
                ["sh", "-c", "(exit 7)" + command],
                env={
                    **os.environ,
                    "STDERR_FILE": str(stderr),
                    "STDOUT_FILE": str(stdout),
                },
                check=False,
            )
            self.assertEqual(7, result.returncode)
            text = message.read_text()
            self.assertLessEqual(len(text.encode()), 4096)
            self.assertTrue(text.startswith("vLLM exited with status 7\nstderr tail:"))
            self.assertIn("FAILED_ENGINE", text)
            self.assertIn("ENGINE_CONTEXT", text)

    def test_success_does_not_write_termination_message(self):
        script = SOURCE.read_text()
        footer = script[script.index("\nstartup_status=$?") :]
        with tempfile.TemporaryDirectory() as work:
            message = Path(work) / "termination-log"
            message.touch()
            result = subprocess.run(
                [
                    "sh",
                    "-c",
                    "(exit 0)" + footer.replace("/dev/termination-log", str(message)),
                ],
                check=False,
            )
            self.assertEqual(0, result.returncode)
            self.assertEqual("", message.read_text())


if __name__ == "__main__":
    unittest.main()
