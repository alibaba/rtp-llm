"""Stop failure propagation and first-start behavior, without production services."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest

BIN = Path(__file__).resolve().parents[1] / "environment/common/bin"


class GracefulStopTest(unittest.TestCase):
    def test_stop_failure_prevents_resource_teardown_and_success(self):
        appctl = (BIN / "appctl.sh").read_text()
        stop = appctl[appctl.index("\nstop() {") : appctl.index("\nbackup() {")]
        with tempfile.TemporaryDirectory() as directory:
            pid = Path(directory) / "service.pid"
            pid.write_text(str(os.getpid()))
            result = subprocess.run(
                ["bash"],
                input=f"""
APP_HOME={directory}
APP_NAME=FlexLB
SERVICE_PID={pid}
source {BIN / 'hook.sh'}
stop_spring_boot() {{ return 1; }}
stop_xagent() {{ echo UNEXPECTED_TEARDOWN; }}
{stop}
stop
""",
                capture_output=True,
                text=True,
                timeout=5,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn("UNEXPECTED_TEARDOWN", result.stdout)
            self.assertNotIn("stop success", result.stdout)

    def test_first_start_without_pid_does_not_call_http_or_signal_java(self):
        appctl = (BIN / "appctl.sh").read_text()
        stop = appctl[appctl.index("\nstop() {") : appctl.index("\nbackup() {")]
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                ["bash"],
                input=f"""
APP_HOME={directory}
APP_NAME=FlexLB
SERVICE_PID={directory}/absent.pid
NGINXCTL=true
source {BIN / 'hook.sh'}
curl() {{ echo UNEXPECTED_HTTP; return 1; }}
stop_spring_boot() {{ echo UNEXPECTED_SIGNAL; return 1; }}
stop_xagent() {{ :; }}
stop_tomcat() {{ :; }}
{stop}
stop
""",
                capture_output=True,
                text=True,
                timeout=5,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn("UNEXPECTED", result.stdout)

    def test_failed_stop_prevents_restart_and_deploy(self):
        appctl = (BIN / "appctl.sh").read_text()
        main = appctl[appctl.index("\nmain() {") : appctl.index("\nmain | tee")]
        for action in ("pubstart", "restart", "deploy"):
            with self.subTest(action=action):
                result = subprocess.run(
                    ["bash"],
                    input=f"""
ACTION={action}
stop() {{ return 1; }}
start() {{ echo UNEXPECTED_START; }}
{main}
main
""",
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn("UNEXPECTED_START", result.stdout)


if __name__ == "__main__":
    unittest.main()
