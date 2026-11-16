"""Process launch and stop behavior without production services."""

import os
import signal
import subprocess
import tempfile
import unittest
from pathlib import Path

BIN = Path(__file__).resolve().parents[1] / "environment/common/bin"


class GracefulStopTest(unittest.TestCase):
    def test_java_output_stays_on_container_stdout_after_launcher_exits(self):
        appctl = (BIN / "appctl.sh").read_text()
        start = appctl[
            appctl.index("\nstart_spring_boot() {") : appctl.index("\nstart_nginx() {")
        ]
        for jar in (True, False):
            with self.subTest(jar=jar), tempfile.TemporaryDirectory() as directory:
                home = Path(directory)
                (home / "bin").mkdir()
                (home / "target").mkdir()
                if jar:
                    (home / "target/FlexLB.jar").touch()
                (home / "bin/preload.sh").write_text(
                    '[[ -f "$APP_HOME/java-ready" ]]\n'
                )
                java = home / "bin/java"
                java.write_text(
                    '#!/bin/bash\nprintf "JVM stderr\\n" >&2\ntouch java-ready\nexec sleep 30\n'
                )
                java.chmod(0o755)
                pid = home / "java.pid"
                output = home / "stdout"
                try:
                    with output.open("w") as stdout:
                        result = subprocess.run(
                            ["bash"],
                            input=f"""
APP_HOME={home}
APP_NAME=FlexLB
JAVA_HOME={home}
SERVICE_PID={pid}
{start}
exec 3>&1
start_spring_boot | tee {home / 'deploy.log'}
""",
                            stdout=stdout,
                            text=True,
                            timeout=5,
                        )
                    self.assertEqual(result.returncode, 0)
                    os.kill(int(pid.read_text()), 0)
                    self.assertIn("JVM stderr", output.read_text())
                    self.assertNotIn("JVM stderr", (home / "deploy.log").read_text())
                finally:
                    if pid.exists():
                        os.kill(int(pid.read_text()), signal.SIGTERM)

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
