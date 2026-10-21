"""Startup deadline and Java cleanup without production services."""

from pathlib import Path
import subprocess
import tempfile
import unittest

BIN = Path(__file__).resolve().parents[1] / "environment/common/bin"


class StartupTimeoutTest(unittest.TestCase):
    def run_startup(self, healthy):
        appctl = (BIN / "appctl.sh").read_text()
        waiting = appctl[appctl.index("    local startup_pid=$!") : appctl.index("\nstart_nginx() {")]
        with tempfile.TemporaryDirectory(dir=Path.home() / "workspace/tmp") as directory:
            root = Path(directory)
            (root / "bin").mkdir()
            (root / "bin/preload.sh").write_text(
                f'echo "probe timeout=$FLEXLB_HEALTH_CHECK_TIMEOUT_SECONDS"\nreturn {0 if healthy else 1}\n'
            )
            result = subprocess.run(
                ["bash"],
                input=f"""
APP_HOME={root}
SERVICE_PID={root}/service.pid
APP_NAME=FlexLB
TOMCAT_PORT=7001
startup_timeout=1
startup_started_at=$SECONDS
exit4() {{ exit 4; }}
run() {{
    sleep 60 &
    echo $! > "$SERVICE_PID"
{waiting}
run
pid=$(cat "$SERVICE_PID")
kill -TERM "$pid"
wait "$pid" 2>/dev/null
exit 0
""",
                capture_output=True,
                text=True,
                timeout=10,
            )
            if not healthy:
                self.assertFalse((root / "service.pid").exists())
            return result

    def test_unhealthy_startup_exits_and_cleans_up_java(self):
        result = self.run_startup(False)
        self.assertEqual(result.returncode, 4, result.stderr)
        self.assertIn("failed to become healthy within 1s", result.stdout)
        self.assertNotIn("ready to serve requests", result.stdout)

    def test_healthy_startup_keeps_java_until_caller_stops_it(self):
        result = self.run_startup(True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ready to serve requests", result.stdout)
        self.assertNotIn("failed to become healthy", result.stdout)


if __name__ == "__main__":
    unittest.main()
