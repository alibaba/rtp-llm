import socket
import tempfile
import unittest
from pathlib import Path

from flexlb_ft.harness import (
    BASE_MASTER_ENV,
    EnvManager,
    EnvSpec,
    FlexEnv,
    port_in_use,
)


class PortProbeTest(unittest.TestCase):

    @unittest.skipUnless(socket.has_ipv6, "IPv6 is unavailable")
    def test_detects_ipv6_dual_stack_listener_from_ipv4_probe(self):
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as listener:
            try:
                listener.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
            except OSError as exc:
                self.skipTest(f"dual-stack sockets are unavailable: {exc}")
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("::", 0))
            listener.listen(1)

            port = listener.getsockname()[1]
            self.assertTrue(port_in_use(port))


class MasterEnvironmentTest(unittest.TestCase):

    def test_exposes_ttl_metric_used_by_standalone_cases(self):
        self.assertEqual("prometheus", BASE_MASTER_ENV["FLEXLB_MONITOR_PROVIDER"])
        self.assertEqual(
            "flexlb_app_flexlb_inflight_ttl",
            BASE_MASTER_ENV["FLEXLB_MONITOR_METRIC_WHITELIST"],
        )

    def test_uses_writable_per_environment_log_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "env"
            env = FlexEnv(
                EnvSpec(master_profile="none", discovery="none"),
                run_dir,
                44999,
            )
            manager = EnvManager(Path(tmp), verbose=False)

            master_env = manager._master_env(env)

            expected = run_dir / "master_logs"
            self.assertEqual(str(expected), master_env["FLEXLB_LOG_PATH"])
            self.assertEqual(str(expected), master_env["FLEXLB_APP_LOG_PATH"])
            self.assertTrue(expected.is_dir())
            self.assertEqual(expected / "flexlb.log", env.flexlb_log_path)
            self.assertEqual(expected / "pv.log", env.pv_log_path)


if __name__ == "__main__":
    unittest.main()
