"""Stress contracts that must survive the shell-to-Python migration."""
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from runtime import stress
from runtime.harness import resolve_java21
from scripts.pipeline.execute_cases import STRESS_BAND_FLOOR


class StressRuntimeTest(unittest.TestCase):
    def test_metric_whitelist_is_exactly_the_prior_producer_contract(self):
        self.assertEqual(13, len(stress.MASTER_METRIC_WHITELIST))
        self.assertEqual(len(stress.MASTER_METRIC_WHITELIST),
                         len(set(stress.MASTER_METRIC_WHITELIST)))
        self.assertEqual("flexlb_app_cache_", stress.MASTER_METRIC_WHITELIST[0])
        self.assertEqual("flexlb_auto_tpm_decode_running_count",
                         stress.MASTER_METRIC_WHITELIST[-1])

    def test_stress_port_window_occupies_reserved_band(self):
        a = stress.parse_args(["--dry-run"])
        ports = stress.required_ports(a)
        self.assertEqual(STRESS_BAND_FLOOR, a.mock_base_grpc_port)
        self.assertEqual(60999, ports[0])
        self.assertEqual(61051, max(ports[:53]))
        self.assertEqual([7001, 7002, 7003], ports[-3:])
        self.assertEqual(len(ports), len(set(ports)))

    def test_jfr_and_shard_arguments(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = stress.parse_args(["--run-dir", tmp, "--workers", "3",
                                   "--max-concurrency", "10", "--dry-run"])
            spec = stress._env_spec(a)
            self.assertIn("filename=" + str(Path(tmp).resolve() / "flexlb_profile.jfr"),
                          spec.master_jvm_args[0])
            self.assertEqual(4, stress.shard_concurrency(10, 3))
            self.assertEqual(13, len(spec.master_env["FLEXLB_MONITOR_METRIC_WHITELIST"].split(",")))
            diagnostic = stress._env_spec(stress.parse_args([
                "--run-dir", tmp, "--collection-profile", "diagnostic", "--dry-run"]))
            self.assertTrue(diagnostic.diagnostic_events)
            self.assertFalse(diagnostic.master_pv_log)

    def test_jfr_option_writes_a_recording_with_jdk21(self):
        try:
            java = resolve_java21()
        except RuntimeError:
            self.skipTest("JDK 21 not installed")
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "smoke.jfr"
            result = subprocess.run([java, stress.jfr_option(output, "5s"), "-version"],
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(0, result.returncode, result.stderr)
            self.assertGreater(output.stat().st_size, 0)

    def test_monitor_scrapes_control_and_management_endpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = stress.parse_args(["--run-dir", tmp, "--dry-run"])
            env = SimpleNamespace(mock_http_port=60999, master_management_port=7002)
            proc = SimpleNamespace(alive=lambda: True)
            with mock.patch.object(stress.ProcessOps, "start", return_value=proc) as start, \
                 mock.patch.object(stress, "wait_for", return_value=True):
                self.assertIs(proc, stress._monitor_start(a, env))
            argv = start.call_args.args[0]
            self.assertIn("mock=http://127.0.0.1:60999/metrics?per_engine=true", argv)
            self.assertIn("master-single=http://127.0.0.1:7002/prometheus", argv)

    def test_replay_and_uniform_pacing_are_client_owned(self):
        replay = stress.parse_args(["--replay-speed", "4", "--dry-run"])
        rv = stress._client_base(replay, Path("/tmp/plan.jsonl"), 12345)
        self.assertEqual((rv["SEND_MODE"], rv["REPLAY_SPEED"], rv["START_AT_EPOCH_MS"]),
                         ("replay", "4", "12345"))
        uniform = stress.parse_args(["--send-mode", "uniform", "--send-mode-qps", "700",
                                     "--ramp-up-s", "17", "--dry-run"])
        uv = stress._client_base(uniform, Path("/tmp/plan.jsonl"), 12345)
        self.assertEqual((uv["SEND_MODE"], uv["SEND_MODE_QPS"], uv["RAMP_UP_SECONDS"]),
                         ("uniform", "700", "17"))

    def test_master_profile_only_selects_matching_mode(self):
        a = stress.parse_args(["--profile", "single-batch", "--dry-run"])
        self.assertEqual("sb", a.master_mode)
        self.assertEqual("explicit", a.mode_plan["profile_source"])

    def test_client_failure_still_stops_monitor_and_archives_incomplete_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = stress.parse_args(["--run-dir", tmp, "--archive", str(Path(tmp).parent / "failure.zip"),
                                   "--dry-run"])
            a.dry_run = False
            alive = SimpleNamespace(alive=lambda: True)
            env = SimpleNamespace(mock=alive, master=alive)
            manager = mock.MagicMock()
            manager.ensure.return_value = env
            with mock.patch.object(stress, "_preflight"), \
                 mock.patch.object(stress, "_traffic"), \
                 mock.patch.object(stress, "_check_services"), \
                 mock.patch.object(stress, "_monitor_start", return_value=alive), \
                 mock.patch.object(stress, "_monitor_stop") as stop, \
                 mock.patch.object(stress, "_run_clients", side_effect=RuntimeError("client failed")), \
                 mock.patch.object(stress, "EnvManager", return_value=manager), \
                 mock.patch.object(stress, "create_archive") as archive, \
                 mock.patch.object(stress.time, "sleep"):
                with self.assertRaisesRegex(RuntimeError, "client failed"):
                    stress.run(a)
            stop.assert_called_once()
            manager.teardown.assert_called_once()
            self.assertEqual("incomplete", archive.call_args.kwargs["status"])


if __name__ == "__main__":
    unittest.main()
