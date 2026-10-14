import os
import pickle
import sys
import unittest
from unittest.mock import patch

from rtp_llm.config.engine_config import EngineConfig
from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.ops import EPLBConfig, RoleType, VitSeparation
from rtp_llm.server.server_args.server_args import setup_args


class EplbMetricsConfigTest(unittest.TestCase):
    def test_default_is_disabled(self):
        with patch.dict(os.environ, {}, clear=True):
            for configs in (PyEnvConfigs(), setup_args([])):
                self.assertEqual(configs.eplb_config.eplb_metrics_report_interval, 0)
                self.assertEqual(configs.eplb_config.eplb_metrics_snapshot_max_files, 0)
                self.assertFalse(
                    hasattr(configs.eplb_config, "eplb_metrics_snapshot_dir")
                )
                EngineConfig.create(configs)
                self.assertEqual(configs.eplb_config.eplb_metrics_report_interval, 0)
                self.assertEqual(configs.eplb_config.eplb_metrics_snapshot_max_files, 0)

    def test_enabling_metrics_does_not_enable_snapshots(self):
        with patch.dict(os.environ, {}, clear=True):
            configs = setup_args(["--eplb_metrics_report_interval", "2"])
            EngineConfig.create(configs)
            self.assertEqual(configs.eplb_config.eplb_metrics_report_interval, 2)
            self.assertEqual(configs.eplb_config.eplb_metrics_snapshot_max_files, 0)

    def test_snapshot_retention_accepts_environment_and_cli(self):
        for source in ("env", "cli"):
            with self.subTest(source=source):
                env = (
                    {"EPLB_METRICS_SNAPSHOT_MAX_FILES": "3"} if source == "env" else {}
                )
                args = (
                    []
                    if source == "env"
                    else ["--eplb_metrics_snapshot_max_files", "3"]
                )
                with patch.dict(os.environ, env, clear=True):
                    configs = setup_args(args)
                    EngineConfig.create(configs)
                self.assertEqual(configs.eplb_config.eplb_metrics_snapshot_max_files, 3)
                self.assertEqual(configs.eplb_config.eplb_metrics_report_interval, 0)

    def test_snapshot_settings_survive_pickle_and_old_state_defaults_to_disabled(self):
        config = EPLBConfig()
        config.eplb_metrics_report_interval = 17
        config.eplb_metrics_snapshot_max_files = 3
        restored = pickle.loads(pickle.dumps(config))
        self.assertEqual(restored.eplb_metrics_report_interval, 17)
        self.assertEqual(restored.eplb_metrics_snapshot_max_files, 3)
        old = EPLBConfig.__new__(EPLBConfig)
        old.__setstate__(config.__getstate__()[:9])
        self.assertEqual(old.eplb_metrics_report_interval, 0)
        self.assertEqual(old.eplb_metrics_snapshot_max_files, 0)

    def test_auto_interval_uses_final_role(self):
        for role, expected in (
            (RoleType.PREFILL, 100),
            (RoleType.DECODE, 1000),
            (RoleType.PDFUSION, 1000),
        ):
            with self.subTest(role=role):
                configs = PyEnvConfigs()
                configs.role_config.role_type = role
                configs.eplb_config.eplb_metrics_report_interval = -1

                EngineConfig.create(configs)

                self.assertEqual(
                    configs.eplb_config.eplb_metrics_report_interval, expected
                )

    def test_final_role_override_precedes_auto_interval(self):
        configs = PyEnvConfigs()
        configs.role_config.role_type = RoleType.PREFILL
        configs.vit_config.vit_separation = VitSeparation.VIT_SEPARATION_ROLE
        configs.eplb_config.eplb_metrics_report_interval = -1

        engine_config = EngineConfig.create(configs)

        self.assertEqual(engine_config.parallelism_config.role_type, RoleType.VIT)
        self.assertEqual(configs.eplb_config.eplb_metrics_report_interval, 1000)

    def test_explicit_intervals_are_preserved(self):
        for role in (RoleType.PREFILL, RoleType.DECODE, RoleType.PDFUSION):
            for interval in (0, 1, 37, 1001):
                with self.subTest(role=role, interval=interval):
                    configs = PyEnvConfigs()
                    configs.role_config.role_type = role
                    configs.eplb_config.eplb_metrics_report_interval = interval

                    EngineConfig.create(configs)

                    self.assertEqual(
                        configs.eplb_config.eplb_metrics_report_interval, interval
                    )

    def test_other_negative_intervals_are_rejected(self):
        for interval in (-2, -100):
            with self.subTest(interval=interval):
                configs = PyEnvConfigs()
                configs.eplb_config.eplb_metrics_report_interval = interval
                with self.assertRaisesRegex(
                    ValueError, "EPLB_METRICS_REPORT_INTERVAL must be"
                ):
                    EngineConfig.create(configs)

    def test_environment_and_cli_reach_engine_config(self):
        for source in ("env", "cli"):
            for role, expected in (
                ("PREFILL", 100),
                ("DECODE", 1000),
                ("PDFUSION", 1000),
            ):
                with self.subTest(source=source, role=role):
                    env = (
                        {
                            "ROLE_TYPE": role,
                            "EPLB_METRICS_REPORT_INTERVAL": "-1",
                        }
                        if source == "env"
                        else {}
                    )
                    args = (
                        []
                        if source == "env"
                        else [
                            "--role_type",
                            role,
                            "--eplb_metrics_report_interval",
                            "-1",
                        ]
                    )
                    with patch.dict(os.environ, env, clear=True), patch.object(
                        sys, "argv", ["server", *args]
                    ):
                        configs = setup_args()
                        EngineConfig.create(configs)
                    self.assertEqual(
                        configs.eplb_config.eplb_metrics_report_interval, expected
                    )

    def test_cli_overrides_environment(self):
        with patch.dict(
            os.environ,
            {"ROLE_TYPE": "PREFILL", "EPLB_METRICS_REPORT_INTERVAL": "-1"},
            clear=True,
        ):
            configs = setup_args(["--eplb_metrics_report_interval=7"])
            EngineConfig.create(configs)
        self.assertEqual(configs.eplb_config.eplb_metrics_report_interval, 7)

    def test_invalid_environment_and_cli_intervals_are_rejected(self):
        for source in ("env", "cli"):
            with self.subTest(source=source):
                env = {"EPLB_METRICS_REPORT_INTERVAL": "-2"} if source == "env" else {}
                args = (
                    [] if source == "env" else ["--eplb_metrics_report_interval", "-2"]
                )
                with patch.dict(os.environ, env, clear=True), patch.object(
                    sys, "argv", ["server", *args]
                ):
                    configs = setup_args()
                    with self.assertRaisesRegex(
                        ValueError, "EPLB_METRICS_REPORT_INTERVAL must be"
                    ):
                        EngineConfig.create(configs)


if __name__ == "__main__":
    unittest.main()
