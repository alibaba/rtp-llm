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
    def test_native_config_defaults_are_disabled(self):
        configs = PyEnvConfigs()
        EngineConfig.create(configs)
        self.assertEqual(configs.eplb_config.eplb_metrics_report_interval, 0)
        self.assertEqual(configs.eplb_config.eplb_metrics_snapshot_max_files, 0)

    def test_default_interval_is_automatic_and_snapshots_are_disabled(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(
            sys, "argv", ["server"]
        ):
            configs = setup_args()
            self.assertEqual(configs.eplb_config.eplb_metrics_report_interval, -1)
            EngineConfig.create(configs)
            self.assertEqual(configs.eplb_config.eplb_metrics_report_interval, 1000)
            self.assertEqual(configs.eplb_config.eplb_metrics_snapshot_max_files, 0)

        for role, expected in (
            ("PREFILL", 100),
            ("DECODE", 1000),
            ("PDFUSION", 1000),
        ):
            with self.subTest(role=role), patch.dict(os.environ, {}, clear=True):
                configs = setup_args(["--role_type", role])
                self.assertEqual(configs.eplb_config.eplb_metrics_report_interval, -1)
                self.assertEqual(configs.eplb_config.eplb_metrics_snapshot_max_files, 0)
                self.assertFalse(
                    hasattr(configs.eplb_config, "eplb_metrics_snapshot_dir")
                )
                EngineConfig.create(configs)
                self.assertEqual(
                    configs.eplb_config.eplb_metrics_report_interval, expected
                )
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
                self.assertEqual(configs.eplb_config.eplb_metrics_report_interval, 1000)

    def test_snapshot_settings_survive_pickle_and_old_state_defaults_to_disabled(self):
        for interval in (-1, 17):
            with self.subTest(interval=interval):
                config = EPLBConfig()
                config.eplb_metrics_report_interval = interval
                config.eplb_metrics_snapshot_max_files = 3
                restored = pickle.loads(pickle.dumps(config))
                self.assertEqual(restored.eplb_metrics_report_interval, interval)
                self.assertEqual(restored.eplb_metrics_snapshot_max_files, 3)
        old = EPLBConfig.__new__(EPLBConfig)
        old.__setstate__((5000, "NONE", 0, "mix", 0, 10, 100, False, 1))
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
        with patch.dict(os.environ, {}, clear=True):
            configs = setup_args([])
        configs.role_config.role_type = RoleType.PREFILL
        configs.vit_config.vit_separation = VitSeparation.VIT_SEPARATION_ROLE

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
        for env_interval, cli_interval, expected in (
            (-1, 0, 0),
            (-1, 7, 7),
            (7, -1, 100),
        ):
            with self.subTest(cli_interval=cli_interval), patch.dict(
                os.environ,
                {
                    "ROLE_TYPE": "PREFILL",
                    "EPLB_METRICS_REPORT_INTERVAL": str(env_interval),
                },
                clear=True,
            ):
                configs = setup_args([f"--eplb_metrics_report_interval={cli_interval}"])
                EngineConfig.create(configs)
            self.assertEqual(configs.eplb_config.eplb_metrics_report_interval, expected)

    def test_environment_overrides_automatic_default(self):
        for interval in (0, 7):
            with self.subTest(interval=interval), patch.dict(
                os.environ,
                {"ROLE_TYPE": "PREFILL", "EPLB_METRICS_REPORT_INTERVAL": str(interval)},
                clear=True,
            ):
                configs = setup_args([])
                EngineConfig.create(configs)
            self.assertEqual(configs.eplb_config.eplb_metrics_report_interval, interval)

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


class EplbRearrangementConfigTest(unittest.TestCase):
    def test_native_and_cli_defaults_match_deepinfer_schedule(self):
        with patch.dict(os.environ, {}, clear=True):
            for config in (EPLBConfig(), setup_args([]).eplb_config):
                self.assertEqual(config.eplb_update_time, 0)
                self.assertEqual(config.eplb_stats_window_size, 100)
                self.assertFalse(config.eplb_rearrange_after_prefill)
                self.assertEqual(config.eplb_rearrange_p2p_groups, 1)

    def test_new_options_flow_from_cli_and_environment_through_pickle(self):
        env = {
            "EPLB_UPDATE_TIME": "24",
            "EPLB_STATS_WINDOW_SIZE": "8",
            "EPLB_REARRANGE_AFTER_PREFILL": "1",
            "EPLB_REARRANGE_P2P_GROUPS": "3",
        }
        for use_cli in (False, True):
            with self.subTest(use_cli=use_cli), patch.dict(os.environ, env, clear=True):
                args = (
                    [
                        "--eplb_update_time",
                        "0",
                        "--eplb_stats_window_size",
                        "0",
                        "--eplb_rearrange_after_prefill",
                        "0",
                        "--eplb_rearrange_p2p_groups",
                        "1",
                    ]
                    if use_cli
                    else []
                )
                configs = setup_args(args)
                EngineConfig.create(configs)
                restored = pickle.loads(pickle.dumps(configs.eplb_config))
                self.assertEqual(
                    (
                        restored.eplb_update_time,
                        restored.eplb_stats_window_size,
                        restored.eplb_rearrange_after_prefill,
                        restored.eplb_rearrange_p2p_groups,
                    ),
                    (0, 0, False, 1) if use_cli else (24, 8, True, 3),
                )

    def test_legacy_pickles_restore_active_fields_without_retired_options(self):
        old = (5000, "STATS", 32, "mix", 1, 10, 100, True, 2)
        for state in (old, old + (7, 3)):
            with self.subTest(fields=len(state)):
                restored = EPLBConfig.__new__(EPLBConfig)
                restored.__setstate__(state)
                self.assertEqual(restored.eplb_update_time, 5000)
                self.assertEqual(restored.eplb_stats_window_size, 10)
                self.assertEqual(restored.redundant_expert, 32)
                self.assertEqual(restored.eplb_rearrange_p2p_groups, 1)
                self.assertFalse(restored.eplb_rearrange_after_prefill)
                self.assertEqual(
                    restored.eplb_metrics_report_interval, 7 if len(state) == 11 else 0
                )
                self.assertEqual(
                    restored.eplb_metrics_snapshot_max_files,
                    3 if len(state) == 11 else 0,
                )
                for name in (
                    "balance_method",
                    "eplb_balance_layer_per_step",
                    "eplb_force_repack",
                    "eplb_test_mode",
                ):
                    self.assertFalse(hasattr(restored, name))

    def test_window_and_interval_boundaries(self):
        for interval, window, valid in (
            (0, 0, True),
            (0, 100, True),
            (1, 1, True),
            (8, 8, True),
            (8, 3, True),
            (-1, 0, False),
            (0, -1, False),
            (4, 0, False),
            (4, 5, False),
        ):
            with self.subTest(interval=interval, window=window), patch.dict(
                os.environ, {}, clear=True
            ):
                configs = setup_args(
                    [
                        "--eplb_update_time",
                        str(interval),
                        "--eplb_stats_window_size",
                        str(window),
                    ]
                )
                if valid:
                    EngineConfig.create(configs)
                else:
                    with self.assertRaisesRegex(
                        ValueError, "EPLB_(UPDATE_TIME|STATS_WINDOW_SIZE)"
                    ):
                        EngineConfig.create(configs)

    def test_p2p_groups_must_be_positive(self):
        for groups in (0, -1):
            with self.subTest(groups=groups), patch.dict(os.environ, {}, clear=True):
                configs = setup_args(["--eplb_rearrange_p2p_groups", str(groups)])
                with self.assertRaisesRegex(ValueError, "EPLB_REARRANGE_P2P_GROUPS"):
                    EngineConfig.create(configs)

    def test_retired_cli_options_are_rejected(self):
        for name in (
            "balance_method",
            "eplb_balance_layer_per_step",
            "eplb_force_repack",
            "eplb_test_mode",
        ):
            with self.subTest(name=name), patch.dict(os.environ, {}, clear=True):
                with self.assertRaises(SystemExit):
                    setup_args(["--" + name, "1"])


if __name__ == "__main__":
    unittest.main()
