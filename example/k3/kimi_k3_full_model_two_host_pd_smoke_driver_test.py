import argparse
import os
import pathlib
import shlex
import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

from example.k3 import kimi_k3_full_model_two_host_pd_smoke_driver as driver


class ForwardedOptionalEnvironmentTest(unittest.TestCase):
    def test_draft_mode_defaults_and_explicit_overrides(self) -> None:
        for settings, expected in (
            ({}, ("mtp", "kimi_k3_mtp")),
            ({"SP_TYPE": "", "SP_MODEL_TYPE": ""}, ("mtp", "kimi_k3_mtp")),
            ({"SP_TYPE": "mtp"}, ("mtp", "kimi_k3_mtp")),
            (
                {"SP_TYPE": "mtp", "SP_MODEL_TYPE": "kimi_k3_mtp"},
                ("mtp", "kimi_k3_mtp"),
            ),
        ):
            with self.subTest(settings=settings), mock.patch.dict(
                os.environ, settings, clear=True
            ):
                for role in ("prefill", "decode"):
                    env = driver.forwarded_optional_environment(role)
                    self.assertEqual((env["SP_TYPE"], env["SP_MODEL_TYPE"]), expected)

    def test_invalid_draft_modes_fail_before_remote_launch(self) -> None:
        for settings in (
            {"SP_TYPE": "none"},
            {"SP_TYPE": "eagle3"},
            {"SP_TYPE": "eagle3", "SP_MODEL_TYPE": "kimi_k3_mla_swa_eagle3"},
            {"SP_MODEL_TYPE": "kimi_k3_mla_swa_eagle3"},
            {"SP_TYPE": "eagle3", "SP_MODEL_TYPE": "kimi_k3_mtp"},
        ):
            with self.subTest(settings=settings), mock.patch.dict(
                os.environ, settings, clear=True
            ):
                for role in ("prefill", "decode"):
                    with self.assertRaises(ValueError):
                        driver.forwarded_optional_environment(role)

    def test_legacy_aux_layers_are_not_forwarded(self) -> None:
        with mock.patch.dict(
            os.environ, {"KIMI_K3_EAGLE3_AUX_LAYER_IDS": "0,44,88"}, clear=True
        ):
            for role in ("prefill", "decode"):
                env = driver.forwarded_optional_environment(role)
                self.assertEqual(env["SP_TYPE"], "mtp")
                self.assertEqual(env["SP_MODEL_TYPE"], "kimi_k3_mtp")
                self.assertNotIn("KIMI_K3_EAGLE3_AUX_LAYER_IDS", env)

    def test_forwards_dcp_padding_regression_to_both_roles(self) -> None:
        with mock.patch.dict(
            os.environ, {"SMOKE_DCP_PADDING_REGRESSION": "1"}, clear=True
        ):
            for role in ("prefill", "decode"):
                self.assertEqual(
                    driver.forwarded_optional_environment(role).get(
                        "SMOKE_DCP_PADDING_REGRESSION"
                    ),
                    "1",
                )

    def test_forwards_page_rr_profile_to_both_roles(self) -> None:
        with mock.patch.dict(os.environ, {"SMOKE_PAGE_RR": "1"}, clear=True):
            for role in ("prefill", "decode"):
                self.assertEqual(
                    driver.forwarded_optional_environment(role)["SMOKE_PAGE_RR"],
                    "1",
                )

    def test_prefill_and_decode_page_rr_are_independent(self) -> None:
        for prefill in ("0", "1"):
            settings = {"SMOKE_PAGE_RR": prefill, "SMOKE_DECODE_PAGE_RR": "1"}
            with self.subTest(prefill=prefill), mock.patch.dict(os.environ, settings, clear=True):
                for role in ("prefill", "decode"):
                    forwarded = driver.forwarded_optional_environment(role)
                    for key, value in settings.items():
                        self.assertEqual(forwarded[key], value)

    def test_precision_modes_and_prefix_budget_reach_both_role_commands(self) -> None:
        args = argparse.Namespace(
            prefill_repo_root="/data1/prefill",
            decode_repo_root="/data1/decode",
            prefill_checkpoint_path="/data1/target",
            decode_checkpoint_path="/data1/target",
            prefill_sp_checkpoint_path="/data1/draft",
            decode_sp_checkpoint_path="/data1/draft",
            prefill_container_runtime="docker",
            decode_container_runtime="docker",
            container="lhc_GPU",
            container_user="luohaocheng.lhc",
            prefill_endpoint="prefill:27188",
            decode_endpoint="decode:28188",
            run_id="precision-test",
            suite="all",
            result_endpoint=None,
        )
        for weight in ("0", "1"):
            for mla in ("0", "1"):
                for budget in ("0", "268435456"):
                    settings = {
                        "FP8_GEMM": weight,
                        "FP8_KV_CACHE": mla,
                        "FP8_MLA": mla,
                        "KIMI_K3_MLA_PREFILL_EXPANDED_KV_BUDGET_BYTES": budget,
                        "LOAD_METHOD": "fastsafetensors",
                    }
                    with self.subTest(
                        weight=weight, mla=mla, budget=budget
                    ), mock.patch.dict(os.environ, settings, clear=True):
                        for role in ("prefill", "decode"):
                            command = driver.build_remote_command(args, role)
                            inner = shlex.split(command)[-1]
                            tokens = shlex.split(inner)
                            for key, value in settings.items():
                                self.assertIn(f"{key}={value}", tokens)
                            self.assertIn("SP_TYPE=mtp", tokens)
                            self.assertIn("SP_MODEL_TYPE=kimi_k3_mtp", tokens)
                            self.assertEqual(tokens[-2:], [driver.ROLE_SCRIPT, role])

    def test_forwards_mtp_mode_to_both_roles(self) -> None:
        mode = {
            "SP_TYPE": "mtp",
            "SP_MODEL_TYPE": "kimi_k3_mtp",
            "TP_SIZE": "4",
            "EP_SIZE": "4",
            "KIMI_K3_TP_SIZE": "4",
            "KIMI_K3_EP_SIZE": "4",
            "GEN_NUM_PER_CIRCLE": "1",
        }
        with mock.patch.dict(os.environ, mode, clear=True):
            for role in ("prefill", "decode"):
                forwarded = driver.forwarded_optional_environment(role)
                for key, value in mode.items():
                    self.assertEqual(forwarded[key], value)

    def test_role_artifact_roots_override_shared_default(self) -> None:
        settings = {
            "SMOKE_ARTIFACT_ROOT": "/tmp/shared-default",
            "PREFILL_SMOKE_ARTIFACT_ROOT": "/data4/user/mtp-smoke",
            "DECODE_SMOKE_ARTIFACT_ROOT": "/data5/user/mtp-smoke",
        }
        with mock.patch.dict(os.environ, settings, clear=True):
            for role in ("prefill", "decode"):
                self.assertEqual(
                    driver.forwarded_optional_environment(role)["SMOKE_ARTIFACT_ROOT"],
                    settings[f"{role.upper()}_SMOKE_ARTIFACT_ROOT"],
                )
            del os.environ["PREFILL_SMOKE_ARTIFACT_ROOT"]
            self.assertEqual(
                driver.forwarded_optional_environment("prefill")["SMOKE_ARTIFACT_ROOT"],
                settings["SMOKE_ARTIFACT_ROOT"],
            )

    def test_forwards_explicit_rdma_hca_allowlist_to_both_roles(self) -> None:
        value = "mlx5_bond_0,mlx5_bond_1"
        with mock.patch.dict(os.environ, {"SMOKE_ACCL_USE_NICS": value}, clear=True):
            for role in ("prefill", "decode"):
                with self.subTest(role=role):
                    self.assertEqual(
                        driver.forwarded_optional_environment(role)[
                            "SMOKE_ACCL_USE_NICS"
                        ],
                        value,
                    )

    def test_does_not_invent_driver_override_when_unset(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertNotIn(
                "SMOKE_ACCL_USE_NICS",
                driver.forwarded_optional_environment("prefill"),
            )

    def test_forwards_keep_cluster_mode_to_both_roles(self) -> None:
        with mock.patch.dict(
            os.environ, {"SMOKE_KEEP_CLUSTER_ON_SUCCESS": "1"}, clear=True
        ):
            for role in ("prefill", "decode"):
                self.assertEqual(
                    driver.forwarded_optional_environment(role)[
                        "SMOKE_KEEP_CLUSTER_ON_SUCCESS"
                    ],
                    "1",
                )


class KimiK3FullModelTwoHostPdSmokeDriverTest(unittest.TestCase):
    def test_role_script_rejects_non_native_mtp_before_host_validation(self):
        role_script = pathlib.Path(driver.__file__).with_name(
            "kimi_k3_full_model_two_host_pd_smoke.sh"
        )
        for role in ("prefill", "decode"):
            for settings, expected in (
                ({"SP_TYPE": "eagle3"}, "SP_TYPE=mtp"),
                (
                    {"SP_TYPE": "eagle3", "SP_MODEL_TYPE": "kimi_k3_mla_swa_eagle3"},
                    "SP_TYPE=mtp",
                ),
                ({"SP_TYPE": "none"}, "SP_TYPE=mtp"),
                (
                    {"SP_TYPE": "mtp", "SP_MODEL_TYPE": "kimi_k3_mla_swa_eagle3"},
                    "SP_MODEL_TYPE=kimi_k3_mtp",
                ),
            ):
                with self.subTest(role=role, settings=settings):
                    result = subprocess.run(
                        [role_script, role],
                        env={"PATH": os.environ["PATH"], **settings},
                        text=True,
                        capture_output=True,
                        check=False,
                        timeout=10,
                    )
                    self.assertEqual(result.returncode, 2, result.stderr)
                    self.assertIn(expected, result.stderr)

    def test_dcp_padding_regression_resolves_and_validates_profile(self):
        source = pathlib.Path(driver.__file__).with_name(
            "kimi_k3_full_model_two_host_pd_smoke.sh"
        ).read_text()
        # Execute real argument/profile handling, stopping before host access.
        prefix = source.split('[[ "$(id -u)" != "0" ]]', 1)[0]
        command = prefix + "\nprintf '%s %s %s' \"$smoke_decode_page_rr\" \"${GEN_NUM_PER_CIRCLE:-unset}\" \"${SMOKE_SUITE:-unset}\"\n"
        for settings, expected in (
            ({}, "0 unset unset"),
            ({"SMOKE_DCP_PADDING_REGRESSION": "0"}, "0 unset unset"),
            ({"SMOKE_DCP_PADDING_REGRESSION": "1"}, "1 2 all"),
            ({"SMOKE_DECODE_PAGE_RR": "1", "GEN_NUM_PER_CIRCLE": "3"}, "1 3 unset"),
            ({"SMOKE_DCP_PADDING_REGRESSION": "1", "GEN_NUM_PER_CIRCLE": "3"}, None),
            ({"SMOKE_DCP_PADDING_REGRESSION": "1", "SMOKE_DECODE_PAGE_RR": "0"}, None),
            ({"SMOKE_DCP_PADDING_REGRESSION": "1", "SMOKE_SUITE": "flow"}, None),
            ({"SMOKE_DCP_PADDING_REGRESSION": "bad"}, None),
        ):
            for role in ("prefill", "decode"):
                with self.subTest(settings=settings, role=role):
                    result = subprocess.run(
                        ["bash", "-c", command, "smoke-profile-test", role],
                        env={"PATH": os.environ["PATH"], **settings},
                        capture_output=True, text=True, timeout=10,
                    )
                    if expected is None:
                        self.assertEqual(result.returncode, 2, result.stderr)
                        self.assertIn("SMOKE_DCP_PADDING_REGRESSION", result.stderr)
                    else:
                        self.assertEqual(result.returncode, 0, result.stderr)
                        self.assertEqual(result.stdout, expected)

    def test_role_script_rejects_deployment_profile_drift(self):
        script = pathlib.Path(driver.__file__).with_name(
            "kimi_k3_full_model_two_host_pd_smoke.sh"
        )
        for settings, message in (
            ({"TP_SIZE": "16", "EP_SIZE": "16"}, "Prefill TP8/EP8"),
            ({"SMOKE_CHUNKWISE_RDMA": "0"}, "SMOKE_CHUNKWISE_RDMA=1"),
        ):
            for role in ("prefill", "decode"):
                with self.subTest(settings=settings, role=role):
                    result = subprocess.run(
                        [script, role],
                        env={"PATH": os.environ["PATH"], **settings},
                        text=True,
                        capture_output=True,
                        check=False,
                        timeout=10,
                    )
                    self.assertEqual(result.returncode, 2, result.stderr)
                    self.assertIn(message, result.stderr)

    def test_decode_profile_preserves_independent_source_and_destination_layouts(self):
        role_script = pathlib.Path(driver.__file__).with_name(
            "kimi_k3_full_model_two_host_pd_smoke.sh"
        )
        source = role_script.read_text()
        # Execute the actual profile function, without starting a service or
        # duplicating its export logic in the test harness.
        definition = (
            "apply_validated_decode_profile() {"
            + source.split("apply_validated_decode_profile() {", 1)[1].split(
                "\n}", 1
            )[0]
            + "\n}"
        )
        for source_rr in ("0", "1"):
            for destination_rr in ("0", "1"):
                topology = "tp8_ep8" if destination_rr == "1" else "dp8_ktp8_ep8"
                env = {
                    **os.environ,
                    "smoke_page_rr": source_rr,
                    "smoke_decode_page_rr": destination_rr,
                    "smoke_decode_topology": topology,
                    "smoke_tp_size": "8",
                    "smoke_decode_kv_cache_mem_mb": "20000",
                    "smoke_decode_kda_pool_blocks": "32",
                }
                command = (
                    definition
                    + "\napply_validated_decode_profile\n"
                    + "printf '%s %s %s' \"$KIMI_K3_DECODE_TOPOLOGY\" "
                    + "\"$DECODE_CP_KV_CACHE_SHARDED\" "
                    + "\"${PREFILL_CP_SIZE:-unset}\"\n"
                )
                with self.subTest(source=source_rr, destination=destination_rr):
                    result = subprocess.run(
                        ["bash", "-eu", "-c", command],
                        env=env,
                        text=True,
                        capture_output=True,
                        check=True,
                    )
                    upstream = "8" if source_rr == "1" else "unset"
                    self.assertEqual(result.stdout, f"{topology} {destination_rr} {upstream}")

    def test_role_script_rejects_page_rr_tp16_before_host_validation(self):
        role_script = pathlib.Path(driver.__file__).with_name(
            "kimi_k3_full_model_two_host_pd_smoke.sh"
        )
        env = {
            **os.environ,
            "SMOKE_PAGE_RR": "1",
            "SMOKE_DECODE_PAGE_RR": "0",
            "TP_SIZE": "16",
            "EP_SIZE": "16",
        }

        result = subprocess.run(
            [role_script, "decode"],
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn(
            "two-host SMOKE_PAGE_RR profile validates only Prefill TP8 -> Decode DP8",
            result.stderr,
        )

    def test_parse_args_requires_both_draft_checkpoints(self):
        argv = [
            "driver",
            "--prefill-ssh-target",
            "prefill-host",
            "--decode-ssh-target",
            "decode-host",
            "--prefill-repo-root",
            "/prefill/repo",
            "--decode-repo-root",
            "/decode/repo",
            "--prefill-checkpoint-path",
            "/prefill/checkpoint",
            "--decode-checkpoint-path",
            "/decode/checkpoint",
            "--prefill-sp-checkpoint-path",
            "/prefill/mtp",
            "--decode-sp-checkpoint-path",
            "/decode/mtp",
            "--prefill-endpoint",
            "10.0.0.1:27188",
            "--decode-endpoint",
            "10.0.0.2:28188",
            "--run-id",
            "projection-ktp",
        ]
        with mock.patch.object(sys, "argv", argv), mock.patch.dict(
            os.environ, {}, clear=True
        ):
            args = driver.parse_args()
        self.assertEqual(args.prefill_sp_checkpoint_path, "/prefill/mtp")
        self.assertEqual(args.decode_sp_checkpoint_path, "/decode/mtp")

    def test_parse_args_rejects_missing_decode_sp_checkpoint(self):
        argv = [
            "driver",
            "--prefill-ssh-target",
            "prefill-host",
            "--decode-ssh-target",
            "decode-host",
            "--prefill-repo-root",
            "/prefill/repo",
            "--decode-repo-root",
            "/decode/repo",
            "--prefill-checkpoint-path",
            "/prefill/checkpoint",
            "--decode-checkpoint-path",
            "/decode/checkpoint",
            "--prefill-sp-checkpoint-path",
            "/mtp/checkpoint",
            "--prefill-endpoint",
            "10.0.0.1:27188",
            "--decode-endpoint",
            "10.0.0.2:28188",
            "--run-id",
            "projection-ktp",
        ]
        with mock.patch.object(sys, "argv", argv), mock.patch.dict(
            os.environ, {}, clear=True
        ):
            with self.assertRaises(SystemExit):
                driver.parse_args()

    def test_role_command_exports_role_local_sp_checkpoint(self):
        args = argparse.Namespace(
            prefill_repo_root="/prefill/repo",
            decode_repo_root="/decode/repo",
            prefill_checkpoint_path="/prefill/checkpoint",
            decode_checkpoint_path="/decode/checkpoint",
            prefill_sp_checkpoint_path="/prefill/mtp",
            decode_sp_checkpoint_path="/decode/mtp",
            prefill_endpoint="10.0.0.1:27188",
            decode_endpoint="10.0.0.2:28188",
            run_id="projection-ktp",
            suite="flow",
            result_endpoint=None,
            container="lhc_GPU",
            prefill_container_runtime="docker",
            decode_container_runtime="docker",
        )
        with mock.patch.dict(os.environ, {}, clear=True):
            _, _, _, command = driver.role_launch_parts(args, "decode")
        self.assertIn("SP_CHECKPOINT_PATH=/decode/mtp", command)

    def test_role_command_preserves_long_prefix_overrides_for_both_hosts(self):
        args = argparse.Namespace(
            prefill_repo_root="/prefill/repo",
            decode_repo_root="/decode/repo",
            prefill_checkpoint_path="/prefill/checkpoint",
            decode_checkpoint_path="/decode/checkpoint",
            prefill_sp_checkpoint_path="/prefill/mtp",
            decode_sp_checkpoint_path="/decode/mtp",
            prefill_endpoint="10.0.0.1:27188",
            decode_endpoint="10.0.0.2:28188",
            run_id="long-prefix-overrides",
            suite="all",
            result_endpoint=None,
            container="lhc_GPU",
            prefill_container_runtime="docker",
            decode_container_runtime="docker",
        )
        settings = {
            "SMOKE_LONG_PREFIX_TARGET_TOKENS": "960000",
            "SMOKE_LONG_PREFIX_TP_SIZE": "1",
        }
        with mock.patch.dict(os.environ, settings, clear=True):
            for role in ("prefill", "decode"):
                with self.subTest(role=role):
                    _, _, _, command = driver.role_launch_parts(args, role)
                    for key, value in settings.items():
                        self.assertIn(f"{key}={value}", command)

    def test_role_command_forwards_core_dump_diagnostic_override(self):
        args = argparse.Namespace(
            prefill_repo_root="/prefill/repo",
            decode_repo_root="/decode/repo",
            prefill_checkpoint_path="/prefill/checkpoint",
            decode_checkpoint_path="/decode/checkpoint",
            prefill_sp_checkpoint_path="/prefill/mtp",
            decode_sp_checkpoint_path="/decode/mtp",
            prefill_endpoint="10.0.0.1:27188",
            decode_endpoint="10.0.0.2:28188",
            run_id="projection-ktp",
            suite="flow",
            result_endpoint=None,
            container="lhc_GPU",
            prefill_container_runtime="docker",
            decode_container_runtime="docker",
        )
        with mock.patch.dict(
            os.environ,
            {"FT_CORE_DUMP_ON_EXCEPTION": "0"},
            clear=True,
        ):
            _, _, _, command = driver.role_launch_parts(args, "prefill")
        self.assertIn("FT_CORE_DUMP_ON_EXCEPTION=0", command)

    def test_role_command_forwards_decode_cache_and_prewarm_timeout(self):
        args = argparse.Namespace(
            prefill_repo_root="/prefill/repo",
            decode_repo_root="/decode/repo",
            prefill_checkpoint_path="/prefill/checkpoint",
            decode_checkpoint_path="/decode/checkpoint",
            prefill_sp_checkpoint_path="/prefill/mtp",
            decode_sp_checkpoint_path="/decode/mtp",
            prefill_endpoint="10.0.0.1:27188",
            decode_endpoint="10.0.0.2:28188",
            run_id="projection-ktp",
            suite="all",
            result_endpoint=None,
            container="lhc_GPU",
            prefill_container_runtime="docker",
            decode_container_runtime="docker",
        )
        with mock.patch.dict(
            os.environ,
            {
                "SMOKE_DECODE_KV_CACHE_MEM_MB": "26000",
                "SMOKE_DECODE_KDA_POOL_BLOCKS": "40",
                "SMOKE_DECODE_ROLE_ADDRS": (
                    "10.0.0.2:28188:28189,10.0.0.2:28197:28198"
                ),
                "SMOKE_RDMA_PREWARM_TIMEOUT_S": "300",
            },
            clear=True,
        ):
            _, _, _, command = driver.role_launch_parts(args, "decode")
        self.assertIn("SMOKE_DECODE_KV_CACHE_MEM_MB=26000", command)
        self.assertIn("SMOKE_DECODE_KDA_POOL_BLOCKS=40", command)
        self.assertIn(
            "SMOKE_DECODE_ROLE_ADDRS=10.0.0.2:28188:28189,10.0.0.2:28197:28198",
            command,
        )
        self.assertIn("SMOKE_RDMA_PREWARM_TIMEOUT_S=300", command)

    def test_start_remote_roles_launches_prefill_without_decode_health_gate(self):
        events = []

        class FakeRole:
            def __init__(self, role):
                self.role = role

            def start(self):
                events.append(f"start:{self.role}")

        args = SimpleNamespace(prefill_start_delay_s=3.5)
        roles = {
            "decode": FakeRole("decode"),
            "prefill": FakeRole("prefill"),
        }
        with mock.patch.object(
            driver.time,
            "sleep",
            side_effect=lambda seconds: events.append(f"sleep:{seconds}"),
        ):
            driver.start_remote_roles(args, roles)

        self.assertEqual(
            events,
            ["start:decode", "sleep:3.5", "start:prefill"],
        )

    def test_detached_control_operations_run_inside_role_container(self):
        args = SimpleNamespace(
            prefill_repo_root="/prefill/repo",
            decode_repo_root="/decode/repo",
            prefill_checkpoint_path="/prefill/checkpoint",
            decode_checkpoint_path="/decode/checkpoint",
            prefill_sp_checkpoint_path="/prefill/mtp",
            decode_sp_checkpoint_path="/decode/mtp",
            prefill_endpoint="10.0.0.1:27188",
            decode_endpoint="10.0.0.2:28188",
            run_id="projection-ktp",
            suite="flow",
            result_endpoint=None,
            container="lhc_GPU",
            container_user="19357313:100",
            prefill_container_runtime="docker",
            decode_container_runtime="docker",
        )

        command = driver.build_detached_control_command(
            args, "decode", "cat /tmp/projection-ktp/decode.status"
        )

        self.assertEqual(
            shlex.split(command),
            [
                "docker",
                "exec",
                "-u",
                "19357313:100",
                "lhc_GPU",
                "bash",
                "-lc",
                "cat /tmp/projection-ktp/decode.status",
            ],
        )

    def test_role_specific_containers_override_shared_container(self):
        args = SimpleNamespace(
            prefill_repo_root="/prefill/repo",
            decode_repo_root="/decode/repo",
            prefill_checkpoint_path="/prefill/checkpoint",
            decode_checkpoint_path="/decode/checkpoint",
            prefill_sp_checkpoint_path="/prefill/mtp",
            decode_sp_checkpoint_path="/decode/mtp",
            prefill_endpoint="10.0.0.1:27188",
            decode_endpoint="10.0.0.2:28188",
            run_id="projection-ktp",
            suite="flow",
            result_endpoint=None,
            container="lhc_GPU",
            prefill_container="lhc_GPU_prefill",
            decode_container="lhc_GPU_decode",
            prefill_container_runtime="pouch",
            decode_container_runtime="pouch",
            container_user="luohaocheng.lhc",
        )

        self.assertEqual(
            driver.role_launch_parts(args, "prefill")[2], "lhc_GPU_prefill"
        )
        self.assertEqual(driver.role_launch_parts(args, "decode")[2], "lhc_GPU_decode")


if __name__ == "__main__":
    unittest.main()
