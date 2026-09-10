import argparse
import os
import shlex
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

from example.k3 import kimi_k3_full_model_two_host_pd_smoke_driver as driver

class ForwardedOptionalEnvironmentTest(unittest.TestCase):
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
        for weight in ("none", "fp8_per_block"):
            for mla in ("0", "1"):
                for budget in ("0", "268435456"):
                    settings = {
                        "KIMI_K3_ATTENTION_QUANTIZATION": weight,
                        "KIMI_K3_MLA_FP8": mla,
                        "KIMI_K3_MLA_FP8_Q_SCALE": "0.5",
                        "KIMI_K3_MLA_FP8_KV_SCALE": "0.5",
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
    def test_parse_args_requires_both_eagle3_checkpoints(self):
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
            "/prefill/eagle3",
            "--decode-sp-checkpoint-path",
            "/decode/eagle3",
            "--prefill-endpoint",
            "10.0.0.1:27188",
            "--decode-endpoint",
            "10.0.0.2:28188",
            "--run-id",
            "projection-ktp",
        ]
        with mock.patch.object(sys, "argv", argv), mock.patch.dict(os.environ, {}, clear=True):
            args = driver.parse_args()
        self.assertEqual(args.prefill_sp_checkpoint_path, "/prefill/eagle3")
        self.assertEqual(args.decode_sp_checkpoint_path, "/decode/eagle3")

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
        with mock.patch.object(sys, "argv", argv), mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(SystemExit):
                driver.parse_args()

    def test_role_command_exports_role_local_sp_checkpoint(self):
        args = argparse.Namespace(
            prefill_repo_root="/prefill/repo",
            decode_repo_root="/decode/repo",
            prefill_checkpoint_path="/prefill/checkpoint",
            decode_checkpoint_path="/decode/checkpoint",
            prefill_sp_checkpoint_path="/prefill/eagle3",
            decode_sp_checkpoint_path="/decode/eagle3",
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
        self.assertIn("SP_CHECKPOINT_PATH=/decode/eagle3", command)

    def test_role_command_forwards_core_dump_diagnostic_override(self):
        args = argparse.Namespace(
            prefill_repo_root="/prefill/repo",
            decode_repo_root="/decode/repo",
            prefill_checkpoint_path="/prefill/checkpoint",
            decode_checkpoint_path="/decode/checkpoint",
            prefill_sp_checkpoint_path="/prefill/eagle3",
            decode_sp_checkpoint_path="/decode/eagle3",
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
            prefill_sp_checkpoint_path="/prefill/eagle3",
            decode_sp_checkpoint_path="/decode/eagle3",
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
            driver.time, "sleep", side_effect=lambda seconds: events.append(f"sleep:{seconds}")
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
            prefill_sp_checkpoint_path="/prefill/eagle3",
            decode_sp_checkpoint_path="/decode/eagle3",
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
            prefill_sp_checkpoint_path="/prefill/eagle3",
            decode_sp_checkpoint_path="/decode/eagle3",
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

        self.assertEqual(driver.role_launch_parts(args, "prefill")[2], "lhc_GPU_prefill")
        self.assertEqual(driver.role_launch_parts(args, "decode")[2], "lhc_GPU_decode")


if __name__ == "__main__":
    unittest.main()
