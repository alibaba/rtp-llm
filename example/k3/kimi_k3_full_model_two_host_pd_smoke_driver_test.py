import argparse
import os
import shlex
import subprocess
import unittest
from types import SimpleNamespace
from unittest import mock

import kimi_k3_full_model_two_host_pd_smoke_driver as driver


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

    def test_parallel_start_defers_readiness_to_role_scripts(self):
        args = SimpleNamespace(parallel_start=True, prefill_start_delay_s=15)
        with mock.patch.object(
            driver, "wait_for_decode_ready"
        ) as ready, mock.patch.object(driver.time, "sleep") as sleep:
            driver.wait_before_prefill(args)
        ready.assert_not_called()
        sleep.assert_not_called()

    def test_sequential_start_retains_decode_readiness_gate(self):
        args = SimpleNamespace(parallel_start=False, prefill_start_delay_s=15)
        with mock.patch.object(
            driver, "wait_for_decode_ready"
        ) as ready, mock.patch.object(driver.time, "sleep") as sleep:
            driver.wait_before_prefill(args)
        sleep.assert_called_once_with(15)
        ready.assert_called_once_with(args)

    def readiness_args(self):
        return SimpleNamespace(
            decode_endpoint="localhost:28188",
            decode_ready_timeout_s=60,
            remote_detached=True,
            remote_control_root="/tmp/mtp-test",
            run_id="failed-start",
            container="lhc_GPU",
            container_user="luohaocheng.lhc",
            prefill_container_runtime="pouch",
            decode_container_runtime="pouch",
        )

    def test_readiness_stops_on_confirmed_detached_role_exit(self):
        result = subprocess.CompletedProcess([], 42, "SMOKE_ROLE_EXIT=1\n", "")
        with mock.patch.object(driver, "run_short_ssh", return_value=result) as run:
            with self.assertRaisesRegex(RuntimeError, "Decode role exited"):
                driver.wait_for_decode_ready(self.readiness_args())
        self.assertIn("controller/decode.status", run.call_args.args[2])
        self.assertTrue(
            run.call_args.args[2].startswith("pouch exec -u luohaocheng.lhc lhc_GPU ")
        )

    def test_readiness_retries_observation_failure_without_restarting_role(self):
        ready = subprocess.CompletedProcess([], 0, "", "")
        with mock.patch.object(
            driver, "run_short_ssh", side_effect=[OSError("relay unavailable"), ready]
        ) as run, mock.patch.object(driver.time, "sleep"):
            driver.wait_for_decode_ready(self.readiness_args())
        self.assertEqual(run.call_count, 2)

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
            self.assertEqual(
                driver.forwarded_optional_environment("prefill")["SMOKE_ACCL_USE_NICS"],
                value,
            )
            self.assertEqual(
                driver.forwarded_optional_environment("decode")["SMOKE_ACCL_USE_NICS"],
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


if __name__ == "__main__":
    unittest.main()
