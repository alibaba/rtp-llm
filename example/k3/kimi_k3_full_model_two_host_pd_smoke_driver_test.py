import os
import argparse
import shlex
import unittest
from unittest import mock

import kimi_k3_full_model_two_host_pd_smoke_driver as driver


class ForwardedOptionalEnvironmentTest(unittest.TestCase):
    def test_precision_modes_and_prefix_budget_reach_both_role_commands(self) -> None:
        args = argparse.Namespace(
            prefill_repo_root="/data1/prefill", decode_repo_root="/data1/decode",
            prefill_checkpoint_path="/data1/target", decode_checkpoint_path="/data1/target",
            prefill_sp_checkpoint_path="/data1/draft", decode_sp_checkpoint_path="/data1/draft",
            prefill_container_runtime="docker", decode_container_runtime="docker",
            container="lhc_GPU", container_user="luohaocheng.lhc",
            prefill_endpoint="prefill:27188", decode_endpoint="decode:28188",
            run_id="precision-test", suite="all", result_endpoint=None,
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
                        "KIMI_K3_FP8_COLLECTIVE_GEMM": "1",
                        "LOAD_METHOD": "fastsafetensors",
                    }
                    with self.subTest(weight=weight, mla=mla, budget=budget), mock.patch.dict(os.environ, settings, clear=True):
                        for role in ("prefill", "decode"):
                            command = driver.build_remote_command(args, role)
                            inner = shlex.split(command)[-1]
                            tokens = shlex.split(inner)
                            for key, value in settings.items():
                                self.assertIn(f"{key}={value}", tokens)
                            self.assertEqual(tokens[-2:], [driver.ROLE_SCRIPT, role])

    def test_forwards_explicit_rdma_hca_allowlist_to_both_roles(self) -> None:
        value = "mlx5_bond_0,mlx5_bond_1"
        with mock.patch.dict(os.environ, {"SMOKE_ACCL_USE_NICS": value}, clear=True):
            self.assertEqual(
                driver.forwarded_optional_environment("prefill")[
                    "SMOKE_ACCL_USE_NICS"
                ],
                value,
            )
            self.assertEqual(
                driver.forwarded_optional_environment("decode")[
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


if __name__ == "__main__":
    unittest.main()
