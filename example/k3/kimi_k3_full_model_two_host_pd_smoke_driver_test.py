import argparse
import os
import shlex
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

from example.k3 import kimi_k3_full_model_two_host_pd_smoke_driver as driver


class KimiK3FullModelTwoHostPdSmokeDriverTest(unittest.TestCase):
    def test_parse_args_accepts_ordinary_decode_without_sp_checkpoint(self):
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
            "--prefill-endpoint",
            "10.0.0.1:27188",
            "--decode-endpoint",
            "10.0.0.2:28188",
            "--run-id",
            "projection-ktp",
        ]
        with mock.patch.object(sys, "argv", argv), mock.patch.dict(os.environ, {}, clear=True):
            args = driver.parse_args()
        self.assertIsNone(args.prefill_sp_checkpoint_path)
        self.assertIsNone(args.decode_sp_checkpoint_path)

    def test_parse_args_rejects_sp_checkpoint(self):
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

    def test_role_command_does_not_export_sp_checkpoint(self):
        args = argparse.Namespace(
            prefill_repo_root="/prefill/repo",
            decode_repo_root="/decode/repo",
            prefill_checkpoint_path="/prefill/checkpoint",
            decode_checkpoint_path="/decode/checkpoint",
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
        self.assertFalse(any(part.startswith("SP_CHECKPOINT_PATH=") for part in command))

    def test_role_command_forwards_core_dump_diagnostic_override(self):
        args = argparse.Namespace(
            prefill_repo_root="/prefill/repo",
            decode_repo_root="/decode/repo",
            prefill_checkpoint_path="/prefill/checkpoint",
            decode_checkpoint_path="/decode/checkpoint",
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


if __name__ == "__main__":
    unittest.main()
