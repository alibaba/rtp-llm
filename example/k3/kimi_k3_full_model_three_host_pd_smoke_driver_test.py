import argparse
import os
import shlex
import sys
import unittest
from unittest import mock

from example.k3 import kimi_k3_full_model_three_host_pd_smoke_driver as driver


def make_args() -> argparse.Namespace:
    return argparse.Namespace(
        prefill_ssh_target="prefill-host",
        decode0_ssh_target="decode0-host",
        decode1_ssh_target="decode1-host",
        prefill_repo_root="/prefill/repo",
        decode0_repo_root="/decode0/repo",
        decode1_repo_root="/decode1/repo",
        prefill_checkpoint_path="/prefill/checkpoint",
        decode0_checkpoint_path="/decode0/checkpoint",
        decode1_checkpoint_path="/decode1/checkpoint",
        prefill_container_runtime="docker",
        decode0_container_runtime="docker",
        decode1_container_runtime="docker",
        prefill_ssh_control_path=None,
        decode0_ssh_control_path=None,
        decode1_ssh_control_path=None,
        prefill_endpoint="10.0.0.1:27188",
        decode0_endpoint="10.0.0.2:28188",
        decode1_endpoint="10.0.0.3:28188",
        result_endpoint="10.0.0.2:28288",
        run_id="dp16-flow",
        suite="flow",
        container="lhc_GPU",
        container_user="luohaocheng.lhc",
        ssh_bin="ssh",
        remote_control_root="/tmp/k3-three-host",
        prefill_start_delay_s=0,
    )


class ThreeHostDriverTest(unittest.TestCase):
    def test_parse_args_and_derive_result_endpoint(self):
        argv = [
            "driver",
            "--prefill-ssh-target", "p",
            "--decode0-ssh-target", "d0",
            "--decode1-ssh-target", "d1",
            "--prefill-repo-root", "/p",
            "--decode0-repo-root", "/d0",
            "--decode1-repo-root", "/d1",
            "--prefill-checkpoint-path", "/cp",
            "--decode0-checkpoint-path", "/cd0",
            "--decode1-checkpoint-path", "/cd1",
            "--prefill-endpoint", "10.0.0.1:27188",
            "--decode0-endpoint", "10.0.0.2:28188",
            "--decode1-endpoint", "10.0.0.3:28188",
            "--run-id", "dp16-flow",
        ]
        with mock.patch.object(sys, "argv", argv), mock.patch.dict(os.environ, {}, clear=True):
            args = driver.parse_args()
        self.assertEqual(args.result_endpoint, "10.0.0.2:28288")
        self.assertEqual(len(driver.decode_role_addresses(args)), 16)

    def test_ordered_decode_addresses_span_both_nodes(self):
        addresses = driver.decode_role_addresses(make_args())
        self.assertEqual(addresses[0], "10.0.0.2:28188:28189")
        self.assertEqual(addresses[7], "10.0.0.2:28251:28252")
        self.assertEqual(addresses[8], "10.0.0.3:28188:28189")
        self.assertEqual(addresses[15], "10.0.0.3:28251:28252")

    def test_decode_gang_roles_have_world_rank_and_common_plan(self):
        args = make_args()
        _, _, _, decode0 = driver.role_launch_parts(args, "decode0")
        _, _, _, decode1 = driver.role_launch_parts(args, "decode1")
        joined0 = " ".join(decode0)
        joined1 = " ".join(decode1)
        self.assertIn("WORLD_RANK=0", decode0)
        self.assertIn("WORLD_RANK=8", decode1)
        self.assertIn("SMOKE_DECODE_NODE_INDEX=0", decode0)
        self.assertIn("SMOKE_DECODE_NODE_INDEX=1", decode1)
        self.assertIn("SMOKE_DECODE_TOPOLOGY=dp16_ktp16_ep16", joined0)
        self.assertIn("name:k3_part0,ip:10.0.0.2,port:28188", joined0)
        self.assertIn("name:k3_part1,ip:10.0.0.3,port:28188", joined1)
        self.assertIn("SMOKE_PRIMARY_READY_FILE=", joined0)
        self.assertIn("SMOKE_PRIMARY_COMPLETION_FILE=", joined0)
        self.assertIn("SMOKE_SECONDARY_COMPLETION_FILE=", joined1)

    def test_prefill_receives_all_sixteen_ordered_role_endpoints(self):
        _, _, _, command = driver.role_launch_parts(make_args(), "prefill")
        role_arg = next(value for value in command if value.startswith("SMOKE_DECODE_ROLE_ADDRS="))
        self.assertEqual(len(role_arg.split("=", 1)[1].split(",")), 16)
        self.assertEqual(command[-1], "prefill")

    def test_completion_command_targets_decode1_container_only(self):
        args = make_args()
        marker = driver.secondary_completion_file(args)
        command = driver.container_control_command(
            args, "decode1", f"test ! -e {shlex.quote(marker)} && touch {shlex.quote(marker)}"
        )
        words = shlex.split(command)
        self.assertEqual(words[:6], ["docker", "exec", "-u", "luohaocheng.lhc", "lhc_GPU", "bash"])
        self.assertIn("decode1.success", words[-1])

    def test_start_order_launches_both_decode_nodes_before_prefill(self):
        events = []

        class Role:
            def __init__(self, name):
                self.name = name

            def start(self):
                events.append(self.name)

        roles = {name: Role(name) for name in driver.ROLES}
        driver.start_roles(make_args(), roles)
        self.assertEqual(events, ["decode0", "decode1", "prefill"])


if __name__ == "__main__":
    unittest.main()
