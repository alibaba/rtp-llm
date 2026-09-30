import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from rtp_llm.utils import scr_vip


class ScrVipTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "ganginfo"
        self.rows = {
            "model-rank-0": {"ip": "22.0.1.2", "real_ip": "10.0.0.2"},
            "model-rank-1": {"ip": "22.0.1.3", "real_ip": "10.0.0.3"},
        }
        self.path.write_text(json.dumps(self.rows))
        path = patch.object(scr_vip, "GANG_INFO_PATH", self.path)
        path.start()
        self.addCleanup(path.stop)
        env = patch.dict(
            os.environ,
            {
                "RTPLLM_ENABLE_SCR": "1",
                "SCR_PHASE": "checkpoint",
                "RANK_SIZE": "2",
                "RANK_ID": "1",
            },
            clear=True,
        )
        env.start()
        self.addCleanup(env.stop)
        self.pc = SimpleNamespace(
            world_size=4, local_world_size=2, world_rank=2, local_rank=0
        )

    def _world(self):
        from rtp_llm.distribute.distributed_server import WorldInfo
        from rtp_llm.distribute.worker_info import WorkerInfo

        members = [
            WorkerInfo(
                f"22.0.1.{2 + rank // 2}", rank % 2, rank, f"rank_{rank}", 22290, 10
            )
            for rank in range(4)
        ]
        return WorldInfo(members, members[0], members[2], 2, True)

    def test_restore_waits_for_merged_gang_info(self):
        from rtp_llm.utils.scr_restore_context import RestoreContext

        world = self._world()
        incomplete = {key: {"ip": value["real_ip"]} for key, value in self.rows.items()}
        self.path.write_text(json.dumps(incomplete))

        def publish_map(_):
            self.path.write_text(json.dumps(self.rows))

        with patch.dict(os.environ, {"SCR_PHASE": "restore"}), patch.object(
            scr_vip.time, "monotonic", side_effect=[0, 1]
        ), patch.object(
            scr_vip.time, "sleep", side_effect=publish_map
        ) as sleep, patch.object(
            scr_vip, "validate_device"
        ) as device:
            restored = RestoreContext("seed", "10.0.0.3").resolve_world_info(
                world, self.pc
            )
        sleep.assert_called_once_with(0.5)
        device.assert_called_once_with("22.0.1.3")
        self.assertEqual(
            [m.ip for m in restored.members], [m.ip for m in world.members]
        )
        self.assertEqual(
            restored.self.cache_store_listen_port, world.self.cache_store_listen_port
        )

    def test_restore_rejects_persistently_unmerged_gang_info(self):
        from rtp_llm.utils.scr_restore_context import RestoreContext

        world = self._world()
        self.path.write_text(
            json.dumps(
                {key: {"ip": value["real_ip"]} for key, value in self.rows.items()}
            )
        )
        with patch.dict(os.environ, {"SCR_PHASE": "restore"}), patch.object(
            scr_vip.time, "monotonic", side_effect=[0, 1, 121]
        ), patch.object(scr_vip.time, "sleep") as sleep, patch.object(
            scr_vip, "validate_device"
        ) as device:
            with self.assertRaisesRegex(KeyError, "real_ip"):
                RestoreContext("seed", "10.0.0.3").resolve_world_info(world, self.pc)
        sleep.assert_called_once_with(0.5)
        device.assert_not_called()
        self.assertEqual(world.self.ip, "22.0.1.3")

    def test_restore_rejects_changed_vip_after_map_becomes_ready(self):
        from rtp_llm.utils.scr_restore_context import RestoreContext

        world = self._world()
        self.rows["model-rank-0"]["ip"] = "22.0.1.4"
        self.path.write_text(json.dumps(self.rows))
        with patch.dict(os.environ, {"SCR_PHASE": "restore"}), patch.object(
            scr_vip, "validate_device"
        ):
            with self.assertRaisesRegex(RuntimeError, "cannot change"):
                RestoreContext("seed", "10.0.0.3").resolve_world_info(world, self.pc)
        self.assertEqual(world.members[0].ip, "22.0.1.2")

    def test_restore_waits_for_network_and_fails_when_it_stays_unready(self):
        from rtp_llm.utils.scr_restore_context import RestoreContext

        for readiness in (
            [RuntimeError("network not ready"), None],
            [RuntimeError("network not ready")] * 2,
        ):
            with self.subTest(recovers=readiness[-1] is None), patch.dict(
                os.environ, {"SCR_PHASE": "restore"}
            ), patch.object(
                scr_vip.time, "monotonic", side_effect=[0, 1, 121]
            ), patch.object(
                scr_vip.time, "sleep"
            ) as sleep, patch.object(
                scr_vip, "validate_device", side_effect=readiness
            ):
                context = RestoreContext("seed", "10.0.0.3")
                if readiness[-1] is None:
                    self.assertEqual(
                        context.resolve_world_info(self._world(), self.pc).self.ip,
                        "22.0.1.3",
                    )
                else:
                    with self.assertRaisesRegex(RuntimeError, "network not ready"):
                        context.resolve_world_info(self._world(), self.pc)
                sleep.assert_called_once_with(0.5)

    def test_control_plane_addresses_and_new_underlay(self):
        self.assertEqual(scr_vip.read_topology(4, 2), {0: "22.0.1.2", 1: "22.0.1.3"})
        self.rows["model-rank-1"]["real_ip"] = "10.1.2.3"
        self.path.write_text(json.dumps(self.rows))
        self.assertEqual(scr_vip.read_topology(4, 2, external=True)[1], "10.1.2.3")
        with patch.object(scr_vip, "validate_device") as check:
            self.assertEqual(scr_vip.internal_ip(self.pc, "10.0.0.3"), "22.0.1.3")
            check.assert_called_once_with("22.0.1.3")

    def test_worker_node_topology_must_match_gpu_topology(self):
        for name, wrong in (("RANK_SIZE", "1"), ("RANK_ID", "0"), ("RANK_SIZE", "")):
            with self.subTest(name=name, value=wrong), patch.dict(
                os.environ, {name: wrong}
            ), patch.object(scr_vip, "validate_device") as device:
                with self.assertRaisesRegex(ValueError, name):
                    scr_vip.topology(self.pc, wait=True)
                device.assert_not_called()

    def test_incomplete_duplicate_and_unmerged_maps_fail(self):
        bad_maps = [
            {},
            {"model-part0": self.rows["model-rank-0"]},
            {**self.rows, "other-rank-0": self.rows["model-rank-0"]},
            {
                "model-rank-0": self.rows["model-rank-0"],
                "model-rank-1": self.rows["model-rank-0"],
            },
            {"model-rank-0": {"ip": "10.0.0.2", "real_ip": "10.0.0.2"}},
        ]
        for rows in bad_maps:
            with self.subTest(rows=rows):
                self.path.write_text(json.dumps(rows))
                with self.assertRaises(ValueError):
                    scr_vip.read_topology(4, 2)

    def test_only_scr_template_multinode_uses_vip(self):
        for switch in (None, "", "0", "false", "1", "true", "yes", "on"):
            for phase in (None, "normal", "checkpoint", "restore"):
                for local_size in (2, 4):
                    with self.subTest(
                        switch=switch, phase=phase, local_size=local_size
                    ):
                        env = {"SCR_ENABLE": "1"}
                        if switch is not None:
                            env["RTPLLM_ENABLE_SCR"] = switch
                        if phase is not None:
                            env["SCR_PHASE"] = phase
                        self.pc.local_world_size = local_size
                        expected = (
                            switch in {"1", "true", "yes", "on"}
                            and phase in {"checkpoint", "restore"}
                            and local_size == 2
                        )
                        with patch.dict(os.environ, env, clear=True), patch.object(
                            scr_vip,
                            "topology",
                            return_value={0: "22.0.1.2", 1: "22.0.1.3"},
                        ) as topology:
                            self.assertEqual(scr_vip.enabled(self.pc), expected)
                            self.assertEqual(
                                scr_vip.internal_ip(self.pc, "10.0.0.3"),
                                "22.0.1.3" if expected else "10.0.0.3",
                            )
                            if expected:
                                topology.assert_called_once_with(self.pc, wait=True)
                            else:
                                topology.assert_not_called()

    def test_missing_or_invalid_platform_map_never_falls_back(self):
        for content, error in ((None, FileNotFoundError), ("{}", ValueError)):
            with self.subTest(content=content):
                if content is None:
                    self.path.unlink()
                else:
                    self.path.write_text(content)
                with patch.object(
                    scr_vip.time, "monotonic", side_effect=[0, 121]
                ), patch.object(scr_vip, "validate_device") as device:
                    with self.assertRaises(error):
                        scr_vip.internal_ip(self.pc, "10.0.0.3")
                    device.assert_not_called()

    def test_unready_network_never_falls_back(self):
        with patch.object(
            scr_vip.time, "monotonic", side_effect=[0, 121]
        ), patch.object(
            scr_vip, "validate_device", side_effect=RuntimeError("network not ready")
        ):
            with self.assertRaisesRegex(RuntimeError, "network not ready"):
                scr_vip.internal_ip(self.pc, "10.0.0.3")

    def test_platform_network_contract(self):
        addresses = json.dumps([{"addr_info": [{"local": "22.0.1.3"}]}])
        with patch.object(
            scr_vip.subprocess, "check_output", return_value=addresses
        ) as command, patch.object(
            scr_vip.Path, "is_socket", autospec=True, return_value=True
        ) as ready, patch.object(
            scr_vip.socket, "socket"
        ) as socket:
            scr_vip.validate_device("22.0.1.3")
            command.assert_called_once_with(
                ["ip", "-j", "-4", "address", "show", "dev", "scr_vxlan0"],
                text=True,
                timeout=5,
            )
            ready.assert_called_once_with(Path("/scr-share/snm/daemon.sock"))
            socket.return_value.__enter__.return_value.bind.assert_called_once_with(
                ("22.0.1.3", 0)
            )

    def test_missing_interface_address_fails(self):
        with patch.object(scr_vip.subprocess, "check_output", return_value="[]"):
            with self.assertRaisesRegex(RuntimeError, "does not own"):
                scr_vip.validate_device("22.0.1.3")

    def test_interface_alone_does_not_prove_network_ready(self):
        addresses = json.dumps([{"addr_info": [{"local": "22.0.1.3"}]}])
        with patch.object(
            scr_vip.subprocess, "check_output", return_value=addresses
        ), patch.object(scr_vip.Path, "is_socket", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "readiness socket"):
                scr_vip.validate_device("22.0.1.3")

    def test_non_uniform_world_fails(self):
        with self.assertRaises(ValueError):
            scr_vip.read_topology(5, 2)


if __name__ == "__main__":
    unittest.main()
