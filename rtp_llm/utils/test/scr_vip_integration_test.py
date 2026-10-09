import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.distribute.distributed_server import (
    DistributedServer,
    get_local_world_info,
    get_master,
)
from rtp_llm.utils import scr_vip
from rtp_llm.utils.scr_restore_context import RestoreContext


class ScrVipIntegrationTest(unittest.TestCase):
    def test_custom_annotation_path_used_by_scr_and_original_c2_discovery(self):
        configs = PyEnvConfigs()
        pc = configs.parallelism_config
        pc.world_size, pc.local_world_size = 4, 2
        pc.world_rank, pc.local_rank = 2, 0
        rows = {
            "model-rank-0": {"ip": "22.0.1.2", "real_ip": "10.0.0.2", "port": 1234},
            "model-rank-1": {"ip": "22.0.1.3", "real_ip": "10.0.0.3"},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "custom-annotations"
            configs.distribute_config.gang_annocation_path = str(path)
            path.write_text(
                'unrelated="value"\napp.c2.io/biz-detail-ganginfo='
                + json.dumps(json.dumps(rows))
                + "\n"
            )
            with patch.dict(
                os.environ,
                {
                    "RTPLLM_ENABLE_SCR": "1",
                    "SCR_PHASE": "checkpoint",
                    "RANK_ID": "1",
                    "RANK_SIZE": "2",
                },
                clear=True,
            ), patch.object(scr_vip, "validate_device"):
                self.assertEqual(
                    get_master(configs.distribute_config, pc), ("22.0.1.2", "")
                )
                world = get_local_world_info(
                    configs.server_config, configs.distribute_config, pc
                )
                self.assertEqual(world.self.ip, "22.0.1.3")
            # The original C2 path still preserves the annotation's port and
            # does not impose SCR rank/real_ip/network requirements.
            path.write_text(
                "app.c2.io/biz-detail-ganginfo="
                + json.dumps(
                    json.dumps(
                        {
                            "model_part0": {"ip": "10.0.0.2", "port": 1234},
                        }
                    )
                )
                + "\n"
            )
            with patch.dict(
                os.environ, {"RTPLLM_ENABLE_SCR": "0"}, clear=True
            ), patch.object(scr_vip, "validate_device") as device:
                self.assertEqual(
                    get_master(configs.distribute_config, pc), ("10.0.0.2", "1234")
                )
                device.assert_not_called()
                # Explicit leader configuration must retain priority over C2.
                configs.distribute_config.leader_address = "10.0.0.9"
                path.unlink()
                self.assertEqual(
                    get_master(configs.distribute_config, pc), ("10.0.0.9", "")
                )

    def test_serving_children_keep_their_node_world_rank(self):
        from rtp_llm import start_dash_sc_server, start_frontend_server

        for launcher, module_name, class_name in (
            (start_frontend_server, "rtp_llm.frontend.frontend_app", "FrontendApp"),
            (start_dash_sc_server, "rtp_llm.dash_sc", "DashScApp"),
        ):
            for node_rank in (0, 2):
                for local_rank in (0, 1):
                    with self.subTest(
                        launcher=launcher.__name__,
                        node_rank=node_rank,
                        local_rank=local_rank,
                    ):
                        configs = PyEnvConfigs()
                        pc = configs.parallelism_config
                        pc.world_size, pc.local_world_size = 4, 2
                        pc.tp_size, pc.dp_size, pc.ep_size = 2, 2, 4
                        pc.world_rank = node_rank
                        app_type = MagicMock()
                        module = ModuleType(module_name)
                        setattr(module, class_name, app_type)
                        with patch.dict(
                            sys.modules, {module_name: module}
                        ), patch.object(
                            launcher, "_install_hot_hook_runtime"
                        ), patch.object(
                            launcher, "set_global_controller"
                        ), patch.object(
                            launcher, "setproctitle"
                        ):
                            entry = getattr(launcher, launcher.__name__.split(".")[-1])
                            entry(local_rank, 0, None, configs)
                        self.assertEqual(pc.world_rank, node_rank + local_rank)
                        self.assertEqual(pc.local_rank, local_rank)
                        self.assertEqual(pc.dp_rank, node_rank // 2)
                        app_type.return_value.start.assert_called_once()

    def test_vip_is_used_for_store_registration_frontend_and_restore(self):
        for rank in range(4):
            with self.subTest(rank=rank):
                configs = PyEnvConfigs()
                pc = configs.parallelism_config
                pc.world_size, pc.local_world_size = 4, 2
                pc.world_rank, pc.local_rank = rank, rank % 2
                configs.server_config.ip = "10.0.0.3"
                configs.server_config.start_port = 22290
                configs.server_config.worker_info_port_num = 10
                with patch.dict(
                    os.environ,
                    {
                        "RTPLLM_ENABLE_SCR": "1",
                        "SCR_PHASE": "checkpoint",
                        "RANK_ID": str(rank // 2),
                    },
                    clear=True,
                ), patch.object(
                    scr_vip, "topology", return_value={0: "22.0.1.2", 1: "22.0.1.3"}
                ), patch(
                    "rtp_llm.distribute.distributed_server.TCPStore"
                ) as store:
                    server = DistributedServer(configs, wait_for_workers=False)
                    server.regist()
                    self.assertEqual(store.call_args.kwargs["host_name"], "22.0.1.2")
                    self.assertEqual(store.call_args.kwargs["port"], 22289)
                    self.assertEqual(server.get_nccl_init_port(), 22279)
                    store.return_value.set.assert_called_with(
                        f"registry_rank_address_{rank}",
                        f"22.0.1.{2 + rank // 2}:{22290 + rank % 2 * 10}",
                    )
                    world = get_local_world_info(
                        configs.server_config, configs.distribute_config, pc
                    )
                    self.assertEqual(
                        [m.ip for m in world.members],
                        [f"22.0.1.{2 + rank // 2}"] * 2,
                    )
                    restored = RestoreContext("seed", "10.1.0.3").resolve_world_info(
                        world, pc, configs.distribute_config.gang_annocation_path
                    )
                    self.assertEqual(restored.self.ip, world.self.ip)
                    self.assertEqual(restored.self.server_port, world.self.server_port)
                    world.members[0].ip = "21.0.0.1"
                    with self.assertRaisesRegex(RuntimeError, "cannot change"):
                        RestoreContext("seed", "10.1.0.3").resolve_world_info(
                            world, pc, configs.distribute_config.gang_annocation_path
                        )

    def test_cpp_broadcast_converts_group_root_for_nonzero_tp_and_dp_groups(self):
        from rtp_llm.models_py.distributed import collective_torch as ct

        pc = PyEnvConfigs().parallelism_config
        pc.world_size, pc.tp_size, pc.dp_size = 4, 2, 2
        for key, mode, rank, members in (("TP1", 0, 2, [2, 3]), ("DP1", 1, 3, [1, 3])):
            for root in (0, 1):
                with self.subTest(group=key, root=root):
                    group = MagicMock()
                    group.size.return_value = 2
                    native = ModuleType("librtp_compute_ops")
                    native.register_comm_ops = MagicMock()
                    tensor = MagicMock(is_cuda=True)
                    with patch.dict(
                        sys.modules, {"librtp_compute_ops": native}
                    ), patch.object(ct, "_group_map", {key: group}), patch.object(
                        ct, "_parallelism_config", pc
                    ), patch.object(
                        ct.torch.distributed, "get_rank", return_value=rank
                    ), patch.object(
                        ct.torch.distributed,
                        "get_global_rank",
                        side_effect=lambda pg, r: members[r],
                    ) as to_global, patch.object(
                        ct.torch.distributed, "broadcast"
                    ) as broadcast, patch.object(
                        ct.torch.cuda, "current_device", return_value=0
                    ):
                        ct._register_process_groups_to_cpp()
                        callback = native.register_comm_ops.call_args.args[0]
                        callback([tensor], root, mode)
                        to_global.assert_called_once_with(group, root)
                        broadcast.assert_called_once_with(
                            tensor, members[root], group=group
                        )


if __name__ == "__main__":
    unittest.main()
