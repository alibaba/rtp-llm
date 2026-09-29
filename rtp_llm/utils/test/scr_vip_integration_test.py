import os
import unittest
from unittest.mock import patch

from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.distribute.distributed_server import (
    DistributedServer,
    get_local_world_info,
)
from rtp_llm.utils import scr_vip
from rtp_llm.utils.scr_restore_context import RestoreContext


class ScrVipIntegrationTest(unittest.TestCase):
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
                        "RTP_LLM_SCR_VIP_INTERFACE": "scr_vxlan0",
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
                        ["22.0.1.2"] * 2 + ["22.0.1.3"] * 2,
                    )
                    restored = RestoreContext("seed", "10.1.0.3").resolve_world_info(
                        world, pc
                    )
                    self.assertEqual(restored.self.ip, world.self.ip)
                    self.assertEqual(restored.self.server_port, world.self.server_port)
                    world.members[0].ip = "21.0.0.1"
                    with self.assertRaisesRegex(RuntimeError, "cannot change"):
                        RestoreContext("seed", "10.1.0.3").resolve_world_info(world, pc)


if __name__ == "__main__":
    unittest.main()
