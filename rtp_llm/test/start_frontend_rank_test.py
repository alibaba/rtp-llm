import os
import unittest
from unittest.mock import Mock, patch

from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.start_server import start_dash_sc_server_impl, start_frontend_server_impl


class StartFrontendRankTest(unittest.TestCase):
    def config(self):
        config = PyEnvConfigs()
        pc = config.parallelism_config
        pc.pp_size = pc.dp_size = pc.tp_size = 2
        pc.ep_size = 0
        pc.world_size = 8
        pc.local_world_size = 4
        pc.world_rank = 4
        config.server_config.start_port = 8000
        config.server_config.worker_info_port_num = 10
        config.server_config.frontend_server_count = 1
        config.distribute_config.remote_server_port = 9000
        config.distribute_config.worker_info_port_num = 10
        return config

    def test_launchers_pass_global_rank_and_children_keep_local_ports(self):
        entry_points = (
            (
                start_frontend_server_impl,
                "rtp_llm.start_frontend_server",
                "rtp_llm.frontend.frontend_app.FrontendApp",
            ),
            (
                start_dash_sc_server_impl,
                "rtp_llm.start_dash_sc_server",
                "rtp_llm.dash_sc.DashScApp",
            ),
        )
        for launch, module, app_class in entry_points:
            parent_config = self.config()
            controller = Mock()
            with (
                self.subTest(entry=module),
                patch.dict(os.environ, {"LOCAL_WORLD_SIZE": "4"}),
                patch("rtp_llm.start_server.multiprocessing.Process") as process,
                patch("rtp_llm.start_server.multiprocessing.Barrier"),
                patch(
                    "rtp_llm.start_server.multiprocessing.Pipe",
                    return_value=(Mock(), Mock()),
                ),
                patch(f"{module}._install_hot_hook_runtime"),
                patch(f"{module}.set_global_controller"),
                patch(f"{module}.setproctitle"),
                patch(app_class),
            ):
                launch(controller, parent_config)
                calls = process.call_args_list
                self.assertEqual([call.kwargs["args"][0] for call in calls], [4, 6])
                for call, local_rank in zip(calls, (0, 2)):
                    # Each spawned child receives its own config copy.
                    args = list(call.kwargs["args"])
                    child_config = self.config()
                    args[3] = child_config
                    call.kwargs["target"](*args)

                    pc = child_config.parallelism_config
                    self.assertEqual(pc.world_rank, 4 + local_rank)
                    self.assertEqual(pc.local_rank, local_rank)
                    self.assertEqual(pc.pp_rank, 1)
                    self.assertEqual(pc.dp_rank, local_rank // 2)
                    self.assertEqual(pc.tp_rank, 0)
                    self.assertEqual(
                        child_config.server_config.rpc_server_port,
                        8001 + local_rank * 10,
                    )
                    self.assertEqual(
                        child_config.distribute_config.remote_rpc_server_port,
                        9001 + local_rank * 10,
                    )


if __name__ == "__main__":
    unittest.main()
