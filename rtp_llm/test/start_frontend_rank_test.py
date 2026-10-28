import os
import unittest
from itertools import product
from unittest.mock import Mock, patch

from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.config.server_config_setup import set_parallelism_config
from rtp_llm.start_server import (
    _iter_serving_world_ranks,
    start_dash_sc_server_impl,
    start_frontend_server_impl,
)


class StartFrontendRankTest(unittest.TestCase):

    def setUp(self):
        hostname_patch = patch(
            "rtp_llm.distribute.distributed_server.socket.gethostbyname",
            return_value="127.0.0.1",
        )
        hostname_patch.start()
        self.addCleanup(hostname_patch.stop)

    def config(self, pp=2, dp=2, tp=2, local_world_size=4, world_rank=4):
        config = PyEnvConfigs()
        pc = config.parallelism_config
        pc.pp_size, pc.dp_size, pc.tp_size = pp, dp, tp
        pc.ep_size = 0
        pc.world_size = pp * dp * tp
        pc.local_world_size = local_world_size
        set_parallelism_config(pc, world_rank)
        config.server_config.ip = "127.0.0.1"
        config.server_config.start_port = 8000
        config.server_config.worker_info_port_num = 10
        config.server_config.frontend_server_count = 1
        config.distribute_config.remote_server_port = 9000
        config.distribute_config.worker_info_port_num = 10
        return config

    def test_serving_world_ranks_keep_local_filter(self):
        cases = (
            ((2, 2, 2, 4, 0), [0, 2]),
            ((2, 2, 2, 4, 4), [4, 6]),
            # The current identity need not be the first member of the node.
            ((2, 2, 2, 4, 6), [4, 6]),
            ((1, 4, 2, 4, 6), [4, 6]),
            # Keep the local heartbeat entry even when its TP rank is nonzero.
            ((2, 1, 4, 2, 6), [6]),
            # A node can contain workers from two PP stages.
            ((2, 3, 2, 4, 6), [4, 6]),
            ((1, 1, 1, 1, 0), [0]),
        )
        for params, expected in cases:
            config = self.config(*params)
            with (
                self.subTest(params=params),
                patch.dict(os.environ, {"LOCAL_WORLD_SIZE": str(params[3])}),
            ):
                self.assertEqual(list(_iter_serving_world_ranks(config)), expected)
                self.assertEqual(config.parallelism_config.world_rank, params[4])

    def test_serving_world_ranks_use_configured_local_world_size(self):
        # Use the resolved config even when the environment is absent or differs.
        for env_size, world_rank in product((None, "1", "4", "8"), (0, 6)):
            config = self.config(pp=1, dp=8, tp=1, world_rank=world_rank)
            expected = [0, 1, 2, 3] if world_rank == 0 else [4, 5, 6, 7]
            with (
                self.subTest(env_size=env_size, world_rank=world_rank),
                patch.dict(os.environ),
            ):
                os.environ.pop("LOCAL_WORLD_SIZE", None)
                if env_size is not None:
                    os.environ["LOCAL_WORLD_SIZE"] = env_size
                self.assertEqual(list(_iter_serving_world_ranks(config)), expected)
                self.assertEqual(config.parallelism_config.local_world_size, 4)
                self.assertEqual(config.parallelism_config.world_rank, world_rank)

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
        for (launch, module, app_class), (pp, dp), parent_rank in product(
            entry_points, ((2, 2), (1, 4)), (4, 6)
        ):
            parent_config = self.config(pp=pp, dp=dp, world_rank=parent_rank)
            parent_config.server_config.frontend_server_count = 2
            controller = Mock()
            with (
                self.subTest(entry=module, pp=pp, parent_rank=parent_rank),
                patch.dict(os.environ, {"LOCAL_WORLD_SIZE": "4"}),
                patch("rtp_llm.start_server.multiprocessing.Process") as process,
                patch("rtp_llm.start_server.multiprocessing.Barrier") as barrier,
                patch(
                    "rtp_llm.start_server.multiprocessing.Pipe",
                    return_value=(Mock(), Mock()),
                ),
                patch(f"{module}._install_hot_hook_runtime") as install_hook,
                patch(f"{module}.set_global_controller"),
                patch(f"{module}.setproctitle"),
                patch(app_class) as app,
            ):
                launch(controller, parent_config)
                calls = process.call_args_list
                expected_specs = [(4, 0), (4, 1), (6, 0), (6, 1)]
                self.assertEqual(
                    [call.kwargs["args"][:2] for call in calls], expected_specs
                )
                is_dash_sc = launch is start_dash_sc_server_impl
                if is_dash_sc:
                    barrier.assert_called_once_with(4)
                for call, (world_rank, server_id) in zip(calls, expected_specs):
                    local_rank = {4: 0, 6: 2}[world_rank]
                    # Each spawned child receives its own config copy.
                    args = list(call.kwargs["args"])
                    child_config = self.config(pp=pp, dp=dp, world_rank=parent_rank)
                    args[3] = child_config
                    call.kwargs["target"](*args)

                    pc = child_config.parallelism_config
                    self.assertEqual(pc.world_rank, world_rank)
                    self.assertEqual(pc.local_rank, local_rank)
                    self.assertEqual(pc.pp_rank, 1 if pp == 2 else 0)
                    self.assertEqual(
                        pc.dp_rank, local_rank // 2 if pp == 2 else world_rank // 2
                    )
                    self.assertEqual(pc.tp_rank, 0)
                    self.assertEqual(
                        child_config.server_config.frontend_server_id, server_id
                    )
                    self.assertEqual(
                        child_config.server_config.rpc_server_port,
                        8001 + local_rank * 10,
                    )
                    self.assertEqual(
                        child_config.distribute_config.remote_rpc_server_port,
                        9001 + local_rank * 10,
                    )
                    role = "dash_sc" if is_dash_sc else "frontend"
                    install_hook.assert_called_with(
                        f"{role}_rank_{local_rank}_server_{server_id}"
                    )
                    if is_dash_sc:
                        self.assertIs(args[5], barrier.return_value)
                        app.return_value.start.assert_called_with(
                            ready_pipe_writer=args[4], bind_barrier=args[5]
                        )
                self.assertEqual(
                    parent_config.parallelism_config.world_rank, parent_rank
                )
                self.assertEqual(
                    parent_config.parallelism_config.local_rank,
                    {4: 0, 6: 2}[parent_rank],
                )


if __name__ == "__main__":
    unittest.main()
