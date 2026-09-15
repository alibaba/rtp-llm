import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from rtp_llm.start_backend_server import _shutdown_order_indices, multi_rank_start


class StartBackendServerShutdownTest(unittest.TestCase):
    def test_single_rank_needs_no_ready_ack(self):
        self.assertEqual(_shutdown_order_indices(1, 0, 1), ([], [0]))

    def test_tp_followers_preserve_per_group_leaders(self):
        self.assertEqual(
            _shutdown_order_indices(4, 0, 2),
            ([1, 3], [0, 2]),
        )

    def test_tp1_dp_ep_followers_stop_before_local_leader(self):
        self.assertEqual(
            _shutdown_order_indices(2, 0, 1),
            ([1], [0]),
        )

    def test_nonzero_local_world_rank_preserves_tp_group_order(self):
        self.assertEqual(
            _shutdown_order_indices(2, 2, 2),
            ([1], [0]),
        )

    def test_installs_signal_handler_before_waiting_for_rank_startup(self):
        processes = [Mock(name="rank_0"), Mock(name="rank_1")]
        shutdown_ready_events = [Mock(name="rank_0_ready"), Mock(name="rank_1_ready")]
        manager = Mock()
        manager.monitor_and_release_processes.return_value = True
        call_order = []

        def create_manager(**kwargs):
            call_order.append("manager")
            return manager

        def set_processes(*args, **kwargs):
            call_order.append("set_processes")

        def wait_for_ranks(*args, **kwargs):
            call_order.append("wait_for_ranks")

        manager.set_processes.side_effect = set_processes
        config = SimpleNamespace(
            distribute_config=SimpleNamespace(fake_gang_env=False),
            parallelism_config=SimpleNamespace(world_rank=0, tp_size=2),
            server_config=SimpleNamespace(shutdown_timeout=50, monitor_interval=1),
        )

        with patch(
            "rtp_llm.start_backend_server._create_rank_processes",
            return_value=(processes, [Mock(), Mock()], shutdown_ready_events),
        ), patch(
            "rtp_llm.start_backend_server.ProcessManager",
            side_effect=create_manager,
        ), patch(
            "rtp_llm.start_backend_server._wait_for_ranks_startup",
            side_effect=wait_for_ranks,
        ):
            self.assertEqual(multi_rank_start(Mock(), config), processes)

        self.assertEqual(
            call_order[:3], ["manager", "set_processes", "wait_for_ranks"]
        )
        manager.monitor_and_release_processes.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
