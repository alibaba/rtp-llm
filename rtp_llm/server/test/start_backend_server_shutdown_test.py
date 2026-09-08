import unittest

from rtp_llm.start_backend_server import _shutdown_order_indices


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


if __name__ == "__main__":
    unittest.main()
