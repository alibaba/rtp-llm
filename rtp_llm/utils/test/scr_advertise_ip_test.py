import os
import socket
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

from rtp_llm.utils.scr_local_comm import cache_store_advertise_ip, current_pod_ip


class ScrAdvertiseIpTest(unittest.TestCase):
    def setUp(self):
        self.pc = NS(world_size=2, local_world_size=2)
        self.world = NS(
            num_nodes=1,
            self=NS(ip="192.0.2.1"),
            members=[NS(ip="127.0.0.1", world_rank=i, local_rank=i) for i in range(2)],
        )
        self.env = patch.dict(os.environ, {"RTP_LLM_SCR_LOCAL_COMM": "1"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_resolves_each_restore_without_using_seed_identity(self):
        with patch("socket.gethostname", return_value="restored-pod"), patch(
            "socket.gethostbyname", side_effect=["192.0.2.20", "192.0.2.21"]
        ) as resolve:
            self.assertEqual(cache_store_advertise_ip(self.world, self.pc), "192.0.2.20")
            self.assertEqual(cache_store_advertise_ip(self.world, self.pc), "192.0.2.21")
        self.assertEqual(resolve.call_count, 2)
        self.assertEqual(self.world.self.ip, "192.0.2.1")
        self.assertEqual([m.ip for m in self.world.members], ["127.0.0.1"] * 2)

    def test_disabled_mode_does_not_require_local_topology_or_resolve(self):
        with patch.dict(os.environ, {"RTP_LLM_SCR_LOCAL_COMM": "0"}), patch(
            "socket.gethostbyname"
        ) as resolve:
            self.assertIsNone(cache_store_advertise_ip(None, None))
        resolve.assert_not_called()

    def test_explicit_manifest_host_preserved(self):
        for member in self.world.members:
            member.ip = "192.0.2.50"
        with patch("socket.gethostbyname") as resolve:
            self.assertIsNone(cache_store_advertise_ip(self.world, self.pc))
        resolve.assert_not_called()

    def test_invalid_topology_rejected_before_resolution(self):
        self.world.members.pop()
        with patch("socket.gethostbyname") as resolve, self.assertRaises(ValueError):
            cache_store_advertise_ip(self.world, self.pc)
        resolve.assert_not_called()

    def test_unusable_addresses_fail_closed(self):
        for address in ["127.0.0.1", "127.0.0.2", "0.0.0.0", "224.0.0.1", "::1", "bad"]:
            with self.subTest(address=address), patch(
                "socket.gethostbyname", return_value=address
            ), self.assertRaises(ValueError):
                current_pod_ip()

    def test_resolution_failure_does_not_fall_back_to_seed(self):
        with patch("socket.gethostbyname", side_effect=socket.gaierror("no address")):
            with self.assertRaises(socket.gaierror):
                cache_store_advertise_ip(self.world, self.pc)


if __name__ == "__main__":
    unittest.main()
