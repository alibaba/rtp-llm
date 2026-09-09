"""Exercise the real config writer and restore hook with separate PD/control IPs."""
import os
import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from rtp_llm.config.engine_config import update_worker_addrs
from rtp_llm.utils.scr_template_utils import _BackendVisitorTemplateHook


class ScrPdAdvertisementTest(unittest.TestCase):
    def setUp(self):
        self.pc = NS(world_size=2, local_world_size=2, tp_size=2, dp_size=1, dp_rank=0, local_rank=0)
        self.world = NS(
            num_nodes=1,
            members=[NS(ip="127.0.0.1", world_rank=i, local_rank=i,
                        cache_store_listen_port=18632 + i * 10,
                        cache_store_rdma_listen_port=18634 + i * 10,
                        rpc_server_port=18631 + i * 10) for i in range(2)],
        )
        self.env = patch.dict(os.environ, {"RTP_LLM_SCR_LOCAL_COMM": "1", "SCR_PHASE": "restore"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_prefill_preserves_control_loopback_and_advertises_current_pod(self):
        runtime = NS()
        with patch("socket.gethostbyname", side_effect=["192.0.2.20", "192.0.2.21"]):
            for ip in ["192.0.2.20", "192.0.2.21"]:
                update_worker_addrs(runtime, self.pc, self.world)
                self.assertEqual(runtime.worker_addrs, [ip + ":18632:18634", ip + ":18642:18644"])
                self.assertEqual(runtime.worker_grpc_addrs, ["127.0.0.1:18631", "127.0.0.1:18641"])

    def test_decode_dp_group_filter_is_preserved(self):
        self.pc.tp_size, self.pc.dp_size, self.pc.dp_rank = 1, 2, 1
        runtime = NS()
        with patch("socket.gethostbyname", return_value="192.0.2.30"):
            update_worker_addrs(runtime, self.pc, self.world)
        self.assertEqual(runtime.worker_addrs, ["192.0.2.30:18642:18644"])
        self.assertEqual(runtime.worker_grpc_addrs, ["127.0.0.1:18641"])

    def test_frontend_identity_refreshes_without_publishing_loopback(self):
        configs = NS(server_config=object(), distribute_config=object(),
                     parallelism_config=self.pc, role_config=NS(role_type="PREFILL"))
        visitor = Mock(source_ip="192.0.2.1")
        with patch("rtp_llm.distribute.distributed_server.get_world_info", return_value=self.world), patch(
            "rtp_llm.utils.scr_endpoint_provider.resolve_world_info", return_value=self.world
        ), patch("rtp_llm.distribute.distributed_server.get_dp_addrs_from_world_info", return_value=["127.0.0.1:18631"]), patch(
            "socket.gethostbyname", return_value="192.0.2.20"
        ):
            _BackendVisitorTemplateHook(visitor, configs).restore_fixup("generation-2")
        self.assertEqual(visitor.source_ip, "192.0.2.20")
        visitor.update_addresses.assert_called_once_with(["127.0.0.1:18631"])


if __name__ == "__main__":
    unittest.main()
