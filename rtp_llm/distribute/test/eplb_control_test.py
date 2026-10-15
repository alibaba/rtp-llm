import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from rtp_llm.distribute.distributed_server import (
    DistributedServer,
    get_eplb_stage_root_addresses,
)
from rtp_llm.distribute.worker_info import WorkerInfo
from rtp_llm.utils.grpc_client_wrapper import GrpcClientWrapper


class EplbStageRootTest(unittest.TestCase):
    def config(self, pp_size):
        return SimpleNamespace(
            parallelism_config=SimpleNamespace(
                pp_size=pp_size, dp_size=2, tp_size=2, local_world_size=8
            ),
            server_config=SimpleNamespace(start_port=8000),
            distribute_config=SimpleNamespace(),
        )

    def member(self, rank):
        return WorkerInfo(
            ip="127.0.0.1", local_rank=rank, world_rank=rank, name="",
            server_port=8000, worker_info_port_num=10,
        )

    def test_local_roots_are_ordered_and_selected_once(self):
        for pp_size, expected in ((1, ["127.0.0.1:8001"]),
                                  (2, ["127.0.0.1:8001", "127.0.0.1:8041"])):
            with self.subTest(pp_size=pp_size), patch(
                "rtp_llm.distribute.distributed_server.get_world_info",
                return_value=SimpleNamespace(
                    members=[self.member(rank) for rank in reversed(range(pp_size * 4))]
                ),
            ), patch("rtp_llm.distribute.distributed_server.TCPStore") as store:
                self.assertEqual(get_eplb_stage_root_addresses(self.config(pp_size)), expected)
                store.assert_not_called()

    def test_missing_root_uses_registered_port_without_another_rank_offset(self):
        # Root rank 4 has a nonzero local rank; its registered port already includes it.
        with patch(
            "rtp_llm.distribute.distributed_server.get_world_info",
            return_value=SimpleNamespace(members=[self.member(0), self.member(2)]),
        ), patch(
            "rtp_llm.distribute.distributed_server.get_master",
            return_value=("127.0.0.1", 8000),
        ), patch("rtp_llm.distribute.distributed_server.TCPStore") as store:
            store.return_value.get.return_value = b"198.51.100.2:8040"
            self.assertEqual(
                get_eplb_stage_root_addresses(self.config(2)),
                ["127.0.0.1:8001", "198.51.100.2:8041"],
            )
            store.return_value.get.assert_called_once_with(
                DistributedServer.REGISTRY_RANK_ADDRESS_KEY + "4"
            )
            self.assertEqual(store.call_args.kwargs["port"], 7999)
            self.assertFalse(store.call_args.kwargs["is_master"])


class EplbConfigUpdateTest(unittest.IsolatedAsyncioTestCase):
    def client(self, addresses=None):
        resolver = Mock(return_value=addresses) if addresses is not None else None
        client = GrpcClientWrapper(
            8001, dp_addresses=["unused-dp:9001"], eplb_address_resolver=resolver
        )
        for address in addresses if addresses is not None else [client.address]:
            client._dp_channels[address] = object()
            client._dp_stubs[address] = SimpleNamespace(UpdateEplbConfig=AsyncMock())
        return client

    async def test_each_stage_receives_the_same_update(self):
        addresses = ["stage0:8001", "stage1:8001"]
        client = self.client(addresses)
        with patch("rtp_llm.utils.grpc_client_wrapper.time.time", return_value=123):
            result = await client.update_eplb_config({"mode": "STATS"})
        self.assertEqual(result, {"status": "ok"})
        client._eplb_address_resolver.assert_called_once_with()
        for address in addresses:
            rpc = client._dp_stubs[address].UpdateEplbConfig
            rpc.assert_awaited_once()
            self.assertEqual(rpc.await_args.args[0].mode, "STATS")
            self.assertEqual(rpc.await_args.args[0].update_time, 123)

    async def test_one_failed_stage_is_reported_after_other_stages_are_called(self):
        client = self.client(["stage0:8001", "stage1:8001"])
        client._dp_stubs["stage1:8001"].UpdateEplbConfig.side_effect = RuntimeError("unavailable")
        result = await client.update_eplb_config({"mode": "STATS"})
        self.assertIn("stage1:8001", result["error"])
        self.assertIn("unavailable", result["error"])
        for stub in client._dp_stubs.values():
            stub.UpdateEplbConfig.assert_awaited_once()

    async def test_pp1_and_legacy_local_client_send_one_request(self):
        for addresses in (["stage0:8001"], None):
            with self.subTest(addresses=addresses):
                client = self.client(addresses)
                self.assertEqual(
                    await client.update_eplb_config('{"mode": "NONE"}'), {"status": "ok"}
                )
                stub, = client._dp_stubs.values()
                stub.UpdateEplbConfig.assert_awaited_once()
                self.assertEqual(stub.UpdateEplbConfig.await_args.args[0].mode, "NONE")


if __name__ == "__main__":
    unittest.main()
