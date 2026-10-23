"""Real gRPC transport failures and all-rank agreement; no native-model coverage."""

import asyncio
import unittest

import grpc

from rtp_llm.cpp.model_rpc.proto import model_rpc_service_pb2 as pb
from rtp_llm.cpp.model_rpc.proto import model_rpc_service_pb2_grpc as rpc
from rtp_llm.RL.replica_rpc_client import ReplicaControlError, ReplicaRpcClient


class Service(rpc.RpcServiceServicer):
    def __init__(self, rank):
        self.reply = pb.RlControlResponsePB(
            protocol_version=2,
            kv_cache_ready=True,
            kv_cache_bytes=1024,
            phase="paused",
            model_version="run:0",
            tp_rank=rank,
            tp_size=2,
            quiescent=True,
        )
        self.gate = None
        self.entered = asyncio.Event()
        self.requests = []

    async def RlControl(self, request, context):
        self.requests.append(request)
        self.entered.set()
        if self.gate is not None:
            await self.gate.wait()
        return self.reply


class ReplicaClientTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.services = [Service(rank) for rank in range(2)]
        self.servers = []
        addresses = []
        for service in self.services:
            server = grpc.aio.server()
            rpc.add_RpcServiceServicer_to_server(service, server)
            port = server.add_insecure_port("127.0.0.1:0")
            await server.start()
            self.servers.append(server)
            addresses.append(f"127.0.0.1:{port}")
        self.client = ReplicaRpcClient(addresses, timeout=0.1)
        await self.client.status()

    async def asyncTearDown(self):
        await self.client.close()
        await asyncio.gather(*(server.stop(0) for server in self.servers))

    async def assert_poisoned(self):
        before = [len(service.requests) for service in self.services]
        with self.assertRaisesRegex(ReplicaControlError, "restart all ranks"):
            await self.client.status()
        self.assertEqual(before, [len(service.requests) for service in self.services])

    async def test_pause_waits_for_every_rank_and_forwards_expected_version(self):
        slow = self.services[1]
        slow.gate = asyncio.Event()
        slow.entered.clear()
        pending = asyncio.create_task(self.client.pause())
        await asyncio.wait_for(slow.entered.wait(), 1)
        self.assertFalse(pending.done())
        slow.gate.set()
        replies = await pending
        self.assertEqual([reply.tp_rank for reply in replies], [0, 1])
        for service in self.services:
            self.assertEqual(service.requests[-1].expected_version, "run:0")
            self.assertEqual(
                service.requests[-1].operation, pb.RlControlRequestPB.PAUSE
            )

    async def test_rank_mapping_mismatch_is_fatal(self):
        self.services[1].reply.tp_rank = 0
        with self.assertRaisesRegex(ReplicaControlError, "rank mapping"):
            await self.client.pause()
        await self.assert_poisoned()

    async def test_pause_without_quiescence_is_fatal(self):
        self.services[1].reply.quiescent = False
        with self.assertRaisesRegex(ReplicaControlError, "quiescence"):
            await self.client.pause()
        await self.assert_poisoned()

    async def test_partial_commit_never_publishes_a_replica_version(self):
        self.services[1].reply.model_version = "run:1"
        with self.assertRaisesRegex(ReplicaControlError, "disagree on model_version"):
            await self.client.commit_update()
        self.assertEqual(self.client.version, "run:0")
        await self.assert_poisoned()

    async def test_weight_count_mismatch_is_fatal(self):
        self.services[1].reply.received_weights = 1
        with self.assertRaisesRegex(
            ReplicaControlError, "disagree on received_weights"
        ):
            await self.client.pause()
        await self.assert_poisoned()

    async def test_transport_deadline_does_not_allow_retry(self):
        self.services[1].gate = asyncio.Event()
        with self.assertRaisesRegex(ReplicaControlError, "control failed"):
            await asyncio.wait_for(self.client.pause(), timeout=7)
        await self.assert_poisoned()

    async def test_partial_release_is_fatal(self):
        self.services[0].reply.kv_cache_ready = False
        self.services[0].reply.kv_cache_bytes = 0
        with self.assertRaisesRegex(ReplicaControlError, "KV residency"):
            await self.client.release_cache()
        await self.assert_poisoned()

    async def test_restore_without_backing_is_fatal(self):
        self.services[1].reply.kv_cache_bytes = 0
        with self.assertRaisesRegex(ReplicaControlError, "KV capacity"):
            await self.client.restore_cache()
        await self.assert_poisoned()

    async def test_restore_waits_for_all_rank_acknowledgements(self):
        slow = self.services[1]
        slow.gate = asyncio.Event()
        slow.entered.clear()
        pending = asyncio.create_task(self.client.restore_cache())
        await asyncio.wait_for(slow.entered.wait(), 1)
        self.assertFalse(pending.done())
        slow.gate.set()
        await pending
        for service in self.services:
            self.assertEqual(
                service.requests[-1].operation, pb.RlControlRequestPB.RESTORE_KV_CACHE
            )
            self.assertEqual(service.requests[-1].expected_version, "run:0")

    async def test_canceling_control_requires_replica_restart(self):
        slow = self.services[1]
        slow.gate = asyncio.Event()
        slow.entered.clear()
        pending = asyncio.create_task(self.client.pause())
        await asyncio.wait_for(slow.entered.wait(), 1)
        pending.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await pending
        await self.assert_poisoned()


if __name__ == "__main__":
    unittest.main()
