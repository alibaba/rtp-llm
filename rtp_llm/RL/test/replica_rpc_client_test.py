"""Unit tests for ReplicaRpcClient cross-DP EP topology and UNIMPLEMENTED stubs."""

import asyncio
import unittest

import grpc

from rtp_llm.cpp.model_rpc.proto import model_rpc_service_pb2 as pb
from rtp_llm.cpp.model_rpc.proto import model_rpc_service_pb2_grpc as rpc
from rtp_llm.RL.replica_rpc_client import ReplicaControlError, ReplicaRpcClient


class _FakeRlService(rpc.RpcServiceServicer):
    def __init__(self, world_rank: int, tp_size: int, dp_size: int, ep_size: int):
        self.world_rank = world_rank
        self.tp_size = tp_size
        self.dp_size = dp_size
        self.ep_size = ep_size
        self.world_size = tp_size * dp_size
        self.requests = []
        self.phase = "paused"
        self.model_version = "run:0"
        self.update_id = ""
        self.global_step = 0
        self.received_weights = 0
        self.applied_weights = 0
        self.fail_next = False
        self.vmm_enabled = False
        self.kv_ready = True
        self.weights_ready = True

    async def RlControl(self, request, context):
        self.requests.append(request)
        if (not self.vmm_enabled and request.operation in (
            pb.RlControlRequestPB.SLEEP,
            pb.RlControlRequestPB.WAKE_UP,
        )) or request.operation == pb.RlControlRequestPB.WRITE_BUCKET:
            await context.abort(grpc.StatusCode.UNIMPLEMENTED, "reserved stub")
        if self.fail_next:
            self.fail_next = False
            await context.abort(grpc.StatusCode.INTERNAL, "injected error")
        if request.operation == pb.RlControlRequestPB.SLEEP:
            self.phase = "paused"
            self.kv_ready = self.weights_ready = False
        elif request.operation == pb.RlControlRequestPB.WAKE_UP:
            if "kv_cache" in request.memory_tags:
                self.kv_ready = True
        elif request.operation == pb.RlControlRequestPB.RESUME:
            self.phase = "running"
        elif request.operation == pb.RlControlRequestPB.BEGIN_UPDATE:
            self.phase = "updating"
            self.update_id = "run:1"
            self.received_weights = 0
            self.applied_weights = 0
        elif request.operation == pb.RlControlRequestPB.WRITE_WEIGHT:
            self.received_weights += 1
        elif request.operation == pb.RlControlRequestPB.COMMIT_UPDATE:
            self.phase = "paused"
            self.model_version = "run:1"
            self.update_id = ""
            self.global_step = 1
            self.applied_weights = self.received_weights
            self.weights_ready = True
        return pb.RlControlResponsePB(
            protocol_version=3,
            phase=self.phase,
            model_version=self.model_version,
            update_id=self.update_id,
            global_step=self.global_step,
            tp_rank=self.world_rank % self.tp_size,
            tp_size=self.tp_size,
            world_rank=self.world_rank,
            world_size=self.world_size,
            dp_rank=self.world_rank // self.tp_size,
            dp_size=self.dp_size,
            ep_rank=self.world_rank % self.ep_size,
            ep_size=self.ep_size,
            received_weights=self.received_weights,
            applied_weights=self.applied_weights,
            quiescent=self.phase != "running",
            kv_cache_ready=self.kv_ready,
            kv_cache_bytes=1024 if self.kv_ready else 0,
            weights_ready=self.weights_ready and self.phase != "updating",
        )


class ReplicaRpcClientTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Cross-DP EP topology: tp=2, dp=2, ep=4 (world_size=4).
        self.tp_size = 2
        self.dp_size = 2
        self.ep_size = 4
        self.services = [
            _FakeRlService(rank, self.tp_size, self.dp_size, self.ep_size)
            for rank in range(4)
        ]
        self.servers = []
        self.addresses = []
        self.client = None
        try:
            for svc in self.services:
                server = grpc.aio.server()
                rpc.add_RpcServiceServicer_to_server(svc, server)
                port = server.add_insecure_port("127.0.0.1:0")
                await server.start()
                self.servers.append(server)
                self.addresses.append(f"127.0.0.1:{port}")
            self.client = ReplicaRpcClient(
                self.addresses,
                timeout=2.0,
                topology={
                    "world_size": 4,
                    "tp_size": self.tp_size,
                    "dp_size": self.dp_size,
                    "ep_size": self.ep_size,
                },
            )
        except BaseException:
            await self.asyncTearDown()
            raise

    async def asyncTearDown(self):
        if self.client is not None:
            await self.client.close()
        if self.servers:
            await asyncio.gather(*(s.stop(0) for s in self.servers))

    async def test_cross_dp_ep_rank_validation_and_unimplemented_stubs(self):
        replies = await self.client.status()
        self.assertEqual(len(replies), 4)
        self.assertEqual(replies[0].protocol_version, 3)
        self.assertEqual(replies[0].ep_size, 4)

        # UNIMPLEMENTED stubs must raise ReplicaControlError(UNIMPLEMENTED) and NOT poison client.
        for coro in (
            self.client.sleep(level=2),
            self.client.wake_up(tags=["weights"]),
            self.client.write_bucket(pb.WriteBucketRequestPB(bucket_id=1)),
        ):
            with self.assertRaises(ReplicaControlError) as ctx:
                await coro
            self.assertIsInstance(ctx.exception.__cause__, grpc.RpcError)
            self.assertEqual(ctx.exception.__cause__.code(), grpc.StatusCode.UNIMPLEMENTED)

        # Client stays healthy and can run a full begin/push/commit cycle.
        started = await self.client.begin_update(
            max_buffer_bytes=1024,
            global_step=1,
            expected_source_names=["model.embed_tokens.weight"],
        )
        self.assertEqual(started[0].phase, "updating")
        self.assertEqual(
            list(self.services[0].requests[-1].expected_source_names),
            ["model.embed_tokens.weight"],
        )
        await self.client.push_weight("model.embed_tokens.weight", [b"ipc"] * 4)
        committed = await self.client.commit_update()
        self.assertEqual(committed[0].model_version, "run:1")
        self.assertEqual(committed[0].received_weights, 1)

    async def test_two_stage_wake_requires_refit_to_restore_weights(self):
        for service in self.services:
            service.vmm_enabled = True
        await self.client.status()
        slept = await self.client.sleep()
        self.assertTrue(all(not reply.weights_ready and reply.kv_cache_bytes == 0 for reply in slept))
        mapped = await self.client.wake_up(tags=["weights"])
        self.assertTrue(all(not reply.weights_ready and not reply.kv_cache_ready for reply in mapped))
        await self.client.begin_update(global_step=1, max_buffer_bytes=1024)
        await self.client.push_weight("model.embed_tokens.weight", [b"ipc"] * 4)
        committed = await self.client.commit_update()
        self.assertTrue(all(reply.weights_ready and not reply.kv_cache_ready for reply in committed))
        await self.client.wake_up(tags=["kv_cache"])
        running = await self.client.resume()
        self.assertTrue(all(reply.phase == "running" for reply in running))

    async def test_partially_implemented_sleep_poisons_client(self):
        self.services[-1].vmm_enabled = True
        await self.client.status()
        with self.assertRaises(ReplicaControlError):
            await self.client.sleep()
        with self.assertRaisesRegex(ReplicaControlError, "restart all ranks"):
            await self.client.status()

    async def test_internal_error_poisons_client(self):
        await self.client.status()
        self.services[2].fail_next = True
        with self.assertRaises(ReplicaControlError):
            await self.client.status()
        with self.assertRaisesRegex(ReplicaControlError, "restart all ranks"):
            await self.client.status()


if __name__ == "__main__":
    unittest.main()
