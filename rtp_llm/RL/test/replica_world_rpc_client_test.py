"""72 real loopback gRPC endpoints and the actual ReplicaRpcClient.

These are CPU transport/state-contract tests, not native-engine, GPU,
collective, model loading or RL acceptance.
"""

import asyncio
import unittest

import grpc

from rtp_llm.cpp.model_rpc.proto import model_rpc_service_pb2 as pb
from rtp_llm.cpp.model_rpc.proto import model_rpc_service_pb2_grpc as rpc
from rtp_llm.RL.replica_rpc_client import ReplicaControlError, ReplicaRpcClient


TOPOLOGY = dict(world_size=72, tp_size=8, dp_size=9, ep_size=72)


class WorldService(rpc.RpcServiceServicer):
    def __init__(self, rank):
        self.rank = rank
        self.reply = pb.RlControlResponsePB(
            protocol_version=3, world_rank=rank, world_size=72,
            tp_rank=rank % 8, tp_size=8, dp_rank=rank // 8, dp_size=9,
            ep_rank=rank, ep_size=72, phase="running", model_version="world:0",
            global_step=0, kv_cache_ready=True, kv_cache_bytes=1024,
            quiescent=False,
        )
        self.requests = []
        self.gate = None
        self.entered = asyncio.Event()
        self.mutate_reply = None
        self.abort = False
        self.sequence = 0
        self.pending_step = None

    async def RlControl(self, request, context):
        self.requests.append(request)
        self.entered.set()
        if self.gate is not None:
            await self.gate.wait()
        if self.abort:
            await context.abort(grpc.StatusCode.UNAVAILABLE, "missing owned world ACK")
        expected = (3, self.rank, 72, 8, 9, 72)
        observed = (
            request.protocol_version, request.expected_world_rank,
            request.expected_world_size, request.expected_tp_size,
            request.expected_dp_size, request.expected_ep_size,
        )
        if observed != expected:
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, "world topology prevalidation")
        op = request.operation
        if op != request.STATUS and request.expected_version != self.reply.model_version:
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, "stale version")
        if op in (request.WRITE_WEIGHT, request.COMMIT_UPDATE):
            if self.reply.phase != "updating" or request.update_id != self.reply.update_id:
                await context.abort(grpc.StatusCode.FAILED_PRECONDITION, "stale transaction")
        if op == request.PAUSE:
            self.reply.phase = "paused"
            self.reply.quiescent = True
        elif op == request.RESUME:
            self.reply.phase = "running"
            self.reply.quiescent = False
        elif op == request.BEGIN_UPDATE:
            self.sequence += 1
            self.reply.phase = "updating"
            self.reply.update_id = "world:" + str(self.sequence)
            self.reply.received_weights = 0
            self.reply.applied_weights = 0
            self.pending_step = request.global_step
        elif op == request.WRITE_WEIGHT:
            self.reply.received_weights += 1
        elif op == request.COMMIT_UPDATE:
            self.reply.phase = "paused"
            self.reply.model_version = self.reply.update_id
            self.reply.update_id = ""
            self.reply.global_step = self.pending_step
            self.reply.applied_weights = 17
        elif op == request.RELEASE_KV_CACHE:
            self.reply.kv_cache_ready = False
            self.reply.kv_cache_bytes = 0
        elif op == request.RESTORE_KV_CACHE:
            self.reply.kv_cache_ready = True
            self.reply.kv_cache_bytes = 1024
        reply = pb.RlControlResponsePB()
        reply.CopyFrom(self.reply)
        if self.mutate_reply is not None:
            self.mutate_reply(reply, request)
        return reply


class WorldClientTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.services = [WorldService(rank) for rank in range(72)]
        self.servers = []
        self.addresses = []
        for service in self.services:
            server = grpc.aio.server()
            rpc.add_RpcServiceServicer_to_server(service, server)
            port = server.add_insecure_port("127.0.0.1:0")
            self.assertGreater(port, 0)
            await server.start()
            self.servers.append(server)
            self.addresses.append(f"127.0.0.1:{port}")
        self.client = self.new_client()
        self.other_clients = []

    def new_client(self):
        return ReplicaRpcClient(self.addresses, timeout=1, topology=TOPOLOGY)

    async def asyncTearDown(self):
        await self.client.close()
        await asyncio.gather(*(client.close() for client in self.other_clients))
        await asyncio.gather(*(server.stop(0) for server in self.servers))

    async def assert_poisoned(self):
        before = [len(service.requests) for service in self.services]
        with self.assertRaisesRegex(ReplicaControlError, "restart all ranks"):
            await self.client.status()
        self.assertEqual(before, [len(service.requests) for service in self.services])

    async def prepare_update(self, step=1):
        await self.client.status()
        await self.client.pause()
        await self.client.begin_update(global_step=step)

    async def write_one(self):
        await self.client.push_weight("model.layers.0.weight", [b"owned IPC metadata"] * 72)

    async def test_two_updates_cache_lifecycle_and_all_nine_tp_zero_ranks(self):
        replies = await self.client.status()
        self.assertEqual([r.world_rank for r in replies if r.tp_rank == 0], list(range(0, 72, 8)))
        for step in (0, 1):
            await self.client.pause()
            await self.client.begin_update(global_step=step)
            await self.write_one()
            await self.write_one()
            replies = await self.client.commit_update()
            self.assertEqual(self.client.version, f"world:{step + 1}")
            self.assertEqual({r.world_rank for r in replies}, set(range(72)))
            self.assertEqual({r.global_step for r in replies}, {step})
            self.assertEqual({r.received_weights for r in replies}, {2})
            await self.client.release_cache()
            await self.client.restore_cache()
            await self.client.resume()
            await self.client.status()
        self.assertFalse(self.client._failed)
        for rank, service in enumerate(self.services):
            for request in service.requests:
                self.assertEqual((request.protocol_version, request.expected_world_rank,
                                  request.expected_world_size, request.expected_tp_size,
                                  request.expected_dp_size, request.expected_ep_size),
                                 (3, rank, 72, 8, 9, 72))
        self.assertEqual(len({len(s.requests) for s in self.services}), 1)

    async def test_begin_waits_for_actual_seventy_second_ack(self):
        await self.client.status()
        await self.client.pause()
        slow = self.services[71]
        slow.gate = asyncio.Event()
        slow.entered.clear()
        pending = asyncio.create_task(self.client.begin_update(global_step=7))
        await asyncio.wait_for(slow.entered.wait(), 2)
        self.assertFalse(pending.done())
        self.assertIsNone(self.client._pending_global_step)
        self.assertEqual(self.client.update_id, "")
        slow.gate.set()
        replies = await pending
        self.assertEqual(len(replies), 72)
        self.assertEqual(self.client._pending_global_step, 7)

    async def test_duplicate_world_rank_is_fatal(self):
        self.services[71].reply.world_rank = 0
        with self.assertRaisesRegex(ReplicaControlError, "world rank mapping"):
            await self.client.status()
        await self.assert_poisoned()

    async def test_wrong_each_topology_coordinate_and_protocol_is_fatal(self):
        changes = dict(world_size=64, tp_size=72, dp_size=1, ep_size=8,
                       tp_rank=7, dp_rank=0, ep_rank=0, protocol_version=2)
        for field, value in changes.items():
            with self.subTest(field=field):
                client = self.new_client()
                self.other_clients.append(client)
                original = getattr(self.services[8].reply, field)
                setattr(self.services[8].reply, field, value)
                with self.assertRaisesRegex(ReplicaControlError, "world rank mapping"):
                    await client.status()
                self.assertTrue(client._failed)
                setattr(self.services[8].reply, field, original)

    async def test_missing_endpoint_ack_poison_has_no_retry(self):
        self.services[71].abort = True
        with self.assertRaisesRegex(ReplicaControlError, "control failed"):
            await self.client.status()
        await self.assert_poisoned()

    async def test_initial_status_requires_every_rank_quiescence_to_match_phase(self):
        self.services[71].reply.quiescent = True
        with self.assertRaisesRegex(ReplicaControlError, "quiescence"):
            await self.client.status()
        await self.assert_poisoned()

    async def test_partial_quiescence_pause_is_fatal(self):
        await self.client.status()
        self.services[71].mutate_reply = lambda r, q: setattr(r, "quiescent", False)
        with self.assertRaisesRegex(ReplicaControlError, "quiescence"):
            await self.client.pause()
        await self.assert_poisoned()

    async def test_same_wrong_begin_count_is_fatal(self):
        await self.client.status()
        await self.client.pause()
        for service in self.services:
            service.mutate_reply = lambda r, q: setattr(r, "received_weights", 1)
        with self.assertRaisesRegex(ReplicaControlError, "begin update/count"):
            await self.client.begin_update(global_step=1)
        await self.assert_poisoned()

    async def test_same_wrong_weight_ack_counter_is_fatal(self):
        await self.prepare_update()
        for service in self.services:
            service.mutate_reply = lambda r, q: setattr(r, "received_weights", 0)
        with self.assertRaisesRegex(ReplicaControlError, "weight ACK count"):
            await self.write_one()
        await self.assert_poisoned()

    async def test_same_wrong_update_id_is_fatal(self):
        await self.prepare_update()
        for service in self.services:
            service.mutate_reply = lambda r, q: setattr(r, "update_id", "foreign:9")
        with self.assertRaisesRegex(ReplicaControlError, "update ID"):
            await self.write_one()
        await self.assert_poisoned()

    async def test_partial_commit_cannot_publish_world_version(self):
        await self.prepare_update()
        await self.write_one()
        self.services[71].mutate_reply = lambda r, q: setattr(r, "model_version", "foreign:1")
        with self.assertRaisesRegex(ReplicaControlError, "disagree on model_version"):
            await self.client.commit_update()
        self.assertEqual(self.client.version, "world:0")
        await self.assert_poisoned()

    async def test_same_wrong_commit_version_step_and_counts_are_fatal(self):
        changes = dict(model_version="world:0", global_step=99, update_id="world:1",
                       received_weights=9, applied_weights=0)
        for field, value in changes.items():
            with self.subTest(field=field):
                # Keep independent worlds for each invalid commit; no rollback
                # or client recovery is simulated after a poisoned transaction.
                if field != "model_version":
                    for service in self.services:
                        service.reply.phase = "running"
                        service.reply.quiescent = False
                        service.reply.model_version = "world:0"
                        service.reply.update_id = ""
                        service.reply.global_step = 0
                        service.reply.received_weights = 0
                        service.reply.applied_weights = 0
                        service.sequence = 0
                        service.mutate_reply = None
                    self.other_clients.append(self.client)
                    self.client = self.new_client()
                await self.prepare_update()
                await self.write_one()
                for service in self.services:
                    service.mutate_reply = lambda r, q, f=field, v=value: setattr(r, f, v)
                with self.assertRaisesRegex(ReplicaControlError, "commit version/step/update/count"):
                    await self.client.commit_update()
                self.assertEqual(self.client.version, "world:0")
                await self.assert_poisoned()

    async def test_partial_release_and_inconsistent_kv_bytes_are_fatal(self):
        await self.client.status()
        await self.client.pause()
        self.services[71].mutate_reply = lambda r, q: setattr(r, "kv_cache_bytes", 1024)
        with self.assertRaisesRegex(ReplicaControlError, "KV capacity"):
            await self.client.release_cache()
        await self.assert_poisoned()

    async def test_same_wrong_kv_capacity_change_outside_cache_operation_is_fatal(self):
        await self.client.status()
        await self.client.pause()
        for service in self.services:
            service.mutate_reply = lambda r, q: setattr(r, "kv_cache_bytes", 2048)
        with self.assertRaisesRegex(ReplicaControlError, "KV capacity changed"):
            await self.client.begin_update(global_step=1)
        await self.assert_poisoned()

    async def test_incomplete_descriptor_coverage_poison_happens_before_any_rpc(self):
        await self.prepare_update()
        before = [len(service.requests) for service in self.services]
        with self.assertRaises(ValueError):
            await self.client.push_weight("w", [b"metadata"] * 71)
        self.assertEqual(before, [len(service.requests) for service in self.services])
        await self.assert_poisoned()

    async def test_empty_rank_descriptor_poison_happens_before_any_rpc(self):
        await self.prepare_update()
        before = [len(service.requests) for service in self.services]
        with self.assertRaises(ValueError):
            await self.client.push_weight("w", [b"metadata"] * 71 + [b""])
        self.assertEqual(before, [len(service.requests) for service in self.services])
        await self.assert_poisoned()

    async def test_status_does_not_accept_uncoordinated_version_change(self):
        await self.client.status()
        for service in self.services:
            service.reply.model_version = "foreign:1"
        with self.assertRaisesRegex(ReplicaControlError, "outside commit"):
            await self.client.status()
        await self.assert_poisoned()

    async def test_client_cancellation_poison_blocks_all_later_rpc(self):
        await self.client.status()
        slow = self.services[71]
        slow.gate = asyncio.Event()
        slow.entered.clear()
        pending = asyncio.create_task(self.client.pause())
        await asyncio.wait_for(slow.entered.wait(), 2)
        pending.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await pending
        await self.assert_poisoned()


class TopologyConstructorTest(unittest.TestCase):
    def test_additive_field_numbers_and_proto3_legacy_defaults(self):
        request_fields = (
            "operation", "timeout_ms", "expected_version", "update_id",
            "weight_name", "cuda_ipc_metadata", "max_buffer_bytes", "global_step",
            "protocol_version", "expected_world_rank", "expected_world_size",
            "expected_tp_size", "expected_dp_size", "expected_ep_size",
        )
        response_fields = (
            "protocol_version", "phase", "model_version", "update_id", "global_step",
            "tp_rank", "tp_size", "received_weights", "applied_weights", "quiescent",
            "kv_cache_ready", "kv_cache_bytes", "torch_allocated_bytes", "torch_reserved_bytes",
            "world_rank", "world_size", "dp_rank", "dp_size", "ep_rank", "ep_size",
        )
        for message, names in ((pb.RlControlRequestPB, request_fields),
                               (pb.RlControlResponsePB, response_fields)):
            self.assertEqual({f.name: f.number for f in message.DESCRIPTOR.fields},
                             {name: number for number, name in enumerate(names, 1)})
        legacy = pb.RlControlRequestPB(operation=pb.RlControlRequestPB.PAUSE,
                                       expected_version="run:0", timeout_ms=100)
        self.assertEqual(legacy.protocol_version, 0)
        self.assertFalse(any(f.number >= 9 for f, value in legacy.ListFields()))

    def test_only_explicit_qualified_world_profile_is_accepted(self):
        endpoints = [f"127.0.0.1:{10000 + r}" for r in range(72)]
        for bad in ({"tp_size": 8}, dict(TOPOLOGY, world_size=64),
                    dict(TOPOLOGY, ep_size=8), dict(TOPOLOGY, dp_size=True)):
            with self.subTest(topology=bad), self.assertRaises(ValueError):
                ReplicaRpcClient(endpoints, topology=bad)
        with self.assertRaises(ValueError):
            ReplicaRpcClient(endpoints[:71], topology=TOPOLOGY)
        with self.assertRaises(ValueError):
            ReplicaRpcClient(endpoints[:71] + [endpoints[0]], topology=TOPOLOGY)


if __name__ == "__main__":
    unittest.main()
