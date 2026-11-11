"""Exercise the complete source schedule and native ACK boundaries with CPU fixtures."""

import asyncio
from collections import Counter
from dataclasses import replace
import hashlib
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import uuid

import torch
import torch.distributed as dist

from full_hf_relay import TensorSpec
from q38_native32_control import Native32Policy, SourceLease
from q38_native32_topology import Native32Topology
from q38_refit_dense import (
    DenseNativeAck, DenseReceipt, DenseReceived, DenseReceiver, DenseSender, dense_dtype,
)
from q38_refit_groups import OwnedRefitGroups, RefitGroupPlan
from q38_refit_stream import EXPERT_ORDER, SplitRefitStream
from qwen38_ep32_transport import ExpertFragment, ExpertReceipt, ExpertReceived
from rtp_llm.utils.qwen38_expert_shards import ExpertShardLayout, PACKED_SHAPES


INCARNATION = "fixture-native-ep32"
ALOG = "model.layers.0.linear_attn.A_log"
DTBIAS = "model.layers.0.linear_attn.dt_bias"
LMDENSE = "lm_head.weight"
DENSE_ORDER = (ALOG, DTBIAS, LMDENSE, *(f"model.dense.{i}" for i in range(1403)))
SHAPES = {**PACKED_SHAPES, **{name: (2,) for name in DENSE_ORDER}}
GPU_UUIDS = tuple(str(uuid.UUID(int=rank+1)) for rank in range(96))
ROLLOUT_UUIDS = GPU_UUIDS[64:]


def digest_bytes(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


def digest_tensor(tensor):
    return hashlib.sha256(bytes(tensor.view(torch.uint8).reshape(-1).tolist())).hexdigest()


def topology():
    return Native32Topology(
        tuple(f"10.0.0.{i+1}" for i in range(8)),
        tuple(tuple(ROLLOUT_UUIDS[4*i:4*i+4]) for i in range(8)), 39870,
    )


class NativeControl:
    def __init__(self, *, partial_dense=False):
        self.phase = "paused"
        self.model_version = f"{INCARNATION}:0"
        self.update_id = ""
        self.step = 0
        self.names = []
        self.partial_dense = partial_dense
        self.commits = 0

    async def status(self):
        return [
            SimpleNamespace(
                world_rank=rank, world_size=32, ep_rank=rank, ep_size=32, phase=self.phase,
                model_version=self.model_version, global_step=self.step, update_id=self.update_id,
                quiescent=self.phase != "running", received_weights=len(self.names),
                applied_weights=int(bool(self.commits)), kv_cache_ready=True,
            )
            for rank in range(32)
        ]

    async def begin_update(self, *, global_step, max_buffer_bytes):
        assert 4*1024**3 < max_buffer_bytes <= 8*1024**3
        self.pending = global_step
        self.update_id = f"{INCARNATION}:{global_step+1}"
        self.phase = "updating"
        self.names = []

    async def push_weight(self, name, descriptors):
        assert len(descriptors) == 32 and name not in self.names
        for rank, descriptor in enumerate(descriptors):
            if name in PACKED_SHAPES:
                assert ExpertShardLayout(rank).unpack_ipc(name, descriptor) == bytes([rank])
            else:
                assert descriptor == bytes([rank])
        self.names.append(name)
        rows = await self.status()
        return rows[:-1] if self.partial_dense and name not in PACKED_SHAPES else rows

    async def commit_update(self):
        self.commits += 1
        self.step = self.pending
        self.model_version, self.update_id = self.update_id, ""
        self.phase = "paused"
        return await self.status()

    async def resume(self):
        self.phase = "running"
        return await self.status()


class FakeWorkers:
    """Fake transport endpoints with independently checked per-role method calls."""

    def __init__(self, control, *, bad_receiver_prepare=False):
        self.control = control
        self.bad_receiver_prepare = bad_receiver_prepare
        self.trainers = tuple(("train", rank) for rank in range(64))
        self.receivers = tuple(("rollout", rank) for rank in range(32))
        self.pairs = {}
        self.prepared = set()
        self.peer_drains = set()
        self.calls = Counter()
        self.dense_dtype_counts = Counter()
        self.dense_export_done = asyncio.Event()

    def prepare_receipt(self, rank, spec):
        self.prepared.add(rank)
        gpu_uuid = ROLLOUT_UUIDS[rank]
        if self.bad_receiver_prepare and rank == 11:
            gpu_uuid = ROLLOUT_UUIDS[10]
        return dict(rank=rank, gpu_uuid=gpu_uuid, native_incarnation=INCARNATION,
                    transaction=spec.transaction, step=spec.step, ordinal=spec.ordinal,
                    name=spec.name, sha256=spec.sha256, size_bytes=spec.validate(4*1024**3),
                    prepared=True)

    async def call(self, worker, method, *args):
        role, rank = worker
        self.calls[role, method] += 1
        if method == "begin_step":
            assert args == (self.control.update_id, 0)
        elif method == "expert_prepare":
            coordinate, = args
            if role == "train":
                return ExpertFragment(coordinate.transaction, coordinate.step, coordinate.ordinal,
                                      coordinate.name, rank, digest_bytes(rank))
            self.pairs[rank] = coordinate
            return self.prepare_receipt(rank, coordinate.spec)
        elif method == "expert_receive":
            pair = self.pairs[rank]
            descriptor = ExpertShardLayout(rank).pack_ipc(pair.spec.name, (2*rank, 2*rank+1), bytes([rank]))
            return ExpertReceived(ExpertReceipt(pair, ROLLOUT_UUIDS[rank], INCARNATION), descriptor)
        elif method == "expert_send":
            assert self.prepared == set(range(32)) and args[0].train_rank == rank
        elif method == "expert_ack":
            ack, = args
            rollout_rank = rank if role == "rollout" else rank // 2
            assert ack.receipt.pair == self.pairs[rollout_rank]
            assert self.control.names[-1] == ack.receipt.pair.spec.name
        elif method == "expert_finish":
            assert args == (self.control.update_id, 0)
        elif method == "dense_finish":
            if rank != 0:
                self.peer_drains.add(rank)
                await self.dense_export_done.wait()
            else:
                self.dense_export_done.set()
        elif method == "dense_prepare":
            if role == "train":
                transaction, step, ordinal = args
                assert self.peer_drains == set(range(1, 8))
                name = DENSE_ORDER[ordinal-len(EXPERT_ORDER)]
                dtype = dense_dtype(name)
                self.dense_dtype_counts[dtype] += 1
                return TensorSpec(transaction, step, ordinal, name, SHAPES[name], dtype, digest_bytes(name))
            return self.prepare_receipt(rank, args[0])
        elif method == "dense_receive":
            return DenseReceived(DenseReceipt(rank, ROLLOUT_UUIDS[rank], INCARNATION, args[0]), bytes([rank]))
        elif method == "dense_send":
            assert role == "train" and rank == 0 and self.prepared == set(range(32))
        elif method == "dense_ack":
            ack, = args
            if role == "train":
                assert len(ack) == 32 and all(isinstance(row, DenseNativeAck) for row in ack)
            else:
                assert isinstance(ack, DenseNativeAck) and ack.receipt.rank == rank
                assert ack.receipt.spec.name == self.control.names[-1]
        elif method == "finish_step":
            assert self.dense_export_done.is_set()
        elif method == "close":
            assert not args
        else:
            raise AssertionError(f"Unexpected worker call {method}")


class CoordinatorTest(unittest.TestCase):
    def test_complete_1590_source_refit_through_the_native_controller(self):
        async def scenario():
            control = NativeControl()
            workers = FakeWorkers(control)
            stream = SplitRefitStream(workers.trainers, workers.receivers, INCARNATION, SHAPES,
                                      ROLLOUT_UUIDS, call=workers.call)
            policy = Native32Policy(topology(), INCARNATION, SHAPES, control, [object()] * 4)
            result = await policy.refit(0, stream.leases(0, control))
            self.assertEqual(result["model_version"], f"{INCARNATION}:1")
            self.assertEqual(result["received_sources"], 1590)
            self.assertEqual(control.names, list(EXPERT_ORDER + DENSE_ORDER))
            self.assertEqual(control.commits, 1)
            self.assertFalse(stream.pending)
            self.assertIsNone(stream.held)
            self.assertEqual(workers.calls["train", "expert_send"], 64 * 184)
            self.assertEqual(workers.calls["rollout", "expert_ack"], 32 * 184)
            self.assertEqual(workers.calls["train", "dense_send"], 1406)
            self.assertEqual(workers.calls["rollout", "dense_ack"], 32 * 1406)
            self.assertEqual(workers.dense_dtype_counts, {"float32": 2, "bfloat16": 1404})
            await stream.close()
            self.assertEqual(workers.calls["train", "close"], 64)
            self.assertEqual(workers.calls["rollout", "close"], 32)
        asyncio.run(scenario())

    def test_failed_receiver_preparation_prevents_all_sends_and_commit(self):
        async def scenario():
            control = NativeControl()
            workers = FakeWorkers(control, bad_receiver_prepare=True)
            stream = SplitRefitStream(workers.trainers, workers.receivers, INCARNATION, SHAPES,
                                      ROLLOUT_UUIDS, call=workers.call)
            policy = Native32Policy(topology(), INCARNATION, SHAPES, control, [object()] * 4)
            with self.assertRaisesRegex(ValueError, "prepare acknowledgement"):
                await policy.refit(0, stream.leases(0, control))
            self.assertTrue(policy.poisoned and stream.poisoned)
            self.assertEqual(control.commits, 0)
            self.assertIsNotNone(stream.held)
            self.assertEqual(workers.calls["train", "expert_send"], 0)
        asyncio.run(scenario())

    def test_partial_dense_native_ack_keeps_lease_held(self):
        async def scenario():
            control = NativeControl(partial_dense=True)
            policy = Native32Policy(topology(), INCARNATION, SHAPES, control, [object()] * 4)
            spec = TensorSpec(f"{INCARNATION}:1", 0, 0, ALOG, (2,), "float32", digest_bytes(ALOG))
            records = tuple(DenseReceived(DenseReceipt(rank, value, INCARNATION, spec), bytes([rank]))
                            for rank, value in enumerate(ROLLOUT_UUIDS))
            release = Mock()
            lease = SourceLease(ALOG, tuple(row.metadata for row in records), (spec.sha256,) * 32,
                                release, dense_records=records)
            async def leases():
                yield lease
            with self.assertRaisesRegex(RuntimeError, "acknowledgements do not cover"):
                await policy.refit(0, leases())
            self.assertIs(policy.held_lease, lease)
            self.assertTrue(policy.poisoned)
            self.assertEqual(control.commits, 0)
            release.assert_not_called()
        asyncio.run(scenario())


class DenseBuffersTest(unittest.TestCase):
    def test_fp32_gdn_bytes_survive_broadcast_and_release_requires_all_acks(self):
        gdn = torch.tensor([0.06253, -0.003087], dtype=torch.float32)
        shapes = {ALOG: tuple(gdn.shape)}
        sent = []
        sender = DenseSender(shapes, 32, broadcast_and_wait=sent.append, digest=digest_tensor)
        with self.assertRaisesRegex(ValueError, "shape or dtype"):
            sender.prepare("txn", 0, 0, ALOG, gdn.bfloat16())
        spec = sender.prepare("txn", 0, 0, ALOG, gdn)
        sender.send(spec)
        self.assertIs(sent[0], gdn)
        self.assertEqual(spec.dtype, "float32")
        self.assertEqual(spec.validate(32), 8)
        receiver = DenseReceiver(
            shapes, 32, rollout_rank=0, gpu_uuid=ROLLOUT_UUIDS[0], native_incarnation=INCARNATION,
            allocate=lambda shape, dtype: torch.empty(shape, dtype=getattr(torch, dtype)),
            broadcast_and_wait=lambda output: output.copy_(sent[0]),
            export_ipc=lambda tensor: digest_tensor(tensor).encode(), digest=digest_tensor,
        )
        receiver.prepare(spec)
        result = receiver.receive(spec)
        self.assertEqual(result.metadata.decode(), digest_tensor(gdn))
        ack = DenseNativeAck(result.receipt)
        with self.assertRaisesRegex(RuntimeError, "all-rank native ACK"):
            sender.native_ack((ack,))
        self.assertIs(sender.buffer, gdn)
        self.assertEqual(sender.state, "poisoned")
        stale = DenseNativeAck(replace(result.receipt, native_incarnation="stale-incarnation"))
        with self.assertRaisesRegex(RuntimeError, "matching native ACK"):
            receiver.native_ack(stale)
        self.assertIsNotNone(receiver.slot.buffer)
        self.assertEqual(receiver.slot.state, "poisoned")


class GroupsTest(unittest.TestCase):
    def test_private_backends_preserve_megatron_registration(self):
        plan = RefitGroupPlan("fixture-refit-0001", "10.0.0.1", 39479, GPU_UUIDS)
        constructors = []
        def backend(store, rank, size, timeout):
            value = SimpleNamespace(rank=rank, size=size, shutdown=Mock())
            constructors.append(value)
            return value
        store = Mock()
        store.add.return_value = 1
        store.multi_get.return_value = [value.encode() for value in GPU_UUIDS]
        with (
            patch.object(dist, "TCPStore", return_value=store) as tcp,
            patch.object(dist, "PrefixStore", return_value=store) as prefixes,
            patch.object(dist, "ProcessGroupNCCL", side_effect=backend),
            patch.object(dist, "is_initialized", return_value=True),
            patch.object(dist, "get_rank", return_value=0),
            patch.object(dist, "get_world_size", return_value=64),
            patch.object(dist, "init_process_group", side_effect=AssertionError("default PG registration")),
            patch.object(dist, "destroy_process_group", side_effect=AssertionError("default PG shutdown")),
            patch.object(torch.cuda, "current_device", return_value=0),
            patch.object(torch.cuda, "get_device_properties", return_value=SimpleNamespace(uuid=GPU_UUIDS[0])),
        ):
            groups = OwnedRefitGroups(plan, 0)
            self.assertEqual([(group.rank, group.size) for group in constructors], [(0, 96), (0, 33)])
            self.assertEqual(tcp.call_args.args[:4], ("10.0.0.1", 39479, 96, True))
            self.assertEqual([call.args[0] for call in prefixes.call_args_list],
                             ["q38-refit-fixture-refit-0001", "experts", "dense"])
            groups.close()
            self.assertTrue(all(group.shutdown.call_count == 1 for group in constructors))
        self.assertIsNone(plan.dense_rank(63))
        self.assertEqual(plan.dense_rank(95), 32)


if __name__ == "__main__":
    unittest.main()
