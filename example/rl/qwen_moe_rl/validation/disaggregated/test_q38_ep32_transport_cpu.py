"""CPU failure checks for the split expert lease and native acknowledgement."""

import asyncio
import hashlib
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from qwen38_ep32_transport import (
    ExpertFragment,
    ExpertNativeAck,
    ExpertPair,
    ExpertReceipt,
    ExpertReceived,
    ExpertReceiver,
    ExpertSender,
    ExpertShardLayout,
    PACKED_SHAPES,
    push_expert_source,
)


NAME = "model.layers.0.mlp.experts.gate_up_proj"
TXN = "native-incarnation:2"


def digest(tensor):
    return hashlib.sha256(bytes(tensor.view(torch.uint8).reshape(-1).tolist())).hexdigest()


def fixture_pair(rollout_rank):
    fragments = tuple(
        ExpertFragment(TXN, 1, 5, NAME, train_rank, hashlib.sha256(bytes([train_rank])).hexdigest())
        for train_rank in (2 * rollout_rank, 2 * rollout_rank + 1)
    )
    return ExpertPair(rollout_rank, fragments)


class Control:
    def __init__(self, *, omit=None, stale=False):
        self.update_id = TXN
        self.omit = omit
        self.stale = stale
        self.calls = 0

    async def push_weight(self, name, metadata):
        self.calls += 1
        assert name == NAME and len(metadata) == 32
        for rank, value in enumerate(metadata):
            assert ExpertShardLayout(rank).unpack_ipc(name, value) == bytes([rank])
        return [
            SimpleNamespace(
                world_rank=rank, world_size=32, ep_rank=rank, ep_size=32,
                update_id="old" if self.stale and rank == 3 else TXN, phase="updating",
                quiescent=True, received_weights=6,
            )
            for rank in range(32) if rank != self.omit
        ]


class ExpertTransportTest(unittest.TestCase):
    def test_reversed_or_cross_generation_fragments_fail_before_allocation(self):
        pair = fixture_pair(7)
        first, second = pair.fragments
        with self.assertRaisesRegex(ValueError, "wrong training owners"):
            ExpertPair(7, (second, first))
        stale = ExpertFragment("old", 0, 5, NAME, second.train_rank, second.sha256)
        with self.assertRaisesRegex(ValueError, "transaction or tensor coordinate"):
            ExpertPair(7, (first, stale))

    def test_receive_digest_failure_keeps_storage_and_blocks_ipc(self):
        export = Mock(return_value=b"fake IPC")
        with patch.dict(PACKED_SHAPES, {NAME: (512, 2, 3)}):
            a = torch.zeros((8, 2, 3), dtype=torch.bfloat16)
            b = torch.ones_like(a)
            pair = ExpertPair(17, (
                ExpertFragment(TXN, 1, 5, NAME, 34, digest(a)),
                ExpertFragment(TXN, 1, 5, NAME, 35, digest(b)),
            ))
            receiver = ExpertReceiver(
                rollout_rank=17, gpu_uuid="gpu-17", native_incarnation="instance",
                allocate=lambda shape, dtype: torch.empty(shape, dtype=getattr(torch, dtype)),
                receive_and_wait=lambda target, source: target.copy_({34: b, 35: a}[source]),
                export_ipc=export, digest=digest,
            )
            receiver.prepare(pair)
            received = receiver.slot.buffer
            with self.assertRaisesRegex(RuntimeError, "differs from trainer EP rank 34"):
                receiver.receive()
            export.assert_not_called()
            self.assertIs(receiver.slot.buffer, received)
            self.assertEqual(receiver.slot.state, "poisoned")

    def test_transport_error_keeps_the_exact_training_buffer(self):
        tensor = torch.zeros((8, 2, 3), dtype=torch.bfloat16)
        send = Mock(side_effect=RuntimeError("uncertain send"))
        sender = ExpertSender(63, send_and_wait=send, digest=digest)
        with patch.dict(PACKED_SHAPES, {NAME: (512, 2, 3)}):
            fragment = sender.prepare(transaction=TXN, step=1, ordinal=5, name=NAME, tensor=tensor)
            with self.assertRaisesRegex(RuntimeError, "uncertain send"):
                sender.send(fragment)
            sent, destination = send.call_args.args
            self.assertIs(sent, tensor)
            self.assertEqual(destination, 95)
            self.assertIs(sender.buffer, tensor)
            self.assertEqual(sender.state, "poisoned")

    def test_native_ack_requires_all_ranks_and_the_active_transaction(self):
        records = []
        for rank in range(32):
            pair = fixture_pair(rank)
            frame = ExpertShardLayout(rank).pack_ipc(NAME, (2 * rank, 2 * rank + 1), bytes([rank]))
            records.append(ExpertReceived(ExpertReceipt(pair, f"gpu-{rank}", "instance"), frame))
        control = Control()
        acks = asyncio.run(push_expert_source(control, records))
        self.assertEqual(acks, tuple(ExpertNativeAck(record.receipt) for record in records))
        self.assertEqual(control.calls, 1)
        for partial in (Control(omit=17), Control(stale=True)):
            with self.assertRaisesRegex(RuntimeError, "do not cover"):
                asyncio.run(push_expert_source(partial, records))
        with self.assertRaisesRegex(ValueError, "native ranks in order"):
            asyncio.run(push_expert_source(control, list(reversed(records))))
        self.assertEqual(control.calls, 1)


if __name__ == "__main__":
    unittest.main()
