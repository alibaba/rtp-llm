"""Bounded EP64-to-EP32 expert transport for a dedicated NCCL group.

Group ranks 0–63 are the live trainer EP ranks. Group rank 64+r is the
rollout relay for native EP rank r. The controller owns group setup, ordered
per-parameter rendezvous, and cleanup after uncertain completion.
"""

from dataclasses import dataclass
import hashlib

from full_hf_relay import RelaySlot, TensorSpec, tensor_digest
from rtp_llm.utils.qwen38_expert_shards import (
    ExpertShardLayout,
    PACKED_SHAPES,
    ROLLOUT_EP,
    TRAIN_EP,
    TRAIN_EXPERTS,
)


MAX_PACKED_EXPERT_BYTES = 1 << 30


@dataclass(frozen=True)
class ExpertFragment:
    transaction: str
    step: int
    ordinal: int
    name: str
    train_rank: int
    sha256: str

    def __post_init__(self):
        if type(self.train_rank) is not int or not 0 <= self.train_rank < TRAIN_EP:
            raise ValueError("Invalid training EP rank")
        self.spec.validate(MAX_PACKED_EXPERT_BYTES // 2)

    @property
    def spec(self):
        if self.name not in PACKED_SHAPES:
            raise ValueError("Expected a canonical Qwen3.8 packed expert source")
        shape = (TRAIN_EXPERTS, *PACKED_SHAPES[self.name][1:])
        return TensorSpec(self.transaction, self.step, self.ordinal, self.name,
                          shape, "bfloat16", self.sha256)


@dataclass(frozen=True)
class ExpertPair:
    rollout_rank: int
    fragments: tuple[ExpertFragment, ExpertFragment]

    def __post_init__(self):
        if type(self.fragments) is not tuple or len(self.fragments) != 2 or not all(
            isinstance(fragment, ExpertFragment) for fragment in self.fragments
        ):
            raise ValueError("An expert receiver requires two ordered source fragments")
        first, second = self.fragments
        if (first.train_rank, second.train_rank) != ExpertShardLayout(self.rollout_rank).train_ranks:
            raise ValueError("Expert fragments have the wrong training owners or order")
        if _coordinate(first) != _coordinate(second):
            raise ValueError("Expert fragments disagree on transaction or tensor coordinate")

    @property
    def spec(self):
        first, second = self.fragments
        digest = hashlib.sha256(bytes.fromhex(first.sha256) + bytes.fromhex(second.sha256)).hexdigest()
        shape = (2 * TRAIN_EXPERTS, *first.spec.shape[1:])
        return TensorSpec(*_coordinate(first), shape, "bfloat16", digest)


def _coordinate(fragment):
    return fragment.transaction, fragment.step, fragment.ordinal, fragment.name


@dataclass(frozen=True)
class ExpertReceipt:
    pair: ExpertPair
    gpu_uuid: str
    native_incarnation: str


@dataclass(frozen=True)
class ExpertReceived:
    receipt: ExpertReceipt
    metadata: bytes


@dataclass(frozen=True)
class ExpertNativeAck:
    receipt: ExpertReceipt


class ExpertSender:
    """Own one trainer slab through the exact native acknowledgement."""

    def __init__(self, train_rank, *, send_and_wait, digest):
        if type(train_rank) is not int or not 0 <= train_rank < TRAIN_EP:
            raise ValueError("Invalid training EP rank")
        self.train_rank = train_rank
        self.send_and_wait = send_and_wait
        self.digest = digest
        self.state = "idle"
        self.fragment = None
        self.buffer = None

    def prepare(self, *, transaction, step, ordinal, name, tensor):
        if self.state != "idle":
            raise RuntimeError("Training expert buffer still has a lease")
        if name not in PACKED_SHAPES:
            raise ValueError("Expected a canonical packed expert source")
        expected = (TRAIN_EXPERTS, *PACKED_SHAPES[name][1:])
        if (tuple(tensor.shape) != expected or str(tensor.dtype) != "torch.bfloat16"
            or not tensor.is_contiguous()):
            raise ValueError("Training expert source must be a contiguous BF16 ETP1 slab")
        self.buffer = tensor
        try:
            self.fragment = ExpertFragment(
                transaction, step, ordinal, name, self.train_rank, self.digest(tensor)
            )
        except BaseException:
            self.state = "poisoned"
            raise
        self.state = "prepared"
        return self.fragment

    def send(self, fragment):
        if self.state != "prepared" or fragment != self.fragment:
            self.state = "poisoned"
            raise RuntimeError("Expert send has no matching prepared slab")
        try:
            self.send_and_wait(self.buffer, TRAIN_EP + self.train_rank // 2)
        except BaseException:
            self.state = "poisoned"
            raise
        self.state = "native_pending"

    def native_ack(self, ack):
        pair = ack.receipt.pair if isinstance(ack, ExpertNativeAck) else None
        if (self.state != "native_pending" or pair is None
            or pair.rollout_rank != self.train_rank // 2
            or pair.fragments[self.train_rank % 2] != self.fragment):
            self.state = "poisoned"
            raise RuntimeError("Trainer expert lease has no matching native ACK")
        self.buffer = None
        self.fragment = None
        self.state = "idle"


class ExpertReceiver:
    """Receive, verify, and expose IPC on the corresponding native GPU."""

    def __init__(self, *, rollout_rank, gpu_uuid, native_incarnation, allocate,
                 receive_and_wait, export_ipc, digest):
        self.layout = ExpertShardLayout(rollout_rank)
        self.pair = None
        self.receive_and_wait = receive_and_wait
        self.digest = digest
        self.slot = RelaySlot(
            target_rank=rollout_rank, gpu_uuid=gpu_uuid, native_incarnation=native_incarnation,
            max_bytes=MAX_PACKED_EXPERT_BYTES, allocate=allocate, broadcast_and_wait=self._receive_pair,
            export_ipc=export_ipc, digest=self._verify_pair,
        )

    def prepare(self, pair):
        if not isinstance(pair, ExpertPair) or pair.rollout_rank != self.layout.rollout_rank:
            raise ValueError("Expert pair belongs to a different rollout rank")
        prepared = self.slot.prepare(pair.spec)
        self.pair = pair
        return prepared

    def _receive_pair(self, tensor):
        for index, fragment in enumerate(self.pair.fragments):
            slab = tensor.narrow(0, index * TRAIN_EXPERTS, TRAIN_EXPERTS)
            self.receive_and_wait(slab, fragment.train_rank)

    def _verify_pair(self, tensor):
        for index, fragment in enumerate(self.pair.fragments):
            slab = tensor.narrow(0, index * TRAIN_EXPERTS, TRAIN_EXPERTS)
            if self.digest(slab) != fragment.sha256:
                raise RuntimeError(f"Expert fragment differs from trainer EP rank {fragment.train_rank}")
        return self.pair.spec.sha256

    def receive(self):
        if self.pair is None:
            raise RuntimeError("No prepared expert pair")
        spec = self.pair.spec
        record = self.slot.receive(spec)
        try:
            metadata = self.layout.pack_ipc(spec.name, self.layout.train_ranks, record["metadata"])
            return ExpertReceived(
                ExpertReceipt(self.pair, record["gpu_uuid"], record["native_incarnation"]), metadata
            )
        except BaseException:
            self.slot.state = "poisoned"
            raise

    def native_ack(self, ack):
        expected = ExpertReceipt(self.pair, self.slot.gpu_uuid, self.slot.native_incarnation)
        if not isinstance(ack, ExpertNativeAck) or ack.receipt != expected:
            self.slot.state = "poisoned"
            raise RuntimeError("Receiver expert lease has no matching native ACK")
        spec = self.pair.spec
        result = self.slot.native_ack(
            transaction=spec.transaction, step=spec.step, ordinal=spec.ordinal, name=spec.name,
            native_incarnation=ack.receipt.native_incarnation,
        )
        self.pair = None
        return result


async def push_expert_source(control, records):
    """Expose one source only after all 32 relays have verified both fragments."""
    if len(records) != ROLLOUT_EP or any(not isinstance(record, ExpertReceived) for record in records):
        raise ValueError("Expert push requires one verified record per native rank")
    receipts = [record.receipt for record in records]
    if [receipt.pair.rollout_rank for receipt in receipts] != list(range(ROLLOUT_EP)):
        raise ValueError("Expert push records must cover native ranks in order")
    if len({receipt.gpu_uuid for receipt in receipts}) != ROLLOUT_EP or len(
        {receipt.native_incarnation for receipt in receipts}
    ) != 1:
        raise ValueError("Expert push records disagree on native GPU or incarnation identity")
    spec = receipts[0].pair.spec
    for receipt in receipts[1:]:
        candidate = receipt.pair.spec
        if (candidate.transaction, candidate.step, candidate.ordinal, candidate.name) != (
            spec.transaction, spec.step, spec.ordinal, spec.name
        ):
            raise ValueError("Expert push records describe different source coordinates")
    if control.update_id != spec.transaction:
        raise ValueError("Expert push transaction differs from native BEGIN_UPDATE")
    replies = await control.push_weight(spec.name, [record.metadata for record in records])
    if len(replies) != ROLLOUT_EP or any(
        reply.world_rank != rank or reply.world_size != ROLLOUT_EP
        or reply.ep_rank != rank or reply.ep_size != ROLLOUT_EP
        or reply.update_id != spec.transaction or reply.phase != "updating"
        or not reply.quiescent or reply.received_weights != spec.ordinal + 1
        for rank, reply in enumerate(replies)
    ):
        raise RuntimeError("Native expert acknowledgements do not cover the source on all ranks")
    return tuple(ExpertNativeAck(receipt) for receipt in receipts)


class NcclExpertEndpoint:
    """Point-to-point operations on an already owned 96-rank process group."""

    def __init__(self, group, rank):
        if group is None or group.size() != TRAIN_EP + ROLLOUT_EP or group.rank() != rank:
            raise ValueError("Dedicated expert NCCL group rank or size differs from the plan")
        self.group = group
        self.rank = rank

    def send(self, tensor, destination):
        if self.rank >= TRAIN_EP or destination != TRAIN_EP + self.rank // 2:
            raise ValueError("Unexpected expert send route")
        self._run(tensor, self.group.send, destination)

    def receive(self, tensor, source):
        if self.rank < TRAIN_EP or source not in ExpertShardLayout(self.rank - TRAIN_EP).train_ranks:
            raise ValueError("Unexpected expert receive route")
        self._run(tensor, self.group.recv, source)

    @staticmethod
    def _run(tensor, operation, peer):
        import torch

        if not tensor.is_cuda or not tensor.is_contiguous():
            raise ValueError("Expert NCCL payload must be contiguous CUDA storage")
        # NCCL does not distinguish tags. The controller orders source
        # ordinals and keeps one outstanding lease on each endpoint.
        if not operation([tensor], peer, 0).wait():
            raise RuntimeError("Expert NCCL work did not complete")
        torch.cuda.synchronize(tensor.device)


def cuda_expert_sender(group, train_rank):
    endpoint = NcclExpertEndpoint(group, train_rank)
    return ExpertSender(train_rank, send_and_wait=endpoint.send, digest=tensor_digest)


def cuda_expert_receiver(group, rollout_rank, *, gpu_uuid, native_incarnation):
    import torch
    from rtp_llm.utils.cuda_ipc import CudaIpcV2

    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expert relay requires one explicitly assigned GPU")
    torch.cuda.set_device(0)
    actual = str(torch.cuda.get_device_properties(0).uuid)
    if actual.lower().removeprefix("gpu-") != gpu_uuid.lower().removeprefix("gpu-"):
        raise RuntimeError("Expert relay and native target GPU UUIDs differ")
    endpoint = NcclExpertEndpoint(group, TRAIN_EP + rollout_rank)
    return ExpertReceiver(
        rollout_rank=rollout_rank, gpu_uuid=gpu_uuid, native_incarnation=native_incarnation,
        allocate=lambda shape, dtype: torch.empty(shape, dtype=getattr(torch, dtype), device="cuda:0"),
        receive_and_wait=endpoint.receive, export_ipc=CudaIpcV2.export_tensor, digest=tensor_digest,
    )
