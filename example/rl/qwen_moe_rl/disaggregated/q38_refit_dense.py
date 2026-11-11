"""Bounded broadcast and all-rank ACKs for Qwen3.8 shared HF sources."""

from dataclasses import dataclass

from full_hf_relay import RelaySlot, TensorSpec, TorchNcclBroadcast, tensor_digest
from q38_refit_groups import normalize_uuid
from rtp_llm.utils.qwen38_expert_shards import PACKED_SHAPES, ROLLOUT_EP


DEFAULT_MAX_DENSE_BYTES = 4 * 1024**3
GDN_FP32_SUFFIXES = (".linear_attn.A_log", ".linear_attn.dt_bias")


def dense_dtype(name):
    return "float32" if name.endswith(GDN_FP32_SUFFIXES) else "bfloat16"


def check_dense_spec(spec, shapes, max_bytes):
    if (not isinstance(spec, TensorSpec) or spec.name in PACKED_SHAPES
            or spec.name not in shapes or spec.shape != shapes[spec.name]
            or spec.dtype != dense_dtype(spec.name)):
        raise ValueError("Dense source does not match its canonical HF shape or dtype")
    return spec.validate(max_bytes)


@dataclass(frozen=True)
class DenseReceipt:
    rank: int
    gpu_uuid: str
    native_incarnation: str
    spec: TensorSpec


@dataclass(frozen=True)
class DenseReceived:
    receipt: DenseReceipt
    metadata: bytes


@dataclass(frozen=True)
class DenseNativeAck:
    receipt: DenseReceipt


class DenseSender:
    """One source buffer retained through all 32 native acknowledgements."""

    def __init__(self, shapes, max_bytes, *, broadcast_and_wait, digest):
        self.shapes = shapes
        self.max_bytes = max_bytes
        self.broadcast_and_wait = broadcast_and_wait
        self.digest = digest
        self.state = "idle"
        self.spec = None
        self.buffer = None

    def prepare(self, transaction, step, ordinal, name, tensor):
        if self.state != "idle":
            raise RuntimeError("Dense sender already holds an outstanding lease")
        spec = TensorSpec(transaction, step, ordinal, name, tuple(tensor.shape),
                          str(tensor.dtype).removeprefix("torch."), "0" * 64)
        check_dense_spec(spec, self.shapes, self.max_bytes)
        if not tensor.is_contiguous():
            raise ValueError("Dense source must use contiguous storage")
        self.buffer = tensor
        try:
            self.spec = TensorSpec(spec.transaction, spec.step, spec.ordinal, spec.name,
                                   spec.shape, spec.dtype, self.digest(tensor))
            self.spec.validate(self.max_bytes)
        except BaseException:
            self.state = "poisoned"
            raise
        self.state = "prepared"
        return self.spec

    def send(self, spec):
        if self.state != "prepared" or spec != self.spec:
            self.state = "poisoned"
            raise RuntimeError("Dense sender has no matching prepared source")
        try:
            self.broadcast_and_wait(self.buffer)
        except BaseException:
            self.state = "poisoned"
            raise
        self.state = "native_pending"

    def native_ack(self, acknowledgements):
        if (self.state != "native_pending" or not isinstance(acknowledgements, (tuple, list))
                or len(acknowledgements) != ROLLOUT_EP
                or any(not isinstance(row, DenseNativeAck) or row.receipt.rank != rank
                       or row.receipt.spec != self.spec
                       for rank, row in enumerate(acknowledgements))):
            self.state = "poisoned"
            raise RuntimeError("Dense sender did not receive matching all-rank native ACK")
        self.spec = None
        self.buffer = None
        self.state = "idle"


class DenseReceiver:
    def __init__(self, shapes, max_bytes, *, rollout_rank, gpu_uuid, native_incarnation,
                 allocate, broadcast_and_wait, export_ipc, digest):
        self.shapes = shapes
        self.slot = RelaySlot(
            target_rank=rollout_rank, gpu_uuid=gpu_uuid, native_incarnation=native_incarnation,
            max_bytes=max_bytes, allocate=allocate, broadcast_and_wait=broadcast_and_wait,
            export_ipc=export_ipc, digest=digest,
        )

    def prepare(self, spec):
        check_dense_spec(spec, self.shapes, self.slot.max_bytes)
        return self.slot.prepare(spec)

    def receive(self, spec):
        record = self.slot.receive(spec)
        receipt = DenseReceipt(self.slot.target_rank, self.slot.gpu_uuid, self.slot.native_incarnation, spec)
        return DenseReceived(receipt, record["metadata"])

    def native_ack(self, ack):
        receipt = DenseReceipt(self.slot.target_rank, self.slot.gpu_uuid, self.slot.native_incarnation, self.slot.spec)
        if not isinstance(ack, DenseNativeAck) or ack.receipt != receipt:
            self.slot.state = "poisoned"
            raise RuntimeError("Dense receiver did not receive its matching native ACK")
        spec = receipt.spec
        return self.slot.native_ack(transaction=spec.transaction, step=spec.step, ordinal=spec.ordinal,
                                    name=spec.name, native_incarnation=receipt.native_incarnation)


def validate_dense_records(records, gpu_uuids, native_incarnation, *, transaction, step, ordinal, name):
    if len(records) != ROLLOUT_EP or len(gpu_uuids) != ROLLOUT_EP:
        raise ValueError("Dense source requires all native ranks")
    if any(not isinstance(row, DenseReceived) for row in records):
        raise ValueError("Invalid dense receive record")
    spec = records[0].receipt.spec
    if (spec.transaction, spec.step, spec.ordinal, spec.name) != (transaction, step, ordinal, name):
        raise ValueError("Dense source differs from the active native transaction")
    for rank, (record, gpu_uuid) in enumerate(zip(records, gpu_uuids)):
        receipt = record.receipt
        if (receipt.rank != rank or normalize_uuid(receipt.gpu_uuid) != normalize_uuid(gpu_uuid)
                or receipt.native_incarnation != native_incarnation or receipt.spec != spec
                or not isinstance(record.metadata, bytes) or not record.metadata):
            raise ValueError("Dense receive record differs from the assigned native rank or source")
    return spec


def dense_native_acks(records, replies):
    if len(records) != ROLLOUT_EP or not isinstance(replies, (tuple, list)) or len(replies) != ROLLOUT_EP:
        raise RuntimeError("Dense native acknowledgements do not cover all ranks")
    spec = records[0].receipt.spec
    if any(
        reply.world_rank != rank or reply.world_size != ROLLOUT_EP
        or reply.ep_rank != rank or reply.ep_size != ROLLOUT_EP
        or reply.update_id != spec.transaction or reply.phase != "updating"
        or not reply.quiescent or reply.received_weights != spec.ordinal + 1
        for rank, reply in enumerate(replies)
    ):
        raise RuntimeError("Dense native acknowledgements do not match the active source")
    return tuple(DenseNativeAck(record.receipt) for record in records)


def cuda_dense_sender(group, shapes, max_bytes=DEFAULT_MAX_DENSE_BYTES):
    if group is None or group.size() != 1 + ROLLOUT_EP or group.rank() != 0:
        raise ValueError("Dense sender requires rank zero in the dedicated broadcast group")
    return DenseSender(shapes, max_bytes, broadcast_and_wait=TorchNcclBroadcast(group).broadcast_and_wait,
                       digest=tensor_digest)


def cuda_dense_receiver(group, shapes, rollout_rank, *, gpu_uuid, native_incarnation,
                        max_bytes=DEFAULT_MAX_DENSE_BYTES):
    import torch
    from rtp_llm.utils.cuda_ipc import CudaIpcV2

    if group is None or group.size() != 1 + ROLLOUT_EP or group.rank() != rollout_rank + 1:
        raise ValueError("Dense receiver requires its assigned broadcast rank")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Dense receiver requires one assigned GPU")
    torch.cuda.set_device(0)
    actual = str(torch.cuda.get_device_properties(0).uuid)
    if normalize_uuid(actual) != normalize_uuid(gpu_uuid):
        raise RuntimeError("Dense relay and native target GPU UUIDs differ")
    return DenseReceiver(
        shapes, max_bytes, rollout_rank=rollout_rank, gpu_uuid=gpu_uuid,
        native_incarnation=native_incarnation,
        allocate=lambda shape, dtype: torch.empty(shape, dtype=getattr(torch, dtype), device="cuda:0"),
        broadcast_and_wait=TorchNcclBroadcast(group).broadcast_and_wait,
        export_ipc=CudaIpcV2.export_tensor, digest=tensor_digest,
    )
