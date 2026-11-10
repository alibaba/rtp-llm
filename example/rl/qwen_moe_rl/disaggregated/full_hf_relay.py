"""Bounded CUDA full-HF relay primitives; experimental, not deployed.

One trainer representative and N isolated destination relays share a dedicated
NCCL ProcessGroup, separate from the trainer's Megatron process groups.  All
NCCL participants must call broadcast in exactly the same tensor order.  The
caller must coordinate metadata and the existing native all-rank transaction.

This is a correctness baseline: every target receives one complete HF source
tensor and the native loader performs its usual model-specific transformations.
It is not a direct expert-shard transfer and has not been GPU-qualified.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass


_DTYPE_BYTES = {"bfloat16": 2, "float32": 4}


@dataclass(frozen=True)
class TensorSpec:
    transaction: str
    step: int
    ordinal: int
    name: str
    shape: tuple[int, ...]
    dtype: str
    sha256: str

    def validate(self, max_bytes):
        if not isinstance(self.transaction, str) or not self.transaction:
            raise ValueError("missing transaction identity")
        if type(self.step) is not int or self.step < 0 or type(self.ordinal) is not int or self.ordinal < 0:
            raise ValueError("invalid step or source ordinal")
        if not isinstance(self.name, str) or not self.name or self.dtype not in _DTYPE_BYTES:
            raise ValueError("invalid source name or dtype")
        if not isinstance(self.shape, tuple) or not self.shape or any(type(n) is not int or n <= 0 for n in self.shape):
            raise ValueError("invalid source shape")
        if (not isinstance(self.sha256, str) or len(self.sha256) != 64
                or any(c not in "0123456789abcdef" for c in self.sha256)):
            raise ValueError("invalid SHA256")
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError("invalid maximum tensor size")
        count = 1
        for dimension in self.shape:
            count *= dimension
        size = count * _DTYPE_BYTES[self.dtype]
        if size > max_bytes:
            raise ValueError("source tensor exceeds complete-receive-buffer bound")
        return size


def tensor_digest(tensor, *, chunk_bytes=8 * 1024 * 1024):
    """Hash GPU tensor bytes with a bounded host staging allocation.

    Each blocking device-to-host copy is complete before the host view is read.
    This measures integrity, not link performance. Caller must first wait for
    the producer/receiver CUDA stream that owns the tensor.
    """
    import torch
    if not isinstance(tensor, torch.Tensor) or not tensor.is_cuda or not tensor.is_contiguous():
        raise ValueError("a contiguous CUDA source is required")
    if type(chunk_bytes) is not int or chunk_bytes <= 0:
        raise ValueError("invalid digest staging limit")
    digest = hashlib.sha256()
    data = tensor.view(torch.uint8).view(-1)
    for offset in range(0, data.numel(), chunk_bytes):
        host = data.narrow(0, offset, min(chunk_bytes, data.numel() - offset)).cpu()
        digest.update(memoryview(host.numpy()).cast("B"))
    return digest.hexdigest()


class TorchNcclBroadcast:
    """Wrap an already initialized dedicated ProcessGroupNCCL.

    The caller owns rendezvous, NCCL timeout, process identities, process-group
    shutdown and failure teardown. `root` is the rank in *this* group, not the
    trainer's global rank. A separate ProcessGroup allows Megatron's default
    process group to remain untouched.
    """
    def __init__(self, group, *, root=0):
        if group is None or type(root) is not int or root < 0:
            raise ValueError("dedicated NCCL group and root are required")
        self.group = group
        self.root = root

    def broadcast_and_wait(self, tensor):
        import torch
        import torch.distributed as dist
        if not tensor.is_cuda or not tensor.is_contiguous():
            raise ValueError("NCCL payload must be contiguous CUDA memory")
        options = dist.BroadcastOptions()
        options.rootRank = self.root
        options.rootTensor = 0
        work = self.group.broadcast([tensor], options)
        if not work.wait():
            raise RuntimeError("NCCL broadcast did not complete")
        torch.cuda.synchronize(tensor.device)


class RelaySlot:
    """One in-flight source tensor per native target GPU.

    The control coordinator must validate exact native rank/UUID/incarnation
    binding before construction, then call `prepare` on every relay and require
    all prepared ACKs before launching `receive` on every relay and invoking
    NCCL broadcast at the trainer. The allocated tensor
    remains owned here until the coordinator confirms the actual all-rank
    native push_weight ACK. Any uncertain operation poisons the slot and keeps
    its buffer alive; process teardown must prove native readers are gone.
    """
    def __init__(self, *, target_rank, gpu_uuid, native_incarnation, max_bytes,
                 allocate, broadcast_and_wait, export_ipc, digest):
        if (type(target_rank) is not int or target_rank < 0 or
                not isinstance(gpu_uuid, str) or not gpu_uuid or
                not isinstance(native_incarnation, str) or not native_incarnation or
                type(max_bytes) is not int or max_bytes <= 0 or
                not all(callable(f) for f in (allocate, broadcast_and_wait, export_ipc, digest))):
            raise ValueError("invalid relay identity or callbacks")
        self.target_rank = target_rank
        self.gpu_uuid = gpu_uuid
        self.native_incarnation = native_incarnation
        self.max_bytes = max_bytes
        self.allocate = allocate
        self.broadcast_and_wait = broadcast_and_wait
        self.export_ipc = export_ipc
        self.digest = digest
        self.state = "idle"
        self.buffer = None
        self.spec = None

    def prepare(self, spec):
        if self.state != "idle":
            raise RuntimeError("relay has an unacknowledged or poisoned buffer")
        try:
            size = spec.validate(self.max_bytes)
            self.spec = spec
            self.state = "allocating"
            self.buffer = self.allocate(spec.shape, spec.dtype)
            self.state = "prepared"
            return {"rank": self.target_rank, "gpu_uuid": self.gpu_uuid,
                    "native_incarnation": self.native_incarnation, "transaction": spec.transaction,
                    "step": spec.step, "ordinal": spec.ordinal, "name": spec.name,
                    "size_bytes": size, "sha256": spec.sha256, "prepared": True}
        except BaseException:
            self.state = "poisoned"
            raise

    def receive(self, spec):
        if self.state != "prepared" or self.spec != spec:
            self.state = "poisoned"
            raise RuntimeError("source was not prepared on this relay")
        try:
            size = spec.validate(self.max_bytes)
            self.state = "receiving"
            self.broadcast_and_wait(self.buffer)
            actual_digest = self.digest(self.buffer)
            if actual_digest != spec.sha256:
                raise RuntimeError("relay content differs from trainer source")
            metadata = self.export_ipc(self.buffer)
            if not isinstance(metadata, bytes) or not metadata:
                raise RuntimeError("invalid CUDA IPC metadata")
            self.state = "native_pending"
            return {"rank": self.target_rank, "gpu_uuid": self.gpu_uuid,
                    "native_incarnation": self.native_incarnation, "transaction": spec.transaction,
                    "step": spec.step, "ordinal": spec.ordinal, "name": spec.name,
                    "size_bytes": size, "sha256": actual_digest, "metadata": metadata}
        except BaseException:
            self.state = "poisoned"
            raise

    def native_ack(self, *, transaction, step, ordinal, name, native_incarnation):
        if self.state != "native_pending":
            raise RuntimeError("native acknowledgement has no pending source")
        spec = self.spec
        if ((transaction, step, ordinal, name, native_incarnation) !=
                (spec.transaction, spec.step, spec.ordinal, spec.name, self.native_incarnation)):
            self.state = "poisoned"
            raise RuntimeError("native acknowledgement identity mismatch")
        self.buffer = None
        self.spec = None
        self.state = "idle"
        return {"rank": self.target_rank, "released_after_native_ack": True}


def cuda_relay_slot(*, target_rank, gpu_uuid, native_incarnation, max_bytes, group):
    """Create relay callbacks after Ray has assigned exactly one GPU.

    UUID lookup and native-process ownership checks are caller responsibilities.
    Device 0 here is relative to Ray's isolated CUDA_VISIBLE_DEVICES.
    """
    import torch
    from rtp_llm.utils.cuda_ipc import CudaIpcV2
    if torch.cuda.device_count() != 1:
        raise RuntimeError("relay must own exactly one visible GPU")
    torch.cuda.set_device(0)
    actual_uuid = str(torch.cuda.get_device_properties(0).uuid).lower()
    if actual_uuid.removeprefix("gpu-") != gpu_uuid.lower().removeprefix("gpu-"):
        raise RuntimeError("relay and native GPU UUID differ")
    nccl = TorchNcclBroadcast(group)
    def allocate(shape, dtype):
        return torch.empty(shape, dtype=getattr(torch, dtype), device="cuda:0")
    return RelaySlot(target_rank=target_rank, gpu_uuid=gpu_uuid,
                     native_incarnation=native_incarnation, max_bytes=max_bytes,
                     allocate=allocate, broadcast_and_wait=nccl.broadcast_and_wait,
                     export_ipc=CudaIpcV2.export_tensor, digest=tensor_digest)
