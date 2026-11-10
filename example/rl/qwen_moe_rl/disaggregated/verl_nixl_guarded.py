"""Guarded adapter for pinned verl's native NIXL checkpoint engine.

Loads the upstream NIXL engine from verl commit 7c760b0; NIXL/UCX remains the
transport. Adds bounded full-HF reassembly and explicit native IPC ACK leases.
This does not implement target-layout expert shards. A new CUDA/UCX/NIXL and
real native IPC qualification is mandatory before production use.
"""
from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

import torch
from verl.checkpoint_engine.base import CheckpointEngineRegistry
from verl.checkpoint_engine import nixl_checkpoint_engine as upstream
from full_hf_relay import tensor_digest


UPSTREAM_SHA256 = "5ba0aa3ecb41ef71849f9919d2e291defa2b78be7e29a3a640ddc25602493a71"
WIRE_PREFIX = "rtp-sha256-v1:"


def _wire_name(name, digest):
    if not isinstance(name, str) or not name or not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("invalid source name or SHA256")
    return WIRE_PREFIX + digest + ":" + name


def _unwire_name(name):
    if not isinstance(name, str) or not name.startswith(WIRE_PREFIX):
        raise RuntimeError("NIXL source lacks trainer digest")
    rest = name[len(WIRE_PREFIX):]
    digest, sep, original = rest.partition(":")
    if not sep or not original or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise RuntimeError("invalid NIXL source digest descriptor")
    return original, digest


class _NativeLeaseIterator:
    def __init__(self, engine, global_steps):
        self.engine = engine
        self.iterator = engine._guarded_receive(global_steps)

    def __aiter__(self):
        return self

    async def __anext__(self):
        return await self.iterator.__anext__()

    def confirm_native_ack(self, name, proof):
        self.engine.confirm_native_ack(name, proof)


@CheckpointEngineRegistry.register("rtp_nixl_guarded")
class RtpGuardedNixl(upstream.NIXLCheckpointEngine):
    """NIXL READ ring plus an explicit all-rank native ACK buffer lease."""
    def __init__(self, bucket_size, is_master=False, *, max_tensor_bytes=16 * 1024**3,
                 timeout_seconds=600, transaction_timeout_seconds=43200, device="cuda"):
        source = Path(upstream.__file__)
        if source.is_symlink() or hashlib.sha256(source.read_bytes()).hexdigest() != UPSTREAM_SHA256:
            raise RuntimeError("NIXL backend source differs from pinned verl commit")
        if (type(bucket_size) is not int or bucket_size <= 0 or type(max_tensor_bytes) is not int or
                max_tensor_bytes <= 0 or type(timeout_seconds) is not int or timeout_seconds <= 0 or
                type(transaction_timeout_seconds) is not int or transaction_timeout_seconds <= 0):
            raise ValueError("invalid NIXL buffer/time bounds")
        super().__init__(bucket_size=bucket_size, is_master=is_master, device=device)
        self.max_tensor_bytes = max_tensor_bytes
        self.timeout_seconds = timeout_seconds
        self.transaction_timeout_seconds = transaction_timeout_seconds
        self.poisoned = False
        self.active = False
        self.finished = False
        self.pending = None
        self.pending_name = None
        self.pending_step = None
        self.native_acknowledged = False
        self.native_binding = None
        self.inflight_task = None

    def prepare(self):
        if self.active or self.pending is not None or self.inflight_task is not None:
            raise RuntimeError("previous NIXL generation not safely finalized")
        self.poisoned = False
        self.finished = False
        self.native_binding = None
        try:
            result = super().prepare()
            self.active = True
            return result
        except BaseException:
            self.poisoned = True
            # Any partially registered upstream buffers remain process-owned;
            # external supervisor must stop peers before releasing this actor.
            raise

    @classmethod
    def build_topology(cls, actor_wg_world_size, rollout_world_size, metadata):
        actor, rollout = super().build_topology(actor_wg_world_size, rollout_world_size, metadata)
        # The pinned upstream computes metadata[-rollout_world_size + 1:].
        # For exactly one receiver, -1 + 1 == 0, so that slice erroneously
        # selects every metadata item instead of no successor at all.
        if rollout_world_size == 1:
            rollout["next_agent_metadata"] = [None]
        return actor, rollout

    async def send_weights(self, weights, global_steps=None):
        if not self.active or self.poisoned or type(global_steps) is not int or global_steps < 0:
            raise RuntimeError("invalid NIXL sender generation or step")
        try:
            self.inflight_task = asyncio.create_task(super().send_weights(
                self._verified_sources(weights), global_steps=global_steps))
            # Do not cancel an in-progress transfer on timeout: the upstream
            # coroutine owns registered buffers/descriptors and transfer handles.
            await asyncio.wait_for(asyncio.shield(self.inflight_task), timeout=self.transaction_timeout_seconds)
            self.inflight_task = None
            self.finished = True
        except SystemExit as error:
            self.poisoned = True
            raise RuntimeError("upstream NIXL worker called exit on transfer error") from error
        except BaseException:
            self.poisoned = True
            raise

    def _verified_sources(self, weights):
        for name, value in weights:
            if self.rank < 0:
                # Megatron collectives may still require all trainer ranks to
                # exhaust the same exporter, but only rank 0 sends data.
                yield name, value
                continue
            if (not isinstance(value, torch.Tensor) or not value.is_cuda or not value.is_contiguous()
                    or value.dtype not in (torch.bfloat16, torch.float32)):
                raise RuntimeError("trainer must export contiguous BF16/FP32 CUDA tensors")
            if value.numel() * value.element_size() > self.max_tensor_bytes:
                raise RuntimeError("trainer source tensor exceeds transfer bound")
            # Encoding the hash in TensorMeta.name preserves the upstream
            # NIXL wire protocol without an uncoordinated side-channel. The
            # adapter sees only the original decoded HF name after verification.
            torch.cuda.synchronize(value.device)
            yield _wire_name(name, tensor_digest(value)), value

    def receive_weights(self, global_steps=None):
        return _NativeLeaseIterator(self, global_steps)

    def confirm_native_ack(self, name, proof):
        if (self.pending is None or self.native_acknowledged or name != self.pending_name or
                not isinstance(proof, dict) or proof.get("name") != name or
                proof.get("global_step") != self.pending_step or not proof.get("native_update_id") or
                not proof.get("native_incarnation") or proof.get("all_rank_acked") is not True):
            self.poisoned = True
            raise RuntimeError("invalid native source acknowledgement")
        count = proof.get("replica_world_size")
        if type(count) is not int or count <= 0 or proof.get("acknowledged_ranks") != list(range(count)):
            self.poisoned = True
            raise RuntimeError("native acknowledgement does not cover the replica")
        binding = (proof["native_update_id"], proof["native_incarnation"], count)
        if self.native_binding is not None and self.native_binding != binding:
            self.poisoned = True
            raise RuntimeError("native transaction changed during source stream")
        self.native_binding = binding
        self.native_acknowledged = True

    async def _guarded_receive(self, global_steps):
        if not self.active or self.poisoned or type(global_steps) is not int or global_steps < 0:
            raise RuntimeError("invalid NIXL receiver generation or step")
        chunks = self._receive_weight_chunks().__aiter__()
        merge_name = merge_shape = merge_dtype = merge_value = merge_bytes = None
        merge_offset = 0
        try:
            while True:
                try:
                    self.inflight_task = asyncio.create_task(chunks.__anext__())
                    meta, chunk = await asyncio.wait_for(asyncio.shield(self.inflight_task),
                                                         timeout=self.timeout_seconds)
                    self.inflight_task = None
                except StopAsyncIteration:
                    self.inflight_task = None
                    break
                except SystemExit as error:
                    raise RuntimeError("upstream NIXL worker called exit on transfer error") from error
                if chunk.dtype != torch.uint8 or not chunk.is_cuda or not chunk.is_contiguous():
                    raise RuntimeError("NIXL returned an invalid GPU byte chunk")
                if meta.dtype not in (torch.bfloat16, torch.float32) or not meta.name:
                    raise RuntimeError("unsupported NIXL source dtype or name")
                original_name, expected_digest = _unwire_name(meta.name)
                shape = tuple(meta.shape)
                if not shape or any(type(n) is not int or n <= 0 for n in shape):
                    raise RuntimeError("invalid source shape")
                size = int(meta.dtype.itemsize)
                for n in shape:
                    size *= n
                if size > self.max_tensor_bytes:
                    raise RuntimeError("complete HF tensor exceeds reassembly bound")
                if (type(meta.chunk_offset) is not int or meta.chunk_offset < 0 or
                        type(meta.chunk_size) is not int or meta.chunk_size <= 0 or
                        meta.chunk_size != chunk.numel() or meta.chunk_offset + meta.chunk_size > size):
                    raise RuntimeError("invalid NIXL source chunk bounds")
                if size <= self.bucket_size:
                    if merge_value is not None or meta.chunk_offset != 0 or meta.chunk_size != size:
                        raise RuntimeError("partial or overlapping short source")
                    value = chunk.view(meta.dtype).view(shape)
                else:
                    if merge_value is None:
                        if meta.chunk_offset != 0:
                            raise RuntimeError("large source did not begin at offset zero")
                        merge_name, merge_shape, merge_dtype = meta.name, shape, meta.dtype
                        merge_value = torch.empty(shape, dtype=meta.dtype, device=chunk.device)
                        merge_bytes = merge_value.view(-1).view(torch.uint8)
                        merge_offset = 0
                    if ((meta.name, shape, meta.dtype, meta.chunk_offset) !=
                            (merge_name, merge_shape, merge_dtype, merge_offset)):
                        raise RuntimeError("large source chunk order/shape mismatch")
                    merge_bytes[merge_offset:merge_offset + meta.chunk_size].copy_(chunk)
                    merge_offset += meta.chunk_size
                    if merge_offset < size:
                        continue
                    value = merge_value
                    merge_name = merge_shape = merge_dtype = merge_value = merge_bytes = None
                    merge_offset = 0
                if tensor_digest(value) != expected_digest:
                    raise RuntimeError("NIXL source differs from trainer SHA256")
                self.pending, self.pending_name, self.pending_step = value, original_name, global_steps
                self.native_acknowledged = False
                yield original_name, value
                if not self.native_acknowledged:
                    raise RuntimeError("adapter advanced without a native all-rank ACK")
                self.pending = self.pending_name = self.pending_step = None
                self.native_acknowledged = False
                # Adapter must discard its own tensor reference before __anext__.
                del value, chunk
            if merge_value is not None:
                raise RuntimeError("NIXL stream ended inside a source tensor")
            self.finished = True
        except BaseException:
            self.poisoned = True
            # Retain current pending allocation; do not call upstream finalize
            # while any native reader or peer NIXL operation may still hold it.
            raise

    def finalize(self):
        if (self.poisoned or not self.active or not self.finished or self.pending is not None
                or self.inflight_task is not None):
            raise RuntimeError("NIXL generation failed or still owns an IPC lease")
        super().finalize()
        self.active = False
        self.native_binding = None
