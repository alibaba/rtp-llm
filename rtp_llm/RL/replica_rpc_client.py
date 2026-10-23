"""Fail-closed, all-rank control for the native synchronous RL replica."""

import asyncio
import math

import grpc

from rtp_llm.cpp.model_rpc.proto import model_rpc_service_pb2 as pb
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2_grpc import RpcServiceStub


class ReplicaControlError(RuntimeError):
    """The owner must terminate every rank before releasing unacknowledged IPC storage."""


class ReplicaRpcClient:
    def __init__(self, addresses, timeout=120):
        if not addresses or len(set(addresses)) != len(addresses):
            raise ValueError("one unique endpoint per TP rank is required")
        if not math.isfinite(timeout) or not 0 < timeout <= 600:
            raise ValueError("control timeout must be in (0, 600] seconds")
        self.addresses = list(addresses)
        self.timeout = timeout
        self.version = None
        self.update_id = ""
        self._channels = None
        self._failed = False
        self._lock = asyncio.Lock()

    async def close(self):
        if self._channels is not None:
            await asyncio.gather(*(channel.close() for channel in self._channels))
            self._channels = None

    async def _call_all(
        self, operation, *, records=None, phase=None, cache_ready=None, **kwargs
    ):
        if self._failed:
            raise ReplicaControlError("replica control has failed; restart all ranks")
        if self._channels is None:
            self._channels = [
                grpc.aio.insecure_channel(address) for address in self.addresses
            ]
        requests = []
        for rank in range(len(self.addresses)):
            extra = {} if records is None else records[rank]
            requests.append(
                pb.RlControlRequestPB(
                    operation=operation,
                    expected_version=self.version or "",
                    timeout_ms=int(self.timeout * 1000),
                    **kwargs,
                    **extra,
                )
            )
        try:
            replies = await asyncio.gather(
                *(
                    RpcServiceStub(channel).RlControl(request, timeout=self.timeout + 5)
                    for channel, request in zip(self._channels, requests)
                ),
                return_exceptions=True,
            )
            for rank, reply in enumerate(replies):
                if isinstance(reply, BaseException):
                    raise ReplicaControlError(
                        f"TP rank {rank} control failed: {reply}"
                    ) from reply
                if (
                    reply.protocol_version != 2
                    or reply.tp_rank != rank
                    or reply.tp_size != len(replies)
                ):
                    raise ReplicaControlError("RL protocol or TP rank mapping mismatch")
                if phase is not None and reply.phase != phase:
                    raise ReplicaControlError(
                        f"TP rank {rank} returned unexpected phase {reply.phase}"
                    )
                if phase in ("paused", "updating") and not reply.quiescent:
                    raise ReplicaControlError(
                        f"TP rank {rank} has not acknowledged quiescence"
                    )
                if cache_ready is not None and reply.kv_cache_ready != cache_ready:
                    raise ReplicaControlError(
                        f"TP rank {rank} returned unexpected KV residency"
                    )
                if reply.kv_cache_ready != (reply.kv_cache_bytes > 0):
                    raise ReplicaControlError(
                        f"TP rank {rank} returned inconsistent KV capacity"
                    )
                if reply.phase == "failed" or (
                    phase == "updating" and not reply.update_id
                ):
                    raise ReplicaControlError(
                        f"TP rank {rank} returned an unusable transaction state"
                    )
            for field in (
                "phase",
                "model_version",
                "update_id",
                "global_step",
                "received_weights",
                "applied_weights",
                "kv_cache_ready",
                "kv_cache_bytes",
            ):
                if len({getattr(reply, field) for reply in replies}) != 1:
                    raise ReplicaControlError(f"TP ranks disagree on {field}")
            if not replies[0].model_version:
                raise ReplicaControlError(
                    "native engine did not publish a model version"
                )
            self.version = replies[0].model_version
            self.update_id = replies[0].update_id
            return replies
        except BaseException:
            self._failed = True
            raise

    async def status(self):
        async with self._lock:
            return await self._call_all(pb.RlControlRequestPB.STATUS)

    async def pause(self):
        async with self._lock:
            if self.version is None:
                await self._call_all(pb.RlControlRequestPB.STATUS)
            return await self._call_all(pb.RlControlRequestPB.PAUSE, phase="paused")

    async def resume(self):
        async with self._lock:
            return await self._call_all(
                pb.RlControlRequestPB.RESUME, phase="running", cache_ready=True
            )

    async def release_cache(self):
        async with self._lock:
            return await self._call_all(
                pb.RlControlRequestPB.RELEASE_KV_CACHE,
                phase="paused",
                cache_ready=False,
            )

    async def restore_cache(self):
        async with self._lock:
            return await self._call_all(
                pb.RlControlRequestPB.RESTORE_KV_CACHE, phase="paused", cache_ready=True
            )

    async def begin_update(self, *, global_step, max_buffer_bytes=2 * 1024**3):
        async with self._lock:
            return await self._call_all(
                pb.RlControlRequestPB.BEGIN_UPDATE,
                phase="updating",
                global_step=global_step,
                max_buffer_bytes=max_buffer_bytes,
            )

    async def push_weight(self, name, metadata_per_rank):
        if len(metadata_per_rank) != len(self.addresses):
            raise ValueError("CUDA IPC descriptors must cover every TP rank")
        async with self._lock:
            return await self._call_all(
                pb.RlControlRequestPB.WRITE_WEIGHT,
                phase="updating",
                update_id=self.update_id,
                weight_name=name,
                records=[{"cuda_ipc_metadata": value} for value in metadata_per_rank],
            )

    async def commit_update(self):
        async with self._lock:
            return await self._call_all(
                pb.RlControlRequestPB.COMMIT_UPDATE,
                phase="paused",
                update_id=self.update_id,
            )
