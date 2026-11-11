"""Fail-closed, all-rank control for the native synchronous RL replica."""

import asyncio
import math
import os

import grpc

from rtp_llm.cpp.model_rpc.proto import model_rpc_service_pb2 as pb
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2_grpc import RpcServiceStub


class ReplicaControlError(RuntimeError):
    """The owner must terminate every rank before releasing unacknowledged IPC storage."""


class ReplicaRpcClient:
    def __init__(self, addresses, timeout=120, *, topology=None):
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
        # The default remains the protocol-2 pure-TP contract. World control is
        # explicitly opt-in and limited to native profiles with corresponding
        # C++ guards; arithmetic alone proves no support.
        self._topology = None
        if topology is not None:
            expected = dict(world_size=72, tp_size=8, dp_size=9, ep_size=72)
            if not isinstance(topology, dict) or set(topology) != set(expected):
                raise ValueError("world control requires explicit world/TP/DP/EP sizes")
            if any(type(topology[key]) is not int for key in expected):
                raise ValueError("world topology sizes must be integers")
            if (topology not in (expected, dict(world_size=32, tp_size=8, dp_size=4, ep_size=32))
                    or len(self.addresses) != topology["world_size"]):
                raise ValueError("unsupported Qwen3.8 world topology or incomplete rank endpoints")
            self._topology = dict(topology)
        self._observed_state = None
        self._pending_global_step = None

    @staticmethod
    def _state(reply):
        return {
            field: getattr(reply, field)
            for field in (
                "phase", "model_version", "update_id", "global_step",
                "received_weights", "applied_weights", "kv_cache_ready",
                "kv_cache_bytes", "quiescent",
            )
        }

    def _check_world_transition(self, operation, reply, kwargs):
        """Validate the actual transition, beyond same-but-wrong all-rank ACKs."""
        state = self._state(reply)
        before = self._observed_state
        if state["phase"] not in ("running", "paused", "updating"):
            raise ReplicaControlError("world returned an unknown or failed phase")
        if state["quiescent"] != (state["phase"] != "running"):
            raise ReplicaControlError("world phase disagrees with quiescence")
        if state["global_step"] < 0:
            raise ReplicaControlError("world returned a negative global step")
        if before is None:
            if operation != pb.RlControlRequestPB.STATUS or state["update_id"] or state["phase"] == "updating":
                raise ReplicaControlError("world control requires a clean initial status")
            return
        if operation not in (pb.RlControlRequestPB.RELEASE_KV_CACHE, pb.RlControlRequestPB.RESTORE_KV_CACHE):
            if state["kv_cache_ready"] != before["kv_cache_ready"] or state["kv_cache_bytes"] != before["kv_cache_bytes"]:
                raise ReplicaControlError("world KV capacity changed outside cache release/restore")
        if operation == pb.RlControlRequestPB.COMMIT_UPDATE:
            if (
                before["phase"] != "updating"
                or not before["update_id"]
                or state["model_version"] != before["update_id"]
                or state["model_version"] == before["model_version"]
                or state["update_id"]
                or state["global_step"] != self._pending_global_step
                or state["received_weights"] != before["received_weights"]
                or state["received_weights"] == 0
                or state["applied_weights"] == 0
            ):
                raise ReplicaControlError("world commit version/step/update/count mismatch")
            return
        if state["model_version"] != before["model_version"] or state["global_step"] != before["global_step"]:
            raise ReplicaControlError("world changed model version or global step outside commit")
        if operation == pb.RlControlRequestPB.BEGIN_UPDATE:
            if (
                before["phase"] != "paused" or before["update_id"]
                or not state["update_id"] or state["update_id"] == state["model_version"]
                or state["received_weights"] != 0 or state["applied_weights"] != 0
            ):
                raise ReplicaControlError("world begin update/count mismatch")
            return
        if state["update_id"] != before["update_id"]:
            raise ReplicaControlError("world changed update ID outside begin/commit")
        expected_received = before["received_weights"] + (operation == pb.RlControlRequestPB.WRITE_WEIGHT)
        if state["received_weights"] != expected_received or state["applied_weights"] != before["applied_weights"]:
            raise ReplicaControlError("world weight ACK count did not match the operation")
        if operation == pb.RlControlRequestPB.WRITE_WEIGHT and before["phase"] != "updating":
            raise ReplicaControlError("world weight ACK has no active transaction")
        if operation == pb.RlControlRequestPB.STATUS and state != before:
            raise ReplicaControlError("world status changed without a coordinated operation")

    async def close(self):
        if self._channels is not None:
            await asyncio.gather(*(channel.close() for channel in self._channels))
            self._channels = None

    async def _call_all(
        self, operation, *, records=None, phase=None, cache_ready=None, **kwargs
    ):
        if self._failed:
            raise ReplicaControlError("replica control has failed; restart all ranks")
        try:
            if self._channels is None:
                self._channels = [
                    grpc.aio.insecure_channel(address) for address in self.addresses
                ]
            if records is not None and len(records) != len(self.addresses):
                raise ReplicaControlError("control records must cover every rank")
            if self._topology is not None and operation != pb.RlControlRequestPB.STATUS and self._observed_state is None:
                raise ReplicaControlError("world control requires an initial all-rank status")
            if self._topology is not None and operation == pb.RlControlRequestPB.BEGIN_UPDATE:
                step = kwargs.get("global_step")
                capacity = kwargs.get("max_buffer_bytes")
                if type(step) is not int or step < self._observed_state["global_step"] or type(capacity) is not int or capacity <= 0:
                    raise ReplicaControlError("invalid world update step or buffer capacity")
            requests = []
            for rank in range(len(self.addresses)):
                extra = {} if records is None else records[rank]
                world = {} if self._topology is None else dict(
                    protocol_version=3, expected_world_rank=rank,
                    expected_world_size=self._topology["world_size"],
                    expected_tp_size=self._topology["tp_size"],
                    expected_dp_size=self._topology["dp_size"],
                    expected_ep_size=self._topology["ep_size"],
                )
                requests.append(
                    pb.RlControlRequestPB(
                        operation=operation,
                        expected_version=self.version or "",
                        timeout_ms=int(self.timeout * 1000),
                        **world, **kwargs, **extra,
                    )
                )
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
                if self._topology is None:
                    if reply.protocol_version != 2 or reply.tp_rank != rank or reply.tp_size != len(replies):
                        raise ReplicaControlError("RL protocol or TP rank mapping mismatch")
                    if os.environ.get('RTP_LLM_RL_Q36_EP4') == '1' and (
                        len(replies) != 4 or reply.world_rank != rank or reply.world_size != 4
                        or reply.dp_rank != 0 or reply.dp_size != 1
                        or reply.ep_rank != rank or reply.ep_size != 4
                    ):
                        raise ReplicaControlError("Q36 EP4 protocol-2 world/expert rank mismatch")
                elif (
                    reply.protocol_version != 3 or reply.world_rank != rank
                    or reply.world_size != self._topology["world_size"]
                    or reply.tp_rank != rank % self._topology["tp_size"]
                    or reply.tp_size != self._topology["tp_size"]
                    or reply.dp_rank != rank // self._topology["tp_size"]
                    or reply.dp_size != self._topology["dp_size"]
                    or reply.ep_rank != rank or reply.ep_size != self._topology["ep_size"]
                ):
                    raise ReplicaControlError(f"RL protocol or world rank mapping mismatch at endpoint {rank}")
                if self._topology is not None and reply.quiescent != (reply.phase in ("paused", "updating")):
                    raise ReplicaControlError(f"world rank {rank} phase disagrees with quiescence")
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
            if self._topology is not None:
                if len(replies) != self._topology["world_size"] or {r.world_rank for r in replies} != set(range(self._topology["world_size"])):
                    raise ReplicaControlError("missing or duplicate world-rank acknowledgements")
                self._check_world_transition(operation, replies[0], kwargs)
                self._observed_state = self._state(replies[0])
                if operation == pb.RlControlRequestPB.BEGIN_UPDATE:
                    self._pending_global_step = kwargs["global_step"]
                elif operation == pb.RlControlRequestPB.COMMIT_UPDATE:
                    self._pending_global_step = None
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
            if self._topology is not None:
                self._failed = True
            raise ValueError("CUDA IPC descriptors must cover every rank")
        if self._topology is not None and (
            not isinstance(name, str) or not name
            or any(not isinstance(value, bytes) or not value for value in metadata_per_rank)
        ):
            self._failed = True
            raise ValueError("each world rank requires a nonempty named CUDA IPC descriptor")
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
