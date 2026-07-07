"""In-process mock rtp-llm engines for FlexLB online evaluation."""

from __future__ import annotations

import asyncio
import json
import logging
import signal
import struct
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .proto_utils import ensure_proto_modules
from .rt_model import (
    PerformanceModel,
    RequestShape,
    compute_block_keys,
    to_signed_int64,
)

SENTINEL = object()

logger = logging.getLogger("mock_engine")


def now_ms() -> int:
    return int(time.time() * 1000)


class LruBlockCache:
    def __init__(self, capacity_blocks: int, block_size: int):
        self.capacity_blocks = max(0, int(capacity_blocks))
        self.block_size = block_size
        self._items: OrderedDict[int, None] = OrderedDict()
        self.evictions = 0

    def prefix_hit_blocks(self, block_keys: Iterable[int]) -> int:
        hit = 0
        touched: List[int] = []
        for key in block_keys:
            key = int(key)
            if key not in self._items:
                break
            hit += 1
            touched.append(key)
        for key in touched:
            self._items.move_to_end(key)
        return hit

    def admit(self, block_keys: Iterable[int]) -> bool:
        changed = False
        for key in block_keys:
            key = int(key)
            if key in self._items:
                self._items.move_to_end(key)
                continue
            if self.capacity_blocks <= 0:
                continue
            self._items[key] = None
            changed = True
            while len(self._items) > self.capacity_blocks:
                self._items.popitem(last=False)
                self.evictions += 1
        return changed

    @property
    def keys(self) -> List[int]:
        return list(self._items.keys())

    @property
    def used_tokens(self) -> int:
        return len(self._items) * self.block_size


@dataclass
class TaskRuntime:
    request_id: int
    batch_id: int
    input_len: int
    output_len: int
    block_keys: List[int]
    prefix_len: int = 0
    phase: int = 0
    start_ms: int = 0
    execution_time_ms: int = 0
    dp_rank: int = 0


class MockEngineState:
    def __init__(
        self,
        *,
        pb2,
        name: str,
        role: str,
        host: str,
        grpc_port: int,
        http_port: int,
        performance: PerformanceModel,
        cache_capacity_blocks: int,
        total_kv_tokens: int,
        block_size: int,
        cluster: "MockEngineCluster",
        performance_override: PerformanceModel | None = None,
    ):
        self.pb2 = pb2
        self.name = name
        self.role = role
        self.host = host
        self.grpc_port = grpc_port
        self.http_port = http_port
        self.performance = performance_override or performance
        self.cache = LruBlockCache(cache_capacity_blocks, block_size)
        self.total_kv_tokens = int(total_kv_tokens)
        self.block_size = block_size
        self.cluster = cluster
        self.inject_config: dict = {}

        self._lock = asyncio.Lock()
        self._running: Dict[int, TaskRuntime] = {}
        self._finished: List[Tuple[int, object]] = []
        self._response_queues: Dict[int, asyncio.Queue] = {}
        self._cancelled: set[int] = set()
        self._status_version = 1
        self._finished_version = 0
        self._cache_version = 1
        self._active_kv_tokens = 0
        self._accepted = 0
        self._completed = 0
        self._rpc_counts: Dict[str, int] = {
            "enqueue_batch": 0,
            "generate_stream": 0,
            "fetch_response": 0,
            "cancel": 0,
        }
        self._cancelled_count = 0
        self._request_lifecycle: Dict[int, dict] = {}
        self._injected_queue_depth = 0

    def set_injection(self, config: dict) -> None:
        """Set error-injection / timeout-simulation config.

        Supported keys:
          enqueue_error  – enqueue_batch returns empty-successes response
          fetch_error    – fetch_response yields one frame then raises grpc.RpcError
          generate_error – generate_stream raises grpc.RpcError immediately
          no_respond     – prefill/decode sleep but never queue responses (stream hangs)
        """
        self.inject_config = dict(config)

    def clear_injection(self) -> None:
        self.inject_config = {}

    @property
    def ip_port(self) -> str:
        return f"{self.host}:{self.grpc_port}"

    @property
    def http_ip_port(self) -> str:
        return f"{self.host}:{self.http_port}"

    async def enqueue_batch(self, request) -> object:
        self._rpc_counts["enqueue_batch"] += 1
        batch_id = int(request.batch_id)
        rids = [
            int(item.input.request_id)
            for slot in request.dp_slots
            for item in slot.requests
        ]
        logger.info(
            "EnqueueBatch arrived engine=%s batch_id=%d rids=%s n=%d",
            self.name,
            batch_id,
            rids,
            len(rids),
        )
        if self.inject_config.get("enqueue_error"):
            return self.pb2.EnqueueBatchResponsePB(batch_id=batch_id)
        inputs = []
        for slot in request.dp_slots:
            for item in slot.requests:
                inputs.append(item.input)
        for input_pb in inputs:
            self._response_queues.setdefault(int(input_pb.request_id), asyncio.Queue())
        asyncio.create_task(self._run_prefill_batch(batch_id, inputs))
        response = self.pb2.EnqueueBatchResponsePB(batch_id=batch_id)
        for input_pb in inputs:
            response.successes.add(request_id=int(input_pb.request_id))
        return response

    async def generate_stream(self, input_pb):
        self._rpc_counts["generate_stream"] += 1
        request_id = int(input_pb.request_id)
        role_addrs_count = len(input_pb.generate_config.role_addrs)
        logger.info(
            "GenerateStreamCall arrived engine=%s rid=%d role_addrs=%d tokens=%d output_len=%d",
            self.name,
            request_id,
            role_addrs_count,
            len(input_pb.token_ids),
            input_pb.generate_config.max_new_tokens,
        )
        if self.inject_config.get("generate_error"):
            import grpc

            raise grpc.RpcError("injected generate_error")
        if self.inject_config.get("enqueue_error"):
            import grpc

            raise grpc.RpcError("injected enqueue_error")
        request_id = int(input_pb.request_id)
        queue = self._response_queues.setdefault(request_id, asyncio.Queue())
        if self.role == "decode":
            asyncio.create_task(self._run_decode(input_pb, -1, queue))
        else:
            asyncio.create_task(self._run_prefill_batch(-1, [input_pb]))
        async for output in self._read_response_queue(request_id):
            yield output

    async def fetch_response(self, request_id: int):
        self._rpc_counts["fetch_response"] += 1
        logger.info(
            "FetchResponse arrived engine=%s rid=%d", self.name, int(request_id)
        )
        self._response_queues.setdefault(int(request_id), asyncio.Queue())
        if self.inject_config.get("fetch_error"):
            import grpc

            yield self._output_pb(
                int(request_id), finished=False, token_id=101, output_len=1
            )
            raise grpc.RpcError("injected fetch_error")
        async for output in self._read_response_queue(int(request_id)):
            yield output

    async def cancel(self, request_id: int) -> None:
        self._rpc_counts["cancel"] += 1
        self._cancelled_count += 1
        was_running = int(request_id) in self._running
        logger.info(
            "Cancel arrived engine=%s rid=%d was_running=%s",
            self.name,
            int(request_id),
            was_running,
        )
        async with self._lock:
            self._cancelled.add(int(request_id))
            self._running.pop(int(request_id), None)
            self._status_version += 1
        queue = self._response_queues.get(int(request_id))
        if queue is not None:
            await queue.put(SENTINEL)

    async def worker_status(self, request) -> object:
        latest = int(getattr(request, "latest_finished_version", -1))
        async with self._lock:
            status = self.pb2.WorkerStatusPB(
                role=self.role.upper(),
                role_type=self._role_pb(),
                available_concurrency=max(0, 4096 - len(self._running)),
                waiting_query_len=(
                    self._injected_queue_depth if self._injected_queue_depth > 0 else 0
                ),
                running_query_len=len(self._running),
                step_latency_ms=0.0,
                iterate_count=self._completed,
                dp_size=1,
                tp_size=1,
                status_version=self._status_version,
                alive=True,
                precision="mock",
                latest_finished_version=self._finished_version,
                dp_rank=0,
                available_kv_cache=max(
                    0, self.total_kv_tokens - self._active_kv_tokens
                ),
                total_kv_cache=self.total_kv_tokens,
            )
            for task in self._running.values():
                status.running_task_info.append(self._task_pb(task))
            for version, task_pb in self._finished:
                if version > latest:
                    status.finished_task_list.append(task_pb)
            return status

    async def cache_status(self, request) -> object:
        need_keys = bool(getattr(request, "need_cache_keys", True))
        async with self._lock:
            response = self.pb2.CacheStatusPB(
                available_kv_cache=max(
                    0, self.total_kv_tokens - self._active_kv_tokens
                ),
                total_kv_cache=self.total_kv_tokens,
                block_size=self.block_size,
                version=self._cache_version,
            )
            if need_keys:
                for key in self.cache.keys:
                    response.cache_keys[int(key)] = True
            return response

    async def snapshot(self) -> dict:
        async with self._lock:
            return {
                "name": self.name,
                "role": self.role,
                "grpc_addr": self.ip_port,
                "http_addr": self.http_ip_port,
                "running": len(self._running),
                "accepted": self._accepted,
                "completed": self._completed,
                "cache_keys": len(self.cache.keys),
                "cache_evictions": self.cache.evictions,
                "active_kv_tokens": self._active_kv_tokens,
                "available_kv_tokens": max(
                    0, self.total_kv_tokens - self._active_kv_tokens
                ),
                "status_version": self._status_version,
                "cache_version": self._cache_version,
                "inject_config": dict(self.inject_config),
                "rpc_counts": dict(self._rpc_counts),
                "cancelled_count": self._cancelled_count,
                "cancelled_rids": sorted(self._cancelled),
                "request_lifecycle": {
                    str(k): v for k, v in self._request_lifecycle.items()
                },
            }

    def _prune_lifecycle(self) -> None:
        """Keep _request_lifecycle bounded; evict oldest 20% when over 10000 entries."""
        if len(self._request_lifecycle) > 10000:
            oldest_keys = sorted(
                self._request_lifecycle.keys(),
                key=lambda k: self._request_lifecycle[k].get("arrived_ms", 0),
            )[:2000]
            for k in oldest_keys:
                del self._request_lifecycle[k]

    async def get_request_lifecycle_snapshot(self) -> dict:
        return dict(self._request_lifecycle)

    async def _run_prefill_batch(self, batch_id: int, inputs: List[object]) -> None:
        if not inputs:
            return
        shapes = [self._shape_from_input(input_pb) for input_pb in inputs]
        start = now_ms()
        async with self._lock:
            self._accepted += len(inputs)
            for shape in shapes:
                task = TaskRuntime(
                    request_id=shape.request_id,
                    batch_id=batch_id,
                    input_len=shape.input_len,
                    output_len=shape.output_len,
                    block_keys=shape.block_keys,
                    prefix_len=shape.hit_tokens,
                    phase=self.pb2.TASK_PHASE_RUNNING,
                    start_ms=start,
                    dp_rank=0,
                )
                self._running[shape.request_id] = task
            arrived_ms = int(time.time() * 1000)
            for shape in shapes:
                self._request_lifecycle[shape.request_id] = {
                    "rid": shape.request_id,
                    "method": "enqueue_batch" if batch_id >= 0 else "generate_stream",
                    "batch_id": batch_id,
                    "arrived_ms": arrived_ms,
                    "running_ms": arrived_ms,
                    "end_ms": 0,
                    "end_state": "running",
                }
            self._prune_lifecycle()
            self._status_version += 1

        prefill_ms = self.performance.prefill_ms(shapes)
        await asyncio.sleep(self.performance.sleep_seconds(prefill_ms))
        end = now_ms()

        async with self._lock:
            for shape in shapes:
                task = self._running.pop(shape.request_id, None)
                if task is None:
                    continue
                task.execution_time_ms = max(1, end - task.start_ms)
                self._finish_task(task)
                if self.cache.admit(shape.block_keys):
                    self._cache_version += 1
            end_ms_lc = int(time.time() * 1000)
            for rid in [s.request_id for s in shapes]:
                lc = self._request_lifecycle.get(rid)
                if lc:
                    lc["end_ms"] = end_ms_lc
                    lc["end_state"] = (
                        "cancelled" if rid in self._cancelled else "completed"
                    )
            self._status_version += 1

        if self.inject_config.get("no_respond"):
            return

        decode_tasks = []
        for input_pb, shape in zip(inputs, shapes):
            queue = self._response_queues.setdefault(shape.request_id, asyncio.Queue())
            if shape.request_id in self._cancelled:
                await queue.put(SENTINEL)
                continue
            decode_state = self.cluster.resolve_decode(input_pb)
            if decode_state is None or decode_state is self:
                decode_tasks.append(
                    asyncio.create_task(self._emit_without_decode(shape, queue))
                )
            else:
                decode_tasks.append(
                    asyncio.create_task(
                        decode_state._run_decode(input_pb, batch_id, queue)
                    )
                )
        if batch_id < 0 and decode_tasks:
            await asyncio.gather(*decode_tasks, return_exceptions=True)

    async def _run_decode(self, input_pb, batch_id: int, queue: asyncio.Queue) -> None:
        shape = self._shape_from_input(input_pb)
        start = now_ms()
        async with self._lock:
            self._accepted += 1
            active_batch = len(self._running) + 1
            self._active_kv_tokens += shape.input_len
            self._running[shape.request_id] = TaskRuntime(
                request_id=shape.request_id,
                batch_id=batch_id,
                input_len=shape.input_len,
                output_len=shape.output_len,
                block_keys=shape.block_keys,
                phase=self.pb2.TASK_PHASE_RUNNING,
                start_ms=start,
            )
            arrived_ms = int(time.time() * 1000)
            self._request_lifecycle[shape.request_id] = {
                "rid": shape.request_id,
                "method": "enqueue_batch" if batch_id >= 0 else "generate_stream",
                "batch_id": batch_id,
                "arrived_ms": arrived_ms,
                "running_ms": arrived_ms,
                "end_ms": 0,
                "end_state": "running",
            }
            self._prune_lifecycle()
            self._status_version += 1

        first_step_ms = self.performance.first_decode_step_ms(active_batch)
        await asyncio.sleep(self.performance.sleep_seconds(first_step_ms))
        if (
            not self.inject_config.get("no_respond")
            and shape.request_id not in self._cancelled
        ):
            await queue.put(
                self._output_pb(
                    shape.request_id, finished=False, token_id=101, output_len=1
                )
            )

        total_decode_ms = self.performance.decode_ms(shape.output_len, active_batch)
        remaining_ms = max(0.0, total_decode_ms - first_step_ms)
        await asyncio.sleep(self.performance.sleep_seconds(remaining_ms))
        end = now_ms()

        async with self._lock:
            task = self._running.pop(shape.request_id, None)
            self._active_kv_tokens = max(0, self._active_kv_tokens - shape.input_len)
            if task is not None:
                task.execution_time_ms = max(1, end - task.start_ms)
                self._finish_task(task)
            if self.cache.admit(shape.block_keys):
                self._cache_version += 1
            end_ms_lc = int(time.time() * 1000)
            lc = self._request_lifecycle.get(shape.request_id)
            if lc:
                lc["end_ms"] = end_ms_lc
                lc["end_state"] = (
                    "cancelled" if shape.request_id in self._cancelled else "completed"
                )
            self._status_version += 1

        if self.inject_config.get("no_respond"):
            return

        if shape.request_id not in self._cancelled:
            await queue.put(
                self._output_pb(
                    shape.request_id,
                    finished=True,
                    token_id=102,
                    output_len=shape.output_len,
                )
            )
        await queue.put(SENTINEL)

    async def _emit_without_decode(
        self, shape: RequestShape, queue: asyncio.Queue
    ) -> None:
        if shape.request_id not in self._cancelled:
            await queue.put(
                self._output_pb(
                    shape.request_id, finished=False, token_id=101, output_len=1
                )
            )
            await queue.put(
                self._output_pb(
                    shape.request_id,
                    finished=True,
                    token_id=102,
                    output_len=shape.output_len,
                )
            )
        await queue.put(SENTINEL)

    async def _read_response_queue(self, request_id: int):
        queue = self._response_queues.setdefault(request_id, asyncio.Queue())
        while True:
            item = await queue.get()
            if item is SENTINEL:
                self._response_queues.pop(request_id, None)
                return
            yield item

    def _shape_from_input(self, input_pb) -> RequestShape:
        meta = parse_unique_key(getattr(input_pb.generate_config, "unique_key", ""))
        token_ids = list(input_pb.token_ids)
        block_keys = meta.get("block_cache_keys")
        if block_keys is None:
            block_keys = compute_block_keys(token_ids, self.block_size)
        input_len = int(meta.get("input_len") or len(token_ids))
        output_len = int(
            meta.get("output_len") or input_pb.generate_config.max_new_tokens or 1
        )
        hit_blocks = self.cache.prefix_hit_blocks(block_keys)
        return RequestShape(
            request_id=int(input_pb.request_id),
            input_len=input_len,
            output_len=output_len,
            block_keys=[to_signed_int64(x) for x in block_keys],
            hit_tokens=hit_blocks * self.block_size,
        )

    def _finish_task(self, task: TaskRuntime) -> None:
        self._finished_version += 1
        self._completed += 1
        self._finished.append((self._finished_version, self._task_pb(task)))
        # Keep bounded history; FlexLB sync polls frequently and only needs recent
        # versions newer than latest_finished_version.
        if len(self._finished) > 10000:
            self._finished = self._finished[-5000:]

    def _task_pb(self, task: TaskRuntime):
        return self.pb2.TaskInfoPB(
            request_id=task.request_id,
            prefix_length=task.prefix_len,
            input_length=task.input_len,
            waiting_time_ms=0,
            iterate_count=1,
            end_time_ms=now_ms() if task.execution_time_ms else -1,
            dp_rank=task.dp_rank,
            batch_id=task.batch_id,
            phase=task.phase,
            execution_time_ms=task.execution_time_ms,
        )

    def _output_pb(
        self, request_id: int, *, finished: bool, token_id: int, output_len: int
    ):
        aux = self.pb2.AuxInfoPB(
            input_len=0,
            output_len=max(1, int(output_len)),
            step_output_len=1,
            iter_count=1,
            total_reuse_len=0,
        )
        output = self.pb2.GenerateOutputsPB(request_id=request_id)
        output.flatten_output.finished.append(bool(finished))
        output.flatten_output.aux_info.append(aux)
        output.flatten_output.output_ids.CopyFrom(_tensor_int32(self.pb2, [token_id]))
        return output

    def _role_pb(self):
        if self.role == "prefill":
            return self.pb2.ROLE_TYPE_PREFILL
        if self.role == "decode":
            return self.pb2.ROLE_TYPE_DECODE
        return self.pb2.ROLE_TYPE_PDFUSION


class MockRpcServicer:
    def __init__(self, state: MockEngineState, pb2):
        self.state = state
        self.pb2 = pb2

    async def GetWorkerStatus(self, request, context):
        return await self.state.worker_status(request)

    async def GetCacheStatus(self, request, context):
        return await self.state.cache_status(request)

    async def CheckHealth(self, request, context):
        return self.pb2.CheckHealthResponsePB(health="OK")

    async def EnqueueBatch(self, request, context):
        return await self.state.enqueue_batch(request)

    async def EnqueueGroup(self, request, context):
        batch = self.pb2.EnqueueBatchRequestPB(batch_id=request.batch_id)
        slot = batch.dp_slots.add(dp_rank=request.dp_rank)
        for item in request.requests:
            external = slot.requests.add()
            external.input.CopyFrom(item.input)
        return await self.state.enqueue_batch(batch)

    async def FetchResponse(self, request, context):
        async for output in self.state.fetch_response(int(request.request_id)):
            yield output

    async def GenerateStreamCall(self, request, context):
        async for output in self.state.generate_stream(request):
            yield output

    async def RemoteLoad(self, request, context):
        return self.pb2.BroadcastLoadResponsePB()

    async def RemoteGenerate(self, request_iterator, context):
        async for _ in request_iterator:
            break
        if False:
            yield self.pb2.GenerateOutputsPB()

    async def RemoteFinish(self, request, context):
        return self.pb2.EmptyPB()

    async def RemoteFinishNew(self, request, context):
        return self.pb2.EmptyPB()

    async def Cancel(self, request, context):
        await self.state.cancel(int(request.request_id))
        return self.pb2.EmptyPB()

    async def SetPause(self, request, context):
        return self.pb2.EmptyPB()

    async def SetRestart(self, request, context):
        return self.pb2.EmptyPB()

    async def SetLogLevel(self, request, context):
        return self.pb2.EmptyPB()

    async def StartProfile(self, request, context):
        return self.pb2.EmptyPB()

    async def StartProfileInternal(self, request, context):
        return self.pb2.EmptyPB()

    async def UpdateSchedulerInfo(self, request, context):
        return self.pb2.EmptyPB()

    async def UpdateEplbConfig(self, request, context):
        return self.pb2.EmptyPB()

    async def UpdateWeights(self, request, context):
        return self.pb2.EmptyPB()

    async def RemoteGenerateNew(self, request, context):
        return self.pb2.RemoteGenerateResponsePBNew(finished=True)

    async def RemoteStore(self, request, context):
        return self.pb2.RemoteStoreResponsePB()

    async def ExecuteFunction(self, request, context):
        return self.pb2.FunctionResponsePB()

    async def CpuTpBroadcast(self, request, context):
        return self.pb2.CpuTpBroadcastResponsePB(success=True)

    async def StartLoad(self, request, context):
        return self.pb2.P2PConnectorStartLoadResponsePB()

    async def GetPeerInfo(self, request, context):
        return self.pb2.GetPeerInfoResponsePB()


class MockEngineCluster:
    def __init__(
        self,
        pb2,
        pb2_grpc,
        performance: PerformanceModel,
        *,
        base_http_port: int = 0,
    ):
        self.pb2 = pb2
        self.pb2_grpc = pb2_grpc
        self.performance = performance
        self.states: List[MockEngineState] = []
        self._by_grpc_addr: Dict[str, MockEngineState] = {}
        self._servers = []
        self._base_http_port = base_http_port
        self._http_runner = None

    async def add_engine(
        self,
        *,
        name: str,
        role: str,
        host: str,
        port: int,
        cache_capacity_blocks: int,
        total_kv_tokens: int,
        block_size: int,
        performance_override: PerformanceModel | None = None,
    ) -> MockEngineState:
        import grpc

        server = grpc.aio.server(
            options=[
                ("grpc.max_send_message_length", 64 * 1024 * 1024),
                ("grpc.max_receive_message_length", 64 * 1024 * 1024),
            ]
        )
        http_port = port - 1
        state = MockEngineState(
            pb2=self.pb2,
            name=name,
            role=role,
            host=host,
            grpc_port=port,
            http_port=http_port,
            performance=self.performance,
            performance_override=performance_override,
            cache_capacity_blocks=cache_capacity_blocks,
            total_kv_tokens=total_kv_tokens,
            block_size=block_size,
            cluster=self,
        )
        servicer = MockRpcServicer(state, self.pb2)
        self.pb2_grpc.add_RpcServiceServicer_to_server(servicer, server)
        bound = server.add_insecure_port(f"{host}:{port}")
        if bound <= 0:
            raise RuntimeError(f"failed to bind {host}:{port}")
        state.grpc_port = bound
        state.http_port = bound - 1
        await server.start()
        self._servers.append(server)
        self.states.append(state)
        self._by_grpc_addr[state.ip_port] = state
        self._by_grpc_addr[f"localhost:{state.grpc_port}"] = state
        self._by_grpc_addr[f"127.0.0.1:{state.grpc_port}"] = state
        return state

    def resolve_decode(self, input_pb) -> Optional[MockEngineState]:
        for role_addr in input_pb.generate_config.role_addrs:
            if role_addr.role_type == self.pb2.ROLE_TYPE_DECODE:
                return self._by_grpc_addr.get(f"{role_addr.ip}:{role_addr.grpc_port}")
        decodes = [s for s in self.states if s.role == "decode"]
        if not decodes:
            return None
        return decodes[int(input_pb.request_id) % len(decodes)]

    async def start_http_server(self, port: int | None = None) -> None:
        """Start a lightweight aiohttp control API alongside the gRPC servers."""
        from aiohttp import web

        port = port or self._base_http_port
        if port <= 0:
            return
        app = web.Application()
        app.router.add_get("/snapshot", self._http_snapshot)
        app.router.add_post("/inject", self._http_inject)
        app.router.add_post("/clear_inject", self._http_clear_inject)
        app.router.add_get("/health", self._http_health)
        app.router.add_get("/requests", self._http_requests)
        app.router.add_post("/set_perf", self._http_set_perf)
        app.router.add_post("/set_kv_pressure", self._http_set_kv_pressure)
        app.router.add_post("/set_queue_depth", self._http_set_queue_depth)
        self._http_runner = web.AppRunner(app)
        await self._http_runner.setup()
        site = web.TCPSite(self._http_runner, "0.0.0.0", port)
        await site.start()

    async def _http_snapshot(self, request) -> object:
        from aiohttp import web

        return web.json_response(await self.snapshot())

    async def _http_inject(self, request) -> object:
        from aiohttp import web

        body = await request.json()
        engine_name = body.get("engine")
        config = body.get("config", {})
        engine = self._find_engine(engine_name)
        if engine is None:
            return web.json_response(
                {"error": f"engine '{engine_name}' not found"}, status=404
            )
        engine.set_injection(config)
        return web.json_response({"status": "ok", "engine": engine_name})

    async def _http_clear_inject(self, request) -> object:
        from aiohttp import web

        body = await request.json()
        engine_name = body.get("engine")
        engine = self._find_engine(engine_name)
        if engine is None:
            return web.json_response(
                {"error": f"engine '{engine_name}' not found"}, status=404
            )
        engine.clear_injection()
        return web.json_response({"status": "ok", "engine": engine_name})

    async def _http_health(self, request) -> object:
        from aiohttp import web

        return web.json_response({"status": "ok"})

    async def _http_requests(self, request):
        from aiohttp import web

        result = {}
        for state in self.states:
            result[state.name] = await state.get_request_lifecycle_snapshot()
        return web.json_response(result)

    async def _http_set_perf(self, request):
        """POST /set_perf — modify prefill_ms / decode_ms for a specific engine.
        Body: {"engine": "prefill-0", "prefill_fixed_ms": 200.0, "decode_scale": 2.0}
        """
        from aiohttp import web

        data = await request.json()
        engine_name = data.get("engine", "")
        for state in self.states:
            if state.name == engine_name:
                if "prefill_fixed_ms" in data:
                    state.performance.prefill_fixed_ms = float(data["prefill_fixed_ms"])
                if "decode_scale" in data:
                    state.performance.decode_scale = float(data["decode_scale"])
                return web.json_response({"status": "ok", "engine": engine_name})
        return web.json_response(
            {"status": "not_found", "engine": engine_name}, status=404
        )

    async def _http_set_kv_pressure(self, request):
        """POST /set_kv_pressure — set decode engine _active_kv_tokens.
        Body: {"engine": "decode-0", "active_kv_tokens": 999000}
        """
        from aiohttp import web

        data = await request.json()
        engine_name = data.get("engine", "")
        for state in self.states:
            if state.name == engine_name:
                state._active_kv_tokens = int(data.get("active_kv_tokens", 0))
                state._status_version += 1
                return web.json_response({"status": "ok", "engine": engine_name})
        return web.json_response(
            {"status": "not_found", "engine": engine_name}, status=404
        )

    async def _http_set_queue_depth(self, request):
        """POST /set_queue_depth — set prefill engine reported waiting_queue_len.
        Body: {"engine": "prefill-0", "queue_depth": 50000}
        """
        from aiohttp import web

        data = await request.json()
        engine_name = data.get("engine", "")
        for state in self.states:
            if state.name == engine_name:
                state._injected_queue_depth = int(data.get("queue_depth", 0))
                state._status_version += 1
                return web.json_response({"status": "ok", "engine": engine_name})
        return web.json_response(
            {"status": "not_found", "engine": engine_name}, status=404
        )

    def _find_engine(self, name: str) -> Optional[MockEngineState]:
        for state in self.states:
            if state.name == name:
                return state
        return None

    async def stop(self) -> None:
        if self._http_runner is not None:
            await self._http_runner.cleanup()
            self._http_runner = None
        for server in self._servers:
            await server.stop(grace=1)
        self._servers.clear()

    async def snapshot(self) -> dict:
        return {"engines": [await state.snapshot() for state in self.states]}

    def service_discovery_env(self, prefill_domain: str, decode_domain: str) -> dict:
        prefill = ",".join(s.ip_port for s in self.states if s.role == "prefill")
        decode = ",".join(s.ip_port for s in self.states if s.role == "decode")
        model_service_config = {
            "service_id": "aigc.text-generation.generation.engine_service",
            "load_balance": True,
            "role_endpoints": [
                {
                    "group": "mock",
                    "prefill_endpoint": {
                        "address": prefill_domain,
                        "protocol": "grpc",
                        "path": "/",
                    },
                    "decode_endpoint": {
                        "address": decode_domain,
                        "protocol": "grpc",
                        "path": "/",
                    },
                }
            ],
        }
        return {
            "MODEL_SERVICE_CONFIG": json.dumps(
                model_service_config, separators=(",", ":")
            ),
            f"DOMAIN_ADDRESS:{prefill_domain}": prefill,
            f"DOMAIN_ADDRESS:{decode_domain}": decode,
        }


def parse_unique_key(value: str) -> dict:
    if not value:
        return {}
    try:
        if value.startswith("flexlb_eval:"):
            value = value[len("flexlb_eval:") :]
        return json.loads(value)
    except Exception:
        return {}


def encode_unique_key(meta: dict) -> str:
    return "flexlb_eval:" + json.dumps(meta, separators=(",", ":"))


def _tensor_int32(pb2, values: List[int]):
    data = b"".join(struct.pack("<i", int(v)) for v in values)
    return pb2.TensorPB(
        data_type=pb2.TensorPB.INT32, shape=[1, 1, len(values)], int32_data=data
    )
