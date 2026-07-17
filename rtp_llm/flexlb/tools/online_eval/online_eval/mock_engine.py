"""In-process mock rtp-llm engines for FlexLB online evaluation."""

from __future__ import annotations

import asyncio
import json
import logging
import math
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


def _avg(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _p99(values: list[float]) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    idx = max(0, min(n - 1, math.ceil(0.99 * n) - 1))
    return sorted_vals[idx]


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


# ATOMICITY INVARIANT: This class uses no locks. All critical sections contain only
# synchronous code (no await). In asyncio's single-threaded model, code between await
# points is atomic. If you add an await inside any method that modifies state, you MUST
# restore asyncio.Lock protection.
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
        self._max_prefill_concurrency = 1
        self._prefill_semaphore = asyncio.Semaphore(self._max_prefill_concurrency)
        self._prefill_waiting = 0
        self._rpc_counts: Dict[str, int] = {
            "enqueue_batch": 0,
            "generate_stream": 0,
            "fetch_response": 0,
            "cancel": 0,
        }
        self._cancelled_count = 0
        self._request_lifecycle: Dict[int, dict] = {}
        self._injected_queue_depth = 0
        self._recent_prefill_times: list[float] = []
        self._recent_decode_times: list[float] = []

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
        return f"{self.host}:{self.http_port}"

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
        self._cancelled.add(int(request_id))
        self._running.pop(int(request_id), None)
        self._status_version += 1
        queue = self._response_queues.get(int(request_id))
        if queue is not None:
            await queue.put(SENTINEL)

    async def worker_status(self, request) -> object:
        latest = int(getattr(request, "latest_finished_version", -1))
        status = self.pb2.WorkerStatusPB(
            role=self.role.upper(),
            role_type=self._role_pb(),
            available_concurrency=max(
                0, self._max_prefill_concurrency - len(self._running)
            ),
            waiting_query_len=max(
                self._injected_queue_depth if self._injected_queue_depth > 0 else 0,
                self._prefill_waiting,
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
            available_kv_cache=max(0, self.total_kv_tokens - self._active_kv_tokens),
            total_kv_cache=self.total_kv_tokens,
        )
        for task in self._running.values():
            status.running_task_info.append(self._task_pb(task))
        filtered_count = 0
        for version, task_pb in self._finished:
            if version > latest:
                status.finished_task_list.append(task_pb)
                filtered_count += 1
        return status

    async def cache_status(self, request) -> object:
        need_keys = bool(getattr(request, "need_cache_keys", True))
        response = self.pb2.CacheStatusPB(
            available_kv_cache=max(0, self.total_kv_tokens - self._active_kv_tokens),
            total_kv_cache=self.total_kv_tokens,
            block_size=self.block_size,
            version=self._cache_version,
        )
        if need_keys:
            for key in self.cache.keys:
                response.cache_keys[int(key)] = True
        return response

    async def snapshot(self) -> dict:
        return {
            "name": self.name,
            "role": self.role,
            "grpc_addr": f"{self.host}:{self.grpc_port}",
            "http_addr": self.http_ip_port,
            "running": len(self._running),
            "waiting": max(self._injected_queue_depth, self._prefill_waiting),
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
            "prefill_ms_avg": _avg(self._recent_prefill_times),
            "prefill_ms_p99": _p99(self._recent_prefill_times),
            "prefill_ms_count": len(self._recent_prefill_times),
            "decode_ms_avg": _avg(self._recent_decode_times),
            "decode_ms_p99": _p99(self._recent_decode_times),
            "decode_ms_count": len(self._recent_decode_times),
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
        self._prefill_waiting += 1
        async with self._prefill_semaphore:
            self._prefill_waiting -= 1
            await asyncio.sleep(self.performance.sleep_seconds(prefill_ms))
        end = now_ms()

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
                lc["end_state"] = "cancelled" if rid in self._cancelled else "completed"
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
            if decode_state is not None and decode_state is not self:
                logger.debug(
                    f"[DIAG] _run_prefill_batch: rid={shape.request_id} "
                    f"decode_target={decode_state.name}"
                )
                decode_tasks.append(
                    asyncio.create_task(
                        decode_state._run_decode(input_pb, batch_id, queue)
                    )
                )
            else:
                remote_addr = self._get_remote_decode_addr(input_pb)
                if remote_addr is not None:
                    logger.debug(
                        f"[DIAG] _run_prefill_batch: rid={shape.request_id} "
                        f"decode_target=remote:{remote_addr}"
                    )
                    decode_tasks.append(
                        asyncio.create_task(self._run_remote_decode(input_pb, queue))
                    )
                else:
                    target_desc = "None" if decode_state is None else "self"
                    logger.debug(
                        f"[DIAG] _run_prefill_batch: rid={shape.request_id} "
                        f"decode_target={target_desc} (emit_without_decode, not tracked)"
                    )
                    decode_tasks.append(
                        asyncio.create_task(self._emit_without_decode(shape, queue))
                    )
        if batch_id < 0 and decode_tasks:
            await asyncio.gather(*decode_tasks, return_exceptions=True)

    async def _run_decode(self, input_pb, batch_id: int, queue: asyncio.Queue) -> None:
        shape = self._shape_from_input(input_pb)
        start = now_ms()
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

    def _get_remote_decode_addr(self, input_pb) -> Optional[str]:
        """Return the decode engine gRPC address from role_addrs, or None."""
        for role_addr in input_pb.generate_config.role_addrs:
            if role_addr.role_type == self.pb2.ROLE_TYPE_DECODE:
                return f"{role_addr.ip}:{role_addr.grpc_port}"
        return None

    async def _run_remote_decode(self, input_pb, queue: asyncio.Queue) -> None:
        """Forward a decode request to a remote engine via gRPC.

        Used when ``resolve_decode`` cannot find the decode engine locally
        (multi-shard mode).  The remote engine runs its own ``_run_decode``
        and streams responses back, which we relay into the local *queue*.
        """
        request_id = int(input_pb.request_id)
        decode_addr = self._get_remote_decode_addr(input_pb)
        if decode_addr is None:
            logger.warning(
                f"[DIAG] _run_remote_decode: rid={request_id} "
                f"no decode role_addr found, falling back to emit_without_decode"
            )
            shape = self._shape_from_input(input_pb)
            await self._emit_without_decode(shape, queue)
            return
        logger.debug(
            f"[DIAG] _run_remote_decode: rid={request_id} "
            f"forwarding to remote decode engine at {decode_addr}"
        )
        MAX_RETRIES = 1  # at most 1 retry (2 attempts total)
        retry_count = 0
        try:
            while True:
                try:
                    channel = await self.cluster._get_grpc_channel(decode_addr)
                    stub = self.cluster.pb2_grpc.RpcServiceStub(channel)
                    stream = stub.GenerateStreamCall(input_pb, timeout=120.0)
                    async for output in stream:
                        if request_id in self._cancelled:
                            # Forward the cancel to the remote engine so it
                            # stops decoding instead of running to completion.
                            self.cluster._grpc_cancel_forward_count += 1
                            try:
                                await stub.Cancel(
                                    self.pb2.CancelRequestPB(request_id=request_id),
                                    timeout=5.0,
                                )
                            except Exception:
                                pass  # best-effort, never block the cancel path
                            break
                        await queue.put(output)
                    break  # stream completed successfully, exit retry loop
                except Exception as exc:
                    retry_count += 1
                    self.cluster._grpc_error_count += 1
                    logger.error(
                        f"[DIAG] _run_remote_decode: rid={request_id} "
                        f"attempt={retry_count} error forwarding to "
                        f"{decode_addr}: {exc}"
                    )
                    if retry_count > MAX_RETRIES:
                        break  # exceeded retries, give up
                    self.cluster._grpc_retry_count += 1
                    await asyncio.sleep(0.5)
        finally:
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
        if task.execution_time_ms > 0:
            if self.role == "prefill":
                self._recent_prefill_times.append(float(task.execution_time_ms))
                if len(self._recent_prefill_times) > 100:
                    self._recent_prefill_times.pop(0)
            elif self.role == "decode":
                self._recent_decode_times.append(float(task.execution_time_ms))
                if len(self._recent_decode_times) > 100:
                    self._recent_decode_times.pop(0)
        logger.info(
            f"[DIAG] _finish_task: requestId={task.request_id}, new_finished_version={self._finished_version}, finished_list_len={len(self._finished)}"
        )
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


def generate_aggregated_prometheus_metrics(
    engines: list[dict],
    cluster_counters: dict | None = None,
) -> str:
    """Generate role-aggregated Prometheus exposition format metrics.

    Groups engines by role (prefill/decode) and aggregates:
    - Counters/state/KV: sum across engines
    - RPC: sum by method name
    - Latency avg: weighted average (sum(avg*count)/sum(count))
    - Latency p99: max across engines
    - Latency count: sum across engines
    """
    lines: list[str] = []
    metrics_meta = [
        ("mock_engine_up", "1 if engine is running, 0 if stopped", "gauge"),
        ("mock_engine_running", "current running requests", "gauge"),
        (
            "mock_engine_waiting",
            "current waiting requests (queued but not yet processing)",
            "gauge",
        ),
        ("mock_engine_accepted_total", "total accepted requests", "counter"),
        ("mock_engine_completed_total", "total completed requests", "counter"),
        ("mock_engine_cancelled_total", "total cancelled requests", "counter"),
        ("mock_engine_cache_keys", "number of cache keys", "gauge"),
        ("mock_engine_cache_evictions_total", "total cache evictions", "counter"),
        ("mock_engine_active_kv_tokens", "active KV cache tokens", "gauge"),
        ("mock_engine_available_kv_tokens", "available KV cache tokens", "gauge"),
        ("mock_engine_rpc_total", "total RPC calls by method", "counter"),
        (
            "mock_engine_prefill_ms_avg",
            "average prefill execution time in ms",
            "gauge",
        ),
        ("mock_engine_prefill_ms_p99", "p99 prefill execution time in ms", "gauge"),
        (
            "mock_engine_prefill_ms_count",
            "count of recent prefill completions",
            "gauge",
        ),
        (
            "mock_engine_decode_ms_avg",
            "average decode execution time in ms",
            "gauge",
        ),
        ("mock_engine_decode_ms_p99", "p99 decode execution time in ms", "gauge"),
        (
            "mock_engine_decode_ms_count",
            "count of recent decode completions",
            "gauge",
        ),
        (
            "flexlb_mock_grpc_error_count",
            "Total gRPC errors in remote decode",
            "counter",
        ),
        (
            "flexlb_mock_grpc_retry_count",
            "Total gRPC retries in remote decode",
            "counter",
        ),
        (
            "flexlb_mock_grpc_cancel_forward_count",
            "Total cancel forwarded to remote engines",
            "counter",
        ),
    ]
    for name, help_text, mtype in metrics_meta:
        lines.append(f"# HELP {name} {help_text}")
        lines.append(f"# TYPE {name} {mtype}")

    # Group engines by role
    buckets: dict[str, list[dict]] = {"prefill": [], "decode": []}
    for e in engines:
        role = e.get("role", "")
        if role in buckets:
            buckets[role].append(e)

    for role in ("prefill", "decode"):
        group = buckets[role]
        if not group:
            continue
        label = f'role="{role}"'

        up = sum(0 if e.get("stopped", False) else 1 for e in group)
        lines.append(f"mock_engine_up{{{label}}} {up}")

        running = sum(e.get("running", 0) for e in group)
        lines.append(f"mock_engine_running{{{label}}} {running}")

        waiting = sum(e.get("waiting", 0) for e in group)
        lines.append(f"mock_engine_waiting{{{label}}} {waiting}")

        accepted = sum(e.get("accepted", 0) for e in group)
        lines.append(f"mock_engine_accepted_total{{{label}}} {accepted}")

        completed = sum(e.get("completed", 0) for e in group)
        lines.append(f"mock_engine_completed_total{{{label}}} {completed}")

        cancelled = sum(e.get("cancelled_count", 0) for e in group)
        lines.append(f"mock_engine_cancelled_total{{{label}}} {cancelled}")

        cache_keys = sum(e.get("cache_keys", 0) for e in group)
        lines.append(f"mock_engine_cache_keys{{{label}}} {cache_keys}")

        cache_evictions = sum(e.get("cache_evictions", 0) for e in group)
        lines.append(f"mock_engine_cache_evictions_total{{{label}}} {cache_evictions}")

        active_kv = sum(e.get("active_kv_tokens", 0) for e in group)
        lines.append(f"mock_engine_active_kv_tokens{{{label}}} {active_kv}")

        available_kv = sum(e.get("available_kv_tokens", 0) for e in group)
        lines.append(f"mock_engine_available_kv_tokens{{{label}}} {available_kv}")

        rpc_totals: dict[str, int] = {}
        for e in group:
            for method, count in e.get("rpc_counts", {}).items():
                rpc_totals[method] = rpc_totals.get(method, 0) + count
        for method in sorted(rpc_totals.keys()):
            rpc_label = f'role="{role}",rpc_method="{method}"'
            lines.append(f"mock_engine_rpc_total{{{rpc_label}}} {rpc_totals[method]}")

        prefill_total_count = sum(e.get("prefill_ms_count", 0) for e in group)
        if prefill_total_count > 0:
            prefill_weighted = sum(
                e.get("prefill_ms_avg", 0.0) * e.get("prefill_ms_count", 0)
                for e in group
            )
            prefill_avg = prefill_weighted / prefill_total_count
        else:
            prefill_avg = 0.0
        prefill_p99 = max((e.get("prefill_ms_p99", 0.0) for e in group), default=0.0)
        lines.append(f"mock_engine_prefill_ms_avg{{{label}}} {prefill_avg:.1f}")
        lines.append(f"mock_engine_prefill_ms_p99{{{label}}} {prefill_p99:.1f}")
        lines.append(f"mock_engine_prefill_ms_count{{{label}}} {prefill_total_count}")

        decode_total_count = sum(e.get("decode_ms_count", 0) for e in group)
        if decode_total_count > 0:
            decode_weighted = sum(
                e.get("decode_ms_avg", 0.0) * e.get("decode_ms_count", 0) for e in group
            )
            decode_avg = decode_weighted / decode_total_count
        else:
            decode_avg = 0.0
        decode_p99 = max((e.get("decode_ms_p99", 0.0) for e in group), default=0.0)
        lines.append(f"mock_engine_decode_ms_avg{{{label}}} {decode_avg:.1f}")
        lines.append(f"mock_engine_decode_ms_p99{{{label}}} {decode_p99:.1f}")
        lines.append(f"mock_engine_decode_ms_count{{{label}}} {decode_total_count}")

    # Cluster-level metrics (no labels)
    if cluster_counters is not None:
        lines.append(
            f"flexlb_mock_grpc_error_count "
            f"{cluster_counters.get('grpc_error_count', 0)}"
        )
        lines.append(
            f"flexlb_mock_grpc_retry_count "
            f"{cluster_counters.get('grpc_retry_count', 0)}"
        )
        lines.append(
            f"flexlb_mock_grpc_cancel_forward_count "
            f"{cluster_counters.get('grpc_cancel_forward_count', 0)}"
        )

    return "\n".join(lines) + "\n"


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
        self._stopped: set[str] = set()
        self._grpc_channels: Dict[str, object] = {}
        # Cluster-level counters for remote decode diagnostics.
        self._grpc_error_count = 0
        self._grpc_retry_count = 0
        self._grpc_cancel_forward_count = 0

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
        self._by_grpc_addr[f"{state.host}:{state.grpc_port}"] = state
        self._by_grpc_addr[f"localhost:{state.grpc_port}"] = state
        self._by_grpc_addr[f"127.0.0.1:{state.grpc_port}"] = state
        return state

    def resolve_decode(self, input_pb) -> Optional[MockEngineState]:
        request_id = int(input_pb.request_id)
        for role_addr in input_pb.generate_config.role_addrs:
            if role_addr.role_type == self.pb2.ROLE_TYPE_DECODE:
                addr = f"{role_addr.ip}:{role_addr.grpc_port}"
                state = self._by_grpc_addr.get(addr)
                if state is not None:
                    logger.debug(
                        f"[DIAG] resolve_decode: rid={request_id} decode_addr={addr} "
                        f"found_in_process=True engine={state.name}"
                    )
                    return state
                logger.debug(
                    f"[DIAG] resolve_decode: rid={request_id} decode_addr={addr} "
                    f"found_in_process=False, returning None for remote routing"
                )
                return None
        decodes = [s for s in self.states if s.role == "decode"]
        if not decodes:
            logger.debug(
                f"[DIAG] resolve_decode: rid={request_id} no decode engines available, returning None"
            )
            return None
        selected = decodes[int(input_pb.request_id) % len(decodes)]
        logger.debug(
            f"[DIAG] resolve_decode: rid={request_id} "
            f"no_role_addrs_decode, round_robin_selected={selected.name}"
        )
        return selected

    async def _get_grpc_channel(self, target: str):
        """Return a cached gRPC channel for *target*, creating one if needed."""
        import grpc

        if target not in self._grpc_channels:
            self._grpc_channels[target] = grpc.aio.insecure_channel(
                target,
                options=[
                    ("grpc.max_send_message_length", 64 * 1024 * 1024),
                    ("grpc.max_receive_message_length", 64 * 1024 * 1024),
                ],
            )
            logger.debug(f"[DIAG] _get_grpc_channel: created new channel for {target}")
        return self._grpc_channels[target]

    async def stop_engine(self, name: str) -> bool:
        """Stop a single engine's gRPC server with grace=0 (simulates process kill).

        The MockEngineState object persists; only the gRPC server is stopped.
        Returns True if the engine was found and stopped (or already stopped).
        """
        for i, state in enumerate(self.states):
            if state.name == name:
                if name in self._stopped:
                    logger.info("Engine %s already stopped", name)
                    return True
                server = self._servers[i]
                await server.stop(grace=0)
                self._stopped.add(name)
                logger.info("Engine %s stopped (gRPC server killed)", name)
                return True
        return False

    async def restart_engine(self, name: str) -> bool:
        """Restart a stopped engine's gRPC server on the same port.

        Creates a new grpc.aio.server, rebinds to the original port, and starts it.
        The MockEngineState (counters, cache, etc.) persists across restart.
        Returns True if the engine was found and restarted (or already running).
        """
        import grpc

        for i, state in enumerate(self.states):
            if state.name == name:
                if name not in self._stopped:
                    logger.info("Engine %s already running", name)
                    return True
                server = grpc.aio.server(
                    options=[
                        ("grpc.max_send_message_length", 64 * 1024 * 1024),
                        ("grpc.max_receive_message_length", 64 * 1024 * 1024),
                    ]
                )
                servicer = MockRpcServicer(state, self.pb2)
                self.pb2_grpc.add_RpcServiceServicer_to_server(servicer, server)
                bound = server.add_insecure_port(f"{state.host}:{state.grpc_port}")
                if bound <= 0:
                    raise RuntimeError(
                        f"failed to rebind {state.host}:{state.grpc_port}"
                    )
                await server.start()
                self._servers[i] = server
                self._stopped.discard(name)
                logger.info(
                    "Engine %s restarted (gRPC server on %s:%d)",
                    name,
                    state.host,
                    state.grpc_port,
                )
                return True
        return False

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
        app.router.add_post("/stop_engine", self._http_stop_engine)
        app.router.add_post("/start_engine", self._http_start_engine)
        app.router.add_get("/metrics", self._http_metrics)
        self._http_runner = web.AppRunner(app)
        await self._http_runner.setup()
        site = web.TCPSite(self._http_runner, "0.0.0.0", port)
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                await site.start()
                break
            except OSError as exc:
                if attempt < max_retries:
                    logger.warning(
                        "failed to bind HTTP port %d (attempt %d/%d): %s; "
                        "retrying in 0.5s",
                        port,
                        attempt,
                        max_retries,
                        exc,
                    )
                    await asyncio.sleep(0.5)
                else:
                    raise RuntimeError(
                        f"failed to bind HTTP port {port} after "
                        f"{max_retries} attempts: {exc}"
                    ) from exc

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
        Body: {"engine": "prefill-0", "prefill_fixed_ms": 200.0, "decode_scale": 2.0, "max_prefill_concurrency": 2}
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
                if "max_prefill_concurrency" in data:
                    new_max = int(data["max_prefill_concurrency"])
                    state._max_prefill_concurrency = new_max
                    state._prefill_semaphore = asyncio.Semaphore(new_max)
                    state._status_version += 1
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

    async def _http_stop_engine(self, request) -> object:
        """POST /stop_engine — stop a single engine's gRPC server (simulates kill).
        Body: {"engine": "prefill-0"}
        """
        from aiohttp import web

        body = await request.json()
        engine_name = body.get("engine", "")
        if not self._find_engine(engine_name):
            return web.json_response(
                {"error": f"engine '{engine_name}' not found"}, status=404
            )
        await self.stop_engine(engine_name)
        return web.json_response(
            {"status": "ok", "engine": engine_name, "action": "stopped"}
        )

    async def _http_start_engine(self, request) -> object:
        """POST /start_engine — restart a stopped engine's gRPC server.
        Body: {"engine": "prefill-0"}
        """
        from aiohttp import web

        body = await request.json()
        engine_name = body.get("engine", "")
        if not self._find_engine(engine_name):
            return web.json_response(
                {"error": f"engine '{engine_name}' not found"}, status=404
            )
        await self.restart_engine(engine_name)
        return web.json_response(
            {"status": "ok", "engine": engine_name, "action": "started"}
        )

    @staticmethod
    def _escape_label_value(value: str) -> str:
        """Escape a label value for Prometheus exposition format."""
        return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")

    async def generate_prometheus_metrics(self) -> str:
        """Generate Prometheus exposition format metrics for all engines."""
        lines: list[str] = []
        metrics_meta = [
            ("mock_engine_up", "1 if engine is running, 0 if stopped", "gauge"),
            ("mock_engine_running", "current running requests", "gauge"),
            (
                "mock_engine_waiting",
                "current waiting requests (queued but not yet processing)",
                "gauge",
            ),
            ("mock_engine_accepted_total", "total accepted requests", "counter"),
            ("mock_engine_completed_total", "total completed requests", "counter"),
            ("mock_engine_cancelled_total", "total cancelled requests", "counter"),
            ("mock_engine_cache_keys", "number of cache keys", "gauge"),
            ("mock_engine_cache_evictions_total", "total cache evictions", "counter"),
            ("mock_engine_active_kv_tokens", "active KV cache tokens", "gauge"),
            ("mock_engine_available_kv_tokens", "available KV cache tokens", "gauge"),
            ("mock_engine_rpc_total", "total RPC calls by method", "counter"),
            (
                "mock_engine_prefill_ms_avg",
                "average prefill execution time in ms",
                "gauge",
            ),
            ("mock_engine_prefill_ms_p99", "p99 prefill execution time in ms", "gauge"),
            (
                "mock_engine_prefill_ms_count",
                "count of recent prefill completions",
                "gauge",
            ),
            (
                "mock_engine_decode_ms_avg",
                "average decode execution time in ms",
                "gauge",
            ),
            ("mock_engine_decode_ms_p99", "p99 decode execution time in ms", "gauge"),
            (
                "mock_engine_decode_ms_count",
                "count of recent decode completions",
                "gauge",
            ),
            (
                "flexlb_mock_grpc_error_count",
                "Total gRPC errors in remote decode",
                "counter",
            ),
            (
                "flexlb_mock_grpc_retry_count",
                "Total gRPC retries in remote decode",
                "counter",
            ),
            (
                "flexlb_mock_grpc_cancel_forward_count",
                "Total cancel forwarded to remote engines",
                "counter",
            ),
        ]
        for name, help_text, mtype in metrics_meta:
            lines.append(f"# HELP {name} {help_text}")
            lines.append(f"# TYPE {name} {mtype}")
        for state in self.states:
            snap = await state.snapshot()
            esc = self._escape_label_value
            labels = (
                f'engine_name="{esc(state.name)}",'
                f'role="{esc(state.role)}",'
                f'grpc_port="{state.grpc_port}",'
                f'engine_ip="{esc(state.host)}"'
            )
            is_up = 0 if state.name in self._stopped else 1
            lines.append(f"mock_engine_up{{{labels}}} {is_up}")
            lines.append(f'mock_engine_running{{{labels}}} {snap["running"]}')
            lines.append(f'mock_engine_waiting{{{labels}}} {snap["waiting"]}')
            lines.append(f'mock_engine_accepted_total{{{labels}}} {snap["accepted"]}')
            lines.append(f'mock_engine_completed_total{{{labels}}} {snap["completed"]}')
            lines.append(
                f'mock_engine_cancelled_total{{{labels}}} {snap["cancelled_count"]}'
            )
            lines.append(f'mock_engine_cache_keys{{{labels}}} {snap["cache_keys"]}')
            lines.append(
                f'mock_engine_cache_evictions_total{{{labels}}} {snap["cache_evictions"]}'
            )
            lines.append(
                f'mock_engine_active_kv_tokens{{{labels}}} {snap["active_kv_tokens"]}'
            )
            lines.append(
                f'mock_engine_available_kv_tokens{{{labels}}} {snap["available_kv_tokens"]}'
            )
            for rpc_method, count in snap.get("rpc_counts", {}).items():
                rpc_labels = f'{labels},rpc_method="{esc(rpc_method)}"'
                lines.append(f"mock_engine_rpc_total{{{rpc_labels}}} {count}")
            lines.append(
                f'mock_engine_prefill_ms_avg{{{labels}}} {snap["prefill_ms_avg"]:.1f}'
            )
            lines.append(
                f'mock_engine_prefill_ms_p99{{{labels}}} {snap["prefill_ms_p99"]:.1f}'
            )
            lines.append(
                f'mock_engine_prefill_ms_count{{{labels}}} {snap["prefill_ms_count"]}'
            )
            lines.append(
                f'mock_engine_decode_ms_avg{{{labels}}} {snap["decode_ms_avg"]:.1f}'
            )
            lines.append(
                f'mock_engine_decode_ms_p99{{{labels}}} {snap["decode_ms_p99"]:.1f}'
            )
            lines.append(
                f'mock_engine_decode_ms_count{{{labels}}} {snap["decode_ms_count"]}'
            )
        # Cluster-level metrics (not per-engine)
        lines.append(f"flexlb_mock_grpc_error_count {self._grpc_error_count}")
        lines.append(f"flexlb_mock_grpc_retry_count {self._grpc_retry_count}")
        lines.append(
            f"flexlb_mock_grpc_cancel_forward_count "
            f"{self._grpc_cancel_forward_count}"
        )
        return "\n".join(lines) + "\n"

    async def _http_metrics(self, request) -> object:
        """GET /metrics — Prometheus exposition format for all engines."""
        from aiohttp import web

        per_engine = request.query.get("per_engine", "").lower() == "true"
        if per_engine:
            metrics_text = await self.generate_prometheus_metrics()
        else:
            snap = await self.snapshot()
            metrics_text = generate_aggregated_prometheus_metrics(
                snap["engines"], snap.get("cluster_counters")
            )
        resp = web.Response(text=metrics_text)
        resp.headers["Content-Type"] = "text/plain; version=0.0.4; charset=utf-8"
        return resp

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
        for channel in self._grpc_channels.values():
            await channel.close()
        self._grpc_channels.clear()

    async def snapshot(self) -> dict:
        engines = []
        for state in self.states:
            snap = await state.snapshot()
            snap["stopped"] = state.name in self._stopped
            engines.append(snap)
        return {
            "engines": engines,
            "cluster_counters": {
                "grpc_error_count": self._grpc_error_count,
                "grpc_retry_count": self._grpc_retry_count,
                "grpc_cancel_forward_count": self._grpc_cancel_forward_count,
            },
        }

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
                        "protocol": "http",
                        "path": "/",
                    },
                    "decode_endpoint": {
                        "address": decode_domain,
                        "protocol": "http",
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
