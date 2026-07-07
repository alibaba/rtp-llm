"""Shared base class for FlexLB smoke tests.

Provides reusable gRPC infrastructure (channel management, proto building,
stream consumption, cancel, recovery verification, and HTTP mock-engine
API access) shared by ``cancel_smoke.py`` and ``scheduling_smoke.py``.
"""

from __future__ import annotations

import argparse
import asyncio
import time
from dataclasses import dataclass, field
from typing import List, Optional

from online_eval.mock_engine import encode_unique_key
from online_eval.proto_utils import ensure_proto_modules

# ---------------------------------------------------------------------------
# Result helpers
# ---------------------------------------------------------------------------


@dataclass
class ScenarioResult:
    name: str
    passed: bool
    detail: str = ""
    duration_s: float = 0.0


@dataclass
class StreamSnapshot:
    """Collected state from a FetchResponse / GenerateStreamCall stream."""

    outputs: List[object] = field(default_factory=list)
    first_received: bool = False
    completed: bool = False
    error: Optional[str] = None
    terminated_s: Optional[float] = None  # monotonic time when stream ended


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class FlexLBSmokeBase:
    """Reusable gRPC + HTTP infrastructure for FlexLB smoke tests."""

    DEFAULT_INPUT_LEN = 2048
    DEFAULT_OUTPUT_LEN = 10
    RECOVERY_TIMEOUT_S = 30.0
    STREAM_CANCEL_TIMEOUT_S = 5.0
    FIRST_OUTPUT_TIMEOUT_S = 15.0

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.pb2, self.pb2_grpc = ensure_proto_modules()
        self._channels: dict[str, object] = {}
        self._request_counter = args.request_id_base
        self.results: List[ScenarioResult] = []

    # -- gRPC channel management ------------------------------------------

    async def close(self) -> None:
        import grpc

        for channel in self._channels.values():
            await channel.close()
        self._channels.clear()

    async def _channel(self, target: str):
        import grpc

        if target not in self._channels:
            self._channels[target] = grpc.aio.insecure_channel(
                target,
                options=[
                    ("grpc.max_receive_message_length", 64 * 1024 * 1024),
                    ("grpc.max_send_message_length", 64 * 1024 * 1024),
                ],
            )
        return self._channels[target]

    def _next_request_id(self) -> int:
        self._request_counter += 1
        return self._request_counter

    def _master_target(self) -> str:
        return f"{self.args.master_ip}:{self.args.master_http_port + 2}"

    # -- Proto builders ---------------------------------------------------

    def _build_generate_input(
        self,
        request_id: int,
        *,
        input_len: int = DEFAULT_INPUT_LEN,
        output_len: int = DEFAULT_OUTPUT_LEN,
        block_keys: Optional[List[int]] = None,
    ):
        """Build a standalone ``GenerateInputPB`` (no schedule wrapper)."""
        meta = {
            "rid": str(request_id),
            "trace_id": f"cancel_smoke_{request_id}",
            "input_len": input_len,
            "output_len": output_len,
            "block_cache_keys": block_keys or [request_id * 100 + 1],
        }
        config = self.pb2.GenerateConfigPB(
            max_new_tokens=max(1, output_len),
            num_return_sequences=1,
            top_p=1.0,
            top_k=0,
            temperature=1.0,
            return_incremental=True,
            is_streaming=True,
            timeout_ms=30_000,
            unique_key=encode_unique_key(meta),
        )
        info = self.pb2.RequestInfoPB(
            request_id=str(request_id),
            trace_id=f"cancel_smoke_{request_id}",
            source_role="cancel_smoke",
        )
        return self.pb2.GenerateInputPB(
            request_id=request_id,
            token_ids=[0] * min(input_len, 4096),
            generate_config=config,
            client_id="cancel_smoke",
            start_time=int(time.time() * 1000),
            request_info=info,
        )

    def _build_schedule_request(
        self,
        request_id: int,
        *,
        input_len: int = DEFAULT_INPUT_LEN,
        output_len: int = DEFAULT_OUTPUT_LEN,
        block_keys: Optional[List[int]] = None,
        schedule_mode: str = "batch",
    ):
        input_pb = self._build_generate_input(
            request_id,
            input_len=input_len,
            output_len=output_len,
            block_keys=block_keys,
        )
        mode_pb = {
            "auto": self.pb2.FLEXLB_SCHEDULE_AUTO,
            "batch": self.pb2.FLEXLB_SCHEDULE_BATCH,
            "direct": self.pb2.FLEXLB_SCHEDULE_DIRECT,
            "queue": self.pb2.FLEXLB_SCHEDULE_QUEUE,
        }[schedule_mode]
        keys = block_keys or [request_id * 100 + 1]
        return self.pb2.FlexlbScheduleRequestPB(
            request_id=request_id,
            generate_input=input_pb,
            block_cache_keys=keys,
            seq_len=input_len,
            generate_timeout=30_000,
            request_time_ms=int(time.time() * 1000),
            max_new_tokens=max(1, output_len),
            num_beams=1,
            force_disable_sp_run=False,
            model="engine_service",
            api_key="",
            schedule_mode=mode_pb,
            cache_key_block_size=1024,
        )

    # -- Master gRPC helpers ----------------------------------------------

    async def _schedule(self, request_id: int, **kwargs):
        """Call ``FlexlbService.Schedule`` and return the response."""
        stub = self.pb2_grpc.FlexlbServiceStub(
            await self._channel(self._master_target())
        )
        req = self._build_schedule_request(request_id, **kwargs)
        return await stub.Schedule(req, timeout=30.0)

    def _role_addr(self, response, role: int) -> str:
        for status in response.server_status:
            if status.role_type == role and status.server_ip:
                return f"{status.server_ip}:{status.grpc_port}"
        return ""

    # -- Dual-path stream helpers -----------------------------------------

    def _copy_role_addrs(self, input_pb, response) -> None:
        """Copy ``server_status`` role addrs into ``input_pb.generate_config``."""
        del input_pb.generate_config.role_addrs[:]
        for status in response.server_status:
            input_pb.generate_config.role_addrs.add(
                role=status.role,
                role_type=status.role_type,
                ip=status.server_ip,
                http_port=status.http_port,
                grpc_port=status.grpc_port,
            )

    async def _start_stream(self, response, request_id: int, input_pb=None):
        """Start FetchResponse (batch) or GenerateStreamCall (direct/queue)."""
        target = self._role_addr(
            response, self.pb2.ROLE_TYPE_PREFILL
        ) or self._role_addr(response, self.pb2.ROLE_TYPE_PDFUSION)
        if not target:
            raise RuntimeError("schedule response has no PREFILL/PDFUSION address")
        stub = self.pb2_grpc.RpcServiceStub(await self._channel(target))
        if response.enqueued_by_master:
            return stub.FetchResponse(
                self.pb2.FetchRequestPB(request_id=request_id),
                timeout=60.0,
            )
        if input_pb is None:
            input_pb = self._build_generate_input(request_id)
        self._copy_role_addrs(input_pb, response)
        return stub.GenerateStreamCall(input_pb, timeout=60.0)

    # -- Stream consumption -----------------------------------------------

    async def _consume_stream(self, stream, snap: StreamSnapshot) -> None:
        """Consume a stream into *snap*."""
        try:
            async for output in stream:
                if not snap.first_received:
                    snap.first_received = True
                snap.outputs.append(output)
                if output.flatten_output.finished and any(
                    output.flatten_output.finished
                ):
                    snap.completed = True
        except asyncio.CancelledError:
            # Task cancelled externally (e.g. cancel test) — not an error
            pass
        except Exception as exc:
            snap.error = repr(exc)
        finally:
            snap.terminated_s = time.monotonic()

    async def _wait_for_first_output(
        self, snap: StreamSnapshot, timeout_s: float = FIRST_OUTPUT_TIMEOUT_S
    ) -> bool:
        deadline = time.monotonic() + timeout_s
        while not snap.first_received and time.monotonic() < deadline:
            await asyncio.sleep(0.02)
        return snap.first_received

    async def _wait_for_stream_end(
        self,
        task: asyncio.Task,
        timeout_s: float = STREAM_CANCEL_TIMEOUT_S,
    ) -> bool:
        try:
            await asyncio.wait_for(task, timeout=timeout_s)
            return True
        except asyncio.TimeoutError:
            task.cancel()
            try:
                await task
            except (Exception, asyncio.CancelledError):
                pass
            return False

    # -- Cancel (dual-path) -----------------------------------------------

    async def _cancel(self, request_id: int, response=None) -> None:
        """Cancel via Master (always) + Worker (direct/queue path only)."""
        stub = self.pb2_grpc.FlexlbServiceStub(
            await self._channel(self._master_target())
        )
        await stub.Cancel(self.pb2.CancelRequestPB(request_id=request_id), timeout=10.0)
        if response is not None and not response.enqueued_by_master:
            await self._worker_cancel(request_id, response)

    async def _worker_cancel(self, request_id: int, response) -> None:
        """Call Worker ``RpcService.Cancel`` directly."""
        target = self._role_addr(
            response, self.pb2.ROLE_TYPE_PREFILL
        ) or self._role_addr(response, self.pb2.ROLE_TYPE_PDFUSION)
        if not target:
            return
        stub = self.pb2_grpc.RpcServiceStub(await self._channel(target))
        await stub.Cancel(self.pb2.CancelRequestPB(request_id=request_id), timeout=10.0)

    # -- Recovery verification --------------------------------------------

    async def _verify_recovery(self) -> tuple[bool, str]:
        """Schedule a fresh request and confirm it completes normally."""
        rid = self._next_request_id()
        try:
            response = await self._schedule(
                rid,
                output_len=2,
                block_keys=[rid * 100 + 1],
                schedule_mode=getattr(self.args, "schedule_mode", "batch"),
            )
            if response.code != 200 or not response.success:
                return False, f"schedule failed: {response.error_message}"
            input_pb = (
                self._build_generate_input(rid)
                if not response.enqueued_by_master
                else None
            )
            stream = await self._start_stream(response, rid, input_pb=input_pb)
            snap = StreamSnapshot()
            await self._consume_stream(stream, snap)
            if snap.error:
                return False, f"stream error: {snap.error}"
            if not snap.completed:
                return False, "recovery request did not complete"
            return True, f"ok (outputs={len(snap.outputs)})"
        except Exception as exc:
            return False, f"exception: {exc!r}"

    # -- HTTP mock-engine API ---------------------------------------------

    async def _get_snapshot(self):
        """Get mock engine cluster snapshot via HTTP."""
        import aiohttp

        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"http://127.0.0.1:{self.args.mock_http_port}/snapshot"
            ) as resp:
                return await resp.json()

    async def _inject(self, engine_name: str, config: dict):
        """Inject error/timeout to a specific engine."""
        import aiohttp

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"http://127.0.0.1:{self.args.mock_http_port}/inject",
                json={"engine": engine_name, "config": config},
            ) as resp:
                return await resp.json()

    async def _clear_inject(self, engine_name: str):
        """Clear injection for a specific engine."""
        import aiohttp

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"http://127.0.0.1:{self.args.mock_http_port}/clear_inject",
                json={"engine": engine_name},
            ) as resp:
                return await resp.json()

    # -- Engine verification helpers -------------------------------------

    async def _verify_engine_received(self, rid: int, method: str) -> tuple:
        """Verify an engine received the specified RPC method for a request.

        Returns (success, detail_str)
        """
        snap = await self._get_snapshot()
        for engine in snap.get("engines", []):
            rpc_counts = engine.get("rpc_counts", {})
            lifecycle = engine.get("request_lifecycle", {})
            if str(rid) in lifecycle:
                lc = lifecycle[str(rid)]
                if lc.get("method") == method:
                    return True, f"engine={engine['name']} method={method}"
            if rpc_counts.get(method, 0) > 0 and str(rid) in {k for k in lifecycle}:
                return True, f"engine={engine['name']} method={method}"
        return False, f"rid={rid} method={method} not found in any engine"

    async def _verify_engine_cancelled(self, rid: int) -> tuple:
        """Verify an engine recorded a cancel operation for the request."""
        snap = await self._get_snapshot()
        for engine in snap.get("engines", []):
            cancelled_rids = engine.get("cancelled_rids", [])
            if rid in cancelled_rids:
                return True, f"engine={engine['name']}"
            lifecycle = engine.get("request_lifecycle", {})
            if (
                str(rid) in lifecycle
                and lifecycle[str(rid)].get("end_state") == "cancelled"
            ):
                return True, f"engine={engine['name']}"
        return False, f"rid={rid} not cancelled in any engine"

    async def _set_perf(self, engine_name: str, **kwargs) -> bool:
        """Modify performance parameters for a specific engine."""
        import aiohttp

        body = {"engine": engine_name, **kwargs}
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"http://127.0.0.1:{self.args.mock_http_port}/set_perf", json=body
            ) as resp:
                return resp.status == 200

    async def _set_kv_pressure(self, engine_name: str, active_kv_tokens: int) -> bool:
        """Set KV pressure on a decode engine."""
        import aiohttp

        body = {"engine": engine_name, "active_kv_tokens": active_kv_tokens}
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"http://127.0.0.1:{self.args.mock_http_port}/set_kv_pressure",
                json=body,
            ) as resp:
                return resp.status == 200

    async def _set_queue_depth(self, engine_name: str, queue_depth: int) -> bool:
        """Set reported queue depth on a prefill engine."""
        import aiohttp

        body = {"engine": engine_name, "queue_depth": queue_depth}
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"http://127.0.0.1:{self.args.mock_http_port}/set_queue_depth",
                json=body,
            ) as resp:
                return resp.status == 200

    async def _verify_inflight_clean(self, timeout_s: float = 10.0) -> tuple:
        """Verify master-side inflight is all zero (batch path only).

        Returns (success, detail_str)
        """
        import time

        import aiohttp

        detail = "no response yet"
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"http://127.0.0.1:{self.args.flexlb_http_port}/rtp_llm/inflight_status"
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            sched = data.get("scheduler_inflight", 0)
                            prefill_eps = data.get("prefill_endpoints", [])
                            decode_eps = data.get("decode_endpoints", [])
                            prefill_clean = all(
                                ep.get("inflight_batches", 0) == 0 for ep in prefill_eps
                            )
                            decode_clean = all(
                                ep.get("inflight_requests", 0) == 0 for ep in decode_eps
                            )
                            if sched == 0 and prefill_clean and decode_clean:
                                return True, "all inflight zero"
                            detail = (
                                f"scheduler={sched}, "
                                f"prefill={[{'ep': ep.get('ip_port'), 'batches': ep.get('inflight_batches', 0)} for ep in prefill_eps]}, "
                                f"decode={[{'ep': ep.get('ip_port'), 'reqs': ep.get('inflight_requests', 0)} for ep in decode_eps]}"
                            )
                            await asyncio.sleep(0.5)
                            continue
            except Exception:
                await asyncio.sleep(0.5)
        return False, f"timeout waiting for inflight clean: {detail}"
