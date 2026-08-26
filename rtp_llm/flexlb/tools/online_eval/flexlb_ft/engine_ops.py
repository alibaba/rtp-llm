"""gRPC engine operations for the functional test framework.

Translated from ``flexlb_smoke_base.py`` (async aiohttp/grpc.aio) to a
synchronous model: streams are consumed on daemon threads, snapshots are
polled from the calling thread.  Behaviour (timeouts, cancel semantics,
recovery verification) mirrors the legacy implementation exactly.

One deliberate fix over the legacy code: ``role_addr`` compares against the
proto ``role`` *string* ("PREFILL"/"DECODE").  The legacy
``scheduling_smoke.py`` passed the enum int ``ROLE_TYPE_PREFILL`` which never
matched the string field, so all prefill-address tracking silently returned
"" — the new framework passes strings, making the S-series distribution
assertions effective for the first time.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional

import grpc

from .harness import (
    encode_unique_key,
    ensure_proto_modules,
    ensure_schedule_proto_modules,
    http_get_json,
    http_post_json,
)

DEFAULT_INPUT_LEN = 2048
DEFAULT_OUTPUT_LEN = 10
RECOVERY_TIMEOUT_S = 30.0
STREAM_CANCEL_TIMEOUT_S = 5.0
FIRST_OUTPUT_TIMEOUT_S = 15.0

CHANNEL_OPTIONS = [
    ("grpc.max_receive_message_length", 64 * 1024 * 1024),
    ("grpc.max_send_message_length", 64 * 1024 * 1024),
]


@dataclass
class StreamSnapshot:
    """Collected state from a FetchResponse / GenerateStreamCall stream."""

    outputs: List[object] = field(default_factory=list)
    first_received: bool = False
    completed: bool = False
    error: Optional[str] = None
    terminated: bool = False
    terminated_s: Optional[float] = None  # monotonic time when stream ended


class StreamHandle:
    """A gRPC stream consumed on a background thread."""

    def __init__(self, call, snap: StreamSnapshot):
        self.call = call
        self.snap = snap
        self.thread = threading.Thread(target=self._consume, daemon=True)
        self.thread.start()

    def _consume(self) -> None:
        try:
            for output in self.call:
                if not self.snap.first_received:
                    self.snap.first_received = True
                self.snap.outputs.append(output)
                finished = output.flatten_output.finished
                if finished and any(finished):
                    self.snap.completed = True
        except grpc.RpcError as exc:
            # Client-side cancellation is not an error (mirrors legacy
            # asyncio.CancelledError handling).
            if exc.code() != grpc.StatusCode.CANCELLED:
                self.snap.error = repr(exc)
        except Exception as exc:
            self.snap.error = repr(exc)
        finally:
            self.snap.terminated = True
            self.snap.terminated_s = time.monotonic()

    def wait_first_output(self, timeout_s: float = FIRST_OUTPUT_TIMEOUT_S) -> bool:
        deadline = time.monotonic() + timeout_s
        while not self.snap.first_received and time.monotonic() < deadline:
            time.sleep(0.02)
        return self.snap.first_received

    def wait_end(self, timeout_s: float = STREAM_CANCEL_TIMEOUT_S) -> bool:
        self.thread.join(timeout_s)
        if self.thread.is_alive():
            self.cancel()
            self.thread.join(5.0)
            return False
        return True

    def cancel(self) -> None:
        try:
            self.call.cancel()
        except Exception:
            pass


class EngineOps:
    """Mock-engine HTTP control plane + master/worker gRPC client."""

    def __init__(
        self,
        master_ip: str,
        master_http_port: int,
        mock_http_port: int,
        deploy_mode: str = "batch",
    ):
        self.master_ip = master_ip
        self.master_http_port = master_http_port
        self.mock_http_port = mock_http_port
        self.deploy_mode = deploy_mode.lower()
        self.pb2, self.pb2_grpc = ensure_proto_modules()
        self.schedule_pb2, self.schedule_pb2_grpc = ensure_schedule_proto_modules()
        self._channels: dict = {}
        self._request_counter = 20000

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        for channel in self._channels.values():
            try:
                channel.close()
            except Exception:
                pass
        self._channels.clear()

    def _channel(self, target: str):
        if target not in self._channels:
            self._channels[target] = grpc.insecure_channel(
                target, options=CHANNEL_OPTIONS
            )
        return self._channels[target]

    def next_request_id(self, base: Optional[int] = None) -> int:
        if base is not None:
            self._request_counter = base
        self._request_counter += 1
        return self._request_counter

    # -- proto builders ----------------------------------------------------

    def build_generate_input(
        self,
        request_id: int,
        *,
        input_len: int = DEFAULT_INPUT_LEN,
        output_len: int = DEFAULT_OUTPUT_LEN,
        block_keys: Optional[List[int]] = None,
    ):
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

    def build_schedule_request(
        self,
        request_id: int,
        *,
        input_len: int = DEFAULT_INPUT_LEN,
        output_len: int = DEFAULT_OUTPUT_LEN,
        block_keys: Optional[List[int]] = None,
    ):
        input_pb = self.build_generate_input(
            request_id,
            input_len=input_len,
            output_len=output_len,
            block_keys=block_keys,
        )
        keys = block_keys or [request_id * 100 + 1]
        return self.schedule_pb2.FlexlbScheduleRequestPB(
            request_id=request_id,
            generate_input=input_pb.SerializeToString(),
            block_cache_keys=keys,
            seq_len=input_len,
            generate_timeout=30_000,
            request_time_ms=int(time.time() * 1000),
            max_new_tokens=max(1, output_len),
            num_beams=1,
            force_disable_sp_run=False,
            model="engine_service",
            api_key="",
            cache_key_block_size=1024,
        )

    # -- master gRPC -------------------------------------------------------

    def master_target(self) -> str:
        return f"{self.master_ip}:{self.master_http_port + 2}"

    def schedule(self, request_id: int, **kwargs):
        stub = self.schedule_pb2_grpc.FlexlbServiceStub(
            self._channel(self.master_target())
        )
        req = self.build_schedule_request(request_id, **kwargs)
        return stub.Schedule(req, timeout=30.0)

    def role_addr(self, response, role: str) -> str:
        """Address of the first server_status entry whose role matches.

        ``role`` must be the proto string ("PREFILL"/"DECODE"/"PDFUSION").
        """
        for status in response.server_status:
            if status.role == role and status.server_ip:
                return f"{status.server_ip}:{status.grpc_port}"
        return ""

    def prefill_addr(self, response) -> str:
        return self.role_addr(response, "PREFILL") or self.role_addr(
            response, "PDFUSION"
        )

    # -- dual-path stream ---------------------------------------------------

    def _copy_role_addrs(self, input_pb, response) -> None:
        del input_pb.generate_config.role_addrs[:]
        for status in response.server_status:
            input_pb.generate_config.role_addrs.add(
                role=status.role,
                role_type=getattr(
                    self.pb2, f"ROLE_TYPE_{status.role}", self.pb2.ROLE_TYPE_PDFUSION
                ),
                ip=status.server_ip,
                http_port=status.http_port,
                grpc_port=status.grpc_port,
            )

    def start_stream(self, response, request_id: int, input_pb=None) -> StreamHandle:
        """Start FetchResponse (batch) or GenerateStreamCall (direct/queue)."""
        target = self.prefill_addr(response)
        if not target:
            raise RuntimeError("schedule response has no PREFILL/PDFUSION address")
        stub = self.pb2_grpc.RpcServiceStub(self._channel(target))
        if response.enqueued_by_master:
            call = stub.FetchResponse(
                self.pb2.FetchRequestPB(request_id=request_id), timeout=60.0
            )
        else:
            if input_pb is None:
                input_pb = self.build_generate_input(request_id)
            self._copy_role_addrs(input_pb, response)
            call = stub.GenerateStreamCall(input_pb, timeout=60.0)
        return StreamHandle(call, StreamSnapshot())

    # -- cancel -------------------------------------------------------------

    def cancel(self, request_id: int, response=None) -> None:
        """Cancel via master (always) + worker (direct/queue path only)."""
        stub = self.schedule_pb2_grpc.FlexlbServiceStub(
            self._channel(self.master_target())
        )
        cancel_request = self.schedule_pb2.FlexlbCancelRequestPB(
            request_id=request_id,
            reason=self.schedule_pb2.CANCEL_REASON_CLIENT_CANCELLED,
        )
        if response is not None and response.HasField("lifecycle"):
            lifecycle = response.lifecycle
            if lifecycle.batch_id:
                cancel_request.batch_id = lifecycle.batch_id
        stub.Cancel(cancel_request, timeout=10.0)
        if response is not None and not response.enqueued_by_master:
            self.worker_cancel(request_id, response)

    def worker_cancel(self, request_id: int, response) -> None:
        target = self.prefill_addr(response)
        if not target:
            return
        stub = self.pb2_grpc.RpcServiceStub(self._channel(target))
        stub.Cancel(self.pb2.CancelRequestPB(request_id=request_id), timeout=10.0)

    # -- recovery -----------------------------------------------------------

    def verify_recovery(self) -> tuple[bool, str]:
        """Schedule a fresh request and confirm it completes normally."""
        rid = self.next_request_id()
        try:
            response = self.schedule(rid, output_len=2, block_keys=[rid * 100 + 1])
            if response.code != 200 or not response.success:
                return False, f"schedule failed: {response.error_message}"
            input_pb = (
                None if response.enqueued_by_master else self.build_generate_input(rid)
            )
            handle = self.start_stream(response, rid, input_pb=input_pb)
            handle.thread.join(RECOVERY_TIMEOUT_S)
            snap = handle.snap
            if snap.error:
                return False, f"stream error: {snap.error}"
            if not snap.completed:
                return False, "recovery request did not complete"
            return True, f"ok (outputs={len(snap.outputs)})"
        except Exception as exc:
            return False, f"exception: {exc!r}"

    # -- mock HTTP control plane -------------------------------------------

    def snapshot(self) -> dict:
        data = http_get_json(f"http://127.0.0.1:{self.mock_http_port}/snapshot")
        if data is None:
            raise RuntimeError(
                f"snapshot failed on mock http port {self.mock_http_port}"
            )
        return data

    def snapshot_by_name(self) -> dict:
        snap = self.snapshot()
        return {e["name"]: e for e in snap.get("engines", [])}

    def addr_to_name(self) -> dict:
        snap = self.snapshot()
        return {e["grpc_addr"]: e["name"] for e in snap.get("engines", [])}

    def requests_data(self) -> dict:
        data = http_get_json(f"http://127.0.0.1:{self.mock_http_port}/requests")
        return data or {}

    def engine_health(self) -> dict:
        data = http_get_json(f"http://127.0.0.1:{self.mock_http_port}/health")
        return data or {}

    def inject(self, engine_name: str, config: dict) -> dict:
        status, body = http_post_json(
            f"http://127.0.0.1:{self.mock_http_port}/inject",
            {"engine": engine_name, "config": config},
        )
        if status != 200:
            raise RuntimeError(f"inject({engine_name}) failed: {status} {body}")
        return body or {}

    def clear_inject(self, engine_name: str) -> dict:
        status, body = http_post_json(
            f"http://127.0.0.1:{self.mock_http_port}/clear_inject",
            {"engine": engine_name},
        )
        if status != 200:
            raise RuntimeError(f"clear_inject({engine_name}) failed: {status} {body}")
        return body or {}

    def set_perf(self, engine_name: str, **kwargs) -> bool:
        status, _ = http_post_json(
            f"http://127.0.0.1:{self.mock_http_port}/set_perf",
            {"engine": engine_name, **kwargs},
        )
        return status == 200

    def set_kv_pressure(self, engine_name: str, active_kv_tokens: int) -> bool:
        status, _ = http_post_json(
            f"http://127.0.0.1:{self.mock_http_port}/set_kv_pressure",
            {"engine": engine_name, "active_kv_tokens": active_kv_tokens},
        )
        return status == 200

    def set_queue_depth(self, engine_name: str, queue_depth: int) -> bool:
        """Java mock: sets FaultInjectionConfig.queueDepthLimit — a *real*
        enqueue rejection gate (``pendingRequests >= limit``), not the legacy
        Python fake display value."""
        status, _ = http_post_json(
            f"http://127.0.0.1:{self.mock_http_port}/set_queue_depth",
            {"engine": engine_name, "queue_depth": queue_depth},
        )
        return status == 200

    def stop_engine(self, engine_name: str) -> dict:
        status, body = http_post_json(
            f"http://127.0.0.1:{self.mock_http_port}/stop_engine",
            {"engine": engine_name},
        )
        if status != 200:
            raise RuntimeError(f"stop_engine({engine_name}) failed: {status} {body}")
        return body or {}

    def start_engine(self, engine_name: str) -> dict:
        status, body = http_post_json(
            f"http://127.0.0.1:{self.mock_http_port}/start_engine",
            {"engine": engine_name},
        )
        if status != 200:
            raise RuntimeError(f"start_engine({engine_name}) failed: {status} {body}")
        return body or {}

    def add_engine(self, role: str, grpc_port: int) -> dict:
        status, body = http_post_json(
            f"http://127.0.0.1:{self.mock_http_port}/add_engine",
            {"role": role, "grpc_port": grpc_port},
        )
        if status != 200:
            raise RuntimeError(f"add_engine failed: {status} {body}")
        return body or {}

    def remove_engine(self, engine_name: str) -> dict:
        status, body = http_post_json(
            f"http://127.0.0.1:{self.mock_http_port}/remove_engine",
            {"engine": engine_name},
        )
        if status != 200:
            raise RuntimeError(f"remove_engine failed: {status} {body}")
        return body or {}

    # -- engine verification helpers ---------------------------------------

    def verify_engine_received(self, rid: int, method: str) -> tuple[bool, str]:
        snap = self.snapshot()
        for engine in snap.get("engines", []):
            lifecycle = engine.get("request_lifecycle", {})
            if str(rid) in lifecycle:
                lc = lifecycle[str(rid)]
                if lc.get("method") == method:
                    return True, f"engine={engine['name']} method={method}"
        return False, f"rid={rid} method={method} not found in any engine"

    def verify_engine_cancelled(self, rid: int) -> tuple[bool, str]:
        snap = self.snapshot()
        for engine in snap.get("engines", []):
            if rid in engine.get("cancelled_rids", []):
                return True, f"engine={engine['name']}"
            lifecycle = engine.get("request_lifecycle", {})
            if (
                str(rid) in lifecycle
                and lifecycle[str(rid)].get("end_state") == "cancelled"
            ):
                return True, f"engine={engine['name']}"
        return False, f"rid={rid} not cancelled in any engine"

    def master_inflight(self) -> Optional[dict]:
        return http_get_json(
            f"http://127.0.0.1:{self.master_http_port}/rtp_llm/inflight_status",
            timeout=5,
        )

    # -- composite request helper ------------------------------------------

    def run_one_request(
        self, rid: int, stream_timeout_s: float = 15.0, **kwargs
    ) -> tuple[str, Optional[str]]:
        """Schedule → stream → consume to completion.

        Returns (prefill_addr, error) — error is None on success.
        """
        try:
            response = self.schedule(rid, **kwargs)
            if response.code != 200 or not response.success:
                return "", f"schedule failed: {response.error_message}"
            addr = self.role_addr(response, "PREFILL")
            input_pb = (
                self.build_generate_input(rid)
                if not response.enqueued_by_master
                else None
            )
            handle = self.start_stream(response, rid, input_pb=input_pb)
            handle.wait_end(stream_timeout_s)
            snap = handle.snap
            if snap.error:
                return addr, snap.error
            if not snap.completed:
                return addr, "stream did not complete"
            return addr, None
        except Exception as exc:
            return "", repr(exc)

    def inflight_count_for_port(self, grpc_port: int) -> int:
        """Master-side prefill inflight_batches for the endpoint at grpc_port."""
        data = self.master_inflight()
        if data is None:
            return -1
        for ep in data.get("prefill_endpoints", []) or []:
            ip_port = ep.get("ip_port", "")
            if ip_port.endswith(f":{grpc_port}"):
                batches = ep.get("inflight_batches", 0)
                return len(batches) if isinstance(batches, list) else int(batches)
        return -1

    def mock_engine_field(self, engine_name: str, field_name: str, default=-1):
        try:
            engines = self.snapshot().get("engines", [])
            for engine in engines:
                if engine.get("name") == engine_name:
                    return engine.get(field_name, default)
        except Exception:
            pass
        return default
