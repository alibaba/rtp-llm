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

import json
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
    wait_for,
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
    # Monotonic time when the FIRST output arrived — the client-observed
    # TTFT anchor for graded property P7 (see balance_overload_avoid_prefill;
    # under BATCH dispatch the first FetchResponse message only surfaces
    # after decode completes, so P7 uses the completion-duration口径 there).
    first_received_s: Optional[float] = None


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
                    self.snap.first_received_s = time.monotonic()
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
    ):
        self.master_ip = master_ip
        self.master_http_port = master_http_port
        self.mock_http_port = mock_http_port
        self.pb2, self.pb2_grpc = ensure_proto_modules()
        self.schedule_pb2, self.schedule_pb2_grpc = ensure_schedule_proto_modules()
        self._channels: dict = {}
        self._request_counter = 20000
        self._rid_lock = threading.Lock()

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
        # Only raise the counter when base exceeds it: repeated calls with the
        # same base (multi-request cases passing base each time) must yield
        # distinct ids, not restart from the same value.  Locked — concurrent
        # callers (background-flow threads, ThreadPoolExecutor bursts) share
        # one EngineOps instance.
        with self._rid_lock:
            if base is not None and base > self._request_counter:
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

    def schedule(self, request_id: int, timeout_s: float = 30.0, **kwargs):
        """Schedule RPC against the master.

        ``timeout_s`` is the *client-side gRPC deadline* — the v2 QUEUE
        scheduler parks capacity-blocked requests (a wait condition, see
        FixedWindowBatcherAlgorithm), so callers probing for parking pass a
        short deadline and expect DEADLINE_EXCEEDED.
        """
        stub = self.schedule_pb2_grpc.FlexlbServiceStub(
            self._channel(self.master_target())
        )
        req = self.build_schedule_request(request_id, **kwargs)
        return stub.Schedule(req, timeout=timeout_s)

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
                ip=status.server_ip,
                http_port=status.http_port,
                grpc_port=status.grpc_port,
            )

    def start_stream(self, response, request_id: int, input_pb=None) -> StreamHandle:
        """Start FetchResponse (BATCH dispatch, enqueued_by_master) or
        GenerateStreamCall (NON_BATCH, frontend-sent) — decided per response
        from the master's enqueued_by_master flag."""
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
                # Default-shape fallback: callers that scheduled with a
                # non-default shape MUST pass input_pb (see run_one_request).
                input_pb = self.build_generate_input(request_id)
            self._copy_role_addrs(input_pb, response)
            call = stub.GenerateStreamCall(input_pb, timeout=60.0)
        return StreamHandle(call, StreamSnapshot())

    # -- cancel -------------------------------------------------------------

    def cancel(self, request_id: int, response=None) -> None:
        """Cancel via master (always) + worker (NON_BATCH/frontend path only)."""
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

    def verify_recovery(self, output_len: int = 2) -> tuple[bool, str]:
        """Schedule a fresh request and confirm it completes normally."""
        rid = self.next_request_id()
        block_keys = [rid * 100 + 1]
        try:
            response = self.schedule(rid, output_len=output_len, block_keys=block_keys)
            if response.code != 200 or not response.success:
                return False, f"schedule failed: {response.error_message}"
            # NON_BATCH re-builds the GenerateInputPB client-side; it must
            # carry the SAME shape/block-keys the ScheduleRequest carried,
            # otherwise the engine sees a different request than the master
            # scheduled (input_len/block cache keys diverge).
            input_pb = (
                None
                if response.enqueued_by_master
                else self.build_generate_input(
                    rid, output_len=output_len, block_keys=block_keys
                )
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

    def add_engine(
        self, role: str, port: Optional[int] = None
    ) -> tuple[int, Optional[dict]]:
        """POST /add_engine {"role": ..., "port": optional} — dynamic scale-out.

        The Java mock's field name is ``port`` (gRPC port; auto-allocated as
        current max + 1 when omitted).  Returns (status, body) WITHOUT raising:
        200 → body carries ``engine`` (name) + ``port`` (gRPC) + ``http_port``;
        409 port-in-use / 400 bad role / 501 (cluster started without
        --discovery-file) are surfaced to the caller (chaos cases exercise
        concurrent add/remove and treat those as expected outcomes).
        """
        body: dict = {"role": role}
        if port is not None:
            body["port"] = port
        return http_post_json(
            f"http://127.0.0.1:{self.mock_http_port}/add_engine", body
        )

    def remove_engine(
        self, engine_name: Optional[str] = None, port: Optional[int] = None
    ) -> tuple[int, Optional[dict]]:
        """POST /remove_engine {"engine": name} or {"port": grpcPort}.

        Permanently detaches the engine (stop semantics + removal from the
        services map and the discovery file).  Returns (status, body) without
        raising — 404 (unknown engine) is an expected outcome under concurrent
        add/remove racing.
        """
        body: dict = {}
        if engine_name:
            body["engine"] = engine_name
        if port is not None:
            body["port"] = port
        if not body:
            raise ValueError("remove_engine needs engine_name or port")
        return http_post_json(
            f"http://127.0.0.1:{self.mock_http_port}/remove_engine", body
        )

    def master_info(self) -> Optional[dict]:
        """POST /rtp_llm/master/info {} → response payload (None on failure)."""
        status, data = http_post_json(
            f"http://127.0.0.1:{self.master_http_port}/rtp_llm/master/info",
            {},
            timeout=5,
        )
        return data if status == 200 else None

    def master_alive_count(self, role: str) -> int:
        """Alive worker count for "PREFILL"/"DECODE" from master info (-1 unknown)."""
        data = self.master_info()
        if not data:
            return -1
        summary = data.get("worker_summary", {}) or {}
        entry = summary.get(role, {}) or {}
        try:
            return int(entry.get("alive", -1))
        except (TypeError, ValueError):
            return -1

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

    def master_scheduler_inflight(self) -> int:
        """Global scheduler inflight request count (-1 on endpoint failure).

        Unlike the per-endpoint view, this survives engine eviction: when a
        dead engine is 3-strike-evicted its endpoint row disappears from
        ``prefill_endpoints`` (per-endpoint lookups return -1) while the
        scheduler-level inflight bookkeeping lingers until the stale-inflight
        TTL / eviction cleanup drains it — which is exactly what the TTL
        cleanup cases need to observe.
        """
        data = self.master_inflight()
        if data is None:
            return -1
        try:
            return int(data.get("scheduler_inflight", -1))
        except (TypeError, ValueError):
            return -1

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
            # NON_BATCH re-builds the GenerateInputPB client-side with the
            # SAME shape/block-keys kwargs the ScheduleRequest carried — a
            # default-shape rebuild would desynchronize the engine's view
            # (admitting default block keys) from the master's routing view.
            input_pb = (
                self.build_generate_input(rid, **kwargs)
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


# ===========================================================================
# Shared control-plane helpers (suite-reorg task #85)
#
# Cross-category helpers that used to live in injection_gate_cases.py /
# status_fault_cases.py (the latter held its own copy while the mock control
# server learned the status-fault types).  They operate purely on the mock
# HTTP control plane and engine snapshots, so engine_ops.py is their shared
# home — every flexlb_ft/cases/ category module imports them from here.
# ===========================================================================


def inject_type(
    ops: "EngineOps", engine_name: str, fault_type: str, enabled: bool = True, **params
) -> dict:
    """POST /inject with the ORIGINAL Java "type" format (MERGE semantics,
    supports all fault types plus their parameters).

    The EngineOps.inject() method uses the Python "config" format which
    REPLACES the whole config and only knows the four boolean flags, so it
    cannot express kv_pressure / queue_depth / crash_after / delays / the
    status_* family.
    """
    payload = {"engine": engine_name, "type": fault_type, "enabled": enabled}
    payload.update(params)
    status, body = http_post_json(
        f"http://127.0.0.1:{ops.mock_http_port}/inject", payload
    )
    if status != 200:
        raise RuntimeError(
            f"inject_type({engine_name}, {fault_type}, {params}) "
            f"failed: {status} {body}"
        )
    return body or {}


def inject_type_all(ops: "EngineOps", names: list, fault_type: str, **params) -> None:
    for name in names:
        inject_type(ops, name, fault_type, **params)


def clear_type_all(ops: "EngineOps", names: list, fault_type: str) -> None:
    for name in names:
        try:
            inject_type(ops, name, fault_type, enabled=False)
        except Exception:
            pass


def engine_inflight_clean(
    ops: "EngineOps", names: list, timeout_s: float = 10.0
) -> tuple:
    """Engine-side leak check: every named engine reports inflight == 0 and
    leak_detected == false in /snapshot."""

    def clean() -> bool:
        snap = ops.snapshot_by_name()
        return all(
            snap.get(n, {}).get("inflight", 0) == 0
            and not snap.get(n, {}).get("leak_detected", False)
            for n in names
        )

    ok = wait_for(clean, timeout_s, 0.5)
    snap = ops.snapshot_by_name()
    detail = {
        n: (
            snap.get(n, {}).get("inflight", -1),
            snap.get(n, {}).get("leak_detected", None),
        )
        for n in names
    }
    return ok, f"{json.dumps(detail, sort_keys=True)}"


def _fence_residue_stable(
    ops: "EngineOps", max_residue: int, settle_s: float = 20.0
) -> tuple:
    """Cross-process contract for an empty-ack (uncertain) enqueue batch.

    The master installs a BATCH_ACK_UNCERTAIN engine fence
    (PriorityScheduler.fenceEntryForUncertainBatchDelivery).  In the
    cross-process production wiring the cancel channel is
    UnsupportedEngineCancelChannel, whose UNSUPPORTED ack is NOT a safe
    release fact (handleEngineFenceOutcome groups it with FAILED /
    NOT_FOUND), so the entry parks in the 60s quarantined-fence sweep
    indefinitely; cleanupInflight explicitly skips engineFence entries
    from the stale TTL.  A bounded, non-growing scheduler-ledger residue
    is therefore the EXPECTED production behaviour, not a leak: assert
    residue <= max_residue (the uncertain batches themselves) and that a
    later sample does not grow (no amplification).
    """
    http = f"http://127.0.0.1:{ops.master_http_port}"
    first = None
    deadline = time.monotonic() + settle_s
    while time.monotonic() < deadline:
        data = http_get_json(f"{http}/rtp_llm/inflight_status", timeout=5)
        if data is not None:
            first = data.get("scheduler_inflight", 0)
            if first <= max_residue:
                break
        time.sleep(1.0)
    if first is None:
        return False, "no inflight_status response"
    if first > max_residue:
        return False, f"residue {first} > bound {max_residue}"
    time.sleep(8.0)
    data = http_get_json(f"{http}/rtp_llm/inflight_status", timeout=5)
    second = -1 if data is None else data.get("scheduler_inflight", 0)
    if second < 0:
        return False, "no inflight_status response (second sample)"
    if second > first:
        return False, f"residue grew {first} -> {second} (leak amplification)"
    return True, f"quarantined residue bounded and stable: {first} -> {second}"
