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
import urllib.request
from dataclasses import dataclass, field
from typing import List, Optional

import grpc

from .harness import (
    DEFAULT_MASTER_MANAGEMENT_PORT,
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

# DashScope inner QoS header — the secondary Auto-TPM priority channel,
# read by the master's GrpcQosHeaderInterceptor into the gRPC Context and
# consumed by PriorityNormalizer when the proto ``priority`` field is unset
# (mirror of flexlb-common PriorityNormalizer.QOS_HEADER_NAME).
QOS_LEVEL_HEADER = "x-dashscope-inner-qos-level"

# Master management-port prometheus exposition: primary Spring Boot
# actuator path first, then the plain /prometheus fallback — the same URL
# ladder as the G3 poller (eval_collectors.py).
MASTER_PROMETHEUS_PATHS = ("actuator/prometheus", "prometheus")


def _http_get_text(url: str, timeout: float = 5.0) -> Optional[str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.read().decode("utf-8", "replace")
    except Exception:
        return None


def _parse_label_block(raw: str) -> dict:
    """Parse the inside of a ``{k1="v1",k2="v2"}`` label block.

    Tolerates the Micrometer/Spring actuator trailing comma
    (``{role="PREFILL",}``).  Label values in this codebase (role /
    engineIp / reason) never carry commas or escapes, so a plain split
    is sufficient — a documented limitation, not a general parser.
    """
    out: dict = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair:
            continue
        key, sep, value = pair.partition("=")
        if not sep:
            continue
        out[key.strip()] = value.strip().strip('"')
    return out


def parse_prometheus_samples(
    body: str, name_prefix: str, labels: Optional[dict] = None
) -> list:
    """Parse a Prometheus text-exposition body into prefix-filtered samples.

    Returns ``[(metric_name, labels_dict, value), ...]`` in file order.
    ``# HELP`` / ``# TYPE`` lines, samples whose name does not start with
    ``name_prefix``, samples missing any required ``labels`` pair, and
    lines with unparseable values are skipped; optional trailing
    timestamps are ignored.  Pure function — locally testable with a
    synthetic body, no network involved.
    """
    samples: list = []
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        # Metric name runs from column 0 to the first "{" or blank.
        name_end = len(line)
        for idx, ch in enumerate(line):
            if ch == "{" or ch == " ":
                name_end = idx
                break
        name = line[:name_end]
        if not name.startswith(name_prefix):
            continue
        rest = line[name_end:]
        label_values: dict = {}
        if rest.startswith("{"):
            close = rest.find("}")
            if close < 0:
                continue
            label_values = _parse_label_block(rest[1:close])
            rest = rest[close + 1 :]
        if labels and any(label_values.get(k) != v for k, v in labels.items()):
            continue
        parts = rest.split()
        if not parts:
            continue
        try:
            value = float(parts[0])
        except ValueError:
            continue
        samples.append((name, label_values, value))
    return samples


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
    # In-band typed error frame (GenerateOutputsPB.error_info, RpcErrorPB):
    # the engine terminates failed streams IN-BAND — a frame carrying
    # error_info is the LAST frame, then the stream completes with gRPC
    # status OK (see priority.py _StreamTerminal's A1 note).  The raw code
    # is an int: proto3 open enums surface non-production values (e.g. an
    # injected 8500) as plain ints on the wire.  These fields are PURE
    # additions — snap.error / snap.completed semantics stay untouched, so
    # every existing consumer keeps its exact prior behavior.
    stream_error_code: Optional[int] = None
    stream_error_message: Optional[str] = None


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
                # In-band typed terminal (error frame): record the raw code
                # and message WITHOUT touching error/completed — the fields
                # above stay exactly as legacy consumers see them, and
                # run_one_request(typed_stream_error=True) surfaces the
                # typed failure to its caller.
                try:
                    if output.HasField("error_info"):
                        self.snap.stream_error_code = int(
                            output.error_info.error_code
                        )
                        self.snap.stream_error_message = (
                            output.error_info.error_message
                        )
                except AttributeError:
                    pass
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
        master_management_port: Optional[int] = None,
    ):
        self.master_ip = master_ip
        self.master_http_port = master_http_port
        self.mock_http_port = mock_http_port
        # Management port serves the actuator/prometheus exposition.
        # Defaults to the harness constant (FLEXLB_FT_MASTER_MANAGEMENT_PORT
        # env, falling back to http+1) — the port start_master actually binds
        # via --management.server.port — so the existing three-argument
        # construction in context.py resolves correctly without changes.
        self.master_management_port = (
            master_management_port
            if master_management_port is not None
            else DEFAULT_MASTER_MANAGEMENT_PORT
        )
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
        # Accepted (and ignored) for kwargs-forwarding symmetry with
        # build_schedule_request: the shared call paths (run_one_request,
        # case _fire helpers) forward ONE kwargs dict to both schedule()
        # and build_generate_input(); GenerateInputPB has no priority
        # field — priority rides the ScheduleRequest proto field (or the
        # QoS header), never the generate input.
        priority: int = 0,
        qos_level: Optional[int] = None,
    ):
        del priority, qos_level
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
        # Auto-TPM QoS priority (proto field 14): 1-100 valid, 0 = unset —
        # the master then normalizes unset to defaultPriority / the QoS
        # header (PriorityNormalizer).  Priority must ride the schedule
        # protocol; embedding it only in unique_key metadata does not reach
        # Auto-TPM admission (same lesson as flexlb_smoke_base.py).
        priority: int = 0,
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
            priority=priority,
        )

    # -- master gRPC -------------------------------------------------------

    def master_target(self) -> str:
        return f"{self.master_ip}:{self.master_http_port + 2}"

    def schedule(
        self,
        request_id: int,
        timeout_s: float = 30.0,
        *,
        qos_level: Optional[int] = None,
        **kwargs,
    ):
        """Schedule RPC against the master.

        ``timeout_s`` is the *client-side gRPC deadline* — the v2 QUEUE
        scheduler parks capacity-blocked requests (a wait condition, see
        FixedWindowBatcherAlgorithm), so callers probing for parking pass a
        short deadline and expect DEADLINE_EXCEEDED.

        ``qos_level`` (optional) attaches the DashScope inner QoS header
        (``x-dashscope-inner-qos-level``) to this Schedule RPC — the
        secondary priority channel read by GrpcQosHeaderInterceptor; the
        proto ``priority`` kwarg (see build_schedule_request) takes
        precedence over it during master-side normalization
        (PriorityNormalizer: proto value > header > defaultPriority).
        """
        stub = self.schedule_pb2_grpc.FlexlbServiceStub(
            self._channel(self.master_target())
        )
        req = self.build_schedule_request(request_id, **kwargs)
        if qos_level is not None:
            return stub.Schedule(
                req,
                timeout=timeout_s,
                metadata=((QOS_LEVEL_HEADER, str(qos_level)),),
            )
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
        self,
        engine_name: Optional[str] = None,
        port: Optional[int] = None,
        mode: str = "graceful",
        drain_timeout_ms: Optional[int] = None,
    ) -> tuple[int, Optional[dict]]:
        """POST /remove_engine {"engine": name} or {"port": grpcPort}.

        Default mode is the mock's GRACEFUL scale-in (strip the discovery
        entry first so the master stops routing, then wait bounded for all
        in-flight work to finish, then tear down) — the production rolling
        scale-in order (user ruling 2026-09: a planned scale-in under load
        must not lose or fail any request).  ``mode="abrupt"`` keeps the
        legacy immediate teardown (in-flight streams cut) for chaos-style
        fault cases.

        The graceful call BLOCKS until the drain settles (mock drain cap
        60s by default), so the HTTP timeout sits well above the bound;
        the response carries ``drained`` / ``drain_ms`` alongside the
        ``running_at_removal`` / ``waiting_at_removal`` counters.  Returns
        (status, body) without raising — 404 (unknown engine) is an
        expected outcome under concurrent add/remove racing.
        """
        body: dict = {}
        if engine_name:
            body["engine"] = engine_name
        if port is not None:
            body["port"] = port
        if not body:
            raise ValueError("remove_engine needs engine_name or port")
        body["mode"] = mode
        if drain_timeout_ms is not None:
            body["drain_timeout_ms"] = drain_timeout_ms
        # Graceful cap is 60s + teardown margin on the Java side; keep the
        # client out of the way of a legitimately slow drain.
        timeout = 5.0 if mode == "abrupt" else 95.0
        return http_post_json(
            f"http://127.0.0.1:{self.mock_http_port}/remove_engine",
            body,
            timeout=timeout,
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

    # -- master prometheus (management port) --------------------------------

    def master_prometheus_text(self) -> Optional[str]:
        """Raw prometheus exposition from the master management port.

        Tries /actuator/prometheus first, then the /prometheus fallback
        (MASTER_PROMETHEUS_PATHS).  Returns None when neither path
        answers (master down / management port not exposed) — distinct
        from an empty exposition string.
        """
        for path in MASTER_PROMETHEUS_PATHS:
            body = _http_get_text(
                f"http://127.0.0.1:{self.master_management_port}/{path}"
            )
            if body is not None:
                return body
        return None

    def master_prometheus_metric(
        self, name_pattern: str, labels: Optional[dict] = None
    ) -> Optional[dict]:
        """Scrape the master prometheus exposition, prefix+label filtered.

        Args:
            name_pattern: metric-name PREFIX.  Prometheus names lose the
                Java dots: the ``app.flexlb.inflight.ttl.expired.qps``
                counter is exposed as
                ``flexlb_app_flexlb_inflight_ttl_expired_qps_total``.
            labels: optional {label_name: required_value} — a sample
                matches only when it carries ALL of these pairs.

        Returns:
            ``{labels_key: value}`` over the matching samples, where
            labels_key is ``name`` for unlabeled samples and
            ``name{k1="v1",k2="v2"}`` (the sample's own label order)
            otherwise — ``.values()`` yields the plain value list.  None
            when the exposition endpoint is unreachable; ``{}`` when
            reachable but no sample matches.

        Sparse-counter semantics (read before computing deltas): the
        master only reports non-zero series, so "sequence absent" and
        "value zero" are indistinguishable — a missing key means "no
        such event has ever happened (yet)".  Before/after delta callers
        must therefore treat a missing key as a 0 baseline; that is safe
        because these counters first APPEAR with a non-zero value (the
        first event), so a None→0 baseline never masks an event nor
        double-counts one.
        """
        body = self.master_prometheus_text()
        if body is None:
            return None
        result: dict = {}
        for name, label_values, value in parse_prometheus_samples(
            body, name_pattern, labels
        ):
            if label_values:
                rendered = ",".join(f'{k}="{v}"' for k, v in label_values.items())
                key = f"{name}{{{rendered}}}"
            else:
                key = name
            result[key] = value
        return result

    def master_ttl_eviction_counts(
        self, engine_ip: Optional[str] = None
    ) -> Optional[dict]:
        """TTL-eviction counters aggregated by ledger role — the assertion
        channel for TTL cases (no G3 timeline file involved).

        Master counter: Java ``app.flexlb.inflight.ttl.expired.qps``, a
        Counter exposed as
        ``flexlb_app_flexlb_inflight_ttl_expired_qps_total`` with tags
        {role, engineIp, reason}, reported at the 60s maintenance-sweep
        granularity, two levels:

          * role=SCHEDULER, engineIp="scheduler" — the scheduler's own
            request-slot ledger sweep (ExpirationTimer);
          * role=PREFILL/DECODE, engineIp=<real engine IP> — per-endpoint
            ledger orphan sweeps (EndpointRegistry).

        Returns ``{"scheduler": v, "prefill": Σ, "decode": Σ}``.  A
        role's value is None when its series is absent — the
        sparse-counter "never happened" state, which delta callers
        treat as a 0 baseline (see master_prometheus_metric).  Overall
        None means the exposition endpoint was unreachable — NOT zero
        evictions; assertions must fail rather than compute a delta
        from it.  ``engine_ip`` restricts PREFILL/DECODE aggregation to
        one engine's series (the scheduler series, tagged
        engineIp="scheduler", only survives that filter when
        engine_ip="scheduler" is passed explicitly).

        Case usage (before/after delta):

            before = ops.master_ttl_eviction_counts()
            ...  # park a request past its inflight TTL
            after = ops.master_ttl_eviction_counts()
            delta = after["decode"] - (before["decode"] or 0)
            assert delta >= 1  # after waiting out the 60s sweep

        The 60s maintenance-sweep granularity means the counter lags
        the eviction event: after-side assertions must poll (wait_for)
        instead of sampling once.
        """
        body = self.master_prometheus_text()
        if body is None:
            return None
        label_filter = {"engineIp": engine_ip} if engine_ip is not None else None
        counts: dict = {"scheduler": None, "prefill": None, "decode": None}
        for _, label_values, value in parse_prometheus_samples(
            body, "flexlb_app_flexlb_inflight_ttl_expired", label_filter
        ):
            role = label_values.get("role", "")
            if role == "SCHEDULER":
                counts["scheduler"] = (counts["scheduler"] or 0.0) + value
            elif role == "PREFILL":
                counts["prefill"] = (counts["prefill"] or 0.0) + value
            elif role == "DECODE":
                counts["decode"] = (counts["decode"] or 0.0) + value
        return counts

    # -- composite request helper ------------------------------------------

    def run_one_request(
        self,
        rid: int,
        stream_timeout_s: float = 15.0,
        typed_stream_error: bool = False,
        **kwargs,
    ) -> tuple[str, Optional[str]]:
        """Schedule → stream → consume to completion.

        Returns (prefill_addr, error) — error is None on success.

        ``typed_stream_error`` (default False, legacy behavior preserved):
        when True, a stream that ended WITHOUT completing but WITH an
        in-band error frame surfaces the typed failure
        ("engine error code=<code>: <message>") instead of the generic
        "stream did not complete" — the execution-phase failure family
        (prefill_async_partial_fail) needs the code visible client-side
        (production stream->reportError half of the terminal), while the
        cancel/timeout family keeps the generic form every existing
        assertion matches on.
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
                if (
                    typed_stream_error
                    and snap.stream_error_code is not None
                ):
                    return addr, (
                        f"engine error code={snap.stream_error_code}: "
                        f"{snap.stream_error_message}"
                    )
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
