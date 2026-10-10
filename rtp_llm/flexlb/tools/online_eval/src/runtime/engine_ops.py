"""gRPC engine operations for the case framework.

Streams are consumed on daemon threads; callers poll their observations.
Operations include scheduling, cancellation, timeout and recovery checks.
Role addresses use the protocol string values PREFILL and DECODE.
"""

from __future__ import annotations

import threading
import time
from typing import List, Optional

import grpc

from runtime.environment_config import DEFAULT_MASTER_MANAGEMENT_PORT
from runtime.proto_utils import encode_unique_key, ensure_proto_modules, ensure_schedule_proto_modules
from runtime.network import http_get_json
from runtime.mock_engine import MockEngineControl
from runtime.stream import StreamHandle, StreamSnapshot

DEFAULT_INPUT_LEN = 2048
DEFAULT_OUTPUT_LEN = 10
RECOVERY_TIMEOUT_S = 30.0

CHANNEL_OPTIONS = [
    ("grpc.max_receive_message_length", 64 * 1024 * 1024),
    ("grpc.max_send_message_length", 64 * 1024 * 1024),
]

# DashScope inner QoS header — the secondary Auto-TPM priority channel,
# read by the master's GrpcQosHeaderInterceptor into the gRPC Context and
# consumed by PriorityNormalizer when the proto ``priority`` field is unset
# (mirror of flexlb-common PriorityNormalizer.QOS_HEADER_NAME).
QOS_LEVEL_HEADER = "x-dashscope-inner-qos-level"


class EngineOps(MockEngineControl):
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
        # via --management.server.port.
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

    def invalidate_channel(self, target: str) -> None:
        """Forget only a restarted endpoint; engine channels remain reusable."""
        channel = self._channels.pop(target, None)
        if channel is not None:
            channel.close()

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
        # build_schedule_request: the shared call paths (request submission,
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
                # non-default shape MUST pass input_pb to preserve the scheduled request shape.
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

    # -- Master debug read -------------------------------------------------

    def master_inflight(self) -> Optional[dict]:
        return http_get_json(
            f"http://127.0.0.1:{self.master_http_port}/rtp_llm/inflight_status",
            timeout=5,
        )
