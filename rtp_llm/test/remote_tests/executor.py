"""Remote Execution client wrapping the REAPI Execute RPC."""

import atexit
import logging
import os
import signal
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import grpc
from google.protobuf import duration_pb2

# Proto modules are generated at build time.  In a source-only checkout they
# may not exist yet — provide a clear error rather than a bare ImportError.
try:
    from . import bytestream_pb2 as bs_pb2
    from . import remote_execution_pb2 as re_pb2
    from . import remote_execution_pb2_grpc as re_grpc
except ImportError as _proto_err:
    raise ImportError(
        f"remote_tests proto modules not found: {_proto_err}. "
        "Run the proto generation step (e.g. `python -m grpc_tools.protoc` "
        "or the project build system) before importing remote_tests, "
        "or ensure generated files are included in the wheel/distribution."
    ) from _proto_err
from .action_cache_client import _encode_varint
from .cas_client import CASClient
from .endpoint_info import (
    ExecutorEndpointPool,
    combine_reapi_endpoints,
    describe_reapi_endpoint,
    extract_remote_worker_ip,
)

log = logging.getLogger(__name__)

StageCallback = Optional[Callable[[str, str], None]]

_EXECUTION_STAGE_ORDER = {
    "UNKNOWN": 0,
    "CACHE_CHECK": 1,
    "QUEUED": 2,
    "EXECUTING": 3,
    "COMPLETED": 4,
}

_DEFAULT_QUEUED_WATCHDOG_SECONDS = 300
_QUEUED_WATCHDOG_ENV = "RTP_REMOTE_QUEUED_TIMEOUT_SECONDS"


def _byte_stream_tail_loop(
    cas: CASClient,
    resource_name: str,
    out_path: Path,
    metadata: List[tuple],
    stop: threading.Event,
) -> None:
    """Poll ByteStream.Read until stop; append chunks to out_path (live remote stdout/stderr)."""
    stub = cas.new_bytestream_stub()
    offset = 0
    with open(out_path, "ab", buffering=0) as f:
        while not stop.is_set():
            req = bs_pb2.ReadRequest(
                resource_name=resource_name, read_offset=offset, read_limit=0
            )
            try:
                for resp in stub.Read(req, metadata=metadata, timeout=300):
                    if stop.is_set():
                        return
                    if resp.data:
                        f.write(resp.data)
                        f.flush()
                        offset += len(resp.data)
            except grpc.RpcError as e:
                if stop.is_set():
                    return
                if e.code() == grpc.StatusCode.OUT_OF_RANGE:
                    return
                time.sleep(0.5)
                continue
            if stop.is_set():
                return
            time.sleep(0.25)


@dataclass
class ExecutionResult:
    exit_code: int
    stdout_raw: bytes = b""
    stderr_raw: bytes = b""
    stdout_digest: Optional[re_pb2.Digest] = None
    stderr_digest: Optional[re_pb2.Digest] = None
    output_files: Dict[str, re_pb2.Digest] = field(default_factory=dict)
    # Filled when remote pytest prints >>>RTP_REMOTE_HOST_IP (actual worker NIC)
    worker_host_ip: Optional[str] = None
    # REAPI ExecutedActionMetadata.worker when the server populates partial_execution_metadata
    metadata_worker: Optional[str] = None
    cached_result: Optional[bool] = None
    response_status_code: Optional[int] = None
    response_status_message: Optional[str] = None
    # Local paths for live-tailed stream logs (ByteStream); same as logged at execute() start
    stream_stdout_path: Optional[str] = None
    stream_stderr_path: Optional[str] = None
    executor_endpoint: Optional[str] = None
    operation_name: Optional[str] = None
    last_stage: Optional[str] = None
    infra_category: Optional[str] = None
    failover_attempts: int = 0


class RemoteExecutor:
    def __init__(
        self,
        executor_endpoint: str,
        cas: CASClient,
        metadata: Optional[List[tuple]] = None,
    ):
        self.grpc_uri = executor_endpoint
        self.reapi_peer_line = describe_reapi_endpoint("executor", executor_endpoint)
        self.reapi_targets_combined = combine_reapi_endpoints(
            cas.grpc_uri, executor_endpoint
        )
        log.info("REAPI %s", self.reapi_targets_combined)

        addr = executor_endpoint.replace("grpc://", "")
        self.channel = grpc.insecure_channel(
            addr,
            options=[
                ("grpc.max_receive_message_length", 64 * 1024 * 1024),
                ("grpc.keepalive_time_ms", 30000),
                ("grpc.keepalive_timeout_ms", 10000),
                ("grpc.keepalive_permit_without_calls", 1),
                ("grpc.http2.max_pings_without_data", 0),
            ],
        )
        self.stub = re_grpc.ExecutionStub(self.channel)
        self.cas = cas
        self.metadata = metadata or []
        self.instance_name = ""

    def close(self) -> None:
        try:
            self.channel.close()
        except Exception:
            pass

    def cancel_operation(self, operation_name: Optional[str], timeout: int = 5) -> bool:
        if not operation_name:
            return False
        try:
            self.channel.unary_unary(
                "/google.longrunning.Operations/CancelOperation",
                request_serializer=lambda req: req,
                response_deserializer=lambda resp: resp,
            )(
                b"\x0a"
                + _encode_varint(len(operation_name.encode("utf-8")))
                + operation_name.encode("utf-8"),
                metadata=self.metadata,
                timeout=timeout,
            )
            log.info("Cancelled remote operation %s", operation_name)
            return True
        except Exception:
            log.warning("Failed to cancel remote operation %s", operation_name)
            return False

    @staticmethod
    def _try_unpack_execute_metadata(
        op: re_pb2.Operation,
    ) -> Optional[re_pb2.ExecuteOperationMetadata]:
        if not op.metadata or not op.metadata.type_url:
            return None
        meta = re_pb2.ExecuteOperationMetadata()
        try:
            if op.metadata.Unpack(meta):
                return meta
        except Exception:
            pass
        return None

    @staticmethod
    def _extract_stage(op: re_pb2.Operation) -> str:
        meta = RemoteExecutor._try_unpack_execute_metadata(op)
        if meta is None:
            return "UNKNOWN"
        try:
            return re_pb2.ExecutionStage.Value.Name(meta.stage)
        except ValueError:
            return "UNKNOWN"

    @staticmethod
    def _queued_watchdog_seconds(
        action_timeout: int, configured_seconds: Optional[int] = None
    ) -> Optional[int]:
        raw_value = os.environ.get(_QUEUED_WATCHDOG_ENV)
        if configured_seconds is not None:
            configured = configured_seconds
        elif raw_value:
            try:
                configured = int(raw_value)
            except ValueError:
                log.warning(
                    "Ignoring invalid %s=%r; using default %ds",
                    _QUEUED_WATCHDOG_ENV,
                    raw_value,
                    _DEFAULT_QUEUED_WATCHDOG_SECONDS,
                )
                configured = _DEFAULT_QUEUED_WATCHDOG_SECONDS
        else:
            configured = _DEFAULT_QUEUED_WATCHDOG_SECONDS

        if configured <= 0:
            return None
        if action_timeout > 0:
            return max(1, min(configured, action_timeout))
        return configured

    def execute(
        self,
        command: List[str],
        input_root_digest: re_pb2.Digest,
        env_vars: Optional[Dict[str, str]] = None,
        platform_properties: Optional[Dict[str, str]] = None,
        timeout: int = 7200,
        action_timeout_seconds: Optional[int] = None,
        rpc_timeout_seconds: Optional[int] = None,
        queued_timeout_seconds: Optional[int] = None,
        global_deadline_epoch: Optional[float] = None,
        output_files: Optional[List[str]] = None,
        on_stage: StageCallback = None,
        stream_stdout_file: Optional[Path] = None,
        stream_stderr_file: Optional[Path] = None,
        no_cache: bool = False,
    ) -> ExecutionResult:
        action_timeout_seconds = int(action_timeout_seconds or timeout)
        rpc_timeout_seconds = int(rpc_timeout_seconds or (action_timeout_seconds + 120))
        if global_deadline_epoch is not None:
            remaining = int(global_deadline_epoch - time.time())
            if remaining <= 0:
                return ExecutionResult(
                    exit_code=-1,
                    stderr_raw=b"REMOTE_GLOBAL_DEADLINE_EXCEEDED before Execute submit",
                    executor_endpoint=self.grpc_uri,
                    infra_category="watchdog_timeout",
                )
            rpc_timeout_seconds = max(1, min(rpc_timeout_seconds, remaining))
            if rpc_timeout_seconds <= action_timeout_seconds:
                action_timeout_seconds = max(1, rpc_timeout_seconds - 1)
        # Build Command proto — platform.properties are the REAPI equivalent of Bazel
        # exec_properties / remote_default_exec_properties (gpu, gpu_count, …).
        cmd = re_pb2.Command(
            arguments=command,
            environment_variables=[
                re_pb2.Command.EnvironmentVariable(name=k, value=v)
                for k, v in (env_vars or {}).items()
            ],
            output_files=output_files or [],
            platform=re_pb2.Platform(
                properties=[
                    re_pb2.Platform.Property(name=k, value=v)
                    for k, v in (platform_properties or {}).items()
                ]
            ),
        )
        cmd_digest = self.cas.upload_blob(cmd.SerializeToString())

        # Build Action proto
        action = re_pb2.Action(
            command_digest=cmd_digest,
            input_root_digest=input_root_digest,
            timeout=duration_pb2.Duration(seconds=action_timeout_seconds),
            do_not_cache=no_cache,
        )
        action_digest = self.cas.upload_blob(action.SerializeToString())

        log.info(
            "[REMOTE_SUBMIT] executor=%s action=%s action_timeout=%ds rpc_timeout=%ds queued_timeout=%s",
            self.grpc_uri,
            action_digest.hash[:12],
            action_timeout_seconds,
            rpc_timeout_seconds,
            queued_timeout_seconds if queued_timeout_seconds is not None else "auto",
        )

        abs_stdout: Optional[str] = None
        abs_stderr: Optional[str] = None
        if stream_stdout_file is not None:
            stream_stdout_file.parent.mkdir(parents=True, exist_ok=True)
            stream_stdout_file.write_bytes(b"")
            abs_stdout = str(stream_stdout_file.resolve())
        if stream_stderr_file is not None:
            stream_stderr_file.parent.mkdir(parents=True, exist_ok=True)
            stream_stderr_file.write_bytes(b"")
            abs_stderr = str(stream_stderr_file.resolve())
        if abs_stdout is not None or abs_stderr is not None:
            log.info(
                "Remote stream logs (tail -f): stdout=%s stderr=%s",
                abs_stdout or "n/a",
                abs_stderr or "n/a",
            )

        # Execute
        request = re_pb2.ExecuteRequest(
            instance_name=self.instance_name,
            action_digest=action_digest,
            skip_cache_lookup=no_cache,
        )

        stop_event = threading.Event()
        stream_threads: List[threading.Thread] = []
        started_stdout = False
        started_stderr = False
        logged_metadata_worker: Optional[str] = None
        last_stage = "SUBMITTED"

        # --- atexit / SIGTERM cancel: abort remote action if local process dies ---
        _op_name_holder: List[Optional[str]] = [None]  # mutable for closure
        _last_op_name_holder: List[Optional[str]] = [None]
        _execute_call_holder = [None]
        _original_sigterm = signal.getsignal(signal.SIGTERM)

        def _cancel_remote():
            name = _op_name_holder[0]
            if not name:
                return
            _op_name_holder[0] = None  # prevent double cancel
            self.cancel_operation(name)

        def _cancel_execute_stream():
            cancel = getattr(_execute_call_holder[0], "cancel", None)
            if cancel is None:
                return
            try:
                cancel()
                log.info("Cancelled local Execute stream")
            except Exception:
                log.warning("Failed to cancel local Execute stream")

        def _sigterm_handler(signum, frame):
            _cancel_remote()
            if callable(_original_sigterm) and _original_sigterm not in (
                signal.SIG_DFL,
                signal.SIG_IGN,
            ):
                _original_sigterm(signum, frame)
            else:
                raise SystemExit(128 + signum)

        atexit.register(_cancel_remote)
        # signal.signal() can only be called from the main thread.
        # In per-test mode, execute() runs in a ThreadPoolExecutor worker thread.
        _is_main_thread = threading.current_thread() is threading.main_thread()
        if _is_main_thread:
            signal.signal(signal.SIGTERM, _sigterm_handler)

        # Wall-clock watchdog. The gRPC `timeout=...` on Execute() is supposed
        # to deadline the entire stream, but in practice when a remote worker
        # pool is congested the server holds the stream open indefinitely
        # without sending any stage updates — gRPC's per-message deadline
        # behavior is not enforced and the call hangs forever (no QUEUED→
        # EXECUTING transition, no error). Fire a Timer that cancels the
        # remote operation after `timeout + 180s`; the cancellation closes
        # the stream server-side, raising RpcError here that breaks the
        # for-loop. +180s buffer over the action timeout (vs the gRPC
        # +120s) so the wall-clock fallback fires AFTER the server-side
        # action timeout has had a chance.
        _watchdog_fired = [False]
        _watchdog_reason = ["wall-clock"]

        def _watchdog():
            _watchdog_fired[0] = True
            _watchdog_reason[0] = "wall-clock"
            log.error(
                "[REMOTE_WATCHDOG] action exceeded %ds wall-clock — cancelling "
                "remote op (server didn't honor stream deadline)",
                action_timeout_seconds + 180,
            )
            _cancel_remote()
            _cancel_execute_stream()

        watchdog_timer = threading.Timer(action_timeout_seconds + 180, _watchdog)
        watchdog_timer.daemon = True
        watchdog_timer.start()

        queued_watchdog_seconds = self._queued_watchdog_seconds(
            action_timeout_seconds, queued_timeout_seconds
        )
        queued_watchdog_timer: Optional[threading.Timer] = None

        def _cancel_queued_watchdog():
            nonlocal queued_watchdog_timer
            if queued_watchdog_timer is not None:
                queued_watchdog_timer.cancel()
                queued_watchdog_timer = None

        def _queued_watchdog():
            if last_stage != "QUEUED":
                return
            _watchdog_fired[0] = True
            _watchdog_reason[0] = "queued"
            log.error(
                "[REMOTE_WATCHDOG] action stayed QUEUED for %ds — cancelling "
                "remote op to allow executor failover",
                queued_watchdog_seconds,
            )
            _cancel_remote()
            _cancel_execute_stream()

        def _ensure_queued_watchdog(stage: str):
            nonlocal queued_watchdog_timer
            if stage == "QUEUED":
                if queued_watchdog_seconds is None or queued_watchdog_timer is not None:
                    return
                queued_watchdog_timer = threading.Timer(
                    queued_watchdog_seconds, _queued_watchdog
                )
                queued_watchdog_timer.daemon = True
                queued_watchdog_timer.start()
                log.info(
                    "[REMOTE_WATCHDOG] queued watchdog armed for %ds op=%s",
                    queued_watchdog_seconds,
                    _last_op_name_holder[0] or "",
                )
                return
            if _EXECUTION_STAGE_ORDER.get(stage, 0) > _EXECUTION_STAGE_ORDER["QUEUED"]:
                _cancel_queued_watchdog()

        try:
            execute_call = self.stub.Execute(
                request, metadata=self.metadata, timeout=rpc_timeout_seconds
            )
            _execute_call_holder[0] = execute_call
            for op in execute_call:
                if _op_name_holder[0] is None and op.name:
                    _op_name_holder[0] = op.name
                if op.name:
                    _last_op_name_holder[0] = op.name
                meta = self._try_unpack_execute_metadata(op)
                if meta is not None:
                    w = (meta.partial_execution_metadata.worker or "").strip()
                    if w and w != logged_metadata_worker:
                        logged_metadata_worker = w
                        st = self._extract_stage(op)
                        log.info(
                            "Execute REAPI worker=%s stage=%s stdout_stream=%s stderr_stream=%s",
                            w,
                            st,
                            bool(meta.stdout_stream_name),
                            bool(meta.stderr_stream_name),
                        )

                    if (
                        stream_stdout_file is not None
                        and meta.stdout_stream_name
                        and not started_stdout
                    ):
                        started_stdout = True
                        t = threading.Thread(
                            target=_byte_stream_tail_loop,
                            args=(
                                self.cas,
                                meta.stdout_stream_name,
                                stream_stdout_file,
                                self.metadata,
                                stop_event,
                            ),
                            name="reapi-stdout-tail",
                            daemon=True,
                        )
                        t.start()
                        stream_threads.append(t)

                    if (
                        stream_stderr_file is not None
                        and meta.stderr_stream_name
                        and not started_stderr
                    ):
                        started_stderr = True
                        t = threading.Thread(
                            target=_byte_stream_tail_loop,
                            args=(
                                self.cas,
                                meta.stderr_stream_name,
                                stream_stderr_file,
                                self.metadata,
                                stop_event,
                            ),
                            name="reapi-stderr-tail",
                            daemon=True,
                        )
                        t.start()
                        stream_threads.append(t)

                if op.done:
                    if on_stage:
                        on_stage("COMPLETED", op.name)
                    last_stage = "COMPLETED"
                    _cancel_queued_watchdog()
                    stop_event.set()
                    for t in stream_threads:
                        t.join(timeout=60)
                    result = self._parse(op)
                    # Stream metadata worker (if any) or ActionResult.execution_metadata.worker
                    result.metadata_worker = (
                        (logged_metadata_worker or "").strip()
                        or (result.metadata_worker or "").strip()
                        or None
                    )
                    result.stream_stdout_path = abs_stdout
                    result.stream_stderr_path = abs_stderr
                    result.executor_endpoint = self.grpc_uri
                    result.operation_name = op.name
                    result.last_stage = last_stage
                    self._write_final_stream_files(
                        stream_stdout_file,
                        stream_stderr_file,
                        result,
                        started_stdout,
                        started_stderr,
                    )
                    if result.worker_host_ip:
                        log.info(
                            "Execute worker host_ip=%s operation=%s",
                            result.worker_host_ip,
                            op.name,
                        )
                    return result

                stage = self._extract_stage(op)
                if (
                    _EXECUTION_STAGE_ORDER.get(stage, 0)
                    < _EXECUTION_STAGE_ORDER.get(last_stage, 0)
                    and last_stage == "EXECUTING"
                ):
                    message = (
                        f"Execute stage regressed from {last_stage} to {stage}; "
                        "treating remote action as infrastructure failure"
                    )
                    log.warning("[REMOTE_STAGE_REGRESSION] %s op=%s", message, op.name)
                    self.cancel_operation(op.name)
                    stop_event.set()
                    for t in stream_threads:
                        t.join(timeout=5)
                    return ExecutionResult(
                        exit_code=-1,
                        stderr_raw=(
                            f"{message}\n[reapi-targets] {self.reapi_targets_combined}"
                        ).encode(),
                        metadata_worker=logged_metadata_worker,
                        stream_stdout_path=abs_stdout,
                        stream_stderr_path=abs_stderr,
                        executor_endpoint=self.grpc_uri,
                        operation_name=op.name,
                        last_stage=stage,
                        infra_category="executor_stage_regressed",
                    )
                last_stage = stage
                _ensure_queued_watchdog(stage)
                if on_stage:
                    on_stage(stage, op.name)
                log.info("[REMOTE_STAGE] stage=%s op=%s", stage, (op.name or "")[:48])
                log.debug("Operation %s stage=%s", op.name, stage)
        except grpc.RpcError as e:
            log.error("Execute RPC failed: %s", e)
            category = "watchdog_timeout" if _watchdog_fired[0] else "executor_rpc"
            log.error(
                "[RESULT] status=blocked category=%s detail=%s",
                category,
                e.code().name,
            )
            stop_event.set()
            for t in stream_threads:
                t.join(timeout=5)
            detail = f"{e.code().name}: {e.details()}"
            if _watchdog_fired[0]:
                detail = (
                    f"WATCHDOG_TIMEOUT reason={_watchdog_reason[0]} "
                    f"last_stage={last_stage}. {detail}"
                )
            tail = f"{detail}\n[reapi-targets] {self.reapi_targets_combined}"
            return ExecutionResult(
                exit_code=-1,
                stderr_raw=tail.encode(),
                metadata_worker=logged_metadata_worker,
                stream_stdout_path=abs_stdout,
                stream_stderr_path=abs_stderr,
                executor_endpoint=self.grpc_uri,
                operation_name=_last_op_name_holder[0],
                last_stage=last_stage,
                infra_category=category,
            )
        finally:
            # Unregister cancel handlers — action completed or errored
            _op_name_holder[0] = None
            _execute_call_holder[0] = None
            atexit.unregister(_cancel_remote)
            watchdog_timer.cancel()
            _cancel_queued_watchdog()
            if _is_main_thread:
                signal.signal(signal.SIGTERM, _original_sigterm)

        stop_event.set()
        for t in stream_threads:
            t.join(timeout=5)
        infra_category = (
            "watchdog_timeout" if _watchdog_fired[0] else "executor_stream_ended"
        )
        detail = "Execute stream ended without result"
        if _watchdog_fired[0]:
            detail = (
                f"WATCHDOG_TIMEOUT reason={_watchdog_reason[0]} "
                f"last_stage={last_stage}; stream ended without result"
            )
        return ExecutionResult(
            exit_code=-1,
            stderr_raw=(
                f"{detail}\n" f"[reapi-targets] {self.reapi_targets_combined}"
            ).encode(),
            metadata_worker=logged_metadata_worker,
            stream_stdout_path=abs_stdout,
            stream_stderr_path=abs_stderr,
            executor_endpoint=self.grpc_uri,
            operation_name=_last_op_name_holder[0],
            last_stage=last_stage,
            infra_category=infra_category,
        )

    def _write_final_stream_files(
        self,
        stdout_file: Optional[Path],
        stderr_file: Optional[Path],
        result: ExecutionResult,
        started_byte_stream_stdout: bool,
        started_byte_stream_stderr: bool,
    ) -> None:
        """Always materialize stream log paths from ActionResult (CAS digest or inline).

        Many schedulers never set ExecuteOperationMetadata.stdout_stream_name; ByteStream
        then stays idle. After completion we have full stdout/stderr in the response.
        """
        if stdout_file is not None:
            stdout_file.write_bytes(result.stdout_raw or b"")
        if stderr_file is not None:
            stderr_file.write_bytes(result.stderr_raw or b"")
        if stdout_file is None and stderr_file is None:
            return
        if not started_byte_stream_stdout and not started_byte_stream_stderr:
            if result.stdout_raw or result.stderr_raw:
                log.info(
                    "REAPI did not expose ByteStream log names; wrote final stdout/stderr "
                    "(%d / %d bytes) to stream log paths",
                    len(result.stdout_raw or b""),
                    len(result.stderr_raw or b""),
                )
            else:
                log.info(
                    "REAPI returned empty stdout/stderr (no stream names, no inline/digest data)."
                )
        elif not started_byte_stream_stdout and result.stdout_raw:
            log.info(
                "REAPI had no stdout_stream_name; filled stdout log from ActionResult (%d bytes)",
                len(result.stdout_raw),
            )
        elif not started_byte_stream_stderr and result.stderr_raw:
            log.info(
                "REAPI had no stderr_stream_name; filled stderr log from ActionResult (%d bytes)",
                len(result.stderr_raw),
            )

    @staticmethod
    def _classify_execute_response_infra(
        *,
        exit_code: int,
        status_code: Optional[int],
        status_message: Optional[str],
        stdout_raw: bytes,
        stderr_raw: bytes,
    ) -> Optional[str]:
        combined_output = (stdout_raw or b"") + b"\n" + (stderr_raw or b"")
        if exit_code != 0 and (
            b">>>RTP_REMOTE_INFRA_STALL" in combined_output
            or b"[action_supervisor] supervisor_timeout" in combined_output
            or b"[action_supervisor] heartbeat_stall" in combined_output
        ):
            return "infra_stall"

        if exit_code != 0 and (
            b">>>PHASE:pip_install_failed" in combined_output
            or (
                b"[prepare_venv]" in combined_output
                and (
                    b"Failed to fetch:" in combined_output
                    or b"Request failed after" in combined_output
                    or b"operation timed out" in combined_output
                )
            )
        ):
            return "worker_setup_network"

        if exit_code != 0 and (
            b">>>RTP_GPU_INFRA_FAILURE" in combined_output
            or b"NVRM: Xid" in combined_output
            or b"nvAssertFailedNoLog" in combined_output
            or b"GPU has fallen off the bus" in combined_output
        ):
            return "worker_gpu_xid"

        if exit_code == 0 or stdout_raw or stderr_raw:
            return None

        message = status_message or ""
        if not message:
            return None

        nativelink_worker_io = (
            "nativelink/work/" in message
            and "Job cancelled because it attempted to execute too many times"
            in message
            and (
                "Could not create directory" in message
                or "Could not remove working directory" in message
                or "Directory not empty" in message
                or "File exists" in message
            )
        )
        if nativelink_worker_io and status_code in {2, 6}:
            return "executor_worker_io"

        return None

    def _parse(self, op) -> ExecutionResult:
        # LRO operation errors indicate infrastructure-level failures; fail fast
        # instead of trying to unpack a response that may be of the wrong type.
        if op.error and op.error.code != 0:
            err_msg = f"LRO operation failed: code={op.error.code}, message={op.error.message!r}"
            log.error(err_msg)
            return ExecutionResult(
                exit_code=1,
                stderr_raw=err_msg.encode("utf-8"),
            )

        resp = re_pb2.ExecuteResponse()
        # Unpack returns False when the type_url does not match ExecuteResponse.
        if not op.response.Unpack(resp):
            try:
                # Fallback: parse raw value bytes (older REAPI endpoints may not
                # set type_url correctly).
                resp.ParseFromString(op.response.value)
            except Exception:
                return ExecutionResult(
                    exit_code=-1,
                    stderr_raw=(
                        b"Failed to unpack response as ExecuteResponse\n[reapi-targets] "
                        + self.reapi_targets_combined.encode()
                    ),
                )

        # An ExecuteResponse without a result is not a successful execution.
        if not resp.HasField("result"):
            status_code = resp.status.code if resp.HasField("status") else None
            status_message = resp.status.message if resp.HasField("status") else ""
            err_msg = (
                f"ExecuteResponse has no result: status={status_code}, "
                f"message={status_message!r}"
            )
            log.error(err_msg)
            return ExecutionResult(
                exit_code=1,
                stderr_raw=err_msg.encode("utf-8"),
                response_status_code=status_code,
                response_status_message=status_message,
            )

        r = resp.result
        log.info(
            "Remote result: exit_code=%d cached=%s status_code=%s status_message=%r stdout_digest=%s stderr_digest=%s",
            r.exit_code,
            resp.cached_result,
            resp.status.code if resp.HasField("status") else None,
            resp.status.message if resp.HasField("status") else "",
            r.stdout_digest.hash[:12] if r.stdout_digest.hash else "none",
            r.stderr_digest.hash[:12] if r.stderr_digest.hash else "none",
        )

        output_files = {f.path: f.digest for f in r.output_files}

        out_raw = r.stdout_raw or b""
        err_raw = r.stderr_raw or b""
        if not out_raw and r.stdout_digest and r.stdout_digest.hash:
            try:
                out_raw = self.cas.download_blob(r.stdout_digest)
            except Exception as e:
                log.warning("Failed to download stdout from digest: %s", e)
        if not err_raw and r.stderr_digest and r.stderr_digest.hash:
            try:
                err_raw = self.cas.download_blob(r.stderr_digest)
            except Exception as e:
                log.warning("Failed to download stderr from digest: %s", e)

        meta_worker = ""
        if r.execution_metadata and r.execution_metadata.worker:
            meta_worker = (r.execution_metadata.worker or "").strip()
            if meta_worker:
                log.info("ActionResult.execution_metadata.worker=%s", meta_worker)
        else:
            log.debug(
                "ActionResult.execution_metadata missing or worker empty "
                "(scheduler may omit REAPI field 9; use >>>RTP_REMOTE_HOST_IP in stdout)"
            )

        out_txt = out_raw.decode("utf-8", errors="replace")
        worker_ip = extract_remote_worker_ip(out_txt)
        status_code = resp.status.code if resp.HasField("status") else None
        status_message = resp.status.message if resp.HasField("status") else None

        # Fail-closed: if the REAPI response status is non-OK, treat the
        # execution as a failure regardless of exit_code.  Without this an
        # infrastructure error (e.g. deadline exceeded, resource exhausted)
        # could be silently treated as a passing test.
        if status_code is not None and status_code != 0:
            err_msg = (
                f"REAPI Execute returned non-OK status: code={status_code}, "
                f"message={status_message!r}"
            ).encode("utf-8")
            return ExecutionResult(
                exit_code=1,
                stdout_raw=out_raw,
                stderr_raw=err_raw + b"\n" + err_msg,
                stdout_digest=r.stdout_digest if r.stdout_digest.hash else None,
                stderr_digest=r.stderr_digest if r.stderr_digest.hash else None,
                output_files=output_files,
                worker_host_ip=worker_ip,
                metadata_worker=meta_worker or None,
                cached_result=resp.cached_result,
                response_status_code=status_code,
                response_status_message=status_message,
                infra_category=self._classify_execute_response_infra(
                    exit_code=1,
                    status_code=status_code,
                    status_message=status_message,
                    stdout_raw=out_raw,
                    stderr_raw=err_raw,
                ),
            )

        infra_category = self._classify_execute_response_infra(
            exit_code=r.exit_code,
            status_code=status_code,
            status_message=status_message,
            stdout_raw=out_raw,
            stderr_raw=err_raw,
        )

        return ExecutionResult(
            exit_code=r.exit_code,
            stdout_raw=out_raw,
            stderr_raw=err_raw,
            stdout_digest=r.stdout_digest if r.stdout_digest.hash else None,
            stderr_digest=r.stderr_digest if r.stderr_digest.hash else None,
            output_files=output_files,
            worker_host_ip=worker_ip,
            metadata_worker=meta_worker or None,
            cached_result=resp.cached_result,
            response_status_code=status_code,
            response_status_message=status_message,
            infra_category=infra_category,
        )

    def download_output(self, digest: re_pb2.Digest) -> str:
        data = self.cas.download_blob(digest)
        return data.decode("utf-8", errors="replace")


class FailoverRemoteExecutor:
    """RemoteExecutor wrapper that rotates executor IPs on infra failures."""

    _FAILOVER_CATEGORIES = {
        "executor_rpc",
        "executor_stream_ended",
        "executor_stage_regressed",
        "executor_worker_io",
        "watchdog_timeout",
        "infra_stall",
        "worker_gpu_xid",
        "worker_setup_network",
    }

    def __init__(
        self,
        executor_endpoint: str,
        cas: CASClient,
        metadata: Optional[List[tuple]] = None,
        *,
        enabled: bool = True,
        max_failovers: int = 3,
        dns_refresh_seconds: int = 60,
        fallback_executor_endpoint: Optional[str] = None,
        executor_factory=RemoteExecutor,
    ):
        self.pool = ExecutorEndpointPool(
            executor_endpoint,
            fallback_uri=fallback_executor_endpoint,
            refresh_seconds=dns_refresh_seconds,
        )
        self.cas = cas
        self.metadata = metadata or []
        self.enabled = enabled
        self.max_failovers = max(0, int(max_failovers))
        self.executor_factory = executor_factory
        self.reapi_targets_combined = combine_reapi_endpoints(
            cas.grpc_uri,
            self.pool.current_endpoint(),
        )

    def _new_executor(self, endpoint: str) -> RemoteExecutor:
        executor = self.executor_factory(endpoint, self.cas, self.metadata)
        self.reapi_targets_combined = executor.reapi_targets_combined
        return executor

    @staticmethod
    def _is_failoverable(result: ExecutionResult) -> bool:
        return result.infra_category in FailoverRemoteExecutor._FAILOVER_CATEGORIES

    @staticmethod
    def _remaining_seconds(deadline: Optional[float]) -> Optional[int]:
        if deadline is None:
            return None
        return int(deadline - time.time())

    @staticmethod
    def _kwargs_for_budget(kwargs: Dict, remaining: Optional[int]) -> Dict:
        if remaining is None:
            return kwargs
        retry_kwargs = dict(kwargs)
        rpc_timeout = min(
            int(retry_kwargs.get("rpc_timeout_seconds") or remaining), remaining
        )
        action_timeout = int(
            retry_kwargs.get("action_timeout_seconds")
            or retry_kwargs.get("timeout")
            or max(1, rpc_timeout - 120)
        )
        if action_timeout + 120 > rpc_timeout:
            action_timeout = max(1, rpc_timeout - 120)
        retry_kwargs["rpc_timeout_seconds"] = max(1, rpc_timeout)
        retry_kwargs["action_timeout_seconds"] = max(1, action_timeout)
        queued_timeout = retry_kwargs.get("queued_timeout_seconds")
        if queued_timeout is not None:
            retry_kwargs["queued_timeout_seconds"] = max(
                1, min(int(queued_timeout), retry_kwargs["action_timeout_seconds"])
            )
        return retry_kwargs

    @staticmethod
    def _kwargs_for_attempt(kwargs: Dict, attempts: int) -> Dict:
        if attempts <= 0:
            return kwargs
        retry_kwargs = dict(kwargs)
        env_vars = dict(retry_kwargs.get("env_vars") or {})
        # NativeLink may repeatedly schedule the same action digest onto the
        # same unhealthy worker. This no-op salt changes only retry action
        # identity, not test behavior.
        env_vars["RTP_REMOTE_EXECUTOR_FAILOVER_ATTEMPT"] = str(attempts)
        retry_kwargs["env_vars"] = env_vars
        return retry_kwargs

    def execute(self, **kwargs) -> ExecutionResult:
        min_retry_remaining_seconds = int(
            kwargs.pop("min_retry_remaining_seconds", 0) or 0
        )
        global_deadline_epoch = kwargs.get("global_deadline_epoch")
        attempts = 0
        endpoint = self.pool.current_endpoint()
        tried = []
        last_result: Optional[ExecutionResult] = None

        while True:
            remaining = self._remaining_seconds(global_deadline_epoch)
            if remaining is not None and remaining <= 0:
                if last_result is not None:
                    log.warning(
                        "[EXECUTOR_FAILOVER] global deadline exhausted; returning last result category=%s",
                        last_result.infra_category,
                    )
                    return last_result
                return ExecutionResult(
                    exit_code=-1,
                    stderr_raw=b"REMOTE_GLOBAL_DEADLINE_EXCEEDED before retry submit",
                    executor_endpoint=endpoint,
                    infra_category="watchdog_timeout",
                )
            if (
                attempts > 0
                and remaining is not None
                and remaining < min_retry_remaining_seconds
            ):
                if last_result is not None:
                    log.warning(
                        "[EXECUTOR_FAILOVER] refusing retry: remaining=%ss below threshold=%ss category=%s",
                        remaining,
                        min_retry_remaining_seconds,
                        last_result.infra_category,
                    )
                    return last_result

            executor = self._new_executor(endpoint)
            try:
                tried.append(endpoint)
                attempt_kwargs = self._kwargs_for_budget(kwargs, remaining)
                attempt_kwargs = self._kwargs_for_attempt(attempt_kwargs, attempts)
                result = executor.execute(**attempt_kwargs)
                last_result = result
                result.failover_attempts = attempts
                if not self.enabled or not self._is_failoverable(result):
                    return result

                self.pool.refresh(force=True)
                next_endpoint = self.pool.current_endpoint()
                if next_endpoint == endpoint:
                    next_endpoint = self.pool.advance()
                if attempts >= self.max_failovers or next_endpoint == endpoint:
                    log.warning(
                        "[EXECUTOR_FAILOVER] exhausted endpoint=%s category=%s "
                        "operation=%s last_stage=%s tried=[%s]",
                        endpoint,
                        result.infra_category,
                        result.operation_name or "n/a",
                        result.last_stage or "n/a",
                        ",".join(tried),
                    )
                    return result

                old_endpoint = endpoint
                old_operation = result.operation_name
                if old_operation:
                    executor.cancel_operation(old_operation)
                endpoint = next_endpoint
                attempts += 1
                log.warning(
                    "[EXECUTOR_FAILOVER] old=%s new=%s category=%s operation=%s "
                    "last_stage=%s will_rerun=true attempt=%d/%d",
                    old_endpoint,
                    endpoint,
                    result.infra_category,
                    old_operation or "n/a",
                    result.last_stage or "n/a",
                    attempts,
                    self.max_failovers,
                )
            finally:
                executor.close()

    def download_output(self, digest: re_pb2.Digest) -> str:
        data = self.cas.download_blob(digest)
        return data.decode("utf-8", errors="replace")
