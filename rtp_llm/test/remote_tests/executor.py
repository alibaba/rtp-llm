"""Remote Execution client wrapping the REAPI Execute RPC."""
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import grpc
from google.protobuf import duration_pb2

from . import bytestream_pb2 as bs_pb2
from . import remote_execution_pb2 as re_pb2
from . import remote_execution_pb2_grpc as re_grpc
from .cas_client import CASClient
from .endpoint_info import (
    combine_reapi_endpoints,
    describe_reapi_endpoint,
    extract_remote_worker_ip,
)

log = logging.getLogger(__name__)

StageCallback = Optional[Callable[[str, str], None]]


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
            req = bs_pb2.ReadRequest(resource_name=resource_name, read_offset=offset, read_limit=0)
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
    # Local paths for live-tailed stream logs (ByteStream); same as logged at execute() start
    stream_stdout_path: Optional[str] = None
    stream_stderr_path: Optional[str] = None


class RemoteExecutor:
    def __init__(self, executor_endpoint: str, cas: CASClient,
                 metadata: Optional[List[tuple]] = None):
        self.grpc_uri = executor_endpoint
        self.reapi_peer_line = describe_reapi_endpoint("executor", executor_endpoint)
        self.reapi_targets_combined = combine_reapi_endpoints(
            cas.grpc_uri, executor_endpoint)
        log.info("REAPI %s", self.reapi_targets_combined)

        addr = executor_endpoint.replace("grpc://", "")
        self.channel = grpc.insecure_channel(
            addr, options=[
                ("grpc.max_receive_message_length", 64 * 1024 * 1024),
                ("grpc.keepalive_time_ms", 30000),
                ("grpc.keepalive_timeout_ms", 10000),
                ("grpc.keepalive_permit_without_calls", 1),
                ("grpc.http2.max_pings_without_data", 0),
            ])
        self.stub = re_grpc.ExecutionStub(self.channel)
        self.cas = cas
        self.metadata = metadata or []
        self.instance_name = ""

    @staticmethod
    def _try_unpack_execute_metadata(op: re_pb2.Operation) -> Optional[re_pb2.ExecuteOperationMetadata]:
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

    def execute(
        self,
        command: List[str],
        input_root_digest: re_pb2.Digest,
        env_vars: Optional[Dict[str, str]] = None,
        platform_properties: Optional[Dict[str, str]] = None,
        timeout: int = 7200,
        output_files: Optional[List[str]] = None,
        on_stage: StageCallback = None,
        stream_stdout_file: Optional[Path] = None,
        stream_stderr_file: Optional[Path] = None,
    ) -> ExecutionResult:
        # Build Command proto — platform.properties are the REAPI equivalent of Bazel
        # exec_properties / remote_default_exec_properties (gpu, gpu_count, …).
        cmd = re_pb2.Command(
            arguments=command,
            environment_variables=[
                re_pb2.Command.EnvironmentVariable(name=k, value=v)
                for k, v in (env_vars or {}).items()
            ],
            output_files=output_files or [],
            platform=re_pb2.Platform(properties=[
                re_pb2.Platform.Property(name=k, value=v)
                for k, v in (platform_properties or {}).items()
            ]),
        )
        cmd_digest = self.cas.upload_blob(cmd.SerializeToString())

        # Build Action proto
        action = re_pb2.Action(
            command_digest=cmd_digest,
            input_root_digest=input_root_digest,
            timeout=duration_pb2.Duration(seconds=timeout),
            do_not_cache=False,
        )
        action_digest = self.cas.upload_blob(action.SerializeToString())

        log.info("Submitting action %s (cmd=%s)", action_digest.hash[:12], " ".join(command[:5]))

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
            skip_cache_lookup=False,
        )

        stop_event = threading.Event()
        stream_threads: List[threading.Thread] = []
        started_stdout = False
        started_stderr = False
        logged_metadata_worker: Optional[str] = None

        try:
            for op in self.stub.Execute(request, metadata=self.metadata, timeout=timeout + 120):
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
                if on_stage:
                    on_stage(stage, op.name)
                log.debug("Operation %s stage=%s", op.name, stage)
        except grpc.RpcError as e:
            log.error("Execute RPC failed: %s", e)
            stop_event.set()
            for t in stream_threads:
                t.join(timeout=5)
            detail = f"{e.code().name}: {e.details()}"
            tail = f"{detail}\n[reapi-targets] {self.reapi_targets_combined}"
            return ExecutionResult(
                exit_code=-1,
                stderr_raw=tail.encode(),
                metadata_worker=logged_metadata_worker,
                stream_stdout_path=abs_stdout,
                stream_stderr_path=abs_stderr,
            )

        stop_event.set()
        for t in stream_threads:
            t.join(timeout=5)
        return ExecutionResult(
            exit_code=-1,
            stderr_raw=(
                "Execute stream ended without result\n"
                f"[reapi-targets] {self.reapi_targets_combined}"
            ).encode(),
            metadata_worker=logged_metadata_worker,
            stream_stdout_path=abs_stdout,
            stream_stderr_path=abs_stderr,
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
            if (result.stdout_raw or result.stderr_raw):
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

    def _parse(self, op) -> ExecutionResult:
        resp = re_pb2.ExecuteResponse()
        try:
            # Try Unpack first (handles type_url matching)
            op.response.Unpack(resp)
        except Exception:
            try:
                # Fallback: parse raw value bytes
                resp.ParseFromString(op.response.value)
            except Exception:
                return ExecutionResult(
                    exit_code=-1,
                    stderr_raw=(
                        b"Failed to unpack response\n[reapi-targets] "
                        + self.reapi_targets_combined.encode()
                    ),
                )

        r = resp.result
        log.info("Remote result: exit_code=%d stdout_digest=%s stderr_digest=%s",
                 r.exit_code, r.stdout_digest.hash[:12] if r.stdout_digest.hash else "none",
                 r.stderr_digest.hash[:12] if r.stderr_digest.hash else "none")

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

        return ExecutionResult(
            exit_code=r.exit_code,
            stdout_raw=out_raw,
            stderr_raw=err_raw,
            stdout_digest=r.stdout_digest if r.stdout_digest.hash else None,
            stderr_digest=r.stderr_digest if r.stderr_digest.hash else None,
            output_files=output_files,
            worker_host_ip=worker_ip,
            metadata_worker=meta_worker or None,
        )

    def download_output(self, digest: re_pb2.Digest) -> str:
        data = self.cas.download_blob(digest)
        return data.decode("utf-8", errors="replace")
