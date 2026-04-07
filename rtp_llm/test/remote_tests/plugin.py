"""pytest plugin: dispatch GPU tests to remote NativeLink workers via REAPI."""

from __future__ import annotations

import logging
import os
import re
import shlex
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import pytest

from .endpoint_info import extract_remote_worker_ip
from .remote_exec_rtp import (
    GPURequest,
    RemoteRuntimeConfig,
    build_runtime_config,
    collect_remote_files,
    collect_session_files,
    infer_gpu_type_from_markexpr,
    quote_args,
    resolve_default_reapi_endpoints,
    resolve_gpu_type_from_items,
    resolve_item_gpu_request,
    should_dispatch_item_remotely,
)

if TYPE_CHECKING:
    from .cas_client import CASClient, UploadProgress
    from .executor import ExecutionResult, RemoteExecutor

log = logging.getLogger(__name__)

DEFAULT_CONCURRENCY = 20
MAX_RETRIES = 2
MAX_REMOTE_TIMEOUT = 7200

_PHASE_LINE_RE = re.compile(r"^>>>PHASE:\S+\s+\d+\s*$")


def _get_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        log.warning("Ignoring invalid %s=%r; using default %d", name, raw, default)
        return default


def _load_remote_execution_types():
    from .cas_client import CASClient, UploadProgress
    from .executor import ExecutionResult, RemoteExecutor

    return CASClient, UploadProgress, ExecutionResult, RemoteExecutor


def _remote_stream_log_paths(rootdir: Path, key: str) -> Tuple[Path, Path]:
    """Per-remote-run files for live ByteStream stdout/stderr (under .pytest_cache)."""
    slug = re.sub(r"[^\w.-]+", "_", key).strip("_")[:200] or "remote"
    base = rootdir / ".pytest_cache" / "remote_stream_logs" / slug
    base.mkdir(parents=True, exist_ok=True)
    return base / "remote_stdout.log", base / "remote_stderr.log"


def _strip_phase_marker_lines(text: str) -> str:
    """Remove >>>PHASE: lines from remote stdout/stderr for display."""
    if not text:
        return text
    lines = [ln for ln in text.splitlines() if not _PHASE_LINE_RE.match(ln.strip())]
    return "\n".join(lines) + ("\n" if text.endswith("\n") and lines else "")


def _uv_cache_hint(combined_output: str) -> Optional[str]:
    """Best-effort extract of uv cache / download hints from remote logs."""
    if not combined_output:
        return None
    for pat in (
        r"(Downloaded|Installed)\s+[\d,]+\s+\w+",
        r"Audited\s+[\d,]+\s+packages",
        r"Using Python",
    ):
        m = re.search(pat, combined_output)
        if m:
            return m.group(0).strip()
    return None


def parse_phase_list(stdout: str) -> List[Tuple[str, float]]:
    phases: List[Tuple[str, float]] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith(">>>PHASE:"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        name = parts[0].split(":", 2)[-1]
        try:
            ts = float(parts[1])
        except ValueError:
            continue
        phases.append((name, ts))
    return phases


def _display_phase_breakdown(
    stdout: str,
    stderr: str,
    tw,
    *,
    prefix: str = "",
) -> None:
    phases = parse_phase_list(stdout)
    if len(phases) < 2:
        return
    combined = (stdout or "") + "\n" + (stderr or "")
    hint = _uv_cache_hint(combined)
    lines = [f"{prefix}Remote execution breakdown (wall-clock phases):"]
    for i in range(len(phases) - 1):
        name, t0 = phases[i]
        _, t1 = phases[i + 1]
        dur = t1 - t0
        extra = ""
        if name == "uv_install" and hint:
            extra = f"  ({hint})"
        lines.append(f"{prefix}  {name}: {dur:.1f}s{extra}")
    total = phases[-1][1] - phases[0][1]
    lines.append(f"{prefix}  total: {total:.1f}s")
    msg = "\n".join(lines)
    if tw:
        tw.line(msg)
    else:
        log.info("%s", msg)


def pytest_addoption(parser):
    g = parser.getgroup("remote-gpu", "Remote GPU execution via REAPI")
    g.addoption(
        "--remote",
        action="store_true",
        default=False,
        help="Execute GPU tests on remote NativeLink workers",
    )
    g.addoption(
        "--remote-session",
        action="store_true",
        default=False,
        help="Submit entire pytest session as single remote action (for py-ut)",
    )
    g.addoption(
        "--remote-executor",
        default=None,
        help="Remote executor gRPC endpoint (grpc://host:port)",
    )
    g.addoption(
        "--remote-cas", default=None, help="CAS gRPC endpoint (grpc://host:port)"
    )
    g.addoption(
        "--remote-header",
        action="append",
        default=[],
        help="gRPC metadata header (key=value), repeatable",
    )
    g.addoption(
        "--remote-timeout",
        type=int,
        default=_get_int_env("REMOTE_TIMEOUT", 7200),
        help="Per-test timeout in seconds (default: 7200)",
    )
    g.addoption(
        "--remote-concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help=f"Max concurrent remote executions (default: {DEFAULT_CONCURRENCY})",
    )
    g.addoption(
        "--remote-env",
        default="daily",
        choices=["daily", "online"],
        help="REAPI environment: 'daily' or 'online' (default: daily)",
    )
    g.addoption(
        "--remote-gpu-type",
        default=os.getenv("REMOTE_GPU_TYPE"),
        help="Override GPU type for session mode (auto-detected from markers if omitted)",
    )
    g.addoption(
        "--remote-workers",
        type=int,
        default=_get_int_env("REMOTE_WORKERS", _get_int_env("REMOTE_GPU_COUNT", 4)),
        help="Number of xdist workers on remote session worker (default: 4)",
    )
    g.addoption(
        "--remote-pytest-args",
        default="",
        help="Extra pytest args forwarded to the remote session command",
    )
    g.addoption(
        "--remote-no-cache",
        action="store_true",
        default=False,
        help="Disable REAPI action cache lookup/store for remote executions",
    )
    g.addoption(
        "--remote-cas-upload-workers",
        type=int,
        default=12,
        help="Max concurrent CAS BatchUpdateBlobs RPCs (default: 12)",
    )
    g.addoption(
        "--remote-log-file",
        default=None,
        metavar="PATH",
        help="Append rtp_llm.test.remote_tests logging to this file (relative to rootdir if not absolute)",
    )
    g.addoption(
        "--remote-collect-outputs",
        action="store_true",
        default=os.environ.get("RTP_REMOTE_COLLECT_OUTPUTS") == "1",
        help=(
            "Download TEST_UNDECLARED_OUTPUTS_DIR from remote worker after execution "
            "(server logs, smoke_actual, OOM state). Env: RTP_REMOTE_COLLECT_OUTPUTS=1"
        ),
    )
    g.addoption(
        "--remote-outputs-dir",
        default=None,
        metavar="PATH",
        help="Local directory for remote outputs (default: .pytest_cache/remote_outputs)",
    )
    g.addoption(
        "--remote-outputs-max-mb",
        type=int,
        default=200,
        help="Skip download if remote outputs archive exceeds this size in MiB (default: 200)",
    )


def _resolve_endpoints(config) -> tuple:
    """Resolve executor/cas endpoints from CLI options + pyproject defaults."""
    rootdir = Path(config.rootdir)
    executor_ep = config.getoption("--remote-executor")
    cas_ep = config.getoption("--remote-cas")
    if not executor_ep or not cas_ep:
        env = config.getoption("--remote-env")
        default_executor_ep, default_cas_ep = resolve_default_reapi_endpoints(
            rootdir, env=env
        )
        executor_ep = executor_ep or default_executor_ep
        cas_ep = cas_ep or default_cas_ep
    if not executor_ep or not cas_ep:
        raise pytest.UsageError(
            "--remote requires REAPI endpoints; pass --remote-executor/--remote-cas "
            "or configure [tool.rtp-llm.remote] in pyproject.toml"
        )
    return executor_ep, cas_ep


def _parse_metadata(config) -> list:
    """Parse gRPC metadata from CLI --remote-header and pyproject [tool.rtp-llm.remote].headers."""
    from .remote_exec_rtp import _load_pyproject

    raw_headers = list(config.getoption("--remote-header") or [])
    # Also load default headers from pyproject.toml [tool.rtp-llm.remote].headers
    rootdir = Path(config.rootdir)
    cfg = _load_pyproject(rootdir).get("tool", {}).get("rtp-llm", {}).get("remote", {})
    pyproject_headers = cfg.get("headers", [])
    if isinstance(pyproject_headers, list):
        for h in pyproject_headers:
            if isinstance(h, str) and "=" in h and h not in raw_headers:
                raw_headers.append(h)
    return [tuple(h.split("=", 1)) for h in raw_headers]


class RemoteDispatchMode(Enum):
    PER_TEST = "per_test"
    SESSION = "session"


def pytest_configure(config):
    use_session = config.getoption("--remote-session", default=False)
    use_remote = config.getoption("--remote", default=False)
    if use_session and use_remote:
        raise pytest.UsageError("Cannot use --remote and --remote-session together")
    if use_session:
        config.pluginmanager.register(
            RemoteREAPIPlugin(config, RemoteDispatchMode.SESSION), "remote-reapi-impl"
        )
    elif use_remote:
        config.pluginmanager.register(
            RemoteREAPIPlugin(config, RemoteDispatchMode.PER_TEST), "remote-reapi-impl"
        )


def _upload_directory_with_progress(
    cas: CASClient,
    rootdir: Path,
    files: List[str],
    config: pytest.Config,
):
    """Run CAS upload in a background thread and refresh a one-line progress display."""
    _, UploadProgress, _, _ = _load_remote_execution_types()
    progress = UploadProgress()
    err_holder: List[BaseException] = []
    result_holder: List = []

    def _run():
        try:
            d = cas.upload_directory(rootdir, files, progress=progress)
            result_holder.append(d)
        except BaseException as e:
            err_holder.append(e)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    tw = config.get_terminal_writer()
    compact = bool(
        config.getoption("--remote-session", default=False)
        or config.getoption("-q", default=0)
    )
    while t.is_alive():
        with progress._lock:
            ub, tb = progress.uploaded_blobs, progress.total_blobs
            ubytes, tbytes = progress.uploaded_bytes, progress.total_bytes
            phase = progress.phase
        mb_up = ubytes / (1024 * 1024)
        mb_tot = tbytes / (1024 * 1024) if tbytes else 0.0
        if not compact:
            if tb > 0:
                tw.write(
                    f"\rCAS upload: {ub}/{tb} blobs ({mb_up:.1f}/{mb_tot:.1f} MiB) [{phase}]",
                    flush=True,
                )
            else:
                tw.write(f"\rCAS upload: [{phase}]", flush=True)
        time.sleep(0.2)
    t.join()
    if not compact:
        tw.line("")
    if err_holder:
        e = err_holder[0]
        raise RuntimeError(f"CAS upload failed [{cas.reapi_peer_line}]: {e}") from e
    return result_holder[0]


class RemoteREAPIPlugin:
    """REAPI remote execution: per-test (--remote) or full session (--remote-session)."""

    def __init__(self, config, mode: RemoteDispatchMode):
        self.mode = mode
        self.config = config
        self.rootdir = Path(config.rootdir)
        self._executor_ep, self._cas_ep = _resolve_endpoints(config)
        self.metadata = _parse_metadata(config)
        requested_timeout = config.getoption("--remote-timeout")
        self.timeout = min(requested_timeout, MAX_REMOTE_TIMEOUT)
        if requested_timeout > MAX_REMOTE_TIMEOUT:
            log.warning(
                "Clamping --remote-timeout from %d to %d to satisfy current REAPI server limit",
                requested_timeout,
                MAX_REMOTE_TIMEOUT,
            )
        self._cas_workers = config.getoption("--remote-cas-upload-workers")
        self.cas: Optional[CASClient] = None
        self.executor: Optional[RemoteExecutor] = None

        self._remote_file_handler: Optional[logging.Handler] = None
        log_file_opt = config.getoption("--remote-log-file", default=None)
        if log_file_opt:
            self._attach_remote_log_file(Path(log_file_opt))

        # Output collection (TEST_UNDECLARED_OUTPUTS_DIR)
        self._collect_outputs = config.getoption("--remote-collect-outputs")
        _odir = config.getoption("--remote-outputs-dir", default=None)
        self._outputs_dir = Path(_odir) if _odir else (self.rootdir / ".pytest_cache" / "remote_outputs")
        self._outputs_max_bytes = config.getoption("--remote-outputs-max-mb") * 1024 * 1024

        if mode == RemoteDispatchMode.PER_TEST:
            self.concurrency = config.getoption("--remote-concurrency")
            self._input_root = None
            self._remote_items: list = []
            self._futures: Dict[str, Future] = {}
            self._pool: Optional[ThreadPoolExecutor] = None
            self._gpu_request = None
            self._result = None
        else:
            self.gpu_type_override = config.getoption("--remote-gpu-type")
            self.workers = config.getoption("--remote-workers")
            self.pytest_args = config.getoption("--remote-pytest-args")
            self.ci_profile = config.getoption("--rtp-ci-profile", default=None)
            self.no_cache = config.getoption("--remote-no-cache")
            self._result = None
            self._gpu_request: Optional[GPURequest] = None

    def _attach_remote_log_file(self, path: Path) -> None:
        """Mirror package logging to a file (in addition to console)."""
        path = path if path.is_absolute() else (self.rootdir / path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(path, mode="a", encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        pkg = logging.getLogger("rtp_llm.test.remote_tests")
        if pkg.level == logging.NOTSET:
            pkg.setLevel(logging.INFO)
        pkg.addHandler(fh)
        self._remote_file_handler = fh
        self._remote_log_file_resolved = path.resolve()
        log.info(
            "Remote diagnostics also logged to file: %s", self._remote_log_file_resolved
        )

    def pytest_sessionstart(self, session):
        p = getattr(self, "_remote_log_file_resolved", None)
        if p is not None:
            tw = self.config.get_terminal_writer()
            if tw:
                tw.line(f"Remote diagnostics log file: {p}")

    def _ensure_remote_clients(self) -> None:
        if self.cas is not None and self.executor is not None:
            return
        CASClient, _, _, RemoteExecutor = _load_remote_execution_types()
        cas = CASClient(
            self._cas_ep,
            self.metadata,
            batch_upload_workers=self._cas_workers,
        )
        self.cas = cas
        self.executor = RemoteExecutor(self._executor_ep, cas, self.metadata)

    def _ensure_uploaded(self, items) -> None:
        if self._input_root:
            return
        self._ensure_remote_clients()
        files = collect_remote_files(self.rootdir, items)
        self._input_root = _upload_directory_with_progress(
            self.cas, self.rootdir, files, self.config
        )
        log.info(
            "Input root uploaded: %s (%d bytes)",
            self._input_root.hash[:12],
            self._input_root.size_bytes,
        )

    def _build_command(self, item, runtime: RemoteRuntimeConfig) -> List[str]:
        test_id = item.name
        test_path = str(Path(str(item.fspath)).relative_to(self.rootdir))
        ignore_args = quote_args(runtime.ignore_args)
        # Forward markexpr so conftest.py doesn't deselect manual tests
        markexpr = getattr(self.config.option, "markexpr", "") or ""
        mark_arg = f"-m {shlex.quote(markexpr)} " if markexpr else ""

        outputs_prefix = ""
        outputs_postscript = ""
        if self._collect_outputs:
            from .output_collector import make_mkdir_prefix, make_tar_postscript
            outputs_prefix = make_mkdir_prefix()
            outputs_postscript = make_tar_postscript() + "; "

        run_cmd = (
            f"{outputs_prefix}"
            "echo \">>>RTP_REMOTE_HOST_IP $(hostname -I 2>/dev/null | awk '{print $1}')\"; "
            'echo ">>>PHASE:pytest_start $(date +%s)"; '
            f"python -m pytest -xvs --tb=long --timeout={self.timeout} "
            f"--override-ini='addopts=' {ignore_args} "
            f"{mark_arg}"
            f"-k {shlex.quote(test_id)} {shlex.quote(test_path)} 2>&1; ec=$?; "
            "echo EXIT_CODE=$ec; "
            'echo ">>>PHASE:pytest_end $(date +%s)"; '
            f"{outputs_postscript}"
            "exit $ec"
        )
        return ["bash", "-c", f"{runtime.remote_setup_prefix}{run_cmd}"]

    def _execute_with_retry(self, **kwargs) -> ExecutionResult:
        self._ensure_remote_clients()
        last_result = None
        for attempt in range(MAX_RETRIES + 1):
            result = self.executor.execute(**kwargs)
            if result.exit_code != -1:
                return result
            last_result = result
            if attempt < MAX_RETRIES:
                wait = 2**attempt
                log.warning(
                    "[RETRY] %s after %ds (attempt %d/%d)",
                    (
                        kwargs.get("command", ["?"])[2][:60]
                        if len(kwargs.get("command", [])) > 2
                        else "?"
                    ),
                    wait,
                    attempt + 1,
                    MAX_RETRIES,
                )
                time.sleep(wait)
        return last_result

    @pytest.hookimpl(trylast=True)
    def pytest_collection_modifyitems(self, items, config=None, session=None):
        if self.mode == RemoteDispatchMode.SESSION:
            self._session_collection_modifyitems(self.config, items)
        else:
            self._per_test_collection_modifyitems(items)

    def _per_test_collection_modifyitems(self, items) -> None:
        self._remote_items = [i for i in items if should_dispatch_item_remotely(i)]
        if not self._remote_items:
            return

        self._ensure_uploaded(self._remote_items)

        workers = min(len(self._remote_items), self.concurrency)
        self._pool = ThreadPoolExecutor(max_workers=workers)
        for item in self._remote_items:
            gpu_req = resolve_item_gpu_request(item)
            runtime = build_runtime_config(
                self.rootdir, gpu_req,
                input_root_hash=self._input_root.hash if self._input_root else None,
            )
            cmd = self._build_command(item, runtime)

            def _on_stage(stage: str, op_name: str, nodeid=item.nodeid):
                log.info(
                    "[REMOTE %s] stage=%s op=%s", nodeid, stage, (op_name or "")[:48]
                )

            log.info(
                "[SUBMIT] %s (gpu=%s x%d)",
                item.nodeid,
                gpu_req.gpu_type,
                gpu_req.gpu_count,
            )
            stdout_log, stderr_log = _remote_stream_log_paths(self.rootdir, item.nodeid)
            env_vars = dict(runtime.env_vars)
            output_files = None
            if self._collect_outputs:
                from .output_collector import make_output_collection_env, make_output_files_decl
                env_vars.update(make_output_collection_env())
                output_files = make_output_files_decl()
            future = self._pool.submit(
                self._execute_with_retry,
                command=cmd,
                input_root_digest=self._input_root,
                env_vars=env_vars,
                platform_properties=runtime.platform_properties,
                timeout=self.timeout,
                output_files=output_files,
                on_stage=_on_stage,
                stream_stdout_file=stdout_log,
                stream_stderr_file=stderr_log,
            )
            self._futures[item.nodeid] = future
        log.info(
            "Submitted %d remote tests (concurrency=%d)", len(self._futures), workers
        )

    def _session_collection_modifyitems(self, config, items) -> None:
        if not items:
            # No tests collected locally (e.g. ROCm/PPU host lacks GPU hardware).
            # Try to infer GPU type from --remote-gpu-type or markexpr.
            gpu_type = self.gpu_type_override
            if not gpu_type:
                markexpr = getattr(config.option, "markexpr", "") or ""
                gpu_type = infer_gpu_type_from_markexpr(markexpr)
            if not gpu_type:
                log.warning("Session mode: no tests collected — nothing to run remotely")
                return
            log.info(
                "Session mode: 0 tests collected locally, gpu_type=%s "
                "(from %s) — submitting session for remote re-collection",
                gpu_type,
                "--remote-gpu-type" if self.gpu_type_override else "markexpr",
            )
        else:
            gpu_type = resolve_gpu_type_from_items(items, override=self.gpu_type_override)

        self._gpu_request = GPURequest(gpu_type=gpu_type, gpu_count=self.workers)
        log.info(
            "Session mode: resolved gpu_type=%s, gpu_count=%d from %d collected items",
            gpu_type,
            self.workers,
            len(items),
        )
        items[:] = []

    @pytest.hookimpl(tryfirst=True)
    def pytest_runtest_protocol(self, item, nextitem):
        if self.mode != RemoteDispatchMode.PER_TEST:
            return None
        if item.nodeid not in self._futures:
            return None

        future = self._futures[item.nodeid]
        log.info("[WAIT] %s", item.nodeid)
        try:
            result = future.result()
        except Exception as e:
            _, _, ExecutionResult, _ = _load_remote_execution_types()
            tail = f"{e}\n[reapi-targets] {self.executor.reapi_targets_combined}"
            result = ExecutionResult(exit_code=-1, stderr_raw=tail.encode())

        self._report_per_test(item, result)
        return True

    def pytest_unconfigure(self, config):
        fh = self._remote_file_handler
        if fh is not None:
            pkg = logging.getLogger("rtp_llm.test.remote_tests")
            pkg.removeHandler(fh)
            fh.close()
            self._remote_file_handler = None
        if self.mode == RemoteDispatchMode.PER_TEST and self._pool:
            self._pool.shutdown(wait=False)

    def _report_per_test(self, item, result: ExecutionResult):
        from _pytest.runner import CallInfo

        stdout = (
            result.stdout_raw.decode("utf-8", errors="replace")
            if result.stdout_raw
            else ""
        )
        if not stdout and result.stdout_digest:
            stdout = self.executor.download_output(result.stdout_digest)
        stderr = (
            result.stderr_raw.decode("utf-8", errors="replace")
            if result.stderr_raw
            else ""
        )
        if not stderr and result.stderr_digest:
            stderr = self.executor.download_output(result.stderr_digest)

        _display_phase_breakdown(stdout, stderr, None)

        worker_ip = result.worker_host_ip or extract_remote_worker_ip(stdout)
        if worker_ip:
            log.info("[remote-worker] %s host_ip=%s", item.nodeid, worker_ip)

        # Download TEST_UNDECLARED_OUTPUTS_DIR artifacts from CAS
        if self._collect_outputs:
            from .output_collector import download_and_extract
            slug = re.sub(r"[^\w.-]+", "_", item.nodeid).strip("_")[:200] or "test"
            out_path = download_and_extract(
                self.cas, result, self._outputs_dir / slug,
                max_bytes=self._outputs_max_bytes,
            )
            if out_path:
                log.info("[remote-outputs] %s -> %s", item.nodeid, out_path)

        tw = self.config.get_terminal_writer()
        if result.stream_stdout_path or result.stream_stderr_path:
            stream_msg = (
                f"Remote stream logs (tail -f): stdout={result.stream_stdout_path or 'n/a'} "
                f"stderr={result.stream_stderr_path or 'n/a'}"
            )
            log.info("%s | %s", item.nodeid, stream_msg)
            if tw:
                tw.line(stream_msg)
        if result.metadata_worker:
            meta_msg = f"Remote REAPI worker (metadata): {result.metadata_worker}"
            log.info("[remote-metadata] %s %s", item.nodeid, meta_msg)
            if tw:
                tw.line(meta_msg)

        setup_call = CallInfo.from_call(lambda: None, when="setup")
        setup_report = pytest.TestReport.from_item_and_call(item, setup_call)
        item.ihook.pytest_runtest_logreport(report=setup_report)

        if result.exit_code == 0:
            # Belt-and-suspenders: also check EXIT_CODE= in stdout
            # (REAPI exit_code may be 0 if bash ends with echo)
            real_exit = 0
            if "EXIT_CODE=" in stdout:
                try:
                    real_exit = int(
                        stdout.rsplit("EXIT_CODE=", 1)[1].strip().split()[0]
                    )
                except (ValueError, IndexError):
                    pass
            if real_exit == 0:
                call = CallInfo.from_call(lambda: None, when="call")
                report = pytest.TestReport.from_item_and_call(item, call)
            else:
                log.warning(
                    "[EXIT_CODE mismatch] %s: REAPI exit=0 but EXIT_CODE=%d",
                    item.nodeid,
                    real_exit,
                )
                stdout = _strip_phase_marker_lines(stdout)
                stderr = _strip_phase_marker_lines(stderr)
                msg = f"Remote execution failed (EXIT_CODE={real_exit}, REAPI exit=0)"
                if stderr:
                    msg += f"\n--- stderr ---\n{stderr[-8000:]}"
                if stdout:
                    msg += f"\n--- stdout ---\n{stdout[-8000:]}"
                msg += (
                    f"\n[remote] worker_host_ip={worker_ip or 'n/a'} | "
                    f"{self.executor.reapi_targets_combined}"
                )
                call = CallInfo.from_call(
                    lambda: pytest.fail(msg, pytrace=False), when="call"
                )
                report = pytest.TestReport.from_item_and_call(item, call)
        else:
            stdout = _strip_phase_marker_lines(stdout)
            stderr = _strip_phase_marker_lines(stderr)
            msg = f"Remote execution failed (exit={result.exit_code})"
            if stderr:
                msg += f"\n--- stderr ---\n{stderr[-8000:]}"
            if stdout:
                msg += f"\n--- stdout ---\n{stdout[-8000:]}"
            msg += (
                f"\n[remote] worker_host_ip={worker_ip or 'n/a'} | "
                f"{self.executor.reapi_targets_combined}"
            )

            call = CallInfo.from_call(
                lambda: pytest.fail(msg, pytrace=False), when="call"
            )
            report = pytest.TestReport.from_item_and_call(item, call)

        item.ihook.pytest_runtest_logreport(report=report)

        teardown_call = CallInfo.from_call(lambda: None, when="teardown")
        teardown_report = pytest.TestReport.from_item_and_call(item, teardown_call)
        item.ihook.pytest_runtest_logreport(report=teardown_report)

    @pytest.hookimpl(tryfirst=True)
    def pytest_runtestloop(self, session):
        if self.mode != RemoteDispatchMode.SESSION:
            return None
        if self._gpu_request is None:
            log.warning(
                "Session mode: no tests to run remotely (0 collected for this profile)"
            )
            self._result = 0
            return True

        gpu_req = self._gpu_request

        self._ensure_remote_clients()
        files = collect_session_files(self.rootdir)
        input_root = _upload_directory_with_progress(
            self.cas, self.rootdir, files, self.config
        )
        log.info(
            "[SESSION_INPUT] files=%d gpu_type=%s gpu_count=%d",
            len(files),
            gpu_req.gpu_type,
            gpu_req.gpu_count,
        )
        log.info(
            "Input root uploaded: %s (%d bytes)",
            input_root.hash[:12],
            input_root.size_bytes,
        )
        log.info(
            "[SESSION_INPUT] input_root=%s size_bytes=%d",
            input_root.hash[:12],
            input_root.size_bytes,
        )

        runtime = build_runtime_config(
            self.rootdir, gpu_req,
            input_root_hash=input_root.hash,
        )
        cmd = self._build_session_command(self.pytest_args, runtime, self.ci_profile)
        log.info(
            "[REMOTE_CMD] mode=session pytest_workers=%d markexpr=%s pytest_args=%s no_cache=%s",
            self.workers,
            bool(getattr(self.config.option, "markexpr", "") or ""),
            bool((self.pytest_args or "").strip()),
            self.no_cache,
        )

        log.info(
            "Submitting session to remote worker (timeout=%ds, gpu=%s x%d)...",
            self.timeout,
            gpu_req.gpu_type,
            gpu_req.gpu_count,
        )

        tw = self.config.get_terminal_writer()
        t0 = time.time()
        last_stage = [None]

        def _on_stage(stage: str, op_name: str):
            if stage != last_stage[0]:
                elapsed = time.time() - t0
                log.info(
                    "[REMOTE_STAGE] stage=%s elapsed=%.0fs op=%s",
                    stage,
                    elapsed,
                    (op_name or "")[:48],
                )
                tw.line(f"Session remote: stage={stage} ({elapsed:.0f}s elapsed)")
                last_stage[0] = stage

        sess_out, sess_err = _remote_stream_log_paths(self.rootdir, "session")
        env_vars = dict(runtime.env_vars)
        output_files = None
        if self._collect_outputs:
            from .output_collector import make_output_collection_env, make_output_files_decl
            env_vars.update(make_output_collection_env())
            output_files = make_output_files_decl()
        result = self.executor.execute(
            command=cmd,
            input_root_digest=input_root,
            env_vars=env_vars,
            platform_properties=runtime.platform_properties,
            timeout=self.timeout,
            output_files=output_files,
            on_stage=_on_stage,
            stream_stdout_file=sess_out,
            stream_stderr_file=sess_err,
            no_cache=self.no_cache,
        )

        if self._collect_outputs:
            from .output_collector import download_and_extract
            out_path = download_and_extract(
                self.cas, result, self._outputs_dir / "session",
                max_bytes=self._outputs_max_bytes,
            )
            if out_path:
                log.info("[remote-outputs] session -> %s", out_path)
                if tw:
                    tw.line(f"Remote outputs: {out_path}")

        self._result = self._parse_remote_output(result, tw)
        log.info("Session completed with exit code %d", self._result)
        return True

    @pytest.hookimpl(trylast=True)
    def pytest_sessionfinish(self, session, exitstatus):
        if self.mode != RemoteDispatchMode.SESSION:
            return
        if self._result is not None:
            session.exitstatus = self._result

    def _parse_remote_output(self, result: ExecutionResult, tw) -> int:
        """Extract stdout/stderr, save junitxml, strip phase lines, show breakdown, return exit code."""
        import sys as _sys

        stdout = (
            result.stdout_raw.decode("utf-8", errors="replace")
            if result.stdout_raw
            else ""
        )
        if not stdout and result.stdout_digest:
            stdout = self.executor.download_output(result.stdout_digest)
        stderr = (
            result.stderr_raw.decode("utf-8", errors="replace")
            if result.stderr_raw
            else ""
        )
        if not stderr and result.stderr_digest:
            stderr = self.executor.download_output(result.stderr_digest)

        worker_ip = extract_remote_worker_ip(stdout) or result.worker_host_ip
        if worker_ip:
            log.info("Session remote worker host_ip=%s", worker_ip)
            if tw:
                tw.line(f"Remote worker host_ip={worker_ip}")

        if result.stream_stdout_path or result.stream_stderr_path:
            stream_msg = (
                f"Remote stream logs (tail -f): stdout={result.stream_stdout_path or 'n/a'} "
                f"stderr={result.stream_stderr_path or 'n/a'}"
            )
            log.info("Session %s", stream_msg)
            if tw:
                tw.line(stream_msg)
        if result.metadata_worker:
            meta_msg = f"Remote REAPI worker (metadata): {result.metadata_worker}"
            log.info("Session %s", meta_msg)
            if tw:
                tw.line(meta_msg)
        if result.response_status_code is not None or result.response_status_message:
            status_msg = (
                f"Remote ExecuteResponse status: code={result.response_status_code} "
                f"message={result.response_status_message or ''}".rstrip()
            )
            log.info("Session %s", status_msg)
            if tw:
                tw.line(status_msg)

        _display_phase_breakdown(stdout, stderr, tw, prefix="")

        if "<<<JUNIT_XML>>>" in stdout and "<<<END_JUNIT_XML>>>" in stdout:
            xml_content = (
                stdout.split("<<<JUNIT_XML>>>", 1)[1]
                .split("<<<END_JUNIT_XML>>>", 1)[0]
                .strip()
            )
            if xml_content:
                junit_path = self.rootdir / "pytest_results.xml"
                junit_path.write_text(xml_content)
                log.info(
                    "Wrote remote junitxml to %s (%d bytes)",
                    junit_path,
                    len(xml_content),
                )
            stdout = stdout.split("<<<JUNIT_XML>>>", 1)[0]

        display_stdout = _strip_phase_marker_lines(stdout)
        if display_stdout:
            print(display_stdout, end="" if display_stdout.endswith("\n") else "\n")
        stderr_disp = _strip_phase_marker_lines(stderr)
        if stderr_disp:
            print(stderr_disp, file=_sys.stderr)

        exit_code = result.exit_code
        if "EXIT_CODE=" in stdout:
            try:
                exit_code = int(stdout.rsplit("EXIT_CODE=", 1)[1].strip().split()[0])
            except (ValueError, IndexError):
                pass

        if exit_code != 0:
            diag = (
                f"[remote] worker_host_ip={worker_ip or 'n/a'} | "
                f"{self.executor.reapi_targets_combined}"
            )
            if (
                result.response_status_code is not None
                or result.response_status_message
            ):
                diag += (
                    f" | response_status={result.response_status_code}:"
                    f"{result.response_status_message or ''}"
                )
            log.error("Session remote diagnostics: %s", diag)
            if tw:
                tw.line(diag)
            print(diag, file=_sys.stderr)

        return exit_code

    def _build_session_command(
        self,
        pytest_args: str,
        runtime: RemoteRuntimeConfig,
        ci_profile: Optional[str] = None,
    ) -> List[str]:
        ignore_args = quote_args(runtime.ignore_args)
        # Forward markexpr to remote worker if set locally
        markexpr = getattr(self.config.option, "markexpr", "") or ""
        mark_arg = f"-m {shlex.quote(markexpr)} " if markexpr else ""
        profile_arg = (
            f"--rtp-ci-profile={shlex.quote(ci_profile)} " if ci_profile else ""
        )

        outputs_prefix = ""
        outputs_postscript = ""
        if self._collect_outputs:
            from .output_collector import make_mkdir_prefix, make_tar_postscript
            outputs_prefix = make_mkdir_prefix()
            outputs_postscript = make_tar_postscript() + "; "

        # pytest_args comes from --remote-pytest-args (operator-controlled), not user input
        run_cmd = (
            f"{outputs_prefix}"
            "echo \">>>RTP_REMOTE_HOST_IP $(hostname -I 2>/dev/null | awk '{print $1}')\"; "
            'echo ">>>PHASE:pytest_start $(date +%s)"; '
            f"python -m pytest {pytest_args} "
            f"{profile_arg}"
            f"{mark_arg}"
            f"-n {self.workers} "
            f"--continue-on-collection-errors "
            f"--junitxml=pytest_results.xml "
            f"--override-ini='addopts=' {ignore_args} "
            f"--tb=short 2>&1; ec=$?; "
            'echo ">>>PHASE:pytest_end $(date +%s)"; '
            "echo EXIT_CODE=$ec; "
            f"echo '<<<JUNIT_XML>>>'; cat pytest_results.xml 2>/dev/null; echo '<<<END_JUNIT_XML>>>'; "
            f"{outputs_postscript}"
            "exit $ec"
        )
        return ["bash", "-c", f"{runtime.remote_setup_prefix}{run_cmd}"]
