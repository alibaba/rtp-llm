"""pytest plugin: dispatch GPU tests to remote NativeLink workers via REAPI."""

from __future__ import annotations

import logging
import os
import re
import shlex
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import Future, ThreadPoolExecutor
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Tuple

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
    resolve_ci_profile_gpu_type,
    resolve_ci_profile_remote_env,
    resolve_default_reapi_endpoint_config,
    resolve_gpu_type_from_items,
    resolve_item_gpu_request,
    should_dispatch_item_remotely,
)
from .remote_timeout_policy import RemoteTimeoutPolicy, select_remote_timeout_policy

if TYPE_CHECKING:
    from .cas_client import CASClient, UploadProgress
    from .executor import ExecutionResult, FailoverRemoteExecutor
    from .test_cache import CacheEntry

log = logging.getLogger(__name__)

from _build.reapi_retry import REAPI_RETRYABLE_EXIT_CODES, reapi_max_retries

DEFAULT_CONCURRENCY = 16
# REAPI retry budget — shared with `setup.py` via _build/reapi_retry.py to keep
# Bazel build retries and pytest-remote retries on the same policy.
MAX_RETRIES = reapi_max_retries()
MAX_REMOTE_TIMEOUT = 7200
# Session mode runs pytest in phases by gpu(count=N). Keep all counts up to the
# default 4-worker session so mixed PD topologies such as 1+2 GPUs do not fall
# back into the 1-GPU phase.
_GPU_COUNT_TIERS = [1, 2, 3, 4]

_PHASE_LINE_RE = re.compile(r"^>>>PHASE:\S+\s+\d+\s*$")
_NODEID_PROPERTY = "nodeid"


def _get_int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        log.warning("Ignoring invalid %s=%r; using default %d", name, raw, default)
        return default


def _get_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw.lower() not in ("0", "false", "no", "off")


def _load_remote_execution_types():
    from .cas_client import CASClient, UploadProgress
    from .executor import ExecutionResult, FailoverRemoteExecutor

    return CASClient, UploadProgress, ExecutionResult, FailoverRemoteExecutor


def _remote_stream_log_paths(rootdir: Path, key: str) -> Tuple[Path, Path]:
    """Per-remote-run files for live ByteStream stdout/stderr (under .pytest_cache)."""
    slug = re.sub(r"[^\w.-]+", "_", key).strip("_")[:200] or "remote"
    base = rootdir / ".pytest_cache" / "remote_stream_logs" / slug
    base.mkdir(parents=True, exist_ok=True)
    return base / "remote_stdout.log", base / "remote_stderr.log"


def _heartbeat_plugin_shell() -> str:
    return "\n".join(
        [
            "cat > /tmp/rtp_remote_heartbeat_plugin.py << '_RTP_HEARTBEAT_PLUGIN_PY_'",
            "import os, time",
            "",
            "def _touch(label):",
            "    path = os.environ.get('RTP_REMOTE_HEARTBEAT_FILE')",
            "    if not path:",
            "        return",
            "    try:",
            "        with open(path, 'a', encoding='utf-8') as f:",
            "            f.write(f'{int(time.time())} {label}\\n')",
            "    except OSError:",
            "        pass",
            "",
            "def pytest_collection_modifyitems(session, config, items):",
            "    _touch(f'collection items={len(items)}')",
            "",
            "def pytest_runtest_logstart(nodeid, location):",
            "    _touch(f'test_start {nodeid}')",
            "",
            "def pytest_runtest_logreport(report):",
            "    _touch(f'test_report {report.nodeid} {report.when} {report.outcome}')",
            "",
            "def pytest_sessionfinish(session, exitstatus):",
            "    _touch(f'sessionfinish {exitstatus}')",
            "_RTP_HEARTBEAT_PLUGIN_PY_",
            "",
        ]
    )


def _heartbeat_shell(label: str) -> str:
    quoted = shlex.quote(label)
    return (
        'if [ -n "${RTP_REMOTE_HEARTBEAT_FILE:-}" ]; then '
        f'printf \'%s %s\\n\' "$(date +%s)" {quoted} >> "$RTP_REMOTE_HEARTBEAT_FILE" 2>/dev/null || true; '
        "fi"
    )


def _artifact_stream_log_paths(rootdir: Path, key: str) -> Tuple[Path, Path]:
    """CI artifact-visible stdout/stderr files for replayed remote cache hits."""
    slug = re.sub(r"[^\w.-]+", "_", key).strip("_")[:200] or "remote"
    base = rootdir / "bazel-testlogs" / "_pytest_remote_streams" / "cache" / slug
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


def _testcase_nodeid(testcase: ET.Element) -> Optional[str]:
    """Return the pytest nodeid stored by the remote worker junit helper."""
    for prop in testcase.findall("./properties/property"):
        if prop.get("name") == _NODEID_PROPERTY and prop.get("value"):
            return prop.get("value")
    return None


def _iter_junit_suites(root: ET.Element) -> Iterable[ET.Element]:
    if root.tag == "testsuite":
        yield root
    elif root.tag == "testsuites":
        for child in list(root):
            if child.tag == "testsuite":
                yield child


def _count_junit(root: ET.Element) -> Tuple[int, int, int, int, float]:
    tests = failures = errors = skipped = 0
    total_time = 0.0
    for testcase in root.iter("testcase"):
        tests += 1
        total_time += _testcase_time(testcase)
        if testcase.find("failure") is not None:
            failures += 1
        if testcase.find("error") is not None:
            errors += 1
        if testcase.find("skipped") is not None:
            skipped += 1
    return tests, failures, errors, skipped, total_time


def _serialize_testcase(testcase: ET.Element) -> str:
    return ET.tostring(testcase, encoding="unicode")


def _testcase_time(testcase: ET.Element) -> float:
    try:
        return float(testcase.get("time", "0") or "0")
    except ValueError:
        return 0.0


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
        "--remote-executor-failover",
        dest="remote_executor_failover",
        action="store_true",
        default=_get_bool_env("RTP_REMOTE_EXECUTOR_FAILOVER", True),
        help="Fail over between executor IPs resolved from the executor host (default: on)",
    )
    g.addoption(
        "--remote-no-executor-failover",
        dest="remote_executor_failover",
        action="store_false",
        help="Disable executor IP failover for remote executions",
    )
    g.addoption(
        "--remote-executor-failover-max",
        type=int,
        default=_get_int_env("RTP_REMOTE_EXECUTOR_FAILOVER_MAX", 3),
        help="Max executor IP failovers per remote action (default: 3)",
    )
    g.addoption(
        "--remote-executor-dns-refresh-seconds",
        type=int,
        default=_get_int_env("RTP_REMOTE_EXECUTOR_DNS_REFRESH_SECONDS", 60),
        help="Refresh executor DNS resolution after this many seconds (default: 60)",
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
    g.addoption(
        "--remote-test-cache",
        dest="remote_test_cache",
        action="store_true",
        default=True,
        help="Enable test result caching for remote tests (default: on)",
    )
    g.addoption(
        "--remote-no-test-cache",
        dest="remote_test_cache",
        action="store_false",
        help="Disable test result caching (force re-run all tests)",
    )
    g.addoption(
        "--remote-test-cache-ttl",
        type=int,
        default=_get_int_env("REMOTE_TEST_CACHE_TTL", 7),
        help="Test result cache TTL in days (default: 7)",
    )


def _resolve_endpoints(config) -> tuple:
    """Resolve executor/cas endpoints from CLI options + pyproject defaults."""
    rootdir = Path(config.rootdir)
    executor_ep = config.getoption("--remote-executor")
    cas_ep = config.getoption("--remote-cas")
    fallback_executor_ep = None
    if not executor_ep or not cas_ep:
        env = config.getoption("--remote-env")
        defaults = resolve_default_reapi_endpoint_config(rootdir, env=env)
        if not executor_ep:
            executor_ep = defaults.executor
            fallback_executor_ep = defaults.fallback_executor
        cas_ep = cas_ep or defaults.cas
    if not executor_ep or not cas_ep:
        raise pytest.UsageError(
            "--remote requires REAPI endpoints; pass --remote-executor/--remote-cas "
            "or configure [tool.rtp-llm.remote] in pyproject.toml"
        )
    return executor_ep, cas_ep, fallback_executor_ep


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
    ci_profile = config.getoption("--rtp-ci-profile", default=None)
    if (
        use_session
        and ci_profile
        and not str(ci_profile).replace("-", "_").startswith("py_ut")
    ):
        raise pytest.UsageError(
            "--remote-session is only supported for py-ut profiles; "
            f"profile {ci_profile!r} must use per-test --remote"
        )
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
        (
            self._executor_ep,
            self._cas_ep,
            self._fallback_executor_ep,
        ) = _resolve_endpoints(config)
        self.metadata = _parse_metadata(config)
        raw_requested_timeout = config.getoption("--remote-timeout")
        requested_timeout = min(raw_requested_timeout, MAX_REMOTE_TIMEOUT)
        if raw_requested_timeout > MAX_REMOTE_TIMEOUT:
            log.warning(
                "Clamping --remote-timeout from %d to %d to satisfy current REAPI server limit",
                raw_requested_timeout,
                MAX_REMOTE_TIMEOUT,
            )
        self.ci_profile = config.getoption("--rtp-ci-profile", default=None)
        remote_timeout_override = (
            requested_timeout if requested_timeout < MAX_REMOTE_TIMEOUT else None
        )
        self.timeout_policy: RemoteTimeoutPolicy = select_remote_timeout_policy(
            self.ci_profile,
            per_test=(mode == RemoteDispatchMode.PER_TEST),
            remote_timeout_override=remote_timeout_override,
        )
        self.timeout = self.timeout_policy.session_budget_seconds
        log.info(
            "Remote timeout policy: profile=%s class=%s session=%ds action=%ds rpc=%ds "
            "supervisor=%ds pytest=%ds queued=%ds min_retry_remaining=%ds",
            self.ci_profile or "none",
            self.timeout_policy.profile_class,
            self.timeout_policy.session_budget_seconds,
            self.timeout_policy.action_timeout_seconds,
            self.timeout_policy.rpc_timeout_seconds,
            self.timeout_policy.supervisor_timeout_seconds,
            self.timeout_policy.pytest_timeout_seconds,
            self.timeout_policy.queued_timeout_seconds,
            self.timeout_policy.min_retry_remaining_seconds,
        )
        self._cas_workers = config.getoption("--remote-cas-upload-workers")
        self.cas: Optional[CASClient] = None
        self.executor: Optional[FailoverRemoteExecutor] = None
        self._executor_failover = config.getoption("remote_executor_failover")
        self._executor_failover_max = config.getoption("--remote-executor-failover-max")
        self._executor_dns_refresh_seconds = config.getoption(
            "--remote-executor-dns-refresh-seconds"
        )

        self._remote_file_handler: Optional[logging.Handler] = None
        log_file_opt = config.getoption("--remote-log-file", default=None)
        if log_file_opt:
            self._attach_remote_log_file(Path(log_file_opt))

        # Output collection (TEST_UNDECLARED_OUTPUTS_DIR)
        self._collect_outputs = config.getoption("--remote-collect-outputs")
        _odir = config.getoption("--remote-outputs-dir", default=None)
        self._outputs_dir = (
            Path(_odir)
            if _odir
            else (self.rootdir / "bazel-testlogs" / "pytest" / "remote_outputs")
        )
        self._outputs_max_bytes = (
            config.getoption("--remote-outputs-max-mb") * 1024 * 1024
        )

        # Test result cache
        self._test_cache_enabled = config.getoption("remote_test_cache")
        self._test_cache_ttl = config.getoption("--remote-test-cache-ttl")
        self._test_cache = None  # lazy init after CAS client is ready
        self._cache_manifest = None
        self._cache_input_root = None
        self._cached_items: Dict[str, "CacheEntry"] = {}  # nodeid -> CacheEntry
        self._session_cached_entries: Dict[str, "CacheEntry"] = {}

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
            self.no_cache = config.getoption("--remote-no-cache")
            self._result = None
            self._gpu_request: Optional[GPURequest] = None

    def _supervised_command(self, inner_script: str, key: str) -> List[str]:
        slug = re.sub(r"[^\w.-]+", "_", key).strip("_")[:80] or "remote"
        session_id = f"rtp-remote-{slug}-{os.getpid()}-{int(time.time())}"
        heartbeat_file = f"/tmp/rtp_remote_heartbeat_{session_id}.log"
        supervised = (
            f"export RTP_REMOTE_SESSION_ID={shlex.quote(session_id)}; "
            f"export RTP_REMOTE_HEARTBEAT_FILE={shlex.quote(heartbeat_file)}; "
            "/opt/conda310/bin/python rtp_llm/test/remote_tests/action_supervisor.py "
            f"--timeout {self.timeout_policy.supervisor_timeout_seconds} "
            f"--heartbeat-file {shlex.quote(heartbeat_file)} "
            f"--heartbeat-timeout {self.timeout_policy.heartbeat_stall_seconds} "
            f"-- bash -lc {shlex.quote(inner_script)}"
        )
        return ["bash", "-c", supervised]

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
        CASClient, _, _, FailoverRemoteExecutor = _load_remote_execution_types()
        cas = CASClient(
            self._cas_ep,
            self.metadata,
            batch_upload_workers=self._cas_workers,
        )
        self.cas = cas
        self.executor = FailoverRemoteExecutor(
            self._executor_ep,
            cas,
            self.metadata,
            enabled=self._executor_failover,
            max_failovers=self._executor_failover_max,
            dns_refresh_seconds=self._executor_dns_refresh_seconds,
            fallback_executor_endpoint=self._fallback_executor_ep,
        )

    def _ensure_test_cache(self) -> None:
        if self._test_cache is not None or not self._test_cache_enabled:
            return
        from .action_cache_client import ActionCacheClient
        from .test_cache import RemoteTestCache

        ac_client = None
        try:
            ac_client = ActionCacheClient(self._cas_ep, self.metadata)
        except Exception as exc:
            log.warning("Failed to create ActionCache client: %s", exc)

        self._test_cache = RemoteTestCache(
            cache_dir=self.rootdir / ".pytest_cache" / "remote_test_cache",
            action_cache_client=ac_client,
            cas_client=self.cas,
            ttl_days=self._test_cache_ttl,
        )

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
        from .remote_exec_rtp import _safe_rel_to_rootdir

        test_id = item.name
        gpu_req = resolve_item_gpu_request(item)
        gpu_count = str(gpu_req.gpu_count)
        gpu_env_exports = (
            f"export GPU_COUNT={shlex.quote(gpu_count)}; "
            f"export WORLD_SIZE={shlex.quote(gpu_count)}; "
            f"export GPU_COUNT_PER_WORKER={shlex.quote(gpu_count)}; "
        )
        # Use _safe_rel_to_rootdir so sibling internal_source suite files map
        # through the uploaded internal_source/ symlink on the worker.
        test_path = _safe_rel_to_rootdir(Path(str(item.fspath)).resolve(), self.rootdir)
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
            f"{_heartbeat_plugin_shell()}"
            f"{outputs_prefix}"
            f"{gpu_env_exports}"
            "echo \">>>RTP_REMOTE_HOST_IP $(hostname -I 2>/dev/null | awk '{print $1}')\"; "
            f"{_heartbeat_shell('per_test_pytest_start')}; "
            'echo ">>>PHASE:pytest_start $(date +%s)"; '
            'export PYTHONPATH="/tmp:$PWD:${PYTHONPATH:-}"; '
            f"python rtp_llm/test/utils/device_resource.py "
            f"python -m pytest -p rtp_remote_heartbeat_plugin -xvs --tb=long "
            f"--timeout={self.timeout_policy.pytest_timeout_seconds} "
            f"--override-ini='addopts=' {ignore_args} "
            f"{mark_arg}"
            f"-k {shlex.quote(test_id)} {shlex.quote(test_path)} 2>&1; ec=$?; "
            "echo EXIT_CODE=$ec; "
            f"{_heartbeat_shell('per_test_pytest_end')}; "
            'echo ">>>PHASE:pytest_end $(date +%s)"; '
            f"{outputs_postscript}"
            "exit $ec"
        )
        return self._supervised_command(
            f"{runtime.remote_setup_prefix}{run_cmd}", item.nodeid
        )

    _GPU_LOCK_RETRY_PATTERNS = (
        b"GpuLockTimeoutError",
        b"GPU lock timed out",
    )

    @staticmethod
    def _is_gpu_lock_failure(result) -> bool:
        """Check if a remote execution failed due to GPU lock contention."""
        if result.exit_code == 0:
            return False
        for blob in (result.stdout_raw, result.stderr_raw):
            if blob:
                for pat in RemoteREAPIPlugin._GPU_LOCK_RETRY_PATTERNS:
                    if pat in blob:
                        return True
        return False

    def _execute_with_retry(self, **kwargs) -> ExecutionResult:
        self._ensure_remote_clients()
        last_result = None
        for attempt in range(MAX_RETRIES + 1):
            global_deadline_epoch = kwargs.get("global_deadline_epoch")
            if global_deadline_epoch is not None and attempt > 0:
                remaining = int(global_deadline_epoch - time.time())
                min_remaining = int(
                    kwargs.get("min_retry_remaining_seconds")
                    or self.timeout_policy.min_retry_remaining_seconds
                )
                if remaining < min_remaining:
                    log.warning(
                        "[RETRY] refusing retry: remaining=%ss below threshold=%ss",
                        remaining,
                        min_remaining,
                    )
                    break
            result = self.executor.execute(**kwargs)
            # REAPI codes 34/38 are transient infra failures — same policy as
            # bazel build (setup.py imports the same constant from _build/reapi_retry).
            should_retry = (
                result.exit_code == -1
                or result.exit_code in REAPI_RETRYABLE_EXIT_CODES
                or self._is_gpu_lock_failure(result)
            )
            if not should_retry:
                return result
            last_result = result
            if attempt < MAX_RETRIES:
                wait = 2 ** (attempt + 1)
                if self._is_gpu_lock_failure(result):
                    reason = "GPU lock contention"
                elif result.exit_code in REAPI_RETRYABLE_EXIT_CODES:
                    reason = f"REAPI transient (exit={result.exit_code})"
                else:
                    reason = "REAPI infra error"
                log.warning(
                    "[RETRY] %s (%s) after %ds (attempt %d/%d)",
                    (
                        kwargs.get("command", ["?"])[2][:60]
                        if len(kwargs.get("command", [])) > 2
                        else "?"
                    ),
                    reason,
                    wait,
                    attempt + 1,
                    MAX_RETRIES,
                )
                time.sleep(wait)
        return last_result

    @pytest.hookimpl(tryfirst=True)
    def pytest_ignore_collect(self, collection_path, config):
        """In session mode, skip local collection entirely.

        The remote worker will re-collect with its own GPU environment.
        This avoids ImportError noise from CUDA extensions on CPU-only
        CI controller nodes.  Unlike pytest_collect_file (where returning
        None falls through to the default collector), returning True here
        unconditionally prevents pytest from importing the file.
        """
        if self.mode == RemoteDispatchMode.SESSION:
            return True

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

        # --- Test cache: load manifest and split cached vs uncached ---
        uncached_items = self._remote_items
        if self._test_cache_enabled and self._input_root:
            self._ensure_test_cache()
            if self._test_cache is not None:
                self._cache_input_root = self._input_root
                self._cache_manifest = self._test_cache.load_manifest(self._input_root)
                cached_count = 0
                new_uncached: list = []
                for item in self._remote_items:
                    gpu_req = resolve_item_gpu_request(item)
                    entry = self._test_cache.lookup(
                        self._cache_manifest,
                        item.nodeid,
                        gpu_req.gpu_type,
                        gpu_req.gpu_count,
                    )
                    if entry is not None:
                        self._cached_items[item.nodeid] = entry
                        cached_count += 1
                    else:
                        new_uncached.append(item)
                uncached_items = new_uncached
                tw = self.config.get_terminal_writer()
                cache_src = (
                    "ActionCache" if self._cache_manifest.get("_from_ac") else "local"
                )
                msg = (
                    f"Test cache: {cached_count} cached, {len(uncached_items)} to run"
                    f" (source: {cache_src})"
                )
                log.info(msg)
                if tw:
                    tw.line(msg)

        if not uncached_items:
            log.info("All %d tests satisfied from cache", len(self._cached_items))
            return

        workers = min(len(uncached_items), self.concurrency)
        self._pool = ThreadPoolExecutor(max_workers=workers)
        for item in uncached_items:
            gpu_req = resolve_item_gpu_request(item)
            runtime = build_runtime_config(
                self.rootdir,
                gpu_req,
                extra_env=resolve_ci_profile_remote_env(self.rootdir, self.ci_profile),
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
                from .output_collector import (
                    make_output_collection_env,
                    make_output_files_decl,
                )

                env_vars.update(make_output_collection_env())
                output_files = make_output_files_decl()
            future = self._pool.submit(
                self._execute_with_retry,
                command=cmd,
                input_root_digest=self._input_root,
                env_vars=env_vars,
                platform_properties=runtime.platform_properties,
                timeout=self.timeout,
                action_timeout_seconds=self.timeout_policy.action_timeout_seconds,
                rpc_timeout_seconds=self.timeout_policy.rpc_timeout_seconds,
                queued_timeout_seconds=self.timeout_policy.queued_timeout_seconds,
                global_deadline_epoch=time.time()
                + self.timeout_policy.session_budget_seconds,
                min_retry_remaining_seconds=self.timeout_policy.min_retry_remaining_seconds,
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
            # No tests collected locally (e.g. platform host lacks GPU hardware).
            # Try to infer GPU type from --remote-gpu-type, CI profile, or markexpr.
            gpu_type = self.gpu_type_override
            gpu_type_source = "--remote-gpu-type" if gpu_type else ""
            if not gpu_type:
                gpu_type = resolve_ci_profile_gpu_type(self.rootdir, self.ci_profile)
                if gpu_type:
                    gpu_type_source = f"profile:{self.ci_profile}"
            if not gpu_type:
                markexpr = getattr(config.option, "markexpr", "") or ""
                gpu_type = infer_gpu_type_from_markexpr(markexpr)
                if gpu_type:
                    gpu_type_source = "markexpr"
            if not gpu_type:
                log.warning(
                    "Session mode: no tests collected — nothing to run remotely"
                )
                return
            log.info(
                "Session mode: 0 tests collected locally, gpu_type=%s "
                "(from %s) — submitting session for remote re-collection",
                gpu_type,
                gpu_type_source,
            )
        else:
            gpu_type = resolve_gpu_type_from_items(
                items, override=self.gpu_type_override
            )

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

        # --- Cache hit path ---
        if item.nodeid in self._cached_items:
            cache_entry = self._cached_items[item.nodeid]
            result = self._reconstruct_result_from_cache(
                cache_entry,
                require_outputs=self._collect_outputs,
            )
            if result is not None:
                log.info(
                    "[CACHED] %s (originally %.1fs on %s)",
                    item.nodeid,
                    cache_entry.duration_s,
                    cache_entry.worker_ip or "n/a",
                )
                self._report_per_test(item, result, cached=True)
                return True
            # CAS blob eviction — fall through to re-dispatch
            log.warning(
                "Cache hit but CAS blob evicted for %s, re-dispatching",
                item.nodeid,
            )
            del self._cached_items[item.nodeid]
            self._dispatch_single_item(item)

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

    def _reconstruct_result_from_cache(
        self,
        entry: "CacheEntry",
        *,
        require_outputs: bool = False,
    ):
        """Reconstruct ExecutionResult from cached CAS digests.

        Returns None if critical blobs are missing from CAS (evicted).
        """
        from . import remote_execution_pb2 as re_pb2
        from .executor import ExecutionResult

        stdout_digest = entry.stdout_digest.to_proto() if entry.stdout_digest else None
        stderr_digest = entry.stderr_digest.to_proto() if entry.stderr_digest else None
        output_files = {k: v.to_proto() for k, v in entry.output_files.items()}
        if require_outputs and not output_files:
            log.info("Cache entry for %s has no output archive", entry.test_nodeid)
            return None

        digests_to_check = []
        if stdout_digest:
            digests_to_check.append(stdout_digest)
        if stderr_digest:
            digests_to_check.append(stderr_digest)
        if require_outputs:
            digests_to_check.extend(output_files.values())

        # Verify referenced blobs still exist in CAS. If any critical blob has
        # been evicted, treat the cache entry as a miss and re-run remotely.
        if digests_to_check and self.cas:
            missing = self.cas._find_missing(digests_to_check)
            if missing:
                log.info(
                    "Cache entry for %s missing CAS blobs: %s",
                    entry.test_nodeid,
                    ",".join(sorted(missing))[:200],
                )
                return None

        return ExecutionResult(
            exit_code=entry.exit_code,
            stdout_digest=stdout_digest,
            stderr_digest=stderr_digest,
            output_files=output_files,
            worker_host_ip=entry.worker_ip,
            cached_result=True,
        )

    def _dispatch_single_item(self, item) -> None:
        """Late-dispatch a single item that was initially cached but had CAS eviction."""
        self._ensure_remote_clients()
        gpu_req = resolve_item_gpu_request(item)
        runtime = build_runtime_config(
            self.rootdir,
            gpu_req,
            extra_env=resolve_ci_profile_remote_env(self.rootdir, self.ci_profile),
            input_root_hash=self._input_root.hash if self._input_root else None,
        )
        cmd = self._build_command(item, runtime)

        def _on_stage(stage: str, op_name: str, nodeid=item.nodeid):
            log.info("[REMOTE %s] stage=%s op=%s", nodeid, stage, (op_name or "")[:48])

        stdout_log, stderr_log = _remote_stream_log_paths(self.rootdir, item.nodeid)
        env_vars = dict(runtime.env_vars)
        output_files = None
        if self._collect_outputs:
            from .output_collector import (
                make_output_collection_env,
                make_output_files_decl,
            )

            env_vars.update(make_output_collection_env())
            output_files = make_output_files_decl()
        future_result = self._execute_with_retry(
            command=cmd,
            input_root_digest=self._input_root,
            env_vars=env_vars,
            platform_properties=runtime.platform_properties,
            timeout=self.timeout,
            action_timeout_seconds=self.timeout_policy.action_timeout_seconds,
            rpc_timeout_seconds=self.timeout_policy.rpc_timeout_seconds,
            queued_timeout_seconds=self.timeout_policy.queued_timeout_seconds,
            global_deadline_epoch=time.time()
            + self.timeout_policy.session_budget_seconds,
            min_retry_remaining_seconds=self.timeout_policy.min_retry_remaining_seconds,
            output_files=output_files,
            on_stage=_on_stage,
            stream_stdout_file=stdout_log,
            stream_stderr_file=stderr_log,
        )
        # Store as a resolved future
        f: Future = Future()
        f.set_result(future_result)
        self._futures[item.nodeid] = f

    def pytest_unconfigure(self, config):
        # Flush test cache
        if (
            self._test_cache is not None
            and self._cache_manifest is not None
            and self._cache_input_root is not None
        ):
            self._test_cache.flush(self._cache_input_root, self._cache_manifest)

        fh = self._remote_file_handler
        if fh is not None:
            pkg = logging.getLogger("rtp_llm.test.remote_tests")
            pkg.removeHandler(fh)
            fh.close()
            self._remote_file_handler = None
        if self.mode == RemoteDispatchMode.PER_TEST and self._pool:
            self._pool.shutdown(wait=False)

        # Close gRPC channels
        if hasattr(self, "cas") and self.cas is not None:
            self.cas.close()
        if self._test_cache is not None:
            self._test_cache.close()

    def _write_cached_stream_logs(self, key: str, stdout: str, stderr: str) -> None:
        stdout_path, stderr_path = _artifact_stream_log_paths(self.rootdir, key)
        stdout_path.write_text(stdout or "", encoding="utf-8")
        stderr_path.write_text(stderr or "", encoding="utf-8")

    def _report_per_test(self, item, result: ExecutionResult, cached: bool = False):
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

        if not cached:
            _display_phase_breakdown(stdout, stderr, None)

        worker_ip = result.worker_host_ip or extract_remote_worker_ip(stdout)
        if worker_ip:
            log.info("[remote-worker] %s host_ip=%s", item.nodeid, worker_ip)

        # Download TEST_UNDECLARED_OUTPUTS_DIR artifacts from CAS
        if self._collect_outputs:
            from .output_collector import download_and_extract

            slug = re.sub(r"[^\w.-]+", "_", item.nodeid).strip("_")[:200] or "test"
            out_path = download_and_extract(
                self.cas,
                result,
                self._outputs_dir / slug,
                max_bytes=self._outputs_max_bytes,
            )
            if out_path:
                log.info("[remote-outputs] %s -> %s", item.nodeid, out_path)

        tw = self.config.get_terminal_writer()
        if cached:
            self._write_cached_stream_logs(f"per_test_{item.nodeid}", stdout, stderr)
            entry = self._cached_items.get(item.nodeid)
            dur = f"{entry.duration_s:.1f}s" if entry else "?"
            wip = entry.worker_ip if entry else "n/a"
            cache_msg = f"(cached, originally {dur} on {wip})"
            log.info("[CACHED] %s %s", item.nodeid, cache_msg)
            if tw:
                tw.line(f"{item.nodeid} {cache_msg}")
        else:
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

                # --- Store successful result in test cache ---
                if (
                    not cached
                    and self._test_cache is not None
                    and self._cache_manifest is not None
                ):
                    self._store_test_result_in_cache(item, result, worker_ip)
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

    def _store_test_result_in_cache(
        self, item, result: ExecutionResult, worker_ip: Optional[str]
    ) -> None:
        """Store a successful test result in the cache manifest."""
        from .test_cache import CacheEntry, DigestRef

        gpu_req = resolve_item_gpu_request(item)
        stdout_ref = self._test_cache.ensure_stdout_digest(
            result.stdout_raw, result.stdout_digest
        )
        stderr_ref = self._test_cache.ensure_stderr_digest(
            result.stderr_raw, result.stderr_digest
        )
        output_refs = {}
        for fname, digest in result.output_files.items():
            ref = DigestRef.from_proto(digest)
            if ref:
                output_refs[fname] = ref

        # Estimate duration from phase markers
        phases = parse_phase_list(
            result.stdout_raw.decode("utf-8", errors="replace")
            if result.stdout_raw
            else ""
        )
        duration = 0.0
        if len(phases) >= 2:
            duration = phases[-1][1] - phases[0][1]

        entry = CacheEntry(
            test_nodeid=item.nodeid,
            gpu_type=gpu_req.gpu_type,
            gpu_count=gpu_req.gpu_count,
            result="passed",
            exit_code=0,
            timestamp=time.time(),
            duration_s=duration,
            worker_ip=worker_ip,
            stdout_digest=stdout_ref,
            stderr_digest=stderr_ref,
            output_files=output_refs,
        )
        self._test_cache.store(self._cache_manifest, entry)
        log.info("[CACHE_STORE] %s (%.1fs)", item.nodeid, duration)

    def _session_cache_scope(self) -> str:
        markexpr = getattr(self.config.option, "markexpr", "") or ""
        keyword = getattr(self.config.option, "keyword", "") or ""
        pytest_args = (self.pytest_args or "").strip()
        return (
            f"profile={self.ci_profile or ''}|markexpr={markexpr}|"
            f"keyword={keyword}|pytest_args={pytest_args}"
        )

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
            self.rootdir,
            gpu_req,
            extra_env=resolve_ci_profile_remote_env(self.rootdir, self.ci_profile),
            input_root_hash=input_root.hash,
        )

        # --- Session test cache: load manifest, build deselect args ---
        deselect_nodeids: List[str] = []
        if self._test_cache_enabled:
            self._ensure_test_cache()
            if self._test_cache is not None:
                self._cache_input_root = input_root
                self._cache_manifest = self._test_cache.load_manifest(input_root)
                if self._cache_manifest.get("tests"):
                    n_cached = len(self._cache_manifest["tests"])
                    deselect_nodeids = self._build_session_deselect_args(
                        self._cache_manifest
                    )
                    log.info(
                        "Session test cache: %d cached entries, %d to deselect",
                        n_cached,
                        len(deselect_nodeids),
                    )

        cmd = self._build_session_command(
            self.pytest_args,
            runtime,
            self.ci_profile,
            extra_deselect_nodeids=deselect_nodeids,
        )
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

        # Include profile name in the stream-log slug so two parallel
        # `--remote-session` runs (e.g. py_ut_sm8x + py_ut_sm9x) don't
        # clobber each other's stdout/stderr files. Falls back to plain
        # "session" when no profile is set.
        _slug_key = f"session_{self.ci_profile}" if self.ci_profile else "session"
        sess_out, sess_err = _remote_stream_log_paths(self.rootdir, _slug_key)
        env_vars = dict(runtime.env_vars)
        output_files = None
        if self._collect_outputs:
            from .output_collector import (
                make_output_collection_env,
                make_output_files_decl,
            )

            env_vars.update(make_output_collection_env())
            output_files = make_output_files_decl()
        # Session path goes through _execute_with_retry so transient REAPI failures
        # (exit_code==-1, codes 34/38) and GPU lock contention get the same 2x
        # exponential-backoff retry as the per-test path. Without this, a single
        # 504 from the REAPI gateway tanks the whole pytest session.
        result = self._execute_with_retry(
            command=cmd,
            input_root_digest=input_root,
            env_vars=env_vars,
            platform_properties=runtime.platform_properties,
            timeout=self.timeout,
            action_timeout_seconds=self.timeout_policy.action_timeout_seconds,
            rpc_timeout_seconds=self.timeout_policy.rpc_timeout_seconds,
            queued_timeout_seconds=self.timeout_policy.queued_timeout_seconds,
            global_deadline_epoch=time.time()
            + self.timeout_policy.session_budget_seconds,
            min_retry_remaining_seconds=self.timeout_policy.min_retry_remaining_seconds,
            output_files=output_files,
            on_stage=_on_stage,
            stream_stdout_file=sess_out,
            stream_stderr_file=sess_err,
            no_cache=self.no_cache,
        )

        if self._collect_outputs:
            from .output_collector import download_and_extract

            out_path = download_and_extract(
                self.cas,
                result,
                self._outputs_dir / "session",
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

    def _cached_testcases_for_session(self) -> List[ET.Element]:
        cached_cases: List[ET.Element] = []
        for entry in self._session_cached_entries.values():
            if not entry.junit_testcase_xml:
                continue
            try:
                testcase = ET.fromstring(entry.junit_testcase_xml)
            except ET.ParseError as exc:
                log.debug("Skipping malformed cached junit testcase: %s", exc)
                continue
            cached_cases.append(testcase)
        return cached_cases

    def _merge_session_junit(self, fresh_xml: str) -> Tuple[str, int, int]:
        """Merge fresh remote junit with locally replayed cached testcases."""
        fresh_suites: List[ET.Element] = []
        fresh_nodeids = set()
        if fresh_xml:
            try:
                fresh_root = ET.fromstring(fresh_xml)
                fresh_suites = list(_iter_junit_suites(fresh_root))
                for testcase in fresh_root.iter("testcase"):
                    nodeid = _testcase_nodeid(testcase)
                    if nodeid:
                        fresh_nodeids.add(nodeid)
            except ET.ParseError as exc:
                log.warning("Failed to parse remote junit XML: %s", exc)

        cached_cases = []
        for testcase in self._cached_testcases_for_session():
            nodeid = _testcase_nodeid(testcase)
            if nodeid and nodeid in fresh_nodeids:
                continue
            cached_cases.append(testcase)

        if not fresh_suites and not cached_cases:
            return "", 0, 0

        root = ET.Element("testsuites")
        for suite in fresh_suites:
            root.append(suite)

        if cached_cases:
            cache_suite = ET.Element(
                "testsuite",
                {
                    "name": "remote-cache",
                    "tests": str(len(cached_cases)),
                    "failures": "0",
                    "errors": "0",
                    "skipped": "0",
                    "time": f"{sum(_testcase_time(tc) for tc in cached_cases):.3f}",
                },
            )
            for testcase in cached_cases:
                cache_suite.append(testcase)
            root.append(cache_suite)

        tests, failures, errors, skipped, total_time = _count_junit(root)
        root.set("tests", str(tests))
        root.set("failures", str(failures))
        root.set("errors", str(errors))
        root.set("skipped", str(skipped))
        root.set("time", f"{total_time:.3f}")
        return ET.tostring(root, encoding="unicode"), tests, len(cached_cases)

    def _replay_cached_outputs(self) -> None:
        if not self._collect_outputs or not self._session_cached_entries:
            return
        from .output_collector import OUTPUTS_ARCHIVE, download_and_extract
        from .executor import ExecutionResult

        seen_archives = set()
        seen_streams = set()
        for entry in self._session_cached_entries.values():
            result = self._reconstruct_result_from_cache(
                entry,
                require_outputs=True,
            )
            if result is None:
                continue
            archive = entry.output_files.get(OUTPUTS_ARCHIVE)
            if archive is not None and archive.hash not in seen_archives:
                seen_archives.add(archive.hash)
                one_file_result = ExecutionResult(
                    exit_code=0,
                    output_files={OUTPUTS_ARCHIVE: archive.to_proto()},
                )
                out_path = download_and_extract(
                    self.cas,
                    one_file_result,
                    self._outputs_dir / "cached" / archive.hash[:16],
                    max_bytes=self._outputs_max_bytes,
                )
                if out_path:
                    log.info(
                        "[remote-outputs] cached %s -> %s",
                        archive.hash[:12],
                        out_path,
                    )
            stream_digest_key = (
                entry.stdout_digest.hash if entry.stdout_digest else "",
                entry.stderr_digest.hash if entry.stderr_digest else "",
            )
            if stream_digest_key in seen_streams:
                continue
            seen_streams.add(stream_digest_key)
            stdout = (
                self.executor.download_output(entry.stdout_digest.to_proto())
                if entry.stdout_digest
                else ""
            )
            stderr = (
                self.executor.download_output(entry.stderr_digest.to_proto())
                if entry.stderr_digest
                else ""
            )
            stream_key = (
                f"session_{self.ci_profile or 'default'}_"
                f"{stream_digest_key[0][:12] or 'stdout'}_"
                f"{stream_digest_key[1][:12] or 'stderr'}"
            )
            self._write_cached_stream_logs(stream_key, stdout, stderr)

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

        xml_content = ""
        if "<<<JUNIT_XML>>>" in stdout and "<<<END_JUNIT_XML>>>" in stdout:
            xml_content = (
                stdout.split("<<<JUNIT_XML>>>", 1)[1]
                .split("<<<END_JUNIT_XML>>>", 1)[0]
                .strip()
            )
            if xml_content:
                # Store passing tests from junit XML in cache manifest
                if self._test_cache is not None and self._cache_manifest is not None:
                    self._store_session_results_from_junit(xml_content, result)
            stdout = stdout.split("<<<JUNIT_XML>>>", 1)[0]

        merged_xml, merged_tests, replayed_cached = self._merge_session_junit(
            xml_content
        )
        if merged_xml:
            junit_path = self.rootdir / "bazel-testlogs" / "pytest" / "test.xml"
            junit_path.parent.mkdir(parents=True, exist_ok=True)
            junit_path.write_text(merged_xml)
            log.info(
                "Wrote remote junitxml to %s (%d bytes, replayed_cached=%d)",
                junit_path,
                len(merged_xml),
                replayed_cached,
            )

        self._replay_cached_outputs()

        display_stdout = _strip_phase_marker_lines(stdout)
        if display_stdout:
            print("::group::Remote Worker Output (stdout)")
            print(display_stdout, end="" if display_stdout.endswith("\n") else "\n")
            print("::endgroup::")
        stderr_disp = _strip_phase_marker_lines(stderr)
        if stderr_disp:
            print("::group::Remote Worker Output (stderr)", file=_sys.stderr)
            print(stderr_disp, file=_sys.stderr)
            print("::endgroup::", file=_sys.stderr)

        exit_code = result.exit_code
        # Do NOT let the remote worker's EXIT_CODE override a non-OK REAPI
        # response status.  A status code != 0 means the execution itself failed
        # at the REAPI layer, regardless of what the worker claims.
        if (
            result.response_status_code is not None
            and result.response_status_code != 0
        ):
            exit_code = result.exit_code or 1
        elif "EXIT_CODE=" in stdout:
            try:
                exit_code = int(stdout.rsplit("EXIT_CODE=", 1)[1].strip().split()[0])
            except (ValueError, IndexError):
                pass
        if exit_code == 5 and merged_tests > 0 and replayed_cached > 0:
            log.info(
                "Remote pytest returned no-tests-ran, but %d cached tests were replayed; treating as success",
                replayed_cached,
            )
            exit_code = 0

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

    def _store_session_results_from_junit(
        self, xml_content: str, result: ExecutionResult
    ) -> None:
        """Parse junit XML and store passing test results in the cache manifest."""
        from .test_cache import CacheEntry, DigestRef

        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError as exc:
            log.warning("Failed to parse junit XML for cache: %s", exc)
            return

        # Capture session-level stdout/stderr digests
        stdout_ref = self._test_cache.ensure_stdout_digest(
            result.stdout_raw, result.stdout_digest
        )
        stderr_ref = self._test_cache.ensure_stderr_digest(
            result.stderr_raw, result.stderr_digest
        )
        output_refs = {}
        for fname, digest in result.output_files.items():
            ref = DigestRef.from_proto(digest)
            if ref:
                output_refs[fname] = ref

        stored = 0
        for testcase in root.iter("testcase"):
            # Skip tests that have failure/error/skipped children
            if (
                testcase.find("failure") is not None
                or testcase.find("error") is not None
                or testcase.find("skipped") is not None
            ):
                continue
            nodeid = _testcase_nodeid(testcase)
            if not nodeid:
                log.debug("Skipping session cache store for testcase without nodeid")
                continue
            time_s = _testcase_time(testcase)
            # Session mode: infer gpu from the session request
            gpu_type = self._gpu_request.gpu_type if self._gpu_request else ""

            entry = CacheEntry(
                test_nodeid=nodeid,
                gpu_type=gpu_type,
                gpu_count=1,
                result="passed",
                exit_code=0,
                timestamp=time.time(),
                duration_s=time_s,
                worker_ip=result.worker_host_ip,
                stdout_digest=stdout_ref,
                stderr_digest=stderr_ref,
                output_files=output_refs,
                junit_testcase_xml=_serialize_testcase(testcase),
                session_profile=self.ci_profile or "",
                session_scope=self._session_cache_scope(),
            )
            self._test_cache.store(self._cache_manifest, entry)
            stored += 1

        log.info("Session cache: stored %d passing test results", stored)

    def _build_session_deselect_args(self, manifest: dict) -> List[str]:
        """Build list of nodeids to deselect from cached test entries."""
        if not manifest or not manifest.get("tests"):
            return []

        self._session_cached_entries = {}
        gpu_type = self._gpu_request.gpu_type if self._gpu_request else ""
        session_profile = self.ci_profile or ""
        session_scope = self._session_cache_scope()
        deselect_nodeids: List[str] = []
        from .test_cache import CacheEntry

        for _key, entry_dict in manifest["tests"].items():
            try:
                entry = CacheEntry.from_dict(entry_dict)
            except Exception:
                continue
            if entry.result != "passed":
                continue
            if entry.gpu_type != gpu_type:
                continue
            if (entry.session_profile or "") != session_profile:
                continue
            if (entry.session_scope or "") != session_scope:
                continue
            if not entry.junit_testcase_xml:
                continue
            if time.time() - entry.timestamp > self._test_cache_ttl * 86400:
                continue
            if (
                self._reconstruct_result_from_cache(
                    entry,
                    require_outputs=self._collect_outputs,
                )
                is None
            ):
                continue
            deselect_nodeids.append(entry.test_nodeid)
            self._session_cached_entries[entry.test_nodeid] = entry

        if deselect_nodeids:
            log.info(
                "Session cache: deselecting %d cached tests", len(deselect_nodeids)
            )
        return deselect_nodeids

    def _build_session_command(
        self,
        pytest_args: str,
        runtime: RemoteRuntimeConfig,
        ci_profile: Optional[str] = None,
        extra_deselect_nodeids: Optional[List[str]] = None,
    ) -> List[str]:
        """Generate a multi-phase bash script that runs tests by gpu_count tier.

        Each phase uses a different GPU_COUNT_PER_WORKER and xdist worker count
        so GPUs are perfectly partitioned with no contention between workers.

        Deselect args are written to a file via heredoc and referenced with
        pytest's @file syntax to avoid exceeding ARG_MAX.
        """
        total_gpus = self.workers
        ignore_args = quote_args(runtime.ignore_args)
        markexpr = getattr(self.config.option, "markexpr", "") or ""
        # Forward client-side -k keyword expression to the remote worker so that
        # `pytest -k 'foo'` only runs matching cases on the worker too. Without
        # this, the worker collects+runs the entire profile suite while the
        # client only reports the filtered subset — wasting GPU minutes.
        keyword = getattr(self.config.option, "keyword", "") or ""
        keyword_arg = f"-k {shlex.quote(keyword)} " if keyword else ""

        outputs_prefix = ""
        outputs_postscript = ""
        if self._collect_outputs:
            from .output_collector import make_mkdir_prefix, make_tar_postscript

            outputs_prefix = make_mkdir_prefix()
            outputs_postscript = make_tar_postscript()

        # Determine phases: only tiers that fit in total_gpus
        higher_tiers = sorted(t for t in _GPU_COUNT_TIERS if t > 1 and total_gpus >= t)
        phases: List[Tuple[int, int, str]] = []
        for tier in _GPU_COUNT_TIERS:
            if total_gpus < tier:
                continue
            n_workers = total_gpus // tier
            if tier == 1:
                excludes = " and ".join(f"not gpu_count_{t}" for t in higher_tiers)
                if markexpr and excludes:
                    phase_mark = f"({markexpr}) and {excludes}"
                elif markexpr:
                    phase_mark = markexpr
                else:
                    phase_mark = excludes
            else:
                phase_mark = (
                    f"({markexpr}) and gpu_count_{tier}"
                    if markexpr
                    else f"gpu_count_{tier}"
                )
            phases.append((tier, n_workers, phase_mark))

        log.info(
            "Multi-phase session: total_gpus=%d, phases=%s",
            total_gpus,
            [(tier, nw) for tier, nw, _ in phases],
        )

        # Resolve CI profile's default_pytest_cli into concrete args for the
        # remote worker (since rtp-ci-profile plugin is blocked there).
        profile_cli_args = ""
        # Resolve profile `paths` and forward as positional collection args.
        # Without this, worker pytest falls back to global [tool.pytest.ini_options]
        # `testpaths` and collects platform-specific tests in the tree → 45+
        # ImportErrors on platform-mismatched test modules.
        #
        # Path-form translation:
        # - Locally, profile paths like "../internal_source/rtp_llm/test/smoke/suites/"
        #   work because cwd is github-opensource/ and ../internal_source/ is the
        #   internal monorepo's sibling directory.
        # - On the worker, file collection follows the symlink
        #   github-opensource/internal_source -> ../internal_source so uploads land
        #   under <work>/internal_source/... (NOT <work>/../internal_source/).
        # - Therefore "../X/" in the local profile must be rewritten to "X/" for
        #   the worker. Plain "rtp_llm/..." paths flow through unchanged.
        profile_paths_args = ""
        if ci_profile:
            try:
                from rtp_llm.test.ci_profile_plugin import (
                    _get_profile,
                    _get_pytest_ci_section,
                )

                section = _get_pytest_ci_section(Path(self.config.rootpath))
                default_cli = (section.get("default_pytest_cli") or "").strip()
                if default_cli:
                    profile_cli_args = default_cli + " "
                prof = _get_profile(Path(self.config.rootpath), ci_profile)
                paths = prof.get("paths") or []
                if isinstance(paths, list) and paths:
                    worker_paths = []
                    for p in paths:
                        # Strip leading "../" so the path resolves against
                        # the symlinked internal_source/ on the worker.
                        wp = p
                        while wp.startswith("../"):
                            wp = wp[3:]
                        worker_paths.append(wp)
                    profile_paths_args = (
                        " ".join(shlex.quote(p) for p in worker_paths) + " "
                    )
            except Exception:
                profile_cli_args = f"-v --tb=short --timeout={self.timeout_policy.pytest_timeout_seconds} "

        # Common pytest arguments shared by all phases
        # Note: profile_arg is NOT forwarded — the rtp-ci-profile plugin is blocked
        # on the remote worker via -p no:rtp-ci-profile. Profile markexpr is already
        # resolved locally and applied via phase_mark. Profile's default_pytest_cli
        # is pre-expanded into profile_cli_args.
        common = (
            f"{pytest_args} {profile_cli_args}"
            f"--continue-on-collection-errors "
            f"--override-ini='addopts=' {ignore_args} "
            f"--tb=short "
            f"--timeout={self.timeout_policy.pytest_timeout_seconds}"
        ).strip()

        # Deselect args via file to avoid ARG_MAX overflow (nodeids can be 100+ chars each,
        # 500+ cached tests => 50KB+ that would overflow bash -c argument limit).
        deselect_file_arg = ""
        deselect_nodeids = extra_deselect_nodeids or []

        # Build multi-phase bash script (newline-separated for heredoc support)
        lines: List[str] = []
        if outputs_prefix:
            lines.append(outputs_prefix.rstrip("; "))
        lines.append(_heartbeat_plugin_shell())
        lines.append("mkdir -p bazel-testlogs/pytest")
        lines.append("cat > /tmp/rtp_remote_nodeid_plugin.py << '_NODEID_PLUGIN_PY_'")
        lines.append("import pytest")
        lines.append("")
        lines.append("@pytest.hookimpl(tryfirst=True, hookwrapper=True)")
        lines.append("def pytest_runtest_makereport(item, call):")
        lines.append("    outcome = yield")
        lines.append("    report = outcome.get_result()")
        lines.append("    if call.when == 'call':")
        lines.append(
            f"        report.user_properties.append(('{_NODEID_PROPERTY}', item.nodeid))"
        )
        lines.append("_NODEID_PLUGIN_PY_")

        if deselect_nodeids:
            lines.append("cat > /tmp/_pytest_deselect.txt << '_DESELECT_EOF_'")
            for nodeid in deselect_nodeids:
                lines.append(f"--deselect={nodeid}")
            lines.append("_DESELECT_EOF_")
            deselect_file_arg = " @/tmp/_pytest_deselect.txt"
            log.info(
                "Session deselect: %d nodeids written to heredoc file (%d bytes cmd saved)",
                len(deselect_nodeids),
                sum(len(f"--deselect={n} ") for n in deselect_nodeids),
            )
        lines.append(
            'echo ">>>RTP_REMOTE_HOST_IP $(hostname -I 2>/dev/null '
            "| awk '{print $1}')\""
        )
        lines.append('echo ">>>PHASE:pytest_start $(date +%s)"')
        lines.append(_heartbeat_shell("session_pytest_start"))
        lines.append("final_ec=0; any_ran=0")
        lines.append('export PYTHONPATH="/tmp:$PWD:${PYTHONPATH:-}"')
        lines.append("export PYTHONFAULTHANDLER=1")
        lines.append("ulimit -c unlimited 2>/dev/null || true")
        lines.append(
            'echo ">>>INHERITED_CVD=${CUDA_VISIBLE_DEVICES:-unset} HVD=${HIP_VISIBLE_DEVICES:-unset}"'
        )

        # Crash diagnostics: all output to /tmp/rtp_crash_diag/ files.
        # Runs before xdist phases to capture import-time crash stacks.
        lines.append(
            "if [ -f rtp_llm/test/utils/crash_diag.py ]; then "
            'echo ">>>CRASH_DIAG_START"; '
            "python rtp_llm/test/utils/crash_diag.py --layers LG 2>&1; "
            'echo ">>>CRASH_DIAG_END"; '
            "fi"
        )

        for tier, n_workers, phase_mark in phases:
            mark_arg = f"-m {shlex.quote(phase_mark)} " if phase_mark else ""
            lines.append(
                f'echo "--- Phase: {tier}-GPU tests, {n_workers} workers ---"; '
                f"{_heartbeat_shell(f'phase_{tier}gpu_start')}; "
                f"export GPU_COUNT={total_gpus}; "
                "unset WORLD_SIZE; "
                f"export GPU_COUNT_PER_WORKER={tier}; "
                f"python rtp_llm/test/utils/device_resource.py "
                f"python -m pytest -p no:remote-gpu -p no:rtp-ci-profile "
                f"-p rtp_remote_nodeid_plugin -p rtp_remote_heartbeat_plugin {common} "
                f"{profile_paths_args}"
                f"{mark_arg}"
                f"{keyword_arg}"
                f"-n {n_workers} "
                f"--junitxml=bazel-testlogs/pytest/test_{tier}gpu.xml"
                f"{deselect_file_arg} "
                f"2>&1; ec=$?; "
                f"[ $ec -ne 5 ] && any_ran=1; "
                f"[ $ec -ne 0 ] && [ $ec -ne 5 ] && [ $final_ec -eq 0 ] && final_ec=$ec; "
                f"{_heartbeat_shell(f'phase_{tier}gpu_done')}; "
                f'echo ">>>PHASE:phase_{tier}gpu_done $(date +%s)"'
            )

        # Merge per-phase junitxml into single file
        lines.extend(
            [
                "python <<'_MERGE_PY_'",
                "import xml.etree.ElementTree as ET, glob",
                "s = ET.Element('testsuites')",
                "for f in sorted(glob.glob('bazel-testlogs/pytest/test_*gpu.xml')):",
                "    try:",
                "        r = ET.parse(f).getroot()",
                "        for c in (list(r) if r.tag == 'testsuites' else [r]): s.append(c)",
                "    except Exception: pass",
                "ET.ElementTree(s).write('bazel-testlogs/pytest/test.xml', xml_declaration=True, encoding='unicode')",
                "_MERGE_PY_",
            ]
        )

        lines.append("[ $any_ran -eq 0 ] && final_ec=5")
        lines.append(_heartbeat_shell("session_pytest_end"))
        lines.append('echo ">>>PHASE:pytest_end $(date +%s)"')
        lines.append(
            'echo ">>>GPU_LOCK_DIAG_START"; '
            "cat /tmp/rtp_llm_gpu_diag/* 2>/dev/null; "
            'echo ">>>GPU_LOCK_DIAG_END"'
        )
        lines.append("echo EXIT_CODE=$final_ec")
        lines.append(
            "echo '<<<JUNIT_XML>>>'; "
            "cat bazel-testlogs/pytest/test.xml 2>/dev/null; "
            "echo '<<<END_JUNIT_XML>>>'"
        )
        if outputs_postscript:
            lines.append(outputs_postscript)
        lines.append("exit $final_ec")

        run_cmd = "\n".join(lines)
        key = f"session_{ci_profile}" if ci_profile else "session"
        return self._supervised_command(f"{runtime.remote_setup_prefix}{run_cmd}", key)
