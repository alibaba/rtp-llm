"""RTP-specific helpers for remote pytest execution.

This module centralizes repository-specific behavior:
- default REAPI endpoint resolution
- remote worker setup command
- input file collection rules
- runtime env/platform properties

It does not register pytest hooks or talk to REAPI directly.
"""

from __future__ import annotations

import logging
import shlex
import socket
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, List, Optional, Tuple

log = logging.getLogger(__name__)

DEFAULT_GPU_TYPE = "gpu_cuda12"

_EXCLUDE_DIRS = frozenset({
    "flash_attn", "bazel-bin", "bazel-out", "bazel-rtp_llm",
    ".git", "__pycache__", "_solib_local", "build_logs",
})


@dataclass(frozen=True)
class GPURequest:
    """Normalized GPU resource request for one remote execution."""

    gpu_type: str
    gpu_count: int


@dataclass(frozen=True)
class RemoteRuntimeConfig:
    """Normalized runtime config consumed by plugin.py.

    Keep the shape stable so plugin code stays simple even if RTP rules change.

    ``platform_properties`` become REAPI ``Command.platform.properties`` (name/value
    pairs). Bazel maps the same keys from per-target or ``--remote_default_exec_properties``
    (e.g. ``gpu=gpu_cuda12`` in ``.cicd_bazelrc``) into that field for remote actions, so
    scheduling semantics match ``exec_properties`` / default exec properties.
    """

    ignore_args: List[str]
    env_vars: dict
    platform_properties: dict
    remote_setup_prefix: str


# ---------------------------------------------------------------------------
# pyproject.toml helpers
# ---------------------------------------------------------------------------

def _load_toml_module():
    if sys.version_info >= (3, 11):
        import tomllib
        return tomllib
    import tomli as tomllib  # type: ignore[import-not-found]
    return tomllib


@lru_cache(maxsize=4)
def _load_pyproject(rootdir: Path) -> dict:
    pyproject_path = rootdir / "pyproject.toml"
    if not pyproject_path.exists():
        return {}
    try:
        tomllib = _load_toml_module()
        with open(pyproject_path, "rb") as f:
            return tomllib.load(f)
    except Exception as exc:
        log.warning("Failed to load %s: %s", pyproject_path, exc)
        return {}


def _get_pytest_ini_options(rootdir: Path) -> dict:
    return _load_pyproject(rootdir).get("tool", {}).get("pytest", {}).get("ini_options", {})


def quote_args(args) -> str:
    """Shell-quote and join a list of arguments."""
    return " ".join(shlex.quote(arg) for arg in args)


def _resolve_host(host: str, port: int) -> str:
    """Resolve hostname to IP (handles vipserver DNS), return grpc:// URI."""
    try:
        results = socket.getaddrinfo(host, port, socket.AF_INET, socket.SOCK_STREAM)
        if results:
            ip = results[0][4][0]
            return f"grpc://{ip}:{port}"
    except (socket.gaierror, OSError):
        pass
    return f"grpc://{host}:{port}"


# ---------------------------------------------------------------------------
# Public helpers called by plugin.py
# ---------------------------------------------------------------------------

def resolve_default_reapi_endpoints(rootdir: Path, env: str = "daily") -> Tuple[Optional[str], Optional[str]]:
    """Resolve executor/cas endpoints from [tool.rtp-llm.remote].

    ``env`` selects which host keys to use: ``"daily"`` or ``"online"``.
    """
    cfg = _load_pyproject(rootdir).get("tool", {}).get("rtp-llm", {}).get("remote", {})
    if not cfg:
        return None, None

    cas_host = cfg.get(f"cas-{env}", "")
    cas_port = int(cfg.get("cas-port", 50051))
    executor_host = cfg.get(f"executor-{env}", "")
    executor_port = int(cfg.get("executor-port", 50052))

    if not cas_host or not executor_host:
        log.warning("No %s endpoints configured in [tool.rtp-llm.remote]", env)
        return None, None

    return _resolve_host(executor_host, executor_port), _resolve_host(cas_host, cas_port)


def get_pytest_ignore_args(rootdir: Path) -> List[str]:
    """Extract --ignore args from pyproject addopts."""
    addopts = _get_pytest_ini_options(rootdir).get("addopts", [])
    return [opt for opt in addopts if isinstance(opt, str) and opt.startswith("--ignore=")]


def build_remote_setup_command(rootdir: Path) -> str:
    """Return shell prefix executed before remote pytest command.

    Calls the shared pip_install.sh which handles:
    - Finding/installing uv
    - Finding base python
    - Creating venv
    - Bootstrap + single-step pip install
    - Error checking

    CWD on the remote worker is github-opensource (CAS rootdir).
    pip_install.sh is included in the CAS upload via _collect_base_files().
    """
    return (
        "export HOME=/home/admin; "
        "echo \">>>PHASE:pip_install_start $(date +%s)\"; "
        "export RTP_SKIP_BAZEL_BUILD=1; "
        "source internal_source/ci/pip_install.sh; "
        "echo \">>>PHASE:pip_install_done $(date +%s)\"; "
    )


# ---------------------------------------------------------------------------
# File collection
# ---------------------------------------------------------------------------

def _collect_base_files(rootdir: Path) -> List[str]:
    """Build config files present in rootdir."""
    files: List[str] = []
    for name in ("pyproject.toml", "open_source/pyproject.toml", "setup.py", "setup.cfg",
                  "conftest.py", "internal_source/ci/pip_install.sh"):
        if (rootdir / name).exists():
            files.append(name)
    return files


def _collect_repo_runtime_files(rootdir: Path) -> List[str]:
    """Python sources, .so libs, tokenizer data, config JSON, and testdata."""
    files: List[str] = []
    for pattern in ("rtp_llm/**/*.py", "internal_source/rtp_llm/**/*.py"):
        for p in rootdir.glob(pattern):
            if p.is_file() and not any(d in p.parts for d in _EXCLUDE_DIRS):
                files.append(str(p.relative_to(rootdir)))
    for pattern in ("rtp_llm/libs/*.so", "rtp_llm/libs/**/*.so"):
        files.extend(str(p.relative_to(rootdir)) for p in rootdir.glob(pattern) if p.is_file())
    for pattern in ("rtp_llm/tokenizer_data/*", "rtp_llm/config/*.json"):
        files.extend(str(p.relative_to(rootdir)) for p in rootdir.glob(pattern) if p.is_file())
    for pattern in ("rtp_llm/**/*.tiktoken", "rtp_llm/**/*.conf"):
        files.extend(str(p.relative_to(rootdir)) for p in rootdir.glob(pattern) if p.is_file())
    for pattern in ("rtp_llm/test/**/testdata/**/*",):
        files.extend(str(p.relative_to(rootdir)) for p in rootdir.glob(pattern) if p.is_file())
    return files


def _collect_smoke_files(rootdir: Path) -> List[str]:
    """Smoke test golden data."""
    files: List[str] = []
    for pattern in (
        "internal_source/rtp_llm/test/smoke/data/**/*.json",
        "internal_source/rtp_llm/test/smoke/data/**/*.pt",
        "internal_source/rtp_llm/test/smoke/data/**/*.jpg",
        "internal_source/rtp_llm/test/smoke/data/**/*.mp4",
    ):
        files.extend(str(p.relative_to(rootdir)) for p in rootdir.glob(pattern) if p.is_file())
    return files


def _collect_session_extra_files(rootdir: Path) -> List[str]:
    """Extra files only needed by session mode (internal testdata, tiktoken, conf)."""
    files: List[str] = []
    for pattern in (
        "internal_source/rtp_llm/**/testdata/**/*",
        "internal_source/rtp_llm/**/test/testdata/**/*",
    ):
        files.extend(str(p.relative_to(rootdir)) for p in rootdir.glob(pattern) if p.is_file())
    for pattern in ("rtp_llm/**/*.tiktoken", "rtp_llm/**/*.conf"):
        files.extend(str(p.relative_to(rootdir)) for p in rootdir.glob(pattern) if p.is_file())
    return files


def collect_remote_files(rootdir: Path, items: List[Any]) -> List[str]:
    """Collect per-test remote execution inputs (only called for gpu-marked items)."""
    files = _collect_base_files(rootdir)

    for item in items:
        rel = str(Path(str(item.fspath)).relative_to(rootdir))
        if rel not in files:
            files.append(rel)

    files.extend(_collect_repo_runtime_files(rootdir))

    if any(item.get_closest_marker("smoke") is not None for item in items):
        files.extend(_collect_smoke_files(rootdir))

    files = sorted(set(files))
    log.info("Collected %d files for CAS upload", len(files))
    return files


def collect_session_files(rootdir: Path) -> List[str]:
    """Collect remote-session execution inputs."""
    files = _collect_base_files(rootdir)
    files.extend(_collect_repo_runtime_files(rootdir))
    files.extend(_collect_session_extra_files(rootdir))

    files = sorted(set(files))
    log.info("Session mode: collected %d files for CAS upload", len(files))
    return files


# ---------------------------------------------------------------------------
# GPU request resolution
# ---------------------------------------------------------------------------

def should_dispatch_item_remotely(item: Any) -> bool:
    """Only tests marked with @pytest.mark.gpu go through per-test remote mode."""
    return item.get_closest_marker("gpu") is not None


def resolve_item_gpu_request(item: Any) -> GPURequest:
    """Resolve GPU request from @pytest.mark.gpu(type=..., count=...)."""
    marker = item.get_closest_marker("gpu")
    if marker is None:
        return GPURequest(DEFAULT_GPU_TYPE, 0)

    gpu_count = int(marker.kwargs.get("count", 1))
    gpu_type = str(marker.kwargs.get("type", DEFAULT_GPU_TYPE))
    return GPURequest(gpu_type, gpu_count)


def resolve_gpu_type_from_items(items: list, *, override: Optional[str] = None) -> str:
    """Determine a single GPU type from collected items' markers.

    If *override* is provided, use it directly.
    Otherwise scan all ``@pytest.mark.gpu(type=...)`` markers:

    - 0 distinct types → DEFAULT_GPU_TYPE
    - 1 distinct type  → that type
    - >1 distinct types → raise ValueError
    """
    if override:
        return override

    gpu_types: set = set()
    for item in items:
        marker = item.get_closest_marker("gpu")
        if marker:
            gpu_types.add(str(marker.kwargs.get("type", DEFAULT_GPU_TYPE)))

    gpu_types.discard(DEFAULT_GPU_TYPE)

    if len(gpu_types) == 0:
        return DEFAULT_GPU_TYPE
    if len(gpu_types) == 1:
        return gpu_types.pop()

    raise ValueError(
        f"Session mode requires a single GPU type but found {len(gpu_types)}: "
        f"{sorted(gpu_types)}. Use --remote-gpu-type to override."
    )


# ---------------------------------------------------------------------------
# Unified runtime config
# ---------------------------------------------------------------------------

def build_runtime_config(
    rootdir: Path,
    gpu_request: GPURequest,
    *,
    extra_env: Optional[dict] = None,
) -> RemoteRuntimeConfig:
    """Build env/platform/ignore/setup from a normalized GPU request."""
    env_vars: dict = {
        "GPU_COUNT": str(gpu_request.gpu_count),
    }
    if extra_env:
        env_vars.update(extra_env)

    platform_properties: dict = {
        "gpu": gpu_request.gpu_type,
        "gpu_count": str(gpu_request.gpu_count),
    }

    return RemoteRuntimeConfig(
        ignore_args=get_pytest_ignore_args(rootdir),
        env_vars=env_vars,
        platform_properties=platform_properties,
        remote_setup_prefix=build_remote_setup_command(rootdir),
    )
