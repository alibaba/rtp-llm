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
import os
import re
import shlex
import sys
import tarfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

DEFAULT_GPU_TYPE = "gpu_cuda12"

_EXCLUDE_DIRS = frozenset(
    {
        "flash_attn",
        "bazel-bin",
        "bazel-out",
        "bazel-rtp_llm",
        ".git",
        "__pycache__",
        "_solib_local",
        "build_logs",
    }
)
_REMOTE_INPUT_DIR = Path(".pytest_cache") / "remote_inputs"

# Non-Python files read directly by test_build_packaging_contract.py. Native
# pytest sessions do not get Bazel runfiles, so these source contracts must be
# explicit CAS inputs.
_SOURCE_CONTRACT_FILES = (
    "3rdparty/cub_compat.h",
    "3rdparty/flashinfer/flashinfer_cu13.BUILD",
    "arch_config/arch_select.bzl",
    "rtp_llm/cpp/cache/test/BUILD",
    "rtp_llm/cpp/cache/test/KVCacheManagerCPSlotMapperTest.cc",
    "rtp_llm/cpp/cache/test/SharedBlockCacheTest.cc",
    "rtp_llm/cpp/cuda_graph/cuda_graph_runner.cc",
    "rtp_llm/cpp/disaggregate/cache_store/test/BUILD",
    "rtp_llm/cpp/model_rpc/test/BUILD",
    "rtp_llm/cpp/models/logits_processor/test/BUILD",
    "rtp_llm/cpp/models/logits_processor/test/SpecLogitsVerifyRunnerPerfTest.cc",
    "rtp_llm/cpp/models/test/BUILD",
    "rtp_llm/cpp/models/test/PyWrappedModelCacheStoreIntegrationTest.cc",
    "rtp_llm/cpp/normal_engine/speculative/test/BUILD",
    "rtp_llm/cpp/normal_engine/speculative/test/MtpBatchStreamProcessorTest.cc",
    "rtp_llm/cpp/pybind/BUILD",
    "rtp_llm/cpp/telemetry/TraceAttributes.h",
    "rtp_llm/cpp/utils/test/BUILD",
    "rtp_llm/dash_sc/BUILD",
    "rtp_llm/dash_sc/test/BUILD",
    "rtp_llm/models_py/bindings/cuda/ops/tests/BUILD",
    "rtp_llm/models_py/bindings/cuda/ops/tests/CudaSamplerTest.cc",
    "rtp_llm/models_py/modules/dsv4/decode/test/BUILD",
    "rtp_llm/utils/test/BUILD",
)
_RUNTIME_LIBS_ARCHIVE = _REMOTE_INPUT_DIR / "rtp_llm_libs.tar"
_PPU_RUNTIME_DIR = _REMOTE_INPUT_DIR / "ppu_runtime"
_CORE_RUNTIME_LIBS = (
    "libth_transformer_config.so",
    "libth_grammar_tokenizer_info.so",
    "libth_transformer.so",
    "librtp_compute_ops.so",
)
_PPU_RUNTIME_LIB_PATTERNS = (
    ("CUDA runtime", "libcudart.so*"),
    ("cuBLAS", "libcublas.so*"),
    ("cuBLAS Lt", "libcublasLt.so*"),
    ("CUDA profiling runtime", "libcupti.so*"),
    ("PPU device runtime", "libhgml.so*"),
    ("PPU kernel runtime", "libuki.so*"),
)


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


@dataclass(frozen=True)
class ReapiEndpointConfig:
    """Default REAPI endpoints loaded from pyproject without DNS resolution."""

    executor: Optional[str]
    cas: Optional[str]
    fallback_executor: Optional[str] = None


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
    """Load pyproject.toml, merging internal_source overrides when present.

    Reads rootdir/pyproject.toml as the base, then deep-merges the internal
    overlay from either a packaged checkout (rootdir/internal_source) or the
    CI monorepo layout (rootdir/../internal_source). This keeps internal
    configs such as [tool.rtp-llm.pytest_ci.gpu_env] available when pytest is
    launched from github-opensource/.
    """
    tomllib = _load_toml_module()
    base: dict = {}
    base_path = rootdir / "pyproject.toml"
    if base_path.exists():
        try:
            with open(base_path, "rb") as f:
                base = tomllib.load(f)
            _normalize_pytest_ci_namespace(base)
        except Exception as exc:
            log.warning("Failed to load %s: %s", base_path, exc)

    # Prefer the current overlay name and retain the older pyproject.toml
    # fallback. CI keeps internal_source next to github-opensource, while the
    # uploaded remote input materializes it below the execution root.
    for internal_dir in (
        rootdir / "internal_source",
        rootdir.parent / "internal_source",
    ):
        for fname in ("pyproject_internal.toml", "pyproject.toml"):
            internal_path = internal_dir / fname
            if not internal_path.exists():
                continue
            try:
                with open(internal_path, "rb") as f:
                    internal = tomllib.load(f)
                _normalize_pytest_ci_namespace(internal)
                _deep_merge(base, internal)
            except Exception as exc:
                log.warning("Failed to load %s: %s", internal_path, exc)
            return base

    return base


def _deep_merge(base: dict, override: dict) -> None:
    """Recursively merge override into base (in-place). Override wins on conflict."""
    for key, val in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(val, dict):
            _deep_merge(base[key], val)
        else:
            base[key] = val


def _normalize_pytest_ci_namespace(config: dict) -> None:
    """Map the legacy internal pytest_ci namespace to the canonical one."""
    tool = config.get("tool", {})
    if not isinstance(tool, dict):
        return
    legacy_namespace = tool.get("rtp-llm", {})
    if not isinstance(legacy_namespace, dict):
        return
    legacy_pytest_ci = legacy_namespace.get("pytest_ci")
    if not isinstance(legacy_pytest_ci, dict):
        return

    canonical_namespace = tool.setdefault("rtp_llm", {})
    if not isinstance(canonical_namespace, dict):
        return
    canonical_pytest_ci = canonical_namespace.setdefault("pytest_ci", {})
    if not isinstance(canonical_pytest_ci, dict):
        return

    merged: dict = {}
    _deep_merge(merged, legacy_pytest_ci)
    _deep_merge(merged, canonical_pytest_ci)
    canonical_namespace["pytest_ci"] = merged


def _get_pytest_ini_options(rootdir: Path) -> dict:
    return (
        _load_pyproject(rootdir)
        .get("tool", {})
        .get("pytest", {})
        .get("ini_options", {})
    )


def quote_args(args) -> str:
    """Shell-quote and join a list of arguments."""
    return " ".join(shlex.quote(arg) for arg in args)


def _endpoint_uri(host: str, port: int) -> str:
    """Return a REAPI endpoint URI without resolving DNS.

    Pytest remote owns executor DNS resolution so it can refresh and
    fail over between resolved IPs. Keep outer config/script layers as stable
    hostnames.
    """
    return f"grpc://{host}:{port}"


def _configured_endpoint(
    cfg: dict,
    *,
    key_prefix: str,
    env: str,
    port_key: str,
    default_port: int,
) -> Optional[str]:
    host = cfg.get(f"{key_prefix}-{env}", "")
    if not host:
        return None
    port = int(cfg.get(port_key, default_port))
    return _endpoint_uri(host, port)


# ---------------------------------------------------------------------------
# Public helpers called by plugin.py
# ---------------------------------------------------------------------------


def resolve_default_reapi_endpoint_config(
    rootdir: Path, env: str = "daily"
) -> ReapiEndpointConfig:
    """Load executor/cas endpoints from [tool.rtp-llm.remote].

    ``env`` selects which host keys to use: ``"daily"`` or ``"online"``.
    Hostnames are intentionally not resolved here. Executor DNS
    resolution happens inside the pytest remote executor pool.
    """
    cfg = _load_pyproject(rootdir).get("tool", {}).get("rtp-llm", {}).get("remote", {})
    if not cfg:
        return ReapiEndpointConfig(None, None)

    cas_ep = _configured_endpoint(
        cfg,
        key_prefix="cas",
        env=env,
        port_key="cas-port",
        default_port=50051,
    )
    executor_ep = _configured_endpoint(
        cfg,
        key_prefix="executor",
        env=env,
        port_key="executor-port",
        default_port=50052,
    )

    if not cas_ep or not executor_ep:
        log.warning("No %s endpoints configured in [tool.rtp-llm.remote]", env)
        return ReapiEndpointConfig(None, None)

    fallback_executor_ep = None
    if env == "online":
        fallback_executor_ep = _configured_endpoint(
            cfg,
            key_prefix="executor",
            env="daily",
            port_key="executor-port",
            default_port=50052,
        )
        if fallback_executor_ep == executor_ep:
            fallback_executor_ep = None

    return ReapiEndpointConfig(executor_ep, cas_ep, fallback_executor_ep)


def resolve_default_reapi_endpoints(
    rootdir: Path, env: str = "daily"
) -> Tuple[Optional[str], Optional[str]]:
    """Compatibility wrapper returning executor/cas endpoints only."""
    endpoints = resolve_default_reapi_endpoint_config(rootdir, env=env)
    return endpoints.executor, endpoints.cas


def get_pytest_ignore_args(rootdir: Path) -> List[str]:
    """Extract --ignore args from pyproject addopts."""
    addopts = _get_pytest_ini_options(rootdir).get("addopts", [])
    return [
        opt for opt in addopts if isinstance(opt, str) and opt.startswith("--ignore=")
    ]


def _shell_export_prefix(env: Dict[str, str]) -> str:
    parts = []
    for key, value in sorted(env.items()):
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key):
            log.warning("Ignoring invalid remote env var name: %s", key)
            continue
        parts.append(f"export {key}={shlex.quote(str(value))}; ")
    return "".join(parts)


def _shell_env_assignments(env: Dict[str, str]) -> str:
    parts = []
    for key, value in sorted(env.items()):
        if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key):
            log.warning("Ignoring invalid remote env var name: %s", key)
            continue
        parts.append(f"{key}={shlex.quote(str(value))}")
    return " ".join(parts)


def build_remote_setup_command(rootdir: Path, *, setup_env: Optional[dict] = None) -> str:
    """Return shell prefix executed before remote pytest command.

    Calls prepare_venv.py which handles:
    - Finding base python + detecting platform
    - Computing isolated venv path (per CAS input root hash)
    - Locking (fcntl.flock) to prevent concurrent install races
    - Caching (.installed_ok) to skip install when venv already ready
    - Creating venv + bootstrap + uv pip install --compile-bytecode

    CWD on the remote worker is github-opensource (CAS rootdir).
    prepare_venv.py is included in the CAS upload via _collect_base_files().
    """
    gpu_diag = (
        'echo ">>>GPU_DIAG_START"; '
        "nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu "
        "--format=csv,noheader 2>/dev/null || true; "
        "nvidia-smi --query-compute-apps=pid,gpu_uuid,used_gpu_memory,name "
        "--format=csv,noheader 2>/dev/null || true; "
        "for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' '); do "
        "  echo \"PID=$p CMD=$(cat /proc/$p/cmdline 2>/dev/null | tr '\\0' ' ' || echo N/A) "
        'CGROUP=$(cat /proc/$p/cgroup 2>/dev/null | head -1 || echo N/A)"; '
        "done; "
        'echo ">>>GPU_DIAG_END"; '
    )
    # PV_RC must be captured *between* assignments — `OUT=$(...); rc=$?` puts the
    # exit code from prepare_venv.py into rc. Without this guard, `eval "$OUT"`
    # silently no-ops on failure and pytest fails downstream with a misleading
    # ImportError. Pattern matches internal_source/ci/basic_test.sh:84-104.
    # PWD on worker is the CAS rootdir (mirrors github-opensource/), so the
    # uploaded `rtp_llm/libs/` ends up at $PWD/rtp_llm/libs. We MUST add it to
    # LD_LIBRARY_PATH because libth_transformer.so DT_NEEDED carries
    # `kv_cache_manager_client.so` and other co-resident libs without rpath
    # (they're sibling .so files dlopen-resolved via the loader's search list).
    # Without this, `from libth_transformer_config import …` raises
    # `ImportError: kv_cache_manager_client.so: cannot open shared object file`,
    # and downstream `torch.ops.rtp_llm.init_engine` is unregistered.
    remote_setup = (
        _load_pyproject(rootdir)
        .get("tool", {})
        .get("rtp_llm", {})
        .get("pytest_ci", {})
        .get("remote_setup", {})
    )
    extra_prefix = str(remote_setup.get("prefix", ""))
    setup_env_prefix = _shell_export_prefix(
        {str(key): str(value) for key, value in (setup_env or {}).items()}
    )
    setup_env_assignments = _shell_env_assignments(
        {
            str(key): str(value)
            for key, value in (setup_env or {}).items()
            if key != "LD_LIBRARY_PATH"
        }
    )
    prepare_env_prefix = f"{setup_env_assignments} " if setup_env_assignments else ""
    setup_env_diag = (
        'echo "[remote_setup] RTP_BAZEL_CONFIG=${RTP_BAZEL_CONFIG:-unset}"; '
        if "RTP_BAZEL_CONFIG" in (setup_env or {})
        else ""
    )
    return (
        "export HOME=/home/admin; "
        "export RTP_SKIP_BAZEL_BUILD=1; "
        # /opt/conda310/lib is critical on ROCm workers: aiter's JIT-built
        # `.so` files reference GCC 11+ libstdc++ symbols (e.g. ref-qualified
        # `std::__cxx11::ostringstream::str() const&` =
        # _ZNKRSt7__cxx1119basic_ostringstream...3strEv). The system
        # /usr/lib64/libstdc++.so.6 on the rocm dev image is GCC 8.5 (no
        # such symbol); /opt/conda310/lib/libstdc++.so.6.0.29 (GCC 11.2)
        # IS the right copy. The bazel-driven test path always set this
        # via .bazelrc `test:rocm --test_env LD_LIBRARY_PATH=.../opt/conda310/lib/...`,
        # but the pytest --remote prologue lost that. Without it, venv
        # `bin/python` (whose `$ORIGIN/../lib` is empty) falls back to
        # /usr/lib64 → ImportError on aiter dlopen. Append gcc-toolset-12
        # + /opt/rocm + /opt/amdgpu paths for symmetry with .bazelrc; ld
        # silently skips non-existent dirs so safe on other workers.
        "export LD_LIBRARY_PATH="
        '"$PWD/.pytest_cache/remote_inputs/ppu_runtime:$PWD/rtp_llm/libs:'
        "/opt/conda310/lib:/opt/rh/gcc-toolset-12/root/usr/lib64:"
        "/opt/rocm/lib:/opt/amdgpu/lib64:"
        '/usr/local/nvidia/lib64:/usr/lib64:/usr/local/cuda/lib64"; '
        # Diagnostic — appears in remote_stdout.log so we can verify the
        # prologue ran. Drops if smoke-amd starts passing consistently.
        'echo "[remote_setup] LD_LIBRARY_PATH=$LD_LIBRARY_PATH"; '
        + setup_env_prefix
        + extra_prefix
        + setup_env_diag
        +
        # Disk eviction: REAPI workers reuse /home/admin/ across pytest
        # sessions; each session's prepare_venv.py creates a NEW venv
        # at /home/admin/venvs/rtp-llm-{platform}-{hash}. Evict only old venvs
        # whose prepare_venv lock can be acquired, so concurrent jobs keep
        # their active environments.
        # Belt-and-suspenders: also evict /tmp/uv-rtp-llm-* caches.
        "evict_locked_venvs() { "
        "  find /home/admin/venvs -maxdepth 1 -type d -name 'rtp-llm-*' \"$@\" -print0 2>/dev/null | "
        "  while IFS= read -r -d '' d; do "
        '    lock="${d}.lock"; '
        '    if flock -n "$lock" rm -rf "$d"; then '
        '      echo "[remote_setup] evicted inactive venv: $d"; '
        "    else "
        '      echo "[remote_setup] skip active venv: $d"; '
        "    fi; "
        "  done; "
        "}; "
        "evict_locked_venvs -mtime +7; "
        "find /tmp -maxdepth 1 -type d -name 'uv-rtp-llm-*' "
        "  -mtime +3 -exec rm -rf {} + 2>/dev/null; "
        "free_kb=$(df -Pk /home/admin 2>/dev/null | awk 'NR==2 {print $4+0}'); "
        'if [ "${free_kb:-0}" -lt 52428800 ]; then '
        '  echo "[remote_setup] low disk (${free_kb} KiB free); evicting 6h-old venv/cache dirs"; '
        "  evict_locked_venvs -mmin +360; "
        "  find /tmp -maxdepth 1 -type d -name 'uv-rtp-llm-*' "
        "    -mmin +360 -exec rm -rf {} + 2>/dev/null; "
        "fi; "
        "free_kb=$(df -Pk /home/admin 2>/dev/null | awk 'NR==2 {print $4+0}'); "
        'if [ "${free_kb:-0}" -lt 20971520 ]; then '
        '  echo "[remote_setup] critically low disk (${free_kb} KiB free); evicting 1h-old venv/cache dirs"; '
        "  evict_locked_venvs -mmin +60; "
        "  find /tmp -maxdepth 1 -type d -name 'uv-rtp-llm-*' "
        "    -mmin +60 -exec rm -rf {} + 2>/dev/null; "
        "fi; "
        'echo "[remote_setup] disk after eviction: $(df -h /home/admin 2>/dev/null | tail -1)"; '
        + gpu_diag
        + 'echo ">>>PHASE:pip_install_start $(date +%s)"; '
        "mkdir -p logs; "
        "if [ -f internal_source/ci/prepare_rocm_deps.sh ]; then "
        "  source internal_source/ci/prepare_rocm_deps.sh >&2 || exit $?; "
        "fi; "
        "if [ -f internal_source/ci/prepare_venv.py ]; then "
        # Dependency installers invoke system Git, whose Kerberos libraries
        # cannot use Conda's OpenSSL (EVP_KDF_ctrl / OPENSSL_1_1_1b). Keep the
        # test runtime library path in the parent shell, not the installer.
        f"  env -u LD_LIBRARY_PATH {prepare_env_prefix}"
        "/opt/conda310/bin/python internal_source/ci/prepare_venv.py "
        ">logs/prepare_venv.out 2>logs/prepare_venv.err & PV_PID=$!; "
        "  (while kill -0 \"$PV_PID\" 2>/dev/null; do "
        '    if [ -n "${RTP_REMOTE_HEARTBEAT_FILE:-}" ]; then '
        '      printf \'%s %s\\n\' "$(date +%s)" pip_install_active '
        '        >> "$RTP_REMOTE_HEARTBEAT_FILE" 2>/dev/null || true; '
        "    fi; sleep 5; done) & PV_HB_PID=$!; "
        '  wait "$PV_PID"; PV_RC=$?; wait "$PV_HB_PID" 2>/dev/null || true; '
        '  OUT=$(cat logs/prepare_venv.out); '
        '  if [ "$PV_RC" -ne 0 ]; then '
        "    cat logs/prepare_venv.err >&2; "
        '    echo ">>>PHASE:pip_install_failed $(date +%s) rc=$PV_RC"; '
        '    exit "$PV_RC"; '
        "  fi; "
        '  eval "$OUT"; cat logs/prepare_venv.err >&2; '
        "else "
        '  echo ">>>PHASE:prepare_venv_skipped $(date +%s)"; '
        "fi; "
        'echo ">>>PHASE:pip_install_done $(date +%s)"; '
        'if [ -n "${_rocm_native_library_path:-}" ]; then '
        '  export LD_LIBRARY_PATH="${_rocm_native_library_path}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"; '
        "fi; "
        f"if [ -f {shlex.quote(str(_RUNTIME_LIBS_ARCHIVE))} ]; then "
        f"  tar -xf {shlex.quote(str(_RUNTIME_LIBS_ARCHIVE))}; "
        '  echo "[remote_setup] restored rtp_llm/libs from runtime libs archive"; '
        "fi; "
        "ls -lh rtp_llm/libs/libth_transformer_config.so "
        "rtp_llm/libs/libth_transformer.so "
        "rtp_llm/libs/librtp_compute_ops.so 2>/dev/null || true; "
        "if [ -d /opt/rocm ] && [ -f _build/rocm_jit.py ]; then "
        '  echo ">>>PHASE:rocm_jit_start $(date +%s)"; '
        "  python -m _build.rocm_jit & RJ_PID=$!; "
        '  (while kill -0 "$RJ_PID" 2>/dev/null; do '
        '    if [ -n "${RTP_REMOTE_HEARTBEAT_FILE:-}" ]; then '
        '      printf \'%s %s\\n\' "$(date +%s)" rocm_jit_active '
        '        >> "$RTP_REMOTE_HEARTBEAT_FILE" 2>/dev/null || true; '
        "    fi; sleep 5; done) & RJ_HB_PID=$!; "
        '  wait "$RJ_PID"; RJ_RC=$?; wait "$RJ_HB_PID" 2>/dev/null || true; '
        '  if [ "$RJ_RC" -ne 0 ]; then exit "$RJ_RC"; fi; '
        '  echo ">>>PHASE:rocm_jit_done $(date +%s)"; '
        "fi; "
    )


# ---------------------------------------------------------------------------
# File collection
# ---------------------------------------------------------------------------


def _collect_base_files(rootdir: Path) -> List[str]:
    """Build config files present in rootdir."""
    files: List[str] = []
    for name in (
        "pyproject.toml",
        "open_source/pyproject.toml",
        "internal_source/pyproject_internal.toml",
        "internal_source/pyproject.toml",
        "setup.py",
        "setup.cfg",
        "conftest.py",
        "internal_source/ci/prepare_venv.py",
        "internal_source/ci/prepare_rocm_deps.sh",
        "internal_source/ci/ci_pip_install.sh",
    ):
        if (rootdir / name).exists():
            files.append(name)
    extra_overlay_files = (
        _load_pyproject(rootdir)
        .get("tool", {})
        .get("rtp-llm", {})
        .get("platform-overlay", {})
        .get("extra_overlay_files", [])
    )
    for name in extra_overlay_files:
        rel = str(name)
        if (rootdir / rel).exists():
            files.append(rel)
    # _build/*.py is imported by setup.py and prepare_venv.py (detect_build_config).
    # Without these the remote venv install fails with: "Cannot import _build.platform".
    for p in (rootdir / "_build").glob("*.py"):
        if p.is_file():
            files.append(str(p.relative_to(rootdir)))
    # _build/oss_optional_extras.toml carries platform-specific torch/flash-attn
    # URL pins (cuda12_9, rocm, cuda12_arm). setup.py's
    # get_merged_optional_dependencies() reads it to inject torch URL into
    # install_requires. Without this file on the REAPI worker, setup.py falls
    # back to pyproject.toml (which only declares dev/docs) → no torch pin →
    # uv installs torch from PyPI (latest) → ABI mismatch with bazel-built .so
    # ("undefined symbol: ..._compute_strides_like_channels_last_3d...").
    extras_toml = rootdir / "_build" / "oss_optional_extras.toml"
    if extras_toml.is_file():
        files.append(str(extras_toml.relative_to(rootdir)))
    # MORI is built with the CI controller's native toolchain. Ship its wheel
    # so GPU workers do not recursively clone GitHub repositories during setup.
    wheelhouse = rootdir / ".pytest_cache/remote_inputs/mori_wheel"
    for pattern in ("manifest.json", "*.whl"):
        files.extend(
            str(path.relative_to(rootdir))
            for path in wheelhouse.glob(pattern)
            if path.is_file()
        )
    return files


def _collect_repo_runtime_files(
    rootdir: Path, *, include_libs: bool = True
) -> List[str]:
    """Runtime sources, .so libs, tokenizer data, config JSON, and testdata."""
    files: List[str] = []
    for pattern in ("rtp_llm/**/*.py", "internal_source/rtp_llm/**/*.py"):
        for p in rootdir.glob(pattern):
            if p.is_file() and not any(d in p.parts for d in _EXCLUDE_DIRS):
                files.append(str(p.relative_to(rootdir)))
    # Some Python compatibility tests inspect type stubs and narrowly scoped
    # C++ source contracts at runtime. Bazel supplied these as target data;
    # native pytest sessions must upload them explicitly.
    for pattern in (
        "rtp_llm/**/*.pyi",
        "internal_source/rtp_llm/**/*.pyi",
        "rtp_llm/model_loader/tipc/csrc/**/*",
    ) + _SOURCE_CONTRACT_FILES:
        files.extend(
            str(p.relative_to(rootdir)) for p in rootdir.glob(pattern) if p.is_file()
        )
    if include_libs:
        for pattern in (
            "rtp_llm/libs/*.so",
            "rtp_llm/libs/*.so.*",
            "rtp_llm/libs/**/*.so",
            "rtp_llm/libs/test/*",
        ):
            files.extend(
                str(p.relative_to(rootdir))
                for p in rootdir.glob(pattern)
                if p.is_file()
            )
        # kv_cache_manager_bin (staged by setup.py from a top-level Bazel output).
        # Required by smoke tests using remote_kv_cache (cuda_remote_cache,
        # eagle_remote_cache_tp2, next_long_reuse_remote, …) under pytest+REAPI
        # dispatch.
        files.extend(
            str(p.relative_to(rootdir))
            for p in rootdir.glob("rtp_llm/libs/kv_cache_manager_server/bin/*")
            if p.is_file()
        )
    for pattern in ("rtp_llm/tokenizer_data/*", "rtp_llm/config/*.json"):
        files.extend(
            str(p.relative_to(rootdir)) for p in rootdir.glob(pattern) if p.is_file()
        )
    for pattern in ("rtp_llm/**/*.tiktoken", "rtp_llm/**/*.conf"):
        files.extend(
            str(p.relative_to(rootdir)) for p in rootdir.glob(pattern) if p.is_file()
        )
    for pattern in (
        "rtp_llm/test/perf_test/*.json",
        "rtp_llm/**/testdata/**/*",
        "rtp_llm/**/test/testdata/**/*",
        "rtp_llm/**/test/data/**/*",
        "rtp_llm/models_py/**/data/**/*",
        "rtp_llm/models_py/kernels/cuda/fp8_kernel/cutlass_groupgemm/*.json",
        "internal_source/rtp_llm/models_py/kernels/cuda/fp8_kernel/cutlass_groupgemm/*.json",
    ):
        files.extend(
            str(p.relative_to(rootdir)) for p in rootdir.glob(pattern) if p.is_file()
        )
    return files


def _runtime_lib_files(rootdir: Path) -> List[Path]:
    libs_dir = rootdir / "rtp_llm" / "libs"
    if not libs_dir.exists():
        return []
    return sorted(p for p in libs_dir.rglob("*") if p.is_file())


def _ppu_runtime_search_dirs() -> List[Path]:
    sdk_roots: List[Path] = []
    for env_name in ("PPU_SDK", "PPU_HOME", "SDK_ROOT"):
        if value := os.environ.get(env_name):
            sdk_roots.append(Path(value))
    sdk_roots.append(Path("/usr/local/PPU_SDK"))

    candidates: List[Path] = []
    for sdk_root in sdk_roots:
        candidates.extend(
            (
                sdk_root / "CUDA_SDK" / "lib64",
                sdk_root / "CUDA_SDK" / "extras" / "CUPTI" / "lib64",
                sdk_root / "CUDA_SDK" / "targets" / "x86_64-linux" / "lib",
                sdk_root / "lib",
                sdk_root / "sailSHMEM" / "lib",
            )
        )
    candidates.extend(
        (
            Path("/usr/local/cuda/lib64"),
            Path("/usr/local/cuda/extras/CUPTI/lib64"),
            Path("/usr/local/cuda/targets/x86_64-linux/lib"),
            Path("/usr/local/nvidia/lib64"),
        )
    )
    return list(dict.fromkeys(candidates))


def _prepare_ppu_runtime_libs(rootdir: Path, items: List[Any]) -> List[str]:
    """Expose controller-side PPU runtimes as explicit REAPI inputs.

    Bazel tests obtain these libraries through target data/runfiles. Native
    pytest dispatch only uploads repository files, while PPU workers are not
    guaranteed to contain the SDK. Stage symlinks in the ignored pytest cache;
    the Merkle builder follows them and materializes regular files remotely.
    """
    if not any(
        resolve_item_gpu_request(item).gpu_type.upper().startswith("PPU")
        for item in items
    ):
        return []

    search_dirs = _ppu_runtime_search_dirs()
    selected: List[Path] = []
    missing: List[str] = []
    for label, pattern in _PPU_RUNTIME_LIB_PATTERNS:
        matches: List[Path] = []
        for directory in search_dirs:
            if directory.is_dir():
                matches = sorted(
                    path
                    for path in directory.glob(pattern)
                    if path.is_file() or path.is_symlink()
                )
            if matches:
                break
        if matches:
            selected.extend(matches)
        else:
            missing.append(f"{label} ({pattern})")

    if missing:
        searched = ", ".join(str(path) for path in search_dirs)
        raise RuntimeError(
            "PPU remote tests require controller runtime libraries missing from "
            f"the REAPI workers: {', '.join(missing)}. Searched: {searched}"
        )

    stage_dir = rootdir / _PPU_RUNTIME_DIR
    stage_dir.mkdir(parents=True, exist_ok=True)
    staged: List[str] = []
    for source in selected:
        destination = stage_dir / source.name
        resolved_source = source.resolve()
        if destination.is_symlink() or destination.exists():
            if destination.is_symlink() and destination.resolve() == resolved_source:
                staged.append(str(destination.relative_to(rootdir)))
                continue
            destination.unlink()
        destination.symlink_to(resolved_source)
        staged.append(str(destination.relative_to(rootdir)))

    log.info(
        "PPU remote tests: staged %d runtime library names from controller SDK",
        len(staged),
    )
    return staged


def _prepare_runtime_libs_archive(rootdir: Path) -> str:
    """Pack runtime libs for remote-session runs.

    Per-test remote execution uploads ``rtp_llm/libs`` as ordinary input files.
    Session mode runs an editable install on the worker before pytest starts;
    in practice that path has lost the staged ``.so`` directory on some workers,
    producing import-time failures for ``libth_transformer_config.so``.  Ship a
    single archive and extract it after ``prepare_venv.py`` so the worker source
    tree has the same built runtime libs that the controller just validated.
    """
    libs_dir = rootdir / "rtp_llm" / "libs"
    missing = [name for name in _CORE_RUNTIME_LIBS if not (libs_dir / name).is_file()]
    if missing:
        raise RuntimeError(
            "remote-session requires staged RTP runtime libs, but missing: "
            f"{', '.join(missing)}. Run `python setup.py build_ext --inplace` first."
        )

    lib_files = _runtime_lib_files(rootdir)
    if not lib_files:
        raise RuntimeError(f"remote-session found no runtime libs under {libs_dir}")

    archive = rootdir / _RUNTIME_LIBS_ARCHIVE
    latest_input_mtime = max(p.stat().st_mtime for p in lib_files)
    if archive.is_file() and archive.stat().st_mtime >= latest_input_mtime:
        return str(archive.relative_to(rootdir))

    archive.parent.mkdir(parents=True, exist_ok=True)
    tmp_archive = archive.with_suffix(".tar.tmp")
    if tmp_archive.exists():
        tmp_archive.unlink()

    with tarfile.open(tmp_archive, "w", dereference=True) as tar:
        for path in lib_files:
            rel = path.relative_to(rootdir)
            info = tar.gettarinfo(str(path), arcname=str(rel))
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            with open(path, "rb") as f:
                tar.addfile(info, f)

    tmp_archive.replace(archive)
    log.info(
        "Session mode: packed %d runtime lib files into %s (%.1f MB)",
        len(lib_files),
        archive.relative_to(rootdir),
        archive.stat().st_size / 1024 / 1024,
    )
    return str(archive.relative_to(rootdir))


def _collect_smoke_files(rootdir: Path) -> List[str]:
    """Smoke test golden data — OSS tree + internal tree, both shipped."""
    files: List[str] = []
    for pattern in (
        "rtp_llm/test/smoke/data/**/*.json",
        "rtp_llm/test/smoke/data/**/*.pt",
        "rtp_llm/test/smoke/data/**/*.bin",
        "rtp_llm/test/smoke/data/**/*.model",
        "rtp_llm/test/smoke/data/**/*.safetensors",
        "rtp_llm/test/smoke/data/**/*.jpg",
        "rtp_llm/test/smoke/data/**/*.jpeg",
        "rtp_llm/test/smoke/data/**/*.mp4",
        "internal_source/rtp_llm/test/smoke/data/**/*.json",
        "internal_source/rtp_llm/test/smoke/data/**/*.pt",
        "internal_source/rtp_llm/test/smoke/data/**/*.bin",
        "internal_source/rtp_llm/test/smoke/data/**/*.model",
        "internal_source/rtp_llm/test/smoke/data/**/*.safetensors",
        "internal_source/rtp_llm/test/smoke/data/**/*.jpg",
        "internal_source/rtp_llm/test/smoke/data/**/*.jpeg",
        "internal_source/rtp_llm/test/smoke/data/**/*.mp4",
        "internal_source/rtp_llm/test/smoke/flexlb_runtime/**/*",
    ):
        files.extend(
            str(p.relative_to(rootdir)) for p in rootdir.glob(pattern) if p.is_file()
        )
    return files


def _collect_session_extra_files(rootdir: Path) -> List[str]:
    """Extra files only needed by session mode (internal testdata, tiktoken, conf)."""
    files: List[str] = []
    for pattern in (
        "internal_source/rtp_llm/**/testdata/**/*",
        "internal_source/rtp_llm/**/test/testdata/**/*",
    ):
        files.extend(
            str(p.relative_to(rootdir)) for p in rootdir.glob(pattern) if p.is_file()
        )
    for pattern in ("rtp_llm/**/*.tiktoken", "rtp_llm/**/*.conf"):
        files.extend(
            str(p.relative_to(rootdir)) for p in rootdir.glob(pattern) if p.is_file()
        )
    files.extend(_collect_perf_files(rootdir))
    return files


def _collect_perf_files(rootdir: Path) -> List[str]:
    """Perf test data: distribution CSVs, batch configs, and baselines."""
    files: List[str] = []
    # perf_test data: distribution CSVs, batch_seq_len configs, baselines.
    # perf tests load these via Path(__file__).parent.parent (one level above suites/).
    for pattern in (
        "internal_source/rtp_llm/test/perf_test/test_data/**/*",
        "internal_source/rtp_llm/test/perf_test/baselines/**/*",
    ):
        files.extend(
            str(p.relative_to(rootdir)) for p in rootdir.glob(pattern) if p.is_file()
        )
    return files


def _safe_rel_to_rootdir(item_path: Path, rootdir: Path) -> str:
    """Turn an absolute test-file path into a rootdir-relative string.

    Internal smoke tests can live as a sibling of rootdir (`github-opensource/`),
    so plain `relative_to(rootdir)` raises ValueError. Map those paths back
    through the checked-out ``internal_source`` symlink so CAS entries and
    worker pytest paths stay inside the uploaded root.
    """
    try:
        return str(item_path.relative_to(rootdir))
    except ValueError:
        rel = os.path.relpath(item_path, rootdir)
        parts = Path(rel).parts
        if len(parts) >= 2 and parts[0] == ".." and parts[1] == "internal_source":
            return str(Path(*parts[1:]))
        return rel


def collect_remote_files(rootdir: Path, items: List[Any]) -> List[str]:
    """Collect per-test remote execution inputs (only called for gpu-marked items)."""
    files = _collect_base_files(rootdir)
    item_rels: List[str] = []

    for item in items:
        rel = _safe_rel_to_rootdir(Path(str(item.fspath)).resolve(), rootdir)
        item_rels.append(rel)
        if rel not in files:
            files.append(rel)

    files.extend(_collect_repo_runtime_files(rootdir))
    files.extend(_prepare_ppu_runtime_libs(rootdir, items))

    has_smoke = any(item.get_closest_marker("smoke") is not None for item in items)
    if has_smoke:
        files.extend(_collect_smoke_files(rootdir))
    has_perf = any(item.get_closest_marker("perf") is not None for item in items)
    if has_perf or any("/perf_test/" in rel for rel in item_rels):
        files.extend(_collect_perf_files(rootdir))

    files = sorted(set(files))
    if has_smoke:
        _check_no_lfs_pointers(rootdir, files)
    log.info("Collected %d files for CAS upload", len(files))
    return files


_OSS_LFS_POINTER_HEADER = b"oss-lfs v1"
_OSS_LFS_BINARY_SUFFIXES = (".pt", ".bin", ".model", ".safetensors")


def _check_no_lfs_pointers(rootdir: Path, rel_files: List[str]) -> None:
    """Fail-fast if smoke binary fixtures are still oss-lfs pointer text.

    Worktrees created via ``git worktree add`` don't inherit
    ``core.hooksPath=.githooks`` so the post-checkout auto-pull never runs,
    leaving 184-byte pointer files in place of real .pt/.bin tensors.
    Uploading those to CAS makes worker `torch.load` fail with the cryptic
    `_pickle.UnpicklingError: could not find MARK`.
    """
    pointers: List[str] = []
    for rel in rel_files:
        if not rel.endswith(_OSS_LFS_BINARY_SUFFIXES):
            continue
        path = rootdir / rel
        try:
            with open(path, "rb") as f:
                head = f.read(len(_OSS_LFS_POINTER_HEADER))
        except OSError:
            continue
        if head == _OSS_LFS_POINTER_HEADER:
            pointers.append(rel)
    if pointers:
        sample = "\n  ".join(pointers[:5])
        more = f"\n  ... and {len(pointers) - 5} more" if len(pointers) > 5 else ""
        raise RuntimeError(
            f"\n{len(pointers)} oss-lfs pointer file(s) detected — would upload garbage to worker:\n  "
            f"{sample}{more}\n"
            f"Fix: cd {rootdir} && git config core.hooksPath .githooks && "
            f"bash scripts/oss-lfs/oss-lfs-pull.sh"
        )


def collect_session_files(rootdir: Path) -> List[str]:
    """Collect remote-session execution inputs."""
    files = _collect_base_files(rootdir)
    files.extend(_collect_repo_runtime_files(rootdir, include_libs=False))
    files.extend(_collect_smoke_files(rootdir))
    files.extend(_collect_session_extra_files(rootdir))
    files.append(_prepare_runtime_libs_archive(rootdir))

    files = sorted(set(files))
    _check_no_lfs_pointers(rootdir, files)
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


# Marker-side GPU type tokens. These match the *underscore* form that pytest
# requires for marker names (markers must be valid Python identifiers — no
# hyphens). Profile markexpr values in pyproject.toml use this form.
#
# Keep this ordered: session mode may infer from broad expressions before
# worker-side collection can inspect concrete pytest items. Prefer concrete
# worker-pool tokens over deprecated aliases.
_KNOWN_GPU_TYPES = (
    "SM100_ARM",
    "SM100",
    "MI308X_ROCM7",
    "MI308X",
    "A10",
    "GeForce_RTX_3090",
    "GeForce_RTX_4090",
    "Tesla_V100S_PCIE_32GB",
    "L20",
    "H20",
)

# Pytest markers must be valid Python identifiers (no hyphens), but some REAPI
# worker pools register GPU types with hyphenated names.
# A worker-pool mismatch (e.g. submitting gpu=MI308X vs the pool's
# gpu=MI308X-ROCM7) silently leaves the action queued forever with no worker
# matching (verified 2026-05-01: amd session sat in QUEUED 12+min until gRPC
# deadline, retried, sat another 12min, gave up — all because the marker
# `MI308X` got submitted as-is instead of being mapped to `MI308X-ROCM7`).
# This mapping MUST stay in sync with .cicd_bazelrc. The `MI308X` →
# `MI308X-ROCM7` entry below covers the existing `pytest.mark.gpu(type="MI308X")`
# call sites in rtp_llm/models_py/{kernels,modules}/.../rocm/test/ — they
# all target the only AMD pool we currently have.
_GPU_TYPE_TO_REAPI: Dict[str, str] = {
    "MI308X": "MI308X-ROCM7",
    "MI308X_ROCM7": "MI308X-ROCM7",
}


def _to_reapi_gpu_type(rootdir: Path, gpu_type: str) -> str:
    """Translate a marker-derived GPU type to the REAPI platform property value."""
    aliases = dict(_GPU_TYPE_TO_REAPI)
    aliases.update(
        _load_pyproject(rootdir)
        .get("tool", {})
        .get("rtp_llm", {})
        .get("pytest_ci", {})
        .get("gpu_aliases", {})
    )
    return str(aliases.get(gpu_type, gpu_type))


def infer_gpu_type_from_markexpr(markexpr: str) -> Optional[str]:
    """Best-effort extraction of GPU type from a pytest ``-m`` expression.

    Scans for known GPU type tokens that appear *positively* (i.e. not preceded
    by ``not``).  Returns the first match, or ``None`` if nothing is found.
    """
    if not markexpr:
        return None
    import re

    # Tokenize: split into words, identify negated tokens
    negated: set = set()
    tokens = re.findall(r"\b\w+\b", markexpr)
    for i, tok in enumerate(tokens):
        if tok == "not" and i + 1 < len(tokens):
            negated.add(tokens[i + 1])
    for gpu_type in _KNOWN_GPU_TYPES:
        if gpu_type in tokens and gpu_type not in negated:
            return gpu_type
    return None


def resolve_ci_profile_gpu_type(
    rootdir: Path, ci_profile: Optional[str]
) -> Optional[str]:
    """Return the configured gpu_type for a pytest CI profile, if present."""
    if not ci_profile:
        return None
    profiles = (
        _load_pyproject(rootdir)
        .get("tool", {})
        .get("rtp_llm", {})
        .get("pytest_ci", {})
        .get("profiles", {})
    )
    profile = profiles.get(ci_profile, {})
    gpu_type = profile.get("gpu_type") if isinstance(profile, dict) else None
    return str(gpu_type) if gpu_type else None


def resolve_ci_profile_remote_env(
    rootdir: Path, ci_profile: Optional[str]
) -> Dict[str, str]:
    """Return REAPI environment variables configured for a pytest CI profile."""
    if not ci_profile:
        return {}
    profiles = (
        _load_pyproject(rootdir)
        .get("tool", {})
        .get("rtp_llm", {})
        .get("pytest_ci", {})
        .get("profiles", {})
    )
    profile = profiles.get(ci_profile, {})
    remote_env = profile.get("remote_env", {}) if isinstance(profile, dict) else {}
    if not isinstance(remote_env, dict):
        return {}
    return {str(key): str(value) for key, value in remote_env.items()}


def resolve_gpu_type_remote_env(rootdir: Path, gpu_type: str) -> Dict[str, str]:
    """Return shared and profile-specific remote env for a GPU type."""
    pytest_ci = (
        _load_pyproject(rootdir)
        .get("tool", {})
        .get("rtp_llm", {})
        .get("pytest_ci", {})
    )
    gpu_env = pytest_ci.get("gpu_env", {}).get(gpu_type, {})
    merged: Dict[str, str] = (
        {str(key): str(value) for key, value in gpu_env.items()}
        if isinstance(gpu_env, dict)
        else {}
    )
    profiles = pytest_ci.get("profiles", {})
    for profile in profiles.values():
        if not isinstance(profile, dict) or str(profile.get("gpu_type", "")) != gpu_type:
            continue
        remote_env = profile.get("remote_env", {})
        if isinstance(remote_env, dict):
            merged.update({str(key): str(value) for key, value in remote_env.items()})
    return merged


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
    input_root_hash: Optional[str] = None,
) -> RemoteRuntimeConfig:
    """Build env/platform/ignore/setup from a normalized GPU request."""
    env_vars: dict = {
        "GPU_COUNT": str(gpu_request.gpu_count),
    }
    reapi_gpu_type = _to_reapi_gpu_type(rootdir, gpu_request.gpu_type)
    if not reapi_gpu_type.startswith("MI308X"):
        env_vars["CUDA_HOME"] = "/usr/local/cuda"
    if input_root_hash:
        env_vars["RTP_CAS_INPUT_ROOT"] = input_root_hash[:12]
    setup_env = resolve_gpu_type_remote_env(rootdir, gpu_request.gpu_type)
    if extra_env:
        setup_env.update({str(key): str(value) for key, value in extra_env.items()})
        env_vars.update(extra_env)
    env_vars.update(setup_env)

    platform_properties: dict = {
        "gpu": reapi_gpu_type,
        "gpu_count": str(gpu_request.gpu_count),
    }

    return RemoteRuntimeConfig(
        ignore_args=get_pytest_ignore_args(rootdir),
        env_vars=env_vars,
        platform_properties=platform_properties,
        remote_setup_prefix=build_remote_setup_command(rootdir, setup_env=setup_env),
    )
