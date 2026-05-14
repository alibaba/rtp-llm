#!/usr/bin/env python3
"""
RTP-LLM Setup Script

This script integrates Bazel C++ extension building with setuptools.
The C++ extensions are built using Bazel and then copied to the package.

Wheel naming follows vllm convention:
  rtp_llm-{VERSION}+{PLATFORM}-cp{PY}-cp{PY}-{PLATFORM_TAG}.whl

Examples:
  rtp_llm-0.2.0+cu126-cp310-cp310-manylinux_2_28_x86_64.whl
  rtp_llm-0.2.0+rocm62-cp310-cp310-linux_x86_64.whl
"""
import datetime
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
from distutils.cmd import Command
from pathlib import Path

from setuptools import Extension
from setuptools import setup as setuptools_setup
from setuptools.command.build_ext import build_ext

# Force line-buffered stdout so prints appear in real-time in CI (no TTY)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

try:
    from wheel.bdist_wheel import bdist_wheel
except ImportError:
    bdist_wheel = None

# Ensure source directory is on sys.path so _build can be found even when
# setuptools exec's this file as '<string>' (prepare_metadata_for_build_*).
_src_dir = (
    os.path.dirname(os.path.abspath(__file__))
    if os.path.isfile(__file__)
    else os.getcwd()
)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

# Platform detection and config constants — shared with prepare_venv.py
from _build.platform import (  # noqa: E402
    _ensure_rocm_target_list,
    detect_build_config,
    extract_platform_from_config,
    get_config_to_extras,
    get_pip_extras,
    get_platform_config_versions,
    parse_bazel_config,
    should_skip_bazel_build,
)

_REMOTE_TESTS_PROTO_DIR = "rtp_llm/test/remote_tests"
# Protobuf sources + outputs for pytest REAPI client (generated during pip install / build_ext).
REMOTE_TESTS_PROTO_SOURCES = [
    f"{_REMOTE_TESTS_PROTO_DIR}/bytestream.proto",
    f"{_REMOTE_TESTS_PROTO_DIR}/remote_execution.proto",
]
REMOTE_TESTS_PROTO_OUTPUTS = [
    f"{_REMOTE_TESTS_PROTO_DIR}/bytestream_pb2.py",
    f"{_REMOTE_TESTS_PROTO_DIR}/bytestream_pb2_grpc.py",
    f"{_REMOTE_TESTS_PROTO_DIR}/remote_execution_pb2.py",
    f"{_REMOTE_TESTS_PROTO_DIR}/remote_execution_pb2_grpc.py",
]

PROTO_OUTPUTS = [
    "rtp_llm/cpp/model_rpc/proto/model_rpc_service_pb2.py",
    "rtp_llm/cpp/model_rpc/proto/model_rpc_service_pb2_grpc.py",
] + REMOTE_TESTS_PROTO_OUTPUTS

PROTO_SOURCES = [
    "rtp_llm/cpp/model_rpc/proto/model_rpc_service.proto",
] + REMOTE_TESTS_PROTO_SOURCES


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.absolute()


def get_release_version(project_root: Path = None) -> str:
    """Read the canonical package version without importing rtp_llm."""
    project_root = project_root or get_project_root()
    version_file = project_root / "rtp_llm" / "release_version.py"
    try:
        text = version_file.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(
            f"Failed to read release version from {version_file}"
        ) from exc

    match = re.search(r'^RELEASE_VERSION\s*=\s*["\']([^"\']+)["\']', text, re.M)
    if not match:
        raise RuntimeError(f"RELEASE_VERSION not found in {version_file}")
    return match.group(1)


def get_version_with_platform() -> str:
    """Get version string with platform suffix.

    Returns version like: 0.2.0+cu129 or 0.2.0+cu126
    Platform is always auto-detected from environment.
    """
    detected = detect_build_config()
    return f"{get_release_version()}+{get_platform_config_versions()[detected]}"


def get_platform_tag() -> str:
    """Get wheel platform tag.

    Returns platform tag like: linux_x86_64, manylinux_2_28_x86_64
    """
    machine = platform.machine()

    # Same platform key as Bazel build (RTP_BAZEL_CONFIG if set, else auto-detect)
    build_config = detect_build_config()

    # ARM builds
    if "arm" in build_config.lower() or machine == "aarch64":
        return f"manylinux_2_28_{machine}"

    # Standard Linux x86_64
    return f"linux_{machine}"


class BuildPrerequisiteError(RuntimeError):
    """Raised when a build prerequisite is missing, with actionable fix instructions."""

    pass


def has_generated_proto_files(project_root: Path = None) -> bool:
    """Return True when checked-in generated proto files are already present."""
    project_root = project_root or get_project_root()
    return all((project_root / rel_path).exists() for rel_path in PROTO_OUTPUTS)


def _installed_protobuf_has_runtime_version() -> bool:
    try:
        from google.protobuf import runtime_version  # noqa: F401

        return True
    except ImportError:
        return False


def generated_proto_files_stale(project_root: Path = None) -> bool:
    """True if on-disk *_pb2.py targets protobuf 5+ but runtime is older (e.g. protobuf 4.x)."""
    project_root = project_root or get_project_root()
    if _installed_protobuf_has_runtime_version():
        return False
    for rel in PROTO_OUTPUTS:
        if not rel.endswith("_pb2.py"):
            continue
        path = project_root / rel
        if not path.exists():
            continue
        try:
            head = path.read_text(encoding="utf-8", errors="replace")[:8000]
        except OSError:
            continue
        if "runtime_version" in head:
            return True
    return False


def check_build_prerequisites() -> None:
    """Validate that all required build tools and packages are available.

    Called early in setup.py before any real work begins.
    Checks differ based on whether we are doing a full build (C++ extensions)
    or a deps-only install (RTP_SKIP_BAZEL_BUILD=1).
    """
    errors = []
    skip_build = should_skip_bazel_build()
    project_root = get_project_root()
    needs_proto_generation = not has_generated_proto_files(project_root)

    if needs_proto_generation:
        try:
            import grpc_tools  # noqa: F401
        except ImportError:
            errors.append(
                "grpcio-tools is required to generate *_pb2.py from .proto when those "
                "files are absent (model_rpc + remote_tests REAPI). "
                "Fix: uv pip install --compile-bytecode grpcio-tools  (match your grpcio version, e.g. grpcio-tools==1.62.0)"
            )

    # -- Always required --
    import importlib.metadata as _meta

    try:
        sv = _meta.version("setuptools")
        major = int(sv.split(".")[0])
        if major < 75:
            errors.append(
                f"setuptools >= 64.0 required (found {sv}). "
                "Fix: uv pip install --compile-bytecode 'setuptools>=75.0,<82'"
            )
        elif major >= 82:
            errors.append(
                f"setuptools {sv} removed pkg_resources which breaks grpc_tools. "
                "Fix: uv pip install --compile-bytecode 'setuptools>=75.0,<82'"
            )
    except _meta.PackageNotFoundError:
        errors.append(
            "setuptools not found. Fix: uv pip install --compile-bytecode 'setuptools>=75.0,<82'"
        )

    if sys.version_info < (3, 11):
        try:
            import tomli  # noqa: F401
        except ImportError:
            errors.append(
                "tomli required for Python <3.11 to parse pyproject.toml. "
                "Fix: uv pip install --compile-bytecode tomli"
            )

    try:
        import wheel  # noqa: F401
    except ImportError:
        errors.append("wheel required. Fix: uv pip install --compile-bytecode wheel")

    # -- Required only for full build (C++ extensions) --
    if not skip_build:
        try:
            import torch  # noqa: F401
        except ImportError:
            errors.append(
                "torch not found. uv pip should have installed it before build_ext.\n"
                "  If running manually, ensure torch is installed first:\n"
                "    uv pip install --compile-bytecode -e '.[dev]' --no-build-isolation-package rtp-llm\n"
                "  Or for deps-only: RTP_SKIP_BAZEL_BUILD=1 uv pip install --compile-bytecode -e '.[dev]' --no-build-isolation"
            )

        if not shutil.which("bazelisk"):
            errors.append(
                "bazelisk not found in PATH. "
                "Install: https://github.com/bazelbuild/bazelisk/releases"
            )

    if errors:
        sep = "=" * 70
        header = f"\n{sep}\nBUILD PREREQUISITES CHECK FAILED\n{sep}"
        footer = (
            f"{sep}\n"
            "If you only need to install Python deps (skip C++ build), run:\n"
            "  RTP_SKIP_BAZEL_BUILD=1 uv pip install --compile-bytecode -e '.[dev]' --no-build-isolation\n"
            f"{sep}"
        )
        detail = "\n".join(f"  [{i+1}] {e}" for i, e in enumerate(errors))
        raise BuildPrerequisiteError(f"{header}\n{detail}\n{footer}")

    print("Build prerequisites check passed.")


def patch_remote_tests_grpc_stubs(project_root: Path) -> None:
    """Strip grpcio 1.63+-only API from generated *_pb2_grpc.py for grpcio 1.62 runtime.

    Newer grpc_tools emit ``_registered_method=True`` and
    ``add_registered_method_handlers`` which break grpcio==1.62 (project pin).
    """
    import re

    rt = project_root / "rtp_llm" / "test" / "remote_tests"
    for name in ("remote_execution_pb2_grpc.py", "bytestream_pb2_grpc.py"):
        path = rt / name
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        orig = text
        text = re.sub(r",\s*_registered_method=True\)", ")", text)
        text = re.sub(
            r"\n\s*server\.add_registered_method_handlers\([^)]*\)\s*",
            "\n",
            text,
        )
        if text != orig:
            path.write_text(text, encoding="utf-8")
            print(f"  Patched {name} for grpcio 1.62 compatibility")


def generate_proto_files(project_root: Path = None) -> None:
    """Generate Python files from .proto definitions.

    Compiles PROTO_SOURCES (model_rpc + remote_tests REAPI). Must run before
    wheel build or when *_pb2.py are missing (including RTP_SKIP_BAZEL_BUILD).
    """
    try:
        import grpc_tools
        from grpc_tools import protoc
    except ImportError as exc:
        raise BuildPrerequisiteError(
            "grpcio-tools is required to generate *_pb2.py from .proto, but it is not installed.\n"
            "Fix: uv pip install --compile-bytecode grpcio-tools  (e.g. grpcio-tools==1.62.0 to match grpcio)"
        ) from exc

    project_root = project_root or get_project_root()

    # grpcio-tools vendors well-known protos under grpc_tools/_proto,
    # e.g. google/protobuf/wrappers.proto. We must add it to proto_path.
    grpc_tools_proto_root = Path(grpc_tools.__file__).resolve().parent / "_proto"

    print("Generating protobuf files...")

    for proto_file in PROTO_SOURCES:
        proto_path = project_root / proto_file
        if not proto_path.exists():
            raise FileNotFoundError(f"Proto file not found: {proto_path}")

        print(f"  Compiling {proto_file}")
        args = [
            "grpc_tools.protoc",
            f"-I{project_root}",
            f"-I{grpc_tools_proto_root}",
            f"--python_out={project_root}",
            f"--grpc_python_out={project_root}",
            str(proto_path),
        ]
        code = protoc.main(args)
        if code != 0:
            raise RuntimeError(
                f"Failed to compile {proto_file}, exit code: {code}, args={args}"
            )

    patch_remote_tests_grpc_stubs(project_root)
    print("Proto file generation complete.")


def ensure_proto_files_generated(project_root: Path = None) -> None:
    """Generate *_pb2.py if any expected output is missing (always, not only with Bazel)."""
    project_root = project_root or get_project_root()
    if has_generated_proto_files(project_root) and not generated_proto_files_stale(
        project_root
    ):
        print("Using existing generated protobuf Python files.")
        return
    if generated_proto_files_stale(project_root):
        print(
            "Regenerating protobuf Python files (protobuf runtime vs generated code mismatch)."
        )
        for rel in PROTO_OUTPUTS:
            p = project_root / rel
            if p.exists():
                p.unlink()
    generate_proto_files(project_root)


def rewrite_torch_root() -> None:
    """Rewrite .torch_bazelrc to replace $(TORCH_ROOT) with actual torch installation path.

    This function detects the torch installation path from the current Python environment
    and updates .torch_bazelrc to use the actual path instead of the environment variable.
    """

    import torch

    # Get torch root directory (parent of torch package directory)
    # torch.__path__[0] gives us the torch package directory
    torch_package_dir = Path(torch.__path__[0])
    torch_root = str(torch_package_dir.absolute())
    print(f"Detected TORCH_ROOT: {torch_root}")

    content = """
build --action_env=TORCH_ROOT=$(TORCH_ROOT)
build --host_action_env=TORCH_ROOT=$(TORCH_ROOT)
"""

    project_root = get_project_root()
    torch_bazelrc_path = project_root / ".torch_bazelrc"

    with open(torch_bazelrc_path, "w+") as f:
        content = content.replace("$(TORCH_ROOT)", torch_root)
        f.write(content)

    # Only write if content changed
    print(f"Updated {torch_bazelrc_path} with TORCH_ROOT={torch_root}")


def _is_arm_bazel_config(bazel_args: list) -> bool:
    """Detect ARM build from resolved --config flags.

    cuda12_9_arm (Blackwell ARM) has no remote executor pool — only the
    REAPI cache is available. Match on the user-resolved --config tokens
    rather than the auto-detected default so explicit RTP_BAZEL_CONFIG
    overrides work too.
    """
    for arg in bazel_args:
        if arg.startswith("--config=") and "arm" in arg.split("=", 1)[1].lower():
            return True
    return False


def _get_bazel_cmd_prefix(build_config: str) -> tuple:
    """Get bazel command prefix and build args, shared by build and test.

    Returns:
        tuple: (cmd_prefix, build_args) where cmd_prefix includes bazelisk + startup options,
               and build_args are the config flags to append after the subcommand.
    """
    bazel_args = parse_bazel_config(default_config=build_config)
    arm_cache_only = _is_arm_bazel_config(bazel_args)
    bazel_args.extend(_get_remote_bazel_args(arm_cache_only=arm_cache_only))
    bazel_args.extend(_get_local_jobs_args(force_local=arm_cache_only))
    has_output_root = any("--output_user_root" in arg for arg in bazel_args)

    cmd = ["bazelisk"]
    if not has_output_root:
        cache_base = os.environ.get(
            "XDG_CACHE_HOME", os.path.join(os.path.expanduser("~"), ".cache")
        )
        cache_dir = os.path.join(cache_base, f"bazel_{build_config}_cache")
        cmd.append(f"--output_user_root={cache_dir}")
        print(f"Using platform-specific cache: {cache_dir}")

    build_args = [arg for arg in bazel_args if "--output_user_root" not in arg]
    return cmd, build_args


def is_remote_enabled() -> bool:
    """Check if remote build/test is enabled via RTP_REMOTE env var."""
    return os.environ.get("RTP_REMOTE", "").lower() in ("1", "true", "yes")


def _resolve_reapi_endpoint(online_host: str, daily_host: str, port: int) -> str:
    """Resolve a REAPI endpoint: try the production -online host first
    (typically service-discovery-backed), fall back to the -daily host if DNS
    doesn't resolve (e.g. dev workstation outside the prod VPC)."""
    try:
        results = socket.getaddrinfo(
            online_host, port, socket.AF_INET, socket.SOCK_STREAM
        )
        if results:
            ip = results[0][4][0]
            return f"grpc://{ip}:{port}"
    except (socket.gaierror, OSError):
        pass
    return f"grpc://{daily_host}:{port}"


def _find_overlay(rel: "str") -> "Optional[Path]":
    """Look up an overlay file under <repo_root>/<rel> then <project_root>/<rel>.

    REAPI workers receive files relative to project_root, so
    `internal_source/` ends up as a SIBLING of setup.py
    (project_root/internal_source/). In local dev
    `github-opensource/internal_source` is a symlink → ../internal_source,
    so project_root/internal_source/ resolves to the same directory as
    repo_root/internal_source/.

    Shared by `_load_remote_config` and `get_merged_optional_dependencies`.
    """
    project_root = get_project_root()
    repo_root = project_root.parent
    for base in (repo_root, project_root):
        cand = base / rel
        if cand.exists():
            return cand
    return None


def _read_toml_file(path: "Path") -> dict:
    """Parse a TOML file; return {} if missing or unparseable."""
    if not path.exists():
        return {}
    try:
        if sys.version_info >= (3, 11):
            import tomllib
        else:
            import tomli as tomllib  # type: ignore[no-redef]
        with open(path, "rb") as f:
            return tomllib.load(f)
    except Exception:
        return {}


def _load_remote_config() -> dict:
    """Load [tool.rtp-llm.remote] from gho pyproject.toml + internal overlay.

    Overlay keys win on conflict. Private endpoint values belong in the local
    overlay and must not be copied into the open-source pyproject.
    """
    cfg: dict = {}
    base = get_project_root() / "pyproject.toml"
    if base.exists():
        cfg.update(
            _read_toml_file(base).get("tool", {}).get("rtp-llm", {}).get("remote", {})
        )
    overlay = _find_overlay("internal_source/pyproject_internal.toml")
    if overlay:
        cfg.update(
            _read_toml_file(overlay)
            .get("tool", {})
            .get("rtp-llm", {})
            .get("remote", {})
        )
    return cfg


def _get_remote_bazel_args(arm_cache_only: bool = False) -> list:
    """Build remote-execution bazel args from pyproject.toml config.

    Returns extra bazel args when RTP_REMOTE is enabled, empty list otherwise.

    When ``arm_cache_only`` is True (cuda12_9_arm builds), only remote cache
    is wired up — the ARM REAPI executor pool does not exist, so the build
    runs locally but still pulls/pushes the cache.
    """
    if not is_remote_enabled():
        return []

    cfg = _load_remote_config()
    if not cfg:
        gho_pp = get_project_root() / "pyproject.toml"
        print(
            "WARNING: RTP_REMOTE=1 but [tool.rtp-llm.remote] missing.\n"
            f"  searched: {gho_pp}\n"
            "  searched: <repo_root|project_root>/internal_source/pyproject_internal.toml"
        )
        return []

    cas_ep = _resolve_reapi_endpoint(
        cfg.get("cas-online", ""),
        cfg.get("cas-daily", ""),
        int(cfg.get("cas-port", 50051)),
    )

    args = [
        "--config=cicd",
        "--config=online_aone_bazel_cache",
        f"--remote_cache={cas_ep}",
    ]

    if arm_cache_only:
        print(
            f"ARM build (cuda12_9_arm): remote cache only, no executor — {' '.join(args)}"
        )
        return args

    exec_ep = _resolve_reapi_endpoint(
        cfg.get("executor-online", ""),
        cfg.get("executor-daily", ""),
        int(cfg.get("executor-port", 50052)),
    )
    jobs = int(cfg.get("jobs", 320))
    args.extend(
        [
            "--config=online_aone_bazel_remote",
            f"--remote_executor={exec_ep}",
            f"--jobs={jobs}",
        ]
    )
    print(f"Remote build enabled: {' '.join(args)}")
    return args


def _get_local_jobs_args(force_local: bool = False) -> list:
    """Get --jobs arg based on local CPU count.

    Skipped when remote execution is on, *unless* ``force_local`` is True
    (e.g. ARM cache-only builds, which still execute actions locally).
    """
    if is_remote_enabled() and not force_local:
        return []
    cpu = int(
        os.environ.get(
            "AONE_CI_REQUESTED_CPU", os.environ.get("AJDK_MAX_PROCESSORS_LIMIT", "0")
        )
    )
    if cpu > 0:
        return [f"--jobs={cpu * 10}"]
    return []


# ---------------------------------------------------------------------------
# REAPI retry logic
# ---------------------------------------------------------------------------

# Source-of-truth lives in _build/reapi_retry.py — shared with the pytest-remote
# plugin (rtp_llm/test/remote_tests/plugin.py) so both wrap-points retry on the
# same set of transient codes. See module docstring for rationale.
from _build.reapi_retry import REAPI_RETRYABLE_EXIT_CODES, reapi_max_retries

_REAPI_RETRYABLE_EXIT_CODES = REAPI_RETRYABLE_EXIT_CODES
_REAPI_MAX_RETRIES = reapi_max_retries()

_STAGED_OUTPUT_CORE = "core"
_STAGED_OUTPUT_RUNTIME = "runtime"

_CORE_BAZEL_STAGED_OUTPUTS = [
    (
        _STAGED_OUTPUT_CORE,
        "//:th_transformer",
        ("libth_transformer.so",),
    ),
    (
        _STAGED_OUTPUT_CORE,
        "//:rtp_compute_ops",
        ("librtp_compute_ops.so",),
    ),
    (
        _STAGED_OUTPUT_CORE,
        "//:th_transformer_config",
        ("libth_transformer_config.so",),
    ),
]

_REMOTE_KVCM_RUNTIME_BAZEL_STAGED_OUTPUTS = [
    (
        _STAGED_OUTPUT_RUNTIME,
        "//3rdparty/remote_kv_cache_manager:remote_kv_cache_manager_client_shared",
        ("kv_cache_manager_client.so",),
    ),
    (
        _STAGED_OUTPUT_RUNTIME,
        "//3rdparty/3fs:hf3fs_shared",
        (
            "libhf3fs_api_shared.so",
            "libboost_atomic.so.1.71.0",
            "libboost_context.so.1.71.0",
            "libboost_filesystem.so.1.71.0",
            "libboost_program_options.so.1.71.0",
            "libboost_regex.so.1.71.0",
            "libboost_system.so.1.71.0",
            "libboost_thread.so.1.71.0",
            "libdouble-conversion.so.3",
            "libgflags.so.2.2",
            "libglog.so.0",
            "libevent-2.1.so.7",
            "libdwarf.so.1",
            "libicui18n.so.66",
            "libicuuc.so.66",
            "libicudata.so.66",
            "libunwind.so.8",
            "libssl.so.1.1",
            "libcrypto.so.1.1",
        ),
    ),
]

_REMOTE_KVCM_SERVER_BAZEL_STAGED_OUTPUTS = [
    (
        _STAGED_OUTPUT_RUNTIME,
        "//:remote_kv_cache_manager_server_bin",
        (("kv_cache_manager_bin", "kv_cache_manager_server/bin/kv_cache_manager_bin"),),
    ),
]


def _bazel_config_names(bazel_args: list) -> set:
    """Return config names from resolved Bazel args."""
    names = set()
    for i, arg in enumerate(bazel_args):
        if arg.startswith("--config="):
            names.add(arg.split("=", 1)[1])
        elif arg == "--config" and i + 1 < len(bazel_args):
            names.add(bazel_args[i + 1])
    return names


def _include_remote_kvcm_runtime_outputs(bazel_args: list) -> bool:
    """Whether to build/package RPM-backed remote-KVCM runtime libraries.

    ARM cache-only builds skip remote-KVCM runtime packaging in this workflow.
    Keep those builds lean and avoid forcing x86 RPM targets onto non-x86
    platforms. Other local overlay configs may opt out with
    [tool.rtp-llm.remote_kvcm_skip_configs].
    """
    config_names = _bazel_config_names(bazel_args)
    overlay = _find_overlay("internal_source/pyproject_internal.toml")
    overlay_toml = _read_toml_file(overlay) if overlay else {}
    skip_configs = set(
        overlay_toml.get("tool", {})
        .get("rtp-llm", {})
        .get("remote_kvcm_skip_configs", [])
    )
    return not any(
        name in skip_configs or "arm" in name.lower() for name in config_names
    )


def _selected_bazel_staged_outputs(build_config: str, bazel_args: list = None) -> list:
    """Return the Bazel targets and outputs that should be staged into libs/."""
    if bazel_args is None:
        bazel_args = parse_bazel_config(default_config=build_config)
    staged_outputs = list(_CORE_BAZEL_STAGED_OUTPUTS)
    include_remote_kvcm = _include_remote_kvcm_runtime_outputs(bazel_args)
    if include_remote_kvcm:
        staged_outputs.extend(_REMOTE_KVCM_RUNTIME_BAZEL_STAGED_OUTPUTS)
        staged_outputs.extend(_REMOTE_KVCM_SERVER_BAZEL_STAGED_OUTPUTS)
    return staged_outputs


def _bazel_targets_for_staged_outputs(staged_outputs: list) -> list:
    """Deduplicate Bazel targets while preserving declaration order."""
    targets = []
    seen = set()
    for _, target, _ in staged_outputs:
        if target not in seen:
            targets.append(target)
            seen.add(target)
    return targets


def _has_remote_download_arg(bazel_args: list) -> bool:
    return any(arg.startswith("--remote_download") for arg in bazel_args)


def _with_default_remote_download(bazel_args: list, mode: str) -> list:
    """Add a remote output download default unless the user already chose one."""
    if not is_remote_enabled() or _has_remote_download_arg(bazel_args):
        return bazel_args
    return [*bazel_args, f"--remote_download_{mode}"]


# Canonical PATH used when invoking bazelisk. Bazel includes PATH in the
# action environment hash by default, so a transient PATH (uv build venv,
# user shell tweaks, /tmp/build-env-…) busts the REAPI cache. Pinning to
# the standard system PATH keeps cache hits stable across local dev,
# CI, and the uv build-isolation env.
_BAZEL_FIXED_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"


def _bazel_subprocess_env() -> dict:
    """Return os.environ with PATH pinned to ``_BAZEL_FIXED_PATH``."""
    env = os.environ.copy()
    env["PATH"] = _BAZEL_FIXED_PATH
    return env


def _resolve_bazel_cmd(cmd: list) -> list:
    """Absolute-resolve ``cmd[0]`` if it's a bare bazelisk/bazel name.

    The fixed PATH (_BAZEL_FIXED_PATH) does not include user-local install
    locations like ``~/.nvm/.../bin`` where bazelisk often lives. Resolve
    against the *current* PATH before swapping the child env, so the spawn
    finds the binary while bazelisk's children still see the canonical PATH.
    """
    if not cmd:
        return cmd
    head = cmd[0]
    if os.path.isabs(head) or "/" in head:
        return cmd
    resolved = shutil.which(head)
    if resolved:
        return [resolved] + cmd[1:]
    return cmd


def _run_bazel_with_retry(
    cmd: list,
    max_retries: int = _REAPI_MAX_RETRIES,
    **kwargs,
) -> subprocess.CompletedProcess:
    """Run a bazel command, retrying on transient REAPI failures."""
    kwargs.setdefault("env", _bazel_subprocess_env())
    cmd = _resolve_bazel_cmd(cmd)
    for attempt in range(max_retries + 1):
        result = subprocess.run(cmd, **kwargs)
        if result.returncode not in _REAPI_RETRYABLE_EXIT_CODES:
            return result
        if attempt < max_retries:
            wait = 10 * (attempt + 1)
            print(
                f"[retry] Bazel REAPI error (exit {result.returncode}), "
                f"retrying in {wait}s ({attempt + 1}/{max_retries})..."
            )
            import time

            time.sleep(wait)
        else:
            print(
                f"[retry] Bazel REAPI error (exit {result.returncode}), "
                f"exhausted {max_retries} retries."
            )
    return result


def build_bazel_extensions(build_config: str) -> None:
    """Build C++ extensions using Bazel.

    Args:
        build_config: Platform name from detect_build_config() (cuda12_9,
            rocm, …) used for logging and as the default ``--config=`` when
            RTP_BAZEL_CONFIG is unset. If RTP_BAZEL_CONFIG is set, it overrides
            that default via parse_bazel_config().
    """
    # Rewrite .torch_bazelrc with actual torch installation path
    rewrite_torch_root()

    # Validate ROCm target list before building
    if build_config == "rocm":
        _ensure_rocm_target_list()

    project_root = get_project_root()

    # Create log directory
    log_dir = project_root / "build_logs"
    log_dir.mkdir(exist_ok=True)

    # Log file with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"bazel_build_{build_config}_{timestamp}.log"

    print(f"=" * 60)
    print(f"Building C++ extensions with config: {build_config}")
    print(f"Project root: {project_root}")
    print(f"Build log: {log_file}")
    print(f"=" * 60)

    # Auto-set output_user_root for platform-specific cache isolation.
    # This keeps accelerator builds from sharing cache state.
    # Note: --output_user_root must be before the 'build' command
    cmd, build_args = _get_bazel_cmd_prefix(build_config)
    build_args = _with_default_remote_download(build_args, "toplevel")
    staged_outputs = _selected_bazel_staged_outputs(build_config, build_args)
    targets = _bazel_targets_for_staged_outputs(staged_outputs)

    # Add 'build' command and targets
    cmd.extend(["build", *targets])

    # Add config args
    cmd.extend(build_args)

    # Note: We don't add --jobs by default, let Bazel decide based on system resources.
    # Users can add --jobs=N via RTP_BAZEL_CONFIG if needed.

    # Pin PATH for the bazelisk subprocess so the action env hash stays stable
    # across local/CI/uv-build-isolation invocations (see _BAZEL_FIXED_PATH).
    # Absolute-resolve cmd[0] first — bazelisk may live off the canonical PATH
    # (e.g. ~/.nvm/.../bin), in which case execvp would fail under fixed PATH.
    bazel_env = _bazel_subprocess_env()
    cmd = _resolve_bazel_cmd(cmd)
    print(f"PATH (bazel subprocess): {bazel_env['PATH']}")
    print(f"Running: {' '.join(cmd)}")
    print(f"This may take a while... Check {log_file} for progress.")

    # Run bazel with output redirected to log file (with REAPI retry)
    last_error = None
    for attempt in range(_REAPI_MAX_RETRIES + 1):
        try:
            with open(log_file, "w") as f:
                # Write command info to log
                f.write(f"Command: {' '.join(cmd)}\n")
                f.write(f"Working directory: {project_root}\n")
                f.write(f"Build config: {build_config}\n")
                f.write(f"Started at: {datetime.datetime.now().isoformat()}\n")
                if attempt > 0:
                    f.write(f"Retry attempt: {attempt}/{_REAPI_MAX_RETRIES}\n")
                f.write("=" * 60 + "\n\n")
                f.flush()

                # Run subprocess with output to both file and terminal (tee-like behavior)
                process = subprocess.Popen(
                    cmd,
                    cwd=project_root,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    env=bazel_env,
                )

                # Stream output to both file and stdout in real-time
                while True:
                    line = process.stdout.readline()
                    if not line and process.poll() is not None:
                        break
                    if line:
                        f.write(line)
                        f.flush()
                        sys.stdout.write(line)
                        sys.stdout.flush()

                process.wait()

                # Write completion info
                f.write("\n" + "=" * 60 + "\n")
                f.write(f"Finished at: {datetime.datetime.now().isoformat()}\n")
                f.write(f"Exit code: {process.returncode}\n")

                if process.returncode != 0:
                    raise subprocess.CalledProcessError(process.returncode, cmd)

            # Success — break out of retry loop
            break

        except subprocess.CalledProcessError as e:
            last_error = e
            if (
                e.returncode in _REAPI_RETRYABLE_EXIT_CODES
                and attempt < _REAPI_MAX_RETRIES
            ):
                wait = 10 * (attempt + 1)
                print(
                    f"\n[retry] Bazel REAPI error (exit {e.returncode}), "
                    f"retrying in {wait}s ({attempt + 1}/{_REAPI_MAX_RETRIES})..."
                )
                import time

                time.sleep(wait)
                continue

            print(f"\n" + "=" * 60)
            print(f"ERROR: Bazel build failed with exit code {e.returncode}")
            print(f"Check the full log at: {log_file}")
            print(f"Last 50 lines of log:")
            print("=" * 60)

            # Print last 50 lines of log
            try:
                with open(log_file, "r") as f:
                    lines = f.readlines()
                    for line in lines[-50:]:
                        print(line.rstrip())
            except Exception:
                pass

            raise RuntimeError(f"Bazel build failed. Check log file: {log_file}") from e

    print(f"\nBazel build completed successfully!")
    print(f"Full log available at: {log_file}")

    # Stage fresh libs only after Bazel succeeded (if the build failed above, we
    # never wipe — previous rtp_llm/libs is left intact).
    _wipe_rtp_llm_libs(project_root)
    copy_extensions(
        project_root,
        build_config,
        staged_outputs=staged_outputs,
        require_all_outputs=True,
    )


def _safe_copy_so(src: Path, dst: Path) -> bool:
    """Copy .so file, handling read-only bazel outputs.

    Returns True on success, False if src is a broken symlink / missing
    (which can happen when copy_extensions runs against a stale bazel-bin
    where symlinks point to garbage-collected paths — common when
    RTP_SKIP_BAZEL_BUILD=1 is used to seed test venvs).
    """
    try:
        if not src.exists():
            print(f"  Warning: source missing or broken symlink: {src}")
            return False
        if dst.exists():
            dst.chmod(0o755)
        shutil.copy2(src, dst)
        dst.chmod(0o755)
        return True
    except (OSError, FileNotFoundError) as e:
        print(f"  Warning: copy failed {src} -> {dst}: {e}")
        return False


def _bazel_target_output_dir(bazel_bin: Path, target: str) -> Path:
    """Return the bazel-bin package directory for a target label."""
    if not target.startswith("//"):
        return bazel_bin
    package = target[2:].split(":", 1)[0]
    return bazel_bin / package if package else bazel_bin


def _find_bazel_output_for_target(
    bazel_bin: Path, target: str, output_name: str
) -> Path:
    """Find an output declared for a specific top-level Bazel target."""
    expected = _bazel_target_output_dir(bazel_bin, target) / output_name
    if expected.exists():
        return expected

    found = [p for p in bazel_bin.glob(f"**/{output_name}") if p.is_file()]
    if not found:
        return None

    # Prefer target package outputs over incidental shared-library symlink trees.
    found.sort(key=lambda p: ("_solib" in p.parts, len(p.parts), str(p)))
    return found[0]


def _normalize_staged_output(output_spec) -> tuple:
    """Return ``(source_name, destination_relpath)`` for a staging output."""
    if isinstance(output_spec, tuple):
        return output_spec
    return output_spec, output_spec


def stage_bazel_outputs(
    project_root: Path,
    staged_outputs: list,
    require_all_outputs: bool = True,
) -> None:
    """Copy Bazel target outputs into ``rtp_llm/libs``.

    Each copied file is declared next to its Bazel target in the staged-output
    tables above. This keeps top-level build targets and packaging
    expectations in one place, which is important when remote builds use
    ``--remote_download_toplevel``.
    """
    bazel_bin = project_root / "bazel-bin"
    target_dir = project_root / "rtp_llm" / "libs"
    target_dir.mkdir(parents=True, exist_ok=True)

    missing = []
    copied_count = 0
    for kind, target, output_specs in staged_outputs:
        print(f"  target {target}:")
        for output_spec in output_specs:
            output_name, dest_relpath = _normalize_staged_output(output_spec)
            dst = target_dir / dest_relpath
            dst.parent.mkdir(parents=True, exist_ok=True)
            src = _find_bazel_output_for_target(bazel_bin, target, output_name)
            if not src:
                print(f"    Warning: {output_name} not found for {target}")
                if require_all_outputs or kind == _STAGED_OUTPUT_CORE:
                    missing.append((target, output_name))
                continue
            print(f"    {src} -> {dst}")
            if _safe_copy_so(src, dst):
                copied_count += 1

    if missing:
        existing = sorted(p.name for p in target_dir.glob("*.so"))
        detail = "\n".join(f"    - {target}: {name}" for target, name in missing)
        raise RuntimeError(
            "Build succeeded but required Bazel outputs are missing:\n"
            f"{detail}\n"
            f"  Existing in {target_dir}: {existing}\n"
            "  Possible causes:\n"
            "    - bazel-bin symlink is broken\n"
            "    - a staging target did not produce its declared output\n"
            "    - remote_download_toplevel was used without a matching top-level target"
        )

    print(f"  Copied {copied_count} staged Bazel output(s)")


def generate_pyi_stubs(project_root: Path) -> None:
    """Generate .pyi type stubs from compiled pybind11 .so modules.

    Hard-fails if any module's stub generation fails: stubgen failure is
    almost always the .so itself being broken (missing symbol, broken
    pybind binding), so surfacing it at build time catches real binding
    bugs that would otherwise show up as runtime ImportError under
    heavy logs.
    """
    import re as _re

    ops_dir = project_root / "rtp_llm" / "ops"
    modules = ["libth_transformer_config", "libth_transformer", "librtp_compute_ops"]
    failures: list = []  # list of (module, exit_code, stderr) — real .so bugs
    skipped: list = []  # list of (module, stderr) — host can't dlopen target

    # Host-environment shape: dynamic linker can't find a system shared lib
    # the .so links against (libcuda.so.1 on a CPU-only frontend container,
    # libtorch.so on a venv that hasn't installed torch yet, etc.). The .so
    # itself is fine for the wheel — runtime hosts will load it normally.
    # Skip stubgen for that target rather than failing the build.
    HOST_LIB_MISSING = _re.compile(
        r"cannot open shared object file: No such file or directory"
    )

    # The .so files live in rtp_llm/libs/. pybind11_stubgen runs as a
    # fresh subprocess and would do a bare `importlib.import_module(
    # "libth_transformer_config")`, which fails because rtp_llm/ops/
    # __init__.py is what normally:
    #   1. adds libs/ to sys.path,
    #   2. `import torch` → torch's _load_global_deps → CDLL of
    #      libtorch_global_deps.so with RTLD_GLOBAL,
    #   3. (ROCm only) cdll.LoadLibrary(libcaffe2_nvrtc.so) — the
    #      explicit hack at rtp_llm/ops/__init__.py:77-88,
    #   4. cdll.LoadLibrary(libpython3.10.so).
    # Mirror production by routing through `python -c "import rtp_llm.ops"`
    # before pybind11_stubgen — that runs the full init sequence in the
    # subprocess, so any present and future preload hacks auto-apply.
    # Project root on PYTHONPATH so `rtp_llm` is importable; libs dir
    # too so pybind11_stubgen's own importlib.import_module resolves.
    libs_path = project_root / "rtp_llm" / "libs"
    stubgen_env = os.environ.copy()
    stubgen_env["PYTHONPATH"] = (
        f"{project_root}{os.pathsep}{libs_path}"
        f"{os.pathsep}{stubgen_env.get('PYTHONPATH', '')}"
    )

    print("\nGenerating .pyi stubs via pybind11_stubgen:")
    for module in modules:
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import rtp_llm.ops, runpy, sys; "
                    f"sys.argv = ['pybind11_stubgen', {module!r}, '-o', {str(ops_dir)!r}]; "
                    "runpy.run_module('pybind11_stubgen', run_name='__main__')",
                ],
                cwd=str(project_root),
                env=stubgen_env,
                timeout=60,
                capture_output=True,
                text=True,
            )
        except Exception as e:
            failures.append((module, -1, repr(e)))
            print(f"  ERROR: pybind11_stubgen could not run for {module}: {e}")
            continue

        if result.returncode == 0:
            print(f"  Generated .pyi for {module}")
        elif HOST_LIB_MISSING.search(result.stderr or ""):
            skipped.append((module, result.stderr))
            print(
                f"  SKIP: pybind11_stubgen for {module} — host missing shared "
                "lib (.so itself OK, but this build host can't dlopen it)"
            )
        else:
            failures.append((module, result.returncode, result.stderr))
            print(
                f"  ERROR: pybind11_stubgen failed for {module} (exit={result.returncode})"
            )
            print("  --- stderr (full) ---")
            print(result.stderr or "<empty>")
            print("  --- end stderr ---")

    if skipped and not failures:
        names = ", ".join(m for m, _ in skipped)
        print(
            f"\nWARNING: pybind11_stubgen skipped for {len(skipped)} module(s): "
            f"{names} (host shared-lib missing). Build proceeds; runtime hosts "
            "with the right libs will load the .so normally."
        )

    if failures:
        names = ", ".join(m for m, _, _ in failures)
        raise RuntimeError(
            f"pybind11_stubgen failed for {len(failures)} module(s): {names}. "
            "This usually indicates a broken pybind binding or missing symbol "
            "in the compiled .so."
        )


def _wipe_rtp_llm_libs(project_root: Path) -> None:
    """Clear ``rtp_llm/libs`` before staging artifacts from ``bazel-bin``.

    Only call this immediately before ``copy_extensions`` when we know we will
    repopulate the directory. Do **not** call when ``RTP_SKIP_BAZEL_BUILD`` is
    set but ``bazel-bin`` has no outputs — otherwise a deps-only ``pip install``
    would erase a previously staged ``libs/`` from an earlier successful build.
    """
    target_dir = project_root / "rtp_llm" / "libs"
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)


def copy_extensions(
    project_root: Path,
    build_config: str,
    staged_outputs: list = None,
    require_all_outputs: bool = False,
) -> None:
    """Copy built Bazel outputs from bazel-bin to package.

    The output list is keyed by Bazel target so packaging stays aligned with
    the targets requested by ``build_bazel_extensions``. Core libraries are
    always required; runtime files are required only for fresh builds.
    """
    target_dir = project_root / "rtp_llm" / "libs"
    target_dir.mkdir(parents=True, exist_ok=True)
    staged_outputs = staged_outputs or _selected_bazel_staged_outputs(build_config)

    print(f"\nCopying extensions to {target_dir}:")
    stage_bazel_outputs(
        project_root,
        staged_outputs=staged_outputs,
        require_all_outputs=require_all_outputs,
    )

    # Auto-generate .pyi type stubs from the freshly compiled .so files
    generate_pyi_stubs(project_root)


def build_all():
    project_root = get_project_root()
    target_dir = project_root / "rtp_llm" / "libs"

    ensure_proto_files_generated(project_root)

    if not should_skip_bazel_build():
        # Auto-detect platform if RTP_BAZEL_CONFIG not set
        build_config = detect_build_config()
        # build_bazel_extensions runs Bazel, then wipes+stages libs only on success.
        build_bazel_extensions(build_config)
        return

    print("Skipping Bazel build (RTP_SKIP_BAZEL_BUILD is set)")
    bazel_bin = project_root / "bazel-bin"
    if bazel_bin.exists() and (bazel_bin / "libth_transformer.so").exists():
        build_config = detect_build_config()
        _wipe_rtp_llm_libs(project_root)
        copy_extensions(project_root, build_config)
    else:
        # Deps-only pass: do not wipe rtp_llm/libs — keep any .so from a prior
        # successful build_ext; otherwise prepare_venv pass 1 empties libs and
        # pass 2 has nothing until Bazel runs.
        print(
            f"  bazel-bin/libth_transformer.so absent — "
            f"leaving {target_dir} unchanged (deps-only install)"
        )


# a placeholder to make setuptools happy
class BazelExtension(Extension):
    def __init__(self):
        super().__init__("BazelExtension", sources=[])

    def build(self):
        build_all()


class BuildBazelExtension(build_ext):
    def run(self):
        check_build_prerequisites()
        for ext in self.extensions:
            if isinstance(ext, BazelExtension):
                ext.build()
        non_bazel = [e for e in self.extensions if not isinstance(e, BazelExtension)]
        if non_bazel:
            orig = self.extensions
            self.extensions = non_bazel
            super().run()
            self.extensions = orig

    def copy_extensions_to_source(self):
        """Override to skip copying BazelExtension .so files.

        BazelExtension is just a placeholder. The actual .so files are built
        by Bazel and copied to rtp_llm/libs/ by build_all(), so we don't need
        to copy them here. This prevents errors in editable mode when the
        placeholder .so file doesn't exist.
        """
        # Filter out BazelExtension from extensions to copy
        original_extensions = self.extensions
        self.extensions = [
            ext for ext in self.extensions if not isinstance(ext, BazelExtension)
        ]

        try:
            # Only copy non-Bazel extensions if any exist
            if self.extensions:
                super().copy_extensions_to_source()
            # If no extensions left, skip the copy step (this is fine for BazelExtension)
        finally:
            # Restore original extensions list
            self.extensions = original_extensions


if bdist_wheel is not None:

    class BdistWheelWithPlatform(bdist_wheel):
        """Custom bdist_wheel that sets platform-specific wheel name.

        Set WHEEL_COMPRESSION=stored (or pass --compression=stored) to skip
        zlib compression.  For large native .so wheels this can cut packaging
        time from minutes to seconds with negligible size difference.
        """

        def finalize_options(self):
            env_comp = os.environ.get("WHEEL_COMPRESSION")
            if env_comp and not getattr(self, "_compression_set_by_cli", False):
                self.compression = env_comp
            super().finalize_options()
            self.root_is_pure = False
            self.plat_name_supplied = True
            self.plat_name = get_platform_tag()

        def get_tag(self):
            # Get Python version
            py_version = f"cp{sys.version_info.major}{sys.version_info.minor}"

            # Return (python_tag, abi_tag, platform_tag)
            return (py_version, py_version, get_platform_tag())

else:
    BdistWheelWithPlatform = None


def _load_toml():
    """Load tomllib or tomli module."""
    try:
        import tomllib

        return tomllib
    except ImportError:
        try:
            import tomli as tomllib

            return tomllib
        except ImportError:
            return None


def get_base_dependencies() -> list:
    """Get base dependencies layered: root [tool.rtp-llm.base-dependencies] +
    optional internal [tool.rtp-llm.base_dependencies_extra].
    """
    tomllib = _load_toml()
    if not tomllib:
        print("Warning: tomllib/tomli not available")
        return []

    project_root = get_project_root()
    pyproject_path = project_root / "pyproject.toml"

    base_deps: list = []
    if pyproject_path.exists():
        try:
            with open(pyproject_path, "rb") as f:
                pyproject = tomllib.load(f)
            base_deps = list(
                pyproject.get("tool", {})
                .get("rtp-llm", {})
                .get("base-dependencies", [])
                or []
            )
        except Exception as e:
            print(f"Warning: Could not read base dependencies: {e}")

    # Internal overlay can contribute additional base deps. Driven purely by
    # file presence — OSS build CI strips internal_source/ via oss_strip.sh
    # so this short-circuits naturally when no overlay file exists.
    repo_root = project_root.parent
    internal_overlay = repo_root / "internal_source" / "pyproject_internal.toml"
    if internal_overlay.exists():
        try:
            with open(internal_overlay, "rb") as f:
                data = tomllib.load(f)
            extra = (
                data.get("tool", {})
                .get("rtp-llm", {})
                .get("base_dependencies_extra", [])
                or []
            )
            if extra:
                print(f"[overlay] internal: extend base-dependencies (+{len(extra)})")
                base_deps.extend(extra)
        except Exception as e:
            print(f"Warning: Could not read internal base-deps overlay: {e}")

    return base_deps


def _load_extras_from_toml(path) -> dict:
    """Load [project.optional-dependencies] from a pyproject*.toml file.

    Returns empty dict if the file is missing or unparseable.
    """
    if not path.exists():
        return {}
    tomllib = _load_toml()
    if not tomllib:
        return {}
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
        return data.get("project", {}).get("optional-dependencies", {}) or {}
    except Exception as e:
        print(f"Warning: failed to load extras from {path}: {e}")
        return {}


def _load_overlay_meta(path) -> dict:
    """Read [tool.rtp-llm.platform-overlay] metadata from an overlay file."""
    if not path.exists():
        return {}
    tomllib = _load_toml()
    if not tomllib:
        return {}
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
        return data.get("tool", {}).get("rtp-llm", {}).get("platform-overlay", {}) or {}
    except Exception:
        return {}


def get_merged_optional_dependencies() -> dict:
    """Return [project.optional-dependencies] merged with internal overlays.

    OSS extras are loaded from ``_build/oss_optional_extras.toml`` when present
    (GPU pins split out of ``pyproject.toml`` to avoid duplicating them under
    PEP621). ``setup()`` only exposes ``dev``/``docs``/``all`` via
    ``extras_require``; platform stacks are merged into ``install_requires``
    (``get_all_dependencies()``) so installers never see two torch pins.
    Otherwise fall back to ``pyproject.toml``.

    Layer 1 — OSS extras file or pyproject.toml.
    Layer 2 — internal_source/pyproject_internal.toml (override / extend).
    Layer 3 — any extra overlay files declared by layer 2.
    """
    project_root = get_project_root()
    repo_root = project_root.parent
    extras_file = project_root / "_build" / "oss_optional_extras.toml"
    if extras_file.exists():
        root_extras = dict(_load_extras_from_toml(extras_file))
    else:
        root_extras = dict(_load_extras_from_toml(project_root / "pyproject.toml"))

    # Local overlays live at <repo_root>/internal_source/... or
    # <project_root>/internal_source/... — see module-level _find_overlay
    # for the layout reasoning.
    internal_overlay = _find_overlay("internal_source/pyproject_internal.toml")
    extra_overlay_files = []
    if internal_overlay:
        meta = _load_overlay_meta(internal_overlay)
        internal_extras = _load_extras_from_toml(internal_overlay)
        extra_overlay_files = list(meta.get("extra_overlay_files", []))
        mode = meta.get("mode", "extend")
        targets = meta.get("target_extras") or list(internal_extras.keys())
        for key in targets:
            if key in internal_extras:
                if mode == "override":
                    print(
                        f"[overlay] internal: override {key} ("
                        f"{len(root_extras.get(key, []))} -> "
                        f"{len(internal_extras[key])} pkgs)"
                    )
                    root_extras[key] = list(internal_extras[key])
                else:
                    root_extras.setdefault(key, []).extend(internal_extras[key])
        for key, deps in internal_extras.items():
            if key in (targets or []):
                continue
            print(f"[overlay] internal: extend {key} (+{len(deps)} pkgs)")
            root_extras.setdefault(key, []).extend(deps)

    for rel in extra_overlay_files:
        overlay = _find_overlay(str(rel))
        if not overlay:
            continue
        extra_extras = _load_extras_from_toml(overlay)
        for key, deps in extra_extras.items():
            print(f"[overlay] extra: extend {key} (+{len(deps)} pkgs)")
            root_extras.setdefault(key, []).extend(deps)

    return root_extras


def get_platform_dependencies() -> list:
    """Resolve platform dependencies via root + overlay chain.

    Same merge rules as get_merged_optional_dependencies(); returns only the
    extra list for the auto-detected build_config.
    """
    build_config = detect_build_config()
    if not build_config:
        return []

    root_extras = get_merged_optional_dependencies()

    dep_key = get_config_to_extras().get(build_config)
    deps = root_extras.get(dep_key, []) if dep_key else []
    print(
        f"Platform dependencies for {build_config} -> extras[{dep_key}]: "
        f"{len(deps)} packages"
    )
    return list(deps)


def get_all_dependencies() -> list:
    """Get all dependencies (base + platform-specific)."""
    base_deps = get_base_dependencies()
    platform_deps = get_platform_dependencies()

    # Merge: platform deps override base deps for same package
    all_deps = list(base_deps)

    # Get package names from platform deps to check for conflicts
    platform_packages = set()
    for dep in platform_deps:
        # Extract package name (before @ or version specifier)
        if "@" in dep:
            pkg = dep.split("@")[0].strip()
        elif ">=" in dep:
            pkg = dep.split(">=")[0].strip()
        elif "==" in dep:
            pkg = dep.split("==")[0].strip()
        elif "<" in dep:
            pkg = dep.split("<")[0].strip()
        else:
            pkg = dep.strip()
        platform_packages.add(pkg.lower().replace("-", "_").replace(".", "_"))

    # Filter out base deps that conflict with platform deps
    filtered_base = []
    for dep in all_deps:
        if "@" in dep:
            pkg = dep.split("@")[0].strip()
        elif ">=" in dep:
            pkg = dep.split(">=")[0].strip()
        elif "==" in dep:
            pkg = dep.split("==")[0].strip()
        elif "<" in dep:
            pkg = dep.split("<")[0].strip()
        else:
            pkg = dep.strip()

        normalized = pkg.lower().replace("-", "_").replace(".", "_")
        if normalized not in platform_packages:
            filtered_base.append(dep)

    # Combine
    result = filtered_base + platform_deps
    print(
        f"Total dependencies: {len(result)} (base: {len(filtered_base)}, platform: {len(platform_deps)})"
    )
    return result


def dynamic_version() -> str:
    """Hook for [tool.setuptools.dynamic] (pyproject.toml) — PEP 621."""

    return get_version_with_platform()


def dynamic_install_requires() -> list:
    """Hook for [tool.setuptools.dynamic] — must match setup(install_requires=...)."""

    return get_all_dependencies()


class BazelTest(Command):
    """Run C++ tests via Bazel.

    Usage:
        python setup.py test                          # run all C++ tests
        python setup.py test --compile-only           # compile test targets only
        python setup.py test --test-target=//rtp_llm/cpp/cache/test:...

    Bazel flags: same as build — RTP_BAZEL_CONFIG / RTP_BAZEL_APPEND_CONFIG when set,
    otherwise auto-detected platform (detect_build_config) supplies ``--config=``.
    Set RTP_REMOTE=1 to enable remote execution (REAPI endpoints from pyproject.toml).
    """

    description = "Run C++ tests via Bazel"
    user_options = [
        ("compile-only", None, "Only compile test targets, do not execute"),
        ("test-target=", None, "Bazel test target (default: //rtp_llm/cpp/...)"),
    ]
    boolean_options = ["compile-only"]

    def initialize_options(self):
        self.compile_only = False
        self.test_target = "//rtp_llm/cpp/..."

    def finalize_options(self):
        pass

    def run(self):
        build_config = detect_build_config()
        rewrite_torch_root()

        # Use same bazel cmd prefix as build_bazel_extensions for cache reuse
        cmd, build_args = _get_bazel_cmd_prefix(build_config)
        build_args = _with_default_remote_download(build_args, "minimal")

        if self.compile_only:
            cmd.extend(["build", self.test_target, "--build_tests_only"])
        else:
            cmd.extend(
                [
                    "test",
                    self.test_target,
                    "--build_tests_only=1",
                    "--run_under=//rtp_llm/test/utils:gpu_lock",
                ]
            )

        # Add config args (same as build)
        cmd.extend(build_args)

        # Add test-specific args from BAZEL_TEST_ARGS env var
        test_args = os.environ.get("BAZEL_TEST_ARGS", "")
        if test_args:
            print(f"Using BAZEL_TEST_ARGS: {test_args}")
            cmd.extend(test_args.split())

        project_root = get_project_root()
        mode = "compile-only" if self.compile_only else "test"
        print(f"Running bazel {mode}: {self.test_target}")
        print(f"Command: {' '.join(cmd)}")

        result = _run_bazel_with_retry(cmd, cwd=project_root)
        if result.returncode != 0 and result.returncode != 4:
            # exit code 4 = no test targets matched, not an error
            raise SystemExit(result.returncode)


# install_requires carries base + auto-detected platform (merged URLs).
# GPU extras (cuda12_9, rocm, …) live in _build/oss_optional_extras.toml for
# reference / tooling only — do NOT also pass them as setuptools extras_require
# or uv sees two torch pins (install_requires + extras).
#
# `dev` / `docs` are now declared statically in pyproject.toml
# [project.optional-dependencies] (uv reads PEP 621 directly and drops
# setup.py-only extras), so we MUST NOT also inject them here — setuptools
# rejects duplicate static + dynamic declarations. Only `all` (which references
# extras across files) stays here.
_merged_extras = get_merged_optional_dependencies()
_non_gpu_extras = {k: v for k, v in _merged_extras.items() if k in ("all",)}

all_deps = dynamic_install_requires()
version = dynamic_version()
print(f"Building rtp-llm version: {version}")

cmdclass = {"build_ext": BuildBazelExtension, "test": BazelTest}
if BdistWheelWithPlatform is not None:
    cmdclass["bdist_wheel"] = BdistWheelWithPlatform

# Only invoke setuptools when running as a script (`python setup.py …`). PEP 517 / setuptools
# imports this file to resolve ``dynamic_version`` / metadata; a top-level ``setup()`` call would
# re-enter setuptools while the module is still loading and break ``read_attr("setup.dynamic_version")``.
if __name__ == "__main__":
    setuptools_setup(
        version=version,
        install_requires=all_deps if all_deps else None,
        extras_require=_non_gpu_extras if _non_gpu_extras else None,
        ext_modules=[BazelExtension()],
        cmdclass=cmdclass,
        entry_points={
            "pytest11": [
                # These plugins intentionally live under rtp_llm.test.* and
                # tests are packaged in the wheel so installed pytest sessions
                # can load the same CI helpers used from a source checkout.
                "remote-gpu = rtp_llm.test.remote_tests.plugin",
                "rtp-ci-profile = rtp_llm.test.ci_profile_plugin",
                "smoke-runs-per-test = rtp_llm.test.smoke_framework.runs_plugin",
            ],
        },
    )
