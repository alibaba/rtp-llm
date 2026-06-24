"""Build-time platform detection for RTP-LLM.

Detects CUDA / ROCm environments from filesystem probes and environment
variables. Local overlays may contribute additional platform detectors. Used by
both setup.py and prepare_venv.py.

No runtime dependencies on rtp_llm — only stdlib + json.
"""

from __future__ import annotations

import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

# ============================================================================
# Bazel Configuration - Unified via RTP_BAZEL_CONFIG
# ============================================================================
#
# RTP_BAZEL_CONFIG is passed directly to bazelisk as command-line arguments.
# RTP_BAZEL_APPEND_CONFIG appends args after the default (auto-detect or RTP_BAZEL_CONFIG).
# Use native Bazel argument format.
#
# Default (recommended): leave RTP_BAZEL_CONFIG unset — detect_build_config()
# infers cuda12_* / rocm from the machine and setup.py passes ``--config=...``.
#
# Optional override when auto-detect is wrong or you need extra flags:
#   export RTP_BAZEL_CONFIG="--config=cuda12_6 --config=sm9x"
#   export RTP_BAZEL_CONFIG="--config=rocm --jobs=200"
#   export RTP_BAZEL_APPEND_CONFIG="--config=sm9x --jobs=200"
#
# Common configs:
#   --config=cuda12_6     CUDA 12.6
#   --config=cuda12_9     CUDA 12.9
#   --config=rocm         ROCm/AMD
#   --config=cpu / --config=arm are deprecated for pip packaging. They remain
#   Bazel concepts only and intentionally have no platform dependency mapping.
#
# ============================================================================

# Platform to version suffix mapping (for wheel naming)
PLATFORM_CONFIG_VERSIONS = {
    "cuda12_6": "cu126",
    "cuda12_9": "cu129",
    "cuda12_9_arm": "cu129",
    "rocm": "rocm62",
}

# Bazel config name -> pyproject.toml extras name
CONFIG_TO_EXTRAS = {
    "cuda12_6": "cuda12",
    "cuda12_9": "cuda12_9",
    "cuda12_9_arm": "cuda12_arm",
    "rocm": "rocm",
}

DEPRECATED_PIP_PLATFORM_CONFIGS = frozenset({"cpu", "arm"})


# ---------------------------------------------------------------------------
# Bazel config parsing
# ---------------------------------------------------------------------------


def _read_toml(path: Path) -> dict:
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


def _read_platform_configs_text(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Read overlay platform configs without tomli.

    prepare_venv.py calls this module before the venv exists, so Python 3.10 CI
    images may not have tomli yet. This tiny fallback only understands the
    platform-configs tables we need for detection.
    """
    versions = {}
    extras = {}
    current = ""
    prefix = "[tool.rtp-llm.platform-configs."
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return versions, extras

    values: dict[str, dict[str, str]] = {}
    for raw_line in lines:
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            current = ""
            if line.startswith(prefix):
                current = line[len(prefix) : -1].strip()
                values.setdefault(current, {})
            continue
        if not current or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key in ("version", "extra") and value:
            values[current][key] = value

    for name, cfg in values.items():
        version = cfg.get("version")
        extra = cfg.get("extra")
        if version and extra:
            versions[name] = version
            extras[name] = extra
    return versions, extras


def _find_overlay(rel: str) -> Path | None:
    project_root = Path(__file__).resolve().parents[1]
    repo_root = project_root.parent
    for base in (repo_root, project_root):
        cand = base / rel
        if cand.exists():
            return cand
    return None


def _overlay_platform_configs() -> tuple[dict[str, str], dict[str, str]]:
    """Return platform version/extras maps contributed by a local overlay."""
    overlay = _find_overlay("internal_source/pyproject_internal.toml")
    if not overlay:
        return {}, {}
    entries = (
        _read_toml(overlay)
        .get("tool", {})
        .get("rtp-llm", {})
        .get("platform-configs", {})
    )
    if not entries:
        return _read_platform_configs_text(overlay)

    versions = {}
    extras = {}
    for name, cfg in entries.items():
        if not isinstance(cfg, dict):
            continue
        version = cfg.get("version")
        extra = cfg.get("extra")
        if version and extra:
            versions[str(name)] = str(version)
            extras[str(name)] = str(extra)
    return versions, extras


def get_platform_config_versions() -> dict[str, str]:
    versions = dict(PLATFORM_CONFIG_VERSIONS)
    overlay_versions, _ = _overlay_platform_configs()
    versions.update(overlay_versions)
    return versions


def get_config_to_extras() -> dict[str, str]:
    extras = dict(CONFIG_TO_EXTRAS)
    _, overlay_extras = _overlay_platform_configs()
    extras.update(overlay_extras)
    return extras


def _get_bazel_config() -> str:
    """Get raw RTP_BAZEL_CONFIG environment variable."""
    return os.environ.get("RTP_BAZEL_CONFIG", "")


def _get_bazel_append_config() -> str:
    """Get raw RTP_BAZEL_APPEND_CONFIG environment variable."""
    return os.environ.get("RTP_BAZEL_APPEND_CONFIG", "")


def parse_bazel_config(default_config: str = "") -> list:
    """Parse RTP_BAZEL_CONFIG/RTP_BAZEL_APPEND_CONFIG and return bazel arguments.

    Returns:
        list: Bazel arguments parsed from environment config(s)

    Examples:
        "--config=custom_accel" -> ["--config=custom_accel"]
        "--config=cuda12_6 --config=sm9x" -> ["--config=cuda12_6", "--config=sm9x"]
    """
    bazel_config = _get_bazel_config()
    append_config = _get_bazel_append_config()
    args = []

    if bazel_config:
        print(f"Using RTP_BAZEL_CONFIG: {bazel_config}")
        args.extend(bazel_config.split())
    elif default_config:
        args.append(f"--config={default_config}")

    if append_config:
        print(f"Using RTP_BAZEL_APPEND_CONFIG: {append_config}")
        args.extend(append_config.split())

    return args


def extract_platform_from_config() -> str:
    """Extract platform name from RTP_BAZEL_CONFIG for wheel naming.

    Returns:
        str: Platform name (cuda12_6, rocm, etc.) or empty string
    """
    bazel_config = _get_bazel_config()
    for arg in bazel_config.split():
        if arg.startswith("--config="):
            config_value = arg.split("=", 1)[1]
            if config_value in get_platform_config_versions():
                return config_value
    return ""


def extract_deprecated_platform_from_config() -> str:
    """Return deprecated CPU/ARM platform configs requested explicitly."""
    bazel_config = _get_bazel_config()
    for arg in bazel_config.split():
        if arg.startswith("--config="):
            config_value = arg.split("=", 1)[1]
            if config_value in DEPRECATED_PIP_PLATFORM_CONFIGS:
                return config_value
    return ""


# ---------------------------------------------------------------------------
# Filesystem platform detection
# ---------------------------------------------------------------------------


def _get_cuda_version_from_json() -> str | None:
    """Read CUDA version from /usr/local/cuda/version.json.

    Returns:
        CUDA version string (e.g. "12.9.20250531"), or None.
    """
    version_json_path = Path("/usr/local/cuda/version.json")
    if not version_json_path.exists():
        return None

    try:
        with open(version_json_path, "r") as f:
            data = json.load(f)
            return data.get("cuda", {}).get("version")
    except (json.JSONDecodeError, KeyError, IOError):
        return None


def _get_cuda_version_from_nvcc() -> str | None:
    """Attempt to get CUDA version from nvcc --version output.

    Returns:
        CUDA version string (e.g. "12.6.0"), or None if nvcc is unavailable.
    """
    nvcc_path = shutil.which("nvcc")
    if not nvcc_path:
        return None
    try:
        result = subprocess.run(
            [nvcc_path, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        # Parse line like: "Cuda compilation tools, release 12.6, V12.6.77"
        match = re.search(r"release\s+(\d+\.\d+)", result.stdout)
        if match:
            return match.group(1) + ".0"
    except (subprocess.TimeoutExpired, OSError):
        pass
    return None


def _parse_cuda_major_minor(version_str: str) -> tuple[int, int] | None:
    parts = version_str.split(".")
    if len(parts) < 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def _get_cuda_config_from_version(version_str: str) -> str:
    """Map CUDA version string to config name.

    Args:
        version_str: e.g. "12.9.20250531" or "12.6.0"

    Returns:
        Config name, e.g. "cuda12_6" or "cuda12_9"
    """
    if not version_str:
        return "cuda12_6"

    version_key = _parse_cuda_major_minor(version_str)
    if version_key:
        if version_key >= (12, 9):
            return "cuda12_9"
        if version_key >= (12, 6):
            return "cuda12_6"
        return "cuda12_6"

    return "cuda12_6"


def _detect_cuda() -> bool:
    """Detect CUDA on the filesystem."""
    return Path("/usr/local/cuda").exists() or bool(shutil.which("nvcc"))


def _detect_rocm() -> bool:
    """Detect ROCm on the filesystem."""
    return Path("/opt/rocm").exists()


def _load_overlay_platform_detector():
    detector = _find_overlay("internal_source/ci/platform_detection.py")
    if not detector:
        return None

    spec = importlib.util.spec_from_file_location(
        "_rtp_llm_overlay_platform_detection",
        detector,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load platform detector overlay: {detector}")

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        raise RuntimeError(
            f"Failed to execute platform detector overlay: {detector}. Error: {e}"
        ) from e
    return module


def _detect_overlay_build_config(verbose: bool = True) -> str:
    """Ask a local overlay for platform detection, if it provides one."""
    module = _load_overlay_platform_detector()
    if module is None:
        return ""
    detector = getattr(module, "detect_build_config", None)
    if detector is None:
        return ""
    if not callable(detector):
        raise RuntimeError(
            "platform detector overlay detect_build_config is not callable"
        )

    known_configs = get_platform_config_versions()
    build_config = detector(known_configs)
    if not build_config:
        return ""
    if not isinstance(build_config, str):
        raise RuntimeError(
            f"platform detector overlay returned non-string config: {build_config!r}"
        )
    if build_config not in known_configs:
        raise RuntimeError(
            f"platform detector overlay returned unknown config {build_config!r}; "
            f"known configs: {sorted(known_configs)}"
        )
    if verbose:
        print(f"Detected overlay platform: {build_config}")
    return build_config


def _ensure_rocm_target_list() -> None:
    """Validate /opt/rocm/bin/target.lst is gfx942 only.

    CI images write this file at container build time. No env override or home
    fallback — keeps detection deterministic and a missing file is a real
    container-setup bug, not something to paper over.
    """
    target_list_path = Path("/opt/rocm/bin/target.lst")
    if not target_list_path.exists():
        raise RuntimeError(
            "ROCm target list not found: /opt/rocm/bin/target.lst. "
            "Container is missing the gfx942 target. Recreate the container "
            "or write it manually: "
            "echo gfx942 | sudo tee /opt/rocm/bin/target.lst"
        )

    try:
        with open(target_list_path, "r") as f:
            lines = [line.strip() for line in f.readlines()]
    except OSError as e:
        raise RuntimeError(
            f"Failed to read ROCm target list: {target_list_path}. Error: {e}"
        ) from e

    targets = [line for line in lines if line]
    if targets != ["gfx942"]:
        raise RuntimeError(
            f"ROCm target list mismatch in {target_list_path}: "
            f"expected only 'gfx942', got {targets}."
        )


# ---------------------------------------------------------------------------
# Main detection entry point
# ---------------------------------------------------------------------------

_cached_build_config: str | None = None


def detect_build_config(verbose: bool = True) -> str:
    """Detect appropriate build config based on environment.

    Checks (in order):
    1. RTP_BAZEL_CONFIG environment variable
    2. Local overlay platform detector
    3. CUDA toolkit (version.json -> cuda12_6 / cuda12_9 / cuda12_9_arm)
    4. ROCm directory

    Result is cached after first call.

    Raises:
        RuntimeError: if no platform can be detected.
    """
    global _cached_build_config
    if _cached_build_config is not None:
        return _cached_build_config

    deprecated_config = extract_deprecated_platform_from_config()
    if deprecated_config:
        raise RuntimeError(
            f"RTP_BAZEL_CONFIG requests deprecated pip platform config "
            f"{deprecated_config!r}. CPU/ARM dependency resolution is no longer "
            "supported by setup.py; use a supported accelerator config "
            "(cuda12_6, cuda12_9, cuda12_9_arm, rocm, or an overlay config)."
        )

    # 1. Check RTP_BAZEL_CONFIG first
    build_config = extract_platform_from_config()
    if build_config:
        _cached_build_config = build_config
        return build_config

    build_config = _detect_overlay_build_config(verbose=verbose)
    if build_config:
        _cached_build_config = build_config
        return build_config

    if _detect_cuda():
        cuda_version = _get_cuda_version_from_json()
        if cuda_version is None:
            cuda_version = _get_cuda_version_from_nvcc()
        if cuda_version is None:
            cuda_version = "12.6.0"
        cuda_config = _get_cuda_config_from_version(cuda_version)
        if verbose:
            print(
                f"Detected CUDA environment (version: {cuda_version}, config: {cuda_config})"
            )

        if platform.machine() == "aarch64" and cuda_config == "cuda12_9":
            _cached_build_config = "cuda12_9_arm"
            return "cuda12_9_arm"

        _cached_build_config = cuda_config
        return cuda_config

    if _detect_rocm():
        if verbose:
            print("Detected ROCm environment")
        _cached_build_config = "rocm"
        return "rocm"

    if should_skip_bazel_build():
        raise RuntimeError(
            "Cannot detect a supported accelerator build configuration for "
            "deps-only install. CPU/ARM pip packaging paths are deprecated; set "
            "RTP_BAZEL_CONFIG to a supported accelerator config such as "
            "'--config=cuda12_9' or '--config=rocm'."
        )

    raise RuntimeError(
        "Cannot detect build configuration. Please set RTP_BAZEL_CONFIG to a "
        "supported accelerator config. Example: export "
        "RTP_BAZEL_CONFIG='--config=cuda12_6 --config=sm9x'. CPU/ARM pip "
        "packaging configs are deprecated."
    )


def get_pip_extras() -> str:
    """Get pyproject.toml extras name for the detected platform.

    Returns extras name like: cuda12_9, cuda12, cuda12_arm, rocm
    """
    return get_config_to_extras()[detect_build_config(verbose=True)]


def should_skip_bazel_build() -> bool:
    """Check if Bazel build should be skipped (RTP_SKIP_BAZEL_BUILD is set)."""
    return bool(os.environ.get("RTP_SKIP_BAZEL_BUILD"))
