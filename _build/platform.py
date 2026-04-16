"""Build-time platform detection for RTP-LLM.

Detects PPU / CUDA / ROCm environments from filesystem probes and
environment variables. Used by both setup.py and prepare_venv.py.

No runtime dependencies on rtp_llm — only stdlib + json.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
from pathlib import Path

# ============================================================================
# Bazel Configuration - Unified via RTP_BAZEL_CONFIG
# ============================================================================
#
# RTP_BAZEL_CONFIG is passed directly to bazelisk as command-line arguments.
# RTP_BAZEL_APPEND_CONFIG appends args to the auto-detected default config.
# Use native Bazel argument format.
#
# Examples:
#   export RTP_BAZEL_CONFIG="--config=ppu"
#   export RTP_BAZEL_CONFIG="--config=cuda12_6 --config=sm9x"
#   export RTP_BAZEL_CONFIG="--config=rocm --jobs=200"
#   export RTP_BAZEL_APPEND_CONFIG="--config=sm9x --jobs=200"
#
# Common configs:
#   --config=ppu          PPU platform
#   --config=cuda12_6     CUDA 12.6
#   --config=cuda12_9     CUDA 12.9
#   --config=rocm         ROCm/AMD
#
# ============================================================================

# Platform to version suffix mapping (for wheel naming)
PLATFORM_CONFIG_VERSIONS = {
    "ppu": "ppu1.5.2",
    "cuda12_6": "cu126",
    "cuda12_9": "cu129",
    "cuda12_9_arm": "cu129",
    "rocm": "rocm62",
}

# Bazel config name -> pyproject.toml extras name
CONFIG_TO_EXTRAS = {
    "ppu": "ppu",
    "cuda12_6": "cuda12",
    "cuda12_9": "cuda12_9",
    "cuda12_9_arm": "cuda12_arm",
    "rocm": "rocm",
}

# PPU detection paths and environment variables
_PPU_SDK_PATHS = [Path("/usr/local/PPU_SDK"), Path("/opt/ppu")]
_PPU_ENV_VARS = ["PPU_SDK_PATH", "PPU_HOME"]


# ---------------------------------------------------------------------------
# Bazel config parsing
# ---------------------------------------------------------------------------

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
        "--config=ppu" -> ["--config=ppu"]
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
        str: Platform name (ppu, cuda12_6, rocm, etc.) or empty string
    """
    bazel_config = _get_bazel_config()
    for arg in bazel_config.split():
        if arg.startswith("--config="):
            config_value = arg.split("=", 1)[1]
            if config_value in PLATFORM_CONFIG_VERSIONS:
                return config_value
    return ""


# ---------------------------------------------------------------------------
# Filesystem platform detection
# ---------------------------------------------------------------------------

def _detect_ppu() -> bool:
    """Detect PPU SDK on the filesystem."""
    if any(p.exists() for p in _PPU_SDK_PATHS):
        return True
    if any(os.environ.get(var) for var in _PPU_ENV_VARS):
        return True
    return False


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


def _get_cuda_config_from_version(version_str: str) -> str:
    """Map CUDA version string to config name.

    Args:
        version_str: e.g. "12.9.20250531" or "12.6.0"

    Returns:
        Config name, e.g. "cuda12_6" or "cuda12_9"
    """
    if not version_str:
        return "cuda12_6"

    parts = version_str.split(".")
    if len(parts) >= 2:
        version_key = f"{parts[0]}.{parts[1]}"
        if version_key >= "12.9":
            return "cuda12_9"
        elif version_key >= "12.6":
            return "cuda12_6"
        else:
            return "cuda12_6"

    return "cuda12_6"


def _detect_cuda() -> bool:
    """Detect CUDA on the filesystem."""
    return Path("/usr/local/cuda").exists() or bool(shutil.which("nvcc"))


def _detect_rocm() -> bool:
    """Detect ROCm on the filesystem."""
    return Path("/opt/rocm").exists()


def _ensure_rocm_target_list() -> None:
    """Validate ROCm target list is gfx942 only."""
    target_list_path = Path("/opt/rocm/bin/target.lst")
    if not target_list_path.exists():
        raise RuntimeError(
            "ROCm target list not found: /opt/rocm/bin/target.lst. "
            "Please follow the ROCm pre-setup step to add gfx942."
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
            "ROCm target list mismatch in /opt/rocm/bin/target.lst: "
            f"expected only 'gfx942', got {targets}. "
            "Please follow the ROCm pre-setup step to set gfx942."
        )


# ---------------------------------------------------------------------------
# Main detection entry point
# ---------------------------------------------------------------------------

_cached_build_config: str | None = None


def detect_build_config(verbose: bool = True) -> str:
    """Detect appropriate build config based on environment.

    Checks (in order):
    1. RTP_BAZEL_CONFIG environment variable
    2. PPU SDK on filesystem
    3. CUDA toolkit (version.json -> cuda12_6 / cuda12_9 / cuda12_9_arm)
    4. ROCm directory

    Result is cached after first call.

    Raises:
        RuntimeError: if no platform can be detected.
    """
    global _cached_build_config
    if _cached_build_config is not None:
        return _cached_build_config

    # 1. Check RTP_BAZEL_CONFIG first
    build_config = extract_platform_from_config()
    if build_config:
        _cached_build_config = build_config
        return build_config

    # 2. Filesystem detection
    if _detect_ppu():
        if verbose:
            print("Detected PPU environment")
        _cached_build_config = "ppu"
        return "ppu"

    if _detect_cuda():
        cuda_version = _get_cuda_version_from_json()
        if cuda_version:
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

    raise RuntimeError(
        "Cannot detect build configuration. Please set RTP_BAZEL_CONFIG environment variable. "
        "Example: export RTP_BAZEL_CONFIG='--config=ppu' or '--config=cuda12_6 --config=sm9x'"
    )


def get_pip_extras() -> str:
    """Get pyproject.toml extras name for the detected platform.

    Returns extras name like: cuda12_9, cuda12, cuda12_arm, rocm, ppu
    """
    return CONFIG_TO_EXTRAS[detect_build_config(verbose=True)]


def should_skip_bazel_build() -> bool:
    """Check if Bazel build should be skipped (RTP_SKIP_BAZEL_BUILD is set)."""
    return bool(os.environ.get("RTP_SKIP_BAZEL_BUILD"))
