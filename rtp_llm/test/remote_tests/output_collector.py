"""Collect and download remote worker test outputs.

Implements the Bazel TEST_UNDECLARED_OUTPUTS_DIR pattern for the REAPI
pytest remote plugin.  On the remote worker, test code writes artifacts
(server logs, smoke_actual, OOM state, coredump summaries) to the
directory pointed to by TEST_UNDECLARED_OUTPUTS_DIR.  After the test
command finishes, a shell postscript tars that directory into a single
file declared as a Command.output_file.  The local plugin then downloads
the tar from CAS and extracts it.

This module is deliberately free of pytest imports so it can be unit-tested
independently.
"""
from __future__ import annotations

import logging
import os
import tarfile
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from . import remote_execution_pb2 as re_pb2
    from .cas_client import CASClient
    from .executor import ExecutionResult

log = logging.getLogger(__name__)


def _safe_extract(tar: tarfile.TarFile, dest: Path) -> None:
    """Extract tar members safely, rejecting path traversal attempts."""
    for member in tar.getmembers():
        member_path = os.path.normpath(member.name)
        if member_path.startswith("..") or os.path.isabs(member_path):
            log.warning("Skipping unsafe tar member: %s", member.name)
            continue
        # Resolve to ensure no symlink tricks
        target = (dest / member_path).resolve()
        if not str(target).startswith(str(dest.resolve())):
            log.warning("Skipping tar member escaping dest: %s", member.name)
            continue
        tar.extract(member, dest)

# Bazel-compatible env var name — existing code already writes to this.
OUTPUTS_ENV_VAR = "TEST_UNDECLARED_OUTPUTS_DIR"

# Directory name on the remote worker (relative to sandbox CWD).
WORKER_OUTPUTS_DIR = "_rtp_test_outputs"

# Archive name declared in Command.output_files.
OUTPUTS_ARCHIVE = "_rtp_test_outputs.tar.gz"


def make_output_collection_env() -> dict:
    """Env vars to inject into the remote Command.

    NOTE: This is intentionally empty.  REAPI Command.environment_variables
    are passed as literal strings (no shell expansion), so ``$PWD`` would
    not be resolved.  Instead, the env var is set via the shell prefix
    returned by :func:`make_mkdir_prefix`.
    """
    return {}


def make_output_files_decl() -> list:
    """output_files entries to declare in the Command proto."""
    return [OUTPUTS_ARCHIVE]


def make_tar_postscript() -> str:
    """Shell snippet appended after pytest.  Runs regardless of exit code.

    Tars the entire working directory, excluding only coredump files.
    """
    return (
        f"dmesg -T 2>/dev/null | tail -200 > {WORKER_OUTPUTS_DIR}/dmesg.log 2>/dev/null; "
        f"tar -czf {OUTPUTS_ARCHIVE} --exclude='core.*' --exclude='core' "
        f"{WORKER_OUTPUTS_DIR} logs 2>/dev/null; "
        f'echo ">>>RTP_OUTPUTS_ARCHIVE size=$(stat -c%s {OUTPUTS_ARCHIVE} 2>/dev/null || echo 0)"'
    )


def make_mkdir_prefix() -> str:
    """Shell snippet prepended before pytest to create the outputs dir.

    Also exports TEST_UNDECLARED_OUTPUTS_DIR so that existing test code
    (smoke comparers, server manager, OOM hooks) writes artifacts there.
    This must happen in the shell (not REAPI env vars) because we need
    ``$PWD`` to be resolved by bash.
    """
    return (
        f"mkdir -p {WORKER_OUTPUTS_DIR}; "
        f"export {OUTPUTS_ENV_VAR}=$PWD/{WORKER_OUTPUTS_DIR}; "
    )


def download_and_extract(
    cas: "CASClient",
    result: "ExecutionResult",
    local_dest: Path,
    *,
    max_bytes: int = 0,
) -> Optional[Path]:
    """Download the outputs archive from CAS and extract to *local_dest*.

    Args:
        max_bytes: If > 0, skip download when the archive exceeds this size.

    Returns the local directory on success, ``None`` if no archive was
    produced.
    """
    digest = result.output_files.get(OUTPUTS_ARCHIVE)
    if digest is None:
        log.debug("No %s in ActionResult — remote produced no outputs", OUTPUTS_ARCHIVE)
        return None

    if max_bytes and digest.size_bytes > max_bytes:
        log.warning(
            "Remote outputs archive too large (%.1f MiB > %.1f MiB limit), skipping download",
            digest.size_bytes / (1024 * 1024),
            max_bytes / (1024 * 1024),
        )
        return None

    log.info(
        "Downloading remote outputs (%.1f MiB) digest=%s",
        digest.size_bytes / (1024 * 1024),
        digest.hash[:12],
    )

    try:
        data = cas.download_blob(digest)
        if not data:
            log.warning("download_blob returned empty for %s", digest.hash[:12])
            return None
    except Exception as exc:
        log.warning("Failed to download remote outputs: %s", exc)
        return None

    local_dest.mkdir(parents=True, exist_ok=True)
    tmp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tf:
            tf.write(data)
            tmp_path = Path(tf.name)
        with tarfile.open(tmp_path, "r:gz") as tar:
            _safe_extract(tar, local_dest)
        log.info("Remote outputs extracted to %s", local_dest)
        return local_dest
    except Exception as exc:
        log.warning("Failed to extract remote outputs: %s", exc)
        return None
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
