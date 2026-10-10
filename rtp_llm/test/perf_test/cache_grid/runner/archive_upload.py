"""Archive a completed cache-grid result directory and upload it to OSS."""

import hashlib
import subprocess
import tarfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit


def validate_destination(destination: str) -> None:
    parsed = urlsplit(destination)
    if (
        parsed.scheme != "oss"
        or not parsed.netloc
        or not parsed.path.lstrip("/")
        or not parsed.path.endswith(".tar.gz")
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "--oss-destination must be a complete OSS object URI ending in .tar.gz"
        )


def create_archive(root: Path) -> dict:
    """Keep the archive beside root so it cannot include itself."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = root.parent / f"{root.name}-{timestamp}-{uuid.uuid4().hex[:8]}.tar.gz"
    try:
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(root, arcname=root.name)
    except BaseException:
        archive.unlink(missing_ok=True)
        raise
    digest = hashlib.sha256()
    with archive.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(archive),
        "bytes": archive.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def upload_archive(archive: Path, destination: str) -> None:
    result = subprocess.run(["ossutil", "cp", str(archive), destination], check=False)
    if result.returncode:
        raise RuntimeError(f"ossutil cp failed with exit code {result.returncode}")
