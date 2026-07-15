import json
import os
import shutil
import socket
import tarfile
import tempfile
import time
import uuid
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Iterator

SNAPSHOT_SUFFIX = ".jit_snapshot.tar.zst"
SNAPSHOT_KEEP = 100
REMOTE_READY_TIMEOUT_S = 120.0
MTIME_MANIFEST = ".jit_mtime_ns.json"


def _needs_exact_mtime(name: str) -> bool:
    # Ninja rebuilds on mtime; keep exact ns for the ninja caches' sources/objects.
    return name.startswith(("flashinfer/", "torch_extensions/")) and name.endswith(
        (".so", ".o", ".cu", ".cpp", ".inc", ".h")
    )


@contextmanager
def zstd_tar(path: Path, mode: str) -> Iterator[tarfile.TarFile]:
    import zstandard as zstd

    file_mode, tar_mode = {"r": ("rb", "r|"), "w": ("wb", "w|")}[mode]
    with zstd.open(path, file_mode) as body, tarfile.open(
        fileobj=body, mode=tar_mode
    ) as archive:
        yield archive


def pack_zstd_tar(archive: Path, source: Path) -> None:
    mtimes = {
        name: path.stat().st_mtime_ns
        for path in source.rglob("*")
        if path.is_file()
        and _needs_exact_mtime(name := path.relative_to(source).as_posix())
    }
    manifest = source / MTIME_MANIFEST
    manifest.write_text(json.dumps(mtimes), encoding="utf-8")
    try:
        with zstd_tar(archive, "w") as output:
            output.add(source, arcname="")
    finally:
        manifest.unlink(missing_ok=True)


def extract_zstd_tar(archive: Path, target: Path) -> None:
    with zstd_tar(archive, "r") as source_archive:
        source_archive.extractall(target)
    manifest = target / MTIME_MANIFEST
    if not manifest.exists():
        return
    mtimes = json.loads(manifest.read_text(encoding="utf-8"))
    manifest.unlink()
    for name, mtime_ns in mtimes.items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or not _needs_exact_mtime(name):
            raise ValueError(f"invalid mtime manifest path: {name}")
        os.utime(target / path, ns=(mtime_ns, mtime_ns), follow_symlinks=False)


class RemoteSnapshotStore:
    def __init__(self, remote_root: Path):
        self.remote_root = remote_root

    def _gc_stale_uploads(self, older_than_s: float = 3600.0) -> None:
        now = time.time()
        for tmp in self.remote_root.glob(".upload.*.tmp"):
            with suppress(OSError):
                if now - tmp.stat().st_mtime > older_than_s:
                    tmp.unlink()

    @staticmethod
    def _overlay_files(staging: Path, files: dict[str, Path]) -> None:
        for name, source in files.items():
            destination = staging / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

    @staticmethod
    def _wait_remote_ready(path: Path, expected_size: int) -> None:
        deadline = time.monotonic() + REMOTE_READY_TIMEOUT_S
        while time.monotonic() < deadline:
            with suppress(OSError):
                if path.stat().st_size == expected_size:
                    with path.open("rb", buffering=0) as source:
                        if source.read(1):
                            return
            time.sleep(0.05)
        raise TimeoutError(f"remote snapshot did not become ready: {path}")

    def restore(self, target: Path) -> bool:
        target.mkdir(parents=True, exist_ok=True)
        self._gc_stale_uploads()
        snapshots = sorted(self.remote_root.glob(f"*{SNAPSHOT_SUFFIX}"))
        if not snapshots:
            return False
        with tempfile.TemporaryDirectory(
            prefix=f"{target.name}.restore.", dir=target.parent
        ) as tmp_name:
            staging = Path(tmp_name)
            extract_zstd_tar(snapshots[-1], staging)
            shutil.copytree(staging, target, dirs_exist_ok=True)
        return True

    def _build_candidate(
        self, workspace: Path, base: Path | None, files: dict[str, Path]
    ) -> Path:
        staging = workspace / "staging"
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir()
        if base is not None:
            extract_zstd_tar(base, staging)
        self._overlay_files(staging, files)
        archive = workspace / "candidate.tar.zst"
        pack_zstd_tar(archive, staging)
        return archive

    def _prune(self, snapshots: list[Path]) -> None:
        for old in sorted(set(snapshots))[:-SNAPSHOT_KEEP]:
            with suppress(OSError):
                old.unlink()

    def publish_snapshot(self, files: dict[str, Path]) -> bool:
        if not files:
            return False

        snapshots = sorted(self.remote_root.glob(f"*{SNAPSHOT_SUFFIX}"))
        base = snapshots[-1] if snapshots else None
        with tempfile.TemporaryDirectory(prefix=".jit_snapshot.") as tmp_name:
            archive = self._build_candidate(Path(tmp_name), base, files)
            # FUSE-safe lock-free commit to a fresh target. Concurrent publishes
            # may miss each other's artifacts; they will recompile and republish.
            committed = self.remote_root / (
                f"{time.time_ns():020d}-{socket.gethostname()}{SNAPSHOT_SUFFIX}"
            )
            remote_tmp = self.remote_root / f".upload.{uuid.uuid4().hex}.tmp"
            try:
                shutil.copyfile(archive, remote_tmp)
                self._wait_remote_ready(remote_tmp, archive.stat().st_size)
                os.rename(remote_tmp, committed)
            finally:
                with suppress(OSError):
                    remote_tmp.unlink()
        self._prune(snapshots + [committed])
        return True
