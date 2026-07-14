import json
import os
import shutil
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


@contextmanager
def zstd_tar(path: Path, mode: str) -> Iterator[tarfile.TarFile]:
    import zstandard as zstd

    file_mode, tar_mode = {"r": ("rb", "r|"), "w": ("wb", "w|")}[mode]
    with zstd.open(path, file_mode) as body, tarfile.open(
        fileobj=body, mode=tar_mode
    ) as archive:
        yield archive


def pack_zstd_tar(archive: Path, source: Path) -> None:
    with zstd_tar(archive, "w") as output:
        output.add(source, arcname="")


def extract_zstd_tar(archive: Path, target: Path) -> None:
    with zstd_tar(archive, "r") as source_archive:
        source_archive.extractall(target)


def _rebase_triton_paths(root: Path, restored_root: Path) -> None:
    for manifest in (root / "triton").rglob("__grp__*.json"):
        with suppress(OSError, json.JSONDecodeError):
            group = json.loads(manifest.read_text("utf-8"))
            paths = group.get("child_paths") if isinstance(group, dict) else None
            if not isinstance(paths, dict):
                continue
            restored_dir = restored_root / manifest.parent.relative_to(root)
            for name in paths:
                if name not in ("", ".", "..") and Path(name).name == name:
                    paths[name] = str(restored_dir / name)
            manifest.write_text(json.dumps(group), "utf-8")


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
        for snapshot in reversed(snapshots):
            with tempfile.TemporaryDirectory(
                prefix=f"{target.name}.restore.", dir=target.parent
            ) as tmp_name:
                staging = Path(tmp_name)
                try:
                    extract_zstd_tar(snapshot, staging)
                    _rebase_triton_paths(staging, target)
                except Exception:
                    if snapshot == snapshots[0]:
                        raise
                    continue
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
            # FUSE-safe lock-free commit to a unique target. Concurrent publishes
            # may miss each other's artifacts; they will recompile and republish.
            committed = self.remote_root / (
                f"{time.time_ns():020d}-{uuid.uuid4().hex}{SNAPSHOT_SUFFIX}"
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
