import fcntl
import io
import json
import logging
import os
import re
import shutil
import socket
import tarfile
import tempfile
import threading
import time
from contextlib import contextmanager, nullcontext, suppress
from pathlib import Path

import zstandard as zstd

SNAPSHOT_SUFFIX = ".jit_snapshot.tar.zst"
# Enough headroom so GC never unlinks a snapshot an in-flight restore is reading.
SNAPSHOT_KEEP = 20
REMOTE_READY_TIMEOUT_S = 120.0
# Age gate: GC never unlinks a file a live upload/restore may still be touching.
IDLE_REAP_S = REMOTE_READY_TIMEOUT_S * 10
STALE_BATON_S = 1800.0
# tarfile only preserves whole-second mtime; ninja compares nanoseconds.
MTIME_MANIFEST = ".jit_mtime_ns.json"


def sanitize(text: str, fallback: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", text).strip("_") or fallback


def pack_zstd_tar(archive: Path, source: Path) -> None:
    # Symlinks are skipped; _safe_members rejects them on restore anyway. dereference
    # packs hardlinked members as full content, never as tar hardlink references.
    mtimes = {}
    with zstd.open(archive, "wb") as body, tarfile.open(
        fileobj=body, mode="w|", dereference=True
    ) as tar:

        def keep(info: tarfile.TarInfo):
            path = source / info.name
            if path.is_symlink() or path.name == MTIME_MANIFEST:
                return None
            if info.isfile():
                mtimes[info.name] = path.stat().st_mtime_ns
            return info

        tar.add(source, arcname=".", filter=keep)
        manifest = json.dumps(mtimes).encode()
        info = tarfile.TarInfo(MTIME_MANIFEST)
        info.size = len(manifest)
        tar.addfile(info, io.BytesIO(manifest))


def _safe_path(root: Path, name: str) -> Path:
    root, path = root.resolve(), (root / name).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"unsafe JIT snapshot path: {name}")
    return path


def _safe_members(archive, target: Path):
    for member in archive:
        if not (member.isfile() or member.isdir()):
            raise ValueError(f"unsafe JIT snapshot member: {member.name}")
        _safe_path(target, member.name)
        yield member


def extract_zstd_tar(archive: Path, target: Path) -> None:
    with zstd.open(archive, "rb") as body, tarfile.open(
        fileobj=body, mode="r|"
    ) as source:
        source.extractall(target, members=_safe_members(source, target))
    manifest = target / MTIME_MANIFEST
    if manifest.exists():
        mtimes = json.loads(manifest.read_text())
        manifest.unlink()
        for name, mtime_ns in mtimes.items():
            os.utime(_safe_path(target, name), ns=(mtime_ns, mtime_ns))


@contextmanager
def restore_lock(target: Path):
    # Pure mutual exclusion: the kernel frees the flock on fd close or process
    # death, so a crash never wedges it. Claiming the tree (.ready) is separate.
    target.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(target.with_name(f"{target.name}.lock"), os.O_CREAT | os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def is_lock_file(name: str) -> bool:
    # torch's FileBaton uses a bare "lock"; our builders use *_lock / *.lock.
    return name == "lock" or name.endswith(("_lock", ".lock"))


def reap_dead_lock(path: Path) -> None:
    # Unlink a lock no live builder holds: a non-blocking flock grab fails iff a
    # builder still holds it. torch's bare "lock" baton is never flocked so it
    # always grabs free; there fall back to mtime and spare a fresh (maybe-live) one.
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return
        with suppress(OSError):
            if (
                path.name == "lock"
                and time.time() - path.stat().st_mtime < STALE_BATON_S
            ):
                return
            path.unlink()
    finally:
        os.close(fd)


def _tree_is_warm(root: Path) -> bool:
    # Warm = holds a real artifact (not a lock corpse or the mtime manifest). Restore
    # must skip a warm tree; artifact-free leftovers (lock corpses, empty scope dirs)
    # are clobberable so a killed cold start can retry.
    with suppress(OSError):
        return any(
            name != MTIME_MANIFEST and not is_lock_file(name)
            for _, _, files in os.walk(root)
            for name in files
        )
    return False


def _signature(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


class RemoteSnapshotStore:
    def __init__(self, remote_root: Path):
        self.remote_root = remote_root

    def _snapshots(self) -> list[Path]:
        return sorted(self.remote_root.glob(f"*{SNAPSHOT_SUFFIX}"))

    @staticmethod
    def _wait_remote_ready(path: Path, source: Path) -> None:
        # Some FUSE mounts expose st_size before the tail is readable.
        def tail(file: Path, offset: int) -> bytes:
            with file.open("rb", buffering=0) as handle:
                handle.seek(offset)
                return handle.read()

        size = source.stat().st_size
        offset = max(0, size - 4096)
        expected = tail(source, offset)
        deadline, delay = time.monotonic() + REMOTE_READY_TIMEOUT_S, 0.05
        while time.monotonic() < deadline:
            with suppress(OSError):
                if path.stat().st_size == size and tail(path, offset) == expected:
                    return
            time.sleep(delay)
            delay = min(delay * 2, 2.0)
        raise TimeoutError(f"remote snapshot did not become ready: {path}")

    def restore(
        self,
        target: Path,
        cancel: threading.Event | None = None,
        commit=None,
    ) -> bool:
        # commit: context manager entered around every tree swap/claim so the
        # caller's adopt-vs-abandon decision is atomic; providing it means the
        # caller already holds restore_lock(target).
        if commit is None:
            with restore_lock(target):
                return self.restore(target, cancel, nullcontext())
        ready = target.with_name(f"{target.name}.ready")
        if ready.exists() or (cancel and cancel.is_set()):
            return False
        if _tree_is_warm(target):
            ready.touch()  # claim: built by a peer that never ran restore
            return False

        snapshots = self._snapshots()
        staging = target.with_name(f"{target.name}.stage.{time.time_ns()}")
        try:
            for snapshot in reversed(snapshots):
                shutil.rmtree(staging, ignore_errors=True)
                staging.mkdir(parents=True)
                try:
                    extract_zstd_tar(snapshot, staging)
                except Exception:
                    logging.warning("JIT snapshot unusable: %s", snapshot)
                    continue
                with commit:
                    if cancel and cancel.is_set():
                        return False
                    shutil.rmtree(target, ignore_errors=True)
                    os.rename(staging, target)
                    ready.touch()
                return True
            if not snapshots:
                with commit:
                    if not (cancel and cancel.is_set()):
                        ready.touch()  # empty remote: claim, cold build fills it
            # all snapshots unusable: leave unclaimed so a later start retries
            return False
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def publish_snapshot(self, scan) -> bool:
        # Each archive is one complete local generation, never an old-snapshot overlay.
        files = scan()
        if not files:
            return True
        with tempfile.TemporaryDirectory(prefix=".jit_snapshot.") as tmp:
            staging = Path(tmp) / "staging"
            staging.mkdir()
            signatures = {name: _signature(path) for name, path in files.items()}
            # Hardlink into staging instead of copying (cheap, no data move);
            # pack_zstd_tar dereferences, so even shared inodes pack as full
            # content, never hardlink members. Fall back to copy2 across devices.
            for name, source in sorted(files.items()):
                destination = _safe_path(staging, name)
                destination.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(source, destination)
                except OSError:
                    shutil.copy2(source, destination)

            archive = Path(tmp) / "candidate.tar.zst"
            pack_zstd_tar(archive, staging)
            # Staging aliases the live tree: only a post-pack rescan proves the
            # packed generation stayed self-consistent. A change is a benign
            # race with a live build; the caller retries after the next quiet period.
            try:
                changed = {
                    name: _signature(path) for name, path in scan().items()
                } != signatures
            except OSError:
                changed = True
            if changed:
                logging.warning("JIT cache changed during snapshot; deferring publish")
                return False
            host = sanitize(socket.gethostname(), "host")
            committed = self.remote_root / (
                f"{time.time_ns():020d}-{host}{SNAPSHOT_SUFFIX}"
            )
            remote_tmp = committed.with_name(f"{committed.name}.tmp")
            try:
                shutil.copyfile(archive, remote_tmp)
                self._wait_remote_ready(remote_tmp, archive)
                os.rename(remote_tmp, committed)
            finally:
                with suppress(OSError):
                    remote_tmp.unlink()

        # Age gate: unlink-while-open is unsafe on OSS/FUSE; backlog <= publish rate * IDLE_REAP_S.
        cutoff = time.time() - IDLE_REAP_S
        stale = self._snapshots()[:-SNAPSHOT_KEEP]
        stale.extend(self.remote_root.glob(f"*{SNAPSHOT_SUFFIX}.tmp"))
        for path in stale:
            with suppress(OSError):
                if path.stat().st_mtime < cutoff:
                    path.unlink()
        return True
