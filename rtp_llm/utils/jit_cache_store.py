import fcntl
import io
import json
import logging
import os
import shutil
import tarfile
import tempfile
import threading
import time
from collections import namedtuple
from contextlib import suppress
from pathlib import Path
from urllib.parse import urlparse

import zstandard as zstd

SNAPSHOT_SUFFIX, MTIME_MANIFEST = ".jit_snapshot.tar.zst", ".jit_mtime_ns.json"
SNAPSHOT_KEEP, STALE_REMOTE_TMP_S, STALE_BATON_S = 20, 1800.0, 7200.0


class SnapshotRaced(RuntimeError):
    """A tracked file changed while packing; defer the generation."""


def _file_sig(path: Path) -> tuple[int, int, int, int]:
    st = path.stat()
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)


def pack_zstd_tar(archive: Path, files: dict[str, Path]) -> None:
    before = {name: _file_sig(path) for name, path in files.items()}
    with zstd.open(
        archive, "wb", cctx=zstd.ZstdCompressor(write_checksum=True)
    ) as body, tarfile.open(fileobj=body, mode="w|", dereference=True) as tar:
        for name, path in sorted(files.items()):
            tar.add(path, arcname=name, recursive=False)
        if any(_file_sig(files[name]) != sig for name, sig in before.items()):
            raise SnapshotRaced("files changed while packing")
        manifest = json.dumps({name: sig[3] for name, sig in before.items()}).encode()
        info = tarfile.TarInfo(MTIME_MANIFEST)
        info.size = len(manifest)
        tar.addfile(info, io.BytesIO(manifest))


def _safe_path(root: Path, name: str) -> Path:
    if root not in (path := (root / name).resolve()).parents:
        raise ValueError(f"unsafe JIT snapshot path: {name}")
    return path


def _safe_members(archive, target: Path):
    for member in archive:
        if not (member.isfile() or member.isdir()):
            raise ValueError(f"unsafe JIT snapshot member: {member.name}")
        _safe_path(target, member.name)
        yield member


def extract_zstd_tar(archive: Path, target: Path) -> None:
    target = target.resolve()  # _safe_path compares against a resolved root
    with zstd.open(archive, "rb") as body, tarfile.open(fileobj=body, mode="r|") as tar:
        kwargs = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
        tar.extractall(target, members=_safe_members(tar, target), **kwargs)
    mtimes = json.loads((manifest := target / MTIME_MANIFEST).read_text())
    manifest.unlink()
    for name, mtime_ns in mtimes.items():
        os.utime(_safe_path(target, name), ns=(mtime_ns, mtime_ns))
    for path in (target, *target.rglob("*")):  # the tar filter drops shared write bits
        with suppress(OSError):
            if not path.is_symlink():  # Linux has no lchmod: chmod would follow it
                path.chmod(0o777 if path.is_dir() else 0o666)


def acquire_flock(path: Path, blocking: bool = True) -> int | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o666)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        return fd
    except BaseException as error:
        os.close(fd)
        if isinstance(error, BlockingIOError):
            return None
        raise


def reap_stale_batons(root: Path) -> None:
    cutoff = time.time() - STALE_BATON_S
    for path in (p for name in ("lock", "lock_*") for p in root.rglob(name)):
        with suppress(OSError):
            if path.stat().st_mtime < cutoff:
                path.unlink()
                logging.warning("reaped stale baton %s", path)


def scope_root_usable(root: Path) -> bool:
    with suppress(OSError):
        return any(path.is_file() for path in root.rglob("*"))
    return False


def commit_restore(staging: Path, scope_root: Path) -> bool:
    with suppress(OSError):
        for path in (*sorted(scope_root.rglob("*"), reverse=True), scope_root):
            with suppress(OSError):
                path.rmdir()
    try:
        os.rename(staging, scope_root)
        return True
    except OSError:
        logging.warning("JIT restore skipped; local tree already in use")
        shutil.rmtree(staging, ignore_errors=True)
        return False


Restored = namedtuple("Restored", "staging snapshot")


class RemoteSnapshotStore:
    def __init__(self, remote_root: Path, mounted: str = ""):
        self.remote_root, self._mounted = remote_root, mounted
        self._mount_lock, self._closed = threading.Lock(), False

    def close(self):
        with self._mount_lock:
            self._closed = True
            if self._mounted:
                try:
                    from rtp_llm.utils.fuser import umount_file

                    umount_file(self._mounted)
                    self._mounted = ""
                except Exception:
                    logging.warning("JIT unmount failed", exc_info=True)

    def prepare_restore(self, staging_root: Path) -> Restored | None:
        staging_root.mkdir(parents=True, exist_ok=True)
        for snapshot in reversed(sorted(self.remote_root.glob(f"*{SNAPSHOT_SUFFIX}"))):
            staging = Path(tempfile.mkdtemp(prefix="stage.", dir=staging_root))
            try:
                extract_zstd_tar(snapshot, staging)
                return Restored(staging, snapshot)
            except Exception:
                logging.warning("JIT snapshot unusable: %s", snapshot)
                shutil.rmtree(staging, ignore_errors=True)

    def publish_snapshot(self, files: dict[str, Path]):
        # close() unmounts: a cut copy truncates, a later one hits a bare mountpoint.
        with self._mount_lock:
            if self._closed or not files:
                return
            with tempfile.TemporaryDirectory(prefix=".jit_snapshot.") as tmp:
                archive = Path(tmp) / "candidate.tar.zst"
                pack_zstd_tar(archive, files)
                name = f"{time.time_ns():020d}-{os.uname().nodename}{SNAPSHOT_SUFFIX}"
                remote_tmp = self.remote_root / f"{name}.tmp"
                try:
                    shutil.copyfile(archive, remote_tmp)
                    os.rename(remote_tmp, self.remote_root / name)
                finally:
                    with suppress(OSError):
                        remote_tmp.unlink()
                cutoff = time.time() - STALE_REMOTE_TMP_S
                old = sorted(self.remote_root.glob(f"*{SNAPSHOT_SUFFIX}"))
                stale = self.remote_root.glob(f"*{SNAPSHOT_SUFFIX}.tmp")
                for path in (*old[:-SNAPSHOT_KEEP], *stale):
                    with suppress(OSError):
                        if path.suffix != ".tmp" or path.stat().st_mtime < cutoff:
                            path.unlink()
                logging.info("JIT published %s: %d bytes", name, archive.stat().st_size)


def resolve_remote(value, version: str, scope_id: str):
    text, mounted = str(value or "").strip(), ""
    try:
        if urlparse(text).scheme:
            from rtp_llm.utils.fuser import MountRwMode
            from rtp_llm.utils.fuser import fetch_remote_file_to_local as fetch

            text = mounted = fetch(text, MountRwMode.RWMODE_RW, True)
        root = Path(text).expanduser()
        if not text or not root.is_absolute() or not root.is_dir():
            raise OSError(f"invalid remote directory {text}")
        # umask is 0: co-tenants can publish into the same version/scope dir
        (root := root / version / scope_id).mkdir(parents=True, exist_ok=True)
        return RemoteSnapshotStore(root, mounted)
    except Exception:
        logging.warning("JIT_CACHE_FAIL_OPEN: remote unavailable", exc_info=True)
        RemoteSnapshotStore(Path(), mounted).close()
