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
from contextlib import ExitStack, suppress
from pathlib import Path
from urllib.parse import urlparse

try:
    import zstandard as zstd
except ImportError:
    zstd = None

SNAPSHOT_SUFFIX = (
    ".jit_snapshot.tar.zst" if zstd is not None else ".jit_snapshot.tar.gz"
)
MTIME_MANIFEST = ".jit_mtime_ns.json"
SNAPSHOT_KEEP, STALE_REMOTE_TMP_S, STALE_BATON_S = 20, 1800.0, 7200.0
Restored = namedtuple("Restored", "staging snapshot")


class SnapshotRaced(RuntimeError): ...


def file_sig(path: Path) -> tuple[int, int, int, int]:
    st = path.stat()
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)


def _add_snapshot_members(tar, files, before) -> None:
    for name, path in sorted(files.items()):
        tar.add(path, arcname=name, recursive=False)
    if any(file_sig(files[name]) != sig for name, sig in before.items()):
        raise SnapshotRaced("files changed while packing")
    manifest = json.dumps({n: s[3] for n, s in before.items()}).encode()
    info = tarfile.TarInfo(MTIME_MANIFEST)
    info.size = len(manifest)
    tar.addfile(info, io.BytesIO(manifest))


def pack_snapshot(archive: Path, files: dict[str, Path]) -> None:
    try:
        before = {name: file_sig(path) for name, path in files.items()}
        if zstd is not None:
            with zstd.open(
                archive, "wb", cctx=zstd.ZstdCompressor(write_checksum=True)
            ) as body, tarfile.open(
                fileobj=body, mode="w|", dereference=True
            ) as tar:
                _add_snapshot_members(tar, files, before)
        else:
            with tarfile.open(archive, mode="w:gz", dereference=True) as tar:
                _add_snapshot_members(tar, files, before)
    except FileNotFoundError as error:
        raise SnapshotRaced("files changed while packing") from error


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


def extract_snapshot(archive: Path, target: Path) -> None:
    target = target.resolve()
    with ExitStack() as stack:
        if zstd is not None:
            body = stack.enter_context(zstd.open(archive, "rb"))
            tar = stack.enter_context(tarfile.open(fileobj=body, mode="r|"))
        else:
            tar = stack.enter_context(tarfile.open(archive, mode="r:gz"))
        kwargs = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
        tar.extractall(target, members=_safe_members(tar, target), **kwargs)
    for name, ns in json.loads((target / MTIME_MANIFEST).read_text()).items():
        os.utime(_safe_path(target, name), ns=(ns, ns))
    (target / MTIME_MANIFEST).unlink()
    for path in (target, *target.rglob("*")):
        with suppress(OSError):
            if not path.is_symlink():
                path.chmod(0o777 if path.is_dir() else 0o666)


def acquire_flock(path: Path, blocking: bool = True, create: bool = True) -> int | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_NOFOLLOW | (os.O_CREAT if create else 0), 0o666)
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
    for path in (p for g in ("lock", "lock_*") for p in root.rglob(g)):
        with suppress(OSError):
            if (fd := acquire_flock(path, False, create=False)) is not None:
                with os.fdopen(fd, "rb") as baton:
                    old, current = os.fstat(baton.fileno()), path.stat()
                    if old.st_mtime < cutoff and os.path.samestat(old, current):
                        path.unlink()


def scope_root_usable(root: Path) -> bool:
    with suppress(OSError):
        return any(path.is_file() for path in root.rglob("*"))
    return False


def commit_restore(staging: Path, scope_root: Path) -> bool:
    try:
        if scope_root_usable(scope_root):
            raise FileExistsError(scope_root)
        with suppress(OSError):
            for path in (*sorted(scope_root.rglob("*"), reverse=True), scope_root):
                path.rmdir()
        os.rename(staging, scope_root)
        return True
    except OSError as error:
        logging.warning("JIT restore skipped; local tree unchanged: %s", error)
    finally:
        shutil.rmtree(staging, ignore_errors=True)  # a committed staging is gone
    return False


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
        for snap in sorted(self.remote_root.glob(f"*{SNAPSHOT_SUFFIX}"), reverse=True):
            staging = Path(tempfile.mkdtemp(prefix="stage.", dir=staging_root))
            try:
                extract_snapshot(snap, staging)
                return Restored(staging, snap)
            except Exception:
                logging.warning("JIT snapshot unusable: %s", snap)
                shutil.rmtree(staging, ignore_errors=True)

    def publish_snapshot(self, files: dict[str, Path], rescan=None):
        with self._mount_lock:
            if self._closed or not files:
                return
            with tempfile.TemporaryDirectory(prefix=".jit_snapshot.") as tmp:
                archive = Path(tmp) / f"candidate{SNAPSHOT_SUFFIX}"
                pack_snapshot(archive, files)
                if rescan and files.keys() != rescan().keys():
                    raise SnapshotRaced("file set changed while packing")
                name = f"{time.time_ns():020d}-{os.uname().nodename}{SNAPSHOT_SUFFIX}"
                remote_tmp = self.remote_root / f"{name}.tmp"
                try:
                    shutil.copyfile(archive, remote_tmp)
                    remote_tmp.chmod(0o644)
                    os.rename(remote_tmp, self.remote_root / name)
                finally:
                    with suppress(OSError):
                        remote_tmp.unlink()
                cutoff, keep = time.time() - STALE_REMOTE_TMP_S, -SNAPSHOT_KEEP
                old = sorted(self.remote_root.glob(f"*{SNAPSHOT_SUFFIX}"))[:keep]
                for path in (*old, *self.remote_root.glob(f"*{SNAPSHOT_SUFFIX}.tmp")):
                    with suppress(OSError):
                        if path.suffix != ".tmp" or path.stat().st_mtime < cutoff:
                            path.unlink()


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
        rwx = os.R_OK | os.W_OK | os.X_OK
        scope_root = root / version / scope_id
        for path in (scope_root.parent, scope_root):
            with suppress(OSError):  # trusted remote; FUSE may ignore chmod
                path.mkdir()
                path.chmod(0o777)
            if not path.is_dir() or path.is_symlink() or not os.access(path, rwx):
                raise OSError(f"unusable remote directory {path}")
        return RemoteSnapshotStore(scope_root, mounted)
    except Exception:
        logging.warning("JIT_CACHE_FAIL_OPEN: remote unavailable", exc_info=True)
        RemoteSnapshotStore(Path(), mounted).close()
        return None
