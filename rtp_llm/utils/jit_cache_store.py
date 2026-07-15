import hashlib
import json
import logging
import os
import re
import shutil
import socket
import tarfile
import tempfile
import time
import uuid
from contextlib import contextmanager, suppress
from pathlib import Path, PurePosixPath
from typing import Iterator

SNAPSHOT_SUFFIX = ".jit_snapshot.tar.zst"
SNAPSHOT_KEEP = 100
REMOTE_READY_TIMEOUT_S = 120.0
MTIME_MANIFEST = ".jit_mtime_ns.json"


class SourceFileChangedError(RuntimeError):
    pass


class SnapshotChecksumError(RuntimeError):
    pass


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
        for member in source_archive:
            relative = PurePosixPath(member.name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"invalid JIT snapshot path: {member.name}")
            parts = tuple(part for part in relative.parts if part != ".")
            if not parts:
                continue
            destination = target.joinpath(*parts)
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise ValueError(
                    f"unsupported JIT snapshot member: {member.name} ({member.type!r})"
                )
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = source_archive.extractfile(member)
            if source is None:
                raise ValueError(f"missing JIT snapshot payload: {member.name}")
            with source, destination.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
            if destination.stat().st_size != member.size:
                destination.unlink(missing_ok=True)
                raise ValueError(f"truncated JIT snapshot member: {member.name}")
            os.chmod(destination, member.mode)
            os.utime(destination, (member.mtime, member.mtime))
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
    def _copy_stable_file(source: Path, destination: Path) -> None:
        """Copy one closed artifact and reject concurrent in-place writes."""
        with source.open("rb", buffering=0) as source_file:
            before = os.fstat(source_file.fileno())
            with destination.open("wb", buffering=0) as output:
                shutil.copyfileobj(source_file, output, length=1024 * 1024)
            after = os.fstat(source_file.fileno())

        identity_before = (before.st_size, before.st_mtime_ns)
        identity_after = (after.st_size, after.st_mtime_ns)
        if (
            identity_before != identity_after
            or destination.stat().st_size != before.st_size
        ):
            destination.unlink(missing_ok=True)
            raise SourceFileChangedError(
                f"JIT artifact changed while copying: {source}"
            )
        os.chmod(destination, before.st_mode)
        os.utime(
            destination,
            ns=(before.st_atime_ns, before.st_mtime_ns),
            follow_symlinks=False,
        )

    @classmethod
    def _overlay_files(cls, staging: Path, files: dict[str, Path]) -> None:
        for name, source in files.items():
            destination = staging / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            cls._copy_stable_file(source, destination)

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @classmethod
    def _verify_snapshot(cls, path: Path) -> None:
        stem = path.name[: -len(SNAPSHOT_SUFFIX)]
        checksum_part = stem.rsplit("-", 1)[-1]
        if not checksum_part.startswith("sha256_"):
            return  # Backward compatibility for snapshots without embedded hashes.
        expected = checksum_part.removeprefix("sha256_")
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise SnapshotChecksumError(f"invalid JIT snapshot checksum name: {path}")
        actual = cls._sha256(path)
        if actual != expected:
            raise SnapshotChecksumError(
                f"JIT snapshot checksum mismatch: {path} expected={expected} "
                f"actual={actual}"
            )

    @classmethod
    def _wait_remote_ready(
        cls, path: Path, expected_size: int, expected_sha256: str
    ) -> None:
        deadline = time.monotonic() + REMOTE_READY_TIMEOUT_S
        while time.monotonic() < deadline:
            with suppress(OSError):
                if (
                    path.stat().st_size == expected_size
                    and cls._sha256(path) == expected_sha256
                ):
                    return
            time.sleep(0.05)
        raise TimeoutError(f"remote snapshot did not become ready: {path}")

    def _overlay_snapshot(self, snapshot: Path, staging: Path) -> None:
        self._verify_snapshot(snapshot)
        with tempfile.TemporaryDirectory(
            prefix=".jit_snapshot_input.", dir=staging.parent
        ) as tmp_name:
            extracted = Path(tmp_name)
            extract_zstd_tar(snapshot, extracted)
            shutil.copytree(extracted, staging, dirs_exist_ok=True)

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
            # Snapshots are immutable. Overlay every retained branch so concurrent
            # publishers cannot hide each other's artifacts.
            restored = 0
            for snapshot in snapshots:
                try:
                    self._overlay_snapshot(snapshot, staging)
                    restored += 1
                except Exception:
                    logging.warning(
                        "skipping invalid JIT snapshot during restore: %s",
                        snapshot,
                        exc_info=True,
                    )
            if not restored:
                return False
            shutil.copytree(staging, target, dirs_exist_ok=True)
        return True

    def _build_candidate(
        self, workspace: Path, bases: list[Path], files: dict[str, Path]
    ) -> Path:
        staging = workspace / "staging"
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir()
        for base in bases:
            try:
                self._overlay_snapshot(base, staging)
            except Exception:
                logging.warning(
                    "skipping invalid JIT snapshot during publish: %s",
                    base,
                    exc_info=True,
                )
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
        with tempfile.TemporaryDirectory(prefix=".jit_snapshot.") as tmp_name:
            archive = self._build_candidate(Path(tmp_name), snapshots, files)
            archive_sha256 = self._sha256(archive)
            # FUSE-safe lock-free commit to a fresh target. Each publisher merges
            # every branch it can see; unseen concurrent branches remain immutable
            # and are overlaid by restore until a later publish consolidates them.
            committed = self.remote_root / (
                f"{time.time_ns():020d}-{socket.gethostname()}-sha256_{archive_sha256}"
                f"{SNAPSHOT_SUFFIX}"
            )
            remote_tmp = self.remote_root / f".upload.{uuid.uuid4().hex}.tmp"
            try:
                shutil.copyfile(archive, remote_tmp)
                self._wait_remote_ready(
                    remote_tmp, archive.stat().st_size, archive_sha256
                )
                os.rename(remote_tmp, committed)
            finally:
                with suppress(OSError):
                    remote_tmp.unlink()
        self._prune(snapshots + [committed])
        return True
