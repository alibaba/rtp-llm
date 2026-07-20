import json
import logging
import os
import shutil
import tarfile
import tempfile
import time
import uuid
from contextlib import suppress
from pathlib import Path

import zstandard as zstd

SNAPSHOT_SUFFIX = ".jit_snapshot.tar.zst"
# Keep enough recent snapshots that GC never unlinks one an in-flight restore is
# still reading (OSS/FUSE unlink-while-open is unsafe).
SNAPSHOT_KEEP = 20
REMOTE_READY_TIMEOUT_S = 120.0
# Idle so far past REMOTE_READY_TIMEOUT_S that no live upload/restore can be
# touching it, so GC (orphan .tmp / over-keep snapshot) never races a live op.
IDLE_REAP_S = REMOTE_READY_TIMEOUT_S * 10
# tarfile keeps only whole-second mtime but ninja's restat compares ns; persist
# ns here and restore via os.utime so a warm cache isn't seen as stale.
MTIME_MANIFEST = ".jit_mtime_ns.json"


def pack_zstd_tar(archive: Path, source: Path) -> None:
    mtimes = {
        path.relative_to(source).as_posix(): path.stat().st_mtime_ns
        for path in source.rglob("*")
        if path.is_file()
    }
    manifest = source / MTIME_MANIFEST
    manifest.write_text(json.dumps(mtimes), encoding="utf-8")
    try:
        with zstd.open(archive, "wb") as body, tarfile.open(
            fileobj=body, mode="w|"
        ) as output:
            output.add(source, arcname=".")
    finally:
        manifest.unlink(missing_ok=True)


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
    ) as source_archive:
        source_archive.extractall(target, members=_safe_members(source_archive, target))
    manifest = target / MTIME_MANIFEST
    if not manifest.exists():
        return
    mtimes = json.loads(manifest.read_text(encoding="utf-8"))
    manifest.unlink()
    for name, mtime_ns in mtimes.items():
        os.utime(
            _safe_path(target, name), ns=(mtime_ns, mtime_ns), follow_symlinks=False
        )


def _reap_idle(paths, cutoff: float) -> None:
    # Skip files newer than the cutoff: they may still belong to a live op.
    for path in paths:
        with suppress(OSError):
            if path.stat().st_mtime < cutoff:
                path.unlink()


class RemoteSnapshotStore:
    def __init__(self, remote_root: Path):
        self.remote_root = remote_root

    @staticmethod
    def _wait_remote_ready(path: Path, source: Path) -> None:
        # OSS/FUSE may expose the final st_size before the tail is flushed; match
        # the trailing bytes (not just size) so we never commit a truncated tar.
        expected_size = source.stat().st_size
        tail_len = min(expected_size, 4096) or 1
        with source.open("rb", buffering=0) as local:
            local.seek(max(0, expected_size - tail_len))
            expected_tail = local.read()
        deadline = time.monotonic() + REMOTE_READY_TIMEOUT_S
        while time.monotonic() < deadline:
            with suppress(OSError):
                if path.stat().st_size == expected_size:
                    with path.open("rb", buffering=0) as remote:
                        remote.seek(max(0, expected_size - tail_len))
                        if remote.read() == expected_tail:
                            return
            time.sleep(0.05)
        raise TimeoutError(f"remote snapshot did not become ready: {path}")

    def restore(self, target: Path) -> bool:
        # Read only the newest usable snapshot — no cross-snapshot merge, by design.
        # Concurrent publishers forking the same baseline (S0+A, S0+B) mean the
        # newest may omit the other's delta; accepted, since this is only a hit-rate
        # cache: the missing artifact is recompiled on demand and republished, so
        # snapshots converge. Eventual consistency keeps restore O(1) and lock-free.
        target.mkdir(parents=True, exist_ok=True)
        for snapshot in sorted(
            self.remote_root.glob(f"*{SNAPSHOT_SUFFIX}"), reverse=True
        ):
            with tempfile.TemporaryDirectory(dir=target.parent) as tmp:
                staging = Path(tmp)
                try:
                    extract_zstd_tar(snapshot, staging)
                except TimeoutError:
                    raise
                except Exception:
                    logging.warning("JIT snapshot unusable, trying older: %s", snapshot)
                    continue
                # Merge file-by-file via same-dir temp + os.replace so a
                # mid-merge failure never exposes a truncated .so/.cubin.
                for source in (p for p in staging.rglob("*") if p.is_file()):
                    destination = target / source.relative_to(staging)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    partial = destination.with_name(
                        f".{destination.name}.{uuid.uuid4().hex}.tmp"
                    )
                    try:
                        shutil.copy2(source, partial)
                        os.replace(partial, destination)
                    finally:
                        partial.unlink(missing_ok=True)
            return True
        return False

    def publish_snapshot(self, files: dict[str, Path]) -> None:
        if not files:
            return

        # Lock-free publish: repack newest → fresh immutable archive → GC,
        # sidestepping OSS/FUSE cross-writer/partial-write hazards. Full repack
        # (not incremental) is fine: cold-start-only + coalesced, so rare.
        with tempfile.TemporaryDirectory(prefix=".jit_snapshot.") as tmp_name:
            workspace = Path(tmp_name)
            staging = workspace / "staging"
            staging.mkdir()
            # Re-base on newest, keeping stale scopes on purpose: one remote cache
            # is shared across services with differing scopes, and storage is cheap.
            self.restore(staging)
            for name, source in files.items():
                destination = staging / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                copy_target = destination.with_name(
                    f".{destination.name}.{uuid.uuid4().hex}.tmp"
                )
                try:
                    shutil.copy2(source, copy_target)
                    os.replace(copy_target, destination)
                except FileNotFoundError:
                    # Some compilers close files in a staging directory and then
                    # atomically move or delete them before the coalesced flush.
                    # Keep the previous snapshot value, if any, and publish the
                    # remaining stable artifacts in this batch.
                    with suppress(OSError):
                        copy_target.unlink()
            archive = workspace / "candidate.tar.zst"
            pack_zstd_tar(archive, staging)
            # Lock-free commit to a fresh target. Concurrent publishes may miss
            # each other's artifacts; they'll recompile and republish.
            token = uuid.uuid4().hex
            committed = self.remote_root / (
                f"{time.time_ns():020d}-{token}{SNAPSHOT_SUFFIX}"
            )
            remote_tmp = self.remote_root / f".upload.{token}.tmp"
            try:
                shutil.copyfile(archive, remote_tmp)
                self._wait_remote_ready(remote_tmp, archive)
                os.rename(remote_tmp, committed)
            finally:
                with suppress(OSError):
                    remote_tmp.unlink()
        # Reap over-KEEP snapshots and orphan .upload.*.tmp (SIGKILL'd pre-rename),
        # age-gated so GC never unlink-races a live reader/writer (see IDLE_REAP_S).
        cutoff = time.time() - IDLE_REAP_S
        snapshots = sorted(self.remote_root.glob(f"*{SNAPSHOT_SUFFIX}"))
        _reap_idle(snapshots[:-SNAPSHOT_KEEP], cutoff)
        _reap_idle(self.remote_root.glob(".upload.*.tmp"), cutoff)
