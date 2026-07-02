from __future__ import annotations

import logging
import os
import re
import shutil
import tarfile
import tempfile
import threading
import time
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from functools import lru_cache
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Callable, Iterator
from urllib.parse import urlparse

import zstandard as zstd
from filelock import FileLock
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from rtp_llm.utils.time_util import current_time_ms, elapsed_ms

SNAPSHOT_NAME = ".jit_snapshot.tar.zst"
SNAPSHOT_LOCK_DIR_NAME = ".jit_snapshot.lock.dir"
SNAPSHOT_EXTRACT_LOCK_NAME = ".jit_snapshot.extract.lock"
SNAPSHOT_LOCK_POLL_S = 0.1
SNAPSHOT_LOCK_TIMEOUT_S = 60.0
WATCHDOG_JOIN_TIMEOUT_S = 2.0
DEBOUNCE_PUBLISH_S = 5.0


@dataclass(frozen=True)
class ComponentSpec:
    name: str
    env_name: str
    sync_suffixes: tuple[str, ...] = (".so", ".cubin", ".json")
    upload_events: frozenset[str] = frozenset({"closed"})
    scope_func: Callable[[], str | None] | None = None

    @property
    def scope(self) -> str | None:
        return self.scope_func() if self.scope_func else None

    def cache_dir(self, root: Path) -> Path:
        return root / self.name / scope if (scope := self.scope) else root / self.name

    def snapshot_rel(self, rel: str) -> str:
        return f"{scope}/{rel}" if (scope := self.scope) else rel

    def local_rel(self, rel: str) -> str | None:
        if not (scope := self.scope):
            return rel
        first, sep, rest = rel.partition("/")
        return rest if sep and first == scope and rest else None


@lru_cache(maxsize=None)
def _package_version(dist_name: str) -> str:
    try:
        return importlib_metadata.version(dist_name)
    except importlib_metadata.PackageNotFoundError:
        return "unknown"


def _sanitize_scope_part(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", value).strip("_") or "unknown"


def _cuda_version() -> str:
    import torch

    return str(torch.version.cuda or "unknown")


@lru_cache(maxsize=1)
def get_gpu_info() -> str:
    gpu_name = os.environ.get("TRITON_AUTOTUNE_GPU_NAME")
    if not gpu_name:
        import torch

        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return _sanitize_scope_part(gpu_name or "unknown")


COMPONENT_SPECS = (
    ComponentSpec(
        "flashinfer",
        "FLASHINFER_WORKSPACE_BASE",
        scope_func=lambda: "cuda-" + _sanitize_scope_part(_cuda_version()),
    ),
    ComponentSpec(
        "deep_gemm",
        "DG_JIT_CACHE_DIR",
        upload_events=frozenset({"created", "moved"}),
        scope_func=lambda: "deep_gemm-"
        + _sanitize_scope_part(_package_version("deep_gemm")),
    ),
    ComponentSpec(
        "torch_extensions",
        "TORCH_EXTENSIONS_DIR",
        scope_func=lambda: "torch-" + _sanitize_scope_part(_package_version("torch")),
    ),
    ComponentSpec("triton", "TRITON_CACHE_DIR", upload_events=frozenset({"moved"})),
    ComponentSpec(
        "triton_autotune",
        "TRITON_AUTOTUNE_CONFIG_DIR",
        sync_suffixes=(".json", ".pkl", ".pickle"),
        scope_func=get_gpu_info,
    ),
)
COMPONENT_BY_NAME = {c.name: c for c in COMPONENT_SPECS}


def _local_path(p: str | Path) -> Path:
    return Path(p).expanduser().absolute()


def component_cache_dir(root: Path, component: ComponentSpec) -> Path:
    return component.cache_dir(root)


def component_snapshot_rel(component: ComponentSpec, rel: str) -> str:
    return component.snapshot_rel(rel)


def snapshot_path(remote_root: Path) -> Path:
    return remote_root / SNAPSHOT_NAME


def should_sync_file(component: ComponentSpec, rel: str) -> bool:
    return rel.endswith(component.sync_suffixes) and not any(
        part.startswith("tmp.pid_") for part in rel.split("/")
    )


def resolve_remote_root(remote_jit_dir: Any) -> Path | None:
    text = str(remote_jit_dir or "").strip()
    if not text:
        return None
    if urlparse(text).scheme:
        from rtp_llm.utils.fuser import MountRwMode, fetch_remote_file_to_local

        try:
            text = fetch_remote_file_to_local(text, MountRwMode.RWMODE_RW)
        except Exception as e:
            logging.warning("failed to mount remote_jit_dir %r: %s", text, e)
            return None
    path = _local_path(text)
    if path.is_dir():
        return path
    logging.warning("remote_jit_dir %r does not exist; remote JIT cache disabled", text)
    return None


def apply_jit_cache_env(local_root: Path | str) -> None:
    root = _local_path(local_root)
    for component in COMPONENT_SPECS:
        os.environ[component.env_name] = str(component.cache_dir(root))
    os.environ.setdefault("TRITON_AUTOTUNE_CACHE_MODE", "cached")


def apply_jit_cache_env_from_env() -> None:
    apply_jit_cache_env(os.environ.get("LOCAL_JIT_DIR", "./.jit_cache"))


def iter_component_sync_files(
    root: Path, component: ComponentSpec
) -> Iterator[tuple[Path, str]]:
    if not root.is_dir():
        return
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for filename in filenames:
            path = Path(dirpath) / filename
            rel = path.relative_to(root).as_posix()
            with suppress(OSError):
                if (
                    should_sync_file(component, rel)
                    and path.stat(follow_symlinks=False).st_size > 0
                ):
                    yield path, rel


@contextmanager
def open_snapshot_reader(path: Path) -> Iterator[tarfile.TarFile]:
    with (
        path.open("rb") as compressed,
        zstd.ZstdDecompressor().stream_reader(compressed) as reader,
        tarfile.open(fileobj=reader, mode="r|") as tar,
    ):
        yield tar


@contextmanager
def open_snapshot_writer(path: Path) -> Iterator[tarfile.TarFile]:
    with (
        path.open("wb") as raw,
        zstd.ZstdCompressor(level=3).stream_writer(raw) as compressed,
        tarfile.open(fileobj=compressed, mode="w|") as tar,
    ):
        yield tar


@contextmanager
def _snapshot_mklock(remote_root: Path) -> Iterator[None]:
    lock_dir = remote_root / SNAPSHOT_LOCK_DIR_NAME
    deadline = time.monotonic() + SNAPSHOT_LOCK_TIMEOUT_S
    while time.monotonic() < deadline:
        try:
            lock_dir.mkdir()
            break
        except FileExistsError:
            time.sleep(SNAPSHOT_LOCK_POLL_S)
    else:
        raise TimeoutError(f"failed to acquire snapshot lock: {lock_dir}")
    try:
        yield
    finally:
        with suppress(OSError):
            lock_dir.rmdir()


class _JitFileEventHandler(FileSystemEventHandler):
    def __init__(
        self,
        component: ComponentSpec,
        root: Path,
        upload: Callable[[ComponentSpec, str], bool],
    ):
        self.component, self.root_prefix, self.upload = (
            component,
            str(root) + os.sep,
            upload,
        )

    def on_any_event(self, event: Any) -> None:
        if event.is_directory or event.event_type not in self.component.upload_events:
            return
        src = event.dest_path if event.event_type == "moved" else event.src_path
        if not src.startswith(self.root_prefix):
            return
        rel = src[len(self.root_prefix) :].replace(os.sep, "/")
        if should_sync_file(self.component, rel):
            self.upload(self.component, rel)


class JitCacheManager:
    def __init__(
        self, jit_config: Any | None = None, *, debounce_s: float = DEBOUNCE_PUBLISH_S
    ):
        from rtp_llm.config.py_config_modules import JITConfig

        jit_config = jit_config or JITConfig()
        self.remote_root = resolve_remote_root(jit_config.remote_jit_dir)
        self.local_root = _local_path(jit_config.local_jit_dir)
        self.component_dirs: dict[str, Path] = {}
        self._observer: Any | None = None
        self._stopping, self._debounce_s = False, debounce_s
        self._debounce_lock = threading.Lock()
        self._debounce_timer: threading.Timer | None = None

    @property
    def enabled(self) -> bool:
        return self.remote_root is not None

    def make_summary(
        self, mode: str, result: str, start_s: float, **extra: Any
    ) -> dict[str, Any]:
        event = {
            "timestamp_ms": int(current_time_ms()),
            "mode": mode,
            "result": result,
            "total_cost_ms": elapsed_ms(start_s),
        }
        event.update(extra)
        return event

    def bootstrap(self) -> None:
        self.local_root.mkdir(parents=True, exist_ok=True)
        apply_jit_cache_env(self.local_root)
        self.component_dirs = {}
        for component in COMPONENT_SPECS:
            local_dir = _local_path(os.environ[component.env_name])
            local_dir.mkdir(parents=True, exist_ok=True)
            if self.remote_root is not None:
                self.component_dirs[component.name] = local_dir
        logging.info("JIT cache local=%s remote=%s", self.local_root, self.remote_root)

    def prepare(self, should_stop: Callable[[], bool] | None = None) -> dict[str, Any]:
        start_s = time.monotonic()
        if not self.enabled:
            return self.make_summary(
                "snapshot_download", "skipped", start_s, cache_state="disabled"
            )
        if should_stop is not None and should_stop():
            raise KeyboardInterrupt("JIT cache prepare cancelled")
        archive = snapshot_path(self.remote_root)
        if not archive.is_file():
            return self.make_summary(
                "snapshot_download", "skipped", start_s, cache_state="snapshot_miss"
            )
        try:
            with FileLock(
                str(self.local_root / SNAPSHOT_EXTRACT_LOCK_NAME),
                timeout=SNAPSHOT_LOCK_TIMEOUT_S,
            ):
                extracted_bytes, extracted_files = self._extract_snapshot(archive)
        except Exception as e:
            logging.exception("failed to extract JIT cache snapshot")
            return self.make_summary(
                "snapshot_download",
                "failed",
                start_s,
                cache_state="snapshot_error",
                message=str(e),
            )
        return self.make_summary(
            "snapshot_download",
            "success",
            start_s,
            cache_state="snapshot_hit",
            extracted_files=extracted_files,
            extracted_bytes=extracted_bytes,
        )

    def _extract_snapshot(self, archive: Path) -> tuple[int, int]:
        files = bytes_ = 0
        with open_snapshot_reader(archive) as tar:
            for member in tar:
                parts = member.name.split("/")
                if member.name.startswith("/") or any(
                    part in ("", "..") for part in parts
                ):
                    raise RuntimeError(f"unsafe snapshot path: {member.name}")
                if not member.isfile() or parts[0] not in self.component_dirs:
                    continue
                component_name, component = parts[0], COMPONENT_BY_NAME[parts[0]]
                local_rel = component.local_rel("/".join(parts[1:]))
                if local_rel is None or not should_sync_file(component, local_rel):
                    continue
                source = tar.extractfile(member)
                if source is None:
                    continue
                target = self.component_dirs[component_name] / local_rel
                target.parent.mkdir(parents=True, exist_ok=True)
                with source, target.open("wb") as out:
                    shutil.copyfileobj(source, out)
                os.utime(target, (member.mtime, member.mtime))
                files, bytes_ = files + 1, bytes_ + member.size
        return bytes_, files

    def _make_handler(
        self, component: ComponentSpec, root: Path
    ) -> _JitFileEventHandler:
        return _JitFileEventHandler(component, root, self.upload_file)

    def start_background_sync(self) -> None:
        if not self.enabled or self._observer is not None:
            return
        observer = Observer()
        try:
            for name, local_dir in self.component_dirs.items():
                observer.schedule(
                    self._make_handler(COMPONENT_BY_NAME[name], local_dir),
                    str(local_dir),
                    recursive=True,
                )
            observer.start()
            self._observer = observer
        except Exception:
            logging.warning("failed to start JIT cache watcher", exc_info=True)
            with suppress(Exception):
                observer.stop()
                observer.join(timeout=WATCHDOG_JOIN_TIMEOUT_S)

    def upload_file(self, component: ComponentSpec, rel_path: str) -> bool:
        if self._stopping or component.name not in self.component_dirs:
            return False
        try:
            if (self.component_dirs[component.name] / rel_path).stat().st_size <= 0:
                return False
        except FileNotFoundError:
            return False
        except Exception:
            logging.exception(
                "failed to stat JIT cache file %s/%s", component.name, rel_path
            )
            return False
        try:
            self._schedule_publish()
            return True
        except Exception:
            logging.exception(
                "failed to publish JIT cache file %s/%s", component.name, rel_path
            )
            return False

    def _schedule_publish(self) -> None:
        if self._debounce_s <= 0:
            self._publish_snapshot()
            return
        with self._debounce_lock:
            if self._stopping or self._debounce_timer is not None:
                return
            self._debounce_timer = threading.Timer(
                self._debounce_s, self._debounce_fire
            )
            self._debounce_timer.daemon = True
            self._debounce_timer.start()

    def _cancel_debounce(self) -> None:
        with self._debounce_lock:
            if self._debounce_timer is not None:
                self._debounce_timer.cancel()
                self._debounce_timer = None

    def _debounce_fire(self) -> None:
        with self._debounce_lock:
            self._debounce_timer = None
        if not self._stopping:
            try:
                self._publish_snapshot()
            except Exception:
                logging.exception("failed to publish JIT cache snapshot")

    def sync_once(self, mode: str = "manual_sync") -> dict[str, Any]:
        start_s = time.monotonic()
        if self._stopping:
            return self.make_summary(mode, "skipped", start_s, reason="stopping")
        if not self.enabled:
            return self.make_summary(mode, "skipped", start_s, cache_state="disabled")
        self._cancel_debounce()
        try:
            self._publish_snapshot()
        except Exception as e:
            logging.exception("failed to publish JIT cache snapshot")
            return self.make_summary(mode, "failed", start_s, message=str(e))
        return self.make_summary(mode, "success", start_s)

    def _publish_snapshot(self) -> None:
        if self.remote_root is None:
            return
        start_s = time.monotonic()
        archive = snapshot_path(self.remote_root)
        files, bytes_ = self._create_snapshot_archive(archive)
        logging.info(
            "published JIT cache snapshot %s files=%d bytes=%d cost_ms=%d",
            archive,
            files,
            bytes_,
            elapsed_ms(start_s),
        )

    def _create_snapshot_archive(self, archive: Path) -> tuple[int, int]:
        local_files: dict[str, Path] = {}
        for name, local_dir in self.component_dirs.items():
            component = COMPONENT_BY_NAME[name]
            local_files.update(
                {
                    f"{name}/{component.snapshot_rel(rel)}": path
                    for path, rel in iter_component_sync_files(local_dir, component)
                }
            )
        if not local_files:
            return 0, 0
        archive.parent.mkdir(parents=True, exist_ok=True)
        with _snapshot_mklock(archive.parent):
            files = bytes_ = 0
            existing_arcnames: set[str] = set()
            with tempfile.TemporaryDirectory(prefix=".jit_snapshot_") as tmp_dir:
                local_tmp = Path(tmp_dir) / SNAPSHOT_NAME
                with open_snapshot_writer(local_tmp) as tar_out:
                    if archive.is_file():
                        files, bytes_ = self._copy_existing_snapshot(
                            archive, tar_out, existing_arcnames
                        )
                    added = self._add_new_snapshot_files(
                        tar_out, local_files, existing_arcnames
                    )
                    if added[0] == 0:
                        return files, bytes_
                    files, bytes_ = files + added[0], bytes_ + added[1]
                remote_tmp: Path | None = archive.with_name(
                    f"{archive.name}.{os.getpid()}.{time.time_ns()}.tmp"
                )
                try:
                    shutil.copyfile(local_tmp, remote_tmp)
                    with suppress(FileNotFoundError):
                        archive.unlink()
                    remote_tmp.rename(archive)
                    remote_tmp = None
                finally:
                    if remote_tmp is not None:
                        with suppress(OSError):
                            remote_tmp.unlink()
        return files, bytes_

    def _copy_existing_snapshot(
        self, archive: Path, tar_out: tarfile.TarFile, existing: set[str]
    ) -> tuple[int, int]:
        files = bytes_ = 0
        with open_snapshot_reader(archive) as tar_in:
            for member in tar_in:
                if not member.isfile():
                    continue
                source = tar_in.extractfile(member)
                if source is None:
                    continue
                with source:
                    tar_out.addfile(member, source)
                existing.add(member.name)
                files, bytes_ = files + 1, bytes_ + member.size
        return files, bytes_

    def _add_new_snapshot_files(
        self, tar_out: tarfile.TarFile, local_files: dict[str, Path], existing: set[str]
    ) -> tuple[int, int]:
        files = bytes_ = 0
        for arcname, local_path in local_files.items():
            if arcname in existing:
                continue
            try:
                size = local_path.stat().st_size
                tar_out.add(str(local_path), arcname=arcname, recursive=False)
            except OSError:
                continue
            files, bytes_ = files + 1, bytes_ + size
        return files, bytes_

    def stop(self) -> None:
        self._stopping = True
        self._cancel_debounce()
        observer, self._observer = self._observer, None
        if observer is not None:
            observer.stop()
            observer.join(timeout=WATCHDOG_JOIN_TIMEOUT_S)
