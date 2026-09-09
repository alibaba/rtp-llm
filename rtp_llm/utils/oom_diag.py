"""Best-effort Torch GPU allocator diagnostics.

Set ``RTP_OOM_RECORD=1`` before startup to retain allocation history and attach
Torch's OOM observer. The observer remains one-shot per process. Fatal C++ OOM
handling and the manual diagnostics endpoint can also request a dump; manual
dumps are repeatable and serialized.
"""

# pyright: reportPrivateUsage=false
import fcntl
import logging
import os
import pickle
import re
import tempfile
import threading
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import AbstractSet, Any, Dict, Iterator, Optional, TextIO

import torch

_LOG = logging.getLogger(__name__)
_install_lock = threading.Lock()
_dump_lock = threading.Lock()
_installed = False
_oom_fired = False
_last_observer_dump: Optional[str] = None
_dump_counter = 0

_RECORD_ENV = "RTP_OOM_RECORD"
_LOG_DIR_ENV = "LOG_PATH"
_OUT_DIR = "logs"
_MAX_TRACE_ENTRIES = 500_000
_DUMP_MAX_COUNT_ENV = "RTP_OOM_DUMP_MAX_COUNT"
_DUMP_MAX_AGE_SECONDS_ENV = "RTP_OOM_DUMP_MAX_AGE_SECONDS"
_DUMP_MAX_BYTES_ENV = "RTP_OOM_DUMP_MAX_BYTES"
_DEFAULT_DUMP_MAX_COUNT = 8
_DEFAULT_DUMP_MAX_AGE_SECONDS = 24 * 60 * 60
_DEFAULT_DUMP_MAX_BYTES = 2 * 1024**3
_RETENTION_LOCK_FILE = ".oom_allocator_retention.lock"
_IN_PROGRESS_PREFIX = ".oom_allocator_in_progress_"
_SAFE_COMPONENT = re.compile(r"[^A-Za-z0-9_.-]+")
_CORRELATION_GROUP = re.compile(r"_cid([A-Za-z0-9-]+)_r")

_BLOCK_USAGE = {
    "active_allocated": "ACTIVE_ALLOCATED",
    "active_awaiting_free": "ACTIVE_AWAITING_FREE",
    "inactive": "CACHED_FREE",
}


def _enabled() -> bool:
    return os.environ.get(_RECORD_ENV) == "1"


def _out_dir() -> Path:
    path = Path(
        os.environ.get(_LOG_DIR_ENV)
        or os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR")
        or _OUT_DIR
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def _sanitize_filename_component(value: Any, fallback: str = "unknown") -> str:
    sanitized = _SAFE_COMPONENT.sub("_", str(value))
    sanitized = re.sub(r"\.{2,}", "_", sanitized).strip(".")[:96]
    return sanitized or fallback


def _sanitize_correlation_id(value: Any) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9-]+", "-", str(value)).strip("-")[:64]
    return sanitized or "unknown"


def _suffix(
    tag: str,
    device: int,
    sequence: int,
    dump_correlation_id: Optional[str] = None,
) -> str:
    world_rank = _sanitize_filename_component(
        os.environ.get("WORLD_RANK", os.environ.get("RANK", "0"))
    )
    server_id = _sanitize_filename_component(os.environ.get("FRONTEND_SERVER_ID", "0"))
    safe_tag = _sanitize_filename_component(tag)
    safe_device = _sanitize_filename_component(device)
    correlation = (
        f"_cid{_sanitize_correlation_id(dump_correlation_id)}"
        if dump_correlation_id
        else ""
    )
    return (
        f"{safe_tag}{correlation}_r{world_rank}_s{server_id}_d{safe_device}"
        f"_pid{os.getpid()}_n{sequence:06d}"
    )


@contextmanager
def _retention_lock(output_dir: Path) -> Iterator[None]:
    lock_path = output_dir / _RETENTION_LOCK_FILE
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _positive_retention_limit(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = int(value)
        if parsed <= 0:
            raise ValueError("must be positive")
        return parsed
    except ValueError:
        _LOG.warning("[OOM_DUMP] ignoring invalid %s=%r", name, value)
        return default


def _group_key_from_suffix(suffix: str) -> str:
    correlations = _CORRELATION_GROUP.findall(suffix)
    return f"correlation:{correlations[-1]}" if correlations else suffix


def _dump_group_key(path: Path) -> Optional[str]:
    name = path.name
    if name.startswith("oom_allocator_snapshot_") and name.endswith(".pickle"):
        suffix = name[len("oom_allocator_snapshot_") : -len(".pickle")]
    elif name.startswith("oom_allocator_") and name.endswith(".log"):
        suffix = name[len("oom_allocator_") : -len(".log")]
    else:
        return None
    return _group_key_from_suffix(suffix)


def _active_dump_group_keys(
    output_dir: Path, current_time: float, stale_after_seconds: int
) -> set[str]:
    """Return live publication groups and remove expired crash markers.

    Publishers hold an exclusive advisory lock on their marker for the entire
    atomic write. An unlocked marker is retained briefly to tolerate startup
    races, then removed once it is older than the configured dump age.
    """
    active = set()
    try:
        markers = list(output_dir.glob(f"{_IN_PROGRESS_PREFIX}*"))
    except OSError as error:
        _LOG.warning("[OOM_DUMP] failed to list in-progress markers: %s", error)
        return active

    for marker in markers:
        group_key = _group_key_from_suffix(marker.name[len(_IN_PROGRESS_PREFIX) :])
        fd: Optional[int] = None
        marker_locked = False
        try:
            fd = os.open(marker, os.O_RDWR)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                marker_locked = True
            except BlockingIOError:
                active.add(group_key)
                continue

            marker_age = max(0.0, current_time - os.fstat(fd).st_mtime)
            if marker_age > stale_after_seconds:
                marker.unlink(missing_ok=True)
            else:
                active.add(group_key)
        except FileNotFoundError:
            continue
        except OSError as error:
            active.add(group_key)
            _LOG.warning(
                "[OOM_DUMP] failed to inspect in-progress marker %s: %s",
                marker,
                error,
            )
        finally:
            if fd is not None:
                if marker_locked:
                    try:
                        fcntl.flock(fd, fcntl.LOCK_UN)
                    except OSError as error:
                        _LOG.warning(
                            "[OOM_DUMP] failed to unlock in-progress marker %s: %s",
                            marker,
                            error,
                        )
                try:
                    os.close(fd)
                except OSError as error:
                    _LOG.warning(
                        "[OOM_DUMP] failed to close in-progress marker %s: %s",
                        marker,
                        error,
                    )
    return active


def _prune_dump_files(
    output_dir: Path,
    now: Optional[float] = None,
    protected_group_keys: AbstractSet[str] = frozenset(),
) -> None:
    """Apply retention while the caller holds thread and interprocess locks."""
    max_count = _positive_retention_limit(_DUMP_MAX_COUNT_ENV, _DEFAULT_DUMP_MAX_COUNT)
    max_age_seconds = _positive_retention_limit(
        _DUMP_MAX_AGE_SECONDS_ENV, _DEFAULT_DUMP_MAX_AGE_SECONDS
    )
    max_bytes = _positive_retention_limit(_DUMP_MAX_BYTES_ENV, _DEFAULT_DUMP_MAX_BYTES)
    current_time = time.time() if now is None else now
    protected = set(protected_group_keys) | _active_dump_group_keys(
        output_dir, current_time, max_age_seconds
    )

    groups: Dict[str, list[tuple[Path, int, float]]] = {}
    try:
        candidates = list(output_dir.glob("oom_allocator_*"))
    except OSError as error:
        _LOG.warning("[OOM_DUMP] failed to list retained dumps: %s", error)
        return

    for path in candidates:
        group_key = _dump_group_key(path)
        if group_key is None:
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        groups.setdefault(group_key, []).append((path, stat.st_size, stat.st_mtime))

    def remove_group(group_key: str) -> int:
        removed_bytes = 0
        for path, size, _ in groups.pop(group_key, []):
            try:
                path.unlink()
                removed_bytes += size
            except FileNotFoundError:
                removed_bytes += size
            except OSError as error:
                _LOG.warning("[OOM_DUMP] failed to prune %s: %s", path, error)
        return removed_bytes

    for group_key, files in list(groups.items()):
        if (
            group_key not in protected
            and current_time - max(item[2] for item in files) > max_age_seconds
        ):
            remove_group(group_key)

    def removable_groups() -> list[str]:
        return sorted(
            (group_key for group_key in groups if group_key not in protected),
            key=lambda group_key: max(item[2] for item in groups[group_key]),
        )

    while len(groups) > max_count:
        candidates_to_remove = removable_groups()
        if not candidates_to_remove:
            break
        remove_group(candidates_to_remove[0])

    retained_bytes = sum(size for files in groups.values() for _, size, _ in files)
    while retained_bytes > max_bytes:
        candidates_to_remove = removable_groups()
        if not candidates_to_remove:
            break
        retained_bytes -= remove_group(candidates_to_remove[0])


def _human_bytes(value: Optional[int]) -> str:
    if value is None:
        return "unknown"
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(amount) < 1024 or unit == "TiB":
            return f"{amount:.2f} {unit}"
        amount /= 1024
    return f"{value} B"


def _hex(value: Any) -> str:
    try:
        return f"0x{int(value):x}"
    except (TypeError, ValueError):
        return str(value)


def _write_frames(output: TextIO, frames: Any, indent: str) -> None:
    if not frames:
        output.write(
            f"{indent}allocation_frames=<not recorded; set {_RECORD_ENV}=1 before startup>\n"
        )
        return
    output.write(f"{indent}allocation_frames={len(frames)}\n")
    for index, frame in enumerate(frames):
        output.write(
            f"{indent}  frame[{index:04d}] {frame.get('name', '<unknown>')} "
            f"at {frame.get('filename', '<unknown>')}:{frame.get('line', 0)}\n"
        )


def _snapshot_counts(snapshot: Optional[Dict[str, Any]]) -> tuple[int, int]:
    if not snapshot:
        return 0, 0
    segments = snapshot.get("segments", [])
    return len(segments), sum(len(segment.get("blocks", [])) for segment in segments)


def _write_allocator_blocks(output: TextIO, snapshot: Dict[str, Any]) -> None:
    segments = snapshot.get("segments", [])
    state_bytes: Dict[str, int] = {}
    state_blocks: Dict[str, int] = {}
    total_segment_bytes = 0
    total_requested_bytes = 0
    block_count = 0
    for segment in segments:
        total_segment_bytes += int(segment.get("total_size", 0))
        for block in segment.get("blocks", []):
            state = str(block.get("state", "unknown"))
            size = int(block.get("size", 0))
            state_bytes[state] = state_bytes.get(state, 0) + size
            state_blocks[state] = state_blocks.get(state, 0) + 1
            total_requested_bytes += int(block.get("requested_size", 0))
            block_count += 1

    output.write(f"allocator_settings={snapshot.get('allocator_settings', {})!r}\n")
    output.write(
        f"segments={len(segments)} blocks={block_count} "
        f"segment_bytes={total_segment_bytes} ({_human_bytes(total_segment_bytes)}) "
        f"requested_bytes={total_requested_bytes} ({_human_bytes(total_requested_bytes)})\n"
    )
    for state in sorted(state_bytes):
        output.write(
            f"state={state} usage={_BLOCK_USAGE.get(state, state.upper())} "
            f"blocks={state_blocks[state]} bytes={state_bytes[state]} "
            f"({_human_bytes(state_bytes[state])})\n"
        )

    for segment_index, segment in enumerate(segments):
        segment_address = int(segment.get("address", 0))
        total_size = int(segment.get("total_size", 0))
        allocated_size = int(segment.get("allocated_size", 0))
        active_size = int(segment.get("active_size", 0))
        requested_size = int(segment.get("requested_size", 0))
        blocks = segment.get("blocks", [])
        output.write("\n")
        output.write(
            f"SEGMENT[{segment_index:06d}] device={segment.get('device', '<unknown>')} "
            f"address={_hex(segment_address)} total_size={total_size} ({_human_bytes(total_size)}) "
            f"allocated_size={allocated_size} ({_human_bytes(allocated_size)}) "
            f"active_size={active_size} ({_human_bytes(active_size)}) "
            f"requested_size={requested_size} ({_human_bytes(requested_size)}) "
            f"cached_free_size={max(total_size - allocated_size, 0)} "
            f"({_human_bytes(max(total_size - allocated_size, 0))}) "
            f"segment_type={segment.get('segment_type', '<unknown>')} "
            f"stream={_hex(segment.get('stream', 0))} "
            f"pool_id={segment.get('segment_pool_id', '<unknown>')!r} "
            f"expandable={bool(segment.get('is_expandable', False))} blocks={len(blocks)}\n"
        )
        if segment.get("frames"):
            output.write("  segment_allocation_frames:\n")
            _write_frames(output, segment.get("frames"), "    ")

        next_address = segment_address
        for block_index, block in enumerate(blocks):
            address = int(block.get("address", next_address))
            size = int(block.get("size", 0))
            requested = int(block.get("requested_size", 0))
            state = str(block.get("state", "unknown"))
            usage = _BLOCK_USAGE.get(state, state.upper())
            offset = max(address - segment_address, 0)
            slack = max(size - requested, 0)
            share = 100.0 * size / total_size if total_size else 0.0
            output.write(
                f"  BLOCK[{segment_index:06d}.{block_index:06d}] "
                f"address={_hex(address)} offset={offset} ({_human_bytes(offset)}) "
                f"size={size} ({_human_bytes(size)}) "
                f"requested_size={requested} ({_human_bytes(requested)}) "
                f"slack={slack} ({_human_bytes(slack)}) "
                f"state={state} usage={usage} segment_share={share:.4f}%\n"
            )
            _write_frames(output, block.get("frames"), "    ")
            next_address = address + size

    traces = snapshot.get("device_traces", [])
    trace_count = sum(len(device_trace) for device_trace in traces)
    output.write(
        f"\ndevice_trace_devices={len(traces)} device_trace_events={trace_count}; "
        "full allocation/free timeline is stored in the pickle snapshot when recording is enabled\n"
    )


def install_oom_dump() -> None:
    """Optionally enable allocation history and keep Torch's OOM observer."""
    if not _enabled():
        return

    global _installed
    with _install_lock:
        if _installed:
            return
        torch.cuda.memory._record_memory_history(
            enabled="all",
            context="all",
            stacks="all",
            max_entries=_MAX_TRACE_ENTRIES,
        )
        torch._C._cuda_attach_out_of_memory_observer(_oom_observer)  # type: ignore[attr-defined]
        _installed = True

    _LOG.info(
        "[OOM_DUMP] allocation history and OOM observer enabled device=%d pid=%d dir=%s stacks=all",
        torch.cuda.current_device(),
        os.getpid(),
        _out_dir(),
    )


def dump_oom_diagnostics(
    tag: str = "allocator_dump",
    device: Optional[int] = None,
    alloc_size: int = 0,
    device_total: Optional[int] = None,
    device_free: Optional[int] = None,
    exception: Optional[str] = None,
    cpp_backtrace: Optional[str] = None,
    reuse_observer_dump: bool = False,
    dump_correlation_id: Optional[str] = None,
) -> Optional[str]:
    """Write a repeatable allocator dump without allowing failures to escape."""
    global _dump_counter

    if reuse_observer_dump:
        with _install_lock:
            if _last_observer_dump is not None and Path(_last_observer_dump).exists():
                return _last_observer_dump

    with _dump_lock:
        sequence = _dump_counter
        _dump_counter += 1
        output_dir: Optional[Path] = None
        marker_path: Optional[Path] = None
        marker_fd: Optional[int] = None
        temporary_paths: list[Path] = []
        correlation_id = (
            _sanitize_correlation_id(dump_correlation_id)
            if dump_correlation_id
            else None
        )
        try:
            output_dir = _out_dir()
            if device is None:
                device = torch.cuda.current_device()
            suffix = _suffix(tag, device, sequence, correlation_id)
            group_key = _group_key_from_suffix(suffix)
            pending_marker_path = output_dir / f"{_IN_PROGRESS_PREFIX}{suffix}"
            with _retention_lock(output_dir):
                _prune_dump_files(output_dir, protected_group_keys={group_key})
                marker_fd = os.open(
                    pending_marker_path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600
                )
                marker_path = pending_marker_path
                fcntl.flock(marker_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

            devices = list(range(torch.cuda.device_count()))
            if device not in devices:
                devices.append(device)

            memory_by_device: Dict[int, tuple[Optional[int], Optional[int]]] = {}
            stats_by_device: Dict[int, Dict[str, Any]] = {}
            summary_by_device: Dict[int, str] = {}
            for current_device in devices:
                current_free = device_free if current_device == device else None
                current_total = device_total if current_device == device else None
                try:
                    queried_free, queried_total = torch.cuda.mem_get_info(
                        current_device
                    )
                    current_free = (
                        queried_free if current_free is None else current_free
                    )
                    current_total = (
                        queried_total if current_total is None else current_total
                    )
                except Exception as error:  # noqa: BLE001
                    _LOG.exception(
                        "[OOM_DUMP] failed to read device=%d memory: %s",
                        current_device,
                        error,
                    )
                memory_by_device[current_device] = (current_free, current_total)

                try:
                    stats_by_device[current_device] = torch.cuda.memory_stats(
                        device=current_device
                    )
                except Exception as error:  # noqa: BLE001
                    _LOG.exception(
                        "[OOM_DUMP] failed to read device=%d allocator stats: %s",
                        current_device,
                        error,
                    )
                    stats_by_device[current_device] = {}

                try:
                    summary_by_device[current_device] = torch.cuda.memory_summary(
                        device=current_device, abbreviated=False
                    )
                except Exception as error:  # noqa: BLE001
                    _LOG.exception(
                        "[OOM_DUMP] failed to render device=%d allocator summary: %s",
                        current_device,
                        error,
                    )
                    summary_by_device[current_device] = (
                        f"<failed to render torch.cuda.memory_summary: {error}>"
                    )

            allocator_snapshot: Optional[Dict[str, Any]] = None
            snapshot_error: Optional[str] = None
            try:
                allocator_snapshot = torch.cuda.memory._snapshot()
            except Exception as error:  # noqa: BLE001
                snapshot_error = str(error)
                _LOG.exception(
                    "[OOM_DUMP] failed to collect allocator blocks: %s", error
                )

            snapshot_path = output_dir / f"oom_allocator_snapshot_{suffix}.pickle"
            snapshot_value = "disabled; set RTP_OOM_RECORD=1 before startup"
            if _enabled() and allocator_snapshot is not None:
                try:
                    with tempfile.NamedTemporaryFile(
                        mode="wb",
                        prefix=f".{snapshot_path.name}.",
                        suffix=".tmp",
                        dir=output_dir,
                        delete=False,
                    ) as snapshot_file:
                        snapshot_tmp_path = Path(snapshot_file.name)
                        temporary_paths.append(snapshot_tmp_path)
                        pickle.dump(allocator_snapshot, snapshot_file)
                    os.replace(snapshot_tmp_path, snapshot_path)
                    temporary_paths.remove(snapshot_tmp_path)
                    snapshot_value = str(snapshot_path)
                except Exception as error:  # noqa: BLE001
                    _LOG.exception(
                        "[OOM_DUMP] failed to dump allocation snapshot: %s", error
                    )
                    snapshot_value = f"failed: {error}"

            output_path = output_dir / f"oom_allocator_{suffix}.log"
            with tempfile.NamedTemporaryFile(
                mode="w",
                prefix=f".{output_path.name}.",
                suffix=".tmp",
                dir=output_dir,
                delete=False,
                encoding="utf-8",
            ) as output:
                output_tmp_path = Path(output.name)
                temporary_paths.append(output_tmp_path)
                output.write("=" * 120 + "\n")
                output.write("RTP-LLM TORCH GPU ALLOCATOR DIAGNOSTICS\n")
                output.write("=" * 120 + "\n")
                output.write(f"tag={tag}\n")
                output.write(f"dump_sequence={sequence}\n")
                output.write(f"dump_id={correlation_id or '<automatic>'}\n")
                output.write(f"primary_device={device}\n")
                output.write(f"logical_devices={devices}\n")
                output.write(
                    f"world_rank={os.environ.get('WORLD_RANK', os.environ.get('RANK', '0'))}\n"
                )
                output.write(
                    f"frontend_server_id={os.environ.get('FRONTEND_SERVER_ID', '0')}\n"
                )
                output.write(f"pid={os.getpid()}\n")
                output.write(f"time={time.time()}\n")
                output.write(f"snapshot={snapshot_value}\n")
                output.write("\n[ORIGINAL EXCEPTION]\n")
                output.write(f"{exception or '<not provided>'}\n")
                output.write("\n[ORIGINAL C++ BACKTRACE]\n")
                output.write(f"{cpp_backtrace or '<not provided>'}\n")
                output.write("\n[GPU DEVICE MEMORY - ALL LOGICAL DEVICES]\n")
                output.write(f"failed_alloc_device={device}\n")
                output.write(
                    f"failed_alloc_bytes={alloc_size} ({_human_bytes(alloc_size)})\n"
                )
                for current_device in devices:
                    current_free, current_total = memory_by_device[current_device]
                    current_used = (
                        None
                        if current_free is None or current_total is None
                        else current_total - current_free
                    )
                    output.write(
                        f"DEVICE[{current_device}] used_bytes={current_used} ({_human_bytes(current_used)}) "
                        f"free_bytes={current_free} ({_human_bytes(current_free)}) "
                        f"total_bytes={current_total} ({_human_bytes(current_total)})\n"
                    )
                output.write(
                    "\n[TORCH ALLOCATOR STATS - ALL KEYS, ALL LOGICAL DEVICES]\n"
                )
                for current_device in devices:
                    output.write(f"\nDEVICE[{current_device}]\n")
                    for name in sorted(stats_by_device[current_device]):
                        value = int(stats_by_device[current_device][name])
                        human = f" ({_human_bytes(value)})" if "bytes" in name else ""
                        output.write(f"{name}={value}{human}\n")
                output.write("\n[TORCH MEMORY SUMMARY - ALL LOGICAL DEVICES]\n")
                for current_device in devices:
                    output.write(
                        f"\nDEVICE[{current_device}]\n{summary_by_device[current_device]}\n"
                    )
                output.write(
                    "\n[TORCH ALLOCATOR SEGMENTS AND BLOCKS - FULL, NOT TRUNCATED]\n"
                )
                if allocator_snapshot is not None:
                    _write_allocator_blocks(output, allocator_snapshot)
                else:
                    output.write(
                        f"<failed to collect allocator snapshot: {snapshot_error}>\n"
                    )
                output.write(
                    "\n[DIAGNOSTIC PYTHON STACK - NOT THE ORIGINAL GPU EXCEPTION STACK]\n"
                )
                output.write("".join(traceback.format_stack()))
            os.replace(output_tmp_path, output_path)
            temporary_paths.remove(output_tmp_path)

            with _retention_lock(output_dir):
                marker_path.unlink(missing_ok=True)
                marker_path = None
                _prune_dump_files(output_dir, protected_group_keys={group_key})

            segment_count, block_count = _snapshot_counts(allocator_snapshot)
            _LOG.error(
                "[OOM_DUMP] tag=%s dump_id=%s sequence=%d device=%d pid=%d exception=%s "
                "file=%s snapshot=%s allocator_segments=%d allocator_blocks=%d\n"
                "[OOM_DUMP] torch allocator summary:\n%s",
                tag,
                correlation_id or "<automatic>",
                sequence,
                device,
                os.getpid(),
                exception or "<not provided>",
                output_path,
                snapshot_value,
                segment_count,
                block_count,
                summary_by_device[device],
            )
            return str(output_path)
        except Exception as error:  # noqa: BLE001
            _LOG.exception(
                "[OOM_DUMP] diagnostic collection failed for tag=%s dump_id=%s: %s",
                tag,
                correlation_id or "<automatic>",
                error,
            )
            return None
        finally:
            for temporary_path in temporary_paths:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass
            if output_dir is not None and marker_path is not None:
                try:
                    with _retention_lock(output_dir):
                        marker_path.unlink(missing_ok=True)
                        _prune_dump_files(output_dir)
                except OSError as cleanup_error:
                    _LOG.warning(
                        "[OOM_DUMP] failed to clear in-progress marker %s: %s",
                        marker_path,
                        cleanup_error,
                    )
            if marker_fd is not None:
                try:
                    fcntl.flock(marker_fd, fcntl.LOCK_UN)
                except OSError as cleanup_error:
                    _LOG.warning(
                        "[OOM_DUMP] failed to unlock in-progress marker: %s",
                        cleanup_error,
                    )
                try:
                    os.close(marker_fd)
                except OSError as cleanup_error:
                    _LOG.warning(
                        "[OOM_DUMP] failed to close in-progress marker: %s",
                        cleanup_error,
                    )


def _oom_observer(
    device: int, alloc_size: int, device_total: int, device_free: int
) -> None:
    """Torch allocator callback; preserve the existing one-shot behavior."""
    global _oom_fired, _last_observer_dump
    with _install_lock:
        if _oom_fired:
            return
        _oom_fired = True

    output_path = dump_oom_diagnostics(
        tag="torch_allocator_oom",
        device=device,
        alloc_size=alloc_size,
        device_total=device_total,
        device_free=device_free,
    )
    with _install_lock:
        _last_observer_dump = output_path
