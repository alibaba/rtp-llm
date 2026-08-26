"""Best-effort Torch CUDA allocator diagnostics for fatal OOMs.

Every dump writes a human-readable ``oom_allocator_*.log`` beside the normal
service logs. The file contains the original exception/backtrace, full
``torch.cuda.memory_summary``, and every allocator segment/block. Set
``RTP_OOM_RECORD=1`` before startup to additionally record Python/C++ allocation
stacks and emit a memory-viz snapshot.
"""

# pyright: reportPrivateUsage=false
import logging
import os
import pickle
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional, TextIO

import torch

_LOG = logging.getLogger(__name__)
_lock = threading.Lock()
_installed = False
_oom_fired = False

_RECORD_ENV = "RTP_OOM_RECORD"
_LOG_DIR_ENV = "LOG_PATH"
_OUT_DIR = "logs"
_MAX_TRACE_ENTRIES = 500_000

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


def _suffix(tag: str, device: int) -> str:
    world_rank = os.environ.get("WORLD_RANK", os.environ.get("RANK", "0"))
    server_id = os.environ.get("FRONTEND_SERVER_ID", "0")
    return (
        f"{tag}_r{world_rank}_s{server_id}_d{device}_pid{os.getpid()}_t{time.time_ns()}"
    )


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
    """Optionally record allocation history and attach the Torch OOM observer."""
    if not _enabled():
        return

    global _installed
    with _lock:
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
        "[OOM_DUMP] allocation history enabled device=%d pid=%d dir=%s stacks=all",
        torch.cuda.current_device(),
        os.getpid(),
        _out_dir(),
    )


def dump_oom_diagnostics(
    tag: str = "oom",
    device: Optional[int] = None,
    alloc_size: int = 0,
    device_total: Optional[int] = None,
    device_free: Optional[int] = None,
    exception: Optional[str] = None,
    cpp_backtrace: Optional[str] = None,
) -> None:
    """Write allocator state once without masking the OOM being diagnosed."""
    global _oom_fired
    with _lock:
        if _oom_fired:
            return
        _oom_fired = True

    try:
        if device is None:
            device = torch.cuda.current_device()

        if device_total is None or device_free is None:
            try:
                current_free, current_total = torch.cuda.mem_get_info(device)
                device_free = current_free if device_free is None else device_free
                device_total = current_total if device_total is None else device_total
            except Exception as error:  # noqa: BLE001 - diagnostics are best effort
                _LOG.exception("[OOM_DUMP] failed to read device memory: %s", error)

        try:
            stats = torch.cuda.memory_stats(device=device)
        except Exception as error:  # noqa: BLE001
            _LOG.exception("[OOM_DUMP] failed to read allocator stats: %s", error)
            stats = {}

        try:
            summary = torch.cuda.memory_summary(device=device, abbreviated=False)
        except Exception as error:  # noqa: BLE001
            _LOG.exception("[OOM_DUMP] failed to render allocator summary: %s", error)
            summary = f"<failed to render torch.cuda.memory_summary: {error}>"

        allocator_snapshot: Optional[Dict[str, Any]] = None
        snapshot_error: Optional[str] = None
        try:
            allocator_snapshot = torch.cuda.memory._snapshot()
        except Exception as error:  # noqa: BLE001
            snapshot_error = str(error)
            _LOG.exception("[OOM_DUMP] failed to collect allocator blocks: %s", error)

        suffix = _suffix(tag, device)
        output_dir = _out_dir()
        snapshot_path = output_dir / f"oom_allocator_snapshot_{suffix}.pickle"
        snapshot_value = "disabled; set RTP_OOM_RECORD=1 before startup"
        if _enabled() and allocator_snapshot is not None:
            try:
                with snapshot_path.open("wb") as snapshot_file:
                    pickle.dump(allocator_snapshot, snapshot_file)
                snapshot_value = str(snapshot_path)
            except Exception as error:  # noqa: BLE001
                _LOG.exception(
                    "[OOM_DUMP] failed to dump allocation snapshot: %s", error
                )
                snapshot_value = f"failed: {error}"

        output_path = output_dir / f"oom_allocator_{suffix}.log"
        with output_path.open("w", encoding="utf-8") as output:
            output.write("=" * 120 + "\n")
            output.write("RTP-LLM TORCH CUDA OOM DIAGNOSTICS\n")
            output.write("=" * 120 + "\n")
            output.write(f"tag={tag}\n")
            output.write(f"device={device}\n")
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
            output.write("\n[CUDA DEVICE MEMORY]\n")
            output.write(
                f"failed_alloc_bytes={alloc_size} ({_human_bytes(alloc_size)})\n"
            )
            output.write(
                f"device_total_bytes={device_total} ({_human_bytes(device_total)})\n"
            )
            output.write(
                f"device_free_bytes={device_free} ({_human_bytes(device_free)})\n"
            )
            output.write("\n[TORCH ALLOCATOR STATS - ALL KEYS]\n")
            for name in sorted(stats):
                value = int(stats[name])
                human = f" ({_human_bytes(value)})" if "bytes" in name else ""
                output.write(f"{name}={value}{human}\n")
            output.write("\n[TORCH MEMORY SUMMARY]\n")
            output.write(f"{summary}\n")
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
                "\n[DIAGNOSTIC PYTHON STACK - NOT THE ORIGINAL CUDA EXCEPTION STACK]\n"
            )
            output.write("".join(traceback.format_stack()))

        segment_count, block_count = _snapshot_counts(allocator_snapshot)
        _LOG.error(
            "[OOM_DUMP] tag=%s device=%d pid=%d exception=%s file=%s snapshot=%s "
            "allocator_segments=%d allocator_blocks=%d\n"
            "[OOM_DUMP] torch allocator summary:\n%s",
            tag,
            device,
            os.getpid(),
            exception or "<not provided>",
            output_path,
            snapshot_value,
            segment_count,
            block_count,
            summary,
        )
    except Exception as error:  # noqa: BLE001
        _LOG.exception(
            "[OOM_DUMP] diagnostic collection failed for tag=%s: %s", tag, error
        )


def _oom_observer(
    device: int, alloc_size: int, device_total: int, device_free: int
) -> None:
    dump_oom_diagnostics(
        tag="torch_allocator_oom",
        device=device,
        alloc_size=alloc_size,
        device_total=device_total,
        device_free=device_free,
    )
