"""GPU OOM diagnostics for RTP-LLM.

Off by default. Set RTP_OOM_RECORD=1 to enable. When enabled, on the first
CUDA allocator OOM in a process, dumps two artifacts:

  1. OOM_MARKER_*.txt -- failed-alloc size, device totals, allocator
     statistics summary, and the Python stack at the call that triggered
     OOM.
  2. snap_*.pickle    -- torch.cuda.memory._dump_snapshot() output with
     full allocation history (every alloc / free event) and per-frame
     Python stacks. Viewable at https://pytorch.org/memory_viz to inspect
     the segment topology and a scrubbable timeline of who allocated what.

Cost when enabled: ~5-15% throughput, plus ~100MB pickle per OOM.

Each rank process must call install_oom_dump() once after CUDA is up.
Output filenames embed rank + pid + timestamp so multi-GPU runs do not
collide on a shared dir.

Output files are written to the current working directory.

Env knobs:
  RTP_OOM_RECORD=1   enable diagnostics (default off)
"""

# pyright: reportPrivateUsage=false
import json
import logging
import os
import threading
import time
import traceback
from pathlib import Path

import torch

_LOG = logging.getLogger(__name__)
_lock = threading.Lock()
_installed = False
_oom_fired = False

_RECORD_ENV = "RTP_OOM_RECORD"
_OUT_DIR = "."
_SNAPSHOT_CONTROL_ENV = "RTP_MEMORY_SNAPSHOT_CONTROL_DIR"
_SNAPSHOT_OUT_ENV = "RTP_MEMORY_SNAPSHOT_DIR"
_SNAPSHOT_RANKS_ENV = "RTP_MEMORY_SNAPSHOT_RANKS"
_SNAPSHOT_MAX_ENTRIES_ENV = "RTP_MEMORY_SNAPSHOT_MAX_ENTRIES"
_snapshot_thread: threading.Thread | None = None

# Ring-buffer cap on the alloc/free event trace. Bounds RAM growth in
# long-running processes; per-block frames on live segments are stored
# separately and not capped, so "who is holding memory now" is unaffected
# when this fills. ~500K events ~= ~70 MB pickle, ~3 min of activity at
# the rate observed in dsv4 prefill smoke runs.
_MAX_TRACE_ENTRIES = 500_000


def _enabled() -> bool:
    return os.environ.get(_RECORD_ENV) == "1"


def _out_dir() -> Path:
    d = Path(os.environ.get("RTP_OOM_OUT_DIR", _OUT_DIR))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _snapshot_requested() -> bool:
    return bool(os.environ.get(_SNAPSHOT_CONTROL_ENV))


def _snapshot_rank_enabled(device: int) -> bool:
    configured = os.environ.get(_SNAPSHOT_RANKS_ENV, "all").strip().lower()
    if configured in ("", "all", "*"):
        return True
    try:
        return device in {int(value.strip()) for value in configured.split(",")}
    except ValueError:
        _LOG.error(
            "[MEMORY_SNAPSHOT] invalid %s=%r; expected all or comma-separated local ranks",
            _SNAPSHOT_RANKS_ENV,
            configured,
        )
        return False


def _write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temporary.replace(path)


def _memory_snapshot_control_loop(device: int) -> None:
    """Dump an opt-in snapshot when the experiment writes arm then dump."""
    control_dir = Path(os.environ[_SNAPSHOT_CONTROL_ENV])
    output_dir = Path(os.environ.get(_SNAPSHOT_OUT_ENV, str(control_dir)))
    control_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    arm_path, dump_path = control_dir / "arm", control_dir / "dump"
    armed_path, done_path = (
        control_dir / f"armed_d{device}.json",
        control_dir / f"done_d{device}.json",
    )
    tag = os.environ.get("RTP_MEMORY_SNAPSHOT_TAG", "request").replace("/", "_")
    max_entries = int(os.environ.get(_SNAPSHOT_MAX_ENTRIES_ENV, _MAX_TRACE_ENTRIES))
    try:
        torch.cuda.set_device(device)
        last_attempt = ""
        while True:
            attempt = ""
            while not attempt or attempt == last_attempt:
                if arm_path.exists():
                    attempt = arm_path.read_text().strip() or str(arm_path.stat().st_mtime_ns)
                if not attempt or attempt == last_attempt:
                    time.sleep(0.05)
            baseline_allocated = int(torch.cuda.memory_allocated(device))
            baseline_reserved = int(torch.cuda.memory_reserved(device))
            torch.cuda.memory._record_memory_history(
                enabled="all", context="all", stacks="python",
                max_entries=max_entries, device=device, clear_history=True,
            )
            torch.cuda.reset_peak_memory_stats(device)
            _write_json(armed_path, {
                "attempt": attempt, "device": device, "pid": os.getpid(),
                "baseline_allocated_bytes": baseline_allocated,
                "baseline_reserved_bytes": baseline_reserved,
                "max_trace_entries": max_entries,
            })
            _LOG.info("[MEMORY_SNAPSHOT] armed attempt=%s device=%d", attempt, device)
            while not dump_path.exists() or (dump_path.read_text().strip() not in ("", attempt)):
                time.sleep(0.05)
            torch.cuda.synchronize(device)
            safe_attempt = "".join(c for c in attempt if c.isalnum() or c in "-_")[-32:]
            snapshot_path = output_dir / f"memory_snapshot_{tag}_a{safe_attempt}_d{device}_pid{os.getpid()}.pickle"
            torch.cuda.memory._dump_snapshot(str(snapshot_path))
            free_bytes, total_bytes = torch.cuda.mem_get_info(device)
            summary = {
                "attempt": attempt, "device": device, "pid": os.getpid(),
                "baseline_allocated_bytes": baseline_allocated,
                "baseline_reserved_bytes": baseline_reserved,
                "end_allocated_bytes": int(torch.cuda.memory_allocated(device)),
                "end_reserved_bytes": int(torch.cuda.memory_reserved(device)),
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                "device_free_bytes": int(free_bytes), "device_total_bytes": int(total_bytes),
                "snapshot": str(snapshot_path), "snapshot_bytes": snapshot_path.stat().st_size,
            }
            _write_json(output_dir / f"memory_summary_{tag}_a{safe_attempt}_d{device}_pid{os.getpid()}.json", summary)
            _write_json(done_path, summary)
            torch.cuda.memory._record_memory_history(enabled=None, device=device)
            last_attempt = attempt
            _LOG.info("[MEMORY_SNAPSHOT] dumped attempt=%s device=%d snapshot=%s", attempt, device, snapshot_path)
    except BaseException:
        _LOG.exception("[MEMORY_SNAPSHOT] controller failed device=%d", device)


def _suffix(tag: str, device: int) -> str:
    return f"{tag}_d{device}_pid{os.getpid()}_t{int(time.time())}"


def install_oom_dump() -> None:
    """Install opt-in OOM and successful-run allocator diagnostics."""
    oom_enabled = _enabled()
    snapshot_requested = _snapshot_requested()
    if not oom_enabled and not snapshot_requested:
        return

    global _installed
    with _lock:
        if _installed:
            return
        _installed = True

    device = torch.cuda.current_device()
    if oom_enabled:
        torch.cuda.memory._record_memory_history(
            enabled="all", context="all", stacks="python", max_entries=_MAX_TRACE_ENTRIES,
        )
        torch._C._cuda_attach_out_of_memory_observer(_oom_observer)  # type: ignore[attr-defined]
        _LOG.info("[OOM_DUMP] installed device=%d pid=%d dir=%s", device, os.getpid(), _out_dir())
    global _snapshot_thread
    if snapshot_requested and _snapshot_rank_enabled(device):
        _snapshot_thread = threading.Thread(
            target=_memory_snapshot_control_loop, args=(device,),
            name=f"memory-snapshot-d{device}", daemon=True,
        )
        _snapshot_thread.start()
        _LOG.info("[MEMORY_SNAPSHOT] controller installed device=%d control=%s", device, os.environ[_SNAPSHOT_CONTROL_ENV])


def _oom_observer(
    device: int, alloc_size: int, device_total: int, device_free: int
) -> None:
    """Allocator-thread callback. Writes marker + snapshot, returns so the
    original OOM exception can propagate. Fires at most once per process."""
    global _oom_fired
    with _lock:
        if _oom_fired:
            return
        _oom_fired = True

    suffix = _suffix("oom", device)
    out = _out_dir()
    marker = out / f"OOM_MARKER_{suffix}.txt"
    snap = out / f"snap_{suffix}.pickle"
    stack = "".join(traceback.format_stack())
    stats = torch.cuda.memory_stats(device=device)

    marker.write_text(
        f"device={device}\n"
        f"failed_alloc={alloc_size}\n"
        f"device_total={device_total}\n"
        f"device_free={device_free}\n"
        f"pid={os.getpid()}\n"
        f"time={time.time()}\n"
        f"allocated_bytes={int(stats.get('allocated_bytes.all.current', 0))}\n"
        f"reserved_bytes={int(stats.get('reserved_bytes.all.current', 0))}\n"
        f"inactive_split_bytes={int(stats.get('inactive_split_bytes.all.current', 0))}\n"
        f"num_alloc_retries={int(stats.get('num_alloc_retries', 0))}\n"
        f"num_ooms={int(stats.get('num_ooms', 0))}\n"
        f"stack=\n{stack}"
    )
    _LOG.error(
        "[OOM_DUMP] OOM device=%d failed_alloc=%d total=%d free=%d " "pid=%d marker=%s",
        device,
        alloc_size,
        device_total,
        device_free,
        os.getpid(),
        marker,
    )

    torch.cuda.memory._dump_snapshot(str(snap))
    _LOG.error("[OOM_DUMP] snapshot dumped: %s", snap)
