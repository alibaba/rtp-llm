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

# Ring-buffer cap on the alloc/free event trace. Bounds RAM growth in
# long-running processes; per-block frames on live segments are stored
# separately and not capped, so "who is holding memory now" is unaffected
# when this fills. ~500K events ~= ~70 MB pickle, ~3 min of activity at
# the rate observed in dsv4 prefill smoke runs.
_MAX_TRACE_ENTRIES = 500_000


def _enabled() -> bool:
    return os.environ.get(_RECORD_ENV) == "1"


def _out_dir() -> Path:
    d = Path(_OUT_DIR)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _suffix(tag: str, device: int) -> str:
    return f"{tag}_d{device}_pid{os.getpid()}_t{int(time.time())}"


def install_oom_dump() -> None:
    """Enable allocator history recording and install the OOM observer.
    No-op unless RTP_OOM_RECORD=1. Idempotent; call once per rank process
    after CUDA is up."""
    if not _enabled():
        return

    global _installed
    with _lock:
        if _installed:
            return
        _installed = True

    torch.cuda.memory._record_memory_history(
        enabled="all",
        context="all",
        stacks="python",
        max_entries=_MAX_TRACE_ENTRIES,
    )
    torch._C._cuda_attach_out_of_memory_observer(_oom_observer)  # type: ignore[attr-defined]
    _LOG.info(
        "[OOM_DUMP] installed device=%d pid=%d dir=%s",
        torch.cuda.current_device(),
        os.getpid(),
        _out_dir(),
    )


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
