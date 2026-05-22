"""DSV4 runtime memory profiler — identifies peak torch allocations.

Trigger: ``touch /tmp/dsv4_mem_profile``
The NEXT qualifying forward (non-warmup, T > 1024) captures a full
allocation snapshot with Python stack traces, then writes a sorted
report of runtime-only allocations (excluding weights/KV cache baseline).

Output: ``$DSV4_MEM_PROFILE_DIR/`` (default ``/tmp/dsv4_mem_profile_out/``)
  - ``snapshot.pickle`` — raw PyTorch memory snapshot (viewable via
    ``torch.cuda.memory._viz`` or https://pytorch.org/memory_viz)
  - ``report.txt`` — sorted runtime allocations at peak

Usage from forward_layers:
    from rtp_llm.models_py.modules.dsv4._mem_profiler import (
        mem_profile_should_capture, mem_profile_start, mem_profile_stop,
    )
"""

import logging
import os
import pickle
from typing import Dict, List, Optional, Tuple

import torch

_TRIGGER = "/tmp/dsv4_mem_profile"
_OUTPUT_DIR = os.environ.get("DSV4_MEM_PROFILE_DIR", "/tmp/dsv4_mem_profile_out")
_MIN_TOKENS = int(os.environ.get("DSV4_MEM_PROFILE_MIN_TOKENS", "1024"))

logger = logging.getLogger(__name__)


def mem_profile_should_capture() -> bool:
    return os.path.exists(_TRIGGER)


def mem_profile_start(device: torch.device) -> int:
    os.makedirs(_OUTPUT_DIR, exist_ok=True)
    try:
        os.remove(_TRIGGER)
    except OSError:
        pass
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    baseline = torch.cuda.memory_allocated(device)
    torch.cuda.memory._record_memory_history(
        enabled="all",
        context="all",
        stacks="python",
    )
    logger.info(
        "[DSV4_MEM_PROFILE] recording started on %s, baseline=%.3f GiB",
        device,
        baseline / 1024**3,
    )
    return baseline


def mem_profile_stop(device: torch.device, baseline: int) -> None:
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device)
    current = torch.cuda.memory_allocated(device)

    snapshot_path = os.path.join(_OUTPUT_DIR, "snapshot.pickle")
    try:
        torch.cuda.memory._dump_snapshot(snapshot_path)
    except Exception as e:
        logger.error("[DSV4_MEM_PROFILE] dump_snapshot failed: %s", e)
    torch.cuda.memory._record_memory_history(enabled=None)

    runtime_peak = peak - baseline
    logger.info(
        "[DSV4_MEM_PROFILE] done. total_peak=%.3f GiB, baseline=%.3f GiB, "
        "runtime_peak=%.3f GiB, current=%.3f GiB. Snapshot: %s",
        peak / 1024**3,
        baseline / 1024**3,
        runtime_peak / 1024**3,
        current / 1024**3,
        snapshot_path,
    )

    try:
        report = _analyze_snapshot(snapshot_path, baseline)
        report_path = os.path.join(_OUTPUT_DIR, "report.txt")
        with open(report_path, "w") as f:
            f.write(report)
        logger.info("[DSV4_MEM_PROFILE] report written to %s", report_path)
        print(report, flush=True)
    except Exception as e:
        logger.error("[DSV4_MEM_PROFILE] analysis failed: %s", e, exc_info=True)


def _format_size(nbytes: int) -> str:
    if nbytes >= 1024**3:
        return f"{nbytes / 1024**3:.3f} GiB"
    elif nbytes >= 1024**2:
        return f"{nbytes / 1024**2:.2f} MiB"
    elif nbytes >= 1024:
        return f"{nbytes / 1024:.1f} KiB"
    return f"{nbytes} B"


def _short_path(filename: str) -> str:
    if not filename:
        return "<unknown>"
    markers = ("rtp_llm/", "models_py/", "modules/")
    for m in markers:
        idx = filename.find(m)
        if idx >= 0:
            return filename[idx:]
    parts = filename.rsplit("/", 2)
    return "/".join(parts[-2:]) if len(parts) >= 2 else filename


def _stack_key(frames: List[Dict]) -> str:
    relevant = []
    for fr in frames[:8]:
        fn = fr.get("filename", "")
        if "torch/" in fn or "site-packages/" in fn or "_record_memory_history" in fn:
            continue
        relevant.append(f"{_short_path(fn)}:{fr.get('line', '?')}:{fr.get('name', '?')}")
        if len(relevant) >= 3:
            break
    return " <- ".join(relevant) if relevant else "<no user frames>"


def _analyze_snapshot(snapshot_path: str, baseline: int) -> str:
    with open(snapshot_path, "rb") as f:
        snapshot = pickle.load(f)

    # PyTorch memory snapshots use 'segments' with per-block allocation info.
    # Each segment has 'blocks', each block has 'state', 'size', 'frames'.
    # Blocks with state='active_allocated' are currently live.
    # Blocks WITH frames were allocated during _record_memory_history (runtime);
    # blocks WITHOUT frames were allocated before recording started (baseline).
    segments = snapshot.get("segments", [])
    if not segments:
        return "No segments found in snapshot."

    all_blocks: List[Tuple[int, List[Dict]]] = []
    runtime_blocks: List[Tuple[int, List[Dict]]] = []
    total_active = 0

    for seg in segments:
        for blk in seg.get("blocks", []):
            if blk.get("state") != "active_allocated":
                continue
            size = blk.get("size", 0)
            frames = blk.get("frames", [])
            total_active += size
            all_blocks.append((size, frames))
            if frames:
                runtime_blocks.append((size, frames))

    runtime_blocks.sort(key=lambda x: x[0], reverse=True)
    runtime_total = sum(s for s, _ in runtime_blocks)

    # Group runtime allocations by call-stack signature
    groups: Dict[str, List[Tuple[int, List[Dict]]]] = {}
    for size, frames in runtime_blocks:
        key = _stack_key(frames)
        groups.setdefault(key, []).append((size, frames))

    # Sort groups by total size descending
    sorted_groups = sorted(
        groups.items(), key=lambda kv: sum(s for s, _ in kv[1]), reverse=True
    )

    lines = []
    lines.append("=" * 72)
    lines.append("DSV4 Runtime Memory Profile")
    lines.append("=" * 72)
    lines.append(f"Total active at dump: {_format_size(total_active)}")
    lines.append(f"Baseline (weights+KV+buffers): {_format_size(baseline)}")
    lines.append(f"Runtime with stacks (allocated during recording): {_format_size(runtime_total)}")
    lines.append(f"Number of runtime blocks (with stacks): {len(runtime_blocks)}")
    lines.append(f"Number of unique call sites: {len(sorted_groups)}")
    lines.append("")
    lines.append("Top allocations at peak (grouped by call site, sorted by total size):")
    lines.append("-" * 72)

    for rank, (key, allocs) in enumerate(sorted_groups[:80], 1):
        total = sum(s for s, _ in allocs)
        count = len(allocs)
        largest = max(s for s, _ in allocs)
        lines.append(
            f"#{rank:3d}  {_format_size(total):>12s}  "
            f"(count={count}, largest={_format_size(largest)})"
        )
        lines.append(f"      {key}")
        # Show full stack for top-20
        if rank <= 20 and allocs:
            _, sample_frames = allocs[0]
            for fr in sample_frames[:6]:
                fn = fr.get("filename", "")
                line = fr.get("line", "?")
                name = fr.get("name", "?")
                if "torch/" in fn or "site-packages/" in fn:
                    continue
                lines.append(f"        {_short_path(fn)}:{line} in {name}")
        lines.append("")

    lines.append("=" * 72)
    return "\n".join(lines)
