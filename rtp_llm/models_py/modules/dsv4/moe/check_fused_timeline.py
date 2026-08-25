#!/usr/bin/env python3
"""Validate DSV4 MoE fused hot ranges in a Chrome/Perfetto timeline.

The checker is intentionally conservative: for each MoE record_function range
it reports the GPU kernel-name sequence whose timestamps fall inside that
range, and fails if any sequence contains known torch native kernels.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


HOT_RANGES = (
    "dsv4.moe.shared_expert_start",
    "dsv4.moe.shared_expert",
    "dsv4.moe.routed_experts",
    "dsv4.moe.shared_expert_finish",
    "dsv4.moe.add_shared",
)

BAD_KERNEL_RE = re.compile(
    r"(at::native|copy_kernel|vectorized_elementwise_kernel|"
    r"unrolled_elementwise_kernel|aten::|torch_)",
    re.IGNORECASE,
)


def _events(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        trace_events = payload.get("traceEvents", payload.get("events", []))
        if isinstance(trace_events, list):
            return [e for e in trace_events if isinstance(e, dict)]
    if isinstance(payload, list):
        return [e for e in payload if isinstance(e, dict)]
    raise ValueError("timeline JSON must be a Chrome trace dict/list")


def _complete_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        e
        for e in events
        if e.get("ph") == "X"
        and isinstance(e.get("name"), str)
        and isinstance(e.get("ts"), (int, float))
        and isinstance(e.get("dur"), (int, float))
    ]


def _is_kernel_event(event: dict[str, Any]) -> bool:
    cat = str(event.get("cat", "")).lower()
    return "kernel" in cat


def check_timeline(path: Path) -> int:
    events = _complete_events(_events(json.loads(path.read_text())))
    gpu_ranges = [
        e
        for e in events
        if e["name"] in HOT_RANGES
        and "gpu_user_annotation" in str(e.get("cat", "")).lower()
    ]
    ranges = gpu_ranges or [
        e
        for e in events
        if e["name"] in HOT_RANGES
        and "user_annotation" in str(e.get("cat", "")).lower()
    ]
    kernels = [e for e in events if _is_kernel_event(e)]
    if not ranges:
        print(f"no DSV4 MoE hot ranges found in {path}; skipped")
        return 0

    if gpu_ranges:
        print(f"using GPU annotation ranges in {path}")
    else:
        print(
            f"GPU annotation ranges not found in {path}; "
            "falling back to CPU annotation ranges",
            file=sys.stderr,
        )

    failures: list[tuple[str, str]] = []
    by_range: dict[str, list[list[str]]] = defaultdict(list)
    for range_event in sorted(ranges, key=lambda e: (e["ts"], e["name"])):
        start = float(range_event["ts"])
        end = start + float(range_event["dur"])
        seq = [
            str(k["name"])
            for k in kernels
            if start <= float(k["ts"]) and float(k["ts"]) < end
        ]
        by_range[str(range_event["name"])].append(seq)
        for name in seq:
            if BAD_KERNEL_RE.search(name):
                failures.append((str(range_event["name"]), name))

    for range_name in HOT_RANGES:
        seqs = by_range.get(range_name, [])
        print(f"\n[{range_name}] {len(seqs)} range(s)")
        for idx, seq in enumerate(seqs):
            compact: list[str] = []
            for name in seq:
                if not compact or compact[-1] != name:
                    compact.append(name)
            print(f"  #{idx}:")
            for name in compact:
                print(f"    {name}")

    if failures:
        print("\nForbidden torch-native kernels found:", file=sys.stderr)
        for range_name, kernel_name in failures:
            print(f"  {range_name}: {kernel_name}", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("timeline", type=Path)
    args = parser.parse_args()
    return check_timeline(args.timeline)


if __name__ == "__main__":
    raise SystemExit(main())
