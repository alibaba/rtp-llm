"""
BlockTreeCache GPU benchmark case registry.

Single source of truth for all smoke and profile benchmark cases.
Each case has complete canonical parameters (no binary defaults).

Profile matrix design (14 cases):
- concurrency is fixed at 8 for all transfer cases (swa was raised from 4
  to 8 so full_context/swa compare fairly at equal worker count),
  no longer expanded as a dimension
- all transfer cases build the real group-set layout from the model profile
  (one multi-layer device pool per member group, e.g. full_context = 3 pools /
  91 copy tiles, swa = 3 pools / 121 tiles), so results reflect the production
  plan shape instead of a single contiguous copy
- the host pair (Device<->Host) runs each group set twice, once with the
  batched-copy path and once with the staged SM copy path, for a direct
  strategy comparison
- transfer directions are mixed per medium pair in one case:
    host pair:      d2h + h2d
    device-disk:    d2disk + disk2d
    host-disk:      h2disk + disk2h
- duration floor enforced by the binary via pilot calibration
  (--min-measured-seconds, default 30)
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class BenchmarkCase:
    """A single benchmark case with complete canonical parameters."""

    name: str
    suite: str  # "smoke" or "profile"
    subcommand: str  # "tree" or "transfer"
    params: Dict[str, str] = field(default_factory=dict)
    description: str = ""
    is_representative_perf: bool = (
        False  # Whether this case gets perf collection in full suite mode
    )
    min_logical_bytes: int = 0  # Minimum cumulative logical bytes for transfer cases
    requires_disk: bool = False


# ---------------------------------------------------------------------------
# Smoke cases
# ---------------------------------------------------------------------------

SMOKE_CASES = [
    BenchmarkCase(
        name="smoke_tree_mini",
        suite="smoke",
        subcommand="tree",
        params={
            "--payload-mode": "scaled",
            "--tree-node-count": "64",
            "--max-path-length": "64",
            "--tree-branching-factor": "4",
            "--initial-min-path-length": "16",
            "--initial-max-path-length": "32",
            "--continuation-ratio": "0.7",
            "--fork-ratio": "0.2",
            "--fork-reuse-min-ratio": "0.25",
            "--fork-reuse-max-ratio": "0.9",
            "--hot-path-ratio": "0.2",
            "--active-path-limit": "64",
            "--append-length": "8",
            "--inserts-per-match": "4",
            "--operation-trace-count": "1000",
            "--steady-threads": "1",
            "--warmup-seconds": "2",
            "--min-measured-seconds": "5",
        },
        description="Minimal stateful tree smoke: match then four incremental inserts, 5s window",
    ),
    BenchmarkCase(
        name="smoke_transfer_d2h_mini",
        suite="smoke",
        subcommand="transfer",
        params={
            "--group-set": "full_context",
            "--transfer-directions": "d2h",
            "--transfer-operation-count": "64",
            "--transfer-concurrency": "1",
            "--host-memory": "pinned",
            "--min-measured-seconds": "5",
        },
        description="Minimal D2H transfer smoke test: 64 operations, full_context, pinned",
    ),
]


# ---------------------------------------------------------------------------
# Profile cases
# ---------------------------------------------------------------------------

_GIB = 1024 * 1024 * 1024


def _transfer_case(
    name,
    directions,
    group_set,
    concurrency,
    disk_mode=None,
    min_bytes=0,
    is_perf_rep=False,
    working_set_blocks=0,
    copy_strategy=None,
    description="",
):
    params = {
        "--group-set": group_set,
        "--transfer-directions": directions,
        "--transfer-operation-count": "4096",
        "--transfer-concurrency": str(concurrency),
        "--host-memory": "pinned",
    }
    if copy_strategy is not None:
        params["--copy-strategy"] = copy_strategy
    if disk_mode is not None:
        params["--disk-io-mode"] = disk_mode
        params["--disk-access-pattern"] = "sequential"
    if working_set_blocks > 0:
        params["--working-set-blocks"] = str(working_set_blocks)
    return BenchmarkCase(
        name=name,
        suite="profile",
        subcommand="transfer",
        params=params,
        min_logical_bytes=min_bytes,
        requires_disk=disk_mode is not None,
        is_representative_perf=is_perf_rep,
        description=description,
    )


PROFILE_CASES = [
    # --- stateful tree stress ---
    BenchmarkCase(
        name="tree_stress_100k",
        suite="profile",
        subcommand="tree",
        params={
            "--payload-mode": "scaled",
            "--tree-node-count": "100000",
            "--max-path-length": "1000",
            "--tree-branching-factor": "16",
            "--initial-min-path-length": "128",
            "--initial-max-path-length": "768",
            "--continuation-ratio": "0.7",
            "--fork-ratio": "0.2",
            "--fork-reuse-min-ratio": "0.25",
            "--fork-reuse-max-ratio": "0.9",
            "--hot-path-ratio": "0.2",
            "--active-path-limit": "4096",
            "--append-length": "32",
            "--inserts-per-match": "4",
            "--operation-trace-count": "20000",
            "--steady-threads": "8",
            "--warmup-seconds": "10",
            "--min-measured-seconds": "30",
        },
        description="100k stateful paths: continuation/fork/cold match then 4x32-node appends, 8 workers, 30s window",
        is_representative_perf=True,
    ),
    # Single-thread baseline of the same steady workload (lock-contention
    # comparison against tree_stress_100k).
    BenchmarkCase(
        name="tree_stress_100k_single",
        suite="profile",
        subcommand="tree",
        params={
            "--payload-mode": "scaled",
            "--tree-node-count": "100000",
            "--max-path-length": "1000",
            "--tree-branching-factor": "16",
            "--initial-min-path-length": "128",
            "--initial-max-path-length": "768",
            "--continuation-ratio": "0.7",
            "--fork-ratio": "0.2",
            "--fork-reuse-min-ratio": "0.25",
            "--fork-reuse-max-ratio": "0.9",
            "--hot-path-ratio": "0.2",
            "--active-path-limit": "4096",
            "--append-length": "32",
            "--inserts-per-match": "4",
            "--operation-trace-count": "20000",
            "--steady-threads": "1",
            "--warmup-seconds": "10",
            "--min-measured-seconds": "30",
        },
        description="Single-thread baseline: same 100k steady workload with 1 worker",
        is_representative_perf=True,
    ),
    # --- host pair: Device <-> Host, mixed d2h+h2d in one case, one case per
    # copy strategy (batched API vs staged SM) for direct comparison ---
    _transfer_case(
        "transfer_device_host_full_context_batch",
        "d2h,h2d",
        "full_context",
        8,
        min_bytes=16 * _GIB,
        is_perf_rep=True,
        copy_strategy="batch",
        description="Host pair d2h+h2d mixed, full_context (3 pools/91 tiles, 0.73MB/coord), batched copy, 8 workers",
    ),
    _transfer_case(
        "transfer_device_host_full_context_staged_sm",
        "d2h,h2d",
        "full_context",
        8,
        min_bytes=16 * _GIB,
        is_perf_rep=True,
        copy_strategy="staged-sm",
        description="Host pair d2h+h2d mixed, full_context (3 pools/91 tiles, 0.73MB/coord), staged SM copy, 8 workers",
    ),
    _transfer_case(
        "transfer_device_host_swa_batch",
        "d2h,h2d",
        "swa",
        8,
        min_bytes=16 * _GIB,
        copy_strategy="batch",
        description="Host pair d2h+h2d mixed, swa (3 pools/121 tiles, 7.03MB/coord), batched copy, 8 workers",
    ),
    _transfer_case(
        "transfer_device_host_swa_staged_sm",
        "d2h,h2d",
        "swa",
        8,
        min_bytes=16 * _GIB,
        copy_strategy="staged-sm",
        description="Host pair d2h+h2d mixed, swa (3 pools/121 tiles, 7.03MB/coord), staged SM copy, 8 workers",
    ),
    # --- device-disk pair: Device <-> Disk, mixed d2disk+disk2d ---
    _transfer_case(
        "transfer_device_disk_full_context_direct",
        "d2disk,disk2d",
        "full_context",
        8,
        disk_mode="direct",
        min_bytes=8 * _GIB,
        is_perf_rep=True,
        description="Device-Disk pair d2disk+disk2d mixed, full_context, direct IO, 8 workers",
    ),
    _transfer_case(
        "transfer_device_disk_full_context_buffered",
        "d2disk,disk2d",
        "full_context",
        8,
        disk_mode="buffered",
        min_bytes=8 * _GIB,
        working_set_blocks=32768,
        description="Device-Disk pair d2disk+disk2d mixed, full_context, buffered, 8 workers",
    ),
    _transfer_case(
        "transfer_device_disk_swa_direct",
        "d2disk,disk2d",
        "swa",
        8,
        disk_mode="direct",
        min_bytes=8 * _GIB,
        description="Device-Disk pair d2disk+disk2d mixed, swa, direct, 8 workers",
    ),
    _transfer_case(
        "transfer_device_disk_swa_buffered",
        "d2disk,disk2d",
        "swa",
        8,
        disk_mode="buffered",
        min_bytes=8 * _GIB,
        working_set_blocks=4096,
        description="Device-Disk pair d2disk+disk2d mixed, swa, buffered, 8 workers",
    ),
    # --- host-disk pair: Host <-> Disk, mixed h2disk+disk2h ---
    _transfer_case(
        "transfer_host_disk_full_context_direct",
        "h2disk,disk2h",
        "full_context",
        8,
        disk_mode="direct",
        min_bytes=8 * _GIB,
        is_perf_rep=True,
        description="Host-Disk pair h2disk+disk2h mixed, full_context, direct IO, 8 workers",
    ),
    _transfer_case(
        "transfer_host_disk_full_context_buffered",
        "h2disk,disk2h",
        "full_context",
        8,
        disk_mode="buffered",
        min_bytes=8 * _GIB,
        working_set_blocks=32768,
        description="Host-Disk pair h2disk+disk2h mixed, full_context, buffered, 8 workers",
    ),
    _transfer_case(
        "transfer_host_disk_swa_direct",
        "h2disk,disk2h",
        "swa",
        8,
        disk_mode="direct",
        min_bytes=8 * _GIB,
        description="Host-Disk pair h2disk+disk2h mixed, swa, direct, 8 workers",
    ),
    _transfer_case(
        "transfer_host_disk_swa_buffered",
        "h2disk,disk2h",
        "swa",
        8,
        disk_mode="buffered",
        min_bytes=8 * _GIB,
        working_set_blocks=4096,
        description="Host-Disk pair h2disk+disk2h mixed, swa, buffered, 8 workers",
    ),
]


# ---------------------------------------------------------------------------
# Combined registry
# ---------------------------------------------------------------------------

ALL_CASES: List[BenchmarkCase] = SMOKE_CASES + PROFILE_CASES

CASE_REGISTRY: Dict[str, BenchmarkCase] = {c.name: c for c in ALL_CASES}


def get_suite_cases(suite: str) -> List[BenchmarkCase]:
    """Get all cases for a given suite."""
    if suite == "all":
        return ALL_CASES
    return [c for c in ALL_CASES if c.suite == suite]


def get_representative_perf_cases(suite: str) -> List[BenchmarkCase]:
    """Get the representative perf cases for a suite."""
    cases = get_suite_cases(suite)
    return [c for c in cases if c.is_representative_perf]


def compute_transfer_operation_count(
    case: BenchmarkCase,
    profile_payload_bytes: int,
    min_measured_seconds: int = 30,
) -> int:
    """Compute minimum transfer operations to meet min_logical_bytes."""
    if case.min_logical_bytes == 0 or profile_payload_bytes == 0:
        return int(case.params.get("--transfer-operation-count", "4096"))
    min_ops = math.ceil(case.min_logical_bytes / profile_payload_bytes)
    final_ops = 1
    while final_ops < min_ops:
        final_ops *= 2
    return final_ops
