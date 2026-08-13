"""
BlockTreeCache GPU benchmark case registry.

Single source of truth for all smoke and profile benchmark cases.
Each case has complete canonical parameters (no binary defaults).

Tree profile design (1 representative case):
- `tree_online_high_variation_c32` is the ONLY active/representative Tree
  perf case: fixed online workload constants (C32 logical request contexts,
  ~20k initial nodes, 32,768-block device/host pools, 20 length buckets,
  13 hit-rate buckets, 100ms forward sleep, 15s warmup, 60s measured) with
  the shared task pool as the only knob (--task-pool-size=4).
- task-pool 4/8 formal comparisons are driver overrides of this same case;
  no task-pool-32 or load-pressure case is registered.

Transfer profile design (12 cases):
- concurrency and transfer-engine descriptor batch size are fixed at 8 for
  all transfer cases, so full_context/swa compare fairly at equal wave width;
  neither is expanded as a dimension
- all transfer cases build the real group-set layout from the model profile
  (one multi-layer device pool per member group, e.g. full_context = 3 pools /
  91 copy tiles, swa = 3 pools / 121 tiles), so results reflect the production
  plan shape instead of a single contiguous copy
- the host pair (Device<->Host) runs each group set twice, once with the CUDA
  batch-copy strategy and once with the staged SM copy strategy, for a direct
  strategy comparison
- transfer directions are mixed per medium pair in one case:
    host pair:      d2h + h2d
    device-disk:    d2disk + disk2d
    host-disk:      h2disk + disk2h
- duration floor enforced by the binary via pilot calibration
  (--min-measured-seconds, default 30)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class BenchmarkCase:
    """A single benchmark case with complete canonical parameters."""

    name: str
    suite: str  # "smoke" or "profile"
    subcommand: str  # "tree" or "transfer"
    params: Dict[str, str] = field(default_factory=dict)
    is_representative_perf: bool = (
        False  # Whether this case gets perf collection in full suite mode
    )
    min_logical_bytes: int = 0  # Minimum cumulative logical bytes for transfer cases
    requires_disk: bool = False
    # Native process timeout override. The online Tree case needs to cover
    # setup + 15s warmup + 60s measured + drain + profiler teardown.
    expected_process_timeout_seconds: Optional[int] = None


# ---------------------------------------------------------------------------
# Smoke cases
# ---------------------------------------------------------------------------

SMOKE_CASES = [
    BenchmarkCase(
        name="smoke_tree_online_mini",
        suite="smoke",
        subcommand="tree",
        params={
            "--task-pool-size": "4",
        },
    ),
    BenchmarkCase(
        name="smoke_transfer_d2h_mini",
        suite="smoke",
        subcommand="transfer",
        params={
            "--group-set": "full_context",
            "--transfer-directions": "d2h",
            "--transfer-operation-count": "64",
            "--transfer-concurrency": "2",
            "--transfer-descriptor-batch-size": "2",
            "--host-memory": "pinned",
            "--min-measured-seconds": "5",
        },
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
):
    params = {
        "--group-set": group_set,
        "--transfer-directions": directions,
        "--transfer-operation-count": "4096",
        "--transfer-concurrency": str(concurrency),
        "--transfer-descriptor-batch-size": str(concurrency),
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
    )


PROFILE_CASES = [
    # --- online scheduler lifecycle tree stress ---
    BenchmarkCase(
        name="tree_online_high_variation_c32",
        suite="profile",
        subcommand="tree",
        params={
            "--task-pool-size": "4",
        },
        is_representative_perf=True,
        expected_process_timeout_seconds=180,
    ),
    # --- host pair: Device <-> Host, mixed d2h+h2d in one case, one case per
    # Device<->Host copy strategy (CUDA batch copy vs staged SM) for direct
    # comparison. Both variants use descriptor batches at the transfer-engine
    # API boundary.
    _transfer_case(
        "transfer_device_host_full_context_batch",
        "d2h,h2d",
        "full_context",
        8,
        min_bytes=16 * _GIB,
        is_perf_rep=True,
        copy_strategy="batch",
    ),
    _transfer_case(
        "transfer_device_host_full_context_staged_sm",
        "d2h,h2d",
        "full_context",
        8,
        min_bytes=16 * _GIB,
        is_perf_rep=True,
        copy_strategy="staged-sm",
    ),
    _transfer_case(
        "transfer_device_host_swa_batch",
        "d2h,h2d",
        "swa",
        8,
        min_bytes=16 * _GIB,
        copy_strategy="batch",
    ),
    _transfer_case(
        "transfer_device_host_swa_staged_sm",
        "d2h,h2d",
        "swa",
        8,
        min_bytes=16 * _GIB,
        copy_strategy="staged-sm",
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
    ),
    _transfer_case(
        "transfer_device_disk_full_context_buffered",
        "d2disk,disk2d",
        "full_context",
        8,
        disk_mode="buffered",
        min_bytes=8 * _GIB,
        working_set_blocks=32768,
    ),
    _transfer_case(
        "transfer_device_disk_swa_direct",
        "d2disk,disk2d",
        "swa",
        8,
        disk_mode="direct",
        min_bytes=8 * _GIB,
    ),
    _transfer_case(
        "transfer_device_disk_swa_buffered",
        "d2disk,disk2d",
        "swa",
        8,
        disk_mode="buffered",
        min_bytes=8 * _GIB,
        working_set_blocks=4096,
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
    ),
    _transfer_case(
        "transfer_host_disk_full_context_buffered",
        "h2disk,disk2h",
        "full_context",
        8,
        disk_mode="buffered",
        min_bytes=8 * _GIB,
        working_set_blocks=32768,
    ),
    _transfer_case(
        "transfer_host_disk_swa_direct",
        "h2disk,disk2h",
        "swa",
        8,
        disk_mode="direct",
        min_bytes=8 * _GIB,
    ),
    _transfer_case(
        "transfer_host_disk_swa_buffered",
        "h2disk,disk2h",
        "swa",
        8,
        disk_mode="buffered",
        min_bytes=8 * _GIB,
        working_set_blocks=4096,
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
