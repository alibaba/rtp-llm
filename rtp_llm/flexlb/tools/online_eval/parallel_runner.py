#!/usr/bin/env python3
"""FlexLB case-test parallel orchestrator (P0 lane parallelism + P1 case sharding).

The single runner (flexlb_functional_tests.py) walks its cases in one
serial for-loop: 99 cases ≈ 35-55 min wall on the remote dev container,
and the bottleneck is WAITING (batch drains, TTL windows, converge
windows), not CPU.  This orchestrator splits the work into N lanes and
runs each lane as an independent runner subprocess tree with explicitly
partitioned ports, so the lanes never touch each other's sockets or run
dirs.

Two sharding granularities (--shard):

  * category (P0, default) — the categories are LPT-packed into lanes.
    Measured ceiling: the heaviest FAMILY caps the wall (status 24 cases
    = 2104s of a 4918s serial run — 43%), so 4 lanes gave 2.01x, not 4x.
  * case (P1 flattening) — every CASE is LPT-packed onto the lanes from
    a measured per-case cost baseline (--timing-json = a prior full
    run's aggregate JSON; cases[].duration_ms).  Same-family cases
    deliberately spread across lanes; the wall tracks the balanced sum,
    not the heaviest family.  Each lane runs ONE runner invocation with
    the new --cases exact-name list.  Expected-fail probes participate
    as ordinary cases.  Without a timing baseline the split degenerates
    to uniform (round-robin) with a stderr warning; individual cases
    missing from the baseline fall back to the family per-case weight.

Isolation contract (why these knobs are enough):

  * master ports — FLEXLB_FT_MASTER_HTTP_PORT (http / mgmt=+1 /
    grpc=+2) and the HA Tier-1 A/B + Tier-3 port groups are ALL
    env-overridable per runner PROCESS (harness.py reads them at import
    time), so per-lane values give disjoint port groups.  Lane i owns
    [18080+10i .. 18089+10i] (Tier-1 A: +0..+2, B: +3..+5; Tier-3 and
    the single-master path share +0..+2 on distinct bind IPs).
  * mock ports — FLEXLB_FT_MOCK_BASE_GRPC_PORT pins the scan base per
    lane (default auto-scan from 55151 has a TOCTOU window when lanes
    scan concurrently).  Lane i owns [base .. base+~152]; stride 2000.
  * ZK helper — launches with --port 0 (auto-allocated); no lane
    partitioning needed (harness.py ZkHelperOps contract).
  * run dirs — the runner's new --run-root flag gives every lane its
    own tree; without it two lanes started in the same wall-clock
    second would merge their env<N>_<label> dirs and interleave logs.
  * cross-lane env passthrough — the lane env is os.environ OVERLAID
    with the port partition, so operator exports (e.g.
    FLEXLB_FT_HA_DUAL_MASTER=1 to arm the HA cases) still apply.
  * base offsets — FLEXLB_FT_PARALLEL_MASTER_BASE (default 18080) and
    FLEXLB_FT_PARALLEL_MOCK_BASE (default 55151) shift the whole
    partition matrix.  Shift both when the DEFAULT bands would collide
    with another flexlb_ft user on the same host (e.g. a concurrent
    serial run on the shared dev container); keep the strides intact so
    lanes stay disjoint.

Lane packing: LPT greedy over per-category cost weights (Tina's
family-time survey: master is heavy HA traffic windows, elastic pays
converge windows, admission pays serialized 3s batch drains, ...).
--parallel 1 degenerates to ONE lane running `--category all` — the
exact legacy serial path (including cross-category env reuse), so a
parallel-1 run is the equivalence smoke against a direct runner run.

The aggregated --json payload keeps the single-runner schema
(summary + cases[]) and adds a lanes[] block; per-case rows gain a
"lane" field.  summary.serial_case_time_s is the SUM of per-case
duration_ms — a lower bound on the serial wall (env builds/reuse are
not counted), so treat speedup vs the 35-55 min measured serial wall,
not vs serial_case_time_s, when reporting.

Usage:
    python3 parallel_runner.py                          # 4 lanes, defaults
    python3 parallel_runner.py --parallel 2 --json out.json
    python3 parallel_runner.py --parallel 1             # serial equivalence
    python3 parallel_runner.py --dry-run                # plan only
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from flexlb_ft.grade import overall_verdict  # noqa: E402

RUNNER = Path(__file__).resolve().parent / "flexlb_functional_tests.py"

# ---------------------------------------------------------------------------
# Cost model — per-category weight ≈ seconds-per-CASE (order-of-magnitude,
# from Tina's family-time survey of the 2026-09 baseline runs).  Lane
# packing multiplies by the LIVE per-category case count (queried from the
# runner's --list at plan time), so family weights stay correct as cases
# are added/removed.  Only the ranking matters for LPT; the printed "est"
# is a planning hint.
CATEGORY_WEIGHTS = {
    "master": 60,  # HA traffic windows 60-150s/case (legacy 3 + 5 gated HA)
    "elastic": 40,  # topology converge windows + background-flow runs
    "admission": 30,  # serialized 3s batch drains + park observation windows
    "engine_fault": 30,  # crash/restart + generation re-converge
    "priority": 30,  # preemption / yield windows (estimate — no baseline yet)
    "kv": 20,  # prime/evict churn + sync convergence settles
    "status": 15,  # TTL / 3-strike / generation windows
    "balance": 15,  # concurrent burst rounds + decode sampling
    "cancel": 12,  # stream lifecycle, mostly short
    "direct": 8,  # single fast-fail probe
}

# ---------------------------------------------------------------------------
# Port partition (per lane index i, 0-based):
#   master group   18080+10i .. 18089+10i  (http=+0 mgmt=+1 grpc=+2;
#                                            Tier-1 B=+3..+5)
#   mock base      55151+S*i .. +151       (scan window incl. victim zone)
# VERIFIED mock window width (harness.py _pick_base_grpc_port + start_victim,
# JavaMockEngineCluster.java: http control = base-1; engines = base ..
# base+nP+nD-1; victim zone = base+149..151): a lane occupies exactly
# [base-1 .. base+151] = 153 ports regardless of engine count.  So any
# stride >= 153 keeps lanes disjoint; the default 2000 is historical
# headroom, and --mock-stride 500 (3x window) is safe — it lifts the lane
# cap from 6 to 21 (port-range-wise; the dev container sustains 4-8).
MASTER_HTTP_BASE = 18080
MASTER_PORT_STRIDE = 10
MOCK_BASE_GRPC_PORT = 55151
MOCK_PORT_STRIDE = 2000
MOCK_PORT_WINDOW_LAST = 151  # lane footprint [base-1 .. base+151]


def max_lanes(mock_stride: int, mock_base: int) -> int:
    """Lane cap implied by the mock band: base+stride*(N-1)+151 <= 65535."""
    return (65535 - MOCK_PORT_WINDOW_LAST - mock_base) // mock_stride + 1


def _env_int(name: str, default: int) -> int:
    """Read a positive int env override (empty/absent → default)."""
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise SystemExit(f"error: {name}={raw!r} is not an integer")
    if value <= 0:
        raise SystemExit(f"error: {name}={raw!r} must be positive")
    return value


def _master_base() -> int:
    """Master-group base for lane 0 (FLEXLB_FT_PARALLEL_MASTER_BASE)."""
    # Range vs the lane count is validated in main() (stride-dependent).
    return _env_int("FLEXLB_FT_PARALLEL_MASTER_BASE", MASTER_HTTP_BASE)


def _mock_base() -> int:
    """Mock-port base for lane 0 (FLEXLB_FT_PARALLEL_MOCK_BASE)."""
    # Range vs the lane count is validated in main() (stride-dependent).
    return _env_int("FLEXLB_FT_PARALLEL_MOCK_BASE", MOCK_BASE_GRPC_PORT)


def lane_env(lane_idx: int, mock_stride: int = MOCK_PORT_STRIDE) -> dict[str, str]:
    """Port-partition env overlay for lane *lane_idx* (0-based).

    Every key below is process-global in harness.py (read at import), so a
    per-runner-subprocess value fully owns that lane's sockets.  Values NOT
    listed here (e.g. FLEXLB_FT_HA_DUAL_MASTER) pass through unchanged from
    the orchestrator's environment.
    """
    m = _master_base() + MASTER_PORT_STRIDE * lane_idx
    return {
        "FLEXLB_FT_MASTER_HTTP_PORT": str(m),
        "FLEXLB_FT_MASTER_MANAGEMENT_PORT": str(m + 1),
        "FLEXLB_FT_HA_MASTER_A_HTTP_PORT": str(m),
        "FLEXLB_FT_HA_MASTER_B_HTTP_PORT": str(m + 3),
        "FLEXLB_FT_HA_TIER3_MASTER_HTTP_PORT": str(m),
        "FLEXLB_FT_MOCK_BASE_GRPC_PORT": str(_mock_base() + mock_stride * lane_idx),
    }


def _mock_stride_of(args: argparse.Namespace) -> int:
    """Effective mock stride (CLI --mock-stride, else the 2000 default)."""
    return getattr(args, "mock_stride", None) or MOCK_PORT_STRIDE


def _list_rows(profile: str) -> list[list[str]]:
    """Whitespace-split token rows from the runner's --list output."""
    proc = subprocess.run(
        [sys.executable, str(RUNNER), "--list", "--profile", profile],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"runner --list failed (rc={proc.returncode}):\n{proc.stderr[-1000:]}"
        )
    return [line.split() for line in proc.stdout.splitlines()]


def category_case_counts(profile: str) -> dict[str, int]:
    """Live per-category case counts from the runner's --list output.

    The list is profile-filtered, so the family weights track the exact
    case set the lanes will run (e.g. the NON_BATCH-only cancel case
    drops out of the batch-window plan).  --list is pure registration
    (no jars booted), safe to call at plan time.
    """
    counts: dict[str, int] = {}
    for fields in _list_rows(profile):
        # Row format: NAME CATEGORY PROFILES ... — name/category are both
        # whitespace-free tokens, so fields[1] is the category even when a
        # long name overflows the 40-char column.
        if len(fields) >= 2 and fields[1] in CATEGORY_WEIGHTS:
            counts[fields[1]] = counts.get(fields[1], 0) + 1
    missing = [c for c in CATEGORY_WEIGHTS if c not in counts]
    if missing:
        raise RuntimeError(f"runner --list produced no rows for: {missing}")
    return counts


def list_case_pairs(profile: str) -> list[tuple[str, str]]:
    """Live (case name, category) pairs from the runner's --list output,
    in runner registration order (the P1 case-shard planning input)."""
    pairs: list[tuple[str, str]] = []
    for fields in _list_rows(profile):
        if len(fields) >= 2 and fields[1] in CATEGORY_WEIGHTS:
            pairs.append((fields[0], fields[1]))
    missing = [c for c in CATEGORY_WEIGHTS if c not in {cat for _, cat in pairs}]
    if missing:
        raise RuntimeError(f"runner --list produced no rows for: {missing}")
    return pairs


def load_timing_baseline(path: str) -> dict[str, float] | None:
    """case name -> measured seconds, from a prior run's JSON.

    Accepts the orchestrator aggregate schema or a bare per-lane runner
    JSON — both carry cases[].duration_ms.  Returns None when the path is
    unreadable/corrupt (the caller falls back to a uniform split).
    """
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    out: dict[str, float] = {}
    for row in payload.get("cases", []):
        name = row.get("name") if isinstance(row, dict) else None
        if not name:
            continue
        try:
            out[name] = float(row.get("duration_ms", 0)) / 1000.0
        except (TypeError, ValueError):
            continue
    return out


def family_weights(profile: str) -> dict[str, float]:
    """Per-category total cost = per-case seconds x live case count."""
    counts = category_case_counts(profile)
    return {
        cat: per_case * counts.get(cat, 0) for cat, per_case in CATEGORY_WEIGHTS.items()
    }


def plan_lanes(weights: dict[str, float], parallel: int) -> list[list[str]]:
    """LPT (longest-processing-time-first) greedy lane packing.

    Categories sorted by FAMILY weight desc, each dropped onto the
    currently lightest lane.  Deterministic (stable sort + lowest-index
    tie-break), balanced, and degrades to sensible layouts at any
    --parallel 1..N.
    """
    lanes: list[list[str]] = [[] for _ in range(parallel)]
    loads = [0.0] * parallel
    for cat in sorted(weights, key=lambda c: (weights[c], c), reverse=True):
        idx = min(range(parallel), key=lambda k: (loads[k], k))
        lanes[idx].append(cat)
        loads[idx] += weights[cat]
    return lanes


def plan_case_lanes(
    case_costs: list[tuple[str, float]], parallel: int
) -> list[list[str]]:
    """LPT greedy over per-CASE costs — the P1 flattening.

    Cases sorted by cost desc (name as the deterministic tie-break), each
    dropped onto the currently lightest lane — a heavy family's cases
    spread across lanes by construction (same-category cases MAY share a
    lane; that is fine, they are independent processes-wise).  Within a
    lane the runner-registration order is restored, so each lane executes
    its slice in the same order as the serial baseline (keeps per-case
    comparisons and log diffs readable).
    """
    lanes: list[list[str]] = [[] for _ in range(parallel)]
    loads = [0.0] * parallel
    ordered = sorted(case_costs, key=lambda nc: (-nc[1], nc[0]))
    for name, cost in ordered:
        idx = min(range(parallel), key=lambda k: (loads[k], k))
        lanes[idx].append(name)
        loads[idx] += cost
    rank = {name: i for i, (name, _) in enumerate(case_costs)}
    for lane in lanes:
        lane.sort(key=rank.__getitem__)
    return lanes


# ---------------------------------------------------------------------------
# Lane execution


class LaneResult:
    def __init__(
        self,
        lane_idx: int,
        categories: list[str],
        case_names: list[str] | None = None,
    ):
        self.lane_idx = lane_idx
        # Category mode: the lane's category list.  Case mode: the sorted
        # set of families the lane's cases belong to (informational).
        self.categories = list(categories)
        # Case mode only: the lane's exact case-name slice (recorded into
        # the aggregate JSON lanes[] block for reproducibility).
        self.case_names = list(case_names) if case_names is not None else None
        self.runs: list[tuple[str, int, Path]] = []  # (label, rc, json_path)
        self.wall_s = 0.0
        self.error: str | None = None


# Active runner subprocesses (for Ctrl-C teardown); guard with a lock
# because lane workers run in threads.
_active_procs: list[subprocess.Popen] = []
_active_lock = threading.Lock()


def _spawn(argv: list[str], env: dict, log_path: Path) -> int:
    """Start one runner subprocess, tee its output to *log_path*, wait."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "wb") as out:
        proc = subprocess.Popen(argv, stdout=out, stderr=subprocess.STDOUT, env=env)
    with _active_lock:
        _active_procs.append(proc)
    try:
        return proc.wait()
    finally:
        with _active_lock:
            if proc in _active_procs:
                _active_procs.remove(proc)


def run_lane(
    lane_idx: int,
    items: list[str],
    args: argparse.Namespace,
    out_dir: Path,
    run_stamp: str,
) -> LaneResult:
    """Run one lane.

    shard=category (P0): *items* are category names; sequential runner
    subprocesses, one per category.  Sequential within the lane (a lane
    is ONE port partition — two concurrent runners inside a lane would
    share it by construction); parallelism lives BETWEEN lanes.  A
    single-lane plan runs `--category all` instead: the exact legacy
    serial path, including cross-category env reuse.

    shard=case (P1): *items* are case names; ONE runner invocation with
    `--cases <comma list>` — the runner's exact-name filter (profile
    filtering still applies inside the runner).
    """
    if getattr(args, "shard", "category") == "case":
        return _run_case_lane(lane_idx, items, args, out_dir, run_stamp)
    categories = items
    result = LaneResult(lane_idx, categories)
    lane_dir = out_dir / f"lane{lane_idx}"
    lane_dir.mkdir(parents=True, exist_ok=True)
    run_root = Path(f"/tmp/flexlb_ft_p{run_stamp}_lane{lane_idx}")
    env = dict(os.environ)
    env.update(lane_env(lane_idx, _mock_stride_of(args)))

    # A single-lane plan arrives as ["all"]: one `--category all` runner —
    # the exact legacy serial path (cross-category env reuse intact).  A
    # multi-category lane runs one runner per category, SEQUENTIALLY: a
    # lane is one port partition by construction, so two concurrent
    # runners inside a lane would share it.
    t0 = time.monotonic()
    for cat in categories:
        json_path = lane_dir / f"{cat}.json"
        argv = [
            sys.executable,
            str(RUNNER),
            "--category",
            _runner_cli_name(cat),
            "--json",
            str(json_path),
            "--run-root",
            str(run_root),
            "--profile",
            args.profile,
            "--grade",
            args.grade,
        ]
        if args.keep:
            argv.append("--keep")
        rc = _spawn(argv, env, lane_dir / f"{cat}.log")
        result.runs.append((cat, rc, json_path))
        if rc != 0:
            # Keep going: the next category boots its own envs; one
            # crashed category must not sink the lane's remaining signal.
            print(
                f"[lane {lane_idx}] runner --category {cat} exited rc={rc} "
                f"(see {lane_dir / (cat + '.log')}); continuing lane",
                file=sys.stderr,
                flush=True,
            )
    result.wall_s = time.monotonic() - t0
    return result


def _run_case_lane(
    lane_idx: int,
    case_names: list[str],
    args: argparse.Namespace,
    out_dir: Path,
    run_stamp: str,
) -> LaneResult:
    """Run one case-shard lane: ONE runner invocation with --cases."""
    cat_of = dict(getattr(args, "case_pairs", None) or [])
    families = sorted({cat_of[n] for n in case_names if n in cat_of})
    result = LaneResult(lane_idx, families, case_names=case_names)
    lane_dir = out_dir / f"lane{lane_idx}"
    lane_dir.mkdir(parents=True, exist_ok=True)
    run_root = Path(f"/tmp/flexlb_ft_p{run_stamp}_lane{lane_idx}")
    env = dict(os.environ)
    env.update(lane_env(lane_idx, _mock_stride_of(args)))
    json_path = lane_dir / "cases.json"
    argv = [
        sys.executable,
        str(RUNNER),
        "--cases",
        ",".join(case_names),
        "--json",
        str(json_path),
        "--run-root",
        str(run_root),
        "--profile",
        args.profile,
        "--grade",
        args.grade,
    ]
    if args.keep:
        argv.append("--keep")
    t0 = time.monotonic()
    rc = _spawn(argv, env, lane_dir / "cases.log")
    result.runs.append(("cases", rc, json_path))
    if rc != 0:
        print(
            f"[lane {lane_idx}] runner --cases ({len(case_names)} cases) "
            f"exited rc={rc} (see {lane_dir / 'cases.log'})",
            file=sys.stderr,
            flush=True,
        )
    result.wall_s = time.monotonic() - t0
    return result


ALL_CATEGORIES = list(CATEGORY_WEIGHTS)  # canonical order (dict order)

# CLI kebab-case <-> python identifier (mirrors the runner's
# CATEGORY_ALIASES both ways: --categories engine-fault normalizes IN,
# and the spawn argv needs the runner's kebab-case choices OUT).
_CATEGORY_ALIASES = {"engine-fault": "engine_fault"}
_RUNNER_CLI_NAMES = {v: k for k, v in _CATEGORY_ALIASES.items()}


def _normalize_category(name: str) -> str:
    return _CATEGORY_ALIASES.get(name, name)


def _runner_cli_name(category: str) -> str:
    """Runner --category choices are kebab-case (engine-fault)."""
    return _RUNNER_CLI_NAMES.get(category, category)


def _plan(
    args: argparse.Namespace,
) -> tuple[list[list[str]], dict[str, float]]:
    """Lane plan over the requested work set.

    shard=category (P0): parallel=1 over the FULL set degenerates to ONE
    `--category all` runner — the exact legacy serial path (cross-category
    env reuse intact), the equivalence reference against a direct runner
    run.  A partial subset at parallel=1 stays per-category (a
    `--category all` runner would run the unrequested categories too).

    shard=case (P1): per-case LPT flattening from the --timing-json
    baseline; see _plan_case_shard.
    """
    if getattr(args, "shard", "category") == "case":
        return _plan_case_shard(args)
    requested = (
        [
            _normalize_category(c.strip())
            for c in args.categories.split(",")
            if c.strip()
        ]
        if args.categories
        else list(CATEGORY_WEIGHTS)
    )
    unknown = [c for c in requested if c not in CATEGORY_WEIGHTS]
    if unknown:
        raise SystemExit(
            f"unknown --categories entries {unknown}; valid: "
            f"{sorted(CATEGORY_WEIGHTS)}"
        )
    weights = {
        cat: w for cat, w in family_weights(args.profile).items() if cat in requested
    }
    if args.parallel == 1 and set(requested) == set(CATEGORY_WEIGHTS):
        return [["all"]], weights
    return plan_lanes(weights, args.parallel), weights


def _plan_case_shard(
    args: argparse.Namespace,
) -> tuple[list[list[str]], dict[str, float]]:
    """Per-case LPT plan (the P1 flattening).

    Cost source: --timing-json (a prior full run's aggregate JSON,
    cases[].duration_ms).  No baseline at all → uniform split (every case
    weighs 1, LPT degenerates to round-robin) with a stderr warning;
    individual cases missing from an otherwise usable baseline fall back
    to the family per-case weight.  --categories still bounds the case
    pool (family-level subset); same-family cases may land on different
    lanes — that is the point of the flattening.  Expected-fail probes
    participate as ordinary cases.
    """
    pairs = list_case_pairs(args.profile)
    if args.categories:
        requested = {
            _normalize_category(c.strip())
            for c in args.categories.split(",")
            if c.strip()
        }
        unknown = sorted(requested - set(CATEGORY_WEIGHTS))
        if unknown:
            raise SystemExit(
                f"unknown --categories entries {unknown}; valid: "
                f"{sorted(CATEGORY_WEIGHTS)}"
            )
        pairs = [pc for pc in pairs if pc[1] in requested]
    # Stash for run_lane / _print_plan (lane family breakdown) without
    # changing the (lanes, weights) return contract.
    args.case_pairs = pairs

    timing_path = getattr(args, "timing_json", None)
    timing = load_timing_baseline(timing_path) if timing_path else None
    if timing is None:
        if timing_path:
            print(
                f"warning: --timing-json unreadable ({timing_path}); "
                "falling back to uniform case split",
                file=sys.stderr,
            )
            args.cost_source = f"uniform (baseline unreadable: {timing_path})"
        else:
            print(
                "warning: --shard case without --timing-json — falling "
                "back to uniform case split",
                file=sys.stderr,
            )
            args.cost_source = "uniform (no baseline)"
    else:
        args.cost_source = (
            f"baseline {timing_path} "
            f"({sum(1 for name, _ in pairs if name in timing)}/{len(pairs)} cases)"
        )

    costs: list[tuple[str, float]] = []
    fallback: list[str] = []
    for name, cat in pairs:
        if timing is None:
            costs.append((name, 1.0))  # uniform split
        elif name in timing:
            costs.append((name, timing[name]))
        else:
            costs.append((name, CATEGORY_WEIGHTS[cat]))
            fallback.append(name)
    if timing is not None and fallback:
        preview = ", ".join(fallback[:10])
        more = f" ... (+{len(fallback) - 10} more)" if len(fallback) > 10 else ""
        print(
            f"warning: {len(fallback)} case(s) missing from the timing "
            f"baseline — family-weight fallback: {preview}{more}",
            file=sys.stderr,
        )

    weights = dict(costs)
    if not costs:
        return [[] for _ in range(args.parallel)], weights
    return plan_case_lanes(costs, args.parallel), weights


# ---------------------------------------------------------------------------
# Aggregation


def _load_runner_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def aggregate(
    lane_results: list[LaneResult],
    args: argparse.Namespace,
    wall_s: float,
) -> dict:
    """Merge per-lane runner JSONs into the single-runner schema + lanes."""
    all_cases: list[dict] = []
    counts = {
        "total": 0,
        "passed": 0,
        "failed": 0,
        "finding_confirmed": 0,
        "finding_resolved": 0,
    }
    serial_case_ms = 0
    achieved: list[str] = []  # normal graded cases only (verdict roll-up)
    lanes_block = []
    any_rc_fail = False

    for lr in sorted(lane_results, key=lambda r: r.lane_idx):
        lane_rows = 0
        lane_rcs: dict[str, int] = {}
        for cat, rc, json_path in lr.runs:
            lane_rcs[cat] = rc
            if rc != 0:
                any_rc_fail = True
            payload = _load_runner_json(json_path)
            if payload is None:
                # Runner died before writing JSON (crash / Ctrl-C): the
                # missing rows are accounted by the lane rc, not faked.
                continue
            # Counts come from the runner's own summary (single source of
            # truth for the four-way classification); case rows are merged
            # verbatim below.
            summary = payload.get("summary", {})
            for key in counts:
                counts[key] += int(summary.get(key, 0))
            for row in payload.get("cases", []):
                row["lane"] = lr.lane_idx
                all_cases.append(row)
                lane_rows += 1
                serial_case_ms += int(row.get("duration_ms", 0))
                if not row.get("expected_fail") and row.get("grade"):
                    achieved.append(row["grade"].get("achieved"))
        lanes_block.append(
            {
                "lane": lr.lane_idx,
                "categories": lr.categories,
                "exit_codes": lane_rcs,
                "cases": lane_rows,
                "wall_time_s": round(lr.wall_s, 1),
                "error": lr.error,
                # Case mode: the exact case-name slice (reproducibility —
                # the shard matrix is part of the record, not just the count).
                **({"case_names": lr.case_names} if lr.case_names is not None else {}),
            }
        )

    exit_code = 1 if (any_rc_fail or counts["failed"] > 0) else 0
    return {
        "summary": {
            "total": counts["total"],
            "passed": counts["passed"],
            "failed": counts["failed"],
            "finding_confirmed": counts["finding_confirmed"],
            "finding_resolved": counts["finding_resolved"],
            "verdict": overall_verdict([a for a in achieved if a]),
            "exit_code": exit_code,
            "parallel": args.parallel,
            "shard": getattr(args, "shard", "category"),
            "profile": args.profile,
            "grade": args.grade,
            "wall_time_s": round(wall_s, 1),
            "serial_case_time_s": round(serial_case_ms / 1000.0, 1),
        },
        "lanes": lanes_block,
        "cases": all_cases,
    }


# ---------------------------------------------------------------------------
# CLI


def _print_plan(
    lanes: list[list[str]], weights: dict[str, float], args: argparse.Namespace
) -> None:
    shard = getattr(args, "shard", "category")
    print(f"== FlexLB case tests — parallel orchestration (shard={shard}) ==")
    print(
        f"parallel={args.parallel} profile={args.profile} grade={args.grade}"
        f" out_dir={args.out_dir}"
    )
    mock_stride = _mock_stride_of(args)
    if shard == "case":
        cat_of = dict(getattr(args, "case_pairs", None) or [])
        src = getattr(
            args, "cost_source", f"baseline {getattr(args, 'timing_json', None)}"
        )
        print(
            f"case plan (LPT greedy over per-case seconds; cost source: {src};"
            " same-family cases spread across lanes):"
        )
        for i, lane in enumerate(lanes):
            est = sum(weights.get(n, 0.0) for n in lane)
            fam: dict[str, int] = {}
            for n in lane:
                if n in cat_of:
                    fam[cat_of[n]] = fam.get(cat_of[n], 0) + 1
            fam_s = " ".join(f"{c}x{k}" for c, k in sorted(fam.items()))
            print(
                f"  lane {i}: {len(lane)} cases   est ~{est:.0f}s"
                + (f"   [{fam_s}]" if fam_s else "")
            )
            if lane:
                print(f"    {', '.join(lane)}")
    else:
        print("lane plan (LPT greedy; est = planning weights, not measurements):")
        for i, lane in enumerate(lanes):
            est = sum(weights.get(c, 0.0) for c in lane)
            cats = " ".join(lane)
            print(
                f"  lane {i}: {cats}"
                + (f"   (est ~{est:.0f}s)" if est else "   (serial all-in-one)")
            )
    print("port partition (master group / mock base per lane):")
    master_base = _master_base()
    mock_base = _mock_base()
    for i in range(len(lanes)):
        m = master_base + MASTER_PORT_STRIDE * i
        mock = mock_base + mock_stride * i
        print(
            f"  lane {i}: master {m}-{m + 5} (http={m} mgmt={m + 1} "
            f"grpc={m + 2}; Tier-1 B={m + 3}-{m + 5}), mock {mock - 1}-"
            f"{mock + MOCK_PORT_WINDOW_LAST} (base {mock}, stride {mock_stride})"
        )
    print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="FlexLB case-test parallel orchestrator (lane parallelism)"
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=4,
        help=(
            "lane count (default 4; 1 degenerates to the legacy serial "
            "`--category all` path in category mode; cap derived from "
            "--mock-stride — 6 at the default 2000, 21 at 500)"
        ),
    )
    parser.add_argument(
        "--profile", default="batch-window", help="passed through to the runner"
    )
    parser.add_argument(
        "--grade", default="normal", help="passed through to the runner"
    )
    parser.add_argument(
        "--json",
        default=None,
        help="aggregated JSON path (default <out-dir>/aggregate.json)",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="lane artifacts dir (default /tmp/flexlb_ft_parallel_<ts>)",
    )
    parser.add_argument(
        "--categories",
        default=None,
        help=(
            "comma-separated category subset to orchestrate (default: all "
            "ten; CLI kebab-case, e.g. engine-fault is accepted as-is)"
        ),
    )
    parser.add_argument(
        "--shard",
        choices=["category", "case"],
        default="category",
        help=(
            "sharding granularity: category (P0 lane packing — the heaviest "
            "family caps the wall) or case (P1 flattening — per-case LPT "
            "from the --timing-json baseline; same-family cases spread "
            "across lanes)"
        ),
    )
    parser.add_argument(
        "--timing-json",
        default=None,
        help=(
            "case-mode cost baseline: a prior full run's aggregate JSON "
            "(cases[].duration_ms). Missing file → uniform split; cases "
            "absent from the baseline → family-weight fallback (both warn "
            "on stderr)"
        ),
    )
    parser.add_argument(
        "--mock-stride",
        type=int,
        default=None,
        help=(
            "mock-port stride between lanes (default 2000). VERIFIED "
            "per-lane mock window: [base-1 .. base+151] = 153 ports "
            "(harness _pick_base_grpc_port / start_victim; JavaMockEngine-"
            "Cluster http=base-1, engines=base..base+n-1, victim zone "
            "base+149..151), so 500 is safe with ~3x headroom and lifts "
            "the lane cap to 8+"
        ),
    )
    parser.add_argument(
        "--keep", action="store_true", help="passed through to the runner"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the lane plan + port matrix and exit (no execution)",
    )
    args = parser.parse_args()

    # Mock stride: CLI --mock-stride over the 2000 default.  The verified
    # per-lane mock footprint is [base-1 .. base+151] (153 ports), so any
    # stride >= 153 keeps lanes disjoint; reject below that outright.
    mock_stride = args.mock_stride if args.mock_stride is not None else MOCK_PORT_STRIDE
    args.mock_stride = mock_stride
    if mock_stride <= MOCK_PORT_WINDOW_LAST + 1:
        parser.error(
            f"--mock-stride must be >= {MOCK_PORT_WINDOW_LAST + 2} "
            f"(verified per-lane mock window = {MOCK_PORT_WINDOW_LAST + 2} "
            "ports: [base-1 .. base+151])"
        )

    # Lane cap is DERIVED from the stride (not the old fixed 6): the last
    # lane's mock window must stay under 65535; master groups (stride 10)
    # only bind for the truly absurd bases.
    mbase = _mock_base()
    mabase = _master_base()
    cap = min(
        max_lanes(mock_stride, mbase),
        (65535 - 5 - mabase) // MASTER_PORT_STRIDE + 1,
    )
    if not 1 <= args.parallel <= cap:
        parser.error(
            f"--parallel must be 1..{cap} (mock stride {mock_stride}, "
            f"mock base {mbase}, master base {mabase})"
        )

    run_stamp = str(int(time.time()))
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else Path(f"/tmp/flexlb_ft_parallel_{run_stamp}")
    )
    args.out_dir = str(out_dir)
    json_path = Path(args.json) if args.json else out_dir / "aggregate.json"

    lanes, weights = _plan(args)
    _print_plan(lanes, weights, args)
    if args.dry_run:
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"lanes started: {len(lanes)} runner subprocess trees → {out_dir}")
    t0 = time.monotonic()
    try:
        with ThreadPoolExecutor(max_workers=len(lanes)) as ex:
            futures = [
                ex.submit(run_lane, i, lane, args, out_dir, run_stamp)
                for i, lane in enumerate(lanes)
            ]
            lane_results = [f.result() for f in futures]
    except KeyboardInterrupt:
        with _active_lock:
            procs = list(_active_procs)
        print(
            f"\nCtrl-C: terminating {len(procs)} active runner subprocess(es) ...",
            file=sys.stderr,
        )
        for p in procs:
            try:
                p.terminate()
            except OSError:
                pass
        deadline = time.monotonic() + 10.0
        for p in procs:
            try:
                p.wait(timeout=max(0.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                try:
                    p.kill()
                except OSError:
                    pass
        return 130

    wall_s = time.monotonic() - t0
    payload = aggregate(lane_results, args, wall_s)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    s = payload["summary"]
    print(f"\n{'=' * 60}")
    print(
        f" Parallel results: {s['passed']} PASS / {s['failed']} FAIL / "
        f"{s['finding_confirmed']} finding-confirmed / "
        f"{s['finding_resolved']} finding-resolved / {s['total']} total"
    )
    print(
        f" Lanes: {s['parallel']} | wall {s['wall_time_s']}s "
        f"| sum(case time) {s['serial_case_time_s']}s (lower-bound serial)"
    )
    for lane in payload["lanes"]:
        rcs = ", ".join(f"{c}={rc}" for c, rc in lane["exit_codes"].items())
        print(
            f"   lane {lane['lane']}: {lane['cases']} cases in "
            f"{lane['wall_time_s']}s  [{rcs}]"
        )
    if s["verdict"] is not None:
        print(f" Overall grade: {s['verdict']}")
    print(f" JSON: {json_path}")
    print(f"{'=' * 60}\n")
    return s["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
