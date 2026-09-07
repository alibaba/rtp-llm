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

  * case (DEFAULT) — every CASE is LPT-packed onto the lanes from a
    per-case cost baseline.  Same-family cases deliberately spread
    across lanes; the wall tracks the balanced sum, not the heaviest
    family.  Each lane runs ONE runner invocation with the --cases
    exact-name list.  Expected-fail probes participate as ordinary
    cases.  Without a timing baseline the split degenerates to uniform
    (round-robin); individual cases missing from the baseline fall
    back to the family per-case weight.
  * category — the categories are LPT-packed into lanes (the runner's
    original grouping; opt-in legacy mode).  Measured ceiling: the
    heaviest FAMILY caps the wall (status 24 cases = 2104s of a 4918s
    serial run — 43%), so 4 lanes gave 2.01x, not 4x.

Timing-baseline self-maintenance: every completed run (any shard mode,
    any subset) MERGES its per-case durations into a shared baseline
    file (default /tmp/flexlb_ft_timing_baseline.json, overridable via
    FLEXLB_FT_TIMING_BASELINE; atomic tmp+rename write).  case mode
    reads that file automatically when --timing-json is absent, so the
    SECOND full run onward is already cost-balanced; an explicit
    --timing-json still overrides.  Merge, not overwrite: a partial
    --categories run refreshes only the cases it ran.

Isolation contract (why these knobs are enough):

  * master ports — FLEXLB_FT_MASTER_HTTP_PORT (http / mgmt=+1 /
    grpc=+2) and the HA Tier-1 A/B port groups are ALL env-overridable
    per runner PROCESS (harness.py reads them at import time), so
    per-lane values give disjoint port groups.  Lane i owns
    [18080+10i .. 18089+10i] (Tier-1 A: +0..+2, B: +3..+5; the
    single-master path shares +0..+2).
  * mock ports — FLEXLB_FT_MOCK_BASE_GRPC_PORT pins the scan base per
    lane (default auto-scan from 55151 has a TOCTOU window when lanes
    scan concurrently).  Lane i owns [base-1 .. base+151]; stride 500
    keeps a 6-lane matrix (the full-suite typical --parallel 6)
    entirely below the 61000 stress band (see STRESS_BAND_FLOOR).
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
    python3 parallel_runner.py                          # 4 lanes, case sharding
    python3 parallel_runner.py --parallel 1             # serial equivalence
    python3 parallel_runner.py --shard category          # family-level lanes
    python3 parallel_runner.py --dry-run                # plan only
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import IO

sys.path.insert(0, str(Path(__file__).resolve().parent))

from flexlb_ft.grade import overall_verdict  # noqa: E402
from flexlb_ft.harness import PROBE_BIND_HOST, port_in_use  # noqa: E402

# (harness.py's import block is pure stdlib — no module-level grpc
# import — so this import has zero side effects.)

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
    # "direct" folded into "master" (dcae01e694) — no longer a runner
    # category; listing it here makes list_case_pairs fail the
    # no-rows-for-category check at plan time.
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
# stride >= 153 keeps lanes disjoint; the default is 500 (~3x window)
# because a --parallel 6 matrix (the full-suite typical lane count,
# NOT the --parallel default of 4) sits at 55150..57802 ENTIRELY
# below the 61000 stress/lease band (run_online_eval.sh pins its mock
# base at ${MOCK_BASE_GRPC_PORT:-61000} and the lease ledger hands out
# port windows from 61000 upward), and it lifts the lane cap from 6
# to 21 (port-range-wise; the dev container sustains 4-8).
MASTER_HTTP_BASE = 18080
MASTER_PORT_STRIDE = 10
MOCK_BASE_GRPC_PORT = 55151
MOCK_PORT_STRIDE = 500
MOCK_PORT_WINDOW_LAST = 151  # lane footprint [base-1 .. base+151]

# Stress/lease port band on the shared dev container: online_eval
# (run_online_eval.sh MOCK_BASE_GRPC_PORT:-61000) and the flexlb lease
# ledger (port_base=61000) both allocate from here upward, so FT port
# matrices must stay strictly below it.  Hard bound for auto-shift
# candidates; an EXPLICIT base crossing it only warns — a leased-but-
# not-yet-listening port is invisible to bind probing, so refusing
# would false-positive (the explicit pair is a contract, not a hint).
STRESS_BAND_FLOOR = 61000

# Machine-level port-window lock dir.  Deliberately NOT env-overridable:
# the lock only means mutual exclusion if every user on the host shares
# this one directory — a per-user override would silently break that.
PORT_WINDOW_LOCK_DIR = Path("/tmp/flexlb_ft_portlocks")

# Safety bound for the auto-shift candidate ladder: k = 0..32 (33
# candidates; a runaway loop with a degenerate stride would otherwise
# scan forever).
PORT_SCAN_MAX_SHIFTS = 32


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
        "FLEXLB_FT_MOCK_BASE_GRPC_PORT": str(_mock_base() + mock_stride * lane_idx),
    }


def _mock_stride_of(args: argparse.Namespace) -> int:
    """Effective mock stride (CLI --mock-stride, else the 500 default)."""
    return getattr(args, "mock_stride", None) or MOCK_PORT_STRIDE


# ---------------------------------------------------------------------------
# Port window preflight & machine-level window lock


# Holder for the flocks of the SELECTED window in a real (non-dry)
# run.  Module-level on purpose: an flock lives exactly as long as its
# open file description, and strong references here keep the file
# objects (and their fds) alive until interpreter exit — unreferenced
# holders would be garbage-collected, closing the fds and silently
# dropping the machine-level mutexes mid-run.
_WINDOW_LOCK_FILES: list[IO] | None = None
_WINDOW_LOCK_DIAGNOSTIC = ""


def _lane_ports(
    lane_idx: int, mock_stride: int, master_base: int, mock_base: int
) -> list[int]:
    """Every port lane *lane_idx* may ever bind — a fixed 159-port set.

    Master group [m .. m+5] (http/mgmt/grpc + Tier-1 B) plus the FULL
    mock window [base-1 .. base+151] (153 ports: mock http control,
    engine grpc range, victim zone), at the same offsets lane_env pins.
    The FULL window (not just the ports the current case set needs) is
    deliberate: dynamic add_engine cases grow the engine set to
    base+nP+nD-1 inside the window and the victim zone sits at
    base+149..151, so a narrower probe would let an elastic case boot
    straight into a port a foreign process just grabbed — the exact
    failure mode this preflight exists to prevent.
    """
    m = master_base + MASTER_PORT_STRIDE * lane_idx
    base = mock_base + mock_stride * lane_idx
    return list(range(m, m + 6)) + list(
        range(base - 1, base + MOCK_PORT_WINDOW_LAST + 1)
    )


def _busy_ports(ports: list[int]) -> list[int]:
    """Ports from *ports* already bound by someone (0.0.0.0 probe).

    Must probe the WILDCARD address, never loopback: harness.py records
    the real lesson — a 127.0.0.1 probe PASSED while a foreign process
    (invisible to ps/ss in the shared network namespace, unreachable by
    kill) held 0.0.0.0:55252, and the JVM died at startup.  Serial on
    purpose: measured ~12ms for 950 ports; thread-parallel probing is
    2-3x SLOWER here (GIL + syscall churn), so do not "optimize" this.
    """
    return [p for p in ports if port_in_use(p, PROBE_BIND_HOST)]


def _window_lock_paths(
    master_base: int, mock_base: int, mock_stride: int, n_lanes: int
) -> list[Path]:
    """One lifetime lock per lane and side, named by its port interval.

    Interval intersections are checked under the admission lock before
    these files are acquired. Master and mock ports share one namespace.
    """
    paths = [
        PORT_WINDOW_LOCK_DIR
        / (
            f"m{master_base + MASTER_PORT_STRIDE * i}_"
            f"{master_base + MASTER_PORT_STRIDE * i + 5}.lock"
        )
        for i in range(n_lanes)
    ]
    paths += [
        PORT_WINDOW_LOCK_DIR
        / (
            f"g{mock_base + mock_stride * i - 1}_"
            f"{mock_base + mock_stride * i + MOCK_PORT_WINDOW_LAST}.lock"
        )
        for i in range(n_lanes)
    ]
    return paths


def _close_window_locks(holders: list[IO] | None) -> None:
    """Close (release) every holder of a window-lock set; None-safe."""
    for holder in holders or []:
        try:
            holder.close()
        except OSError:
            pass


def _window_lock_note(
    master_base: int, mock_base: int, mock_stride: int, n_lanes: int
) -> str:
    """Explain the last failed reservation and list its requested windows."""
    names = [
        p.name for p in _window_lock_paths(master_base, mock_base, mock_stride, n_lanes)
    ]
    shown = ", ".join(names[:6])
    if len(names) > 6:
        shown += f" (+{len(names) - 6} more)"
    detail = _WINDOW_LOCK_DIAGNOSTIC or "conflicting or inaccessible window"
    return f"window lock unavailable: {detail} ({shown} in {PORT_WINDOW_LOCK_DIR})"


def _open_window_lock(path: Path, *, create: bool = True) -> IO:
    """Open a persistent lock inode, including another uid's existing file."""
    flags = os.O_RDWR | os.O_NOFOLLOW
    if create:
        try:
            fd = os.open(path, flags | os.O_CREAT | os.O_EXCL, 0o666)
        except FileExistsError:
            # Existing files must be opened without O_CREAT in sticky
            # directories on Linux with fs.protected_regular enabled.
            fd = os.open(path, flags)
    else:
        fd = os.open(path, flags)
    try:
        try:
            os.fchmod(fd, 0o666)
        except OSError:
            pass  # another uid owns the inode
        return os.fdopen(fd, "r+")
    except BaseException:
        os.close(fd)
        raise


def _lock_interval(path: Path) -> tuple[int, int] | None:
    match = re.fullmatch(r"[mg](\d+)_(\d+)\.lock", path.name)
    return (int(match[1]), int(match[2])) if match else None


def _try_window_lock(
    master_base: int, mock_base: int, mock_stride: int, n_lanes: int
) -> list[IO] | None:
    """Reserve the whole matrix before probing sockets or starting runners.

    A short admission flock serializes discovery and acquisition, including
    simultaneous first use of different overlapping interval filenames.
    Live intersecting intervals block admission regardless of their side or
    endpoints. Stale unlocked files do not reserve ports and are left intact.
    After admission only the per-window flocks remain held, until process
    exit. Any acquisition or filesystem failure releases partial ownership.
    """
    global _WINDOW_LOCK_DIAGNOSTIC
    _WINDOW_LOCK_DIAGNOSTIC = ""
    holders: list[IO] = []
    lock_path = PORT_WINDOW_LOCK_DIR
    try:
        PORT_WINDOW_LOCK_DIR.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(PORT_WINDOW_LOCK_DIR, 0o1777)
        except OSError:
            pass  # pre-existing directory owned by another uid
        paths = _window_lock_paths(master_base, mock_base, mock_stride, n_lanes)
        intervals = [_lock_interval(path) for path in paths]
        for i, (lo, hi) in enumerate(intervals):
            for j in range(i):
                start, end = intervals[j]
                if lo <= end and start <= hi:
                    _WINDOW_LOCK_DIAGNOSTIC = (
                        f"matrix overlaps itself: {paths[j].name} / {paths[i].name}"
                    )
                    return None
        lock_path = PORT_WINDOW_LOCK_DIR / ".admission.lock"
        with _open_window_lock(lock_path) as gate:
            fcntl.flock(gate, fcntl.LOCK_EX)
            for path in PORT_WINDOW_LOCK_DIR.iterdir():
                interval = _lock_interval(path)
                if interval is None:
                    continue
                lo, hi = interval
                if not any(lo <= end and start <= hi for start, end in intervals):
                    continue
                lock_path = path
                with _open_window_lock(path, create=False) as probe:
                    fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for path in paths:
                lock_path = path
                holder = _open_window_lock(path)
                holders.append(holder)
                fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for holder in holders:
            try:
                holder.seek(0)
                holder.truncate()
                holder.write(f"{os.getpid()} {' '.join(sys.argv[:3])}\n")
                holder.flush()
            except OSError:
                pass  # diagnostics only; flock ownership is authoritative
        return holders
    except OSError as exc:
        _close_window_locks(holders)
        _WINDOW_LOCK_DIAGNOSTIC = f"{lock_path.name}: {exc}"
        return None
    except BaseException:
        _close_window_locks(holders)
        raise


def _matrix_tail(mock_base: int, mock_stride: int, n_lanes: int) -> int:
    """Highest port the LAST lane's mock window touches."""
    return mock_base + mock_stride * (n_lanes - 1) + MOCK_PORT_WINDOW_LAST


def _resolve_port_bases(args: argparse.Namespace) -> None:
    """Port-window preflight: probe every lane, lock, and pick the bases.

    EXPLICIT mode — FLEXLB_FT_PARALLEL_MASTER_BASE or
    FLEXLB_FT_PARALLEL_MOCK_BASE (either one, non-empty) pins the pair
    as a CONTRACT: one candidate, never shifted (half contract, half
    auto would silently move ports an operator deliberately pinned).
    Busy or locked → fail fast with a per-lane diagnosis: the run must
    die BEFORE burning 120s-per-case timeouts, not after (a pinned-but-
    occupied port once took the whole lane matrix down this way).  A
    matrix tail crossing STRESS_BAND_FLOOR only WARNS — a leased-but-
    not-yet-listening stress-band port is invisible to bind probing, so
    refusing would false-positive.  --dry-run never exits here: it
    shows the per-lane status and warns that a real run fails fast.

    DEFAULT mode — candidate k shifts the whole matrix DOWN by P
    strides (master and mock together, uniform direction: shifting
    master UP would walk into the harness auto-hunt band 18080..18580,
    shifting mock UP would creep toward the 61000 stress band / 65535
    ceiling):  m_k = m0 - MASTER_PORT_STRIDE*P*k,  b_k = b0 - stride*P*k.
    Shifted candidates never overlap each other: master side
    10*P > 10*(P-1)+5 holds for every P >= 1; mock side
    stride*P > stride*(P-1)+153  <=>  stride >= 153 (the CLI floor).
    Gates per candidate, cheapest first: (a) unprivileged range;
    (b) matrix tail < STRESS_BAND_FLOOR (hard — the online-eval/lease
    band starts there); (c) the machine-level window locks; (d) every
    lane's full 159-port window free under a 0.0.0.0 bind probe.
    First survivor wins; k > 0 injects the shifted bases into
    os.environ (lane_env / _print_plan / aggregate follow them
    automatically) with a loud stderr warning.

    Stash on args: port_provenance ("default 18080/55151" | "auto
    18080/55151 -> 18020/52151 (default window busy)" | "explicit
    18300/55151") and lane_port_status ({lane: "FREE" | "BUSY(:18080,
    18082)"}).

    Invariant: main()'s lane-cap check runs BEFORE this function, on
    the INITIAL bases; shifting moves bases DOWN only, which can only
    RAISE the cap, so the validated cap stays valid for the shifted
    matrix.

    Fail-closed: a probe/lock INFRASTRUCTURE exception (not plain
    busyness — port_in_use *returning True* is the busy signal; note
    the distinction from port_in_use RAISING) exits instead of running:
    a false rejection costs minutes (the message names the exact
    lane/port), a false "all free" costs 120s x N hung cases — hours.
    --dry-run downgrades it to a warning and omits the status column.

    Window locks: taken before any lane subprocess starts and held
    until process exit (the module-level holder), so a second
    instance fails fast instead of trampling this one's sockets.
    Admission checks interval intersections while holding a shared gate
    (see _try_window_lock); any overlap with a live run blocks this
    matrix, including partial and cross-side overlaps.  A resolve
    that SystemExits holds no lock.  Dry-run side effects: only the
    /tmp lock dir plus inert lock files (probe-and-release — nothing
    is held past the call).
    """
    global _WINDOW_LOCK_FILES
    dry = bool(getattr(args, "dry_run", False))
    n_lanes = args.parallel
    stride = _mock_stride_of(args)
    explicit = any(
        (os.environ.get(key) or "").strip()
        for key in (
            "FLEXLB_FT_PARALLEL_MASTER_BASE",
            "FLEXLB_FT_PARALLEL_MOCK_BASE",
        )
    )
    m0, b0 = _master_base(), _mock_base()

    def _probe(m: int, b: int) -> dict[int, list[int]]:
        """Busy ports per lane for the (m, b) matrix (full lane map)."""
        return {
            lane: _busy_ports(_lane_ports(lane, stride, m, b))
            for lane in range(n_lanes)
        }

    def _fmt_taken(items: list) -> str:
        """First 5 entries comma-joined, (+N more) tail when capped."""
        text = ",".join(str(x) for x in items[:5])
        if len(items) > 5:
            text += f" (+{len(items) - 5} more)"
        return text

    def _busy_note(m: int, b: int, lane: int, taken: list[int]) -> str:
        head = m + MASTER_PORT_STRIDE * lane
        parts = []
        for side, in_master in (("master", True), ("mock", False)):
            ports = [p for p in taken if (head <= p < head + 6) == in_master]
            if ports:
                parts.append(f"{side} {_fmt_taken(ports)}")
        return f"lane {lane}: {' / '.join(parts)} BUSY"

    def _status_of(probe: dict[int, list[int]]) -> dict[int, str]:
        return {
            lane: (f"BUSY(:{_fmt_taken(taken)})" if taken else "FREE")
            for lane, taken in probe.items()
        }

    def _infra_fail(exc: Exception, where: str) -> None:
        if dry:
            print(
                f"warning: port-window preflight infrastructure failure "
                f"({where}: {exc!r}) — dry-run continues without the "
                "port status column; a real run fails fast here",
                file=sys.stderr,
            )
            return
        raise SystemExit(
            f"error: port-window preflight infrastructure failure "
            f'({where}: {exc!r}) — refusing to run: a false "all free" '
            "would burn 120s per case on port timeouts"
        )

    if explicit:
        args.port_provenance = f"explicit {m0}/{b0}"
        tail = _matrix_tail(b0, stride, n_lanes)
        if tail >= STRESS_BAND_FLOOR:
            print(
                f"warning: explicit mock matrix tail {tail} >= stress "
                f"band floor {STRESS_BAND_FLOOR} (mock base {b0}, stride "
                f"{stride}, {n_lanes} lanes) — crossing the online-eval "
                "/ lease port band; continuing because a leased-but-not-"
                "yet-listening port is invisible to bind probing "
                "(explicit base env is a contract)",
                file=sys.stderr,
            )
        if not (b0 - 1 >= 1024 and m0 >= 1024):
            detail = (
                f"base below the privileged floor 1024 (master {m0}, "
                f"mock window starts {b0 - 1})"
            )
            if dry:
                print(
                    f"warning: {detail} — dry-run continues, a real run "
                    "will fail-fast here (explicit base env is a "
                    "contract)",
                    file=sys.stderr,
                )
                return
            raise SystemExit(
                f"error: explicit port window unusable: {detail} "
                "(explicit base env is a contract, never shifted)"
            )
        holders = None
        try:
            holders = _try_window_lock(m0, b0, stride, n_lanes)
            probe = _probe(m0, b0)
        except Exception as exc:
            _close_window_locks(holders)
            _infra_fail(exc, f"explicit window probe (bases {m0}/{b0})")
            return
        busy = {lane: taken for lane, taken in probe.items() if taken}
        args.lane_port_status = _status_of(probe)
        if holders is None:
            detail = _window_lock_note(m0, b0, stride, n_lanes)
        elif busy:
            detail = "; ".join(
                _busy_note(m0, b0, lane, taken) for lane, taken in sorted(busy.items())
            )
        else:
            detail = None
        if detail is not None:
            _close_window_locks(holders)
            if dry:
                print(
                    f"warning: {detail} — dry-run continues, a real run "
                    "will fail-fast here (explicit base env is a "
                    "contract)",
                    file=sys.stderr,
                )
            else:
                raise SystemExit(
                    f"error: explicit port window unusable: {detail} "
                    "(explicit base env is a contract, never shifted)"
                )
            return
        if dry:
            _close_window_locks(holders)  # probe-and-release
        else:
            _WINDOW_LOCK_FILES = holders
        return

    # DEFAULT mode: walk the shift ladder until a candidate passes every
    # gate; the FIRST rejection detail feeds the auto-shift warning.
    rejections: list[str] = []
    first_detail: str | None = None
    for k in range(PORT_SCAN_MAX_SHIFTS + 1):
        m_k = m0 - MASTER_PORT_STRIDE * n_lanes * k
        b_k = b0 - stride * n_lanes * k
        tag = f"k={k} bases {m_k}/{b_k}"
        detail: str | None
        if not (b_k - 1 >= 1024 and m_k >= 1024):
            detail = "below the privileged floor 1024"
        elif _matrix_tail(b_k, stride, n_lanes) >= STRESS_BAND_FLOOR:
            detail = (
                f"matrix tail {_matrix_tail(b_k, stride, n_lanes)} "
                f"enters the stress band >= {STRESS_BAND_FLOOR}"
            )
        else:
            holders = _try_window_lock(m_k, b_k, stride, n_lanes)
            if holders is None:
                detail = _window_lock_note(m_k, b_k, stride, n_lanes)
            else:
                try:
                    probe = _probe(m_k, b_k)
                except Exception as exc:
                    _close_window_locks(holders)
                    _infra_fail(exc, f"probe of {tag}")
                    return
                busy = {lane: taken for lane, taken in probe.items() if taken}
                if busy:
                    _close_window_locks(holders)
                    detail = "; ".join(
                        _busy_note(m_k, b_k, lane, taken)
                        for lane, taken in sorted(busy.items())
                    )
                else:
                    # SELECTED — the only path through every gate.
                    args.lane_port_status = _status_of(probe)
                    if k == 0:
                        args.port_provenance = f"default {m0}/{b0}"
                    else:
                        os.environ["FLEXLB_FT_PARALLEL_MASTER_BASE"] = str(m_k)
                        os.environ["FLEXLB_FT_PARALLEL_MOCK_BASE"] = str(b_k)
                        args.port_provenance = (
                            f"auto {m0}/{b0} -> {m_k}/{b_k} " "(default window busy)"
                        )
                        print(
                            f"warning: default port window busy "
                            f"({first_detail}) — auto-shifting bases "
                            f"{m0}/{b0} -> {m_k}/{b_k}; pin "
                            "FLEXLB_FT_PARALLEL_MASTER_BASE / "
                            "FLEXLB_FT_PARALLEL_MOCK_BASE to make it a "
                            "contract",
                            file=sys.stderr,
                        )
                    if dry:
                        _close_window_locks(holders)  # probe-and-release
                    else:
                        _WINDOW_LOCK_FILES = holders
                    return
        rejections.append(f"{tag}: {detail}")
        if first_detail is None:
            first_detail = detail
    raise SystemExit(
        "error: no usable port window in "
        f"{PORT_SCAN_MAX_SHIFTS + 1} auto-shift candidates (k=0.."
        f"{PORT_SCAN_MAX_SHIFTS}) — "
        + "; ".join(rejections)
        + "; pin FLEXLB_FT_PARALLEL_MASTER_BASE / "
        "FLEXLB_FT_PARALLEL_MOCK_BASE to an explicitly free pair"
    )


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
    if not counts:
        # ZERO valid category rows at all (runner rc was 0): the runner's
        # registration drifted out of sync with CATEGORY_WEIGHTS, or the
        # profile is broken — hard failure.
        raise RuntimeError(
            f"runner --list produced no category rows for profile {profile!r}"
        )
    missing = [c for c in CATEGORY_WEIGHTS if c not in counts]
    if missing:
        # PARTIAL emptiness is legal: profiles legitimately drop whole
        # families (elastic/status are batch-window-only), so warn + skip
        # instead of aborting; family_weights scales such a family to 0.
        print(
            f"warning: profile {profile!r}: runner --list produced no rows "
            f"for {missing}; skipping these categories",
            file=sys.stderr,
        )
    return counts


def list_case_pairs(profile: str) -> list[tuple[str, str]]:
    """Live (case name, category) pairs from the runner's --list output,
    in runner registration order (the P1 case-shard planning input)."""
    pairs: list[tuple[str, str]] = []
    for fields in _list_rows(profile):
        if len(fields) >= 2 and fields[1] in CATEGORY_WEIGHTS:
            pairs.append((fields[0], fields[1]))
    if not pairs:
        # Zero valid category rows at all — registration drift / broken
        # profile; hard failure (mirrors category_case_counts).
        raise RuntimeError(
            f"runner --list produced no category rows for profile {profile!r}"
        )
    missing = [c for c in CATEGORY_WEIGHTS if c not in {cat for _, cat in pairs}]
    if missing:
        # Same semantics as category_case_counts: partially empty families
        # are legal under profile filtering — warn + skip, don't abort.
        print(
            f"warning: profile {profile!r}: runner --list produced no rows "
            f"for {missing}; skipping these categories",
            file=sys.stderr,
        )
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


# Shared per-case timing baseline (case-mode cost source when no
# --timing-json is given).  Lives OUTSIDE the repo (a cross-run,
# machine-local artifact, like the /tmp run roots); the env override
# exists so two operators sharing a host can keep separate baselines.
TIMING_BASELINE_PATH = Path("/tmp/flexlb_ft_timing_baseline.json")


def _default_timing_baseline() -> Path:
    """Shared baseline location (FLEXLB_FT_TIMING_BASELINE overrides)."""
    raw = os.environ.get("FLEXLB_FT_TIMING_BASELINE")
    return Path(raw) if raw else TIMING_BASELINE_PATH


def write_timing_baseline(payload: dict, path: Path | None = None) -> tuple[Path, int]:
    """Merge a finished run's per-case durations into the shared baseline.

    *payload* is the orchestrator aggregate (or a bare runner JSON —
    both carry cases[].duration_ms).  MERGE, not overwrite: entries for
    cases this run did not touch (a --categories subset, or a baseline
    from a fuller run) keep their previous measurements, so a small
    targeted run never destroys a full-run baseline.  Atomic tmp+rename
    write — a concurrent reader never sees a half-written file.  Returns
    (path, entry count) so callers can report what the baseline now holds.
    """
    target = path if path is not None else _default_timing_baseline()
    entries: dict[str, int] = {}  # name -> duration_ms
    if target.exists():
        try:
            old = json.loads(target.read_text(encoding="utf-8"))
            if isinstance(old, dict):
                for row in old.get("cases", []):
                    if isinstance(row, dict) and row.get("name"):
                        try:
                            entries[row["name"]] = int(row["duration_ms"])
                        except (KeyError, TypeError, ValueError):
                            continue
        except (OSError, ValueError):
            entries = {}  # corrupt baseline: start fresh from this run
    for row in payload.get("cases", []):
        if not (isinstance(row, dict) and row.get("name")):
            continue
        try:
            entries[row["name"]] = int(row["duration_ms"])
        except (KeyError, TypeError, ValueError):
            continue
    doc = {"cases": [{"name": n, "duration_ms": ms} for n, ms in entries.items()]}
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    tmp.replace(target)
    return target, len(entries)


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
    if args.categories and not requested:
        # "--categories ,,," parses to the empty set: weights would then
        # be {} and the run would silently exit 0 on N empty lanes —
        # reject before the runner --list subprocess (family_weights).
        raise SystemExit("error: --categories contained no non-empty entries")
    unknown = [c for c in requested if c not in CATEGORY_WEIGHTS]
    if unknown:
        raise SystemExit(
            f"unknown --categories entries {unknown}; valid: "
            f"{sorted(CATEGORY_WEIGHTS)}"
        )
    # w > 0: a family with zero live cases under this profile (warned
    # about in category_case_counts) takes no lane and spawns nothing.
    weights = {
        cat: w
        for cat, w in family_weights(args.profile).items()
        if cat in requested and w > 0
    }
    if requested and not weights:
        # Explicit --categories whose families are ALL empty here: an
        # empty lane plan would silently do nothing — fail loudly.
        raise SystemExit(
            f"error: profile {args.profile!r}: none of the requested "
            f"categories {sorted(requested)} have any case under this profile"
        )
    if args.parallel == 1 and set(requested) == set(CATEGORY_WEIGHTS):
        return [["all"]], weights
    return plan_lanes(weights, args.parallel), weights


def _plan_case_shard(
    args: argparse.Namespace,
) -> tuple[list[list[str]], dict[str, float]]:
    """Per-case LPT plan (the P1 flattening).

    Cost source for case mode: an explicit --timing-json (a prior run's
    aggregate JSON, cases[].duration_ms) always wins.  Without one, the
    shared self-maintained baseline (see write_timing_baseline) is read
    automatically: absent = first run, quiet uniform split; present but
    corrupt = stderr warning + uniform.  Individual cases missing from
    an otherwise usable baseline fall back to the family per-case
    weight.  No baseline at all → uniform split (every case weighs 1,
    LPT degenerates to round-robin).  --categories still bounds the case
    pool (family-level subset); same-family cases may land on different
    lanes — that is the point of the flattening.  Expected-fail probes
    participate as ordinary cases.
    """
    requested: set[str] | None = None
    if args.categories:
        requested = {
            _normalize_category(c.strip())
            for c in args.categories.split(",")
            if c.strip()
        }
        if not requested:
            # "--categories ,,," parses to the empty set: the pool filter
            # below would then hand `--cases ""` to the runner, whose
            # parser treats an empty value as "no filter" = run
            # EVERYTHING — reject before the runner --list subprocess.
            raise SystemExit("error: --categories contained no non-empty entries")
        unknown = sorted(requested - set(CATEGORY_WEIGHTS))
        if unknown:
            raise SystemExit(
                f"unknown --categories entries {unknown}; valid: "
                f"{sorted(CATEGORY_WEIGHTS)}"
            )
    pairs = list_case_pairs(args.profile)
    if requested:
        pairs = [pc for pc in pairs if pc[1] in requested]
        if not pairs:
            # All requested families are empty under this profile.  An
            # empty lane slice would pass `--cases ""` to the runner,
            # whose parser treats an empty value as "no filter" = run
            # EVERYTHING — fail loudly instead.
            raise SystemExit(
                f"error: profile {args.profile!r}: none of the requested "
                f"categories {sorted(requested)} have any case under this "
                "profile"
            )
    # Stash for run_lane / _print_plan (lane family breakdown) without
    # changing the (lanes, weights) return contract.
    args.case_pairs = pairs

    explicit = getattr(args, "timing_json", None)
    timing: dict[str, float] | None = None
    auto = False
    if explicit:
        timing_path = explicit
        timing = load_timing_baseline(timing_path)
        if timing is None:
            print(
                f"warning: --timing-json unreadable ({timing_path}); "
                "falling back to uniform case split",
                file=sys.stderr,
            )
            args.cost_source = f"uniform (baseline unreadable: {timing_path})"
    else:
        timing_path = str(_default_timing_baseline())
        auto = True
        if Path(timing_path).exists():
            timing = load_timing_baseline(timing_path)
            if timing is None:
                print(
                    f"warning: default timing baseline corrupt "
                    f"({timing_path}); falling back to uniform case split",
                    file=sys.stderr,
                )
                args.cost_source = f"uniform (baseline corrupt: {timing_path})"
        else:
            # First run on this host (no baseline yet): uniform split is
            # the expected state, not an error — stay quiet on stderr;
            # the plan print shows the cost source.
            args.cost_source = "uniform (no baseline yet — this run establishes it)"
    if timing is not None:
        covered = sum(1 for name, _ in pairs if name in timing)
        label = "auto baseline" if auto else "baseline"
        args.cost_source = f"{label} {timing_path} ({covered}/{len(pairs)} cases)"

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
            # Port bases actually in effect (env may have been shifted by
            # _resolve_port_bases); getattr keeps old Namespaces working.
            "master_base": _master_base(),
            "mock_base": _mock_base(),
            "port_provenance": getattr(args, "port_provenance", None),
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
    provenance = getattr(args, "port_provenance", None)
    if provenance:
        print(f"  bases: {provenance}")
    lane_status = getattr(args, "lane_port_status", None) or {}
    for i in range(len(lanes)):
        m = master_base + MASTER_PORT_STRIDE * i
        mock = mock_base + mock_stride * i
        tag = lane_status.get(i)
        print(
            f"  lane {i}: master {m}-{m + 5} (http={m} mgmt={m + 1} "
            f"grpc={m + 2}; Tier-1 B={m + 3}-{m + 5}), mock {mock - 1}-"
            f"{mock + MOCK_PORT_WINDOW_LAST} (base {mock}, stride {mock_stride})"
            + (f"  [{tag}]" if tag else "")
        )
    print()


# Process-lifetime flock holder: _acquire_run_lock stashes the open lock
# file object here so its fd (and the advisory lock) survives until the
# interpreter exits — no caller needs the handle afterwards, but an
# unreferenced file object would be GC'd, closing the fd and dropping the
# lock mid-run.
_RUN_LOCK_FILE = None


def _acquire_run_lock(out_dir: Path) -> None:
    """Exclusive lock on <out-dir>/.parallel_runner.lock.

    Two orchestrators sharing one --out-dir overwrite each other's
    lane*/cases.json and run IDENTICAL port windows (same master/mock
    bases per lane index) — one real double-run measured 39 false
    failures.  flock is released by the kernel at process exit; the lock
    file itself is left behind (harmless).
    """
    global _RUN_LOCK_FILE
    lock_path = out_dir / ".parallel_runner.lock"
    lock_file = open(lock_path, "w")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        lock_file.close()
        raise SystemExit(
            f"error: {lock_path} is held by another parallel_runner "
            "instance — a second orchestrator on the same --out-dir would "
            "overwrite lane artifacts and share port windows; rerun with a "
            "different --out-dir"
        )
    _RUN_LOCK_FILE = lock_file


def main() -> int:
    parser = argparse.ArgumentParser(
        description="FlexLB case-test parallel orchestrator (lane parallelism)"
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=4,
        help=(
            "lane count (default 4; 1 = single-lane serial run; cap "
            "derived from --mock-stride — 21 at the default 500, 6 at "
            "2000)"
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
            "nine; CLI kebab-case, e.g. engine-fault is accepted as-is)"
        ),
    )
    parser.add_argument(
        "--shard",
        choices=["category", "case"],
        default="case",
        help=(
            "sharding granularity: case (DEFAULT — per-case LPT from the "
            "timing baseline; same-family cases spread across lanes) or "
            "category (family-level lane packing — the heaviest family "
            "caps the wall; opt-in)"
        ),
    )
    parser.add_argument(
        "--timing-json",
        default=None,
        help=(
            "case-mode cost baseline: a prior run's aggregate JSON "
            "(cases[].duration_ms). Overrides the auto-maintained shared "
            "baseline (default /tmp/flexlb_ft_timing_baseline.json, env "
            "FLEXLB_FT_TIMING_BASELINE). Missing file → uniform split; "
            "cases absent from the baseline → family-weight fallback "
            "(both warn on stderr)"
        ),
    )
    parser.add_argument(
        "--mock-stride",
        type=int,
        default=None,
        help=(
            "mock-port stride between lanes (default 500). VERIFIED "
            "per-lane mock window: [base-1 .. base+151] = 153 ports "
            "(harness _pick_base_grpc_port / start_victim; JavaMockEngine-"
            "Cluster http=base-1, engines=base..base+n-1, victim zone "
            "base+149..151), so 500 keeps ~3x headroom and a lane cap "
            "of 21 (the dev container sustains 4-8)"
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

    # Mock stride: CLI --mock-stride over the 500 default.  The verified
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

    # Port-window preflight BEFORE any lane subprocess starts: the
    # machine-level window locks are taken here and held until process
    # exit, so a second instance of this orchestrator fails fast
    # instead of trampling this one's sockets.  A resolve that
    # SystemExits holds no lock.
    _resolve_port_bases(args)

    run_stamp = str(int(time.time()))
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else Path(f"/tmp/flexlb_ft_parallel_{run_stamp}")
    )
    args.out_dir = str(out_dir)
    json_path = Path(args.json) if args.json else out_dir / "aggregate.json"

    # Dry-run has released the window locks; it creates no out-dir and
    # takes no out-dir lock. Other paths grab the exclusive out-dir lock
    # BEFORE the first child
    # process — _plan itself launches `runner --list`, and a second
    # instance planning concurrently into the same out-dir is exactly the
    # collision the lock guards against (same lane ports, same
    # lane*/cases.json targets).
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        _acquire_run_lock(out_dir)
    lanes, weights = _plan(args)
    _print_plan(lanes, weights, args)
    if args.dry_run:
        return 0
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

    # Self-maintained timing baseline: every completed run (any shard
    # mode / subset) refreshes the shared per-case durations — the next
    # case-mode run reads them back automatically.  Dry-run never gets
    # here (early return above), so a plan-only invocation never writes.
    baseline_path, baseline_n = write_timing_baseline(payload)
    print(f" timing baseline updated: {baseline_path} ({baseline_n} cases)")

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
