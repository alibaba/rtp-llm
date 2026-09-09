#!/usr/bin/env python3
"""Parallel execution of YAML-configured Python cases with owned port windows."""
from __future__ import annotations

import argparse
import fcntl
import os
import re
import sys
from pathlib import Path
from typing import IO

from flexlb_test_framework.harness import PROBE_BIND_HOST, port_in_use
from flexlb_test_framework.resource_plan import (
    MOCK_WINDOW_LAST,
    child_port_env,
    port_intervals,
)

MASTER_HTTP_BASE = 18080
MASTER_PORT_STRIDE = 10
MOCK_BASE_GRPC_PORT = 55151
MOCK_PORT_STRIDE = 500
MOCK_PORT_WINDOW_LAST = MOCK_WINDOW_LAST  # lane footprint [base-1 .. base+151]

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
    return child_port_env(
        _master_base() + MASTER_PORT_STRIDE * lane_idx,
        _mock_base() + mock_stride * lane_idx,
    )


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
    intervals = port_intervals(
        master_base + MASTER_PORT_STRIDE * lane_idx, mock_base + mock_stride * lane_idx
    )
    return [port for _, lo, hi in intervals for port in range(lo, hi + 1)]


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
    lanes = [
        port_intervals(
            master_base + MASTER_PORT_STRIDE * i, mock_base + mock_stride * i
        )
        for i in range(n_lanes)
    ]
    return [
        PORT_WINDOW_LOCK_DIR / f"{prefix}{lo}_{hi}.lock"
        for side_index, prefix in [(0, "m"), (1, "g")]
        for intervals in lanes
        for _, lo, hi in [intervals[side_index]]
    ]


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
        "--grade",
        default="normal",
        choices=("strict", "normal", "loose"),
        help="passed through to the runner",
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
        choices=["case"],
        default="case",
        help="partition by compiled instance",
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
        "--dry-run",
        action="store_true",
        help="print the lane plan + port matrix and exit (no execution)",
    )
    parser.add_argument(
        "--source",
        choices=["yaml"],
        default="yaml",
        help="instance source (default yaml: Python programs configured by YAML)",
    )
    parser.add_argument(
        "--case-dir",
        default=None,
        help="configuration file/directory (default: bundled scenarios/) for Python case configurations",
    )
    parser.add_argument(
        "--instances",
        default=None,
        help="exact compiled instance IDs for compiled Python cases",
    )
    args = parser.parse_args()
    if args.case_dir is None:
        args.case_dir = str(Path(__file__).resolve().parent / "scenarios")
    elif not args.case_dir:
        parser.error("--case-dir must not be empty")

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

    from flexlb_test_framework.instance_runner import run_structured

    return run_structured(args, sys.modules[__name__])


if __name__ == "__main__":
    sys.exit(main())
