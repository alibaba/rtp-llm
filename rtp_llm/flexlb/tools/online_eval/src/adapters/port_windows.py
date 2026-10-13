"""Port-window admission, socket preflight and process-lifetime run locks."""
from __future__ import annotations

import fcntl
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO

from runtime.network import PROBE_BIND_HOST, port_in_use
from adapters.java_mock_resources import MOCK_WINDOW_LAST, child_port_env, port_intervals

MASTER_HTTP_BASE = 18080
MASTER_PORT_STRIDE = 10
MOCK_BASE_GRPC_PORT = 55151
MOCK_PORT_STRIDE = 500
MOCK_PORT_WINDOW_LAST = MOCK_WINDOW_LAST  # lane footprint [base-1 .. base+151]

# The FlexLB lease ledger allocates from 61000 upward. Auto-shift candidates
# stay below it; an explicit leased window only warns because a reserved port
# may not yet be listening. Keep allocation behavior independent of bind probes.
LEASE_BAND_FLOOR = 61000

# Machine-level port-window lock dir.  Deliberately NOT env-overridable:
# the lock only means mutual exclusion if every user on the host shares
# this one directory — a per-user override would silently break that.
PORT_WINDOW_LOCK_DIR = Path("/tmp/flexlb_ft_portlocks")

# Safety bound for the auto-shift candidate ladder: k = 0..32 (33
# candidates; a runaway loop with a degenerate stride would otherwise
# scan forever).
PORT_SCAN_MAX_SHIFTS = 32


class PortAdmissionError(ValueError):
    """No safe window or output ownership can be admitted."""


@dataclass
class PortAdmission:
    master_base: int
    mock_base: int
    port_provenance: str
    lane_port_status: dict[int, str] | None = None
    warnings: tuple[str, ...] = ()
    admitted: bool = True
    _holders: list[IO] = field(default_factory=list, repr=False)

    def close(self):
        _close_window_locks(self._holders)
        self._holders.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


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
        raise PortAdmissionError(f"error: {name}={raw!r} is not an integer")
    if value <= 0:
        raise PortAdmissionError(f"error: {name}={raw!r} must be positive")
    return value


def master_base() -> int:
    """Master-group base for lane 0 (FLEXLB_FT_PARALLEL_MASTER_BASE)."""
    # Range vs the lane count is validated in main() (stride-dependent).
    return _env_int("FLEXLB_FT_PARALLEL_MASTER_BASE", MASTER_HTTP_BASE)


def mock_base() -> int:
    """Mock-port base for lane 0 (FLEXLB_FT_PARALLEL_MOCK_BASE)."""
    # Range vs the lane count is validated in main() (stride-dependent).
    return _env_int("FLEXLB_FT_PARALLEL_MOCK_BASE", MOCK_BASE_GRPC_PORT)


def lane_env(lane_idx: int, mock_stride: int = MOCK_PORT_STRIDE,
             master: int | None = None, mock: int | None = None) -> dict[str, str]:
    """Port-partition env overlay for lane *lane_idx* (0-based).

    Every key below is process-global in adapters.environment_config (read at import), so a
    per-runner-subprocess value fully owns that lane's sockets.  Values NOT
    listed here (e.g. FLEXLB_FT_HA_DUAL_MASTER) pass through unchanged from
    the orchestrator's environment.
    """
    return child_port_env(
        (master_base() if master is None else master) + MASTER_PORT_STRIDE * lane_idx,
        (mock_base() if mock is None else mock) + mock_stride * lane_idx,
    )


_WINDOW_LOCK_DIAGNOSTIC = ""


def lane_ports(
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


def busy_ports(ports: list[int]) -> list[int]:
    """Ports from *ports* already bound by someone (0.0.0.0 probe).

    Must probe the WILDCARD address, never loopback: runtime.network records
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


def _fmt_taken(items):
    text = ",".join(str(x) for x in items[:5])
    return text + (f" (+{len(items) - 5} more)" if len(items) > 5 else "")


def _busy_note(master, lane, taken):
    head = master + MASTER_PORT_STRIDE * lane
    parts = []
    for side, in_master in (("master", True), ("mock", False)):
        ports = [p for p in taken if (head <= p < head + 6) == in_master]
        if ports:
            parts.append(f"{side} {_fmt_taken(ports)}")
    return f"lane {lane}: {' / '.join(parts)} BUSY"


def _probe_candidate(master, mock, stride, lanes):
    """Own reservation until it is admitted or closed, even on probe failure."""
    holders = _try_window_lock(master, mock, stride, lanes)
    if holders is None:
        return [], None, _window_lock_note(master, mock, stride, lanes)
    try:
        probe = {lane: busy_ports(lane_ports(lane, stride, master, mock))
                 for lane in range(lanes)}
        busy = {lane: ports for lane, ports in probe.items() if ports}
        detail = "; ".join(_busy_note(master, lane, ports) for lane, ports in sorted(busy.items()))
        if detail:
            _close_window_locks(holders)
            return [], probe, detail
        return holders, probe, None
    except BaseException:
        _close_window_locks(holders)
        raise


def _admission(master, mock, provenance, holders, probe=None, warnings=(), *,
               dry_run=False, admitted=True):
    status = ({lane: f"BUSY(:{_fmt_taken(ports)})" if ports else "FREE"
               for lane, ports in probe.items()} if probe is not None else None)
    if dry_run:
        _close_window_locks(holders)
        holders = []
    return PortAdmission(master, mock, provenance, status, tuple(warnings), admitted, holders)


def _infrastructure_failure(exc, where, master, mock, provenance, warnings, dry_run):
    detail = (f"port-window preflight infrastructure failure ({where}: {exc!r})")
    if not dry_run:
        raise PortAdmissionError(
            f"error: {detail} — refusing to run: a false all-free result would burn port timeouts"
        ) from exc
    return _admission(master, mock, provenance, [], warnings=[*warnings,
        f"warning: {detail} — dry-run continues without the port status column; a real run fails fast here"],
        admitted=False)


def admit_port_window(lanes: int, stride: int = MOCK_PORT_STRIDE, *,
                      master: int = MASTER_HTTP_BASE, mock: int = MOCK_BASE_GRPC_PORT,
                      explicit: bool = False, dry_run: bool = False) -> PortAdmission:
    """Reserve and probe one matrix, returning owned locks and diagnostics.

    Explicit bases never shift. Defaults walk downward together and stay
    below the lease band. Dry-run probes and releases; infrastructure errors
    never become successful observations. No args, environment or output is
    mutated, and callers close the result on every exit path.
    """
    if type(lanes) is not int or lanes < 1 or type(stride) is not int or stride <= MOCK_PORT_WINDOW_LAST + 1:
        raise PortAdmissionError("invalid lane count or mock stride")
    if type(master) is not int or type(mock) is not int:
        raise PortAdmissionError("port bases must be integers")
    provenance = f"{'explicit' if explicit else 'default'} {master}/{mock}"
    warnings = []
    if explicit:
        tail = _matrix_tail(mock, stride, lanes)
        if tail >= LEASE_BAND_FLOOR:
            warnings.append(f"warning: explicit mock matrix tail {tail} >= lease band floor {LEASE_BAND_FLOOR} "
                            f"(mock base {mock}, stride {stride}, {lanes} lanes) — continuing because "
                            "a leased-but-not-listening port is invisible to bind probing (explicit base env is a contract)")
        if mock - 1 < 1024 or master < 1024:
            detail = f"base below the privileged floor 1024 (master {master}, mock window starts {mock - 1})"
            if not dry_run:
                raise PortAdmissionError(f"error: explicit port window unusable: {detail} (explicit base env is a contract, never shifted)")
            return _admission(master, mock, provenance, [], warnings=[*warnings,
                f"warning: {detail} — dry-run continues, a real run will fail-fast here (explicit base env is a contract)"], admitted=False)
        try:
            holders, probe, detail = _probe_candidate(master, mock, stride, lanes)
        except Exception as exc:
            return _infrastructure_failure(exc, f"explicit window probe (bases {master}/{mock})",
                                           master, mock, provenance, warnings, dry_run)
        if detail:
            if not dry_run:
                raise PortAdmissionError(f"error: explicit port window unusable: {detail} (explicit base env is a contract, never shifted)")
            warnings.append(f"warning: {detail} — dry-run continues, a real run will fail-fast here (explicit base env is a contract)")
        return _admission(master, mock, provenance, holders, probe, warnings,
                          dry_run=dry_run, admitted=detail is None)

    rejections = []
    for shift in range(PORT_SCAN_MAX_SHIFTS + 1):
        m = master - MASTER_PORT_STRIDE * lanes * shift
        b = mock - stride * lanes * shift
        tag = f"k={shift} bases {m}/{b}"
        if b - 1 < 1024 or m < 1024:
            detail = "below the privileged floor 1024"
        elif _matrix_tail(b, stride, lanes) >= LEASE_BAND_FLOOR:
            detail = f"matrix tail {_matrix_tail(b, stride, lanes)} enters the lease band >= {LEASE_BAND_FLOOR}"
        else:
            try:
                holders, probe, detail = _probe_candidate(m, b, stride, lanes)
            except Exception as exc:
                return _infrastructure_failure(exc, f"probe of {tag}", master, mock,
                                               provenance, warnings, dry_run)
            if detail is None:
                if shift:
                    provenance = f"auto {master}/{mock} -> {m}/{b} (default window busy)"
                    warnings.append(f"warning: default port window busy ({rejections[0]}) — auto-shifting bases "
                                    f"{master}/{mock} -> {m}/{b}; pin FLEXLB_FT_PARALLEL_MASTER_BASE / "
                                    "FLEXLB_FT_PARALLEL_MOCK_BASE to make it a contract")
                return _admission(m, b, provenance, holders, probe, warnings, dry_run=dry_run)
        rejections.append(f"{tag}: {detail}")
    raise PortAdmissionError("error: no usable port window in "
        f"{PORT_SCAN_MAX_SHIFTS + 1} auto-shift candidates (k=0..{PORT_SCAN_MAX_SHIFTS}) — "
        + "; ".join(rejections) + "; pin FLEXLB_FT_PARALLEL_MASTER_BASE / "
        "FLEXLB_FT_PARALLEL_MOCK_BASE to an explicitly free pair")


def acquire_run_lock(out_dir: Path) -> IO:
    """Return explicit ownership of the output lock; close on every exit path."""
    lock_path = out_dir / ".parallel_runner.lock"
    holder = open(lock_path, "w")
    try:
        fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        holder.close()
        raise PortAdmissionError(f"error: {lock_path} is held by another parallel_runner "
            "instance — a second orchestrator on the same --out-dir would overwrite "
            "lane artifacts and share port windows; rerun with a different --out-dir") from exc
    return holder
