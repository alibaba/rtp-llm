"""HA case-test support: dual-master specs, client traffic runner, row stats.

Shared plumbing for the six HA dual-master cases in cases/master.py
(brief: flexlb-ha-casetest-brief.md, p1-p10):

  * Tier-1 dual standalone — two flexlb masters on DISTINCT port groups
    (A: 18080/18081/18082, B: 18083/18084/18085), needConsistency stays
    off, no ZK: each master routes independently and the client's sticky
    target decides who serves.  Zero production prerequisites.
  * Tier-3 full-chain ZK — same port group on distinct loopback IPs
    (127.0.0.1 vs 127.0.0.2) + FLEXLB_ADVERTISED_IP + the ZK helper JVM
    (FLEXLB_SYNC_CONSISTENCY_CONFIG).  RULING (2026-09-02, harness.py):
    this same-host distinct-IP layout is DEAD (localIp has no env
    override, wildcard bind, SELF_TARGET) — Tier-3 moves to the
    phase-2 dual-container topology; the 127.0.0.1/.2 wiring is kept
    only as the env-injection contract reference.  Tier-2 forwarding
    semantics are owned by the JUnit layer (master_forward_matrix),
    not this harness.

Deferred (documented cut): the eval collector's per-master/per-route
dimensions — the HA assertion surface is already covered by
client_events.jsonl's master_target/route_path dual dimensions, and
run_online_eval.sh keeps a single master, so the pressure-test line is
unaffected.

client_events.jsonl row contract (JavaLoadClient HA mode, delivered):
  route_path    "master" | "fallback" | "failed"   (value-domain extension)
  master_target actually-served / last-attempted flexlb gRPC address
  failover      bool — same-request retry switch happened
  error_kind    "none" | "transport" | "business" | "deadline"
  status        "ok" | "scheduled" | "schedule_error" | "exception" | ...
"""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Optional

from ..engine_ops import EngineOps
from ..harness import (
    HA_TIER1_MASTER_A_HTTP_PORT,
    HA_TIER1_MASTER_B_HTTP_PORT,
    HA_TIER3_MASTER_B_BIND_IP,
    HA_TIER3_MASTER_HTTP_PORT,
    ClientOps,
    EnvSpec,
    MasterSpec,
    default_perf,
)

# ---------------------------------------------------------------------------
# Env switches (all HA behaviour is gated — the legacy single-master path
# never reads these)
# ---------------------------------------------------------------------------

# "1" switches master_kill to its per-master generalized (Tier-1 dual)
# branch; default (unset) keeps the historical single-master flow.
HA_DUAL_MASTER_ENV = "FLEXLB_FT_HA_DUAL_MASTER"


def ha_dual_enabled() -> bool:
    return os.environ.get(HA_DUAL_MASTER_ENV, "").strip() == "1"


# Layout selector for the ZK-tier cases (master_ha_failover,
# failback_wraparound): "tier3" (default — brief p5/p6/p9/p10 attribution)
# or "tier1" (immediate-smoke fallback: same client-side assertion surface,
# dual-standalone servers).  Tier-3 needs the production-side prerequisites
# (see module docstring) — flip to tier1 to validate the harness plumbing
# and the client failover contract before they land.
HA_LAYOUT_ENV = "FLEXLB_FT_HA_LAYOUT"


def ha_layout() -> str:
    return os.environ.get(HA_LAYOUT_ENV, "tier3").strip().lower() or "tier3"


# ---------------------------------------------------------------------------
# Dual-master EnvSpec constructors (same registry, different configuration
# combos — the brief's Tier-1 vs Tier-2/3 split)
# ---------------------------------------------------------------------------


def tier1_dual_spec(ctx) -> EnvSpec:
    """Tier-1: two standalone masters on distinct port groups, no ZK.

    needConsistency stays off (the mock-line default), so the three
    same-host assumptions (ZK leader id, port stitching, SELF_TARGET) are
    inert — distinct ports are the zero-risk layout (see the MasterSpec
    docstring evidence chain in harness.py).
    """
    return EnvSpec(
        label=f"ha_t1_{ctx.profile}",
        n_prefill=2,
        n_decode=4,
        perf=default_perf(),
        master_profile=ctx.profile,
        discovery="file",
        masters=[
            MasterSpec(name="A", http_port=HA_TIER1_MASTER_A_HTTP_PORT),
            MasterSpec(name="B", http_port=HA_TIER1_MASTER_B_HTTP_PORT),
        ],
    )


def tier3_dual_spec(ctx) -> EnvSpec:
    """Tier-3: same port group on distinct loopback IPs + ZK helper.

    FLEXLB_ADVERTISED_IP and FLEXLB_SYNC_CONSISTENCY_CONFIG are injected
    per the cross-agent contract (flexlb-sync owner); both instances share
    one HIPPO_ROLE (mutual master/follower over /master_lb_leader/{role}).
    """
    return EnvSpec(
        label=f"ha_t3_{ctx.profile}",
        n_prefill=2,
        n_decode=4,
        perf=default_perf(),
        master_profile=ctx.profile,
        discovery="file",
        masters=[
            MasterSpec(
                name="A",
                http_port=HA_TIER3_MASTER_HTTP_PORT,
                bind_ip="127.0.0.1",
                advertised_ip="127.0.0.1",
            ),
            MasterSpec(
                name="B",
                http_port=HA_TIER3_MASTER_HTTP_PORT,
                bind_ip=HA_TIER3_MASTER_B_BIND_IP,
                advertised_ip=HA_TIER3_MASTER_B_BIND_IP,
            ),
        ],
        zk_consistency={"zkTimeoutMs": 10_000},
    )


def dual_spec_for_layout(ctx, layout: Optional[str] = None) -> EnvSpec:
    chosen = (layout or ha_layout()).lower()
    if chosen in ("tier1", "t1", "1"):
        return tier1_dual_spec(ctx)
    if chosen in ("tier3", "t3", "3"):
        return tier3_dual_spec(ctx)
    raise ValueError(f"unknown FLEXLB_FT_HA_LAYOUT '{chosen}' (expected tier1|tier3)")


# ---------------------------------------------------------------------------
# Per-instance helpers
# ---------------------------------------------------------------------------


def instance_ops(ctx, env, name: str) -> EngineOps:
    """EngineOps bound to ONE master instance (schedule/recovery/info probes
    against its own bind_ip:http_port — the _recovery_rate/topology-wait
    judgement generalized per master, brief p9)."""
    key = (id(env), f"master:{name}")
    if key not in ctx._ops_cache:
        mspec = env.master_specs[name]
        ctx._ops_cache[key] = EngineOps(
            mspec.bind_ip, mspec.http_port, env.mock_http_port
        )
    return ctx._ops_cache[key]


def restore_masters(ctx, env) -> int:
    """Restart every dead registry slot (case finally-path hygiene: the
    shared env must stay usable for the next case — mirrors the single
    master_kill finally)."""
    restarted = 0
    for mspec in env.spec.masters:
        if env.masters.get(mspec.name) is None:
            try:
                ctx.env_manager.restart_master_instance(env, mspec.name)
                restarted += 1
            except Exception:
                pass
    return restarted


def instance_alive_full(ops: EngineOps, env, timeout_s: float = 60.0) -> bool:
    """Per-master topology convergence: alive == discovered == the spec
    topology within the brief's 60s cold-convergence window."""
    deadline = time.monotonic() + timeout_s
    want_p = env.spec.n_prefill
    want_d = env.spec.n_decode
    while time.monotonic() < deadline:
        try:
            if (
                ops.master_alive_count("PREFILL") >= want_p
                and ops.master_alive_count("DECODE") >= want_d
            ):
                return True
        except Exception:
            pass
        time.sleep(1.0)
    return False


def recovery_rate(ops: EngineOps, n: int = 20) -> tuple:
    """The brief's _recovery_rate judgement generalized to whichever master
    the caller binds: n fresh requests, >= 95% ok within the 30s recovery
    window (RECOVERY_TIMEOUT_S per request stream)."""
    ok = 0
    errors: list = []
    for _ in range(n):
        _ok, msg = ops.verify_recovery()
        if _ok:
            ok += 1
        else:
            errors.append(str(msg)[:60])
    rate = ok / n if n else 0.0
    return rate >= 0.95, f"recovery={ok}/{n} ({rate:.0%}, >=95% required)"


# ---------------------------------------------------------------------------
# Synthetic mini-trace (JavaLoadClient replays TRACE_FILE rows; LOOP=true
# cycles them until DURATION_S wall-clock expiry)
# ---------------------------------------------------------------------------

HA_TRACE_ROWS = 20
HA_TRACE_IL = 16
HA_TRACE_OL = 4
HA_TRACE_SPACING_MS = 100


def write_ha_trace(case_dir: Path) -> Path:
    path = case_dir / "ha_trace.jsonl"
    lines = []
    for i in range(HA_TRACE_ROWS):
        lines.append(
            json.dumps(
                {
                    "ts": i * HA_TRACE_SPACING_MS,
                    "il": HA_TRACE_IL,
                    "ol": HA_TRACE_OL,
                    "bh": [i * 1_000_003 + 7],
                    "priority": 50,
                },
                separators=(",", ":"),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Traffic runner — JavaLoadClient kept alive ACROSS fault injections
# ---------------------------------------------------------------------------


class HaTrafficRunner:
    """Background JavaLoadClient over GRPC_TARGETS (the HA multi-target
    mode: sticky target + same-request transport-failure retry).

    Phase bookkeeping: the runner stamps wall-clock epoch seconds at
    ``mark()`` call sites; row windows are then sliced offline by
    send_start_epoch_ms (falling back to wall_clock_ts) — the assertions
    compare pre/post-injection WINDOWS, never exact totals (rows buffered
    at a SIGTERM instant may be lost; natural DURATION_S exit flushes all).
    """

    def __init__(
        self,
        ctx,
        env,
        case_dir: Path,
        name: str,
        targets: list,
        *,
        duration_s: int = 60,
        timeout_ms: int = 30_000,
        enable_fallback: bool = False,
        replay_speed: float = 2.0,
        max_concurrency: int = 8,
    ):
        self.ctx = ctx
        self.env = env
        self.name = name
        self.targets = list(targets)
        self.out_dir = case_dir / f"{name}_out"
        self.log_file = case_dir / f"{name}.log"
        self._client = ClientOps(ctx.env_manager, "1g", "1g")
        overrides = {
            "TRACE_FILE": str(write_ha_trace(case_dir)),
            "GRPC_TARGETS": ",".join(self.targets),
            "DURATION_S": str(int(duration_s)),
            "REPLAY_SPEED": str(replay_speed),
            "MAX_CONCURRENCY": str(max_concurrency),
            "TIMEOUT_MS": str(int(timeout_ms)),
            "LOOP": "true",
            "N_CHANNELS": "2",
            "EVENT_LOOP_THREADS": "4",
            "SKIP_SERVER_LATENCY": "true",
            "PRIORITY": "50",
        }
        if enable_fallback:
            # Direct-connect engine addresses: the mock's endpoints.json
            # snapshot (brief p7 — static engine set, equivalent to the
            # production domain query).
            overrides["ENABLE_FALLBACK"] = "true"
            overrides["ENDPOINTS_FILE"] = str(env.endpoint_file)
        self._overrides = overrides
        self.proc = None

    def start(self) -> None:
        self.proc, self.out_dir = self._client.run_async(
            self._overrides, self.out_dir, self.log_file, label=self.name
        )

    @staticmethod
    def now() -> float:
        return time.time()

    def wait_finish(self, extra_s: float = 60.0):
        """Wait for the natural DURATION_S exit (all rows flushed); the
        stop_async SIGTERM path is only a timeout fallback."""
        result = None
        if self.proc is not None:
            if not self.proc.wait(extra_s):
                # Timeout fallback: SIGTERM (buffered rows may be lost —
                # the window-comparison assertions tolerate that).
                result = self._client.stop_async(self.proc, self.out_dir)
            else:
                result = self._client.stop_async(self.proc, self.out_dir)
        return result

    def rows(self) -> list:
        path = self.out_dir / "client_events.jsonl"
        rows = []
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
        return rows


# ---------------------------------------------------------------------------
# Row statistics + window slicing
# ---------------------------------------------------------------------------


def row_ts_ms(row: dict) -> Optional[float]:
    """Row send timestamp (ms epoch) — send_start_epoch_ms preferred,
    wall_clock_ts (s) as the fallback."""
    v = row.get("send_start_epoch_ms")
    if isinstance(v, (int, float)) and v > 0:
        return float(v)
    w = row.get("wall_clock_ts")
    if isinstance(w, (int, float)) and w > 0:
        return float(w) * 1000.0
    return None


def rows_between(rows: list, lo_s: Optional[float], hi_s: Optional[float]) -> list:
    """Rows whose send timestamp falls in [lo_s, hi_s) (epoch seconds)."""
    out = []
    for row in rows:
        ts = row_ts_ms(row)
        if ts is None:
            continue
        if lo_s is not None and ts < lo_s * 1000.0:
            continue
        if hi_s is not None and ts >= hi_s * 1000.0:
            continue
        out.append(row)
    return out


class HaRows:
    """Counter-style view over client_events.jsonl rows."""

    def __init__(self, rows: list):
        self.rows = rows
        missing = [
            k
            for k in ("route_path", "master_target", "failover", "error_kind")
            if any(k not in r for r in rows)
        ]
        # Fail-closed on a stale client build: the HA observability fields
        # are part of the delivered contract (route_path value-domain
        # extension + master_target/failover/error_kind keys) — a client
        # without them must surface, not silently downgrade assertions.
        self.missing_fields = missing

    # -- predicates ---------------------------------------------------------

    def route(self, value: str) -> list:
        return [r for r in self.rows if r.get("route_path") == value]

    def target(self, value: str) -> list:
        return [r for r in self.rows if r.get("master_target") == value]

    def failover_rows(self) -> list:
        return [r for r in self.rows if r.get("failover") is True]

    def error_kind(self, value: str) -> list:
        return [r for r in self.rows if r.get("error_kind") == value]

    def status(self, value: str) -> list:
        return [r for r in self.rows if r.get("status") == value]

    def ok_rows(self) -> list:
        return [r for r in self.rows if r.get("status") in ("ok", "scheduled")]

    def dup_rids(self) -> list:
        seen = Counter(r.get("rid") for r in self.rows)
        return [rid for rid, n in seen.items() if n > 1]

    def by_route(self) -> Counter:
        return Counter(r.get("route_path", "<missing>") for r in self.rows)

    def by_target(self) -> Counter:
        return Counter(r.get("master_target", "<missing>") for r in self.rows)

    def ok_rate(self) -> float:
        return len(self.ok_rows()) / len(self.rows) if self.rows else 0.0

    def detail(self) -> str:
        return (
            f"n={len(self.rows)}, route={dict(self.by_route())}, "
            f"target={dict(self.by_target())}, "
            f"ok_rate={self.ok_rate():.0%}"
        )


def ha_gate() -> Optional[tuple]:
    """Uniform arm-gate for every dual-master HA case: None when armed
    (FLEXLB_FT_HA_DUAL_MASTER=1), else a (True, SKIP) verdict.

    Keeps the compat hard constraint: with the gate off, a default
    --category master run executes the new case names as SKIPs (no dual
    env boot, no Java prerequisites touched) and every legacy case keeps
    its byte-identical behaviour.
    """
    if ha_dual_enabled():
        return None
    return True, (
        "SKIP (FLEXLB_FT_HA_DUAL_MASTER!=1): dual-master HA case not "
        "armed — default run keeps the legacy single-master behaviour"
    )
