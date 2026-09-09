"""HA traffic and observation helpers used by the case executor."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from .harness import ClientOps

HA_TRACE_ROWS = 20
HA_TRACE_SPACING_MS = 100
HA_TRACE_IL = 16
HA_TRACE_OL = 4


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
        manager,
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
        live_events: bool = False,
    ):
        self.manager = manager
        self.env = env
        self.name = name
        self.targets = list(targets)
        self.out_dir = case_dir / f"{name}_out"
        self.log_file = case_dir / f"{name}.log"
        self._client = ClientOps(manager, "1g", "1g")
        overrides = {
            "TRACE_FILE": str(write_ha_trace(case_dir)),
            "LIVE_CLIENT_EVENTS": str(live_events).lower(),
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


class LiveClientEvents:
    """Incremental journal reader; incomplete trailing writes are retried, not lost."""

    def __init__(self, path):
        self.path = Path(path)
        self.offset = 0
        self.pending = b""
        self.sequence = 0
        self.issued = {}
        self.terminal = {}

    def read(self):
        import math

        if not self.path.exists():
            return
        if self.path.stat().st_size < self.offset:
            raise ValueError("live client journal was truncated")
        with self.path.open("rb") as stream:
            stream.seek(self.offset)
            chunk = stream.read(8_000_001)
        if len(chunk) > 8_000_000:
            raise ValueError("live client journal exceeds read budget")
        self.offset += len(chunk)
        lines = (self.pending + chunk).split(b"\n")
        self.pending = lines.pop()
        for line in lines:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("invalid live client row")
            rid = row.get("rid")
            if (
                not isinstance(rid, str)
                or not rid
                or type(row.get("sequence")) is not int
                or row.get("sequence") != self.sequence + 1
            ):
                raise ValueError("live client identity or sequence gap")
            for field in ("send_start_epoch_ms", "recorded_epoch_ms"):
                value = row.get(field)
                if (
                    type(value) not in (int, float)
                    or not math.isfinite(value)
                    or value <= 0
                ):
                    raise ValueError("live client timestamp is missing or invalid")
            event = row.get("event")
            if event == "issued":
                if rid in self.issued:
                    raise ValueError("duplicate live request issue")
                self.issued[rid] = row
            elif event == "terminal":
                if rid not in self.issued or rid in self.terminal:
                    raise ValueError("orphan or duplicate live terminal")
                if (
                    row["send_start_epoch_ms"]
                    != self.issued[rid]["send_start_epoch_ms"]
                ):
                    raise ValueError("live terminal does not match issue")
                if row.get("status") not in {"ok", "schedule_error", "exception"}:
                    raise ValueError("live terminal is not a completed stream outcome")
                if (
                    row.get("route_path") not in {"master", "fallback", "failed"}
                    or type(row.get("failover")) is not bool
                ):
                    raise ValueError("live terminal lacks route evidence")
                self.terminal[rid] = row
            else:
                raise ValueError("unknown live client event")
            self.sequence += 1
            if self.sequence > 50_000:
                raise ValueError("live client journal exceeds event budget")

    def cohort(self, lower_ms, upper_ms, transition=False):
        selected = {}
        for rid, row in self.issued.items():
            if row["send_start_epoch_ms"] >= upper_ms:
                continue
            terminal = self.terminal.get(rid)
            if row["send_start_epoch_ms"] < lower_ms and not (
                transition
                and (terminal is None or terminal["recorded_epoch_ms"] >= lower_ms)
            ):
                continue
            selected[rid] = row
        return selected
