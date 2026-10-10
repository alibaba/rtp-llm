"""Bounded incremental request-ledger reader shared by traffic executors."""

import json
from pathlib import Path


class LiveClientEvents:
    """Incremental journal reader; incomplete trailing writes are retried, not lost."""

    def __init__(self, path, *, max_events=50_000):
        if type(max_events) is not int or not 1 <= max_events <= 2_000_000:
            raise ValueError("live client event budget must be in 1..2000000")
        self.max_events = max_events
        self.path = Path(path)
        self.offset = 0
        self.pending = b""
        self.sequence = 0
        self.issued = {}
        self.terminal = {}

    def read(self):
        if not self.path.exists():
            return
        if self.path.stat().st_size < self.offset:
            raise ValueError("live client journal was truncated")
        # Read a finite snapshot in bounded chunks; a delayed poll is not data loss.
        end = self.path.stat().st_size
        with self.path.open("rb") as stream:
            stream.seek(self.offset)
            while self.offset < end:
                chunk = stream.read(min(8_000_000, end - self.offset))
                if not chunk:
                    raise ValueError("live client journal was truncated")
                self.offset += len(chunk)
                self._consume(chunk)

    def _consume(self, chunk):
        import math

        lines = (self.pending + chunk).split(b"\n")
        self.pending = lines.pop()
        if len(self.pending) > 8_000_000 or any(len(line) > 8_000_000 for line in lines):
            raise ValueError("live client row exceeds read budget")
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
                if row.get("status") not in {
                    "ok",
                    "schedule_error",
                    "exception",
                    "engine_error",
                    "empty_response",
                    "incomplete_response",
                    "timeout",
                    "scheduled",
                }:
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
            if self.sequence > self.max_events:
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


def first_request(rows):
    """Freeze actual send onset from issued requests, never successful terminals only."""
    import math
    stamps = []
    for row in rows:
        value = row.get("send_start_epoch_ms")
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError("request onset requires positive finite send_start_epoch_ms")
        stamps.append((value, row.get("rid")))
    if not stamps:
        return None
    value, identity = min(stamps, key=lambda item: item[0])
    return dict(epoch_s=value / 1000, rid=identity, field="send_start_epoch_ms")


def request_timing(rows):
    """Keep corrupt/partial timing visible in resource evidence without losing rows."""
    try:
        return dict(traffic_start=first_request(rows), errors=[])
    except ValueError as exc:
        return dict(traffic_start=None, errors=[str(exc)])
