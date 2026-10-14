"""HA client metrics computed from frozen rows and explicit topology inputs."""

import math

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Optional, Sequence

from analysis.time_buckets import TimeBuckets


def row_ts_ms(row: dict) -> float:
    """Cohorts require the issue timestamp, never a terminal wall-clock fallback."""
    value = row.get("send_start_epoch_ms")
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError("HA request requires finite positive send_start_epoch_ms")
    return float(value)


def prefill_assignment_buckets(rows: list) -> dict:
    """Count assigned Prefill endpoints by send second, including failed requests."""
    buckets = defaultdict(Counter)
    grid = TimeBuckets(0, 1)
    for row in rows:
        address = row.get("prefill")
        if not address:
            continue
        timestamp = row_ts_ms(row)
        buckets[grid.index(timestamp)][address] += 1
    return dict(buckets)


def prefill_assignment_windows(rows: list, seconds: int = 5) -> dict:
    """Rolling Prefill counts at one-second steps over complete windows."""
    buckets = prefill_assignment_buckets(rows)
    if not buckets:
        return {}
    first, last = min(buckets), max(buckets)
    return {
        end: sum((buckets.get(second, Counter())
                  for second in range(end - seconds + 1, end + 1)), Counter())
        for end in range(first + seconds - 1, last + 1)
    }


@dataclass(frozen=True)
class _ClientMetricInput:
    params: dict
    rows: list
    target: Optional[str]
    prefill_pool: Sequence[str]

    def count(self, field, value):
        return sum(row[field] == value for row in self.rows)

    def share(self, count):
        return count / len(self.rows) if self.rows else 0


def _prefill_max_share(source):
    pool = set(source.prefill_pool)
    addresses = [r.get("prefill") for r in source.rows if r["status"] == "ok"]
    if not addresses or any(address not in pool for address in addresses):
        raise ValueError("successful HA requests lack known Prefill endpoints")
    return max(Counter(addresses).values()) / len(addresses)


def _visible_terminal_count(source):
    return sum(row["status"] == "ok" or row["error_kind"] in {"deadline", "transport", "business"}
               for row in source.rows)


_CLIENT_METRICS = {
    "success_rate": lambda s: s.share(s.count("status", "ok")),
    "non_ok_count": lambda s: sum(r["status"] != "ok" for r in s.rows),
    "target_share": lambda s: s.share(s.count("master_target", s.target)),
    "route_count": lambda s: s.count("route_path", s.params["route"]),
    "duplicate_ids": lambda s: sum(count > 1 for count in Counter(r["rid"] for r in s.rows).values()),
    "failed_count": lambda s: s.count("route_path", "failed"),
    "visible_terminal_share": lambda s: s.share(_visible_terminal_count(s)),
    "prefill_max_share": _prefill_max_share,
}
HA_METRICS = frozenset(_CLIENT_METRICS)


def measure_client_metric(params, rows, *, target=None, prefill_pool=()):
    """Measure one declared metric; the executor supplies live target/pool identity."""
    metric = params["metric"].partition("/")[2]
    calculator = _CLIENT_METRICS.get(metric)
    if calculator is None:
        raise ValueError(f"unknown HA metric: {params['metric']}")
    return calculator(_ClientMetricInput(params, rows, target, prefill_pool))
