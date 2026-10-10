"""HA client metrics computed from frozen rows and explicit topology inputs."""

from collections import Counter
from dataclasses import dataclass
from typing import Optional, Sequence

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


def prefill_assignment_buckets(rows: list) -> dict:
    """Count assigned Prefill endpoints by send second, including failed requests."""
    from collections import Counter, defaultdict

    buckets = defaultdict(Counter)
    for row in rows:
        address = row.get("prefill")
        if not address:
            continue
        timestamp = row_ts_ms(row)
        if timestamp is None:
            raise ValueError("assigned HA request lacks send timestamp")
        buckets[int(timestamp // 1000)][address] += 1
    return dict(buckets)


def prefill_assignment_windows(rows: list, seconds: int = 5) -> dict:
    """Rolling Prefill counts at one-second steps over complete windows."""
    from collections import Counter

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

    def count_param(self, field, parameter):
        return sum(row[field] == self.params[parameter] for row in self.rows)

    def share(self, count):
        return count / len(self.rows) if self.rows else 0

    def rate_above_one(self, count):
        return self.share(count) if count > 1 else 0


def _prefill_max_share(source):
    pool = set(source.prefill_pool)
    addresses = [r.get("prefill") for r in source.rows if r["status"] == "ok"]
    if not addresses or any(address not in pool for address in addresses):
        raise ValueError("successful HA requests lack known Prefill endpoints")
    return max(Counter(addresses).values()) / len(addresses)


def _prefill_peak_skew(source):
    pool = set(source.prefill_pool)
    windows = prefill_assignment_windows(source.rows)
    assigned = {row.get("prefill") for row in source.rows if row.get("prefill")}
    if not pool or not assigned <= pool:
        raise ValueError("HA requests reference unknown Prefill endpoints")
    eligible = [counts for counts in windows.values()
                if sum(counts.values()) >= source.params["min_samples"]]
    if not eligible:
        raise ValueError("no HA rolling 5-second window has enough assigned Prefill samples")
    return max(max(counts.values()) * len(pool) / sum(counts.values())
               for counts in eligible)


def _visible_terminal_count(source):
    return sum(row["status"] == "ok" or row["error_kind"] in {"deadline", "transport", "business"}
               for row in source.rows)


def _wrong_error_code(source):
    # This is a literal substring predicate; exact matching requires a structured code field.
    return sum(str(source.params["code"]) not in str(row.get("error", "")) for row in source.rows)


_CLIENT_METRICS = {
    "sample_count": lambda s: len(s.rows),
    "success_rate": lambda s: s.share(s.count("status", "ok")),
    "non_ok_count": lambda s: sum(r["status"] != "ok" for r in s.rows),
    "target_count": lambda s: s.count("master_target", s.target),
    "target_share": lambda s: s.share(s.count("master_target", s.target)),
    "route_count": lambda s: s.count_param("route_path", "route"),
    "route_share": lambda s: s.share(s.count_param("route_path", "route")),
    "failover_count": lambda s: sum(r["failover"] is True for r in s.rows),
    "duplicate_ids": lambda s: sum(count > 1 for count in Counter(r["rid"] for r in s.rows).values()),
    "error_kind_count": lambda s: s.count_param("error_kind", "error_kind"),
    "wrong_error_code": _wrong_error_code,
    "failed_count": lambda s: s.count("route_path", "failed"),
    "failed_rate_above_one": lambda s: s.rate_above_one(s.count("route_path", "failed")),
    "business_rate_above_one": lambda s: s.rate_above_one(s.count("error_kind", "business")),
    "visible_terminal_count": _visible_terminal_count,
    "visible_terminal_share": lambda s: s.share(_visible_terminal_count(s)),
    "prefill_max_share": _prefill_max_share,
    "prefill_peak_skew": _prefill_peak_skew,
}
HA_METRICS = frozenset(_CLIENT_METRICS)


def measure_client_metric(params, rows, *, target=None, prefill_pool=()):
    """Measure one declared metric; the executor supplies live target/pool identity."""
    metric = params["metric"].partition("/")[2]
    calculator = _CLIENT_METRICS.get(metric)
    if calculator is None:
        raise ValueError(f"unknown HA metric: {params['metric']}")
    return calculator(_ClientMetricInput(params, rows, target, prefill_pool))
