"""HA client metrics computed from frozen rows and explicit topology inputs."""

from collections import Counter
from typing import Optional

HA_METRICS = {
    "sample_count",
    "success_rate",
    "non_ok_count",
    "target_share",
    "target_count",
    "route_share",
    "route_count",
    "failover_count",
    "duplicate_ids",
    "error_kind_count",
    "wrong_error_code",
    "failed_count",
    "failed_rate_above_one",
    "business_rate_above_one",
    "visible_terminal_count",
    "visible_terminal_share",
    "prefill_max_share",
    "prefill_peak_skew",
}


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


def measure_client_metric(params, rows, *, target=None, prefill_pool=()):
    """Measure one declared metric; the executor supplies live target/pool identity."""
    metric = params["metric"].split("/", 1)[1]
    if metric not in HA_METRICS:
        raise ValueError(f"unknown HA metric: {params['metric']}")
    n = len(rows)
    if metric == "sample_count":
        actual = n
    elif metric == "success_rate":
        actual = sum(r["status"] == "ok" for r in rows) / n if n else 0
    elif metric == "non_ok_count":
        actual = sum(r["status"] != "ok" for r in rows)
    elif metric in {"target_share", "target_count"}:
        count = sum(r["master_target"] == target for r in rows)
        actual = count / n if metric == "target_share" and n else count
    elif metric in {"route_share", "route_count"}:
        count = sum(r["route_path"] == params["route"] for r in rows)
        actual = count / n if metric == "route_share" and n else count
    elif metric == "failover_count":
        actual = sum(r["failover"] is True for r in rows)
    elif metric == "prefill_max_share":
        pool = set(prefill_pool)
        addresses = [r.get("prefill") for r in rows if r["status"] == "ok"]
        if not addresses or any(address not in pool for address in addresses):
            raise ValueError("successful HA requests lack known Prefill endpoints")
        actual = max(Counter(addresses).values()) / len(addresses)
    elif metric == "prefill_peak_skew":
        pool = set(prefill_pool)
        windows = prefill_assignment_windows(rows)
        assigned = {row.get("prefill") for row in rows if row.get("prefill")}
        if not pool or not assigned <= pool:
            raise ValueError("HA requests reference unknown Prefill endpoints")
        eligible = [counts for counts in windows.values()
                    if sum(counts.values()) >= params["min_samples"]]
        if not eligible:
            raise ValueError("no HA rolling 5-second window has enough assigned Prefill samples")
        actual = max(max(counts.values()) * len(pool) / sum(counts.values())
                     for counts in eligible)
    elif metric == "duplicate_ids":
        actual = sum(count > 1 for count in Counter(r["rid"] for r in rows).values())
    elif metric == "error_kind_count":
        actual = sum(r["error_kind"] == params["error_kind"] for r in rows)
    elif metric in {"failed_rate_above_one", "business_rate_above_one"}:
        count = sum(
            (
                r["route_path"] == "failed"
                if metric == "failed_rate_above_one"
                else r["error_kind"] == "business"
            )
            for r in rows
        )
        actual = count / n if count > 1 and n else 0
    elif metric in {"visible_terminal_count", "visible_terminal_share"}:
        actual = sum(
            r["status"] == "ok"
            or r["error_kind"] in {"deadline", "transport", "business"}
            for r in rows
        )
        if metric == "visible_terminal_share":
            actual = actual / n if n else 0
    elif metric == "wrong_error_code":
        # Preserve the legacy literal substring predicate, not a typed/exact
        # code claim. A strict check needs a structured client error-code field.
        actual = sum(str(params["code"]) not in str(r.get("error", "")) for r in rows)
    else:
        actual = sum(r["route_path"] == "failed" for r in rows)
    return actual
