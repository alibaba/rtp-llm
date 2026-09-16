"""Window request statistics reused by scenario views, over archived client rows."""

import math

from stress.compare_twin import percentile_nr

FIELDS = {
    "ttft_ms": "ms",
    "total_ms": "ms",
    "schedule_ms": "ms",
    "input_len": "tokens",
    "output_len": "tokens",
    "count": "requests",
    "errors": "ratio",
}


def measure(report, selection, lower, upper):
    if (
        set(selection) != {"requests", "reducer", "cohort", "outcomes"}
        or selection["requests"] not in FIELDS
        or selection["cohort"] not in {"sent", "completed"}
        or selection["outcomes"] not in {"success", "all"}
    ):
        raise ValueError("invalid request metric definition")
    field, reducer = selection["requests"], selection["reducer"]
    if (
        (field == "count" and reducer not in {"count", "rate"})
        or (
            field == "errors"
            and (reducer != "fraction" or selection["outcomes"] != "all")
        )
        or (field not in {"count", "errors"} and reducer not in {"p50", "p95", "p99"})
    ):
        raise ValueError("invalid request metric reduction")
    result = dict(
        selection=selection,
        source=report.get("request_sources", []),
        window=[lower, upper],
        unit="requests/s" if reducer == "rate" else FIELDS[field],
        value=None,
    )
    if report["workload"]["runtime_validity"] != "VALID":
        return dict(result, status="INVALID_EVIDENCE")
    if report.get("request_source_state") != "VERIFIED":
        return dict(result, status="MISSING_DATA")
    rows = []
    for row in report.get("_request_rows", []):
        epoch = (
            row.get("send_start_epoch_ms")
            if selection["cohort"] == "sent"
            else row.get("wall_clock_ts")
        )
        if type(epoch) not in (int, float) or not math.isfinite(epoch) or epoch <= 0:
            return dict(result, status="MISSING_TIMESTAMP")
        seconds = (epoch / 1000 if selection["cohort"] == "sent" else epoch) - report[
            "clock_anchor"
        ]["epoch_s"]
        if lower <= seconds < upper:
            rows.append(row)
    ok = lambda row: row.get("status") == "ok"
    selected = [r for r in rows if selection["outcomes"] == "all" or ok(r)]
    if field == "count":
        value = len(selected) / (upper - lower) if reducer == "rate" else len(selected)
    elif field == "errors":
        if not rows:
            return dict(result, status="MISSING_DATA")
        value = sum(not ok(r) for r in rows) / len(rows)
    else:
        values = [r.get(field) for r in selected]
        if not values or any(
            type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values
        ):
            return dict(result, status="MISSING_DATA")
        value = percentile_nr(values, int(reducer[1:]) / 100)
    return dict(
        result,
        status="AVAILABLE",
        value=value,
        samples=len(selected),
        semantics="half-open window; explicit send/completion cohort; pooled nearest-rank percentile (stress.compare_twin.percentile_nr)",
    )
