"""Absolute single-run performance decisions from frozen request evidence."""

import bisect
import math

from schema_contract import matches_schema
from analysis.checks import compare
from analysis.statistics import percentile_nr
from cases.master_performance.inputs import validate
from input_contract import finite_number as finite


REQUIRED_PROVENANCE = (
    "instance",
    "configuration_sha256",
    "master_artifact",
    "mock_jar_sha256",
    "actual_master_config",
    "performance",
    "topology",
    "capacity",
    "trace",
    "client_environment",
)



def engine_tps_checks(evidence):
    """Scrape-time samples; sum priorities per engine, then equally weight engines.

    No Prometheus lookback filling, idle filtering, or cluster TPS summation.
    An engine restart changes incarnation and therefore invalidates coverage.
    """
    bounds = evidence["criteria"].get("engine_tps")
    if bounds is None:
        return {}, []
    lo = evidence["window"]["start_epoch_ms"] / 1000
    hi = evidence["window"]["end_epoch_ms"] / 1000
    gap = evidence["criteria"]["max_gap_s"]
    from cases.master_performance.inputs import engine_roles
    roles = engine_roles(evidence.get("gate_input"), bounds)
    groups = {name: {} for name in roles}
    for row in evidence.get("engine_tps_samples", []):
        labels = row["metric"]
        name = row.get("metric_id", labels.get("__name__"))
        if name not in groups:
            continue
        role = roles[name]
        if labels.get("role") != role or not labels.get("engine_name"):
            raise ValueError("engine TPS lacks role/engine identity")
        key = (labels["engine_name"], labels.get("engine_incarnation", ""))
        priorities = groups[name].setdefault(key, {})
        samples = priorities.setdefault(labels.get("priority", "aggregate"), {})
        for stamp, raw in row["values"]:
            value = float(raw)
            if not finite(stamp) or not math.isfinite(value) or value < 0:
                raise ValueError("invalid engine TPS sample")
            if lo <= stamp <= hi:
                if stamp in samples and samples[stamp] != value:
                    raise ValueError("conflicting engine TPS samples")
                samples[stamp] = value
    metrics, checks = {}, []
    for name, engines in groups.items():
        expected = evidence["provenance"]["topology"][roles[name]]
        if (type(expected) is not int or expected <= 0 or len(engines) != expected
                or len({key[0] for key in engines}) != expected):
            raise ValueError("engine TPS coverage/epoch mismatch: " + name)
        means = []
        for priorities in engines.values():
            stamp_sets = [set(s) for s in priorities.values()]
            if not stamp_sets or not stamp_sets[0] or any(s != stamp_sets[0] for s in stamp_sets):
                raise ValueError("missing engine TPS priority samples: " + name)
            stamps = sorted(stamp_sets[0])
            if (stamps[0] - lo > gap or hi - stamps[-1] > gap
                    or any(b - a > gap for a, b in zip(stamps, stamps[1:]))):
                raise ValueError("engine TPS scrape gap: " + name)
            means.append(sum(sum(s[t] for s in priorities.values()) for t in stamps) / len(stamps))
        value = sum(means) / expected
        metrics[name] = value
        metrics[name + "_engine_count"] = len(engines)
        checks.append(dict(metric=name, actual=value, bound=bounds[name], direction="min",
                           status="PASS" if compare(value, "ge", bounds[name]) else "FAIL"))
    return metrics, checks


def percentile(values, q=0.99):
    return percentile_nr(values, q)


def analyze(evidence):
    errors = []
    result = dict(
        performance_analysis_schema_version=1,
        verdict="INVALID",
        errors=errors,
        checks=[],
        metrics={},
        windows=[],
    )
    try:
        if not isinstance(evidence, dict):
            raise ValueError("evidence must be an object")
        if not isinstance(evidence.get("errors", []), list):
            raise ValueError("errors must be a list")
        errors.extend(evidence.get("errors", []))
        c = validate(evidence["criteria"], evidence.get("gate_input"))
        if not matches_schema(evidence, "performance_evidence_schema_version", 2):
            raise ValueError("unsupported evidence version")
        p = evidence["provenance"]
        for k in REQUIRED_PROVENANCE:
            if not p.get(k):
                raise ValueError("missing provenance: " + k)
        if not isinstance(p["instance"], str) or not p["instance"].strip():
            raise ValueError("missing/invalid instance identity")
        for v in (
            p["configuration_sha256"],
            p["mock_jar_sha256"],
            p.get("analyzer_sha256"),
            p["master_artifact"].get("jar_sha256"),
            p["trace"].get("sha256"),
            p["trace"].get("workload_sha256"),
        ):
            if (
                not isinstance(v, str)
                or len(v) != 64
                or any(x not in "0123456789abcdef" for x in v)
            ):
                raise ValueError("missing/invalid artifact or workload SHA256")
        if str(p["client_environment"].get("FETCH_OUTPUT_STREAM")).lower() not in {
            "1",
            "true",
        }:
            raise ValueError("complete Fetch chain required")
        lo, hi = (
            evidence["window"]["start_epoch_ms"],
            evidence["window"]["end_epoch_ms"],
        )
        if not all(finite(x) for x in (lo, hi)) or hi <= lo:
            raise ValueError("invalid measurement window")
        duration = (hi - lo) / 1000
        if abs(duration - c["measure_s"]) > 0.001:
            raise ValueError("measurement duration differs from contract")
        snap = evidence["flow"]
        if snap.get("complete") is not True or snap.get("errors"):
            raise ValueError(
                "incomplete request accounting: " + str(snap.get("errors", []))
            )
        issued, records = snap["issued"], snap["records"]
        a, b = {}, {}
        for source, target in ((issued, a), (records, b)):
            for row in source:
                rid = row.get("rid")
                if not isinstance(rid, str) or not rid or rid in target:
                    raise ValueError("missing or duplicate request identity")
                target[rid] = row
        if not a or a.keys() != b.keys():
            raise ValueError("issued/terminal request identities differ")
        for rid, row in a.items():
            terminal = b[rid]
            for k in ("send_start_epoch_ms", "input_len", "output_len"):
                if not finite(row.get(k)) or row[k] <= 0 or terminal.get(k) != row[k]:
                    raise ValueError("missing or inconsistent request field: " + k)
            if not finite(row.get("pacing_lag_ms")) or row["pacing_lag_ms"] < 0:
                raise ValueError("missing pacing lag")
            if not isinstance(terminal.get("status"), str) or terminal["status"] in {
                "scheduled",
                "unknown",
                "",
            }:
                raise ValueError("request lacks a full inference terminal status")
            if not finite(terminal.get("total_ms")) or terminal["total_ms"] < 0:
                raise ValueError("missing terminal duration")
            if terminal["status"] == "ok":
                n = terminal.get("observed_output_tokens")
                if type(n) is not int or n <= 0:
                    raise ValueError("successful request lacks observed output tokens")
                if (
                    not finite(terminal.get("ttft_ms"))
                    or not 0 < terminal["ttft_ms"] <= terminal["total_ms"]
                ):
                    raise ValueError("invalid successful TTFT")
        stamps = [x["epoch_ms"] for x in evidence["samples"]]
        if len(stamps) < 2 or any(not finite(x) for x in stamps):
            raise ValueError("missing observer coverage")
        if (
            stamps[0] > lo
            or stamps[-1] < hi
            or any(
                y <= x or y - x > c["max_gap_s"] * 1000
                for x, y in zip(stamps, stamps[1:])
            )
        ):
            # Coverage invalidates the run, not the independently validated request ledger.
            errors.append("observer coverage gap")
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        errors.append(str(exc))
        return result

    cohort = [b[rid] for rid, r in a.items() if lo <= r["send_start_epoch_ms"] < hi]
    if len(cohort) < c["min_requests"]:
        errors.append("insufficient arrival cohort")
    sent_qps = len(cohort) / duration
    if abs(sent_qps / c["qps"] - 1) > c["qps_tolerance"]:
        errors.append("offered QPS outside contract")
    if (
        cohort
        and max(a[r["rid"]]["pacing_lag_ms"] for r in cohort) > c["max_pacing_lag_ms"]
    ):
        errors.append("client pacing lag exceeds budget")
    ok = [r for r in cohort if r["status"] == "ok"]

    def tpot(r):
        n = r["observed_output_tokens"]
        return (r["total_ms"] - r["ttft_ms"]) / (n - 1) if n > 1 else None

    def slo(r):
        return (
            r["ttft_ms"] <= c["slo_ttft_ms"]
            and r["total_ms"] <= c["slo_e2e_ms"]
            and (tpot(r) is None or tpot(r) <= c["slo_tpot_ms"])
        )

    good = [r for r in ok if slo(r)]
    all_rows = list(b.values())

    def finished(r):
        return r["send_start_epoch_ms"] + r["total_ms"]

    # Index once: scanning every request for every one-second bucket made the
    # full-scale gate spend its finish deadline analyzing an already drained run.
    completed = sorted((r for r in all_rows if r["status"] == "ok"), key=finished)
    completed_stamps = [finished(r) for r in completed]

    def completions(start, end):
        return completed[bisect.bisect_left(completed_stamps, start):
                         bisect.bisect_left(completed_stamps, end)]

    sends = sorted(r["send_start_epoch_ms"] for r in all_rows)
    ends = sorted(finished(r) for r in all_rows)

    def inflight(t):
        return bisect.bisect_left(sends, t) - bisect.bisect_left(ends, t)

    done = completions(lo, hi)
    m = dict(
        sent_qps=sent_qps,
        offered_qps_deviation=abs(sent_qps / c["qps"] - 1),
        pacing_lag_max_ms=max((a[r["rid"]]["pacing_lag_ms"] for r in cohort), default=None),
        cohort_requests=len(cohort),
        success_requests=len(ok),
        error_rate=sum(r["status"] != "ok" for r in all_rows) / len(all_rows),
        cohort_error_rate=1 - len(ok) / len(cohort) if cohort else None,
        slo_fraction=len(good) / len(cohort) if cohort else None,
        goodput_rps=len(good) / duration,
        input_goodput=sum(r["input_len"] for r in good) / duration,
        output_goodput=sum(r["observed_output_tokens"] for r in good) / duration,
        input_tps=sum(r["input_len"] for r in done) / duration,
        output_tps=sum(r["observed_output_tokens"] for r in done) / duration,
        ttft_p99_ms=percentile([r["ttft_ms"] for r in ok]),
        e2e_p99_ms=percentile([r["total_ms"] for r in ok]),
        tpot_p99_ms=percentile([tpot(r) for r in ok if tpot(r) is not None]),
        inflight_start=inflight(lo),
        inflight_end=inflight(hi),
        inflight_growth_rps=(inflight(hi) - inflight(lo)) / duration,
    )
    # Single-token output has no TPOT. No successful cohort is a business failure.
    checks = []
    for metric, key, direction in (
        ("input_tps", "min_input_tps", "min"),
        ("output_tps", "min_output_tps", "min"),
        ("goodput_rps", "min_goodput_rps", "min"),
        ("slo_fraction", "min_slo_fraction", "min"),
        ("error_rate", "max_error_rate", "max"),
        ("ttft_p99_ms", "max_ttft_p99_ms", "max"),
        ("e2e_p99_ms", "max_e2e_p99_ms", "max"),
        ("tpot_p99_ms", "max_tpot_p99_ms", "max"),
        ("inflight_growth_rps", "max_inflight_growth_rps", "max"),
    ):
        v = m[metric]
        na = (
            metric == "tpot_p99_ms"
            and bool(ok)
            and all(r["observed_output_tokens"] == 1 for r in ok)
        )
        passed = na or (
            v is not None and compare(v, "ge" if direction == "min" else "le", c[key])
        )
        checks.append(
            dict(
                metric=metric,
                actual=v,
                bound=c[key],
                direction=direction,
                status="NOT_APPLICABLE" if na else "PASS" if passed else "FAIL",
            )
        )
    # Curves use disjoint completion buckets; cohort goodput above includes delayed terminals.
    try:
        engine_metrics, engine_checks = engine_tps_checks(evidence)
        m.update(engine_metrics)
        checks.extend(engine_checks)
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        errors.append("engine TPS: " + str(exc))
    for i in range(math.ceil(duration)):
        start, end = lo + i * 1000, min(hi, lo + (i + 1) * 1000)
        rows = completions(start, end)
        dt = (end - start) / 1000
        result["windows"].append(
            dict(
                t=i,
                input_tps=sum(r["input_len"] for r in rows) / dt,
                output_tps=sum(r["observed_output_tokens"] for r in rows) / dt,
                inflight=inflight(end),
            )
        )
    result.update(
        metrics=m,
        checks=checks,
        verdict=(
            "INVALID"
            if errors
            else "FAIL" if any(x["status"] == "FAIL" for x in checks) else "PASS"
        ),
    )
    return result
