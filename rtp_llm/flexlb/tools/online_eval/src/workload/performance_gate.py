"""Absolute single-run performance contract. A/B never changes these verdicts."""

import argparse
import bisect
import hashlib
import json
import math
from pathlib import Path

from reporting import write_bundle, run_meta, details, table

NUMERIC = {
    "warmup_s",
    "measure_s",
    "sample_s",
    "max_gap_s",
    "qps",
    "qps_tolerance",
    "min_requests",
    "max_pacing_lag_ms",
    "min_input_tps",
    "min_output_tps",
    "min_goodput_rps",
    "min_slo_fraction",
    "max_error_rate",
    "max_ttft_p99_ms",
    "max_e2e_p99_ms",
    "max_tpot_p99_ms",
    "slo_ttft_ms",
    "slo_e2e_ms",
    "slo_tpot_ms",
    "max_inflight_growth_rps",
}
REQUIRED_PROVENANCE = (
    "benchmark_id",
    "master_artifact",
    "mock_jar_sha256",
    "actual_master_config",
    "performance",
    "topology",
    "capacity",
    "trace",
    "client_environment",
)

ENGINE_TPS = {
    "rtp_llm_context_tps": "prefill",
    "rtp_llm_context_tps_with_cache": "prefill",
    "rtp_llm_generate_tps": "decode",
}


def finite(v):
    return type(v) in (int, float) and math.isfinite(v)


def validate(criteria):
    if not isinstance(criteria, dict) or set(criteria) - {"engine_tps"} != NUMERIC | {"benchmark_id"}:
        raise ValueError(
            "performance criteria must explicitly supply every contract field"
        )
    if (
        not isinstance(criteria["benchmark_id"], str)
        or not criteria["benchmark_id"].strip()
    ):
        raise ValueError("benchmark_id required")
    for k in NUMERIC:
        if not finite(criteria[k]) or criteria[k] < 0:
            raise ValueError(k + " must be finite and nonnegative")
    for k in (
        "qps",
        "measure_s",
        "min_requests",
        "min_input_tps",
        "min_output_tps",
        "min_goodput_rps",
        "slo_ttft_ms",
        "slo_e2e_ms",
        "slo_tpot_ms",
        "max_ttft_p99_ms",
        "max_e2e_p99_ms",
        "max_tpot_p99_ms",
    ):
        if criteria[k] <= 0:
            raise ValueError(k + " must be positive")
    if type(criteria["min_requests"]) is not int:
        raise ValueError("min_requests must be an integer")
    for k in ("qps_tolerance", "min_slo_fraction", "max_error_rate"):
        if criteria[k] > 1:
            raise ValueError(k + " must be a fraction")
    if not 0 < criteria["sample_s"] <= criteria["max_gap_s"] <= criteria["measure_s"]:
        raise ValueError("invalid sample coverage budget")
    if criteria["max_error_rate"] != 0:
        raise ValueError("performance gate requires 100% request success")
    if "engine_tps" in criteria:
        bounds = criteria["engine_tps"]
        if (not isinstance(bounds, dict) or set(bounds) != set(ENGINE_TPS)
                or any(not finite(v) or v <= 0 for v in bounds.values())):
            raise ValueError("engine_tps requires all three positive absolute floors")
    return criteria


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
    groups = {name: {} for name in ENGINE_TPS}
    for row in evidence.get("engine_tps_samples", []):
        labels = row["metric"]
        name = labels.get("__name__")
        if name not in groups:
            continue
        role = ENGINE_TPS[name]
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
        expected = evidence["provenance"]["topology"][ENGINE_TPS[name]]
        if type(expected) is not int or expected <= 0 or len(engines) != expected:
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
        metrics[name + "_engine_count"] = expected
        checks.append(dict(metric=name, actual=value, bound=bounds[name], direction="min",
                           status="PASS" if value >= bounds[name] else "FAIL"))
    return metrics, checks


def trace_workload_sha(path):
    """Ignore only run-local request identity; keep exact tokens, order and lengths."""
    h = hashlib.sha256()
    with Path(path).open() as stream:
        for line in stream:
            row = json.loads(line)
            row.pop("rid", None)
            h.update(
                (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
            )
    return h.hexdigest()


def percentile(values, q=0.99):
    return sorted(values)[max(0, math.ceil(len(values) * q) - 1)] if values else None


def analyze(evidence):
    errors = []
    result = dict(
        schema_version=1,
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
        c = validate(evidence["criteria"])
        if evidence.get("schema_version") != 1:
            raise ValueError("unsupported evidence version")
        p = evidence["provenance"]
        for k in REQUIRED_PROVENANCE:
            if not p.get(k):
                raise ValueError("missing provenance: " + k)
        if p["benchmark_id"] != c["benchmark_id"]:
            raise ValueError("benchmark identity mismatch")
        for v in (
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
            raise ValueError("observer coverage gap")
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
            v is not None and (v >= c[key] if direction == "min" else v <= c[key])
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
    except (KeyError, TypeError, ValueError) as exc:
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


def write_evidence(path, evidence):
    """Atomic compact JSON; raw journals remain the source of detailed RPC fields."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(evidence, stream, separators=(",", ":"), allow_nan=False)
    temporary.replace(path)


def report(directory, evidence, result=None, telemetry_directory=None):
    result = analyze(evidence) if result is None else result
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    write_evidence(directory / "performance-gate-evidence.json", evidence)
    p = evidence.get("provenance", {})
    from workload.performance_views import panel

    chart, monitoring = panel(telemetry_directory or directory, evidence, result)
    spec = dict(
        title="Master 性能绝对门禁",
        subtitle=result["verdict"],
        timeAxis=dict(min=0, max=evidence.get("criteria", {}).get("measure_s", 1)),
        panels=[chart],
        sections=[
            table(
                "绝对标准",
                ["指标", "实际值", "标准", "结果"],
                [
                    [
                        x["metric"],
                        x["actual"],
                        str(x["direction"]) + " " + str(x["bound"]),
                        x["status"],
                    ]
                    for x in result["checks"]
                ],
            ),
            details("监控曲线来源与缺采", monitoring),
            details("有效性", result["errors"]),
            details("指标", result["metrics"]),
        ],
    )
    return write_bundle(
        directory,
        "run",
        "master-performance",
        result,
        spec,
        meta=run_meta(
            dict(id="master-performance", verdict=result["verdict"]),
            implementation=p.get("master_artifact"),
            workload=p.get("trace"),
            configuration=p,
            evidence=dict(file="performance-gate-evidence.json"),
        ),
        producer="performance-gate",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--json-only", action="store_true", help="write evidence and verdict without HTML")
    args = parser.parse_args()
    e = json.loads(args.evidence.read_text())
    r = analyze(e)
    if args.json_only:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "performance-gate-evidence.json").write_text(json.dumps(e, allow_nan=False))
        (args.output / "analysis.json").write_text(json.dumps(r, indent=2, allow_nan=False))
    else:
        report(args.output, e, r, args.evidence.parent)
    print(json.dumps(r, allow_nan=False))
    return {"PASS": 0, "FAIL": 1, "INVALID": 2}[r["verdict"]]


if __name__ == "__main__":
    raise SystemExit(main())
