"""Offline cache-collapse adjudication from identity-preserving counter samples.

No HTTP, wall-clock waits or implicit threshold tuning. The same JSON supplies
both CI checks and the existing Chart.js renderer. Queues are diagnostic only.
"""

import argparse
import hashlib
from bisect import bisect_right
import json
import math
from pathlib import Path

from online_eval.reporting import (
    details,
    run_meta,
    write_bundle,
)
from online_eval.reporting.statistics import select_window, counter_delta

COUNTERS = (
    "hit_tokens_total",
    "context_tokens_total",
    "context_requests_total",
    "cache_evictions",
)


def align_send_counters(evidence, issued):
    """Use actual send times, not the time a buffered lifecycle journal was read.

    The caller must first prove complete client accounting. Preserve the live
    observed counts so delayed journal consumption remains independently visible.
    """
    times = [row.get("send_start_epoch_ms") for row in issued]
    if not times or any(
        type(t) not in (int, float) or not math.isfinite(t) or t < 0 for t in times
    ):
        raise ValueError("complete finite issued timestamps are required")
    times.sort()
    for row in evidence["samples"]:
        epoch = row.get("epoch_s")
        if type(epoch) not in (int, float) or not math.isfinite(epoch):
            raise ValueError("finite sample epoch is required")
        row.setdefault("journal_observed_started", row["started"])
        row["started"] = bisect_right(times, epoch * 1000)
    evidence["send_counter_alignment"] = dict(
        method="issued send_start_epoch_ms at each sample epoch",
        issued_count=len(times),
        first_send_epoch_ms=times[0],
        last_send_epoch_ms=times[-1],
    )


def window(rows, start, end, names, max_gap_s):
    selected = select_window(rows, start, end, time=lambda r: r["t"], include_end=True)
    result = dict(
        start=start,
        end=end,
        hit=None,
        completed=0,
        sent_qps=None,
        terminal_qps=None,
        waiting=None,
        running=None,
        evictions=None,
        forward_ms=None,
        errors=[],
    )
    if (
        len(selected) < 2
        or selected[0]["t"] - start > max_gap_s
        or end - selected[-1]["t"] > max_gap_s
    ):
        result["errors"].append("incomplete window coverage")
        return result
    elapsed = selected[-1]["t"] - selected[0]["t"]
    if elapsed <= 0 or any(
        b["t"] - a["t"] > max_gap_s for a, b in zip(selected, selected[1:])
    ):
        result["errors"].append("sampling gap or non-increasing clock")
        return result
    totals = dict.fromkeys(COUNTERS, 0)
    for name in names:
        values = [r["engines"].get(name) for r in selected]
        if (
            any(v is None for v in values)
            or len({(v.get("grpc_addr"), v.get("engine_incarnation")) for v in values})
            != 1
        ):
            result["errors"].append("missing or changed engine " + name)
            continue
        for field in COUNTERS:
            counter = [v.get(field) for v in values]
            delta, state = counter_delta(counter)
            if state == "MISSING_COUNTER":
                result["errors"].append("missing counter " + name + "/" + field)
            elif state == "COUNTER_RESET":
                result["errors"].append("counter reset " + name + "/" + field)
            else:
                totals[field] += delta
    for field, target in (("started", "sent_qps"), ("terminal", "terminal_qps")):
        counts = [r[field] for r in selected]
        if any(b < a for a, b in zip(counts, counts[1:])):
            result["errors"].append("client counter reset")
        else:
            result[target] = (counts[-1] - counts[0]) / elapsed
    if not result["errors"]:
        tokens = totals["context_tokens_total"]
        hits = totals["hit_tokens_total"]
        if hits > tokens:
            result["errors"].append("hit tokens exceed context tokens")
        else:
            result["hit"] = hits / tokens if tokens else None
        result.update(
            completed=totals["context_requests_total"],
            context_tokens=tokens,
            evictions=totals["cache_evictions"],
        )
        result["waiting"] = max(
            sum(r["engines"][n]["waiting"] for n in names) for r in selected
        )
        result["running"] = max(
            sum(r["engines"][n]["running"] for n in names) for r in selected
        )
        result["forward_ms"] = sum(
            selected[-1]["engines"][n]["prefill_ms_avg"] for n in names
        ) / len(names)
    return result


def analyze(evidence):
    p, rows = evidence["criteria"], evidence["samples"]
    errors = list(evidence.get("errors", []))
    if any(b["t"] <= a["t"] for a, b in zip(rows, rows[1:])):
        errors.append("sample clock must increase")
    baseline = window(
        rows,
        evidence["baseline_start"],
        evidence["baseline_end"],
        evidence["initial_engines"],
        p["max_gap_s"],
    )
    errors.extend(baseline["errors"])
    if baseline["hit"] is None or baseline["hit"] < p["baseline_min_hit"]:
        errors.append("baseline not warm")
    if baseline["completed"] < p["min_completed"]:
        errors.append("insufficient baseline prefill completions")
    if (
        baseline["sent_qps"] is None
        or abs(baseline["sent_qps"] / p["qps"] - 1) > p["qps_tolerance"]
    ):
        errors.append("baseline sending rate differs from target")
    start, end = evidence["post_start"], evidence["post_end"]
    if end - start < p["observe_s"] - p["max_gap_s"]:
        errors.append("observation too short")
    threshold = max(p["absolute_min_hit"], (baseline["hit"] or 0) - p["max_drop"])
    windows = []
    t = start + p["window_s"]
    while t <= end + 1e-8:
        w = window(rows, t - p["window_s"], t, evidence["survivors"], p["max_gap_s"])
        w["low"] = w["hit"] is not None and w["hit"] < threshold
        w["valid"] = (
            not w["errors"]
            and w["hit"] is not None
            and w["completed"] >= p["min_completed"]
            and w["sent_qps"] is not None
            and abs(w["sent_qps"] / p["qps"] - 1) <= p["qps_tolerance"]
        )
        windows.append(w)
        t += p["step_s"]
    if not windows or any(not w["valid"] for w in windows):
        errors.append(
            "missing samples, insufficient prefill completions or off-target load"
        )
    post = [r for r in rows if start <= r["t"] <= end]
    if not post or any(r["master_p"] != len(evidence["survivors"]) for r in post):
        errors.append("target topology not stable")
    if (
        not windows
        or max((w["waiting"] or 0 for w in windows), default=0) < p["min_waiting"]
    ):
        errors.append("overload not exercised")
    lag = evidence.get("max_pacing_lag_ms")
    if lag is None or lag > p["max_pacing_lag_ms"]:
        errors.append("client pacing lag missing or exceeds budget")
    longest = run = 0.0
    first_low = first_collapse = recovery = None
    previous_end = None
    for w in windows:
        if w["valid"] and w["low"]:
            if first_low is None:
                first_low = w["end"]
            run = 0 if previous_end is None else run + w["end"] - previous_end
            if run >= p["sustain_s"] and first_collapse is None:
                first_collapse = w["end"]
            longest = max(longest, run)
            previous_end = w["end"]
        else:
            if first_collapse is not None and w["valid"] and recovery is None:
                recovery = w["end"]
            run, previous_end = 0, None
    verdict = (
        "INVALID" if errors else ("FAIL" if first_collapse is not None else "PASS")
    )
    return dict(
        schema_version=1,
        verdict=verdict,
        errors=sorted(set(errors)),
        threshold=threshold,
        baseline=baseline,
        windows=windows,
        longest_low_s=longest,
        first_low_s=first_low,
        first_collapse_s=first_collapse,
        first_recovery_s=recovery,
        semantics="completed-prefill token-weighted reuse; waiting diagnostic only; sustained collapse fails even if later recovered",
        criteria=p,
        events=evidence.get("events", []),
    )


def build_spec(directory, evidence, result):
    from online_eval.monitoring import archived_series

    rows = evidence["samples"]
    anchor = rows[0]["epoch_s"] - rows[0]["t"] if rows else 0
    series, sources, gaps, errors = archived_series(directory, anchor)
    axes = {
        key: dict(label=label, position=position)
        for key, label, position in (
            ("queue", "streams", "left"),
            ("p", "engines", "right"),
            ("qps", "requests/s", "right"),
            ("tokens", "tokens/s", "right"),
            ("ms", "milliseconds", "right"),
            ("blocks", "blocks", "right"),
            ("seconds", "seconds", "right"),
            ("ratio", "cache hit fraction", "right"),
        )
    }
    curves = []
    for key, points in series.items():
        if ("/mock/" not in key and "/client-" not in key) or "/up/" in key:
            continue
        if '"role": "decode"' in key:
            continue
        metric = key.split("/")[2]
        axis = (
            "ratio"
            if "ratio" in metric
            else (
                "tokens"
                if "tps" in metric
                else (
                    "qps"
                    if "qps" in metric
                    else (
                        "p"
                        if "count" in metric
                        else (
                            "seconds"
                            if "seconds" in metric
                            else (
                                "ms"
                                if "_ms_" in metric
                                else "blocks" if "blocks" in metric else "queue"
                            )
                        )
                    )
                )
            )
        )
        curves.append(
            dict(
                name=key,
                axis=axis,
                points=[dict(x=t, y=v) for t, v in points],
                hidden=metric.endswith(("_sum", "_max")),
            )
        )
    return dict(
        run_id="cache-scale-in",
        title="P scale-in cache-collapse gate",
        subtitle=result["verdict"],
        meta=dict(
            params=evidence["criteria"],
            sampling="Prometheus queries only",
            sources=sources,
        ),
        timeOriginLabel="Seconds since observation began",
        events=evidence.get("events", []),
        kpis=[dict(label="Verdict", value=result["verdict"])],
        panels=[
            dict(
                id="cache-overlay",
                title="监控聚合曲线",
                overlay=True,
                axes=axes,
                series=curves,
                caption=(
                    "Prometheus 聚合；waiting/running 默认按存活上报引擎平均。门禁保留组判据见独立证据。"
                    if curves
                    else "缺少监控数据；不从 snapshot、日志或请求文件补算曲线。"
                ),
            )
        ],
        sections=[
            details("专用门禁判据（非监控曲线）", result),
            details("监控来源与缺采", dict(queries=sources, gaps=gaps, errors=errors)),
        ],
        timeAxis=dict(min=0, max=max((r["t"] for r in rows), default=1)),
    )


def write_report(directory, evidence, result):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "cache-gate-evidence.json").write_text(
        json.dumps(evidence, indent=2, allow_nan=False)
    )
    spec = build_spec(directory, evidence, result)
    provenance = evidence.get("provenance", {})
    meta = run_meta(
        dict(id="cache-scale-in", instance=provenance.get("instance")),
        implementation=dict(
            files=provenance.get("files"), master=provenance.get("historical_master")
        ),
        workload=provenance.get("trace"),
        configuration={
            k: provenance.get(k)
            for k in ("topology", "performance", "master_config", "mock_formula_config")
        },
        environment=provenance.get("client_environment"),
        evidence=[
            dict(
                path="../../../cache-gate-evidence.json",
                sha256=hashlib.sha256(
                    (directory / "cache-gate-evidence.json").read_bytes()
                ).hexdigest(),
            )
        ],
    )
    write_bundle(
        directory,
        "run",
        "cache-scale-in",
        result,
        spec,
        meta=meta,
        producer="cache-gate",
    )
    return spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence = json.loads(args.evidence.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    result = analyze(evidence)
    write_report(args.output, evidence, result)
    print(result["verdict"])
    return {"PASS": 0, "FAIL": 1, "INVALID": 2}[result["verdict"]]


def report_series(directory, anchor_epoch_s):
    from online_eval.monitoring import archived_series

    series, sources, _, _ = archived_series(directory, anchor_epoch_s)
    return series, sources


if __name__ == "__main__":
    raise SystemExit(main())
