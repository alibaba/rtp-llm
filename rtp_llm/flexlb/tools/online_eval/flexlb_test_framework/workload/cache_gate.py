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
    table,
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
    from statistics import median

    rows = evidence["samples"]
    anchor = rows[0]["epoch_s"] - rows[0]["t"] if rows else 0
    series, sources, gaps, errors = archived_series(directory, anchor)
    axes = {
        key: dict(title=label, position=position)
        for key, label, position in (
            ("queue", "每引擎 streams", "left"),
            ("count", "数量", "right"),
            ("p", "引擎数", "right"),
            ("qps", "请求 / 秒", "right"),
            ("tokens", "tokens / 秒", "right"),
            ("ms", "毫秒", "right"),
            ("blocks", "KV blocks", "right"),
            ("seconds", "秒", "right"),
            ("ratio", "命中率", "right"),
        )
    }
    # Report vocabulary is intentionally explicit. Prometheus identities remain
    # available in the audit table, but never leak into the visual legend.
    metric_defs = {
        ("mock", "running_avg"): ("P Running / engine", "队列", "queue", "streams", "#1677ff", False),
        ("mock", "running_max"): ("P Running max", "队列", "queue", "streams", "#69b1ff", True),
        ("mock", "waiting_avg"): ("P Waiting / engine", "队列", "queue", "streams", "#f5222d", False),
        ("mock", "waiting_max"): ("P Waiting max", "队列", "queue", "streams", "#ff7875", True),
        ("mock", "engine_count"): ("P engine count", "规模", "count", "engines", "#722ed1", False),
        ("mock", "cache_hit_ratio"): ("P cache hit ratio", "缓存", "ratio", "", "#13c2c2", False),
        ("mock", "context_wall_tps"): ("P compute token throughput", "性能", "tokens", "tokens/s", "#52c41a", True),
        ("mock", "context_execution_tps_avg"): ("P model forward TPS", "性能", "tokens", "tokens/s", "#389e0d", False),
        ("mock", "context_execution_tps_with_cache_avg"): ("P model forward TPS incl. cache", "性能", "tokens", "tokens/s", "#95de64", True),
        ("mock", "simulated_prefill_ms_avg"): ("P simulated model forward", "性能", "ms", "ms", "#fa8c16", True),
        ("mock", "context_completed_qps"): ("P completed QPS", "流量", "qps", "req/s", "#08979c", False),
        ("mock", "accepted_qps"): ("P accepted QPS", "流量", "qps", "req/s", "#36cfc9", True),
        ("mock", "rtp_llm_kv_cache_pool_total_blocks"): ("P KV total blocks", "KV", "blocks", "blocks", "#531dab", True),
        ("mock", "rtp_llm_kv_cache_pool_available_blocks"): ("P KV available blocks", "KV", "blocks", "blocks", "#b37feb", True),
        ("mock", "mock_engine_held_blocks"): ("P held blocks", "KV", "blocks", "blocks", "#ad6800", True),
        ("mock", "mock_engine_referenced_blocks"): ("P referenced blocks", "KV", "blocks", "blocks", "#d48806", True),
        ("client", "actual_send_qps"): ("Client sent QPS", "流量", "qps", "req/s", "#2f54eb", False),
        ("client", "success_qps"): ("Client success QPS", "流量", "qps", "req/s", "#52c41a", False),
        ("client", "error_qps"): ("Client error QPS", "流量", "qps", "req/s", "#cf1322", False),
        ("client", "completed_qps"): ("Client completed QPS", "流量", "qps", "req/s", "#597ef7", True),
        ("client", "ttft_p99_seconds"): ("TTFT p99", "延迟", "seconds", "s", "#fa541c", True),
        ("client", "total_p99_seconds"): ("Total latency p99", "延迟", "seconds", "s", "#faad14", True),
        ("client", "schedule_p99_seconds"): ("Schedule latency p99", "延迟", "seconds", "s", "#d4b106", True),
        ("master", "arrivals_qps"): ("Master arrival QPS", "流量", "qps", "req/s", "#1d39c4", True),
        ("master", "completions_qps"): ("Master completion QPS", "流量", "qps", "req/s", "#237804", True),
        ("master", "flexlb_app_flexlb_batcher_queue_size"): ("Master batcher queue", "Master", "count", "requests", "#c41d7f", True),
        ("master", "flexlb_app_flexlb_scheduler_inflight_size"): ("Master scheduler inflight", "Master", "count", "requests", "#eb2f96", True),
        ("master", "flexlb_app_flexlb_inflight_request_count"): ("Master inflight requests", "Master", "count", "requests", "#9e1068", False),
        ("master", "flexlb_auto_tpm_decode_reserved_count"): ("Master decode reserved", "Master", "count", "requests", "#7cb305", True),
        ("master", "flexlb_auto_tpm_decode_running_count"): ("Master decode running", "Master", "count", "requests", "#a0d911", True),
    }
    curves = []
    audit = []
    found = set()
    for key, points in series.items():
        _, source, metric, label_json = key.split("/", 3)
        if metric == "up":
            continue
        labels = json.loads(label_json)
        if labels.get("role") == "decode":
            continue
        kind = "mock" if source == "mock" else "client" if source.startswith("client-") else "master"
        definition = metric_defs.get((kind, metric))
        if not definition:
            continue
        name, group, axis, unit, color, hidden = definition
        qualifiers = [str(value) for label, value in sorted(labels.items()) if label not in {"role"}]
        if qualifiers:
            name += " · " + ", ".join(qualifiers)
        found.add((kind, metric))
        visible_end = max((r["t"] for r in rows), default=0)
        valid_points = sum(
            value is not None and 0 <= timestamp <= visible_end
            for timestamp, value in points
        )
        expected = max(1, round((visible_end + 1) / max(sources[key].get("step", 1), 0.001)))
        coverage = min(1, valid_points / expected)
        curves.append(
            dict(
                name=name,
                group=group,
                axis=axis,
                unit=unit,
                color=color,
                points=[dict(x=t, y=v) for t, v in points],
                hidden=hidden,
                description=sources[key]["promql"],
            )
        )
        audit.append([name, f"{coverage:.0%}", "OK" if coverage >= 0.8 else "SPARSE", sources[key]["promql"]])
    for identity, definition in metric_defs.items():
        if identity not in found and identity in {
            ("mock", "context_completed_qps"), ("master", "completions_qps")
        }:
            audit.append([definition[0], "0%", "MISSING", "本次归档没有该监控序列"])

    by_name = {curve["name"]: curve for curve in curves}
    def values(name, start=None, end=None):
        return [
            point["y"]
            for point in by_name.get(name, {}).get("points", [])
            if point["y"] is not None
            and (start is None or point["x"] >= start)
            and (end is None or point["x"] <= end)
        ]
    baseline_start, baseline_end = evidence.get("baseline_start"), evidence.get("baseline_end")
    sent = values("Client sent QPS", baseline_start, baseline_end)
    success = values("Client success QPS", baseline_start, baseline_end)
    failures = values("Client error QPS", baseline_start, baseline_end)
    monitor_warnings = [str(error) for error in errors]
    if sent and success and median(sent) > 0 and median(success) / median(sent) < 0.9:
        monitor_warnings.append(
            f"缩容前负载无效：success/send 中位数仅 {median(success) / median(sent):.1%}"
        )
    if sent and failures and median(sent) > 0 and median(failures) / median(sent) > 0.05:
        monitor_warnings.append(
            f"客户端错误已主导流量：error/send 中位数 {median(failures) / median(sent):.1%}"
        )
    monitoring_status = "INVALID" if monitor_warnings else "OK"
    presets = {
        "核心": ["P cache hit ratio", "P Waiting / engine", "P Running / engine", "P engine count", "Client sent QPS", "Client success QPS", "Client error QPS", "Master inflight requests"],
        "队列": [name for name in by_name if "Waiting" in name or "Running" in name or "queue" in name or "inflight" in name],
        "流量": [curve["name"] for curve in curves if curve["group"] == "流量"],
        "性能": [curve["name"] for curve in curves if curve["group"] in {"性能", "延迟"}],
        "KV": [curve["name"] for curve in curves if curve["group"] == "KV"],
    }
    return dict(
        run_id="cache-scale-in",
        title="P scale-in cache-collapse gate",
        subtitle=f'{result["verdict"]} · monitoring {monitoring_status}',
        meta=dict(
            params=evidence["criteria"],
            sampling="Prometheus queries only",
            sources=dict(
                aggregate="Prometheus queries.json",
                engineDist="not used",
                runDir=str(Path(directory).resolve()),
            ),
        ),
        timeOriginLabel="Seconds since observation began",
        events=evidence.get("events", []),
        kpis=[
            dict(label="Gate verdict", value=result["verdict"]),
            dict(label="Monitoring / load validity", value=monitoring_status, tone="danger" if monitor_warnings else "success"),
        ],
        panels=[
            dict(
                id="cache-overlay",
                title="监控聚合曲线",
                overlay=True,
                axes=axes,
                series=curves,
                presets=presets,
                caption=(
                    "全部曲线来自 Prometheus；默认展示关键指标。单击勾选，双击图例可隔离曲线，悬停图例可高亮。waiting/running 使用每引擎平均值。"
                    if curves
                    else "缺少监控数据；不从 snapshot、日志或请求文件补算曲线。"
                ),
            )
        ],
        sections=[
            table("监控曲线审计", ["曲线", "覆盖率", "状态", "PromQL / 说明"], audit),
            details("实验与监控有效性", dict(status=monitoring_status, warnings=monitor_warnings)),
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
