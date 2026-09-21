"""Offline cache-collapse adjudication from identity-preserving counter samples.

No HTTP, wall-clock waits or implicit threshold tuning. The same JSON supplies
both CI checks and the existing Chart.js renderer. Queues are diagnostic only.
"""

import argparse
from bisect import bisect_right
import json
import math
from pathlib import Path

from stress.canvas_report_render_html import render

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
    if not times or any(type(t) not in (int, float) or not math.isfinite(t) or t < 0
                        for t in times):
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
        issued_count=len(times), first_send_epoch_ms=times[0], last_send_epoch_ms=times[-1])


def window(rows, start, end, names, max_gap_s):
    selected = [r for r in rows if start <= r["t"] <= end]
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
            if any(
                type(v) not in (int, float) or not math.isfinite(v) or v < 0
                for v in counter
            ):
                result["errors"].append("missing counter " + name + "/" + field)
            elif any(b < a for a, b in zip(counter, counter[1:])):
                result["errors"].append("counter reset " + name + "/" + field)
            else:
                totals[field] += counter[-1] - counter[0]
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


def write_report(directory, evidence, result):
    directory = Path(directory)
    (directory / "cache-gate-evidence.json").write_text(
        json.dumps(evidence, indent=2, allow_nan=False)
    )
    (directory / "cache-gate-result.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)
    )
    rows, windows = evidence["samples"], result["windows"]
    panels = []

    def panel(title, points, fields, caption):
        panels.append(
            dict(
                id="gate-" + str(len(panels)),
                title=title,
                caption=caption,
                type="line",
                timeX=True,
                x=[str(r["t"]) for r in points],
                xNums=[r["t"] for r in points],
                series=[
                    dict(name=label, data=[r.get(key) for r in points], color=color)
                    for key, label, color in fields
                ],
            )
        )

    hit_points = []
    timeline_windows = []
    withdrew = evidence["post_start"] > 0 or any(event["name"] == "withdraw_start" for event in evidence["events"])
    t = evidence["criteria"]["step_s"]
    while rows and t <= rows[-1]["t"]:
        start = max(0, t - evidence["criteria"]["window_s"])
        names = (
            evidence["initial_engines"]
            if not withdrew or t <= evidence["baseline_end"]
            else evidence["survivors"]
        )
        crossing = start < evidence["post_start"] and t > evidence["baseline_end"]
        w = window(rows, start, t, names, evidence["criteria"]["max_gap_s"])
        timeline_windows.append(w)
        survivor = window(rows, start, t, evidence["survivors"], evidence["criteria"]["max_gap_s"])
        hit_points.append(
            dict(
                t=t,
                hit=None if crossing or w["errors"] else w["hit"],
                survivor_hit=None if survivor["errors"] else survivor["hit"],
                floor=result["threshold"],
            )
        )
        t += evidence["criteria"]["step_s"]
    panel(
        "Token cache hit rate",
        hit_points,
        [("hit", "rolling token hit", "#2563eb"), ("survivor_hit", "survivor token hit", "#0d9488"), ("floor", "gate floor", "#dc2626")],
        result["semantics"],
    )
    panel(
        "P topology",
        rows,
        [("master_p", "master alive P", "#2563eb")],
        "Actual master topology; event times are listed below.",
    )
    panel(
        "Prefill backlog",
        rows,
        [
            ("waiting", "waiting total", "#dc2626"),
            ("running", "running total", "#16a34a"),
        ],
        "Observation only; backlog does not independently fail the gate.",
    )
    points = [
        dict(
            t=w["end"],
            **{
                k: w[k] for k in ("sent_qps", "terminal_qps", "completed", "forward_ms")
            },
        )
        for w in timeline_windows
    ]
    panel(
        "Client traffic",
        points,
        [
            ("sent_qps", "actual sent QPS", "#2563eb"),
            ("terminal_qps", "terminal QPS (success + error)", "#dc2626"),
        ],
        "Rolling windows; rejected requests are included in terminal QPS.",
    )
    panel(
        "Model forward",
        points,
        [("forward_ms", "mean engine recent forward ms", "#2563eb")],
        "Unweighted mean of per-engine recent averages; diagnostic, not pooled request latency.",
    )
    evictions = [
        dict(
            t=w["end"],
            rate=(
                w["evictions"] / (w["end"] - w["start"])
                if w["evictions"] is not None
                else None
            ),
        )
        for w in timeline_windows
    ]
    panel(
        "Device eviction rate",
        evictions,
        [("rate", "evicted blocks/s", "#dc2626")],
        "Counter deltas over surviving P, in the same rolling windows as the gate.",
    )
    completed = evidence.get("completion_series", [])
    if completed:
        panel(
            "Successful / failed completion QPS",
            completed,
            [
                ("success", "success", "#16a34a"),
                ("failure", "failure / rejection", "#dc2626"),
                ("prefill", "completed prefill", "#2563eb"),
            ],
            "One-second completion-time cohorts, including rejected requests.",
        )
        panel(
            "Device / Memory reuse",
            completed,
            [
                ("device_tokens", "device tokens/request", "#2563eb"),
                ("memory_tokens", "memory tokens/request", "#9333ea"),
            ],
            "Mean over completed prefill requests; unavailable instrumentation is a gap.",
        )
        panel(
            "Executed context batch size",
            completed,
            [("batch_size", "request-weighted batch size", "#2563eb")],
            "Request-weighted mean of executed prefill batch sizes.",
        )
        panel(
            "TTFT",
            completed,
            [("ttft_p95_ms", "successful TTFT p95 ms", "#2563eb")],
            "Completion-time cohorts; no success means a gap.",
        )
    # The shared panel accepts independently sampled curves on one time axis.
    mapping = {
        "rolling token hit": ("命中率", "pct", "%", 100, True),
        "survivor token hit": ("保留 P 命中率", "pct", "%", 100, True),
        "gate floor": ("命中率门限", "pct", "%", 100, False),
        "master alive P": ("P 数", "p", "P", 1, True),
        "waiting total": ("waiting 总数", "queue", "请求", 1, True),
        "running total": ("running 总数", "queue", "请求", 1, False),
        "actual sent QPS": ("发送 QPS", "qps", "QPS", 1, True),
        "terminal QPS (success + error)": ("完成 QPS", "qps", "QPS", 1, False),
        "success": ("成功 QPS", "qps", "QPS", 1, True),
        "failure / rejection": ("失败 QPS", "qps", "QPS", 1, True),
        "completed prefill": ("Prefill 完成 QPS", "qps", "QPS", 1, False),
        "mean engine recent forward ms": ("Model forward", "ms", "ms", 1, False),
        "successful TTFT p95 ms": ("TTFT p95", "ms", "ms", 1, False),
        "device tokens/request": ("Device reuse", "tokens", "tokens/request", 1, False),
        "memory tokens/request": ("Memory reuse", "tokens", "tokens/request", 1, False),
        "request-weighted batch size": (
            "Context batch size",
            "batch",
            "请求/batch",
            1,
            False,
        ),
        "evicted blocks/s": ("Device 驱逐", "evictions", "blocks/s", 1, False),
    }
    colors = [
        "#2563eb",
        "#dc2626",
        "#111827",
        "#e76f00",
        "#9333ea",
        "#0891b2",
        "#64748b",
        "#16a34a",
        "#e11d48",
        "#0d9488",
        "#7c3aed",
        "#be185d",
        "#0284c7",
        "#a16207",
        "#4f46e5",
        "#b91c1c",
    ]
    semantic_colors = {
        "命中率": "#2563eb",
        "命中率门限": "#dc2626",
        "P 数": "#111827",
        "waiting 总数": "#e76f00",
        "running 总数": "#9333ea",
        "发送 QPS": "#0891b2",
        "成功 QPS": "#16a34a",
        "失败 QPS": "#dc2626",
        "Model forward": "#7c3aed",
        "TTFT p95": "#be185d",
    }
    curves = []
    for p in panels:
        for series in p["series"]:
            name, axis, unit, factor, visible = mapping[series["name"]]
            curves.append(
                dict(
                    name=name,
                    axis=axis,
                    unit=unit,
                    hidden=not visible,
                    color=semantic_colors.get(name, colors[len(curves) % len(colors)]),
                    points=[
                        dict(x=t, y=value * factor if value is not None else None)
                        for t, value in zip(p["xNums"], series["data"])
                    ],
                )
            )
    axes = {
        key: dict(title=title, position=side)
        for key, title, side in (
            ("pct", "命中率 %", "left"),
            ("qps", "QPS", "right"),
            ("queue", "排队 / 运行请求数", "left"),
            ("p", "P 数", "right"),
            ("ms", "耗时 ms", "right"),
            ("tokens", "复用 tokens/request", "left"),
            ("batch", "请求/batch", "left"),
            ("evictions", "驱逐 blocks/s", "right"),
        )
    }
    axes["pct"].update(min=0, max=100)
    panels = [
        dict(
            id="cache-overlay",
            title="缩 P 实验 · 关键曲线",
            overlay=True,
            caption=f"共用时间轴；各单位独立坐标轴。滚动窗口为 {evidence['criteria']['window_s']} 秒，起始阶段使用已有采样，至少两个采样点才计算速率。命中率取完成侧 token：主曲线隐藏跨缩容及收敛期的窗口；保留 P 命中率全程只统计最终保留的同一组 P，用于观察过渡，不参与门禁判定。零完成 token 或采样异常仍留空。成功/失败按完成时刻统计，不能与同窗发送直接相减。",
            axes=axes,
            series=curves,
            presets={
                "核心指标": [
                    "命中率",
                    "保留 P 命中率",
                    "P 数",
                    "waiting 总数",
                    "发送 QPS",
                    "成功 QPS",
                    "失败 QPS",
                ],
                "缓存与排队": [
                    "命中率",
                    "保留 P 命中率",
                    "命中率门限",
                    "P 数",
                    "waiting 总数",
                    "Device reuse",
                    "Memory reuse",
                ],
                "耗时与负载": [
                    "P 数",
                    "Model forward",
                    "TTFT p95",
                    "Context batch size",
                    "running 总数",
                ],
            },
        )
    ]
    import html

    decode_count = evidence.get("provenance", {}).get("topology", {}).get("decode")
    decode_label = f"{decode_count}D · " if decode_count is not None else ""
    spec = dict(
        run_id="cache-scale-in",
        title="P scale-in cache-collapse gate",
        subtitle=(
            f"{result['verdict']} · {len(evidence['initial_engines'])}P → {evidence['criteria'].get('target_p', len(evidence['survivors']))}P · "
            f"{decode_label}{evidence['criteria']['qps']} QPS · 缩后观察 {evidence['criteria']['observe_s']}s"
        ),
        meta=dict(
            params=evidence["criteria"],
            sampling="Token hit uses pooled completion-counter deltas; traffic curves use completion-time cohorts.",
            sources=dict(
                runDir=str(directory),
                aggregate=str(directory / "cache-gate-result.json"),
            ),
        ),
        timeOriginLabel="Seconds since observation began",
        events=evidence.get("events", []),
        kpis=[
            dict(label="Verdict", value=result["verdict"]),
            dict(label="Longest low hit", value=str(result["longest_low_s"]) + "s"),
        ],
        panels=panels,
        timeAxis=dict(min=0, max=max((r["t"] for r in rows), default=1)),
    )
    detail = (
        "<details><summary>判据明细与事件</summary><pre>"
        + html.escape(json.dumps(result, indent=2))
        + "</pre></details>"
    )
    (directory / "cache-gate.html").write_text(
        render(spec).replace("</body>", detail + "</body>")
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


def completion_series(records, engine_events, samples):
    """Completed-time cohorts; never label issue-time counts as completion QPS."""
    if not samples:
        return []
    origin = samples[0]["epoch_s"] - samples[0]["t"]
    bins = {
        i: dict(
            t=i,
            success=0,
            failure=0,
            prefill=0,
            input_tokens=0,
            device_tokens=0,
            memory_tokens=0,
            exec_ms=0,
            batch_size=0,
            ttft=[],
        )
        for i in range(math.ceil(samples[-1]["t"]) + 1)
    }
    for r in records:
        sent, total = r.get("send_start_epoch_ms"), r.get("total_ms")
        if type(sent) not in (int, float) or type(total) not in (int, float):
            continue
        bucket = math.floor((sent + total) / 1000 - origin)
        if bucket not in bins:
            continue
        b = bins[bucket]
        b["success" if r.get("status") == "ok" else "failure"] += 1
        if r.get("status") == "ok" and type(r.get("ttft_ms")) in (int, float):
            b["ttft"].append(r["ttft_ms"])
    if Path(engine_events).is_file():
        with Path(engine_events).open() as source:
            for line in source:
                r = json.loads(line)
                if r.get("event") != "prefill_done" or r.get("cancelled"):
                    continue
                bucket = math.floor(r["prefill_done_ms"] / 1000 - origin)
                if bucket not in bins:
                    continue
                b = bins[bucket]
                b["prefill"] += 1
                for key, field in (
                    ("input_tokens", "input_len"),
                    ("device_tokens", "cache_device_hit_tokens"),
                    ("memory_tokens", "cache_memory_hit_tokens"),
                    ("exec_ms", "exec_ms"),
                    ("batch_size", "batch_size"),
                ):
                    value = r.get(field)
                    b[key] = None if value is None or b[key] is None else b[key] + value
    for b in bins.values():
        n = b["prefill"]
        for key in ("device_tokens", "memory_tokens", "exec_ms", "batch_size"):
            b[key] = b[key] / n if n and b[key] is not None else None
        times = sorted(b.pop("ttft"))
        b["ttft_p95_ms"] = times[math.ceil(0.95 * len(times)) - 1] if times else None
    return list(bins.values())


def report_series(directory, anchor_epoch_s):
    """Expose gate curves through the common offline timeline/sweep interface."""
    directory = Path(directory)
    path = directory / "cache-gate-evidence.json"
    result_path = directory / "cache-gate-result.json"
    if not path.is_file() or not result_path.is_file():
        return {}, {}
    e = json.loads(path.read_text())
    r = json.loads(result_path.read_text())
    rows = e["samples"]
    if not rows:
        return {}, {}
    offset = rows[0]["epoch_s"] - rows[0]["t"] - anchor_epoch_s
    series = {
        "gate/token_hit": [[w["end"] + offset, w["hit"]] for w in r["windows"]],
        "gate/hit_floor": [[w["end"] + offset, r["threshold"]] for w in r["windows"]],
        "gate/p": [[row["t"] + offset, row["master_p"]] for row in rows],
        "gate/waiting": [[row["t"] + offset, row["waiting"]] for row in rows],
        "gate/running": [[row["t"] + offset, row["running"]] for row in rows],
    }
    for field in (
        "success",
        "failure",
        "prefill",
        "device_tokens",
        "memory_tokens",
        "batch_size",
        "ttft_p95_ms",
    ):
        series["gate/" + field] = [
            [row["t"] + offset, row[field]] for row in e.get("completion_series", [])
        ]
    sources = {
        key: dict(
            path=str(path),
            result=str(result_path),
            time_basis="observed sample / completion time",
            gate_verdict=r["verdict"],
        )
        for key in series
    }
    return series, sources


if __name__ == "__main__":
    raise SystemExit(main())
