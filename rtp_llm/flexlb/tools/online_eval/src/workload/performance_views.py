"""Selectable request and archived Prometheus views; never affect gate decisions."""

import math
from collections import defaultdict

from monitoring.session import archived_series

COLORS = [
    "#1677ff",
    "#13c2c2",
    "#fa541c",
    "#722ed1",
    "#52c41a",
    "#eb2f96",
    "#faad14",
    "#2f54eb",
]


def panel(directory, evidence, result):
    import json

    lo = evidence.get("window", {}).get("start_epoch_ms", 0)
    duration = evidence.get("criteria", {}).get("measure_s", 1)
    curves, audit = [], []
    axes = {
        k: dict(title=title, position="left" if i == 0 else "right")
        for i, (k, title) in enumerate(
            [
                ("input", "输入 tok/s"),
                ("output", "输出 tok/s"),
                ("qps", "req/s"),
                ("ms", "ms"),
                ("count", "数量"),
                ("ratio", "比例"),
                ("tokens", "tokens"),
                ("blocks", "KV blocks"),
                ("seconds", "s"),
                ("forward", "执行 tok/s"),
            ]
        )
    }

    def add(name, group, axis, points, description, hidden=True):
        curves.append(
            dict(
                name=name,
                group=group,
                axis=axis,
                unit=axes[axis]["title"],
                points=[dict(x=t, y=v) for t, v in points],
                hidden=hidden,
                color=COLORS[len(curves) % len(COLORS)],
                description=description,
            )
        )

    for key, name, group, axis in [
        ("input_tps", "完成输入 TPS", "客户端吞吐", "input"),
        ("output_tps", "完成输出 TPS", "客户端吞吐", "output"),
        ("inflight", "Client inflight", "队列", "count"),
    ]:
        add(
            name,
            group,
            axis,
            [(w["t"], w[key]) for w in result["windows"]],
            "逐请求终态重建；TPS 按完成时间分桶，inflight 为桶末值",
            True,
        )
    sent, done = defaultdict(list), defaultdict(list)
    records = evidence.get("flow", {}).get("records", [])
    terminals_by_id = {r.get("rid"): r for r in records}
    for issued in evidence.get("flow", {}).get("issued", []):
        r = dict(
            issued,
            **{
                k: v
                for k, v in terminals_by_id.get(issued.get("rid"), {}).items()
                if k not in issued
            },
        )
        if isinstance(r.get("send_start_epoch_ms"), (int, float)):
            sent[math.floor((r["send_start_epoch_ms"] - lo) / 1000)].append(r)
    for r in records:
        if isinstance(r.get("send_start_epoch_ms"), (int, float)) and isinstance(
            r.get("total_ms"), (int, float)
        ):
            done[
                math.floor((r["send_start_epoch_ms"] + r["total_ms"] - lo) / 1000)
            ].append(r)

    def p99(values):
        values = sorted(
            v for v in values if isinstance(v, (int, float)) and math.isfinite(v)
        )
        return values[max(0, math.ceil(0.99 * len(values)) - 1)] if values else None

    metrics = defaultdict(list)
    for i in range(math.ceil(duration)):
        arrivals, terminals = sent[i], done[i]
        ok = [r for r in arrivals if r.get("status") == "ok"]
        dt = min(1, duration - i)
        vals = {
            "发送 QPS": len(arrivals) / dt,
            "完成 QPS": len(terminals) / dt,
            "成功 QPS": sum(r.get("status") == "ok" for r in terminals) / dt,
            "错误 QPS": sum(r.get("status") != "ok" for r in terminals) / dt,
            "到达 cohort 成功率": len(ok) / len(arrivals) if arrivals else None,
            "TTFT p99": p99([r.get("ttft_ms") for r in ok]),
            "E2E p99": p99([r.get("total_ms") for r in ok]),
            "TPOT p99": p99(
                [
                    (r["total_ms"] - r["ttft_ms"]) / (r["observed_output_tokens"] - 1)
                    for r in ok
                    if r.get("observed_output_tokens", 0) > 1
                    and isinstance(r.get("total_ms"), (int, float))
                    and isinstance(r.get("ttft_ms"), (int, float))
                ]
            ),
            "输入长度均值": (
                sum(r["input_len"] for r in arrivals) / len(arrivals)
                if arrivals
                else None
            ),
            "实际输出长度均值": (
                sum(r.get("observed_output_tokens", 0) for r in ok) / len(ok)
                if ok
                else None
            ),
        }
        for k, v in vals.items():
            metrics[k].append((i, v))
    for name, points in metrics.items():
        group, axis = (
            ("延迟", "ms")
            if "p99" in name
            else (
                "流量",
                "ratio" if "率" in name else "tokens" if "长度" in name else "qps",
            )
        )
        add(
            name,
            group,
            axis,
            points,
            "逐请求证据；1 秒到达 cohort 的终态/延迟，完成与错误 QPS 按完成时间；不替代整窗门禁 p99",
            True,
        )

    series, sources, gaps, errors = archived_series(directory, lo / 1000)
    for key, points in series.items():
        epoch, source, metric, label_json = key.split("/", 3)
        if metric == "up":
            continue
        labels = json.loads(label_json)
        role = {"prefill": "P", "decode": "D"}.get(labels.pop("role", ""), "")
        primary = metric in {"rtp_llm_context_tps_engine_mean", "rtp_llm_context_tps_with_cache_engine_mean"}
        if primary:
            group, axis = "Prefill TPS", "forward"
        elif metric in {"rtp_llm_context_tps_per_engine", "rtp_llm_context_tps_with_cache_per_engine"}:
            group, axis = "Prefill 逐引擎 TPS", "forward"
        elif "blocks" in metric:
            group, axis = "KV", "blocks"
        elif "ratio" in metric:
            group, axis = "KV", "ratio"
        elif "engine_count" in metric:
            group, axis = "规模", "count"
        elif any(
            s in metric for s in ("running", "waiting", "queue", "inflight", "reserved")
        ):
            group, axis = "队列", "count"
        elif "qps" in metric:
            group, axis = "流量", "qps"
        elif "seconds" in metric:
            group, axis = "延迟", "seconds"
        elif "_ms" in metric:
            group, axis = "模拟执行", "ms"
        else:
            group, axis = "模拟执行", "forward"
        name = " ".join(
            s
            for s in [
                role,
                metric.replace("flexlb_app_flexlb_", "")
                .replace("flexlb_auto_tpm_", "")
                .replace("rtp_llm_", "")
                .replace("mock_engine_", "")
                .replace("_", " "),
            ]
            if s
        )
        name = f'{source.split("-")[0]} · {name}'
        if labels:
            name += " · " + ", ".join(f"{k}={v}" for k, v in sorted(labels.items()))
        if any(c["name"] == name for c in curves):
            name += " · epoch " + epoch
        visible = [(t, v) for t, v in points if 0 <= t <= duration]
        add(name, group, axis, visible, sources[key]["promql"], not primary)
        audit.append(
            dict(
                name=name,
                samples=sum(v is not None for _, v in visible),
                **sources[key],
            )
        )
    # Frozen diagnostics also carry exact scrape samples outside queries.json.
    # In particular, decode generate TPS was absent from older query catalogs.
    raw_available = False
    archived_metrics = {key.split("/", 3)[2] for key in series}
    for name, role in (("rtp_llm_context_tps", "prefill"),
                       ("rtp_llm_context_tps_with_cache", "prefill"),
                       ("rtp_llm_generate_tps", "decode")):
        if name + "_engine_mean" in archived_metrics:
            continue
        raw_curves = raw_engine_curves(evidence, name, role)
        group = "Prefill TPS" if role == "prefill" else "Decode TPS"
        for label, points, mean in raw_curves:
            add(label, group if mean else group.replace(" TPS", " 逐引擎 TPS"),
                "forward", points, "engine_tps_samples: 同一抓取点 priority 求和，再按引擎等权平均；缺失不补零", not mean)
        if raw_curves:
            raw_available = True
            audit.append(dict(name=name, source="evidence.engine_tps_samples", series=len(raw_curves)))
    # Never open an empty chart when only request-level evidence survived.
    if not any(not c["hidden"] and any(p["y"] is not None for p in c["points"]) for c in curves):
        for c in curves:
            if c["group"] == "客户端吞吐": c["hidden"] = False
    axes["ratio"].update(min=0, max=1)
    presets = {"核心": [c["name"] for c in curves if not c["hidden"]]}
    for group in ["Prefill TPS", "Prefill 逐引擎 TPS", "Decode TPS", "Decode 逐引擎 TPS", "客户端吞吐", "延迟", "流量", "队列", "规模", "KV", "模拟执行"]:
        presets[group] = [c["name"] for c in curves if c["group"] == group]
    return dict(
        id="performance",
        title="性能与运行状态",
        overlay=True,
        axes=axes,
        series=curves,
        presets=presets,
        caption="Prefill TPS 按引擎/DP 汇总 priority，与线上 context TPS、with cache TPS 口径对应；不对引擎执行速率求集群总和。时间按测量起点对齐。Client 曲线来自逐请求证据，mock/master 曲线来自归档 Prometheus（具体查询见审计）。"
        + (" 本报告缺少监控归档，只有请求级曲线。" if not series and not raw_available else ""),
    ), dict(queries=audit, gaps=gaps, errors=errors, available=bool(series) or raw_available)


def raw_engine_curves(evidence, metric, role):
    """Plot exact scrape samples without hiding missing priorities/engines or gaps."""
    lo = evidence.get("window", {}).get("start_epoch_ms", 0) / 1000
    duration = evidence.get("criteria", {}).get("measure_s", 1)
    gap = evidence.get("criteria", {}).get("max_gap_s", 3)
    engines = {}
    for row in evidence.get("engine_tps_samples", []):
        labels = row.get("metric", {})
        if labels.get("__name__") != metric or labels.get("role") != role or not labels.get("engine_name"):
            continue
        key = (labels["engine_name"], labels.get("engine_incarnation", ""))
        values = engines.setdefault(key, {}).setdefault(labels.get("priority", "aggregate"), {})
        for stamp, value in row.get("values", []):
            if not 0 <= stamp - lo <= duration: continue
            value = float(value)
            value = value if math.isfinite(value) and value >= 0 else None
            if stamp in values and values[stamp] != value: value = None
            values[stamp] = value
    if not engines: return []
    def points(values):
        out, previous = [], None
        for stamp, value in sorted(values.items()):
            if previous is not None and stamp - previous > gap:
                out.append(((previous + stamp) / 2 - lo, None))
            out.append((stamp - lo, value)); previous = stamp
        return out
    per_engine = {}
    for key, priorities in engines.items():
        stamps = set().union(*(set(v) for v in priorities.values()))
        per_engine[key] = {t: (sum(v[t] for v in priorities.values())
            if all(v.get(t) is not None for v in priorities.values()) else None) for t in stamps}
    expected = evidence.get("provenance", {}).get("topology", {}).get(role)
    full = len(per_engine) == expected and len({k[0] for k in per_engine}) == expected
    stamps = set().union(*(set(v) for v in per_engine.values()))
    means = {t: (sum(v[t] for v in per_engine.values()) / expected
        if full and all(v.get(t) is not None for v in per_engine.values()) else None) for t in stamps}
    prefix = ("P" if role == "prefill" else "D") + " · " + metric.removeprefix("rtp_llm_").replace("_", " ")
    return [(prefix + " · engine mean", points(means), True)] + [
        (prefix + " · " + key[0] + " · " + key[1], points(values), False)
        for key, values in sorted(per_engine.items())]
