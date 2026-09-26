"""Shared report vocabulary and series colors.

Metric identities are data keys; names, axes and color are presentation only.
"""

PALETTE = (
    "#1677ff", "#52c41a", "#faad14", "#f5222d",
    "#722ed1", "#13c2c2", "#eb2f96", "#fa8c16",
    "#a0d911", "#2f54eb", "#fadb14", "#08979c",
)
WORKLOAD_COLORS = ("#2563eb", "#dc2626")
VIEW_COLORS = ("#2563eb", "#dc2626", "#16a34a", "#9333ea", "#d97706", "#0891b2")
PERFORMANCE_COLORS = ("#1677ff", "#13c2c2", "#fa541c", "#722ed1",
                      "#52c41a", "#eb2f96", "#faad14", "#2f54eb")


def performance_axes(*, include_hit_pct=False):
    names = [
        ("input", "输入 tok/s"), ("output", "输出 tok/s"),
        ("qps", "req/s"), ("ms", "ms"), ("count", "数量"),
        ("ratio", "比例"),
    ]
    if include_hit_pct:
        names.append(("hit_pct", "命中率 %"))
    names.extend((
        ("tokens", "tokens"), ("blocks", "KV blocks"),
        ("seconds", "s"), ("forward", "执行 tok/s"),
    ))
    return {key: dict(title=title, position="left" if index == 0 else "right")
            for index, (key, title) in enumerate(names)}


def performance_metric_style(source, metric, role, *, include_hit_pct=False):
    """Name and place an archived metric without changing its measured values."""
    if include_hit_pct and metric == "cache_hit_ratio" and role == "P":
        return "P 实际 token 命中率", "缓存命中率", "hit_pct", True
    primary = metric in {"rtp_llm_context_tps_engine_mean",
                         "rtp_llm_context_tps_with_cache_engine_mean"}
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
    elif any(part in metric for part in ("running", "waiting", "queue", "inflight", "reserved")):
        group, axis = "队列", "count"
    elif "qps" in metric:
        group, axis = "流量", "qps"
    elif "seconds" in metric:
        group, axis = "延迟", "seconds"
    elif "_ms" in metric:
        group, axis = "模拟执行", "ms"
    else:
        group, axis = "模拟执行", "forward"
    name = " ".join(part for part in (
        role,
        metric.replace("flexlb_app_flexlb_", "")
        .replace("flexlb_auto_tpm_", "")
        .replace("rtp_llm_", "")
        .replace("mock_engine_", "")
        .replace("_", " "),
    ) if part)
    return f'{source.split("-")[0]} · {name}', group, axis, primary

TONE_TO_COLOR = {
    "primary": PALETTE[0], "success": PALETTE[1],
    "warning": PALETTE[2], "warn": PALETTE[2],
    "danger": PALETTE[3], "info": PALETTE[5],
    "secondary": PALETTE[4], "tertiary": PALETTE[6],
    "quaternary": PALETTE[7], "neutral": "#8c8c8c",
}
KPI_TONE_COLOR = {
    "success": PALETTE[1], "danger": PALETTE[3],
    "warn": PALETTE[2], "warning": PALETTE[2],
    "info": PALETTE[0], "primary": PALETTE[0],
}


def series_color(tone, index):
    return TONE_TO_COLOR.get(tone, PALETTE[index % len(PALETTE)])


CACHE_METRICS = {
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
        ("master", "completions_qps"): ("Master legacy schedule response QPS", "流量", "qps", "req/s", "#237804", True),
        ("master", "schedule_responses_qps"): ("Master schedule response QPS", "流量", "qps", "req/s", "#237804", True),
        ("master", "flexlb_app_flexlb_batcher_queue_size"): ("Master batcher queue", "Master", "count", "requests", "#c41d7f", True),
        ("master", "flexlb_app_flexlb_scheduler_inflight_size"): ("Master scheduler inflight", "Master", "count", "requests", "#eb2f96", True),
        ("master", "flexlb_app_flexlb_inflight_request_count"): ("Master inflight requests", "Master", "count", "requests", "#9e1068", False),
        ("master", "flexlb_auto_tpm_decode_reserved_count"): ("Master decode reserved", "Master", "count", "requests", "#7cb305", True),
        ("master", "flexlb_auto_tpm_decode_running_count"): ("Master decode running", "Master", "count", "requests", "#a0d911", True),
    }

CACHE_AB_COLORS = {
    "P cache hit ratio": ("#d4380d", PALETTE[0]),
    "P Waiting / engine": ("#cf1322", PALETTE[9]),
    "P engine count": (PALETTE[7], PALETTE[4]),
    "Client success QPS": ("#ad6800", PALETTE[5]),
}


def cache_ab_color(name, side, default):
    return CACHE_AB_COLORS.get(name, (default, default))[side == "B"]


# Source paths and display semantics for archived stress A/B curves.
STRESS_AB_METRICS = (
    ("per_second", None, "arrivals", "发射 QPS", "req/s", False),
    ("per_second", None, "success", "成功 QPS", "req/s", False),
    ("per_second", None, "errors", "错误 QPS", "req/s", False),
    ("per_second", None, "sched_p95", "调度耗时 P95", "ms", False),
    ("per_second", None, "ttft_p95", "首 token 耗时 P95", "ms", False),
    ("inflight_ts", None, "prefill_requests", "Prefill 在飞", "requests", False),
    ("inflight_ts", None, "decode_reserved", "Decode 预留", "requests", False),
    ("mock_tps_ts", None, "context_tps", "Prefill 计算 TPS", "tokens/s", False),
    (
        "mock_tps_ts",
        None,
        "context_tps_with_cache",
        "Prefill 含缓存 TPS",
        "tokens/s",
        False,
    ),
    (
        "mock_tps_ts",
        None,
        "context_wall_tps",
        "Prefill 墙钟计算 TPS",
        "tokens/s",
        False,
    ),
    (
        "mock_tps_ts",
        None,
        "context_wall_tps_with_cache",
        "Prefill 墙钟含缓存 TPS",
        "tokens/s",
        False,
    ),
    ("mock_tps_ts", None, "generate_tps", "Decode 生成 TPS", "tokens/s", False),
    ("cache_hit_ts", None, "engine_token", "引擎 token 命中比例", "ratio", False),
    ("cache_hit_ts", None, "master_routing", "Master 路由命中比例", "ratio", False),
    (
        "kv_blocks_ts_by_role",
        "prefill",
        "available_blocks",
        "P 可用 KV 块",
        "blocks",
        False,
    ),
    (
        "kv_blocks_ts_by_role",
        "decode",
        "available_blocks",
        "D 可用 KV 块",
        "blocks",
        False,
    ),
    (
        "kv_blocks_ts_by_role",
        "prefill",
        "cache_evictions",
        "P KV 驱逐速率",
        "blocks/s",
        True,
    ),
    (
        "kv_blocks_ts_by_role",
        "decode",
        "cache_evictions",
        "D KV 驱逐速率",
        "blocks/s",
        True,
    ),
)


FIDELITY_THEME = {
    "BACKGROUND": "#F5F6FA", "CARD": "#fff", "REAL": "#2563eb",
    "INDEPENDENT": "#d97706", "JOINT": "#059669",
    "BAD": "#be123c", "GRID": "#e5e7eb",
}


RENDERER_THEME = {
    "PRIMARY": "#1677ff",
    "DARK_TEXT": "#333",
    "SUCCESS": "#52c41a",
    "EVENT_TEXT": "#64748b",
    "MUTED_TEXT": "#888",
    "EVENT_LINE": "#94a3b8",
    "HOVER_BORDER": "#999",
    "BORDER": "#d9d9d9",
    "LIGHT_BORDER": "#ddd",
    "HOVER_BACKGROUND": "#eef4ff",
    "DANGER": "#f5222d",
    "BACKGROUND": "#f5f6fa",
    "CODE_BACKGROUND": "#f7f8fa",
    "WARNING": "#faad14",
    "MUTED_BACKGROUND": "#fafafa",
    "CARD": "#fff",
}
