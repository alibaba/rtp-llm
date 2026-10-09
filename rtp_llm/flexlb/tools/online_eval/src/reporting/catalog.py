"""Shared report vocabulary and series colors.

Metric identities are data keys; names, axes and color are presentation only.
"""

PALETTE = (
    "#1677ff", "#52c41a", "#faad14", "#f5222d",
    "#722ed1", "#13c2c2", "#eb2f96", "#fa8c16",
    "#a0d911", "#2f54eb", "#fadb14", "#08979c",
)
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
    names.append(("batch", "请求 / 执行批"))
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
                         "rtp_llm_context_tps_with_cache_engine_mean",
                         "rtp_llm_generate_tps_engine_mean"}
    if metric == "rtp_llm_generate_tps_engine_mean":
        group, axis = "Decode TPS", "forward"
    elif primary:
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
