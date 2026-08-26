#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FlexLB 压测报告生成器：v4 骨架纯图表 Canvas 模板，数据全部由脚本填充。

模板纪律（与 v4 报告 flexlb-run-20260825-201548-report-v4.canvas.tsx 对齐）：
  * 组件只用 Stack/H1/H2/Text/Grid/Stat/Divider/ChartComparisonGrid/
    ChartContainer/LineChart/BarChart/Table；仅导入 'qoder/canvas'，默认导出。
  * 图表高度：延迟节 250，其余 230；所有图表统一挂 valueFormatter={fmt2}。
  * caption 只写 x/y 轴说明（可含采样粒度）；不写 footer，不含任何
    分析/解释/归因文字；汇总表两列（指标 / 数值）。
  * 数值无千分位逗号；JSX 文本节点对 & < > { } 一律实体转义；
    JS 字符串字面量按 JS 规则转义（反斜杠/单引号/换行）。

用法：
  python3 canvas_report_gen.py --aggregate <agg.json> \
      [--engine-dist <engine_dist.json>] [--summary <summary.json>] \
      [--slo <slo_batch_analysis.json>] \
      --out <out.canvas.tsx> [--run-id <id>] \
      [--p-engines 750] [--d-engines 500] [--shards 8] [--replay 1000]

缺省规则：
  * --summary / --slo 未指定时取 aggregate 同目录同名文件（存在才读）；
  * engine_dist 来源优先级：--engine-dist 显式指定 > aggregate 顶层内嵌键
    （aggregate_canvas_run.py 已把 engine_dist 计算进 aggregate，一个脚本
    出全部数据）> aggregate 同目录 engine_dist.json；
  * --run-id 未指定时取 aggregate 的 meta.run_dir；
  * P/D 引擎数优先取 engine_dist 的 engine_count，其次 --p-engines/--d-engines；
  * shards 优先取 summary.json 的 load_client_workers，其次 --shards，再缺省 8。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

TAG = "[canvas_report_gen]"


# ---------------------------------------------------------------------------
# 转义与 JS 字面量
# ---------------------------------------------------------------------------


def esc_text(s):
    """转义用于 JSX 文本节点/属性值的字符串：& < > { } 一律实体化。"""
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("{", "&#123;")
        .replace("}", "&#125;")
    )


def attr(s):
    """转义用于 JSX 双引号属性值的字符串。"""
    return esc_text(s).replace('"', "&quot;")


def js_str(s):
    """转义用于单引号 JS 字符串字面量的字符（JS 语法级转义，非 HTML 实体）。"""
    return (
        str(s)
        .replace("\\", "\\\\")
        .replace("'", "\\'")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
    )


def num(v):
    """数字 -> 紧凑 JS 数字字面量（无千分位逗号）。"""
    if v is None:
        return "0"
    if isinstance(v, bool):
        return "1" if v else "0"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "0"
    if math.isnan(f) or math.isinf(f):
        return "0"
    if f == int(f) and abs(f) < 1e15:
        return str(int(f))
    return repr(round(f, 4))


def num_arr(vals):
    return "[" + ", ".join(num(v) for v in vals) + "]"


def str_arr(vals):
    return "[" + ", ".join("'" + js_str(v) + "'" for v in vals) + "]"


# ---------------------------------------------------------------------------
# 轴标签 / 降采样 / domain
# ---------------------------------------------------------------------------


def rel_times(values):
    """时间戳归一化为相对秒：lean 线 per_second.t / window_gini.t 为相对秒，
    intake 线为 epoch 秒（>1e6）；epoch 值减去最小 epoch，非 epoch 值（如
    首桶 t=0）保持原样，保证 x 轴可读且 duration 计算正确。"""
    nums = []
    for v in values:
        try:
            nums.append(float(v))
        except (TypeError, ValueError):
            nums.append(0.0)
    epoch = [v for v in nums if v > 1_000_000]
    if not epoch:
        return [int(round(v)) for v in nums]
    base = min(epoch)
    return [int(round(v - base)) if v > 1_000_000 else int(round(v)) for v in nums]


def sparse_cats(values, max_labels=6):
    """时间轴类目：仅约 max_labels 个位置带标签，其余为空串（长序列防拥挤）。"""
    n = len(values)
    if n == 0:
        return []
    if n <= max_labels * 3:
        return [str(v) for v in values]
    every = max(1, int(round(n / float(max_labels))))
    out = []
    for i, v in enumerate(values):
        out.append(str(v) if (i % every == 0 or i == n - 1) else "")
    return out


def downsample_idx(n, target=40):
    """取 [0, n-1] 上约 target 个采样下标（首尾必含，保序去重）。"""
    if n <= target:
        return list(range(n))
    step = (n - 1) / float(target - 1)
    idx = []
    last = -1
    for i in range(target):
        j = int(round(i * step))
        if j > last:
            idx.append(j)
            last = j
    if idx[-1] != n - 1:
        idx.append(n - 1)
    return idx


def nice_max(v):
    """不小于 v 的“整数感”上限值（用于 chart domain 上界）。"""
    if v is None:
        return 1
    v = float(v)
    if v <= 0:
        return 1
    mag = 10 ** math.floor(math.log10(v))
    for m in (1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
        cand = m * mag
        if cand >= v * 1.02:
            return round(cand, 10)
    return mag * 10


# ---------------------------------------------------------------------------
# 展示用数字格式（无千分位逗号）
# ---------------------------------------------------------------------------


def fmt_int_trunc(v):
    """QPS 取整口径：向零截断（与 v4 报告一致：7968.684 -> 7968）。"""
    try:
        return str(int(float(v)))
    except (TypeError, ValueError):
        return "0"


def fmt_pct(rate):
    """0.025784 -> 2.58%；0.871 -> 87.1%。"""
    try:
        p = float(rate) * 100.0
    except (TypeError, ValueError):
        return "0%"
    if p == 0:
        return "0%"
    return ("%.1f%%" if p >= 10 else "%.2f%%") % p


def fmt_ms(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "0"
    if f == int(f):
        return str(int(f))
    return ("%.1f" % f).rstrip("0").rstrip(".")


def fmt_g3(v):
    return "—" if v is None else "%.3f" % float(v)


def token_scale_label(max_v):
    """tokens 缩放档位：取“最大值/缩放 >= 10”的最大档，标签用于系列名。"""
    scales = [
        (1, ""),
        (100, "百"),
        (1000, "千"),
        (10000, "万"),
        (100000, "10 万"),
        (1000000, "百万"),
        (10000000, "千万"),
        (100000000, "亿"),
    ]
    chosen_sc, chosen_label = 1, ""
    for sc, label in scales:
        try:
            if float(max_v) / float(sc) >= 10:
                chosen_sc, chosen_label = sc, label
        except (TypeError, ValueError):
            break
    return chosen_sc, chosen_label


# ---------------------------------------------------------------------------
# JSX 生成
# ---------------------------------------------------------------------------


def series_obj(key, name, data_ref, tone=None):
    parts = [
        "key: '" + js_str(key) + "'",
        "name: '" + js_str(name) + "'",
        "data: " + data_ref,
    ]
    if tone:
        parts.append("tone: '" + tone + "'")
    return "{ " + ", ".join(parts) + " }"


def emit_chart(chart, cats_ref, height, series, suffix=None, domain=None):
    """生成 LineChart/BarChart 元素行（v4 排版）。series: [(key, name, data_ref, tone)]"""
    props = ["categories={" + cats_ref + "}", "height={" + str(height) + "}"]
    if suffix is not None:
        props.append('valueSuffix="' + attr(suffix) + '"')
    if domain is not None:
        props.append("domain={" + domain + "}")
    props.append("valueFormatter={fmt2}")
    head = " ".join(props)
    lines = ["          <" + chart + " " + head]
    if len(series) == 1:
        lines.append("            series={[" + series_obj(*series[0]) + "]} />")
    else:
        lines.append("            series={[")
        for s in series:
            lines.append("              " + series_obj(*s) + ",")
        lines.append("            ]} />")
    return lines


def emit_container(title, caption, inner_lines):
    out = [
        '        <ChartContainer title="'
        + attr(title)
        + '" caption="'
        + attr(caption)
        + '">'
    ]
    out.extend(inner_lines)
    out.append("        </ChartContainer>")
    return out


def emit_grid(container_blocks):
    out = ["      <ChartComparisonGrid>"]
    for c in container_blocks:
        out.extend(c)
    out.append("      </ChartComparisonGrid>")
    return out


def emit_stat(value, label, tone=None):
    tone_attr = ' tone="' + tone + '"' if tone else ""
    return (
        '        <Stat value="'
        + attr(value)
        + '" label="'
        + attr(label)
        + '"'
        + tone_attr
        + " />"
    )


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser(
        description="FlexLB 压测纯图表 Canvas 报告生成器（v4 骨架，数据脚本填充）"
    )
    ap.add_argument(
        "--aggregate",
        required=True,
        help="aggregate_canvas_run.py 输出的 aggregate JSON",
    )
    ap.add_argument(
        "--engine-dist",
        help="引擎维度分布 JSON（缺省取 aggregate 同目录 engine_dist.json，存在才读）",
    )
    ap.add_argument(
        "--summary",
        help="load_client summary.json（缺省取 aggregate 同目录 summary.json，存在才读）",
    )
    ap.add_argument(
        "--slo",
        help="slo_batch_analysis.json（缺省取 aggregate 同目录同名文件，存在才读）",
    )
    ap.add_argument("--out", required=True, help="输出 .canvas.tsx 路径")
    ap.add_argument("--run-id", help="run 标识（缺省取 aggregate meta.run_dir）")
    ap.add_argument(
        "--p-engines",
        type=int,
        default=750,
        help="prefill 引擎数缺省（engine_dist.engine_count 优先）",
    )
    ap.add_argument(
        "--d-engines",
        type=int,
        default=500,
        help="decode 引擎数缺省（engine_dist.engine_count 优先）",
    )
    ap.add_argument(
        "--shards",
        type=int,
        help="JavaLoadClient 分片数（缺省取 summary.load_client_workers，再缺省 8）",
    )
    ap.add_argument(
        "--replay",
        type=int,
        default=1000,
        help="replay 倍速（数据文件不含此参数，CLI 提供）",
    )
    args = ap.parse_args()

    warnings = []
    agg = load_json(args.aggregate)
    agg_dir = os.path.dirname(os.path.abspath(args.aggregate))

    summary_path = args.summary or os.path.join(agg_dir, "summary.json")
    slo_path = args.slo or os.path.join(agg_dir, "slo_batch_analysis.json")
    ed_path = args.engine_dist or os.path.join(agg_dir, "engine_dist.json")

    summary_standalone = (
        load_json(summary_path) if os.path.isfile(summary_path) else None
    )
    slo = load_json(slo_path) if os.path.isfile(slo_path) else None
    # engine_dist 来源优先级：显式 --engine-dist > aggregate 顶层内嵌键
    # （aggregate_canvas_run.py 一个脚本出全部数据）> 同目录独立文件。
    if args.engine_dist:
        if not os.path.isfile(ed_path):
            sys.exit("%s --engine-dist 文件不存在: %s" % (TAG, ed_path))
        ed = load_json(ed_path)
    elif isinstance(agg.get("engine_dist"), dict):
        ed = agg.get("engine_dist")
    elif os.path.isfile(ed_path):
        ed = load_json(ed_path)
    else:
        ed = None

    sm = agg.get("summary") or {}
    per_second = agg.get("per_second") or []
    queue_ts = agg.get("queue_timeseries") or []
    mock_last = (agg.get("batch") or {}).get("mock_last") or {}
    if not mock_last and slo:
        mock_last = (slo.get("mock") or {}).get("last") or {}
    validity = (
        sm.get("validity_checks")
        or (summary_standalone or {}).get("validity_checks")
        or {}
    )

    run_id = args.run_id or (agg.get("meta") or {}).get("run_dir") or "unknown"

    # ---- 引擎数 / 分片数 / 时长 ----
    p_engines, d_engines = args.p_engines, args.d_engines
    if ed:
        pe = (ed.get("prefill") or {}).get("engine_count")
        de = (ed.get("decode") or {}).get("engine_count")
        if pe:
            p_engines = int(pe)
        if de:
            d_engines = int(de)
    shards = args.shards
    if shards is None:
        shards = int((summary_standalone or {}).get("load_client_workers") or 8)

    if per_second:
        rel_ts = rel_times([p.get("t", 0) for p in per_second])
        duration_s = rel_ts[-1] - rel_ts[0] + 1
    else:
        total0 = sm.get("total_requests") or 0
        qps0 = sm.get("actual_send_qps") or 0
        duration_s = int(round(total0 / float(qps0))) if qps0 else 0

    # ---- Stat 六连 ----
    total_req = sm.get("total_requests")
    success_n = sm.get("success_count")
    error_n = sm.get("error_count")
    error_rate = sm.get("error_rate")
    if error_rate is None and total_req:
        error_rate = (error_n or 0) / float(total_req)
    send_qps = sm.get("actual_send_qps") or 0
    ok_qps = (float(success_n) / duration_s) if (success_n and duration_s) else 0

    if validity:
        all_valid = all(bool(v) for v in validity.values())
        leak_label, leak_tone = (
            ("clean", "success") if all_valid else ("leak", "danger")
        )
    else:
        leak_label, leak_tone = "—", None
    pacing_ok = bool(validity.get("client_pacing_p99_within_limit"))
    pacing_label, pacing_tone = ("good", "success") if pacing_ok else ("bad", "danger")

    pg = (ed.get("prefill") or {}).get("gini_cum") if ed else None
    dg = (ed.get("decode") or {}).get("gini_cum") if ed else None
    if pg is None and dg is None:
        gini_stat, gini_tone = "—", None
    else:
        gini_stat, gini_tone = fmt_g3(pg) + " / " + fmt_g3(dg), "success"

    # ---- 数据常量 ----
    consts = []  # (name, js_expr)

    def const(name, expr):
        consts.append((name, expr))
        return name

    # 每秒时序（1s 粒度）
    TSEC = None
    qps_arrivals = qps_success = qps_errors = None
    sched_p50 = sched_p95 = sched_p99 = None
    if per_second:
        tsec_vals = rel_ts
        TSEC = const("TSEC", str_arr(sparse_cats(tsec_vals)))
        qps_arrivals = const(
            "qpsArrivals", num_arr([p.get("arrivals", 0) for p in per_second])
        )
        qps_success = const(
            "qpsSuccess", num_arr([p.get("success", 0) for p in per_second])
        )
        qps_errors = const(
            "qpsErrors", num_arr([p.get("errors", 0) for p in per_second])
        )
        sched_p50 = const(
            "schedP50", num_arr([p.get("sched_p50", 0) for p in per_second])
        )
        sched_p95 = const(
            "schedP95", num_arr([p.get("sched_p95", 0) for p in per_second])
        )
        sched_p99 = const(
            "schedP99", num_arr([p.get("sched_p99", 0) for p in per_second])
        )

    # 阶段延迟（终态分位）：取数键带 _ms 后缀，展示类目去后缀
    STAGE_KEYS = [
        "grpc_queue_ms",
        "route_submit_ms",
        "batch_wait_ms",
        "dispatch_ack_ms",
        "ack_response_ms",
    ]
    STAGE_LABELS = [k[:-3] for k in STAGE_KEYS]
    stage_lat = sm.get("server_stage_latency_ms") or {}
    stage_p50 = stage_p95 = None
    if stage_lat:
        stage_p50 = const(
            "stageP50",
            num_arr([(stage_lat.get(s) or {}).get("p50", 0) for s in STAGE_KEYS]),
        )
        stage_p95 = const(
            "stageP95",
            num_arr([(stage_lat.get(s) or {}).get("p95", 0) for s in STAGE_KEYS]),
        )

    # 队列时序（5s 粒度；集群总量 -> 每引擎均值）
    TQ = p_run_req = p_run_batch = p_wait = None
    d_run_req = d_wait = avg_batch = heap_used = None
    q_step = 5
    if queue_ts:
        tq_vals = [q.get("t_offset_s", i * 5) for i, q in enumerate(queue_ts)]
        TQ = const("TQ", str_arr(sparse_cats(tq_vals)))
        p_run_req = const(
            "pRunReq",
            num_arr(
                [
                    round((q.get("prefill_running_reqs", 0) or 0) / float(p_engines), 4)
                    for q in queue_ts
                ]
            ),
        )
        p_run_batch = const(
            "pRunBatch",
            num_arr(
                [
                    round((q.get("prefill_running", 0) or 0) / float(p_engines), 4)
                    for q in queue_ts
                ]
            ),
        )
        p_wait = const(
            "pWait",
            num_arr(
                [
                    round((q.get("prefill_waiting", 0) or 0) / float(p_engines), 4)
                    for q in queue_ts
                ]
            ),
        )
        d_run_req = const(
            "dRunReq",
            num_arr(
                [
                    round((q.get("decode_running", 0) or 0) / float(d_engines), 4)
                    for q in queue_ts
                ]
            ),
        )
        d_wait = const(
            "dWait",
            num_arr(
                [
                    round((q.get("decode_waiting", 0) or 0) / float(d_engines), 4)
                    for q in queue_ts
                ]
            ),
        )
        avg_batch = const(
            "avgBatch", num_arr([q.get("cum_avg_batch_size", 0) for q in queue_ts])
        )
        heap_used = const(
            "heapUsed", num_arr([q.get("heap_used_mb", 0) for q in queue_ts])
        )
        deltas = [b - a for a, b in zip(tq_vals, tq_vals[1:]) if b > a]
        q_step = max(set(deltas), key=deltas.count) if deltas else 5

    # engine_dist：窗口 Gini（按池独立，过滤 null 点）
    wg = (ed or {}).get("window_gini") or {}
    wg_ts = rel_times(wg.get("t") or [])
    p_wg_pts = [(t, v) for t, v in zip(wg_ts, wg.get("prefill") or []) if v is not None]
    d_wg_pts = [(t, v) for t, v in zip(wg_ts, wg.get("decode") or []) if v is not None]

    def wg_axes(pts, cats_name, data_name):
        if len(pts) > 48:
            pts = [pts[i] for i in downsample_idx(len(pts), 40)]
        cats = const(cats_name, str_arr([str(t) for t, _ in pts]))
        data = const(data_name, num_arr([v for _, v in pts]))
        return cats, data, pts

    p_wg_cats = p_wg_data = None
    d_wg_cats = d_wg_data = None
    if p_wg_pts:
        p_wg_cats, p_wg_data, p_wg_pts = wg_axes(p_wg_pts, "pWinT", "pWinG")
    if d_wg_pts:
        d_wg_cats, d_wg_data, d_wg_pts = wg_axes(d_wg_pts, "dWinT", "dWinG")

    # engine_dist：引擎维度分布（降采样到 40 点，x = 池内引擎排名）
    ed0 = ed or {}
    ed_p = ed0.get("prefill") or {}
    ed_d = ed0.get("decode") or {}
    p_reqs = ed_p.get("requests_per_engine") or []
    d_reqs = ed_d.get("requests_per_engine") or []
    p_toks = ed0.get("prefill_tokens_per_engine") or []
    p_util = ed0.get("prefill_util_pct") or []
    d_util = ed0.get("decode_util_pct") or []
    lorenz = ed0.get("lorenz") or {}
    lorenz_x = lorenz.get("x_pct") or list(range(0, 101, 5))
    p_ly = lorenz.get("prefill_y_pct") or []
    d_ly = lorenz.get("decode_y_pct") or []

    def rank_axes(vals, cats_name, data_name):
        idx = downsample_idx(len(vals), 40)
        cats = const(cats_name, str_arr([str(i + 1) for i in idx]))
        data = const(data_name, num_arr([vals[i] for i in idx]))
        return cats, data

    PRANK = p_req_curve = None
    DRANK = d_req_curve = None
    PRANK_UTIL = p_util_curve = None
    DRANK_UTIL = d_util_curve = None
    LORENZ_X = p_lorenz_y = d_lorenz_y = None
    tok_series_name = None
    if p_reqs:
        PRANK, p_req_curve = rank_axes(p_reqs, "PRANK", "pReqCurve")
    if p_toks:
        max_tok = max(p_toks)
        sc, label = token_scale_label(max_tok)
        p_tok_idx = downsample_idx(len(p_toks), 40)
        const("PRANK_TOK", str_arr([str(i + 1) for i in p_tok_idx]))
        if sc != 1:
            const(
                "pTokCurve",
                num_arr([round(p_toks[i] / float(sc), 4) for i in p_tok_idx]),
            )
        else:
            const("pTokCurve", num_arr([p_toks[i] for i in p_tok_idx]))
        tok_series_name = "每引擎 input tokens（" + (
            "个）" if not label else "×" + label + "）"
        )
    if p_util:
        PRANK_UTIL, p_util_curve = rank_axes(p_util, "PRANK_UTIL", "pUtilCurve")
    if d_util:
        DRANK_UTIL, d_util_curve = rank_axes(d_util, "DRANK_UTIL", "dUtilCurve")
    if d_reqs:
        DRANK, d_req_curve = rank_axes(d_reqs, "DRANK", "dReqCurve")
    if p_ly or d_ly:
        LORENZ_X = const("LORENZ_X", str_arr(lorenz_x))
        if p_ly:
            p_lorenz_y = const("pLorenzY", num_arr(p_ly))
        if d_ly:
            d_lorenz_y = const("dLorenzY", num_arr(d_ly))

    # engine_dist：decode KV 时序（时间轴优先对齐 queue_timeseries）
    dkv = (ed or {}).get("decode_kv") or {}
    kv_used = dkv.get("used_avg_series") or []
    kv_util = dkv.get("util_pct_series") or []

    def kv_time_axis(series):
        if queue_ts and len(series) == len(queue_ts):
            vals = [q.get("t_offset_s", i * 5) for i, q in enumerate(queue_ts)]
            return vals, q_step
        return [i * 5 for i in range(len(series))], 5

    TKV = kv_avg = None
    TKV_UTIL = kv_util_data = None
    kv_step = kv_util_step = 5
    if kv_used:
        kv_t_vals, kv_step = kv_time_axis(kv_used)
        TKV = const("TKV", str_arr(sparse_cats(kv_t_vals)))
        kv_avg = const("dKvAvg", num_arr(kv_used))
        if not (queue_ts and len(kv_used) == len(queue_ts)):
            warnings.append(
                "decode_kv.used_avg_series 长度与 queue_timeseries 不一致，x 轴按 5s 采样推定"
            )
    if kv_util:
        if kv_used and len(kv_util) == len(kv_used):
            TKV_UTIL, kv_util_step = TKV, kv_step
        else:
            kv_t_vals2, kv_util_step = kv_time_axis(kv_util)
            TKV_UTIL = const("TKV_UTIL", str_arr(sparse_cats(kv_t_vals2)))
            if not (queue_ts and len(kv_util) == len(queue_ts)):
                warnings.append(
                    "decode_kv.util_pct_series 长度与 queue_timeseries 不一致，x 轴按 5s 采样推定"
                )
        kv_util_data = const("dKvUtil", num_arr(kv_util))

    # ---- 身份行 / 采样说明 ----
    sampling_note = "时间序列 1s 采样（QPS / 延迟）"
    if queue_ts:
        sampling_note += "，队列 " + str(q_step) + "s 采样"
    identity = (
        str(p_engines)
        + "P + "
        + str(d_engines)
        + "D mock · JavaLoadClient "
        + str(shards)
        + " shards · replay@"
        + str(args.replay)
        + "x · "
        + str(duration_s)
        + "s · SCHEDULE_ONLY=1 · "
        + num(total_req)
        + " 请求 · "
        + sampling_note
    )

    # ---- JSX 组装 ----
    lines = []
    lines.append("    <Stack gap={20}>")
    lines.append("      <H1>FlexLB 压测报告 · run " + esc_text(run_id) + "</H1>")
    lines.append('      <Text tone="secondary">')
    lines.append("        " + esc_text(identity))
    lines.append("      </Text>")
    lines.append("")
    lines.append("      <Grid columns={6} gap={10}>")
    lines.append(emit_stat(fmt_int_trunc(send_qps), "发送 QPS"))
    lines.append(emit_stat(fmt_int_trunc(ok_qps), "成功调度 QPS", "success"))
    lines.append(emit_stat(fmt_pct(error_rate), "错误率", "danger"))
    lines.append(emit_stat(leak_label, "泄漏判定", leak_tone))
    lines.append(emit_stat(gini_stat, "P / D 路由 Gini", gini_tone))
    lines.append(emit_stat(pacing_label, "pacing 质量", pacing_tone))
    lines.append("      </Grid>")
    lines.append("")

    # 无节标题：每秒 QPS（发送 / 成功 / 失败）
    if per_second:
        qps_max = max(
            max((p.get("arrivals", 0) or 0) for p in per_second),
            max((p.get("success", 0) or 0) for p in per_second),
            max((p.get("errors", 0) or 0) for p in per_second),
        )
        qps_chart = emit_container(
            "每秒 QPS：发送 / 成功 / 失败",
            "x = 压测时间（s）；y = 每秒请求数",
            emit_chart(
                "LineChart",
                TSEC,
                230,
                [
                    ("arr", "发送（arrivals）", qps_arrivals, "neutral"),
                    ("ok", "成功（success）", qps_success, "success"),
                    ("err", "失败（errors）", qps_errors, "danger"),
                ],
                domain="[0, " + num(nice_max(qps_max * 1.05)) + "]",
            ),
        )
        lines.extend(emit_grid([qps_chart]))
        lines.append("")

    # 1. 延迟
    latency_containers = []
    if per_second:
        all_lat = []
        for p in per_second:
            all_lat.append(p.get("sched_p50", 0) or 0)
            all_lat.append(p.get("sched_p95", 0) or 0)
            all_lat.append(p.get("sched_p99", 0) or 0)
        all_lat.sort()
        v95 = all_lat[int(len(all_lat) * 0.95)] if all_lat else 0
        latency_containers.append(
            emit_container(
                "schedule 延迟 p50 / p95 / p99",
                "x = 压测时间（s，1s 采样）；y = 延迟（ms）",
                emit_chart(
                    "LineChart",
                    TSEC,
                    250,
                    [
                        ("p50", "p50", sched_p50, None),
                        ("p95", "p95", sched_p95, "info"),
                        ("p99", "p99", sched_p99, "warning"),
                    ],
                    suffix=" ms",
                    domain="[0, " + num(nice_max(v95 * 2)) + "]",
                ),
            )
        )
    if stage_lat:
        latency_containers.append(
            emit_container(
                "master 内部分阶段延迟（p50 / p95）",
                "x = master 调度链路阶段；y = 阶段延迟（ms，全程终态分位，非时序）",
                emit_chart(
                    "BarChart",
                    str_arr(STAGE_LABELS),
                    250,
                    [
                        ("p50", "p50", stage_p50, None),
                        ("p95", "p95", stage_p95, "info"),
                    ],
                    suffix=" ms",
                ),
            )
        )
    if latency_containers:
        lines.append("      <Divider />")
        lines.append("")
        lines.append("      <H2>1. 延迟</H2>")
        lines.extend(emit_grid(latency_containers))
        lines.append("")

    # 2. 队列
    if queue_ts:
        q_cap = "x = 压测时间（s，" + str(q_step) + "s 采样）"
        queue_containers = [
            emit_container(
                "Prefill 队列：running（请求 / 批）+ waiting（avg per engine）",
                q_cap + "；y = 每引擎平均并发量（请求与批数）",
                emit_chart(
                    "LineChart",
                    TQ,
                    230,
                    [
                        ("rr", "running 请求数（avg/engine）", p_run_req, "success"),
                        ("rb", "running 批数（avg/engine）", p_run_batch, "info"),
                        ("w", "waiting 请求数（avg/engine）", p_wait, "neutral"),
                    ],
                ),
            ),
            emit_container(
                "Decode 队列：running + waiting（avg per engine）",
                q_cap
                + "；y = 每引擎平均并发请求数（集群换算 ×"
                + str(d_engines)
                + "）",
                emit_chart(
                    "LineChart",
                    TQ,
                    230,
                    [
                        ("r", "running 请求数（avg/engine）", d_run_req, "success"),
                        ("w", "waiting 请求数（avg/engine）", d_wait, "neutral"),
                    ],
                ),
            ),
        ]
        batch_max = max((q.get("cum_avg_batch_size", 0) or 0) for q in queue_ts)
        queue_containers.append(
            emit_container(
                "平均 batch size",
                q_cap + "；y = 请求/批（集群累计均值）",
                emit_chart(
                    "LineChart",
                    TQ,
                    230,
                    [("bs", "avg batch size", avg_batch, "info")],
                    suffix=" 请求/批",
                    domain="[0, " + num(nice_max(batch_max * 1.2)) + "]",
                ),
            )
        )
        lines.append("      <Divider />")
        lines.append("")
        lines.append("      <H2>2. 队列（avg per engine，running / waiting）</H2>")
        lines.extend(emit_grid(queue_containers))
        lines.append("")

    # 3. 调度均衡性（窗口 Gini，仅当 engine_dist 有数据）
    if p_wg_pts or d_wg_pts:
        gini_containers = []
        if p_wg_pts:
            p_gini_max = max(v for _, v in p_wg_pts)
            p_cap = 0.5 if p_gini_max <= 0.5 else nice_max(p_gini_max)
            gini_containers.append(
                emit_container(
                    "Prefill 路由均衡（窗口 Gini）",
                    "x = 压测时间（s）；y = 窗口内各引擎请求数的 Gini（0-1）",
                    emit_chart(
                        "LineChart",
                        p_wg_cats,
                        230,
                        [("pg", "prefill 窗口 Gini", p_wg_data, "info")],
                        domain="[0, " + num(p_cap) + "]",
                    ),
                )
            )
        if d_wg_pts:
            d_gini_max = max(v for _, v in d_wg_pts)
            d_cap = 0.5 if d_gini_max <= 0.5 else nice_max(d_gini_max)
            gini_containers.append(
                emit_container(
                    "Decode 路由均衡（窗口 Gini）",
                    "x = 压测时间（s）；y = 窗口内各 decode 引擎请求数的 Gini（0-1）",
                    emit_chart(
                        "LineChart",
                        d_wg_cats,
                        230,
                        [("dg", "decode 窗口 Gini", d_wg_data, "success")],
                        domain="[0, " + num(d_cap) + "]",
                    ),
                )
            )
        lines.append("      <Divider />")
        lines.append("")
        lines.append("      <H2>3. 调度均衡性（P / D 分开）</H2>")
        lines.extend(emit_grid(gini_containers))
        lines.append("")

    # 3.1 引擎维度分布（x = 池内引擎排名，按指标值降序）
    has_31 = bool(p_reqs or d_reqs or p_util or d_util or p_ly or d_ly)
    if has_31:
        if not (p_wg_pts or d_wg_pts):
            lines.append("      <Divider />")
            lines.append("")
        dist_containers = []
        if p_reqs:
            p_series = [("req", "每引擎请求数（个）", p_req_curve, "info")]
            if p_toks:
                p_series.append(("tok", tok_series_name, "pTokCurve", "neutral"))
            cap = (
                "x = prefill 引擎排名（1.."
                + str(len(p_reqs))
                + "，左端负载最重）；y = 每引擎请求数（个）"
            )
            if p_toks:
                cap += "与 input tokens"
            dist_containers.append(
                emit_container(
                    "Prefill 引擎负载分布",
                    cap,
                    emit_chart("LineChart", PRANK, 230, p_series),
                )
            )
        if p_util:
            dist_containers.append(
                emit_container(
                    "Prefill 引擎利用率分布",
                    "x = prefill 引擎排名（1.." + str(len(p_util)) + "）；y = 利用率 %",
                    emit_chart(
                        "LineChart",
                        PRANK_UTIL,
                        230,
                        [("util", "prefill 利用率（%）", p_util_curve, "warning")],
                        suffix="%",
                    ),
                )
            )
        if d_reqs:
            d_series = [("req", "每引擎请求数（个）", d_req_curve, "success")]
            d_title = "Decode 引擎负载分布"
            d_cap = (
                "x = decode 引擎排名（1.."
                + str(len(d_reqs))
                + "）；y = 每引擎请求数（个）"
            )
            if d_util:
                d_series.append(("util", "decode 利用率（%）", "dUtilCurve", "warning"))
                d_title = "Decode 引擎负载与利用率分布"
                d_cap += "与利用率 %"
            dist_containers.append(
                emit_container(
                    d_title,
                    d_cap,
                    emit_chart("LineChart", DRANK, 230, d_series),
                )
            )
        if p_ly or d_ly:
            lz_series = []
            if p_ly:
                lz_series.append(("p", "prefill 洛伦兹", p_lorenz_y, "info"))
            if d_ly:
                lz_series.append(("d", "decode 洛伦兹", d_lorenz_y, "success"))
            dist_containers.append(
                emit_container(
                    "洛伦兹曲线（P / D 请求数）",
                    "x = 引擎累计占比 %（从最轻到最重）；y = 请求累计占比 %",
                    emit_chart("LineChart", LORENZ_X, 230, lz_series, suffix="%"),
                )
            )
        lines.append(
            "      <H2>3.1 引擎维度分布（x = 池内引擎排名，按指标值降序）</H2>"
        )
        lines.extend(emit_grid(dist_containers))
        lines.append("")

    # 5. KV（仅当 engine_dist 带 decode_kv 时序）
    if kv_used or kv_util:
        kv_containers = []
        if kv_used:
            kv_containers.append(
                emit_container(
                    "Decode KV 已用量",
                    "x = 压测时间（s，"
                    + str(kv_step)
                    + "s 采样）；y = 每引擎平均已用 KV tokens",
                    emit_chart(
                        "LineChart",
                        TKV,
                        230,
                        [("avg", "decode KV 已用（每引擎均值）", kv_avg, "info")],
                    ),
                )
            )
        if kv_util:
            kv_containers.append(
                emit_container(
                    "Decode KV 利用率",
                    "x = 压测时间（s，" + str(kv_util_step) + "s 采样）；y = 利用率 %",
                    emit_chart(
                        "LineChart",
                        TKV_UTIL,
                        230,
                        [("util", "decode KV 利用率（%）", kv_util_data, "warning")],
                        suffix="%",
                    ),
                )
            )
        lines.append("      <Divider />")
        lines.append("")
        lines.append("      <H2>5. KV</H2>")
        lines.extend(emit_grid(kv_containers))
        lines.append("")

    # 6. 资源（mock engine heap）
    has_heap = bool(queue_ts) and any("heap_used_mb" in q for q in queue_ts)
    if has_heap:
        lines.append("      <Divider />")
        lines.append("")
        lines.append("      <H2>6. 资源</H2>")
        lines.extend(
            emit_grid(
                [
                    emit_container(
                        "mock engine heap（MB）",
                        "x = 压测时间（s，"
                        + str(q_step)
                        + "s 采样）；y = mock engine JVM 堆已用（MB）",
                        emit_chart(
                            "LineChart",
                            TQ,
                            230,
                            [("heap", "heap used（MB）", heap_used, "info")],
                        ),
                    ),
                ]
            )
        )
        lines.append("")

    # 汇总表（两列）
    lat_summary = sm.get("schedule_latency_ms") or {}
    rows = []
    rows.append(
        [
            "吞吐",
            "发送 "
            + fmt_int_trunc(send_qps)
            + " / 成功 "
            + fmt_int_trunc(ok_qps)
            + " QPS",
        ]
    )
    err_disp = (
        num(error_n) + " · " + fmt_pct(error_rate) if error_n is not None else "—"
    )
    rows.append(["错误", err_disp])
    if lat_summary:
        rows.append(
            [
                "调度延迟",
                "p50 "
                + fmt_ms(lat_summary.get("p50"))
                + " / p99 "
                + fmt_ms(lat_summary.get("p99"))
                + " ms",
            ]
        )
    else:
        rows.append(["调度延迟", "—"])
    pcv = (ed.get("prefill") or {}).get("cv") if ed else None
    dcv = (ed.get("decode") or {}).get("cv") if ed else None
    bal_parts = []
    if pg is not None or dg is not None:
        bal_parts.append("Gini " + fmt_g3(pg) + " / " + fmt_g3(dg))
    if pcv is not None or dcv is not None:
        bal_parts.append("CV " + fmt_g3(pcv) + " / " + fmt_g3(dcv))
    rows.append(["P/D 均衡", " · ".join(bal_parts) if bal_parts else "—"])
    if queue_ts:
        rows.append(
            [
                "队列（P/D waiting 峰值，集群）",
                "P "
                + num(max((q.get("prefill_waiting", 0) or 0) for q in queue_ts))
                + " / D "
                + num(max((q.get("decode_waiting", 0) or 0) for q in queue_ts)),
            ]
        )
    else:
        rows.append(["队列（P/D waiting 峰值，集群）", "—"])
    if mock_last:
        rows.append(
            [
                "batch",
                "avg size "
                + num(mock_last.get("avg_batch_size"))
                + " / avg ms "
                + num(mock_last.get("avg_batch_ms")),
            ]
        )
    else:
        rows.append(["batch", "—"])
    rows.append(["泄漏判定", leak_label])

    table_lines = ["      <Table"]
    table_lines.append("        headers={['指标', '数值']}")
    table_lines.append("        rows={[")
    for r in rows:
        table_lines.append(
            "          ['" + js_str(r[0]) + "', '" + js_str(r[1]) + "'],"
        )
    table_lines.append("        ]}")
    table_lines.append("      />")

    lines.append("      <Divider />")
    lines.append("")
    lines.append("      <H2>汇总</H2>")
    lines.extend(table_lines)

    src_names = [os.path.basename(args.aggregate)]
    if summary_standalone is not None:
        src_names.append(os.path.basename(summary_path))
    if slo is not None:
        src_names.append(os.path.basename(slo_path))
    if ed is not None:
        embedded = args.engine_dist is None and isinstance(agg.get("engine_dist"), dict)
        src_names.append(
            "(aggregate 内嵌 engine_dist)" if embedded else os.path.basename(ed_path)
        )
    sources = "数据源：" + " · ".join(src_names) + " · run 目录 " + str(run_id)
    lines.append('      <Text tone="secondary" size="small">')
    lines.append("        " + esc_text(sources))
    lines.append("      </Text>")
    lines.append("    </Stack>")

    # ---- 写文件 ----
    header = []
    header.append(
        "import { BarChart, ChartComparisonGrid, ChartContainer, Divider, Grid, H1, H2, "
        "LineChart, Stack, Stat, Table, Text } from 'qoder/canvas';"
    )
    header.append("")
    for name, expr in consts:
        header.append("const " + name + " = " + expr + ";")
    header.append("")
    header.append("const fmt2 = (v: number) => v.toFixed(2);")
    header.append("")
    header.append("export default function FlexlbRunReport() {")
    header.append("  return (")

    footer = []
    footer.append("  );")
    footer.append("}")

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(header + lines + footer) + "\n")

    # ---- stdout 摘要 ----
    sections = ["qps"] if per_second else []
    if latency_containers:
        sections.append("latency")
    if queue_ts:
        sections.append("queue")
    if p_wg_pts or d_wg_pts:
        sections.append("balance")
    if has_31:
        sections.append("dist")
    if kv_used or kv_util:
        sections.append("kv")
    if has_heap:
        sections.append("resource")
    sections.append("summary")
    print(TAG + " run_id=" + str(run_id))
    print(
        TAG
        + " inputs: aggregate="
        + os.path.basename(args.aggregate)
        + " summary="
        + ("yes" if summary_standalone is not None else "no")
        + " slo="
        + ("yes" if slo is not None else "no")
        + " engine_dist="
        + ("yes" if ed is not None else "no")
    )
    print(TAG + " sections: " + ", ".join(sections))
    for w in warnings:
        print(TAG + " warning: " + w)
    if ed:
        for note in ed.get("notes") or []:
            print(TAG + " engine_dist note: " + str(note))
    print(TAG + " written: " + args.out)


if __name__ == "__main__":
    main()
