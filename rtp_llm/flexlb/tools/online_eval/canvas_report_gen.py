#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FlexLB 压测报告生成器：直接吐 self-contained Chart.js 4.4.7 HTML。

数据管线保持不变：aggregate JSON → 统计 → panel（LineChart / BarChart）+ KPI +
汇总表。渲染层拆到 canvas_report_render_html.py，页面观感对齐既有
outputs/flexlb-run-*-chartjs.html（浅色主题 / 白卡 / 6 列 KPI / 2 列 panel /
280px chart，legend 单击切换、tooltip x 轴 index 联动，无 zoom 插件）。

内部实现：main() 用 emit_ 系列先在内存拼一份完整 tsx 字符串（保留原有全部
41 图 / 154 组数据的正确性），末端通过 _extract_spec_from_tsx() 反抽出
{run_id/title/subtitle/kpis/panels} spec，喂 canvas_report_render_html.render()
写入 --out 指向的 HTML 文件；不再写 .tsx 中间产物。

用法：
  python3 canvas_report_gen.py --aggregate <agg.json> \
      [--engine-dist <engine_dist.json>] [--summary <summary.json>] \
      [--slo <slo_batch_analysis.json>] \
      --out <out.html> [--run-id <id>] \
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
import bisect
import json
import math
import os
import re
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


def ts_step_values(pts, times):
    """有序 [(t, v)] 序列按 times 逐点对齐（前向 step 插值）：
    取 t' <= t 的最后一个样本值；t 早于首样本取首值，晚于尾样本取尾值。"""
    if not pts:
        return [0 for _ in times]
    ts = [p[0] for p in pts]
    vals = [p[1] for p in pts]
    out = []
    for t in times:
        i = bisect.bisect_right(ts, t)
        out.append(vals[i - 1] if i > 0 else vals[0])
    return out


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


# ---------------------------------------------------------------------------
# tsx → spec 反抽（末端渲染桥）
# ---------------------------------------------------------------------------
#
# emit_ 系列在内存里拼一份完整 tsx（保留原 41 图/154 组数据的算法路径不动），
# 这里从 tsx 字符串反抽出 render_html 需要的 spec dict：
#   {run_id, title, subtitle, kpis:[{label,value,tone}], panels:[panel]}
#   panel: {id,title,caption,type('line'|'bar'),x,yMax,unit,series:[{name,data,color,tone}]}
# 好处：不重构那 2000 行 emit_ 逻辑（碰它风险大），只在末端做一次纯正则/JSON
# 转换。同一套抽取器也用于 preview_gen.py 手动预览。
#
# 注意事项：
#  * const 命名同时含 UPPER_CASE（X 轴）与 camelCase（Y 轴 series），正则须
#    放宽到 [A-Za-z_]+。
#  * ChartContainer body 里可能含多个 LineChart/BarChart（ChartComparisonGrid
#    形态），因此按"container 开合位定位 + body 内循环抽 chart"两段处理，
#    不用一个 regex 匹整体。
#  * chart 属性顺序不稳定（valueSuffix 有时在 domain 前），故 attrs 用独立
#    子正则各抽（ATTR_CATS/HEIGHT/DOMAIN/SUFFIX）。


_TSX_CONST_RE = re.compile(
    r"^const ([A-Za-z_][A-Za-z0-9_]*) = (\[.*?\])\s*;\s*$", re.M | re.S
)
_TSX_STAT_RE = re.compile(
    r'<Stat value="([^"]*)" label="([^"]*)"(?: tone="([^"]*)")? />'
)
_TSX_CONTAINER_OPEN = re.compile(
    r'<ChartContainer title="([^"]*)" caption="([^"]*)">', re.S
)
_TSX_CHART_TAG = re.compile(
    r"<(LineChart|BarChart)\s+([^>]*?)\s+series=\{\[(.*?)\]\}\s*/>", re.S
)
_TSX_ATTR_CATS = re.compile(r"categories=\{([A-Za-z_][A-Za-z0-9_]*)\}")
_TSX_ATTR_DOMAIN = re.compile(r"domain=\{\[([^\]]+)\]\}")
_TSX_ATTR_SUFFIX = re.compile(r'valueSuffix="([^"]*)"')
_TSX_SERIES_RE = re.compile(
    r"\{\s*key:\s*'([^']*)'\s*,\s*name:\s*'([^']*)'\s*,\s*data:\s*"
    r"([A-Za-z_][A-Za-z0-9_]*)(?:\s*,\s*tone:\s*'([^']*)')?\s*\}",
    re.S,
)

_TSX_TONE_TO_KPI = {
    "success": "success",
    "danger": "danger",
    "warning": "warn",
    "warn": "warn",
}


def _extract_spec_from_tsx(tsx_src, run_id, subtitle):
    """从内存 tsx 字符串反抽 render_html.render() 需要的 spec dict。"""
    # 惰性 import，避免测试或 --help 场景强依赖 render_html
    import canvas_report_render_html as _rh

    # 1) 抽 const NAME = [...]; → dict[str, list]
    consts = {}
    for m in _TSX_CONST_RE.finditer(tsx_src):
        name, expr = m.group(1), m.group(2)
        try:
            consts[name] = json.loads(expr.replace("'", '"'))
        except Exception:
            pass

    # 2) 抽 KPI（<Stat …/>）
    kpis = []
    for m in _TSX_STAT_RE.finditer(tsx_src):
        val, label, tone = m.group(1), m.group(2), m.group(3)
        kpis.append(
            {"label": label, "value": val, "tone": _TSX_TONE_TO_KPI.get(tone or "", "")}
        )

    # 3) 抽 ChartContainer + body 内多个 chart
    panels = []
    pid = 0
    for om in _TSX_CONTAINER_OPEN.finditer(tsx_src):
        close_pos = tsx_src.find("</ChartContainer>", om.end())
        if close_pos < 0:
            continue
        body = tsx_src[om.end() : close_pos]
        title, caption = om.group(1), om.group(2)
        for tm in _TSX_CHART_TAG.finditer(body):
            ctype, attrs, series_body = tm.groups()
            cats_m = _TSX_ATTR_CATS.search(attrs)
            if not cats_m:
                continue
            dom_m = _TSX_ATTR_DOMAIN.search(attrs)
            suf_m = _TSX_ATTR_SUFFIX.search(attrs)
            x_ref = cats_m.group(1)
            x_vals = consts.get(x_ref, [])
            series = []
            for i, sm in enumerate(_TSX_SERIES_RE.finditer(series_body)):
                _key, s_name, data_ref, tone = sm.groups()
                data = consts.get(data_ref, [])
                series.append(
                    {
                        "name": s_name,
                        "data": data,
                        "color": _rh.series_color(tone, i),
                        "tone": tone or "",
                    }
                )
            if not series:
                continue
            pid += 1
            y_max = None
            if dom_m:
                try:
                    y_max = float(dom_m.group(1).split(",")[1])
                except Exception:
                    y_max = None
            panels.append(
                {
                    "id": "p%d" % pid,
                    "title": title,
                    "caption": caption,
                    "type": "bar" if ctype == "BarChart" else "line",
                    "x": x_vals,
                    "yMax": y_max,
                    "unit": (suf_m.group(1).strip() if suf_m else ""),
                    "series": series,
                }
            )

    return {
        "run_id": run_id,
        "title": "FlexLB 压测报告 · run " + run_id,
        "subtitle": subtitle,
        "kpis": kpis[:6],
        "panels": panels,
    }


def normalize_out_runid(path):
    """--out 文件名中 RUNID 段（8 位日期_6 位时间）下划线统一转连字符。

    防回归（Canvas 预览 ENOENT 历史坑已复发 4 次）：run 目录本身用
    下划线 RUNID（如 20260828_155349），而 Canvas 预览引用的报告命名
    规范是 flexlb-run-<RUNID>-report.canvas.tsx 且 RUNID 用连字符
    （flexlb-run-20260828-155349-report.canvas.tsx）。生成器在输出处
    强制规范化文件名，调用方传下划线 RUNID 也不会产出坏文件名。
    只匹配 8 位日期 + "_" + 6 位时间的 RUNID 形态，不碰文件名其它
    下划线；报告内部 run_id 展示保留 meta.run_dir 原样（与远端 run
    目录名对账）。
    """
    base = os.path.basename(path)
    fixed = re.sub(r"(?<!\d)(\d{8})_(\d{6})(?!\d)", r"\1-\2", base)
    if fixed != base:
        return os.path.join(os.path.dirname(path), fixed)
    return path


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
    ap.add_argument(
        "--out", required=True, help="输出 .html 路径（self-contained Chart.js HTML）"
    )
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

    # RUNID 文件名规范化（防 ENOENT 复发）：详见 normalize_out_runid
    # 文档字符串；规范化后仍有下划线 RUNID 段则断言失败（自检）。
    out_normalized = normalize_out_runid(args.out)
    if out_normalized != args.out:
        print(
            TAG
            + " normalized RUNID in --out filename: "
            + os.path.basename(args.out)
            + " -> "
            + os.path.basename(out_normalized)
        )
        args.out = out_normalized
    assert not re.search(r"(?<!\d)\d{8}_\d{6}(?!\d)", os.path.basename(args.out)), (
        "--out filename RUNID segment must use hyphen "
        "(flexlb-run-YYYYMMDD-HHMMSS-report.html), got: " + args.out
    )

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
    # ---- 错误桶定义（旧 7 桶 + err_other 细分 9 子桶；label 供图例/汇总表） ----
    # 新子桶（20260828+ aggregate 才有）：旧 aggregate 无这些键 -> 全零
    # -> 图/表自适应跳过，输出与旧版一致（向后兼容）。
    ERR_DEFS = [
        ("err_no_decode", "no worker", "danger"),
        ("err_no_prefill", "no prefill worker", "danger"),
        ("err_queue_full", "queue full", "warning"),
        ("err_deadline", "deadline", "info"),
        ("err_priority", "priority ahead", "warning"),
        ("err_preempted", "preempted", None),
        ("err_yielded", "yielded", None),
        ("err_backpressure", "backpressure (843x)", "warning"),
        ("err_queue_timeout", "queue timeout (8503)", "warning"),
        ("err_rst_stream", "rst_stream", "danger"),
        ("err_goaway", "goaway", "danger"),
        ("err_unavailable", "unavailable", "danger"),
        ("err_cancelled", "cancelled", None),
        ("err_internal", "internal", "danger"),
        ("err_empty_response", "empty response", "warning"),
        ("err_duplicate_rid", "duplicate rid", "warning"),
        ("err_other", "other", "neutral"),
    ]
    # 各桶 per_second 口径总量（仅带时间戳行；0 = 该桶不存在/无数据）
    err_totals = {k: 0 for k, _, _ in ERR_DEFS}
    for p in per_second:
        for k, _, _ in ERR_DEFS:
            err_totals[k] += p.get(k, 0) or 0
    queue_ts = agg.get("queue_timeseries") or []
    # compact time series (aggregate_canvas_run.py 861f3a9+；旧 aggregate 无这些键 ->
    # 空 list，对应图条件渲染)
    stage_ts = agg.get("stage_latency_ts") or []
    engine_exec = agg.get("engine_exec_ts") or []
    process_ts = agg.get("process_ts") or []
    inflight_ts = agg.get("inflight_ts") or []
    inflight_age = agg.get("inflight_age_ts") or []
    inflight_age_by_role = agg.get("inflight_age_by_role") or {}
    kv_ts = agg.get("kv_ts") or []
    batcher_ts = agg.get("batcher_ts") or []
    batcher_ts_by_role = agg.get("batcher_ts_by_role") or []
    batcher_engine_quantile_ts = agg.get("batcher_engine_quantile_ts") or []
    batcher_top_engines_ts = agg.get("batcher_top_engines_ts") or []
    queue_top_bottom_ts = agg.get("queue_top_bottom_ts") or {}
    dispatch_reason_ts = agg.get("dispatch_reason_ts") or []
    dispatch_batch_size_ts = agg.get("dispatch_batch_size_ts") or []
    cancel_ts = agg.get("cancel_qps_ts") or []
    integrity = agg.get("integrity") or {}
    batch_size_final = agg.get("batch_size_final") or {}
    _meta = agg.get("meta") or {}
    # Newer aggregates report fetch_output_stream (True = client read streams);
    # legacy ones recorded the inverted switch as schedule_only.
    fetch_output_stream = (
        bool(_meta["fetch_output_stream"])
        if "fetch_output_stream" in _meta
        else not _meta.get("schedule_only")
    )
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

    # 队列时序（5s 粒度，avg/engine 口径：集群总量 ÷ 引擎数，
    # engine_dist.engine_count 优先）
    TQ = p_run_req = p_run_batch = p_wait = p_master_bq = None
    d_run_req = d_wait = avg_batch = heap_used = None
    cum_batch_ref = None
    q_step = 5
    if queue_ts:
        tq_vals = [q.get("t_offset_s", i * 5) for i, q in enumerate(queue_ts)]
        TQ = const("TQ", str_arr(sparse_cats(tq_vals)))
        p_run_req = const(
            "pRunReq",
            num_arr(
                [
                    round(
                        (q.get("prefill_running_reqs", 0) or 0) / max(1, p_engines), 2
                    )
                    for q in queue_ts
                ]
            ),
        )
        p_run_batch = const(
            "pRunBatch",
            num_arr(
                [
                    round((q.get("prefill_running", 0) or 0) / max(1, p_engines), 2)
                    for q in queue_ts
                ]
            ),
        )
        p_wait = const(
            "pWait",
            num_arr(
                [
                    round((q.get("prefill_waiting", 0) or 0) / max(1, p_engines), 2)
                    for q in queue_ts
                ]
            ),
        )
        d_run_req = const(
            "dRunReq",
            num_arr(
                [
                    round((q.get("decode_running", 0) or 0) / max(1, d_engines), 2)
                    for q in queue_ts
                ]
            ),
        )
        d_wait = const(
            "dWait",
            num_arr(
                [
                    round((q.get("decode_waiting", 0) or 0) / max(1, d_engines), 2)
                    for q in queue_ts
                ]
            ),
        )
        # 口径修复（Jack 诊断）：主线换区间均值 interval_avg_batch_size
        # （相邻采样间隔内 enqueued 增量 ÷ batches 增量，反映真实波动），
        # 集群累计均值降为参照淡线；旧 aggregate 无 interval 键时退回
        # cum 单线（向后兼容）。
        has_interval_batch = any("interval_avg_batch_size" in q for q in queue_ts)
        if has_interval_batch:
            avg_batch = const(
                "ivBatch",
                num_arr([q.get("interval_avg_batch_size", 0) for q in queue_ts]),
            )
            cum_batch_ref = const(
                "cumBatch",
                num_arr([q.get("cum_avg_batch_size", 0) for q in queue_ts]),
            )
        else:
            avg_batch = const(
                "avgBatch", num_arr([q.get("cum_avg_batch_size", 0) for q in queue_ts])
            )
        heap_used = const(
            "heapUsed", num_arr([q.get("heap_used_mb", 0) for q in queue_ts])
        )
        deltas = [b - a for a, b in zip(tq_vals, tq_vals[1:]) if b > a]
        q_step = max(set(deltas), key=deltas.count) if deltas else 5
        if batcher_ts_by_role:
            # 第四线：master batcher 队列（avg/engine，master 侧口径）。
            # 数据源与 2.1 节 p_master_batcher 同源（batcher_ts_by_role 的
            # prefill 集群总量 ÷ P 引擎数）；master 序列 t0 可为负（启动
            # warmup），按 TQ 时间轴前向 step 对齐（尾部超出取尾值，序列
            # 尾部已归零）。TQ 重锚 epoch0 后两轴同源，对齐语义准确。
            btr_pts = [
                (float(r.get("t", 0) or 0), r.get("prefill", 0) or 0)
                for r in batcher_ts_by_role
            ]
            p_master_bq = const(
                "pMasterBq",
                num_arr(
                    [
                        round(v / max(1, p_engines), 2)
                        for v in ts_step_values(btr_pts, tq_vals)
                    ]
                ),
            )

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
    # engine_dist：三口径数据（请求数 / token / 利用率）。新 aggregate 内嵌
    # 键（prefill.tokens_per_engine、utilization.prefill.per_engine_pct）与旧
    # 独立 engine_dist.json 顶层键（prefill_tokens_per_engine / prefill_util_pct）
    # 双向兼容。
    ed0 = ed or {}
    ed_p = ed0.get("prefill") or {}
    ed_d = ed0.get("decode") or {}
    util_block = ed0.get("utilization") or {}
    p_reqs = ed_p.get("requests_per_engine") or []
    d_reqs = ed_d.get("requests_per_engine") or []
    p_toks = ed_p.get("tokens_per_engine") or ed0.get("prefill_tokens_per_engine") or []
    d_toks = ed_d.get("tokens_per_engine") or []
    p_util = (util_block.get("prefill") or {}).get("per_engine_pct") or (
        ed0.get("prefill_util_pct") or []
    )
    d_util = (util_block.get("decode") or {}).get("per_engine_pct") or (
        ed0.get("decode_util_pct") or []
    )
    lorenz = ed0.get("lorenz") or {}
    lorenz_x = lorenz.get("x_pct") or list(range(0, 101, 5))
    p_ly = lorenz.get("prefill_y_pct") or []
    d_ly = lorenz.get("decode_y_pct") or []
    p_tok_ly = lorenz.get("prefill_tokens_y_pct") or []
    d_tok_ly = lorenz.get("decode_tokens_y_pct") or []

    def rank_axes(vals, cats_name, data_name):
        idx = downsample_idx(len(vals), 40)
        cats = const(cats_name, str_arr([str(i + 1) for i in idx]))
        data = const(data_name, num_arr([vals[i] for i in idx]))
        return cats, data

    PRANK = p_req_curve = None
    DRANK = d_req_curve = None
    PRANK_TOK = p_tok_curve = p_tok_series_name = None
    DRANK_TOK = d_tok_curve = d_tok_series_name = None
    PRANK_UTIL = p_util_curve = None
    DRANK_UTIL = d_util_curve = None
    LORENZ_X = p_lorenz_y = d_lorenz_y = None
    LORENZ_TOK_X = p_tok_lorenz_y = d_tok_lorenz_y = None
    if p_reqs:
        PRANK, p_req_curve = rank_axes(p_reqs, "PRANK", "pReqCurve")
    if d_reqs:
        DRANK, d_req_curve = rank_axes(d_reqs, "DRANK", "dReqCurve")
    if p_toks:
        sc, label = token_scale_label(max(p_toks))
        p_tok_idx = downsample_idx(len(p_toks), 40)
        PRANK_TOK = const("PRANK_TOK", str_arr([str(i + 1) for i in p_tok_idx]))
        if sc != 1:
            p_tok_curve = const(
                "pTokCurve",
                num_arr([round(p_toks[i] / float(sc), 4) for i in p_tok_idx]),
            )
        else:
            p_tok_curve = const("pTokCurve", num_arr([p_toks[i] for i in p_tok_idx]))
        p_tok_series_name = "每引擎 input tokens（" + (
            "个）" if not label else "×" + label + "）"
        )
    if d_toks:
        sc_d, label_d = token_scale_label(max(d_toks))
        d_tok_idx = downsample_idx(len(d_toks), 40)
        DRANK_TOK = const("DRANK_TOK", str_arr([str(i + 1) for i in d_tok_idx]))
        if sc_d != 1:
            d_tok_curve = const(
                "dTokCurve",
                num_arr([round(d_toks[i] / float(sc_d), 4) for i in d_tok_idx]),
            )
        else:
            d_tok_curve = const("dTokCurve", num_arr([d_toks[i] for i in d_tok_idx]))
        d_tok_series_name = "每引擎 output tokens（" + (
            "个）" if not label_d else "×" + label_d + "）"
        )
    if p_util:
        PRANK_UTIL, p_util_curve = rank_axes(p_util, "PRANK_UTIL", "pUtilCurve")
    if d_util:
        DRANK_UTIL, d_util_curve = rank_axes(d_util, "DRANK_UTIL", "dUtilCurve")
    if p_ly or d_ly:
        LORENZ_X = const("LORENZ_X", str_arr(lorenz_x))
        if p_ly:
            p_lorenz_y = const("pLorenzY", num_arr(p_ly))
        if d_ly:
            d_lorenz_y = const("dLorenzY", num_arr(d_ly))
    if p_tok_ly or d_tok_ly:
        LORENZ_TOK_X = const("LORENZ_TOK_X", str_arr(lorenz_x))
        if p_tok_ly:
            p_tok_lorenz_y = const("pTokLorenzY", num_arr(p_tok_ly))
        if d_tok_ly:
            d_tok_lorenz_y = const("dTokLorenzY", num_arr(d_tok_ly))

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
    # fetch_output_stream=False 仅在 aggregate meta 明确报告时展示（旧 aggregate 无该键则省略）
    sched_seg = "FETCH_OUTPUT_STREAM=0 · " if fetch_output_stream is False else ""
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
        + "s · "
        + sched_seg
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
    # consolidate 完整性声明：final_snapshot 非 live / slo 陈旧时在报告头
    # 显式降级声明，避免读者把田数据当作本 run 终态。
    integrity_notes = []
    _fss = integrity.get("final_snapshot_source")
    if _fss and _fss != "live":
        _fss_label = {
            "fallback": "回退旧值（fallback）",
            "missing": "缺失（missing）",
        }.get(_fss, str(_fss))
        integrity_notes.append(
            "final_snapshot 为" + _fss_label + "，引擎利用率/终态不代表本 run"
        )
    _si = integrity.get("slo_integrity") or {}
    if _si and not _si.get("fresh", True):
        integrity_notes.append(
            "slo_batch_analysis.json 早于 per_request.jsonl（陈旧残留），SLO/批决策结论不可信"
        )
    _unstamped = integrity.get("per_second_rows_without_send_ts")
    if _unstamped:
        integrity_notes.append(
            "per_second 序列不含 "
            + fmt_int_trunc(_unstamped)
            + " 条无发送时间戳的请求行，sum(arrivals) ≠ total_requests"
        )
    if integrity_notes:
        # Text tone accepts only primary/secondary/tertiary/quaternary
        # ("warning" is a chart-series tone, not a Text tone).
        lines.append('      <Text tone="secondary">')
        for _note in integrity_notes:
            lines.append("        ⚠ 数据完整性: " + esc_text(_note))
            warnings.append("integrity: " + _note)
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

    # 无节标题：每秒 QPS（发送 / 成功 / 失败）+ 失败按原因
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
        # 失败按原因分曲线：曲线按「实际存在的桶」自适应（具名桶优先，
        # 总曲线数不超 ~8，超出时总量排序保留 top，其余合并为一条曲线）；
        # 零失败 run 画一条全零 err_other 曲线保持「无失败」证据。
        # 桶含义：deadline=客户端/调度超时；backpressure=master 准入拒绝
        # （8430/8431/8432）；queue timeout=master 排队超时（8503）；
        # rst_stream/goaway/unavailable/cancelled/internal=gRPC 传输层错误；
        # empty response=零输出流；duplicate rid=请求 ID 重复；
        # other=未归类残渣。旧 aggregate 无新子桶键 -> 全零 -> 自动跳过。
        MAX_ERR_CURVES = 8
        err_total_errors = sum(p.get("errors", 0) or 0 for p in per_second)
        err_nonzero = [(k, lb, tn) for k, lb, tn in ERR_DEFS if err_totals.get(k)]
        merged_keys = []
        if err_total_errors > 0 and len(err_nonzero) > MAX_ERR_CURVES:
            ranked = sorted(err_nonzero, key=lambda x: -err_totals[x[0]])
            sel = ranked[: MAX_ERR_CURVES - 1] + [
                ("__merged__", "small buckets merged", "neutral")
            ]
            merged_keys = [k for k, _, _ in ranked[MAX_ERR_CURVES - 1 :]]
        elif err_total_errors > 0:
            sel = err_nonzero
        else:
            sel = [("err_other", "other", "neutral")]
        err_series = []
        err_max = 0
        for k, lb, tn in sel:
            if k == "__merged__":
                cname = "errMergedSmall"
                vals = [
                    sum(p.get(mk, 0) or 0 for mk in merged_keys) for p in per_second
                ]
            else:
                cname = "err" + k[4:].title().replace("_", "")
                vals = [p.get(k, 0) or 0 for p in per_second]
            ref = const(cname, num_arr(vals))
            err_series.append((k, lb, ref, tn))
            if vals:
                err_max = max(err_max, max(vals))
        fail_chart = emit_container(
            "每秒失败 QPS：按原因",
            "x = 压测时间（s）；y = 每秒失败请求数。桶：deadline=客户端/调度超时；"
            "backpressure=master 准入拒绝（8430/8431/8432）；queue timeout=master"
            " 排队超时（8503）；rst_stream/goaway/unavailable/cancelled/internal"
            "=gRPC 传输层错误；empty response=零输出流；duplicate rid=请求 ID"
            " 重复；other=未归类残渣。曲线按实际存在的桶自适应，小桶自动合并",
            emit_chart(
                "LineChart",
                TSEC,
                230,
                err_series,
                domain="[0, " + num(nice_max(err_max * 1.2)) + "]",
            ),
        )
        qps_grid = [qps_chart, fail_chart]
        # 引擎侧事件速率（cancel / decode 进入 running / decode 完成）：
        # cancel_rpcs / decode_admitted / decode_done 是 mock 集群累计计数，
        # 聚合端按 epoch 对齐差分；与客户端 QPS 同节呈现。全零也画：
        # 零 cancel 曲线本身就是「无抢占/取消」的正面证据。
        if cancel_ts:
            tcxl = const(
                "TCXL", str_arr(sparse_cats([r.get("t", 0) for r in cancel_ts]))
            )
            cancel_defs = [
                ("cancel_qps", "cancel（引擎侧）", "danger"),
                ("decode_admitted_qps", "decode 进入 running", "info"),
                ("decode_done_qps", "decode 完成", "success"),
            ]
            cancel_series = []
            for k, label, tone in cancel_defs:
                cancel_series.append(
                    (
                        "cx_" + k,
                        label,
                        const("cx" + k, num_arr([r.get(k, 0) or 0 for r in cancel_ts])),
                        tone,
                    )
                )
            cancel_max = max(
                (max((r.get(k, 0) or 0) for r in cancel_ts) for k, _, _ in cancel_defs),
                default=0,
            )
            qps_grid.append(
                emit_container(
                    "每秒引擎侧事件速率：cancel / decode 进入 running / decode 完成",
                    "x = 压测时间（s，stats 采样轴，epoch 对齐）；y = 每秒事件数（累计计数差分归一）",
                    emit_chart(
                        "LineChart",
                        tcxl,
                        230,
                        cancel_series,
                        domain="[0, " + num(nice_max(cancel_max * 1.2)) + "]",
                    ),
                )
            )
        lines.extend(emit_grid(qps_grid))
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
    # 反馈 1：schedule 分位 + master 链路五阶段 p95 合成一张时序图
    # （master 10s 窗口；schedule 分位重采样为每个窗口内 1s 桶的中值）
    if stage_ts:
        stage_t_vals = [r.get("t", 0) for r in stage_ts]
        STAGE_T = const("STAGE_T", str_arr(sparse_cats(stage_t_vals)))
        stage_series = []
        if per_second:
            sched_map = {}
            for p in per_second:
                try:
                    sched_map[int(p.get("t", 0))] = p
                except (TypeError, ValueError):
                    pass

            def stage_resample(key, cname):
                vals = []
                for r in stage_ts:
                    try:
                        tc = int(round(float(r.get("t", 0))))
                    except (TypeError, ValueError):
                        tc = 0
                    win = [
                        sched_map[k].get(key, 0) or 0
                        for k in range(tc - 9, tc + 1)
                        if k in sched_map
                    ]
                    win.sort()
                    vals.append(win[len(win) // 2] if win else 0)
                return const(cname, num_arr(vals))

            stage_series.append(
                (
                    "sp50",
                    "schedule p50（10s 窗口中值）",
                    stage_resample("sched_p50", "stageSchedP50"),
                    None,
                )
            )
            stage_series.append(
                (
                    "sp95",
                    "schedule p95（10s 窗口中值）",
                    stage_resample("sched_p95", "stageSchedP95"),
                    "info",
                )
            )
            stage_series.append(
                (
                    "sp99",
                    "schedule p99（10s 窗口中值）",
                    stage_resample("sched_p99", "stageSchedP99"),
                    "warning",
                )
            )
        stage_defs = [
            ("grpc_queue_p95_ms", "grpc_queue p95", "grpcQueueP95"),
            ("route_submit_p95_ms", "route_submit p95", "routeSubmitP95"),
            ("batch_wait_p95_ms", "batch_wait p95", "batchWaitP95"),
            ("dispatch_ack_p95_ms", "dispatch_ack p95", "dispatchAckP95"),
            ("ack_response_p95_ms", "ack_response p95", "ackResponseP95"),
        ]
        for key, label, cname in stage_defs:
            ref = const(cname, num_arr([r.get(key, 0) or 0 for r in stage_ts]))
            stage_series.append((key[:4], label, ref, None))
        stage_all_max = max(
            (
                max(
                    (r.get(k, 0) or 0)
                    for r in stage_ts
                    for k in (
                        "server_p99_ms",
                        "grpc_queue_p95_ms",
                        "route_submit_p95_ms",
                        "batch_wait_p95_ms",
                        "dispatch_ack_p95_ms",
                        "ack_response_p95_ms",
                    )
                )
                if stage_ts
                else 0
            ),
            max((p.get("sched_p99", 0) or 0) for p in per_second) if per_second else 0,
        )
        latency_containers.append(
            emit_container(
                "调度延迟：schedule 分位 + master 链路阶段 p95",
                "x = 压测时间（s，master 10s 窗口）；y = 延迟（ms）",
                emit_chart(
                    "LineChart",
                    STAGE_T,
                    250,
                    stage_series,
                    suffix=" ms",
                    domain="[0, " + num(nice_max(stage_all_max * 1.15)) + "]",
                ),
            )
        )
    # 反馈 2：五延迟合一（e2e / ttft / schedule / prefill exec / decode exec，
    # 全部 p95；engine exec 按最近整秒对齐到 per_second 桶）
    if per_second:
        has_e2e = any((p.get("e2e_p95", 0) or 0) for p in per_second)
        has_ttft = any((p.get("ttft_p95", 0) or 0) for p in per_second)
        if has_e2e or has_ttft or engine_exec:
            five_series = []
            if has_e2e:
                five_series.append(
                    (
                        "e2e",
                        "e2e（p95）",
                        const(
                            "e2eP95", num_arr([p.get("e2e_p95", 0) for p in per_second])
                        ),
                        None,
                    )
                )
            if has_ttft:
                five_series.append(
                    (
                        "ttft",
                        "ttft（p95）",
                        const(
                            "ttftP95",
                            num_arr([p.get("ttft_p95", 0) for p in per_second]),
                        ),
                        "info",
                    )
                )
            if sched_p95:
                five_series.append(("sch", "schedule（p95）", sched_p95, "warning"))
            exec_map = {}
            for r in engine_exec:
                try:
                    exec_map[int(round(float(r.get("t", 0))))] = r
                except (TypeError, ValueError):
                    pass
            tsec_int = []
            for p in per_second:
                try:
                    tsec_int.append(int(p.get("t", 0)))
                except (TypeError, ValueError):
                    tsec_int.append(0)
            if engine_exec:
                de95 = [
                    (exec_map.get(t) or {}).get("decode_exec_p95_ms", 0) or 0
                    for t in tsec_int
                ]
                five_series.append(
                    (
                        "de",
                        "decode exec（p95）",
                        const("decodeExecP95", num_arr(de95)),
                        "success",
                    )
                )
                if "prefill_exec_p95_ms" in (engine_exec[0] or {}):
                    pe95 = [
                        (exec_map.get(t) or {}).get("prefill_exec_p95_ms", 0) or 0
                        for t in tsec_int
                    ]
                    five_series.append(
                        (
                            "pe",
                            "prefill exec（p95）",
                            const("prefillExecP95", num_arr(pe95)),
                            "danger",
                        )
                    )
            e2e_max = max((p.get("e2e_p95", 0) or 0) for p in per_second)
            ttft_max = max((p.get("ttft_p95", 0) or 0) for p in per_second)
            five_max = max(e2e_max, ttft_max, 1)
            five_cap = (
                "x = 压测时间（s，1s 采样）；y = 延迟 p95（ms）。口径："
                "e2e/ttft = 成功请求按发送秒的分位（幸存者口径，过载下慢"
                "请求已转为错误被排除）；prefill/decode exec = 完成流（含 "
                "cancel）按完成秒窗口的分位（全量口径）"
            )
            if any("e2e_n" in p for p in per_second):
                five_cap += "；e2e 每秒样本量见 e2e_n"
            latency_containers.append(
                emit_container(
                    "五延迟：e2e / ttft / schedule / prefill exec / decode exec",
                    five_cap,
                    emit_chart(
                        "LineChart",
                        TSEC,
                        250,
                        five_series,
                        suffix=" ms",
                        domain="[0, " + num(nice_max(five_max * 1.15)) + "]",
                    ),
                )
            )
    if stage_lat:
        latency_containers.append(
            emit_container(
                "master 内部分阶段延迟（p50 / p95，全程终态分位）",
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

    # 2. 队列（avg/engine：集群总量 ÷ 引擎数，engine_dist.engine_count 优先）
    if queue_ts:
        q_cap = "x = 压测时间（s，" + str(q_step) + "s 采样）"
        pq_series = [
            ("rr", "running 请求数（引擎侧）", p_run_req, "success"),
            ("rb", "running 批数（引擎侧）", p_run_batch, "info"),
            ("w", "waiting 请求数（引擎侧）", p_wait, "neutral"),
        ]
        if p_master_bq:
            pq_series.append(
                ("mb", "master batcher 队列（master 侧）", p_master_bq, "danger")
            )
            pq_cap = (
                q_cap
                + "；y = 请求数 / 批数（avg/engine = 集群总量 ÷ "
                + num(p_engines)
                + " 引擎）；前三条为引擎侧队列，第四条为 master 侧 batcher "
                "队列（master per-engine batcher 集群总量 ÷ 同引擎数，1s 采样 "
                "step 对齐）"
            )
        else:
            pq_cap = (
                q_cap
                + "；y = 请求数 / 批数（avg/engine = 集群总量 ÷ "
                + num(p_engines)
                + " 引擎）；均为引擎侧队列（无 master 侧序列）"
            )
        queue_containers = [
            emit_container(
                "Prefill 队列（avg/engine）",
                pq_cap,
                emit_chart("LineChart", TQ, 230, pq_series),
            ),
            emit_container(
                "Decode 队列（avg/engine）",
                q_cap
                + "；y = 请求数（avg/engine = 集群总量 ÷ "
                + num(d_engines)
                + " 引擎）",
                emit_chart(
                    "LineChart",
                    TQ,
                    230,
                    [
                        ("r", "running 请求数", d_run_req, "success"),
                        ("w", "waiting 请求数", d_wait, "neutral"),
                    ],
                ),
            ),
        ]
        if cum_batch_ref is not None:
            batch_max = max(
                max((q.get("interval_avg_batch_size", 0) or 0) for q in queue_ts),
                max((q.get("cum_avg_batch_size", 0) or 0) for q in queue_ts),
            )
            batch_series = [
                ("iv", "区间均值（interval）", avg_batch, "info"),
                ("cm", "累计均值（参照）", cum_batch_ref, "neutral"),
            ]
            batch_cap = (
                q_cap + "；y = 请求/批。主线 = 区间均值（interval：相邻采样间隔内 "
                "enqueued 增量 ÷ batches 增量，反映真实波动）；淡线 = 集群"
                "累计均值（cum，全程平均，含启动稀释）"
            )
        else:
            batch_max = max((q.get("cum_avg_batch_size", 0) or 0) for q in queue_ts)
            batch_series = [("bs", "avg batch size", avg_batch, "info")]
            batch_cap = q_cap + "；y = 请求/批（集群累计均值）"
        queue_containers.append(
            emit_container(
                "平均 batch size",
                batch_cap,
                emit_chart(
                    "LineChart",
                    TQ,
                    230,
                    batch_series,
                    suffix=" 请求/批",
                    domain="[0, " + num(nice_max(batch_max * 1.2)) + "]",
                ),
            )
        )
        # master 侧队列深度（master prometheus G3，1s 采样；label 变体已聚合）。
        # 新口径（batcher_ts_by_role）：per-engine batcher 队列按 role 拆分；
        # 只画 prefill avg/engine + 128 容量线（decode 侧队列语义不同，不画）；
        # prefill 另给 top-5 引擎序列（决策时点深度的 1s 采样近似；
        # Top/Bottom-5 全集见 2.1 节）。旧口径（仅 batcher_ts，P+D 合计）
        # 退化画集群总量。
        if batcher_ts_by_role:
            BRT = const(
                "BRT",
                str_arr(sparse_cats([r.get("t", 0) for r in batcher_ts_by_role])),
            )
            p_bq_avg = const(
                "batcherPrefillAvg",
                num_arr(
                    [
                        round((r.get("prefill", 0) or 0) / max(1, p_engines), 2)
                        for r in batcher_ts_by_role
                    ]
                ),
            )
            queue_containers.append(
                emit_container(
                    "master 队列深度：batcher prefill（avg/engine）",
                    "x = 压测时间（s，1s 采样）；y = 队列深度 / 引擎（prefill 集群总量 ÷ "
                    + num(p_engines)
                    + "）；容量线 = maxWaitingRequestsPerPrefillWorker 128（prefill 口径）",
                    emit_chart(
                        "LineChart",
                        BRT,
                        230,
                        [
                            ("pq", "prefill（avg/engine）", p_bq_avg, "info"),
                            (
                                "cap",
                                "容量 128",
                                const(
                                    "batchCap",
                                    num_arr([128] * len(batcher_ts_by_role)),
                                ),
                                "danger",
                            ),
                        ],
                    ),
                )
            )
            if batcher_top_engines_ts and not (
                (queue_top_bottom_ts.get("p_master_batcher") or {}).get("top")
            ):
                # 旧 aggregate（无 queue_top_bottom_ts 键）：退化渲染 P
                # master-batcher Top-5（命名与 2.1 节统一）；新键存在时由
                # 独立节渲染 top+bottom 全集，此处跳过避免重复。
                top_engine_keys = []
                for r in batcher_top_engines_ts:
                    for k in r:
                        if k != "t" and k not in top_engine_keys:
                            top_engine_keys.append(k)
                top_engine_keys = top_engine_keys[:5]
                if top_engine_keys:
                    TET = const(
                        "TET",
                        str_arr(
                            sparse_cats([r.get("t", 0) for r in batcher_top_engines_ts])
                        ),
                    )
                    te_colors = ["danger", "warning", "info", "success", "neutral"]
                    top_lines = []
                    for i, ekey in enumerate(top_engine_keys):
                        top_lines.append(
                            (
                                "te%d" % i,
                                ekey,
                                const(
                                    "teV%d" % i,
                                    num_arr(
                                        [
                                            r.get(ekey, 0) or 0
                                            for r in batcher_top_engines_ts
                                        ]
                                    ),
                                ),
                                te_colors[i % len(te_colors)],
                            )
                        )
                    queue_containers.append(
                        emit_container(
                            "P master-batcher 队列深度 Top-5",
                            "master batcher 队列深度（决策时点 1s 采样近似，5s 窗口）；"
                            "数据源 master.json prometheus_timeseries per-engine；"
                            "按峰值排序；容量上限 128（maxWaitingRequestsPerPrefillWorker）",
                            emit_chart("LineChart", TET, 230, top_lines),
                        )
                    )
        if batcher_ts:
            BT = const("BT", str_arr(sparse_cats([r.get("t", 0) for r in batcher_ts])))
            # routing 队列口径：与 batcher_queue_size 同源（同一 per-engine
            # batcher 队列的集群合计）。旧 priority 桶口径（routing.queue.length）
            # 尾部 stale 冻结为上报伪影，已弃用；旧 aggregate 数据仍为旧口径。
            routing_q = const(
                "routingQueue",
                num_arr([r.get("routing_queue", 0) or 0 for r in batcher_ts]),
            )
            if not batcher_ts_by_role:
                # 旧口径：batcher_queue 是 PREFILL+DECODE 引擎合计，不除
                # 引擎数、不画容量线（分母口径不匹配会误导）。
                batcher_q_total = const(
                    "batcherQueueTotal",
                    num_arr([r.get("batcher_queue", 0) or 0 for r in batcher_ts]),
                )
                queue_containers.append(
                    emit_container(
                        "master 队列深度：batcher（集群总量，P+D 合计）",
                        "x = 压测时间（s，1s 采样）；y = 队列深度（请求数，PREFILL+DECODE 集群合计）",
                        emit_chart(
                            "LineChart",
                            BT,
                            230,
                            [
                                (
                                    "bq",
                                    "batcher 队列（P+D 合计）",
                                    batcher_q_total,
                                    "info",
                                )
                            ],
                        ),
                    )
                )
            if any(r.get("routing_queue") for r in batcher_ts):
                queue_containers.append(
                    emit_container(
                        "master 队列深度：routing（集群总量，batcher 同源口径）",
                        "x = 压测时间（s，1s 采样）；y = 队列深度（请求数，集群总量）。"
                        "口径：与 batcher_queue_size 同源（同一 per-engine batcher 队列的集群合计，"
                        "尾部正确归零）；旧 priority 桶口径（routing.queue.length）尾部 stale 冻结"
                        "为上报伪影，已弃用；旧 aggregate 数据仍为旧口径",
                        emit_chart(
                            "LineChart",
                            BT,
                            230,
                            [
                                (
                                    "rq",
                                    "routing 队列（batcher 同源）",
                                    routing_q,
                                    "warning",
                                )
                            ],
                        ),
                    )
                )
        # dispatch 决策原因构成（G3 dispatch_reason_total counter 差分出的
        # 每秒 dispatch 速率；与 batcher 队列同 grid 便于对照）
        if dispatch_reason_ts:
            DRT = const(
                "DRT",
                str_arr(sparse_cats([r.get("t", 0) for r in dispatch_reason_ts])),
            )
            dr_known = (
                ("fixed_window_timeout", "fixed_window_timeout", "warning"),
                ("batch_full", "batch_full", "info"),
                ("predicted_execution_cap", "predicted_execution_cap", "success"),
            )
            dr_lines = [
                (
                    key,
                    label,
                    const(
                        "dr" + key,
                        num_arr([r.get(key, 0) or 0 for r in dispatch_reason_ts]),
                    ),
                    color,
                )
                for key, label, color in dr_known
            ]
            other_keys = set()
            for r in dispatch_reason_ts:
                for k in r:
                    if k != "t" and all(k != key for key, _, _ in dr_known):
                        other_keys.add(k)
            if other_keys:
                dr_lines.append(
                    (
                        "drOther",
                        "其他 reason",
                        const(
                            "drOther",
                            num_arr(
                                [
                                    sum((r.get(k, 0) or 0) for k in other_keys)
                                    for r in dispatch_reason_ts
                                ]
                            ),
                        ),
                        "neutral",
                    )
                )
            queue_containers.append(
                emit_container(
                    "dispatch 决策原因（每秒批次数）",
                    "x = 压测时间（s，1s 采样）；y = dispatch 批次数 / 秒（按 reason）",
                    emit_chart("LineChart", DRT, 230, dr_lines),
                )
            )
        if dispatch_batch_size_ts:
            BST = const(
                "BST",
                str_arr(sparse_cats([r.get("t", 0) for r in dispatch_batch_size_ts])),
            )
            bs_known = (
                ("fixed_window_timeout", "fixed_window_timeout", "warning"),
                ("batch_full", "batch_full", "info"),
                ("predicted_execution_cap", "predicted_execution_cap", "success"),
            )
            bs_lines = []
            for key, label, color in bs_known:
                vals = [r.get(key, 0) or 0 for r in dispatch_batch_size_ts]
                if not any(vals):
                    continue
                bs_lines.append(
                    (
                        "bs_" + key,
                        label,
                        const("bs" + key, num_arr(vals)),
                        color,
                    )
                )
            if bs_lines:
                queue_containers.append(
                    emit_container(
                        "dispatch 批大小（按 reason，引擎平均）",
                        "x = 压测时间（s，1s 采样）；y = dispatch 批大小（请求/批，按 reason 的引擎平均）",
                        emit_chart("LineChart", BST, 230, bs_lines),
                    )
                )
        lines.append("      <Divider />")
        lines.append("")
        lines.append("      <H2>2. 队列（集群总量）</H2>")
        lines.extend(emit_grid(queue_containers))
        lines.append("")

    # 2.1 队列 Top/Bottom-5 引擎同图对比（queue_top_bottom_ts；每队列
    # 一张合并图：Top-5（负载最重）与 Bottom-5（最轻）同图，图例带
    # Top / Bottom / Top+Bottom 前缀。引擎数 <10 时 top 与 bottom 集合
    # 可能重叠，交集引擎只画一条曲线（标 Top+Bottom），避免同引擎双线；
    # 引擎总数 <=5 时两集合完全重合，全部标 Top+Bottom。颜色：tone 全空
    # 走 PALETTE 分段——top 段 idx 0-4（前 5 色）、bottom 段 idx 5-9
    # （后 5 色），一图内 10 色不撞。旧 aggregate 无此键 -> 整节省略
    # （P master-batcher Top-5 由队列节退化路径渲染）。
    if queue_top_bottom_ts:
        tb_containers = []
        # (键, 标题基名, y 轴语义 + 数据源说明)
        TB_META = (
            (
                "p_master_batcher",
                "P master-batcher 队列深度",
                "per-engine master batcher 队列深度（决策时点 1s 采样近似，"
                "5s 窗口）；数据源 master.json prometheus_timeseries；"
                "按峰值排序；容量上限 128（maxWaitingRequestsPerPrefillWorker）",
            ),
            (
                "p_running",
                "P running",
                "per-engine prefill running 请求数（mock 引擎 1s 采样，5s 窗口）；"
                "数据源 mock_per_engine_timeseries.json.gz；按峰值排序",
            ),
            (
                "p_waiting",
                "P waiting",
                "per-engine prefill waiting 请求数（mock 引擎 1s 采样，5s 窗口）；"
                "数据源 mock_per_engine_timeseries.json.gz；按峰值排序；"
                "全零 = 引擎侧无等待积压",
            ),
            (
                "d_running",
                "D running",
                "per-engine decode running 请求数（mock 引擎 1s 采样，5s 窗口）；"
                "数据源 mock_per_engine_timeseries.json.gz；按峰值排序",
            ),
            (
                "d_waiting",
                "D waiting",
                "per-engine decode waiting 请求数（mock 引擎 1s 采样，5s 窗口）；"
                "数据源 mock_per_engine_timeseries.json.gz；按峰值排序；"
                "全零 = 引擎侧无等待积压",
            ),
        )

        def _tb_ips(rows):
            """从 rows 提取保序去重的引擎 IP 列表（插入序 = 峰值序）。"""
            out = []
            for r in rows:
                for k in r:
                    if k != "t" and k not in out:
                        out.append(k)
            return out

        for tb_key, tb_title, tb_cap in TB_META:
            tb_entry = queue_top_bottom_ts.get(tb_key) or {}
            top_rows = tb_entry.get("top") or []
            bot_rows = tb_entry.get("bottom") or []
            top_ips = _tb_ips(top_rows)[:5]
            bot_ips = _tb_ips(bot_rows)[:5]
            if not top_ips and not bot_ips:
                continue
            bot_set = set(bot_ips)
            top_set = set(top_ips)
            # 合并序列：Top 集合（含交集，标 Top / Top+Bottom）在前，
            # Bottom 减交集（纯 Bottom）在后；交集引擎只画一条线。
            merged = []
            for ip in top_ips:
                tag = ("Top+Bottom·" if ip in bot_set else "Top·") + ip
                merged.append((ip, tag, top_rows))
            for ip in bot_ips:
                if ip in top_set:
                    continue
                merged.append((ip, "Bottom·" + ip, bot_rows))
            if not merged:
                continue
            # x 轴：优先 top rows 的 t 网格（实测 top/bottom 一致）；
            # bottom-only 曲线若 t 网格不同则按前向 step 对齐。
            rows_ref = top_rows or bot_rows
            t_grid = [r.get("t", 0) for r in rows_ref]
            tb_cats = const(
                "tbT" + tb_key,
                str_arr(sparse_cats(t_grid)),
            )
            tb_series = []
            for i, (ip, label, rows) in enumerate(merged):
                if rows is rows_ref:
                    vals = [r.get(ip, 0) or 0 for r in rows]
                else:
                    pts = [(r.get("t", 0), r.get(ip, 0) or 0) for r in rows]
                    vals = ts_step_values(pts, t_grid)
                tb_series.append(
                    (
                        "tb%d" % i,
                        label,
                        const("tbV" + tb_key + str(i), num_arr(vals)),
                        None,
                    )
                )
            tb_containers.append(
                emit_container(
                    tb_title + " Top-5 / Bottom-5",
                    "x = 压测时间（s，5s 窗口采样）；y = "
                    + tb_cap
                    + "；Top-5（负载最重）与 Bottom-5（最轻）同图对比，"
                    "图例 Top/Bottom 前缀区分（交集引擎仅一条线，标"
                    " Top+Bottom）",
                    emit_chart("LineChart", tb_cats, 230, tb_series),
                )
            )
        if tb_containers:
            lines.append("      <Divider />")
            lines.append("")
            lines.append("      <H2>2.1 队列 Top-5 / Bottom-5 引擎（同图对比）</H2>")
            lines.extend(emit_grid(tb_containers))
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

    # 3.1 引擎维度分布（x = 池内引擎排名，按指标值降序；三口径各自独立图）
    has_31 = bool(
        p_reqs
        or d_reqs
        or p_toks
        or d_toks
        or p_util
        or d_util
        or p_ly
        or d_ly
        or p_tok_ly
        or d_tok_ly
    )
    if has_31:
        if not (p_wg_pts or d_wg_pts):
            lines.append("      <Divider />")
            lines.append("")
        dist_containers = []
        if p_reqs:
            dist_containers.append(
                emit_container(
                    "Prefill 引擎请求数分布",
                    "x = prefill 引擎排名（1.."
                    + str(len(p_reqs))
                    + "，左端请求数最多）；y = 每引擎请求数（个）",
                    emit_chart(
                        "LineChart",
                        PRANK,
                        230,
                        [("req", "每引擎请求数（个）", p_req_curve, "info")],
                    ),
                )
            )
        if p_toks:
            dist_containers.append(
                emit_container(
                    "Prefill 引擎 input tokens 分布",
                    "x = prefill 引擎排名（1.."
                    + str(len(p_toks))
                    + "，左端 tokens 最多）；y = "
                    + p_tok_series_name,
                    emit_chart(
                        "LineChart",
                        PRANK_TOK,
                        230,
                        [("tok", p_tok_series_name, p_tok_curve, "neutral")],
                    ),
                )
            )
        if p_util:
            dist_containers.append(
                emit_container(
                    "Prefill 引擎利用率分布",
                    "x = prefill 引擎排名（1.."
                    + str(len(p_util))
                    + "）；y = busy/elapsed 利用率 %（并发 1，≤100%）",
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
            dist_containers.append(
                emit_container(
                    "Decode 引擎请求数分布",
                    "x = decode 引擎排名（1.."
                    + str(len(d_reqs))
                    + "，左端请求数最多）；y = 每引擎请求数（个）",
                    emit_chart(
                        "LineChart",
                        DRANK,
                        230,
                        [("req", "每引擎请求数（个）", d_req_curve, "success")],
                    ),
                )
            )
        if d_toks:
            dist_containers.append(
                emit_container(
                    "Decode 引擎 output tokens 分布",
                    "x = decode 引擎排名（1.."
                    + str(len(d_toks))
                    + "，左端 tokens 最多）；y = "
                    + d_tok_series_name,
                    emit_chart(
                        "LineChart",
                        DRANK_TOK,
                        230,
                        [("tok", d_tok_series_name, d_tok_curve, "neutral")],
                    ),
                )
            )
        if d_util:
            dist_containers.append(
                emit_container(
                    "Decode 引擎利用率分布",
                    "x = decode 引擎排名（1.."
                    + str(len(d_util))
                    + "）；y = busy/elapsed = 平均并发请求数（软并发，可超 100%）",
                    emit_chart(
                        "LineChart",
                        DRANK_UTIL,
                        230,
                        [("util", "decode 利用率（%）", d_util_curve, "warning")],
                        suffix="%",
                    ),
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
                    "洛伦兹曲线：请求数（P / D）",
                    "x = 引擎累计占比 %（从最轻到最重）；y = 请求数累计占比 %",
                    emit_chart("LineChart", LORENZ_X, 230, lz_series, suffix="%"),
                )
            )
        if p_tok_ly or d_tok_ly:
            lzt_series = []
            if p_tok_ly:
                lzt_series.append(("p", "prefill 洛伦兹", p_tok_lorenz_y, "info"))
            if d_tok_ly:
                lzt_series.append(("d", "decode 洛伦兹", d_tok_lorenz_y, "success"))
            dist_containers.append(
                emit_container(
                    "洛伦兹曲线：tokens（P / D）",
                    "x = 引擎累计占比 %（从最轻到最重）；y = tokens 累计占比 %",
                    emit_chart("LineChart", LORENZ_TOK_X, 230, lzt_series, suffix="%"),
                )
            )
        lines.append(
            "      <H2>3.1 引擎维度分布（x = 池内引擎排名，按指标值降序）</H2>"
        )
        lines.extend(emit_grid(dist_containers))
        lines.append("")

    # 4. In-flight（master scheduler + P/D 引擎侧，master G4 快照）
    if inflight_ts or inflight_age or inflight_age_by_role:
        inf_containers = []
        if inflight_ts:
            IFT = const(
                "IFT", str_arr(sparse_cats([r.get("t", 0) for r in inflight_ts]))
            )
            inf_sched = const(
                "infSched",
                num_arr([r.get("scheduler", 0) or 0 for r in inflight_ts]),
            )
            inf_pb = const(
                "infPB",
                num_arr([r.get("prefill_batches", 0) or 0 for r in inflight_ts]),
            )
            inf_dr = const(
                "infDR",
                num_arr([r.get("decode_requests", 0) or 0 for r in inflight_ts]),
            )
            inf_containers.append(
                emit_container(
                    "In-flight：master scheduler / prefill 批 / decode 请求",
                    "x = 压测时间（s，快照采样）；y = in-flight 数（集群总量）",
                    emit_chart(
                        "LineChart",
                        IFT,
                        230,
                        [
                            ("sch", "scheduler in-flight", inf_sched, "warning"),
                            ("pb", "prefill in-flight 批数", inf_pb, "info"),
                            ("dr", "decode in-flight 请求数", inf_dr, "success"),
                        ],
                    ),
                )
            )
        if inflight_age or inflight_age_by_role:
            age_series = []
            if inflight_age_by_role:
                # ad2d6224+: INFLIGHT_MAX_AGE_MS 按 role 拆（scheduler ledger /
                # prefill、decode per-worker ledger，各 role 跨引擎取 max）。
                # 各 role 时间轴同源（1s prom 采样），缺失点填 0（与 ledger
                # 空闲时上报 0 的语义一致）。
                t_grid = sorted(
                    {
                        row.get("t", 0)
                        for rows in inflight_age_by_role.values()
                        for row in rows
                    }
                )
                IAT = const("IAT", str_arr(sparse_cats(t_grid)))
                age_role_known = (
                    ("scheduler", "scheduler ledger", "danger"),
                    ("prefill", "prefill 引擎（max）", "info"),
                    ("decode", "decode 引擎（max）", "success"),
                )
                role_maps = {
                    role: {row.get("t", 0): row.get("age_ms", 0) or 0 for row in rows}
                    for role, rows in inflight_age_by_role.items()
                }
                for key, label, color in age_role_known:
                    if key not in role_maps:
                        continue
                    age_series.append(
                        (
                            key,
                            label,
                            const(
                                "age" + key.capitalize(),
                                num_arr(
                                    [round(role_maps[key].get(t, 0), 1) for t in t_grid]
                                ),
                            ),
                            color,
                        )
                    )
                other_roles = sorted(
                    r for r in role_maps if r not in {k for k, _, _ in age_role_known}
                )
                for key in other_roles:
                    age_series.append(
                        (
                            key,
                            key + "（max）",
                            const(
                                "age" + key.capitalize(),
                                num_arr(
                                    [round(role_maps[key].get(t, 0), 1) for t in t_grid]
                                ),
                            ),
                            "neutral",
                        )
                    )
                age_max = max(
                    (
                        row.get("age_ms", 0) or 0
                        for rows in inflight_age_by_role.values()
                        for row in rows
                    )
                )
                age_caption = (
                    "x = 压测时间（s，1s 采样）；y = in-flight 最大 age（ms，按 role 拆分："
                    "scheduler ledger vs prefill / decode per-worker ledger）"
                )
            else:
                IAT = const(
                    "IAT", str_arr(sparse_cats([r.get("t", 0) for r in inflight_age]))
                )
                inf_age = const(
                    "infAge", num_arr([r.get("age_ms", 0) or 0 for r in inflight_age])
                )
                age_series = [("age", "max age（ms）", inf_age, "danger")]
                age_max = max((r.get("age_ms", 0) or 0) for r in inflight_age)
                age_caption = (
                    "x = 压测时间（s，1s 采样）；y = in-flight 请求最大 age（ms，"
                    "集群 max，未按 role 拆分——旧 aggregate 无 role 维度）"
                )
            inf_containers.append(
                emit_container(
                    "In-flight 最长滞留时间（按 role）",
                    age_caption,
                    emit_chart(
                        "LineChart",
                        IAT,
                        230,
                        age_series,
                        suffix=" ms",
                        domain="[0, " + num(nice_max(age_max * 1.15)) + "]",
                    ),
                )
            )
        lines.append("      <Divider />")
        lines.append("")
        lines.append("      <H2>4. In-flight</H2>")
        lines.extend(emit_grid(inf_containers))
        lines.append("")

    # 5. KV（kv_ts 集群口径优先；旧 engine_dist decode_kv 每引擎均值回退）
    kv_containers = []
    if kv_ts:
        KVT = const("KVT", str_arr(sparse_cats([r.get("t", 0) for r in kv_ts])))
        kv_used_tok = const(
            "kvUsedTokens",
            num_arr([r.get("used_tokens", 0) or 0 for r in kv_ts]),
        )
        kv_cap_vals = [r.get("capacity_tokens", 0) or 0 for r in kv_ts]
        kv_cap_max = max(kv_cap_vals) if kv_cap_vals else 0
        kv_used_pct = const(
            "kvUsedPct", num_arr([r.get("used_pct", 0) or 0 for r in kv_ts])
        )
        kv_used_lines = [("used", "KV 已用 tokens", kv_used_tok, "info")]
        if kv_cap_max > 0:
            # 总容量参考线（used+available 之和，有限容量才画；旧 aggregate
            # 无 capacity_tokens 字段则不画）
            kv_used_lines.append(
                (
                    "cap",
                    "集群总容量",
                    const("kvCapTokens", num_arr(kv_cap_vals)),
                    "danger",
                )
            )
        kv_containers.append(
            emit_container(
                "KV cache 已用（集群总量）",
                "x = 压测时间（s，1s 采样）；y = 已用 KV tokens（集群总量）"
                + (
                    "；参考线 = 集群总容量（used + available 之和）"
                    if kv_cap_max > 0
                    else "（aggregate 无 capacity_tokens，不画容量线）"
                ),
                emit_chart("LineChart", KVT, 230, kv_used_lines),
            )
        )
        kv_containers.append(
            emit_container(
                "KV cache 利用率（集群）",
                "x = 压测时间（s，1s 采样）；y = 已用 /（已用 + 可用）%",
                emit_chart(
                    "LineChart",
                    KVT,
                    230,
                    [("pct", "KV 利用率（%）", kv_used_pct, "warning")],
                    suffix="%",
                ),
            )
        )
    else:
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
    if kv_containers:
        lines.append("      <Divider />")
        lines.append("")
        lines.append("      <H2>5. KV</H2>")
        lines.extend(emit_grid(kv_containers))
        lines.append("")

    # 6. 资源（mock heap + 进程 CPU/RSS，run_meta process_usage）
    has_heap = bool(queue_ts) and any("heap_used_mb" in q for q in queue_ts)
    has_proc = bool(process_ts)
    res_containers = []
    if has_heap:
        res_containers.append(
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
            )
        )
    if has_proc:
        PT = const("PT", str_arr(sparse_cats([r.get("t", 0) for r in process_ts])))
        proc_cpu_series = []
        proc_rss_series = []
        if any("mock_cpu_pct" in r for r in process_ts):
            proc_cpu_series.append(
                (
                    "mc",
                    "mock engine",
                    const(
                        "cpuMock",
                        num_arr([r.get("mock_cpu_pct", 0) or 0 for r in process_ts]),
                    ),
                    "info",
                )
            )
            proc_rss_series.append(
                (
                    "mr",
                    "mock engine",
                    const(
                        "rssMock",
                        num_arr([r.get("mock_rss_mb", 0) or 0 for r in process_ts]),
                    ),
                    "info",
                )
            )
        if any("master_cpu_pct" in r for r in process_ts):
            proc_cpu_series.append(
                (
                    "ma",
                    "master",
                    const(
                        "cpuMaster",
                        num_arr([r.get("master_cpu_pct", 0) or 0 for r in process_ts]),
                    ),
                    "warning",
                )
            )
            proc_rss_series.append(
                (
                    "mar",
                    "master",
                    const(
                        "rssMaster",
                        num_arr([r.get("master_rss_mb", 0) or 0 for r in process_ts]),
                    ),
                    "warning",
                )
            )
        if any("client_cpu_pct" in r for r in process_ts):
            proc_cpu_series.append(
                (
                    "cl",
                    "load client（分片均值）",
                    const(
                        "cpuClient",
                        num_arr([r.get("client_cpu_pct", 0) or 0 for r in process_ts]),
                    ),
                    "success",
                )
            )
            proc_rss_series.append(
                (
                    "clr",
                    "load client（分片均值）",
                    const(
                        "rssClient",
                        num_arr([r.get("client_rss_mb", 0) or 0 for r in process_ts]),
                    ),
                    "success",
                )
            )
        if proc_cpu_series:
            res_containers.append(
                emit_container(
                    "进程 CPU 使用率",
                    "x = 压测时间（s，1s 采样）；y = CPU 使用率（%）",
                    emit_chart("LineChart", PT, 230, proc_cpu_series, suffix="%"),
                )
            )
        if proc_rss_series:
            res_containers.append(
                emit_container(
                    "进程 RSS（MB）",
                    "x = 压测时间（s，1s 采样）；y = 常驻内存（MB）",
                    emit_chart("LineChart", PT, 230, proc_rss_series),
                )
            )
    if res_containers:
        lines.append("      <Divider />")
        lines.append("")
        lines.append("      <H2>6. 资源</H2>")
        lines.extend(emit_grid(res_containers))
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
    # 错误构成（top 子桶）：优先 summary.error_breakdown（含无时间戳行，
    # 与 error_count 同口径）；旧 aggregate 无此键时退化为 per_second
    # 汇总口径（仅带时间戳行）。
    err_breakdown = dict(sm.get("error_breakdown") or {})
    if not err_breakdown and per_second:
        err_breakdown = {k: v for k, v in err_totals.items() if v}
    if err_breakdown:
        _eb_label = {k: lb for k, lb, _ in ERR_DEFS}
        _eb_items = sorted(err_breakdown.items(), key=lambda kv: -kv[1])
        eb_parts = []
        for k, v in _eb_items[:6]:
            if not v:
                continue
            eb_parts.append((_eb_label.get(k) or k) + " " + fmt_int_trunc(v))
        if len(_eb_items) > 6:
            eb_parts.append("+" + str(len(_eb_items) - 6) + " 桶")
        if eb_parts:
            rows.append(["错误构成（top 子桶）", " · ".join(eb_parts)])
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
    p_tg = ed_p.get("tokens_gini_cum")
    d_tg = ed_d.get("tokens_gini_cum")
    p_ug = (util_block.get("prefill") or {}).get("gini_cum")
    d_ug = (util_block.get("decode") or {}).get("gini_cum")
    bal_parts = []
    if pg is not None or dg is not None:
        bal_parts.append("请求 Gini " + fmt_g3(pg) + " / " + fmt_g3(dg))
    if p_tg is not None or d_tg is not None:
        bal_parts.append("token Gini " + fmt_g3(p_tg) + " / " + fmt_g3(d_tg))
    if p_ug is not None or d_ug is not None:
        bal_parts.append("利用率 Gini " + fmt_g3(p_ug) + " / " + fmt_g3(d_ug))
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
    if kv_ts:
        rows.append(
            [
                "KV 峰值利用率（集群）",
                fmt_ms(max((r.get("used_pct", 0) or 0) for r in kv_ts)) + "%",
            ]
        )
    if batcher_ts_by_role:
        b_parts = []
        p_vals = [
            r.get("prefill", 0) or 0 for r in batcher_ts_by_role if "prefill" in r
        ]
        d_vals = [r.get("decode", 0) or 0 for r in batcher_ts_by_role if "decode" in r]
        if p_vals:
            b_parts.append(
                "prefill " + num(round(max(p_vals) / p_engines, 2)) + "/引擎"
            )
        if d_vals:
            if d_engines:
                b_parts.append(
                    "decode " + num(round(max(d_vals) / d_engines, 2)) + "/引擎"
                )
            else:
                b_parts.append("decode " + num(max(d_vals)) + "（集群）")
        if b_parts:
            rows.append(["master 队列深度峰值（by role）", " / ".join(b_parts)])
    elif batcher_ts:
        bq_vals = [
            r.get("batcher_queue", 0) or 0 for r in batcher_ts if "batcher_queue" in r
        ]
        rq_vals = [
            r.get("routing_queue", 0) or 0 for r in batcher_ts if "routing_queue" in r
        ]
        b_parts = []
        if bq_vals:
            b_parts.append("batcher " + num(max(bq_vals)) + "（P+D 集群）")
        if rq_vals:
            b_parts.append("routing " + num(max(rq_vals)) + "（集群）")
        if b_parts:
            rows.append(["master 队列深度峰值", " / ".join(b_parts)])
    if dispatch_reason_ts:
        dr_sums = {}
        for r in dispatch_reason_ts:
            for k, v in r.items():
                if k != "t" and v:
                    dr_sums[k] = dr_sums.get(k, 0.0) + v
        dr_order = [
            "fixed_window_timeout",
            "batch_full",
            "predicted_execution_cap",
        ]
        dr_order += sorted(k for k in dr_sums if k not in dr_order)
        dr_parts = [
            k + " " + num(round(dr_sums[k], 1)) for k in dr_order if dr_sums.get(k)
        ]
        if dr_parts:
            rows.append(["dispatch reason 批次数", " / ".join(dr_parts)])
    if batch_size_final:
        bs_order = [
            "fixed_window_timeout",
            "batch_full",
            "predicted_execution_cap",
        ]
        bs_order += sorted(k for k in batch_size_final if k not in bs_order)
        bs_parts = []
        for k in bs_order:
            e = batch_size_final.get(k) or {}
            if not e:
                continue
            bs_parts.append(
                k
                + " p50 "
                + num(e.get("p50"))
                + " / max "
                + num(e.get("max"))
                + "（"
                + num(e.get("engines"))
                + " 引擎终值）"
            )
        if bs_parts:
            rows.append(["dispatch 批大小分布（终值）", " / ".join(bs_parts)])
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

    # subtitle（HTML 页头副标）：走绝对路径 + P/D 规模，比 tsx 表内 sources
    # 更详细。tsx 内 sources 保持相对文件名/run_id，方便报告作为附件流转。
    src_abs = [os.path.abspath(args.aggregate)]
    if summary_standalone is not None:
        src_abs.append(os.path.abspath(summary_path))
    if slo is not None:
        src_abs.append(os.path.abspath(slo_path))
    if ed is not None:
        embedded = args.engine_dist is None and isinstance(agg.get("engine_dist"), dict)
        src_abs.append(
            "(aggregate 内嵌 engine_dist)" if embedded else os.path.abspath(ed_path)
        )
    _run_dir_abs = os.path.abspath(os.path.dirname(args.aggregate) or ".")
    if os.path.basename(_run_dir_abs) in ("analysis", "load_client"):
        _run_dir_abs = os.path.dirname(_run_dir_abs)
    scale_bits = []
    if p_engines is not None:
        scale_bits.append("P=" + str(p_engines))
    if d_engines is not None:
        scale_bits.append("D=" + str(d_engines))
    if shards is not None:
        scale_bits.append("shards=" + str(shards))
    if args.replay is not None:
        scale_bits.append("replay=" + str(args.replay))
    if duration_s:
        scale_bits.append("duration=" + str(int(duration_s)) + "s")
    subtitle_html = (
        "数据源："
        + " · ".join(src_abs)
        + " · run 目录 "
        + _run_dir_abs
        + (" · 规模 " + " ".join(scale_bits) if scale_bits else "")
    )

    # ---- 拼 in-memory tsx（供末端反抽 spec，不写盘）----
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

    footer = ["  );", "}"]
    tsx_src = "\n".join(header + lines + footer) + "\n"

    # ---- 反抽 spec → 渲染 Chart.js HTML ----
    # canvas_report_render_html.py 是同目录的 sibling 模块；作为脚本运行时
    # 脚本目录自动在 sys.path 中，作为库被 import 时也依赖同目录可达。
    _this_dir = os.path.dirname(os.path.abspath(__file__))
    if _this_dir not in sys.path:
        sys.path.insert(0, _this_dir)
    import canvas_report_render_html  # noqa: E402

    spec = _extract_spec_from_tsx(tsx_src, run_id=str(run_id), subtitle=subtitle_html)
    html_out = canvas_report_render_html.render(spec)

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html_out)

    # ---- stdout 摘要 ----
    sections = ["qps"] if per_second else []
    if latency_containers:
        sections.append("latency")
    if queue_ts:
        sections.append("queue")
    if (
        batcher_ts
        or batcher_ts_by_role
        or batcher_top_engines_ts
        or queue_top_bottom_ts
        or dispatch_reason_ts
        or dispatch_batch_size_ts
    ):
        sections.append("batcher")
    if p_wg_pts or d_wg_pts:
        sections.append("balance")
    if has_31:
        sections.append("dist")
    if inflight_ts or inflight_age:
        sections.append("inflight")
    if kv_containers:
        sections.append("kv")
    if res_containers:
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
