#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FlexLB 压测多 aggregate 对照报告生成器：self-contained Chart.js 4.4.7 HTML。

与 canvas_report_gen.py（单 run 深度报告）互补：本脚本把多个
aggregate_canvas_run.py 输出的 agg JSON 拼成一份**纯图表对照页** ——
头部 KPI 紧凑表 + 图表区（queueWait 分位 / waiting 曲线 / 每引擎
waiting / 引擎均衡 / 双口径延迟 / flush 健康度 / 10s 调度分位）+
页脚溯源。零分析/解读/结论文字；全部数据由 agg 字段驱动，字段缺失
的图自动跳过并在「已跳过图表」小注中列出，缺失的 KPI 单元格显示
—，不编造。

用法：
  python3 html_report_gen.py \
      --aggregate pre-300=agg_pre300.json \
      --aggregate pre-600=agg_pre600.json \
      --out report.html \
      [--run-id <id>] [--title <标题>]

  --aggregate 语法为 [label=]path（可重复传入）；label 缺省取 agg 的
  meta.run_dir（再缺省取文件名去扩展名）。--out 必填；--run-id 可选
  （写入页头）；--title 可选（缺省自动拼）。

agg 字段探测（aggregate_canvas_run.py 输出；缺失即降级）：
  summary.error_count / test_valid / actual_send_qps /
  summary.schedule_latency_ms.p99（调度口径 KPI，缺则回退
  latency_summary.sched_p99_ms）
  latency_summary.err_rows / sched_p99_ms / e2e_p99_ms / status_dist
  navi_queue_wait_stats（图 a 分位 + KPI 非零占比）
  engine_waiting_ts（图 b/c 曲线 + KPI waiting 峰值 / 均衡比）
  engine_accepted（图 d per-engine accepted 柱状）
  navi_flush_stats / navi_flush_ts（图 f flush 健康度）
  sched_latency_10s（图 g 逐 10s 调度分位 + completion_qps 双轴）

生成后内置自检：HTML 标签闭合（剥离 script/style 的栈配对）、
panel id 唯一且 canvas 创建与 Chart 初始化同循环（运行时防御）、
注入数据串无 NaN/Infinity/undefined 字面量；折线类 panel 的 series
数据必须是 [数字, 数字] 数组对且 HTML 中每个 parsing:false 声明都有
xyPairs() 转换配对（Chart.js 跳过解析时 [x,y] 数组对会静默空图，
此防线让该不兼容在生成期报出）；本机有 node 时对主 script 做
--check 语法冒烟。
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile

TAG = "[html_report_gen]"

# 系列/aggregate 调色板（与 canvas_report_render_html.py 同源）
PALETTE = [
    "#1677ff",
    "#52c41a",
    "#faad14",
    "#f5222d",
    "#722ed1",
    "#13c2c2",
    "#eb2f96",
    "#fa8c16",
    "#a0d911",
    "#2f54eb",
    "#fadb14",
    "#08979c",
]

# per_request 成功态 status 值域（status_dist 排除法回退口径用；
# aggregate 主口径 latency_summary.err_rows 不依赖该集合）
OK_STATUS = ("ok", "scheduled")


def esc(s):
    """HTML 实体转义（label / title 等用户可控串进 innerHTML 前必转。"""
    return html.escape(str(s), quote=True)


def num(v):
    """数值清洗：None/NaN/Inf/非数 -> None（JS null，图上断点/KPI —）。"""
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    if f == int(f) and abs(f) < 1e15:
        return int(f)
    return round(f, 4)


def fmt(v, nd=1):
    """数值 -> 展示串：None -> '—'；整数千分位；小数保留 nd 位。"""
    if v is None:
        return "—"
    if isinstance(v, int) or (isinstance(v, float) and v == int(v)):
        return "{:,}".format(int(v))
    return "{:,.{nd}f}".format(v, nd=nd)


def parse_aggregate_arg(s):
    """'label=path' 或 'path' -> (label|None, path)。"""
    if "=" in s:
        label, _, path = s.partition("=")
        label = label.strip()
        return (label or None), path.strip()
    return None, s.strip()


# ---------------------------------------------------------------------------
# 单个 agg 的字段提取（全部存在性探测，缺 -> None，不抛错）
# ---------------------------------------------------------------------------


def agg_err_e2e(agg):
    """端到端口径错误行数：优先 latency_summary.err_rows，缺则用
    status_dist 排除成功态回退；再缺 -> None。"""
    ls = agg.get("latency_summary") or {}
    if isinstance(ls, dict) and ls.get("err_rows") is not None:
        return ls["err_rows"]
    sd = ls.get("status_dist")
    if isinstance(sd, dict) and sd:
        return sum(v for k, v in sd.items() if k not in OK_STATUS and k != "")
    return None


def agg_err_sched(agg):
    """调度口径错误数：客户端 summary.error_count。"""
    return (agg.get("summary") or {}).get("error_count")


def agg_err_main_class(agg):
    """端到端口径主错误类（非成功态 status 中计数最大者）。"""
    sd = (agg.get("latency_summary") or {}).get("status_dist")
    if not isinstance(sd, dict):
        return None
    bad = {k: v for k, v in sd.items() if k not in OK_STATUS and k != ""}
    if not bad:
        return None
    return max(bad.items(), key=lambda kv: kv[1])[0]


def agg_sched99(agg):
    """调度口径 sched p99：优先客户端 summary.schedule_latency_ms.p99
    （与旧 export_chart_data.py KPI 同源），缺则回退 per_request 口径
    latency_summary.sched_p99_ms。"""
    sl = (agg.get("summary") or {}).get("schedule_latency_ms")
    if isinstance(sl, dict) and sl.get("p99") is not None:
        return num(sl["p99"])
    return num((agg.get("latency_summary") or {}).get("sched_p99_ms"))


def agg_total99(agg):
    """端到端口径 total p99：latency_summary.e2e_p99_ms（成功行
    total_ms 最近秩 p99，与旧 e2e_caliber 同口径）。"""
    return num((agg.get("latency_summary") or {}).get("e2e_p99_ms"))


def agg_qw_stats(agg):
    st = agg.get("navi_queue_wait_stats")
    return st if isinstance(st, dict) and st else None


def agg_ewt(agg):
    ew = agg.get("engine_waiting_ts")
    if isinstance(ew, dict) and ew.get("rows"):
        return ew
    return None


def agg_flush_stats(agg):
    st = agg.get("navi_flush_stats")
    return st if isinstance(st, dict) and st else None


def engine_short(name):
    """引擎名缩写：'prefill-0:1.2.3.4:8000' -> 'prefill-0'；无冒号原样。"""
    s = str(name)
    return s.split(":", 1)[0] if ":" in s else s


def kpi_row(label, agg):
    """头部 KPI 表一行（7 个指标单元格，缺失 -> None 显示 —）。"""
    sm = agg.get("summary") or {}
    qws = agg_qw_stats(agg) or {}
    ews = (agg_ewt(agg) or {}).get("summary") or {}
    err_e2e = agg_err_e2e(agg)
    err_sched = agg_err_sched(agg)
    err_class = agg_err_main_class(agg)
    if err_e2e:
        err_v, err_sub = err_e2e, "端到端口径" + (
            " · " + err_class if err_class else ""
        )
    elif err_sched:
        err_v, err_sub = err_sched, "调度口径（客户端 summary）"
    else:
        err_v, err_sub = 0, "调度 / 端到端口径均 0"
    valid = sm.get("test_valid")
    return {
        "label": label,
        "err": num(err_v),
        "errSub": err_sub,
        "valid": None if valid is None else ("通过" if valid else "未通过"),
        "aqps": num(sm.get("actual_send_qps")),
        "sched99": agg_sched99(agg),
        "total99": agg_total99(agg),
        "qwShare": num(qws.get("nonzero_share_pct")),
        "wPeak": num(ews.get("peak")),
    }


# ---------------------------------------------------------------------------
# 面板构建（缺数据的图跳过并记录原因）
# ---------------------------------------------------------------------------


def build_panels(entries, warnings):
    """entries: [(label, path, agg)]。返回 (panels, skipped)。"""
    panels = []
    skipped = []

    def series_colors(n):
        return [PALETTE[i % len(PALETTE)] for i in range(n)]

    # ---- 图 a：queueWait 分位（p50/p99/max 分组柱状，对数轴） ----
    a_labels, a_p50, a_p99, a_max, a_zero = [], [], [], [], []
    for label, _path, agg in entries:
        qws = agg_qw_stats(agg)
        if not qws:
            continue
        p50, p99, mx = (
            num(qws.get(k)) for k in ("node_p50_ms", "node_p99_ms", "node_max_ms")
        )
        if p50 is None and p99 is None and mx is None:
            continue
        a_labels.append(label)
        a_p50.append(p50)
        a_p99.append(p99)
        a_max.append(mx)
        if p50 == 0 and p99 == 0 and mx == 0:
            a_zero.append(label)
    if a_labels:
        panels.append(
            {
                "id": "qw-quantiles",
                "title": "queueWait 分位对照",
                "caption": "单位 ms · 口径：flexlb_navi_queue_wait 节点最终估计"
                "（Java 侧 max(ledger_ms, engine_ms) 后打出）· y 对数轴"
                "（全 0 组不显示柱，图内标注「恒 0」）",
                "kind": "bar",
                "labels": a_labels,
                "series": [
                    {"name": "p50", "data": a_p50, "color": PALETTE[1]},
                    {"name": "p99", "data": a_p99, "color": PALETTE[0]},
                    {"name": "max", "data": a_max, "color": PALETTE[3]},
                ],
                "logY": True,
                "unit": "ms",
                "zeroLabels": a_zero,
            }
        )
    else:
        skipped.append(
            {
                "name": "queueWait 分位对照",
                "reason": "无 aggregate 携带 navi_queue_wait_stats"
                "（非 navi run 或旧 run 无该日志行）",
            }
        )

    # ---- 图 b：prefill waiting 总量时间曲线（逐秒，系列 = aggregate） ----
    b_series = []
    for label, _path, agg in entries:
        ew = agg_ewt(agg)
        if not ew:
            continue
        data = [[num(r[0]), num(sum(r[1:]))] for r in ew["rows"]]
        data = [p for p in data if p[0] is not None and p[1] is not None]
        if data:
            b_series.append({"name": label, "data": data})
    if b_series:
        for i, s in enumerate(b_series):
            s["color"] = PALETTE[i % len(PALETTE)]
        panels.append(
            {
                "id": "waiting-total",
                "title": "prefill waiting 总量时间曲线",
                "caption": "单位 tokens · 口径：mock 引擎 prefill 队列逐秒采样，"
                "全引擎求和 · t=0 为首个客户端发送时刻（负值为预热段）",
                "kind": "xy",
                "series": b_series,
                "unit": "tokens",
            }
        )
    else:
        skipped.append(
            {
                "name": "prefill waiting 总量时间曲线",
                "reason": "无 aggregate 携带 engine_waiting_ts.rows"
                "（缺 mock_per_engine_timeseries.json.gz）",
            }
        )

    # ---- 图 c：每引擎 waiting 折线（每 aggregate 一个子图） ----
    c_subs = []
    for label, _path, agg in entries:
        ew = agg_ewt(agg)
        if not ew:
            continue
        engs = ew.get("engines") or []
        series = []
        for i, en in enumerate(engs):
            data = [
                [num(r[0]), num(r[i + 1]) if i + 1 < len(r) else None]
                for r in ew["rows"]
            ]
            data = [p for p in data if p[0] is not None and p[1] is not None]
            if data:
                series.append(
                    {
                        "name": engine_short(en),
                        "data": data,
                        "color": PALETTE[i % len(PALETTE)],
                        "axis": 1,
                    }
                )
        if series:
            c_subs.append(
                {"label": label, "series": series, "y1Unit": "tokens", "y2Unit": None}
            )
    if c_subs:
        panels.append(
            {
                "id": "waiting-per-engine",
                "title": "每引擎 prefill waiting",
                "caption": "单位 tokens · 口径：各 prefill 引擎队列逐秒采样 · "
                "每个 aggregate 一个子图",
                "kind": "per-agg",
                "subs": c_subs,
            }
        )
    else:
        skipped.append(
            {
                "name": "每引擎 prefill waiting",
                "reason": "无 aggregate 携带 engine_waiting_ts.rows",
            }
        )

    # ---- 图 d1：per-engine accepted 柱状（x=引擎，系列=aggregate） ----
    d_engs = sorted(
        {e for _l, _p, agg in entries for e in (agg.get("engine_accepted") or {})}
    )
    if d_engs:
        d_series = []
        for i, (label, _path, agg) in enumerate(entries):
            ea = agg.get("engine_accepted") or {}
            if not ea:
                continue
            d_series.append(
                {
                    "name": label,
                    "data": [num(ea.get(e)) for e in d_engs],
                    "color": PALETTE[i % len(PALETTE)],
                }
            )
        if d_series:
            panels.append(
                {
                    "id": "engine-accepted",
                    "title": "per-engine accepted（prefill 路由计数）",
                    "caption": "单位 req · 口径：per_request ok 行按 prefill 引擎"
                    "路由计数（成功请求分布）",
                    "kind": "bar",
                    "labels": [engine_short(e) for e in d_engs],
                    "series": d_series,
                    "unit": "req",
                }
            )
        else:
            skipped.append(
                {
                    "name": "per-engine accepted",
                    "reason": "engine_accepted 均为空（无 per_request "
                    "数据或全部行失败）",
                }
            )
    else:
        skipped.append(
            {
                "name": "per-engine accepted",
                "reason": "无 aggregate 携带 engine_accepted",
            }
        )

    # ---- 图 d2：last60 max/min ratio 对比条 ----
    r_labels, r_data = [], []
    for label, _path, agg in entries:
        ratio = num((agg_ewt(agg) or {}).get("summary", {}).get("last60_max_min_ratio"))
        if ratio is None:
            continue
        r_labels.append(label)
        r_data.append(ratio)
    if r_labels:
        panels.append(
            {
                "id": "balance-ratio",
                "title": "引擎均衡：last60 max/min waiting 比",
                "caption": "口径：1s 序列最后 12 个样本（约 last 60s）各引擎 "
                "waiting max/min 比值的均值 · 1.0 = 完全均衡",
                "kind": "bar",
                "labels": r_labels,
                "series": [
                    {"name": "max/min ratio", "data": r_data, "color": PALETTE[0]}
                ],
                "unit": "",
            }
        )
    else:
        skipped.append(
            {
                "name": "引擎均衡 last60 max/min ratio",
                "reason": "无 aggregate 携带 "
                "engine_waiting_ts.summary."
                "last60_max_min_ratio",
            }
        )

    # ---- 图 e：双口径延迟分组柱状（sched p99 vs total p99） ----
    e_labels, e_sched, e_total = [], [], []
    for label, _path, agg in entries:
        s99, t99 = agg_sched99(agg), agg_total99(agg)
        if s99 is None and t99 is None:
            continue
        e_labels.append(label)
        e_sched.append(s99)
        e_total.append(t99)
    if e_labels:
        vals = [v for v in e_sched + e_total if v]
        log_y = bool(vals) and max(vals) / max(min(vals), 1) > 20
        panels.append(
            {
                "id": "latency-dual",
                "title": "双口径 p99 延迟对照",
                "caption": "单位 ms · sched p99 = 客户端 summary "
                "schedule_latency_ms（调度口径）；total p99 = 成功行 "
                "total_ms 分位（端到端口径）"
                + (" · y 对数轴（跨度大自动启用）" if log_y else ""),
                "kind": "bar",
                "labels": e_labels,
                "series": [
                    {
                        "name": "sched p99（调度口径）",
                        "data": e_sched,
                        "color": PALETTE[0],
                    },
                    {
                        "name": "total p99（端到端口径）",
                        "data": e_total,
                        "color": PALETTE[3],
                    },
                ],
                "logY": log_y,
                "unit": "ms",
            }
        )
    else:
        skipped.append(
            {
                "name": "双口径 p99 延迟对照",
                "reason": "无 aggregate 携带 latency_summary / "
                "summary.schedule_latency_ms",
            }
        )

    # ---- 图 f1：flush gap gt200ms 占比 ----
    f_labels, f_data = [], []
    for label, _path, agg in entries:
        share = num((agg_flush_stats(agg) or {}).get("gap_gt200ms_share_pct"))
        if share is None:
            continue
        f_labels.append(label)
        f_data.append(share)
    if f_labels:
        panels.append(
            {
                "id": "flush-gt200",
                "title": "flush 健康度：行间隔 >200ms 占比",
                "caption": "单位 % · 口径：相邻 flexlb_navi_queue_wait 日志行间隔"
                "（flush 窗口节拍）超过 200ms 的窗口占比（全 run）",
                "kind": "bar",
                "labels": f_labels,
                "series": [{"name": "gt200 占比", "data": f_data, "color": PALETTE[2]}],
                "unit": "%",
            }
        )
    else:
        skipped.append(
            {
                "name": "flush gt200 占比",
                "reason": "无 aggregate 携带 navi_flush_stats" ".gap_gt200ms_share_pct",
            }
        )

    # ---- 图 f2：flush p50 逐 30s 桶曲线 ----
    f2_series = []
    for i, (label, _path, agg) in enumerate(entries):
        fts = agg.get("navi_flush_ts")
        if not isinstance(fts, list) or not fts:
            continue
        data = []
        for r in fts:
            if not isinstance(r, list) or len(r) < 2:
                continue
            x, y = num(r[0]), num(r[1])
            if x is not None and y is not None:
                data.append([x, y])
        if data:
            f2_series.append(
                {"name": label, "data": data, "color": PALETTE[i % len(PALETTE)]}
            )
    if f2_series:
        panels.append(
            {
                "id": "flush-p50-buckets",
                "title": "flush 行间隔 p50 · 逐 30s 桶",
                "caption": "单位 ms · 口径：相邻 flexlb_navi_queue_wait 日志行"
                "间隔按 30s 桶统计的 p50（窗口节拍漂移曲线）",
                "kind": "xy",
                "series": f2_series,
                "unit": "ms",
            }
        )
    else:
        skipped.append(
            {
                "name": "flush p50 逐 30s 桶曲线",
                "reason": "无 aggregate 携带 navi_flush_ts",
            }
        )

    # ---- 图 g：逐 10s 调度 p50/p95/p99 + completion_qps 双轴 ----
    g_subs = []
    for label, _path, agg in entries:
        s10 = agg.get("sched_latency_10s")
        if not isinstance(s10, list) or not s10:
            continue
        series = []
        for si, (name, ci, color) in enumerate(
            [("p50", 1, PALETTE[1]), ("p95", 2, PALETTE[2]), ("p99", 3, PALETTE[3])]
        ):
            data = []
            for r in s10:
                if not isinstance(r, list) or len(r) <= ci:
                    continue
                x, y = num(r[0]), num(r[ci])
                if x is not None and y is not None:
                    data.append([x, y])
            if data:
                series.append({"name": name, "data": data, "color": color, "axis": 1})
        qdata = []
        for r in s10:
            if not isinstance(r, list) or len(r) < 5:
                continue
            x, y = num(r[0]), num(r[4])
            if x is not None and y is not None:
                qdata.append([x, y])
        if qdata:
            series.append(
                {
                    "name": "completion_qps",
                    "data": qdata,
                    "color": PALETTE[0],
                    "axis": 2,
                    "dashed": True,
                }
            )
        if series:
            g_subs.append(
                {"label": label, "series": series, "y1Unit": "ms", "y2Unit": "req/s"}
            )
    if g_subs:
        panels.append(
            {
                "id": "sched-10s",
                "title": "逐 10s 调度延迟分位 + completion_qps",
                "caption": "单位 ms / req/s · 口径：per_request schedule_ms 按 "
                "10s 发送桶分位（左轴）；master counters completion"
                "_count 差分 / 10s（右轴虚线）· 每个 aggregate 一个"
                "子图",
                "kind": "per-agg",
                "subs": g_subs,
            }
        )
    else:
        skipped.append(
            {
                "name": "逐 10s 调度分位 + completion_qps",
                "reason": "无 aggregate 携带 sched_latency_10s",
            }
        )

    return panels, skipped


# ---------------------------------------------------------------------------
# HTML 模板（自包含；唯一外部依赖 Chart.js 4.4.7 CDN）
# ---------------------------------------------------------------------------

_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__PAGE_TITLE__</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<style>
:root {
  --bg: #F5F6FA; --card: #FFFFFF;
  --ink: rgba(0,0,0,.85); --ink-2: rgba(0,0,0,.65); --ink-3: rgba(0,0,0,.45);
  --line: rgba(0,0,0,.06);
  --danger: #f5222d; --success: #52c41a;
  --r-input: 4px; --r-card: 8px; --r-panel: 16px;
  --sp-1: 4px; --sp-2: 8px; --sp-3: 16px; --sp-4: 24px; --sp-5: 32px;
  --font-body: -apple-system,BlinkMacSystemFont,'PingFang SC','Hiragino Sans GB','Microsoft YaHei','Helvetica Neue',Arial,sans-serif;
  --font-num: 'DIN Alternate','Helvetica Neue',Arial,sans-serif;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
html { -webkit-text-size-adjust: 100%; }
body { background: var(--bg); color: var(--ink); font-family: var(--font-body); font-size: 14px; line-height: 1.6; }
.wrap { max-width: 1280px; margin: 0 auto; padding: var(--sp-4) var(--sp-3) var(--sp-5); }
.page-header { background: var(--card); border: 1px solid var(--line); border-radius: var(--r-panel); padding: var(--sp-4) var(--sp-5); margin-bottom: var(--sp-4); }
h1 { font-size: 22px; font-weight: 700; letter-spacing: .2px; }
.sub { font-size: 13px; color: var(--ink-2); margin-top: var(--sp-1); }
.sub-2 { font-size: 12px; color: var(--ink-3); margin-top: var(--sp-1); }
.card { background: var(--card); border: 1px solid var(--line); border-radius: var(--r-card); padding: var(--sp-4); margin-bottom: var(--sp-3); }
.card h2 { font-size: 16px; font-weight: 700; margin-bottom: var(--sp-1); }
.cap { font-size: 12px; color: var(--ink-3); margin-bottom: var(--sp-2); }
.box { height: 300px; position: relative; }
.tbl-wrap { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { padding: 7px 10px; text-align: right; border-bottom: 1px solid var(--line); white-space: nowrap; }
th { color: var(--ink-2); font-weight: 600; background: rgba(0,0,0,.02); }
th small { color: var(--ink-3); font-weight: 400; }
td.txt, th.txt { text-align: left; }
td .cal { display: block; font-size: 11px; color: var(--ink-3); }
td .ok-t { color: var(--success); font-weight: 600; }
td .bad-t { color: var(--danger); font-weight: 600; }
td .err-n { color: var(--danger); font-weight: 700; font-family: var(--font-num); }
td.num { font-family: var(--font-num); }
.sub-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: var(--sp-3); }
.sub-item h3 { font-size: 13px; font-weight: 700; color: var(--ink-2); margin: var(--sp-2) 0 var(--sp-1); }
.sub-item .box { height: 240px; }
.hint { margin: var(--sp-2) 0 var(--sp-3); color: var(--ink-3); font-size: 12px; }
.hint b { color: var(--ink-2); }
.foot { margin-top: var(--sp-3); color: var(--ink-3); font-size: 12px; line-height: 1.9; }
.foot code { font-family: var(--font-num); background: rgba(0,0,0,.03); border: 1px solid var(--line); border-radius: var(--r-input); padding: 1px 6px; }
@media (max-width: 900px) { .sub-grid { grid-template-columns: 1fr; } .box { height: 260px; } }
</style>
</head>
<body>
<div class="wrap">
<header class="page-header">
  <h1>__TITLE__</h1>
  <div class="sub" id="subtitle"></div>
  <div class="sub-2" id="subtitle2"></div>
</header>
<main>
<section class="card" aria-label="KPI 对照表">
  <h2>KPI 对照</h2>
  <div class="tbl-wrap"><table id="kpi-table"></table></div>
</section>
<div id="panels"></div>
<div id="skipped" class="hint"></div>
</main>
<footer class="card" aria-label="溯源">
  <h2>溯源</h2>
  <div class="tbl-wrap"><table id="trace-table"></table></div>
  <div class="foot" id="foot-meta"></div>
</footer>
</div>
<script>
const SPEC = __SPEC_JSON__;

// ---- Chart.js 全局默认（viz-html-render 通用规范） ----
Chart.defaults.font.size = 12;
Chart.defaults.font.family = getComputedStyle(document.body).fontFamily;
Chart.defaults.color = 'rgba(0,0,0,.45)';
Chart.defaults.borderColor = 'rgba(0,0,0,.06)';
const TOOLTIP_STYLE = {
  backgroundColor: 'rgba(0,0,0,.78)', titleColor: '#fff', bodyColor: '#fff',
  cornerRadius: 4, padding: 10, boxPadding: 4,
};
const LEGEND_OPTS = {
  position: 'bottom',
  labels: { boxWidth: 10, boxHeight: 10, usePointStyle: true, padding: 14 },
};

function fmtNum(v) {
  if (v === null || v === undefined) return '—';
  if (Math.abs(v) >= 1e15) return v.toExponential(2);
  if (Number.isInteger(v)) return v.toLocaleString('en-US');
  return Number(v.toFixed(2)).toLocaleString('en-US',
    { minimumFractionDigits: 0, maximumFractionDigits: 2 });
}

// ---- [x,y] 数组对 → {x,y} 对象：parsing:false 要求数据为预解析对象，
// SPEC 为省体积用数组对紧凑序列化（数百点 × 多图），折线 dataset
// 构造时经此统一转换；已是 {x,y} 形态的元素透传（防御）。 ----
const xyPairs = (arr) => arr.map(d => Array.isArray(d) ? { x: d[0], y: d[1] } : d);

// ---- 「恒 0」标注插件：对数轴下全 0 组无柱，图内补文字 ----
const zeroMarkPlugin = {
  id: 'zeroMark',
  afterDatasetsDraw(chart, _args, opts) {
    if (!opts || !opts.labels || !opts.labels.length) return;
    const ctx = chart.ctx, area = chart.chartArea;
    ctx.save();
    ctx.fillStyle = 'rgba(0,0,0,.45)';
    ctx.font = '12px ' + Chart.defaults.font.family;
    ctx.textAlign = 'center';
    opts.labels.forEach(l => {
      const x = chart.scales.x.getPixelForValue(l);
      if (x >= area.left - 1 && x <= area.right + 1) {
        ctx.fillText('恒 0', x, area.top + 26);
      }
    });
    ctx.restore();
  }
};

function makeSection(p) {
  const sec = document.createElement('section');
  sec.className = 'card';
  const h2 = document.createElement('h2');
  h2.textContent = p.title;
  const cap = document.createElement('div');
  cap.className = 'cap';
  cap.textContent = p.caption;
  sec.appendChild(h2); sec.appendChild(cap);
  return sec;
}
function makeCanvas(sec, id, h) {
  const fig = document.createElement('figure');
  fig.className = 'box';
  if (h) fig.style.height = h + 'px';
  const cv = document.createElement('canvas');
  cv.id = id;
  fig.appendChild(cv);
  sec.appendChild(fig);
  return cv;
}
function tooltipLabel(unit) {
  return { callbacks: { label: c => {
    const v = c.parsed.y !== undefined ? c.parsed.y : c.parsed;
    return c.dataset.label + ': ' + fmtNum(v) + (unit || '');
  } } };
}
function axisMoney(o) { return Object.assign({ beginAtZero: true }, o || {}); }

// ---- 渲染器 1：分组柱状（labels × series） ----
function renderBar(p, host) {
  const sec = makeSection(p);
  const cv = makeCanvas(sec, 'c-' + p.id);
  host.appendChild(sec);
  const datasets = p.series.map(s => ({
    label: s.name, data: s.data, backgroundColor: s.color + 'D9',
    borderColor: s.color, borderWidth: 1, borderRadius: 4,
    maxBarThickness: 36,
  }));
  const yScale = p.logY
    ? { type: 'logarithmic', ticks: { callback: v => fmtNum(v) } }
    : axisMoney();
  new Chart(cv, {
    type: 'bar',
    data: { labels: p.labels, datasets: datasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: LEGEND_OPTS,
        tooltip: Object.assign({}, TOOLTIP_STYLE, tooltipLabel(p.unit)),
        zeroMark: { labels: p.zeroLabels || [] },
      },
      scales: {
        x: { grid: { display: false } },
        y: yScale,
      },
    },
    plugins: [zeroMarkPlugin],
  });
}

// ---- 渲染器 2：{x,y} 折线（linear x 轴，多序列） ----
function renderXY(p, host) {
  const sec = makeSection(p);
  const cv = makeCanvas(sec, 'c-' + p.id);
  host.appendChild(sec);
  const datasets = p.series.map(s => ({
    label: s.name, data: xyPairs(s.data), borderColor: s.color,
    backgroundColor: s.color, borderWidth: 2, pointRadius: 0,
    tension: 0.3, fill: false,
  }));
  new Chart(cv, {
    type: 'line',
    data: { datasets: datasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      interaction: { mode: 'nearest', intersect: false },
      parsing: false,
      plugins: {
        legend: LEGEND_OPTS,
        tooltip: Object.assign({}, TOOLTIP_STYLE, {
          callbacks: { label: c => c.dataset.label + ': ' + fmtNum(c.parsed.y) + (p.unit || '') },
        }),
      },
      scales: {
        x: { type: 'linear', grid: { display: false },
             ticks: { maxRotation: 0, autoSkip: true, maxTicksLimit: 12 } },
        y: axisMoney(),
      },
    },
  });
}

// ---- 渲染器 3：每 aggregate 一个子图（series 可带 axis:1/2 双 y 轴） ----
function renderPerAgg(p, host) {
  const sec = makeSection(p);
  const grid = document.createElement('div');
  grid.className = 'sub-grid';
  sec.appendChild(grid);
  host.appendChild(sec);
  p.subs.forEach((sub, idx) => {
    const item = document.createElement('div');
    item.className = 'sub-item';
    const h3 = document.createElement('h3');
    h3.textContent = sub.label;
    item.appendChild(h3);
    const cv = document.createElement('canvas');
    cv.id = 'c-' + p.id + '-' + idx;
    const fig = document.createElement('figure');
    fig.className = 'box';
    fig.appendChild(cv);
    item.appendChild(fig);
    grid.appendChild(item);
    const hasAxis2 = sub.series.some(s => s.axis === 2);
    const datasets = sub.series.map(s => ({
      label: s.name, data: xyPairs(s.data), borderColor: s.color,
      backgroundColor: s.color, borderWidth: 2, pointRadius: 0,
      tension: 0.3, fill: false, yAxisID: s.axis === 2 ? 'y2' : 'y',
      borderDash: s.dashed ? [6, 4] : undefined,
    }));
    const scales = {
      x: { type: 'linear', grid: { display: false },
           ticks: { maxRotation: 0, autoSkip: true, maxTicksLimit: 10 } },
      y: axisMoney({ position: 'left',
                     title: { display: !!sub.y1Unit, text: sub.y1Unit || '' } }),
    };
    if (hasAxis2) {
      scales.y2 = axisMoney({ position: 'right',
                              title: { display: !!sub.y2Unit, text: sub.y2Unit || '' },
                              grid: { drawOnChartArea: false } });
    }
    new Chart(cv, {
      type: 'line',
      data: { datasets: datasets },
      options: {
        responsive: true, maintainAspectRatio: false,
        interaction: { mode: 'nearest', intersect: false },
        parsing: false,
        plugins: {
          legend: LEGEND_OPTS,
          tooltip: Object.assign({}, TOOLTIP_STYLE, {
            callbacks: { label: c => c.dataset.label + ': ' + fmtNum(c.parsed.y) +
              (c.dataset.yAxisID === 'y2' ? ' ' + (sub.y2Unit || '') : ' ' + (sub.y1Unit || '')) },
          }),
        },
        scales: scales,
      },
    });
  });
}

// ---- KPI 表 ----
function renderKpi() {
  const cols = SPEC.kpi.cols;
  const table = document.getElementById('kpi-table');
  const thead = document.createElement('thead');
  const hr = document.createElement('tr');
  cols.forEach(c => {
    const th = document.createElement('th');
    th.innerHTML = c.h;
    if (c.cls) th.className = c.cls;
    hr.appendChild(th);
  });
  thead.appendChild(hr); table.appendChild(thead);
  const tbody = document.createElement('tbody');
  SPEC.kpi.rows.forEach(r => {
    const tr = document.createElement('tr');
    const lt = document.createElement('td');
    lt.className = 'txt'; lt.textContent = r.label;
    tr.appendChild(lt);
    r.cells.forEach(c => {
      const td = document.createElement('td');
      td.className = 'num';
      if (c.ok === true || c.ok === false) {
        const t = document.createElement('span');
        t.className = c.ok ? 'ok-t' : 'bad-t';
        t.textContent = c.v;
        td.appendChild(t);
      } else {
        const v = document.createElement('span');
        if (c.err) v.className = 'err-n';
        v.textContent = c.v;
        td.appendChild(v);
        if (c.sub) {
          const s = document.createElement('span');
          s.className = 'cal'; s.textContent = c.sub;
          td.appendChild(s);
        }
      }
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
}

// ---- 溯源表 ----
function renderTrace() {
  const table = document.getElementById('trace-table');
  const thead = document.createElement('thead');
  const hr = document.createElement('tr');
  ['aggregate', 'run_id (meta.run_dir)', '数据文件', 'meta 字段'].forEach(h => {
    const th = document.createElement('th');
    th.className = 'txt'; th.textContent = h;
    hr.appendChild(th);
  });
  thead.appendChild(hr); table.appendChild(thead);
  const tbody = document.createElement('tbody');
  SPEC.trace.rows.forEach(r => {
    const tr = document.createElement('tr');
    [r.label, r.runDir, r.file, r.meta].forEach(v => {
      const td = document.createElement('td');
      td.className = 'txt'; td.textContent = v || '—';
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
}

// ---- 主流程 ----
document.title = SPEC.title;
document.querySelector('h1').textContent = SPEC.title;
document.getElementById('subtitle').textContent = SPEC.subtitle;
document.getElementById('subtitle2').textContent = SPEC.subtitle2;
renderKpi();
const host = document.getElementById('panels');
SPEC.panels.forEach(p => {
  if (p.kind === 'bar') renderBar(p, host);
  else if (p.kind === 'xy') renderXY(p, host);
  else if (p.kind === 'per-agg') renderPerAgg(p, host);
});
if (SPEC.skipped && SPEC.skipped.length) {
  const el = document.getElementById('skipped');
  const b = document.createElement('b');
  b.textContent = '已跳过图表（' + SPEC.skipped.length + '）：';
  el.appendChild(b);
  el.appendChild(document.createTextNode(
    SPEC.skipped.map(s => s.name + '（' + s.reason + '）').join(' · ')));
}
renderTrace();
document.getElementById('foot-meta').innerHTML = SPEC.footHtml;
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# 生成后自检
# ---------------------------------------------------------------------------

_VOID_TAGS = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "param",
    "source",
    "track",
    "wbr",
}


def _check_tags_balanced(html_text):
    """剥离 script/style 后做开/闭标签栈配对；返回问题串或 None。"""
    body = re.sub(r"<script\b.*?</script>", "", html_text, flags=re.S | re.I)
    body = re.sub(r"<style\b.*?</style>", "", body, flags=re.S | re.I)
    stack = []
    for m in re.finditer(r"<(/?)([a-zA-Z][a-zA-Z0-9]*)\b[^>]*?(/?)>", body):
        close, name, self_close = m.group(1), m.group(2).lower(), m.group(3)
        if name in _VOID_TAGS or self_close:
            continue
        if close:
            if not stack or stack[-1] != name:
                top = stack[-1] if stack else "<empty>"
                return "unbalanced </%s> (stack top: %s)" % (name, top)
            stack.pop()
        else:
            stack.append(name)
    if stack:
        return "unclosed tags: %s" % ", ".join(stack)
    return None


def _check_main_script_syntax(html_text):
    """node 可用时对主 <script>（无 src）做语法冒烟；返回问题串或 None。"""
    node = shutil.which("node")
    if not node:
        return None
    blocks = re.findall(r"<script>(.*?)</script>", html_text, flags=re.S)
    if not blocks:
        return None
    with tempfile.NamedTemporaryFile(
        "w", suffix=".js", delete=False, encoding="utf-8"
    ) as tf:
        tf.write(blocks[-1])
        path = tf.name
    try:
        r = subprocess.run(
            [node, "--check", path], capture_output=True, text=True, timeout=30
        )
        if r.returncode != 0:
            return "node --check failed: " + (r.stderr or "").strip()[:400]
        return None
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def self_check(html_text, spec, spec_json_text):
    """内置自检；返回问题列表（空 = 通过）。"""
    problems = []
    tag_problem = _check_tags_balanced(html_text)
    if tag_problem:
        problems.append("HTML 标签不闭合: " + tag_problem)
    if re.search(r"\bNaN\b|\bInfinity\b|\b-Infinity\b|\bundefined\b", spec_json_text):
        problems.append("注入数据串含 NaN/Infinity/undefined 字面量")
    ids = [p["id"] for p in spec.get("panels", [])]
    if len(ids) != len(set(ids)):
        problems.append("panel id 重复")
    # canvas id 与 Chart 初始化一一对应：canvas 与 Chart 在同一渲染函数/
    # 同一循环内创建（模板结构保证），此处校验每个 panel 数据结构完整，
    # 保证运行时循环不会中途抛错（series.data 必须是数组）。
    for p in spec.get("panels", []):
        for s in p.get("series", []) or []:
            if not isinstance(s.get("data"), list):
                problems.append(
                    "panel %s 系列 %s 数据非数组" % (p["id"], s.get("name"))
                )
        for sub in p.get("subs", []) or []:
            for s in sub.get("series", []) or []:
                if not isinstance(s.get("data"), list):
                    problems.append(
                        "panel %s 子图 %s 系列 %s 数据非数组"
                        % (p["id"], sub.get("label"), s.get("name"))
                    )
    # ---- 防线：parsing:false 与数据形态兼容性（浏览器静默空图教训） ----
    # Chart.js parsing:false 要求数据为预解析 {x,y} 对象；SPEC 序列化为
    # [x,y] 数组对（省体积），折线渲染器经模板内 xyPairs() 统一转换。
    # ① 折线类 panel 的 series 数据必须是 [数字, 数字] 数组对（无 null
    #    分量）——形态漂移在生成期报出，而非浏览器端空图；
    # ② HTML 中 parsing:false 声明数不得超过 xyPairs() 转换调用数
    #    ——新增跳过解析的 dataset 忘记转换时在此拦截。
    for p in spec.get("panels", []):
        if p.get("kind") not in ("xy", "per-agg"):
            continue
        line_series = list(p.get("series", []) or [])
        for sub in p.get("subs", []) or []:
            line_series.extend(sub.get("series", []) or [])
        for s in line_series:
            for i, pt in enumerate(s.get("data", []) or []):
                ok_pt = (
                    isinstance(pt, (list, tuple))
                    and len(pt) == 2
                    and all(
                        isinstance(v, (int, float)) and not isinstance(v, bool)
                        for v in pt
                    )
                )
                if not ok_pt:
                    problems.append(
                        "panel %s 系列 %s 第 %d 个点非 [数字, 数字] 数组对"
                        "（parsing:false 渲染器要求可经 xyPairs 转换的形态）"
                        % (p["id"], s.get("name"), i)
                    )
                    break
    n_parse = len(re.findall(r"parsing:\s*false\s*[,\n]", html_text))
    n_conv = len(re.findall(r"xyPairs\(", html_text))
    if "xyPairs" not in html_text or n_conv < n_parse:
        problems.append(
            "parsing:false 声明 %d 处但 xyPairs() 转换仅 %d 处"
            "（跳过解析的 dataset 数据未转换会静默空图）" % (n_parse, n_conv)
        )
    syntax_problem = _check_main_script_syntax(html_text)
    if syntax_problem:
        problems.append(syntax_problem)
    return problems


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(
        description="FlexLB 压测多 aggregate 纯图表对照 HTML 报告生成器"
        "（Chart.js 4.4.7 self-contained；字段缺失自动降级）"
    )
    ap.add_argument(
        "--aggregate",
        required=True,
        action="append",
        metavar="[LABEL=]PATH",
        help="aggregate_canvas_run.py 输出的 agg JSON，可重复；"
        "语法 label=path，label 缺省取 agg 的 meta.run_dir",
    )
    ap.add_argument(
        "--out", required=True, help="输出 .html 路径（self-contained Chart.js HTML）"
    )
    ap.add_argument("--run-id", help="run 标识（写入页头，可选）")
    ap.add_argument("--title", help="报告标题（缺省自动拼）")
    args = ap.parse_args()

    warnings = []
    entries = []  # (label, path, agg)
    seen_labels = set()
    for raw in args.aggregate:
        label, path = parse_aggregate_arg(raw)
        if not path:
            print(TAG + " empty --aggregate path: " + raw, file=sys.stderr)
            sys.exit(2)
        if not os.path.isfile(path):
            print(TAG + " aggregate not found: " + path, file=sys.stderr)
            sys.exit(2)
        try:
            with open(path) as f:
                agg = json.load(f)
        except (OSError, ValueError) as e:
            print(
                TAG + " aggregate JSON 解析失败: %s (%s)" % (path, e), file=sys.stderr
            )
            sys.exit(2)
        if not isinstance(agg, dict):
            print(TAG + " aggregate 顶层不是 JSON object: " + path, file=sys.stderr)
            sys.exit(2)
        if label is None:
            label = (agg.get("meta") or {}).get("run_dir") or os.path.splitext(
                os.path.basename(path)
            )[0]
        if label in seen_labels:
            warnings.append("label 重复，自动加序号: " + label)
            i = 2
            while "%s #%d" % (label, i) in seen_labels:
                i += 1
            label = "%s #%d" % (label, i)
        seen_labels.add(label)
        entries.append((label, path, agg))

    title = args.title or (
        "FlexLB 压测对照报告" + (" · " + args.run_id if args.run_id else "")
    )
    labels_txt = " / ".join(l for l, _p, _a in entries)
    subtitle = "%d 个 aggregate：%s" % (len(entries), labels_txt)
    subtitle2 = (
        "数据：aggregate_canvas_run.py 聚合输出 · 生成："
        "html_report_gen.py · 图表：Chart.js 4.4.7（CDN）· "
        "口径标注见各图标题下小注"
    )

    panels, skipped = build_panels(entries, warnings)

    kpi_cols = [
        {"h": "aggregate"},
        {"h": "错误数", "sub": True},
        {"h": "test_valid"},
        {"h": "actual_qps<br><small>req/s</small>"},
        {"h": "sched p99<br><small>ms · 调度口径</small>"},
        {"h": "total p99<br><small>ms · 端到端口径</small>"},
        {"h": "queueWait 非零占比<br><small>%</small>"},
        {"h": "waiting 峰值<br><small>tokens</small>"},
    ]
    kpi_rows = []
    trace_rows = []
    for label, path, agg in entries:
        k = kpi_row(label, agg)
        kpi_rows.append(
            {
                "label": label,
                "cells": [
                    {"v": fmt(k["err"]), "sub": k["errSub"], "err": bool(k["err"])},
                    {
                        "v": k["valid"] or "—",
                        "ok": None if k["valid"] is None else (k["valid"] == "通过"),
                    },
                    {"v": fmt(k["aqps"], 3)},
                    {"v": fmt(k["sched99"])},
                    {"v": fmt(k["total99"])},
                    {"v": fmt(k["qwShare"], 2)},
                    {"v": fmt(k["wPeak"])},
                ],
            }
        )
        meta = agg.get("meta") or {}
        meta_bits = []
        for key in sorted(meta):
            if key != "run_dir":
                meta_bits.append("%s=%s" % (key, meta[key]))
        sm = agg.get("summary") or {}
        if sm.get("total_requests") is not None:
            meta_bits.append("summary.total_requests=%s" % sm["total_requests"])
        trace_rows.append(
            {
                "label": label,
                "runDir": meta.get("run_dir") or "—",
                "file": os.path.basename(path),
                "meta": " · ".join(meta_bits) or "—",
            }
        )

    foot_lines = [
        "时间锚点：t=0 为首个客户端发送时刻（per_request epoch0）；负值为预热段。",
        "queueWait 口径：navi 节点最终估计 = Java 侧 max(ledger_ms, engine_ms)"
        " 后打出的 queue_wait_ms 列。",
        "flush 口径：相邻 flexlb_navi_queue_wait 日志行间隔（flush 窗口节拍），"
        "按 30s 桶统计。",
        "延迟口径：sched = 客户端 summary schedule_latency_ms（调度）；"
        "total = 成功行 total_ms 分位（端到端）。",
    ]
    foot_html = "".join("<p>%s</p>" % esc(x) for x in foot_lines)

    spec = {
        "title": title,
        "subtitle": subtitle,
        "subtitle2": subtitle2,
        "kpi": {"cols": kpi_cols, "rows": kpi_rows},
        "panels": panels,
        "skipped": skipped,
        "trace": {"rows": trace_rows},
        "footHtml": foot_html,
    }
    spec_json = json.dumps(
        spec, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )

    html_text = (
        _TEMPLATE.replace("__PAGE_TITLE__", esc(title))
        .replace("__TITLE__", esc(title))
        .replace("__SPEC_JSON__", spec_json)
    )

    problems = self_check(html_text, spec, spec_json)
    if problems:
        for p in problems:
            print(TAG + " SELF-CHECK FAIL: " + p, file=sys.stderr)
        sys.exit(1)

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html_text)

    print(TAG + " aggregates=%d labels=%s" % (len(entries), labels_txt))
    print(TAG + " panels=%d skipped=%d" % (len(panels), len(skipped)))
    for s in skipped:
        print(TAG + " skipped: " + s["name"] + " (" + s["reason"] + ")")
    print(TAG + " warnings=%d" % len(warnings))
    for w in warnings:
        print(TAG + " warning: " + w)
    print(
        TAG + " self-check: OK (tags balanced, panel ids unique, "
        "no NaN/Infinity/undefined, xy series are [x,y] number pairs, "
        "parsing:false datasets pass xyPairs()%s)"
        % (", node --check passed" if shutil.which("node") else "")
    )
    print(TAG + " written=%s (%.1f KB)" % (args.out, len(html_text) / 1024.0))


if __name__ == "__main__":
    main()
