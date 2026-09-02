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

头部三层规范化（2026-09）：
  * subtitle（一眼看清实验条件）：拓扑 + 发送模式/倍率 + ramp/duration/
    shards，形如「12P + 40D mock · replay@82x（名义 650 QPS）· ramp 30s
    · 120s · 8 shards」。倍率取数源 = aggregate meta replay_speed
    （REPLAY_SPEED 自动校准值），不再吃 CLI 缺省（replay@1000x bug：
    旧版 argparse default=1000 被当倍率显示，而 1000 是旧档显式速度
    恰与 trace 行数同值的巧合）；
  * KPI 结果行（头部第二行 chip）：请求数量 / 成功数量 / 失败·cancel /
    成功率 / 持续时间（summary 结果类字段；cancel 为 error_count 的
    具名子桶 err_cancelled，chip 上注记子集数）；
  * detail（<details> 折叠，默认收起）：代码版本（branch/commit）、测试
    数据集（trace 路径/行数/sha256）、实验参数（run_meta.params 全量）、
    环境变量（client_env / flexlb_env FINAL ENV 快照）、数据源（绝对
    路径）。旧 meta 三分区中的数据源从可见面板移入 detail；规模信息由
    subtitle 实验条件行承担，不设分区（信息重复）；时间轴口径 + 采样
    说明保留在可见 meta 面板（口径标注纪律）。

报告级统一时间轴：全部时序面板（x = 压测时间）共享同一 x 轴 [0, T_END]。
T_END = 全部时序面板最大采样点（ceil 整秒，含收尾排空）；min 固定 0
（t=0 = 压测正式开始，warmup 后；warmup 负值段在渲染层被轴裁剪，数据
保留不删）。实现：main() 里各时间轴 cats const 生成时注册到 cats_time
（const 名 → 数值时间序列），反抽时按 x_ref 给面板附 timeX/xNums，spec
顶层注入 timeAxis，渲染层用 linear x 轴钉 TA_MIN/TA_MAX。非时间轴面板
（分布/排名/洛伦兹/分阶段 BarChart）不注册、保持类目轴。

用法：
  python3 canvas_report_gen.py --aggregate <agg.json> \
      [--engine-dist <engine_dist.json>] \
      [--slo <slo_batch_analysis.json>] \
      --out <out.html> [--run-id <id>] \
      [--p-engines 750] [--d-engines 500] [--shards 8] \
      [--send-mode replay] [--replay <speed>] \
      [--git-branch <b>] [--git-commit <c>]

缺省规则：
  * --slo 未指定时取 aggregate 同目录同名文件（存在才读）；
  * engine_dist 来源优先级：--engine-dist 显式指定 > aggregate 顶层内嵌键
    （aggregate_canvas_run.py 已把 engine_dist 计算进 aggregate，一个脚本
    出全部数据）> aggregate 同目录 engine_dist.json；
  * --run-id 未指定时取 aggregate 的 meta.run_dir；
  * P/D 引擎数优先取 engine_dist 的 engine_count，其次 --p-engines/--d-engines；
  * shards 优先取 aggregate.summary 的 load_client_workers，其次 --shards，再缺省 8；
  * 实验条件（send_mode/replay_speed/名义 QPS/ramp/数据集/配置）优先取
    aggregate meta（aggregate_canvas_run.py 20260902+ 写入），其次同目录
    run_meta.json params，最后 CLI 显式参数；均缺则 subtitle 对应段省略
    （fail-closed，不再回退硬编码缺省倍率）；
  * git branch/commit 优先取 aggregate meta（远端重聚合时经
    FLEXLB_GIT_BRANCH/FLEXLB_GIT_COMMIT 注入），其次 CLI，均缺则 detail
    显示 —（未提供）；
  * 2.3 TPS P/D 主图为每引擎平均（集群和 ÷ 引擎数，生产大盘单实例
    series 同构读法；mock_tps_ts/aggregate 语义不动，折算在呈现层）。
    折算引擎数可靠链：aggregate 同目录 run_meta.json 的
    params.n_prefill/n_decode > engine_dist.engine_count > mock.json
    final_snapshot 角色计数；均缺失时回退集群和呈现（caption 明示
    「集群和（引擎数未知）」+ stderr 告警），不吃 --p-engines CLI 缺省
    （750/500 为面板刻度缺省而非真实引擎数）。
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


def _extract_spec_from_tsx(tsx_src, run_id, subtitle, time_axes=None, time_axis=None):
    """从内存 tsx 字符串反抽 render_html.render() 需要的 spec dict。

    time_axes: {cats const 名: 数值时间序列}——时间轴面板注册表，
    panel 的 x_ref 命中时附 timeX=True + xNums（与类目标签等长同序）。
    time_axis: 报告级统一时间轴 {min, max}，写入 spec 顶层供渲染层钉轴。
    """
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
            panel = {
                "id": "p%d" % pid,
                "title": title,
                "caption": caption,
                "type": "bar" if ctype == "BarChart" else "line",
                "x": x_vals,
                "yMax": y_max,
                "unit": (suf_m.group(1).strip() if suf_m else ""),
                "series": series,
            }
            t_vals = (time_axes or {}).get(x_ref)
            if t_vals is not None:
                panel["timeX"] = True
                panel["xNums"] = t_vals
            panels.append(panel)

    spec = {
        "run_id": run_id,
        "title": "FlexLB 压测报告 · run " + run_id,
        "subtitle": subtitle,
        # KPI 两行：第一行五连（发送 QPS / 成功调度 QPS / 错误率 / Gini /
        # pacing）+ 第二行结果五连（请求数量 / 成功 / 失败·cancel / 成功率 /
        # 持续时间），共 10 chip；头部 Grid 区之外无其它 Stat 使用点。
        "kpis": kpis[:10],
        "panels": panels,
    }
    if time_axis:
        spec["timeAxis"] = time_axis
    return spec


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
        "--send-mode",
        help=(
            "发送模式（replay/uniform；缺省取 aggregate meta.send_mode，"
            "再缺省同目录 run_meta.json params.send_mode，最后回退 replay）"
        ),
    )
    ap.add_argument(
        "--replay",
        type=int,
        default=None,
        help=(
            "replay 倍速（优先取 aggregate meta.replay_speed 即 REPLAY_SPEED"
            " 自动校准值，再取同目录 run_meta.json params.replay_speed；"
            "均缺则 subtitle 不显示倍率段——不再回退硬编码缺省，"
            "旧版 default=1000 造成的 replay@1000x 误显示即此根因）"
        ),
    )
    ap.add_argument(
        "--git-branch",
        help="代码分支（优先取 aggregate meta.git_branch；均缺则 detail 显示 —）",
    )
    ap.add_argument(
        "--git-commit",
        help="代码 commit（优先取 aggregate meta.git_commit；均缺则 detail 显示 —）",
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

    slo_path = args.slo or os.path.join(agg_dir, "slo_batch_analysis.json")
    ed_path = args.engine_dist or os.path.join(agg_dir, "engine_dist.json")

    # no-backward-compat：--summary / summary_standalone 独立输入已删
    # （旧 run 不再支持）；summary 仅从 aggregate.summary 单键直读，
    # 缺失即无数据按可选逻辑省略。
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
    batcher_top_engines_ts = agg.get("batcher_top_engines_ts") or []
    queue_top_bottom_ts = agg.get("queue_top_bottom_ts") or {}
    mock_tps_rows = agg.get("mock_tps_ts") or []
    # 引擎侧 KV v2 块池时序（aggregate kv_blocks_ts_by_role：P/D 桶
    # 跨引擎求和，三态 gauge + 准入/复用/淘汰累计 counter；旧
    # aggregate 无键 -> 空表 -> 5. KV 块池面板整组省略）。
    kv_blocks_by_role = agg.get("kv_blocks_ts_by_role") or {}
    # cache 命中率三口径（aggregate cache_hit_ts 窗口命中率列 +
    # summary.cache_hit_summary run 级汇总；旧 aggregate 无键 ->
    # 空表/空 dict -> 5c 面板与 KPI 读数行整体省略）。
    cache_hit_rows = agg.get("cache_hit_ts") or []
    cache_hit_sm = sm.get("cache_hit_summary") or {}
    dispatch_reason_ts = agg.get("dispatch_reason_ts") or []
    dispatch_batch_size_ts = agg.get("dispatch_batch_size_ts") or []
    cancel_ts = agg.get("cancel_qps_ts") or []
    master_arrivals_ts = agg.get("master_arrivals_ts") or []
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
    validity = sm.get("validity_checks") or {}

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

    # ---- 2.3 TPS 每引擎平均折算的引擎数（20260901 呈现口径改版）----
    # TPS P/D 主图从「集群和」改为「每引擎平均」（集群和 ÷ N），与生产
    # 大盘单实例 series 的读法同构——集群和呈现下 12P 求和 p50 3.88M vs
    # 生产单实例 ~58k，观感差 67 倍。引擎数按可靠性取链：
    #   1. run_meta.json params.n_prefill/n_decode（部署配置真值；标准
    #      管线 consolidate 步骤必写，aggregate 同目录兄弟文件）；
    #   2. engine_dist.prefill/decode.engine_count（观察值：收到过流量
    #      的引擎数，健康 run 等于部署数，短/断 run 可偏小）；
    #   3. mock.json final_snapshot.engines 按角色计数（mock 自报终态）。
    # 均不可得 → None：2.3 主图回退集群和呈现（caption 明示「集群和
    # （引擎数未知）」+ stderr 告警；标准 run 引擎数恒可得，回退只是
    # 防御）。刻意不吃 --p-engines/--d-engines：CLI 缺省 750/500 是
    # 其它面板的刻度缺省而非真实引擎数，拿缺省折算会引入数十倍系统
    # 性偏差，宁可回退集群和。aggregate 的 mock_tps_ts 语义不动（保留
    # 原始集群和，无损原始数据）；折算只在 canvas 呈现层做。
    def _tps_engine_count(raw):
        try:
            iv = int(raw)
        except (TypeError, ValueError):
            return None
        return iv if iv > 0 else None

    _rm_path = os.path.join(agg_dir, "run_meta.json")
    _rm_params = (
        (load_json(_rm_path) or {}).get("params") or {}
        if os.path.isfile(_rm_path)
        else {}
    )

    # ---- 实验条件取数（subtitle 条件行 + detail 层，20260902 三层重构）----
    # 优先级：aggregate meta（aggregate_canvas_run.py 20260902+ 写入）>
    # 同目录 run_meta.json params > CLI 显式参数。全链 fail-closed：皆缺
    # 则对应段省略（replay 速度绝不回退硬编码缺省 —— replay@1000x bug
    # 根因即旧版 argparse default=1000 被直接当倍率显示）。
    _agg_meta = agg.get("meta") or {}

    def _cond_int(*raws):
        for raw in raws:
            if raw is None:
                continue
            try:
                iv = int(str(raw).strip())
            except (TypeError, ValueError):
                continue
            if iv > 0:
                return iv
        return None

    send_mode = (
        str(
            _agg_meta.get("send_mode")
            or _rm_params.get("send_mode")
            or args.send_mode
            or ""
        )
        .strip()
        .lower()
        or "replay"
    )
    replay_speed = _cond_int(
        _agg_meta.get("replay_speed"), _rm_params.get("replay_speed"), args.replay
    )
    # 名义 QPS：replay 模式 = 校准目标（650 这类）；uniform 模式 = 发送 QPS
    nominal_qps = _cond_int(
        _agg_meta.get("send_mode_qps"), _rm_params.get("send_mode_qps")
    )
    ramp_s = _cond_int(
        _agg_meta.get("ramp_up_seconds"), _rm_params.get("ramp_up_seconds")
    )
    # detail 层：代码版本 / 数据集 / 配置（同优先级链）
    git_branch = (
        str(_agg_meta.get("git_branch") or args.git_branch or "").strip() or None
    )
    git_commit = (
        str(_agg_meta.get("git_commit") or args.git_commit or "").strip() or None
    )

    tps_p_engines = _tps_engine_count(_rm_params.get("n_prefill"))
    tps_d_engines = _tps_engine_count(_rm_params.get("n_decode"))
    if tps_p_engines is None and ed:
        tps_p_engines = _tps_engine_count((ed.get("prefill") or {}).get("engine_count"))
    if tps_d_engines is None and ed:
        tps_d_engines = _tps_engine_count((ed.get("decode") or {}).get("engine_count"))
    if tps_p_engines is None or tps_d_engines is None:
        _mock_fs_path = os.path.join(agg_dir, "mock.json")
        _mock_fs = (
            (load_json(_mock_fs_path) or {}).get("final_snapshot") or {}
            if os.path.isfile(_mock_fs_path)
            else {}
        )
        _fs_roles = {}
        for _e in _mock_fs.get("engines") or []:
            if isinstance(_e, dict) and _e.get("role"):
                _fs_roles[_e["role"]] = _fs_roles.get(_e["role"], 0) + 1
        if tps_p_engines is None:
            tps_p_engines = _tps_engine_count(_fs_roles.get("prefill"))
        if tps_d_engines is None:
            tps_d_engines = _tps_engine_count(_fs_roles.get("decode"))
    # 折算除数（引擎数未知 = 1.0，即集群和原值——回退呈现语义）
    tps_p_div = float(tps_p_engines) if tps_p_engines else 1.0
    tps_d_div = float(tps_d_engines) if tps_d_engines else 1.0

    shards = args.shards
    if shards is None:
        # load_client_workers：aggregate 透传的元数据键（client.json 直读，
        # Phase B 后自然缺失 → 缺省 8）。
        shards = int(sm.get("load_client_workers") or 8)

    if per_second:
        rel_ts = rel_times([p.get("t", 0) for p in per_second])
        duration_s = rel_ts[-1] - rel_ts[0] + 1
    else:
        total0 = sm.get("total_requests") or 0
        # duration 回退估算优先 master arrival 口径：client 自估的
        # actual_send_qps 在过载 run 下明显偏低（客户端时钟窗口失真），
        # 会把 duration_s 拉长、ok_qps 压低。
        qps0 = sm.get("server_arrival_qps") or sm.get("actual_send_qps") or 0
        duration_s = int(round(total0 / float(qps0))) if qps0 else 0

    # ---- Stat 六连 ----
    total_req = sm.get("total_requests")
    success_n = sm.get("success_count")
    error_n = sm.get("error_count")
    error_rate = sm.get("error_rate")
    if error_rate is None and total_req:
        error_rate = (error_n or 0) / float(total_req)
    # 发送 QPS 口径（20260829 修正）：主值取 master arrival_qps（server 侧
    # 全量计数，docs 手册指定口径）；client 自估 actual_send_qps 降为参考值
    # 双值透出——过载 run 下 client 自估值被压低（实测 475 vs 2002），
    # 单独展示会误导容量结论。旧 aggregate 无 server_arrival_qps 时回退。
    send_qps_master = sm.get("server_arrival_qps") or 0
    send_qps_client = sm.get("actual_send_qps") or 0
    send_qps = send_qps_master or send_qps_client
    send_qps_label = (
        "发送 QPS（master）" if send_qps_master else "发送 QPS（client 自估）"
    )
    ok_qps = (float(success_n) / duration_s) if (success_n and duration_s) else 0

    # 泄漏判定：头部 KPI chip 已下线（2026-08 需求，见自检负向断言）；
    # leak_label 仅为 tsx 汇总表行保留（反抽不抽 Table，不进 HTML）。
    if validity:
        all_valid = all(bool(v) for v in validity.values())
        leak_label = "clean" if all_valid else "leak"
    else:
        leak_label = "—"
    pacing_ok = bool(validity.get("client_pacing_p99_within_limit"))
    pacing_label, pacing_tone = ("good", "success") if pacing_ok else ("bad", "danger")
    # Phase A：chip 附 pacing p99 数值（聚合层自算 client_pacing_lag_ms；
    # 无样本或旧 run 无键时退回纯 good/bad 文案不变）。
    _pacing_dist = sm.get("client_pacing_lag_ms") or {}
    if _pacing_dist.get("count") and _pacing_dist.get("p99") is not None:
        pacing_label += " · p99 " + fmt_ms(_pacing_dist["p99"]) + "ms"

    pg = (ed.get("prefill") or {}).get("gini_cum") if ed else None
    dg = (ed.get("decode") or {}).get("gini_cum") if ed else None
    # Gini 口径（20260829 修正）：主值用全量口径（全部 placement 决策行，
    # 含失败/超时行）——高错误率 run 下成功行只是幸存者子集，成功口径
    # Gini 会低估真实路由不均；成功口径保留在汇总表与洛伦兹双线对照。
    # 旧 aggregate 无 _all 键时回退成功口径。
    pg_all = (ed.get("prefill_all") or {}).get("gini_cum") if ed else None
    dg_all = (ed.get("decode_all") or {}).get("gini_cum") if ed else None
    pg_disp = pg_all if pg_all is not None else pg
    dg_disp = dg_all if dg_all is not None else dg
    gini_is_all = pg_all is not None or dg_all is not None
    if pg_disp is None and dg_disp is None:
        gini_stat, gini_tone, gini_label = "—", None, "P / D 路由 Gini"
    else:
        gini_stat, gini_tone = (
            fmt_g3(pg_disp) + " / " + fmt_g3(dg_disp),
            "success",
        )
        gini_label = "P / D 路由 Gini（全量）" if gini_is_all else "P / D 路由 Gini"

    # ---- 数据常量 ----
    consts = []  # (name, js_expr)

    def const(name, expr):
        consts.append((name, expr))
        return name

    # 报告级统一时间轴：时间轴面板的 cats const 名 → 数值 x 序列（与
    # sparse 后类目标签等长同序）。各节 cats 在生成处注册；T_END =
    # 全部注册序列的最大采样点（ceil 整秒）。非时间轴面板（分布/排名/
    # 洛伦兹/分阶段 BarChart）不注册，保持类目轴（向后兼容）。
    cats_time = {}

    def reg_time(cats_name, t_vals):
        cats_time[cats_name] = [float(v) for v in t_vals]
        return cats_name

    # 每秒时序（1s 粒度）
    # 发送序列双口径（20260830 修正）：master 到达差分序列为主发送线
    # （master_counters 1s 累计计数器差分，覆盖全部到达）；客户端
    # per_request.arrivals 降为参考线——收集器截断时它只覆盖部分窗口
    # （A 档 33,372 行止于 0-70s，每秒 ~476），不能再冒充主发送曲线。
    # master 样本丢弃负 t（首请求发送前的暖机零值），int 秒桶化
    # （~1.001s 采样间隔下每桶 0-2 样本，桶内取均值）；零值桶保留
    # （发送停止后的冻结尾巴就是曲线归零的直接证据）。
    m_arr_bucket = {}
    for r in master_arrivals_ts:
        try:
            t = float(r.get("t", 0) or 0)
            v = float(r.get("arrivals", 0) or 0)
        except (TypeError, ValueError):
            continue
        if t < 0:
            continue
        m_arr_bucket.setdefault(int(t), []).append(v)
    m_arr_by_t = {b: round(sum(vs) / len(vs), 1) for b, vs in m_arr_bucket.items()}
    TSEC = None
    qps_arrivals = qps_success = qps_errors = None
    qps_master_arrivals = None
    sched_p50 = sched_p95 = sched_p99 = None
    has_full_e2e = False
    # 出生轴引擎执行分位（per_second 新键）：has_birth_pe/has_birth_de 分别
    # 标记 prefill/decode 侧是否有非零出生轴样本；used_completion_pe/de
    # 记录面板是否画了完成轴回退线（自检断言用）。提前初始化保证非
    # per_second 分支下自检块引用安全。
    has_birth_pe = has_birth_de = False
    used_completion_pe = used_completion_de = False
    # token 长度时序（20260901）：input/output len per-second p50/p95；
    # 旧 aggregate 无 input_len_n 键时保持 None -> 2.2 节整体省略。
    input_len_p50 = input_len_p95 = None
    output_len_p50 = output_len_p95 = None
    # mock 自报 TPS 序列（20260901，同日纠偏）：2.3 节 P/D 角色主图；
    # 旧 aggregate 无 mock_tps_ts 键时保持 None -> 省略。client 侧
    # token 对账序列不再进报告（对账降级为 aggregate validity_checks
    # 的 token_reconciliation_ok 断言，检测能力保留但不占版面）。
    mock_ctx_tps = mock_ctx_cache_tps = mock_gen_tps = None
    if per_second:
        ps_by_t = {int(p.get("t", 0) or 0): p for p in per_second}
        tsec_vals = rel_ts
        if m_arr_by_t:
            # 合并时间轴：客户端行桶 ∪ master 差分桶。截断 run 的客户端
            # arrivals 只覆盖部分窗口，master 序列补全发送全程与冻结尾巴；
            # 反向桶缺失补 0（无到达/无成功/无失败样本秒）。
            tsec_vals = sorted(set(ps_by_t) | set(m_arr_by_t))
        TSEC = const("TSEC", str_arr(sparse_cats(tsec_vals)))
        reg_time(TSEC, tsec_vals)
        qps_arrivals = const(
            "qpsArrivals",
            num_arr([(ps_by_t.get(t) or {}).get("arrivals", 0) for t in tsec_vals]),
        )
        qps_success = const(
            "qpsSuccess",
            num_arr([(ps_by_t.get(t) or {}).get("success", 0) for t in tsec_vals]),
        )
        qps_errors = const(
            "qpsErrors",
            num_arr([(ps_by_t.get(t) or {}).get("errors", 0) for t in tsec_vals]),
        )
        if m_arr_by_t:
            qps_master_arrivals = const(
                "qpsMasterArrivals",
                num_arr([m_arr_by_t.get(t, 0) for t in tsec_vals]),
            )
        sched_p50 = const(
            "schedP50",
            num_arr([(ps_by_t.get(t) or {}).get("sched_p50", 0) for t in tsec_vals]),
        )
        sched_p95 = const(
            "schedP95",
            num_arr([(ps_by_t.get(t) or {}).get("sched_p95", 0) for t in tsec_vals]),
        )
        sched_p99 = const(
            "schedP99",
            num_arr([(ps_by_t.get(t) or {}).get("sched_p99", 0) for t in tsec_vals]),
        )
        # token 长度时序常量：仅当 per_second 带 input_len_n/output_len_n
        # 且存在非零样本秒时注册（区分「旧 aggregate 无键」与「全零」）。
        if any((p.get("input_len_n") or 0) for p in per_second):
            input_len_p50 = const(
                "inputLenP50",
                num_arr(
                    [(ps_by_t.get(t) or {}).get("input_len_p50", 0) for t in tsec_vals]
                ),
            )
            input_len_p95 = const(
                "inputLenP95",
                num_arr(
                    [(ps_by_t.get(t) or {}).get("input_len_p95", 0) for t in tsec_vals]
                ),
            )
        if any((p.get("output_len_n") or 0) for p in per_second):
            output_len_p50 = const(
                "outputLenP50",
                num_arr(
                    [(ps_by_t.get(t) or {}).get("output_len_p50", 0) for t in tsec_vals]
                ),
            )
            output_len_p95 = const(
                "outputLenP95",
                num_arr(
                    [(ps_by_t.get(t) or {}).get("output_len_p95", 0) for t in tsec_vals]
                ),
            )
        # mock 自报生产口径 TPS（20260901）：rtp_llm_* 集群级行序列
        # （aggregate mock_tps_ts，完成事件记账，1s scrape 窗口）桶化到
        # 整秒（桶内均值，与 master arrivals 差分同规则）画上 TSEC——
        # context 对 = P 角色主图、generate = D 角色主图（生产大盘
        # hippo_role 切分同构读法）。全零序列不注册（与 2.2 节
        # input_len_n 的「非零才画」同规则，区分无键/全零）。
        # 20260901 呈现口径：主图值为每引擎平均（集群和 ÷ 引擎数，
        # tps_p_div/tps_d_div；引擎数未知时除数 1 = 集群和回退，引擎数
        # 可靠链见 tps_p_engines 解析处注释）——mock_tps_ts/aggregate
        # 语义不动，折算只在 canvas 呈现层（与 k/M 单位换算同类的
        # 呈现层单位选择）。
        _mtps_bucket = {}
        for r in mock_tps_rows:
            try:
                _t = float(r.get("t", 0) or 0)
            except (TypeError, ValueError):
                continue
            if _t < 0:
                continue
            for _col in ("context_tps", "context_tps_with_cache", "generate_tps"):
                _v = r.get(_col)
                if _v is None:
                    continue
                _mtps_bucket.setdefault(int(_t), {}).setdefault(_col, []).append(
                    float(_v)
                )
        _mtps_by_t = {
            _b: {c: round(sum(vs) / len(vs), 1) for c, vs in _cols.items()}
            for _b, _cols in _mtps_bucket.items()
        }
        if any(v for _cols in _mtps_by_t.values() for v in _cols.values()):
            mock_ctx_tps = const(
                "mockCtxTps",
                num_arr(
                    [
                        round(
                            (_mtps_by_t.get(t) or {}).get("context_tps", 0) / tps_p_div,
                            1,
                        )
                        for t in tsec_vals
                    ]
                ),
            )
            mock_ctx_cache_tps = const(
                "mockCtxCacheTps",
                num_arr(
                    [
                        round(
                            (_mtps_by_t.get(t) or {}).get("context_tps_with_cache", 0)
                            / tps_p_div,
                            1,
                        )
                        for t in tsec_vals
                    ]
                ),
            )
            mock_gen_tps = const(
                "mockGenTps",
                num_arr(
                    [
                        round(
                            (_mtps_by_t.get(t) or {}).get("generate_tps", 0)
                            / tps_d_div,
                            1,
                        )
                        for t in tsec_vals
                    ]
                ),
            )
        # client 侧 token 对账序列（per_second.input_tokens /
        # output_tokens / output_tokens_completed）不再构造：IO 对账
        # 面板已移除（20260901 纠偏），对账检测能力降级为 aggregate
        # validity_checks 的 token_reconciliation_ok 断言；
        # per_second 上述字段本身保留（②字段不变，喂 validity 对账）。

    # 阶段延迟（终态分位）：取数键带 _ms 后缀，展示类目去后缀。
    # sched_lat_count：schedule 分位全终态样本量（幸存者口径阶段样本量
    # 对照的基准，见阶段 BarChart caption）。
    STAGE_KEYS = [
        "grpc_queue_ms",
        "route_submit_ms",
        "batch_wait_ms",
        "dispatch_ack_ms",
        "ack_response_ms",
    ]
    STAGE_LABELS = [k[:-3] for k in STAGE_KEYS]
    stage_lat = sm.get("server_stage_latency_ms") or {}
    sched_lat_count = (sm.get("schedule_latency_ms") or {}).get("count")
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
        reg_time(TQ, tq_vals)
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
        reg_time(cats_name, [t for t, _ in pts])
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
    # 全量口径洛伦兹（全部 placement 决策行；旧 aggregate 无此键为空）
    p_ly_all = lorenz.get("prefill_all_y_pct") or []
    d_ly_all = lorenz.get("decode_all_y_pct") or []

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
    p_lorenz_all_y = d_lorenz_all_y = None
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
    if p_ly or d_ly or p_ly_all or d_ly_all:
        LORENZ_X = const("LORENZ_X", str_arr(lorenz_x))
        if p_ly:
            p_lorenz_y = const("pLorenzY", num_arr(p_ly))
        if d_ly:
            d_lorenz_y = const("dLorenzY", num_arr(d_ly))
        if p_ly_all:
            p_lorenz_all_y = const("pLorenzAllY", num_arr(p_ly_all))
        if d_ly_all:
            d_lorenz_all_y = const("dLorenzAllY", num_arr(d_ly_all))
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
        reg_time(TKV, kv_t_vals)
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
            reg_time(TKV_UTIL, kv_t_vals2)
            if not (queue_ts and len(kv_util) == len(queue_ts)):
                warnings.append(
                    "decode_kv.util_pct_series 长度与 queue_timeseries 不一致，x 轴按 5s 采样推定"
                )
        kv_util_data = const("dKvUtil", num_arr(kv_util))

    # ---- 身份行（subtitle 第一层：一眼看清实验条件，20260902 三层规范化）----
    # 拓扑 + 发送模式/倍率 + ramp/duration/shards。倍率取数源 =
    # replay_speed（REPLAY_SPEED 自动校准值，如 82），非 CLI 缺省
    # （replay@1000x bug 修复）。原 identity 中的请求总数移入 KPI 结果
    # 行、采样说明移入可见 meta 面板，subtitle 只保留实验条件。
    sampling_note = "时间序列 1s 采样（QPS / 延迟）"
    if queue_ts:
        sampling_note += "，队列 " + str(q_step) + "s 采样"
    cond_parts = [str(p_engines) + "P + " + str(d_engines) + "D mock"]
    if send_mode == "replay":
        if replay_speed:
            _mode_seg = "replay@" + str(replay_speed) + "x"
            if nominal_qps:
                _mode_seg += "（名义 " + str(nominal_qps) + " QPS）"
            cond_parts.append(_mode_seg)
        elif nominal_qps:
            cond_parts.append("replay（名义 " + str(nominal_qps) + " QPS）")
        else:
            cond_parts.append("replay")
    else:
        cond_parts.append(
            ("uniform " + str(nominal_qps) + " QPS") if nominal_qps else "uniform"
        )
    if ramp_s:
        cond_parts.append("ramp " + str(ramp_s) + "s")
    if duration_s:
        cond_parts.append(str(int(duration_s)) + "s")
    cond_parts.append(str(shards) + " shards")
    identity = " · ".join(cond_parts)

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
            "slo_batch_analysis.json 早于 client_events.jsonl（陈旧残留），SLO/批决策结论不可信"
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
    lines.append("      <Grid columns={5} gap={10}>")
    lines.append(emit_stat(fmt_int_trunc(send_qps), send_qps_label))
    lines.append(emit_stat(fmt_int_trunc(ok_qps), "成功调度 QPS", "success"))
    lines.append(emit_stat(fmt_pct(error_rate), "错误率", "danger"))
    lines.append(emit_stat(gini_stat, gini_label, gini_tone))
    lines.append(emit_stat(pacing_label, "pacing 质量", pacing_tone))
    lines.append("      </Grid>")
    lines.append("")

    # ---- KPI 第二行：结果行（三层规范化第二层，20260902）----
    # 请求数量 / 成功数量 / 失败·cancel / 成功率 / 持续时间，全部自
    # summary 结果类字段（total_req/success_n/error_n 在上方 Stat 段已
    # 取）。error_count 与 error_breakdown 同口径（全量行），err_cancelled
    # 是其具名子桶——失败 chip 主值用总数、cancel 子集数括号注记，
    # 避免两数相加造成口径重叠误导。旧 aggregate 无 summary 值时（rows
    # 缺失 → None）chip 显示 —。
    res_cancel = (sm.get("error_breakdown") or {}).get("err_cancelled") or 0
    if error_n is not None and res_cancel:
        fail_stat = (
            fmt_int_trunc(error_n) + "（cancel " + fmt_int_trunc(res_cancel) + "）"
        )
    elif error_n is not None:
        fail_stat = fmt_int_trunc(error_n)
    else:
        fail_stat = "—"
    res_ok_rate = (
        (float(success_n) / float(total_req))
        if (success_n is not None and total_req)
        else None
    )
    ok_rate_tone = (
        "success"
        if res_ok_rate is not None and res_ok_rate >= 0.99
        else ("warning" if res_ok_rate is not None and res_ok_rate >= 0.9 else "danger")
    )
    lines.append("      <Grid columns={5} gap={10}>")
    lines.append(
        emit_stat(
            fmt_int_trunc(total_req) if total_req is not None else "—", "请求数量"
        )
    )
    lines.append(
        emit_stat(
            fmt_int_trunc(success_n) if success_n is not None else "—",
            "成功数量",
            "success",
        )
    )
    lines.append(
        emit_stat(fail_stat, "失败 / cancel", "danger" if error_n else "success")
    )
    lines.append(
        emit_stat(
            fmt_pct(res_ok_rate) if res_ok_rate is not None else "—",
            "成功率",
            ok_rate_tone,
        )
    )
    lines.append(
        emit_stat((str(int(duration_s)) + "s") if duration_s else "—", "持续时间")
    )
    lines.append("      </Grid>")
    lines.append("")

    # 无节标题：每秒 QPS（发送 / 成功 / 失败）+ 失败按原因
    if per_second:
        qps_max = max(
            max((p.get("arrivals", 0) or 0) for p in per_second),
            max(m_arr_by_t.values()) if m_arr_by_t else 0,
            max((p.get("success", 0) or 0) for p in per_second),
            max((p.get("errors", 0) or 0) for p in per_second),
        )
        # 发送 QPS 双口径 foot：master arrival（主口径）与 client 自估
        # 并列透出——过载 run 下二者差距即客户端时钟窗口失真的直观证据。
        qps_foot = ""
        if send_qps_master:
            qps_foot = (
                "；全程口径：master 到达 " + fmt_int_trunc(send_qps_master) + " QPS"
            )
            if send_qps_client:
                qps_foot += (
                    " · client 自估 " + fmt_int_trunc(send_qps_client) + " QPS（参考）"
                )
        elif send_qps_client:
            qps_foot = (
                "；全程口径：client 自估 " + fmt_int_trunc(send_qps_client) + " QPS"
            )
        # 发送序列换源（20260830 修正）：master 到达口径为主发送线
        # （覆盖全部到达，含冻结尾巴）；客户端 arrivals 降为参考线，
        # 截断 run 下它只覆盖部分窗口（如 0-70s，每秒 ~476 vs master
        # ~2000）。成功/失败维持客户端终态口径（本来就是客户端行）。
        if qps_master_arrivals is not None:
            qps_series = [
                (
                    "arrM",
                    "发送（master 到达口径）",
                    qps_master_arrivals,
                    "info",
                ),
                (
                    "arrC",
                    "发送（客户端口径）",
                    qps_arrivals,
                    "neutral",
                ),
                ("ok", "成功（客户端）", qps_success, "success"),
                ("err", "失败（客户端）", qps_errors, "danger"),
            ]
            qps_caption = (
                "x = 压测时间（s）；y = 每秒请求数。发送 = master 到达口径"
                "（master 计数器差分，覆盖全部到达，尾部冻结即发送停止）；"
                "客户端发送为参考线（收集器截断时只覆盖部分窗口）；"
                "成功/失败 = 客户端终态口径"
            )
        else:
            qps_series = [
                ("arr", "发送（arrivals）", qps_arrivals, "neutral"),
                ("ok", "成功（success）", qps_success, "success"),
                ("err", "失败（errors）", qps_errors, "danger"),
            ]
            qps_caption = "x = 压测时间（s）；y = 每秒请求数"
        qps_chart = emit_container(
            "每秒 QPS：发送 / 成功 / 失败",
            qps_caption + qps_foot,
            emit_chart(
                "LineChart",
                TSEC,
                230,
                qps_series,
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
                    sum((ps_by_t.get(t) or {}).get(mk, 0) or 0 for mk in merged_keys)
                    for t in tsec_vals
                ]
            else:
                cname = "err" + k[4:].title().replace("_", "")
                vals = [(ps_by_t.get(t) or {}).get(k, 0) or 0 for t in tsec_vals]
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
        # 引擎侧 cancel 事件速率（按角色）：cancel_rpcs 是 mock 集群累计计数，
        # 聚合端按 epoch 对齐差分；与客户端 QPS 同节呈现。全零也画：
        # 零 cancel 曲线本身就是「无抢占/取消」的正面证据。
        # 20260830+ 聚合端拆 cancel 角色（新 aggregate 才有这些键）：master 侧
        # = census unknown/finished/tombstone 差分（cancel RPC 到达引擎时
        # 引擎已无该请求活跃条目，即 master 调度层发起的取消：queueTimeout/
        # deadline 到期、decode generation retired 批量取消等）；prefill/
        # decode 侧 = 引擎仍在跟踪该请求时的真实取消（final_snapshot 每引擎
        # cancelled_rids + master 终态行时刻重建）。三条 cancel 线之和≈
        # 原 cancel 总量（rid 时刻缺失的少量事件被丢弃，见 integrity）。
        # 旧 aggregate 无新键 -> 回退旧单 cancel 线。
        if cancel_ts:
            tcxl_t = [r.get("t", 0) for r in cancel_ts]
            tcxl = const("TCXL", str_arr(sparse_cats(tcxl_t)))
            reg_time(tcxl, tcxl_t)
            if any(
                "master_cancel_qps" in r or "prefill_cancel_qps" in r for r in cancel_ts
            ):
                cancel_defs = [
                    ("master_cancel_qps", "cancel（master 侧）", "danger"),
                    ("prefill_cancel_qps", "cancel（prefill 侧）", "warning"),
                    ("decode_cancel_qps", "cancel（decode 侧）", None),
                ]
                cancel_title = "每秒引擎侧 cancel 事件速率（按角色）"
                cancel_caption = (
                    "x = 压测时间（s，stats 采样轴，epoch 对齐）；y = 每秒 cancel 事件数。"
                    "cancel 按角色拆分：master 侧 = cancel 到达引擎时引擎无该请求"
                    "活跃条目（master 调度层取消：queueTimeout/deadline 到期、"
                    "generation retired 批量取消）；prefill / decode 侧 = 引擎仍"
                    "跟踪该请求时的真实取消（cancelled_rids 按 deadline 到期"
                    "时刻重建，无法定位时刻的少量事件不计入）；三条 cancel 线"
                    "之和 ≈ 引擎侧 cancel 总量"
                )
            else:
                cancel_defs = [
                    ("cancel_qps", "cancel（引擎侧）", "danger"),
                ]
                cancel_title = "每秒引擎侧 cancel 事件速率"
                cancel_caption = (
                    "x = 压测时间（s，stats 采样轴，epoch 对齐）；"
                    "y = 每秒 cancel 事件数（累计计数差分归一）"
                )
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
                    cancel_title,
                    cancel_caption,
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
                "x = 压测时间（s，1s 采样）；y = 延迟（ms）"
                "；出生秒分桶（client 发出时刻，全终态含失败行）",
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
    # 反馈 1：schedule p95 + master 链路五阶段 p95 合成一张时序图
    # （master 10s 窗口；schedule p95 重采样为每个窗口内 1s 桶的中值。
    # 20260830 精简：p50/p99 线下线，只留 p95——分位细节已在上方
    # 「schedule 延迟 p50 / p95 / p99」单独面板覆盖）
    if stage_ts:
        stage_t_vals = [r.get("t", 0) for r in stage_ts]
        STAGE_T = const("STAGE_T", str_arr(sparse_cats(stage_t_vals)))
        reg_time(STAGE_T, stage_t_vals)
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
                    "sp95",
                    "schedule p95（10s 窗口中值）",
                    stage_resample("sched_p95", "stageSchedP95"),
                    "info",
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
        # 口径标注：schedule p95 为全终态口径（n=sched_lat_count）；
        # master 链路各阶段 p95 为幸存者口径（仅完成该阶段的行上报，
        # 样本量见分阶段 BarChart caption）。
        # 完成轴错位标注（20260830）：SERVER_LAT 行是 master 侧 10s 完成窗
        # 直方图（ServerScheduleLatencyRecorder 无 per-rid 阶段时刻日志
        # 可 join），与出生轴指标（e2e/full_e2e/exec 出生轴）存在结构性
        # 错位——临界爬升期最明显，同图对比时须计入。
        stage_ts_cap = (
            "x = 压测时间（s，master 10s 完成窗口）；y = 延迟（ms）。"
            "master 阶段分位按 10s 完成窗口径统计（完成轴）——与出生轴"
            "指标（e2e/full_e2e/exec 出生轴）存在结构性错位，临界爬升期"
            "最明显"
        )
        if sched_lat_count:
            stage_ts_cap += "；schedule p95 为全终态口径 n=" + fmt_int_trunc(
                sched_lat_count
            )
        stage_ts_cap += "；master 链路阶段 p95 为幸存者口径（仅计完成该阶段的行）"
        # 轴标注（20260901）：末尾一句话点明每条线的分桶时刻——schedule
        # 线按 client 发出（出生秒），master 链路阶段线按完成（完成秒）。
        stage_ts_cap += (
            "；schedule = 出生秒（client 发出时刻分桶，全终态）"
            "；master 链路阶段 = 完成秒（10s 窗口）"
        )
        latency_containers.append(
            emit_container(
                "调度延迟：schedule p95 + master 链路阶段 p95",
                stage_ts_cap,
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
    # 反馈 2：多延迟合一（full_e2e / ttft / schedule / prefill exec /
    # decode exec，全部 p95；e2e 系列已于 20260831 删除——与
    # full_e2e 口径重叠且易误读，只保留 full_e2e）。
    # full_e2e（20260830）：client 发出 → 引擎 decode 正常终态，按
    # request_id 关联的跨两侧全链路口径——schedule-only（FETCH=0）下
    # 覆盖调度+prefill+传输+decode 完整链路；旧 aggregate 无该字段
    # 时线与标题自动回退为四延迟。
    # 引擎 exec 出生轴（20260830 第二批）：prefill/decode exec 分位优先取
    # per_second 的出生轴键（rid join 引擎终态行、按 send_start 出生秒
    # 分桶，与 full_e2e 同轴可比）；旧聚合/旧引擎 build 无该键时回退
    # engine_exec_ts 完成轴窗口快照，caption 明确标注口径与错位风险
    # （完成轴 vs 出生轴结构性错位，临界爬升期最明显——full_e2e 与
    # decode exec 同图对比时曾两次误导排障方向）。
    if per_second:
        has_ttft = any((p.get("ttft_p95", 0) or 0) for p in per_second)
        has_full_e2e = any((p.get("full_e2e_p95", 0) or 0) for p in per_second)
        has_birth_pe = any((p.get("prefill_exec_p95", 0) or 0) for p in per_second)
        has_birth_de = any((p.get("decode_exec_p95", 0) or 0) for p in per_second)
        if has_ttft or has_full_e2e or has_birth_pe or has_birth_de or engine_exec:
            five_series = []
            if has_full_e2e:
                five_series.append(
                    (
                        "fe",
                        "full_e2e（p95）",
                        const(
                            "fullE2eP95",
                            num_arr(
                                [
                                    (ps_by_t.get(t) or {}).get("full_e2e_p95", 0)
                                    for t in tsec_vals
                                ]
                            ),
                        ),
                        "neutral",
                    )
                )
            if has_ttft:
                five_series.append(
                    (
                        "ttft",
                        "ttft（p95）",
                        const(
                            "ttftP95",
                            num_arr(
                                [
                                    (ps_by_t.get(t) or {}).get("ttft_p95", 0)
                                    for t in tsec_vals
                                ]
                            ),
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
            # 引擎 exec 双口径：出生轴（per_second 新键，rid join 引擎终态
            # 行、按 send_start 出生秒分桶）优先；旧聚合/旧引擎 build 无
            # 该键时回退完成轴（engine_exec_ts 按最近整秒对齐到合并轴）。
            # 两侧独立判断：decode_done 行（bcf3e672 起）与 prefill_done 行
            # （本批新增）可能只存在一侧，混合口径在 caption 分别标注。
            _has_completion_pe = bool(engine_exec) and "prefill_exec_p95_ms" in (
                engine_exec[0] or {}
            )
            if has_birth_de:
                de95 = [
                    (ps_by_t.get(t) or {}).get("decode_exec_p95", 0) or 0
                    for t in tsec_vals
                ]
                five_series.append(
                    (
                        "de",
                        "decode exec（p95·出生轴）",
                        const("decodeExecP95Birth", num_arr(de95)),
                        "success",
                    )
                )
            elif engine_exec:
                used_completion_de = True
                de95 = [
                    (exec_map.get(t) or {}).get("decode_exec_p95_ms", 0) or 0
                    for t in tsec_vals
                ]
                five_series.append(
                    (
                        "de",
                        "decode exec（p95·完成轴）",
                        const("decodeExecP95", num_arr(de95)),
                        "success",
                    )
                )
            if has_birth_pe:
                pe95 = [
                    (ps_by_t.get(t) or {}).get("prefill_exec_p95", 0) or 0
                    for t in tsec_vals
                ]
                five_series.append(
                    (
                        "pe",
                        "prefill exec（p95·出生轴）",
                        const("prefillExecP95Birth", num_arr(pe95)),
                        "danger",
                    )
                )
            elif _has_completion_pe:
                used_completion_pe = True
                pe95 = [
                    (exec_map.get(t) or {}).get("prefill_exec_p95_ms", 0) or 0
                    for t in tsec_vals
                ]
                five_series.append(
                    (
                        "pe",
                        "prefill exec（p95·完成轴）",
                        const("prefillExecP95", num_arr(pe95)),
                        "danger",
                    )
                )
            ttft_max = max((p.get("ttft_p95", 0) or 0) for p in per_second)
            full_e2e_max = max((p.get("full_e2e_p95", 0) or 0) for p in per_second)
            de95_birth_max = max((p.get("decode_exec_p95", 0) or 0) for p in per_second)
            pe95_birth_max = max(
                (p.get("prefill_exec_p95", 0) or 0) for p in per_second
            )
            exec_max = 0
            if not has_birth_de and engine_exec:
                exec_max = max(
                    exec_max,
                    max((r.get("decode_exec_p95_ms", 0) or 0) for r in engine_exec),
                )
            if not has_birth_pe and _has_completion_pe:
                exec_max = max(
                    exec_max,
                    max((r.get("prefill_exec_p95_ms", 0) or 0) for r in engine_exec),
                )
            five_max = max(
                ttft_max,
                full_e2e_max,
                de95_birth_max,
                pe95_birth_max,
                exec_max,
                1,
            )
            five_cap = (
                "x = 压测时间（s，1s 采样）；y = 延迟 p95（ms）。口径："
                "ttft = 成功请求按发送秒的分位（幸存者口径，过载下慢"
                "请求已转为错误被排除）"
            )
            # exec 线口径段：出生轴与完成轴分别标注（可能混合——如旧引擎
            # build 有 decode_done 无 prefill_done 时 decode 出生轴 +
            # prefill 完成轴）。
            _birth_axes = []
            if has_birth_pe:
                _birth_axes.append("prefill")
            if has_birth_de:
                _birth_axes.append("decode")
            if _birth_axes:
                five_cap += (
                    "；" + "/".join(_birth_axes) + " exec（出生轴）= 引擎终态"
                    "行按 request_id 关联回成功请求、按请求出生秒（send_start）"
                    "分桶，与 full_e2e 同轴可比（幸存者口径）"
                )
            if used_completion_pe or used_completion_de:
                _completion_axes = []
                if used_completion_de:
                    _completion_axes.append("decode")
                if used_completion_pe:
                    _completion_axes.append("prefill")
                five_cap += (
                    "；" + "/".join(_completion_axes) + " exec（完成轴）= 完成流"
                    "（含 cancel）按完成秒窗口的分位——与出生轴指标存在结构性"
                    "错位，临界爬升期最明显，勿与 full_e2e 直接对比"
                )
            if has_full_e2e:
                five_cap += (
                    "；full_e2e = client 发出 → decode 执行结束（引擎侧终态"
                    "行按 request_id 关联，schedule-only 下覆盖调度+prefill+"
                    "传输+decode 全链路）"
                )
                if any("full_e2e_n" in p for p in per_second):
                    five_cap += "；full_e2e 每秒样本量见 full_e2e_n"
            if has_birth_pe and any("prefill_exec_n" in p for p in per_second):
                five_cap += "；prefill exec 每秒样本量见 prefill_exec_n"
            if has_birth_de and any("decode_exec_n" in p for p in per_second):
                five_cap += "；decode exec 每秒样本量见 decode_exec_n"
            # 轴标注（20260901）：末尾一句话点明各线分桶时刻——e2e 族按
            # client 发出（出生秒），route_submit 按成功发布（完成秒）。
            five_cap += (
                "；e2e/full_e2e/exec = 出生秒（client 发出时刻分桶）"
                "；route_submit = 完成秒（成功发布时刻，幸存者口径）"
            )
            _five_title = (
                "五延迟：full_e2e / ttft / schedule / prefill exec / decode exec"
                if has_full_e2e
                else "四延迟：ttft / schedule / prefill exec / decode exec"
            )
            latency_containers.append(
                emit_container(
                    _five_title,
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
        # 各阶段样本量透出：route_submit 等 master 侧阶段仅由完成该阶段的
        # 终态行上报（幸存者口径），样本量 < schedule 全终态样本时即存在
        # 未走完链路的行——过载 run 下必须可见，否则 p50/p95 会被误读
        # 为全量分布。
        stage_n_parts = []
        for s in STAGE_KEYS:
            _n = (stage_lat.get(s) or {}).get("count")
            if _n:
                stage_n_parts.append(s[:-3] + " n=" + fmt_int_trunc(_n))
        stage_bar_cap = (
            "x = master 调度链路阶段；y = 阶段延迟（ms，全程终态分位，非时序）"
        )
        if stage_n_parts:
            stage_bar_cap += "；样本量：" + " / ".join(stage_n_parts)
        _rs_n = (stage_lat.get("route_submit_ms") or {}).get("count")
        if _rs_n and sched_lat_count and _rs_n < sched_lat_count:
            stage_bar_cap += (
                "；route_submit 为幸存者口径（n="
                + fmt_int_trunc(_rs_n)
                + " < 全终态调度样本 "
                + fmt_int_trunc(sched_lat_count)
                + "，未走完链路的行不含该阶段计时）"
            )
        # 类目轴 const 化：反抽层 _TSX_ATTR_CATS 只认 const 引用，内联
        # str_arr(STAGE_LABELS) 会让整个面板被丢弃（历史遗留——修复后
        # 本面板才进 HTML，各阶段样本量/幸存者标注随之可见）。
        STAGE_CATS = const("STAGE_CATS", str_arr(STAGE_LABELS))
        latency_containers.append(
            emit_container(
                "master 内部分阶段延迟（p50 / p95，全程终态分位）",
                stage_bar_cap,
                emit_chart(
                    "BarChart",
                    STAGE_CATS,
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
            brt_t = [r.get("t", 0) for r in batcher_ts_by_role]
            BRT = const("BRT", str_arr(sparse_cats(brt_t)))
            reg_time(BRT, brt_t)
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
                    tet_t = [r.get("t", 0) for r in batcher_top_engines_ts]
                    TET = const("TET", str_arr(sparse_cats(tet_t)))
                    reg_time(TET, tet_t)
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
            bt_t = [r.get("t", 0) for r in batcher_ts]
            BT = const("BT", str_arr(sparse_cats(bt_t)))
            reg_time(BT, bt_t)
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
            drt_t = [r.get("t", 0) for r in dispatch_reason_ts]
            DRT = const("DRT", str_arr(sparse_cats(drt_t)))
            reg_time(DRT, drt_t)
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
            bst_t = [r.get("t", 0) for r in dispatch_batch_size_ts]
            BST = const("BST", str_arr(sparse_cats(bst_t)))
            reg_time(BST, bst_t)
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
            # avg（混合）线：总入队请求 ÷ 总批数——全部 reason 批混合的
            # 真实加权平均，非三条 reason 线的算术均值。主口径直接取
            # queue_timeseries.interval_avg_batch_size（相邻采样窗内
            # enqueued 增量 ÷ batches 增量，语义完全一致），前向 step
            # 对齐到 BST 1s 轴（queue 采样窗宽可 >1s，值呈阶梯）；旧
            # aggregate 无该字段时回退：从 dispatch 数据推导，
            # Σ(reason 批数 × reason 批大小) ÷ Σ(reason 批数)，批数取
            # dispatch_reason_ts 前向 step 对齐行。
            bs_avg_vals = None
            if queue_ts and any("interval_avg_batch_size" in q for q in queue_ts):
                bs_avg_vals = ts_step_values(
                    [
                        (
                            q.get("t_offset_s", 0),
                            q.get("interval_avg_batch_size", 0) or 0,
                        )
                        for q in queue_ts
                    ],
                    bst_t,
                )
            elif dispatch_reason_ts:
                dr_rows = [(float(r.get("t", 0) or 0), r) for r in dispatch_reason_ts]
                dr_t_axis = [t for t, _ in dr_rows]
                bs_avg_vals = []
                for r in dispatch_batch_size_ts:
                    i = bisect.bisect_right(dr_t_axis, float(r.get("t", 0) or 0))
                    dr_row = dr_rows[i - 1][1] if i > 0 else dr_rows[0][1]
                    num_sum = den_sum = 0.0
                    for k, cnt in dr_row.items():
                        if k == "t" or not cnt:
                            continue
                        size = r.get(k)
                        if size:
                            num_sum += cnt * size
                            den_sum += cnt
                    bs_avg_vals.append(
                        round(num_sum / den_sum, 2) if den_sum > 0 else 0
                    )
            if bs_avg_vals is not None and any(bs_avg_vals):
                bs_lines.append(
                    (
                        "bs_avg",
                        "avg（混合）",
                        const("bsAvgMix", num_arr(bs_avg_vals)),
                        "primary",
                    )
                )
            if bs_lines:
                bs_caption = (
                    "x = 压测时间（s，1s 采样）；y = dispatch 批大小"
                    "（请求/批，按 reason 的引擎平均）；"
                    "avg = 总请求 ÷ 总批（加权混合，非三 reason 均值；"
                    "queue_timeseries 计数器口径，按采样窗 step 对齐）"
                )
                queue_containers.append(
                    emit_container(
                        "dispatch 批大小（按 reason，引擎平均）",
                        bs_caption,
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
            reg_time(tb_cats, t_grid)
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

    # 2.2 输入/输出 token 长度时序（出生秒分桶）：replay run 的长度
    # 组成随 trace loop 周期变化，与「平均 batch size」/队列时序上下
    # 对照可识别输入侧驱动（长请求簇 -> 攒批慢 -> 小批）。口径：全部
    # 带时间戳请求行（含错误行——形状刻画输入组成）；p50/p95 为
    # nearest-rank 分位；旧 aggregate 无 input_len_n 键时整节省略。
    tok_containers = []
    if input_len_p50 is not None:
        il_max = max((p.get("input_len_p95", 0) or 0) for p in per_second)
        tok_containers.append(
            emit_container(
                "输入 token 长度（每秒 p50 / p95）",
                "x = 压测时间（s，1s 采样）；y = input_len（token）；"
                "按出生秒（send_start）分桶，含全部带时间戳请求行",
                emit_chart(
                    "LineChart",
                    TSEC,
                    230,
                    [
                        ("ilp50", "input p50", input_len_p50, "info"),
                        ("ilp95", "input p95", input_len_p95, "warning"),
                    ],
                    suffix=" tok",
                    domain="[0, " + num(nice_max(il_max * 1.1)) + "]",
                ),
            )
        )
    if output_len_p50 is not None:
        ol_max = max((p.get("output_len_p95", 0) or 0) for p in per_second)
        tok_containers.append(
            emit_container(
                "输出 token 长度（每秒 p50 / p95）",
                "x = 压测时间（s，1s 采样）；y = output_len（token）；"
                "按出生秒（send_start）分桶，含全部带时间戳请求行",
                emit_chart(
                    "LineChart",
                    TSEC,
                    230,
                    [
                        ("olp50", "output p50", output_len_p50, "success"),
                        ("olp95", "output p95", output_len_p95, "danger"),
                    ],
                    suffix=" tok",
                    domain="[0, " + num(nice_max(ol_max * 1.1)) + "]",
                ),
            )
        )
    if tok_containers:
        lines.append("      <Divider />")
        lines.append("")
        lines.append("      <H2>2.2 输入 / 输出 token 长度（出生秒分桶）</H2>")
        lines.extend(emit_grid(tok_containers))
        lines.append("")

    # 2.3 TPS P/D 角色主图（20260901，同日纠偏）：mock 自报生产口径
    # TPS（rtp_llm_*，完成事件记账，1s scrape 窗口）按 P/D 角色切分
    # 展示——与生产大盘同构读法（引擎自报、hippo_role tag 切分，无
    # client 侧 TPS 概念）：P 角色（prefill 引擎聚合）= context
    # with/without cache 双曲线，差值 = cache 复用等效吞吐（KV 容量
    # 对齐任务的收益面）；D 角色（decode 引擎聚合）= generate 单曲线。
    # mock_tps_ts 为集群级时序但语义天然按角色切分（context_* 只来自
    # P 引擎、generate 只来自 D 引擎），无需 role 维度数据改造。
    # 20260901 呈现口径改版：主图画每引擎平均（集群和 ÷ 引擎数，
    # tps_p_engines/tps_d_engines 可靠链解析），与生产大盘单实例
    # series 的读法同构——此前集群和呈现下 12P 求和 p50 3.88M vs
    # 生产单实例 ~58k，观感差 67 倍。引擎数未知时回退集群和呈现，
    # caption 明示「集群和（引擎数未知）」+ stderr 告警（标准 run
    # 引擎数恒可得，回退只是防御）。
    # 口径提醒：mock TPS 是记账式模拟读数（分母固定 1s 窗口），
    # 衡量调度组织效率而非 GPU 算力，不可与生产数值直接对表（口径
    # 语义一一对应）。
    tps_containers = []
    if mock_ctx_tps is not None:
        if tps_p_engines:
            _p_y_scope = (
                "y = context token/s，每引擎平均（集群和 ÷ P 引擎数 "
                + str(tps_p_engines)
                + "，生产大盘单实例 series 同构读法）；集群口径 = 生产同名指标 "
                "rtp_llm_context_tps* 跨 P 引擎求和"
            )
            _p_cache_name = "with cache（Σil ÷ N）"
            _p_compute_name = "compute（(Σil−hit) ÷ N）"
        else:
            sys.stderr.write(
                TAG + " warning: 2.3 P 角色 TPS 主图 P 引擎数不可得（run_meta "
                "params / engine_dist / mock final_snapshot 均未给出），"
                "回退集群和呈现\n"
            )
            _p_y_scope = (
                "y = context token/s 集群和（引擎数未知）；P 引擎数在 "
                "run_meta params / engine_dist / mock final_snapshot 均缺失，"
                "回退集群和呈现，与生产大盘单实例 series 不可直接对表；"
                "集群口径 = 生产同名指标 rtp_llm_context_tps* 跨 P 引擎求和"
            )
            _p_cache_name = "with cache（Σil）"
            _p_compute_name = "compute（Σil−hit）"
        _ctx_cap = (
            max((p.get("context_tps_with_cache", 0) or 0) for p in mock_tps_rows)
            / tps_p_div
        )
        tps_containers.append(
            emit_container(
                "P 角色 context TPS：with cache vs compute（cache 复用等效吞吐）",
                "P 角色（prefill，生产大盘同款 hippo_role 切分读法）；"
                "x = 压测时间（s，1s 窗口）；"
                + _p_y_scope
                + "，完成事件记账：compute = "
                "Σ(il−hit)，with cache = Σil；两线差值 = cache 复用等效吞吐"
                + (
                    "；累计复用 cache_saved_tokens = "
                    + fmt_int_trunc(sm.get("cache_saved_tokens"))
                    + " tokens（final_snapshot 累计口径）"
                    if sm.get("cache_saved_tokens") is not None
                    else ""
                ),
                emit_chart(
                    "LineChart",
                    TSEC,
                    230,
                    [
                        (
                            "mcc",
                            _p_cache_name,
                            mock_ctx_cache_tps,
                            "success",
                        ),
                        ("mct", _p_compute_name, mock_ctx_tps, "info"),
                    ],
                    suffix=" tok/s",
                    domain="[0, " + num(nice_max(_ctx_cap * 1.15)) + "]",
                ),
            )
        )
    if mock_gen_tps is not None:
        if tps_d_engines:
            _d_y_scope = (
                "y = generate token/s，每引擎平均（集群和 ÷ D 引擎数 "
                + str(tps_d_engines)
                + "，生产大盘单实例 series 同构读法）；集群口径 = 生产同名指标 "
                "rtp_llm_generate_tps 跨 D 引擎求和"
            )
            _d_series_name = "generate（Σol ÷ N）"
        else:
            sys.stderr.write(
                TAG + " warning: 2.3 D 角色 TPS 主图 D 引擎数不可得（run_meta "
                "params / engine_dist / mock final_snapshot 均未给出），"
                "回退集群和呈现\n"
            )
            _d_y_scope = (
                "y = generate token/s 集群和（引擎数未知）；D 引擎数在 "
                "run_meta params / engine_dist / mock final_snapshot 均缺失，"
                "回退集群和呈现，与生产大盘单实例 series 不可直接对表；"
                "集群口径 = 生产同名指标 rtp_llm_generate_tps 跨 D 引擎求和"
            )
            _d_series_name = "generate（Σol）"
        _gen_cap = (
            max((p.get("generate_tps", 0) or 0) for p in mock_tps_rows) / tps_d_div
        )
        tps_containers.append(
            emit_container(
                "D 角色 generate TPS（rtp_llm_generate_tps）",
                "D 角色（decode，生产大盘同款 hippo_role 切分读法）；"
                "x = 压测时间（s，1s 窗口）；" + _d_y_scope + "，完成事件记账：Σol",
                emit_chart(
                    "LineChart",
                    TSEC,
                    230,
                    [
                        ("mgt", _d_series_name, mock_gen_tps, "success"),
                    ],
                    suffix=" tok/s",
                    domain="[0, " + num(nice_max(_gen_cap * 1.15)) + "]",
                ),
            )
        )
    if tps_containers:
        lines.append("      <Divider />")
        lines.append("")
        lines.append("      <H2>2.3 TPS（生产同构，P/D 角色切分）</H2>")
        lines.extend(emit_grid(tps_containers))
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
        if p_ly or d_ly or p_lorenz_all_y or d_lorenz_all_y:
            lz_series = []
            if p_ly:
                lz_series.append(("p", "prefill 洛伦兹（成功行）", p_lorenz_y, "info"))
            if d_ly:
                lz_series.append(
                    ("d", "decode 洛伦兹（成功行）", d_lorenz_y, "success")
                )
            if p_lorenz_all_y:
                lz_series.append(
                    ("pa", "prefill 洛伦兹（全量）", p_lorenz_all_y, "neutral")
                )
            if d_lorenz_all_y:
                lz_series.append(
                    ("da", "decode 洛伦兹（全量）", d_lorenz_all_y, "warning")
                )
            lz_cap = "x = 引擎累计占比 %（从最轻到最重）；y = 请求数累计占比 %"
            if p_lorenz_all_y or d_lorenz_all_y:
                lz_cap += (
                    "；成功行 = is_ok 行（旧口径），全量 = 全部 placement "
                    "决策行（含失败/超时，高错误率 run 下二者差距即幸存者偏差）"
                )
            dist_containers.append(
                emit_container(
                    "洛伦兹曲线：请求数（P / D）",
                    lz_cap,
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
            ift_t = [r.get("t", 0) for r in inflight_ts]
            IFT = const("IFT", str_arr(sparse_cats(ift_t)))
            reg_time(IFT, ift_t)
            inf_sched = const(
                "infSched",
                num_arr([r.get("scheduler", 0) or 0 for r in inflight_ts]),
            )
            inf_pb = const(
                "infPB",
                num_arr([r.get("prefill_batches", 0) or 0 for r in inflight_ts]),
            )
            inf_pr = const(
                "infPR",
                num_arr([r.get("prefill_requests", 0) or 0 for r in inflight_ts]),
            )
            inf_dres = const(
                "infDRes",
                num_arr([r.get("decode_reserved", 0) or 0 for r in inflight_ts]),
            )
            inf_drun = const(
                "infDRun",
                num_arr(
                    [r.get("decode_confirmed_running", 0) or 0 for r in inflight_ts]
                ),
            )
            # 拆双面板（20260831）：scheduler/decode 侧量级 0~5500，prefill
            # 请求/批 0~600，单 y 轴会把 prefill 批（~24）压成直线。
            # 面板 A：scheduler（master 账本请求数）+ decode reserved（master
            # 预约未确认）+ decode confirmed running（引擎确认运行）。
            inf_containers.append(
                emit_container(
                    "In-flight：scheduler / decode",
                    "x = 压测时间（s，快照采样）；y = in-flight 请求数（集群总量）。"
                    "scheduler = master 账本请求数；decode reserved = master 预约"
                    "未确认（G4 reserved_total，含尚未被引擎确认的预约）；"
                    "decode confirmed running = 引擎确认运行中（confirmed_running）",
                    emit_chart(
                        "LineChart",
                        IFT,
                        230,
                        [
                            ("sch", "scheduler in-flight", inf_sched, "warning"),
                            (
                                "dres",
                                "decode reserved（master 预约）",
                                inf_dres,
                                "success",
                            ),
                            (
                                "drun",
                                "decode confirmed running（引擎确认）",
                                inf_drun,
                                "info",
                            ),
                        ],
                    ),
                )
            )
            # 面板 B：prefill 请求（master 账本请求数）与 prefill 批
            # （引擎侧 in-flight 批）同轴，量级 0~600 两条均可见。
            inf_containers.append(
                emit_container(
                    "Prefill in-flight：请求 / 批",
                    "x = 压测时间（s，快照采样）；y = in-flight 数（集群总量）。"
                    "请求数 = prefill_endpoints[].inflight_requests（master "
                    "账本口径）；批数 = inflight_batches（引擎侧 in-flight 批）",
                    emit_chart(
                        "LineChart",
                        IFT,
                        230,
                        [
                            ("pr", "prefill in-flight 请求数", inf_pr, "info"),
                            ("pb", "prefill in-flight 批数", inf_pb, "warning"),
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
                reg_time(IAT, t_grid)
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
                    "scheduler ledger vs prefill / decode per-worker ledger）。"
                    "三口径互不可比：prefill = 距上次引擎观察的 staleness"
                    "（20ms 轮询地板，非批寿命）；decode = 预约→引擎确认窗口"
                    "（非 decode 执行时长）；scheduler = 接入起全程寿命"
                )
            else:
                iat_t = [r.get("t", 0) for r in inflight_age]
                IAT = const("IAT", str_arr(sparse_cats(iat_t)))
                reg_time(IAT, iat_t)
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
    # 上方 master 侧面板 + 下方 5b 引擎侧块池面板（两种视角同节并存，
    # caption 各自注明口径）。
    kv_containers = []
    if kv_ts:
        kvt_t = [r.get("t", 0) for r in kv_ts]
        KVT = const("KVT", str_arr(sparse_cats(kvt_t)))
        reg_time(KVT, kvt_t)
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
    # 5b. 引擎侧 KV v2 块池面板（20260902）：aggregate kv_blocks_ts_by_role
    # （引擎 /metrics 三态 gauge + 准入/复用/淘汰累计 counter，P/D 桶跨
    # 引擎求和的集群级行）→ 每引擎平均呈现（÷ tps_p_div / tps_d_div，
    # 与 2.3 TPS 同一三级回退链：引擎数未知时除数 1 = 集群和回退）。
    # 与上方 master 侧 kv_ts 面板分工：master 侧 = 调度器聚合视角的
    # KV tokens 占用，本组 = 引擎侧视角的块池三态分解 / 准入 / 复用
    # （KV v2 容量模型直接读数）。counter 列在呈现层做相邻有效桶累计
    # 差分 ÷ 桶间隔（归一 blocks/s；计数器回退钳 0——注入 clear()
    # 归零等场景不产生负速率）。旧 aggregate 无键 -> 空表 -> 本组
    # 面板整组省略（与上方 master 侧面板互不影响）。
    kv_pool_containers = []
    kv_pool_gauge_roles = []
    kv_pool_rate_present = False
    kv_pool_adm_present = False
    _KVP_GAUGE_DEFS = (
        ("available_blocks", "available（free+LRU 可淘汰）", "success", "Avail"),
        ("held_blocks", "held（运行中裸块）", "warning", "Held"),
        ("referenced_blocks", "referenced（被引用不可淘汰）", "danger", "Ref"),
    )
    _KVP_COUNTER_COLS = (
        "cache_evictions",
        "kv_admission_fails",
        "lack_mem_rejects",
        "decode_reuse_blocks",
    )
    # 桶化：int(t) 桶内均值（aggregate 行已跨引擎求和，同秒多行时
    # 防御性取均值）——与 mock TPS 桶化同规则。
    _kvp_mean = {}
    for _role, _rows in kv_blocks_by_role.items():
        _buckets = {}
        for r in _rows or []:
            try:
                _t = float(r.get("t", 0) or 0)
            except (TypeError, ValueError):
                continue
            if _t < 0:
                continue
            for _col in (
                ("total_blocks",)
                + tuple(c for c, _, _, _ in _KVP_GAUGE_DEFS)
                + _KVP_COUNTER_COLS
            ):
                _v = r.get(_col)
                if _v is None:
                    continue
                _buckets.setdefault(int(_t), {}).setdefault(_col, []).append(float(_v))
        _kvp_mean[_role] = {
            _b: {c: sum(vs) / len(vs) for c, vs in _cols.items()}
            for _b, _cols in _buckets.items()
        }
    # counter 差分速率：相邻有效桶 (v2−v1)/(t2−t1)，计数器回退钳 0
    _kvp_rate = {}
    for _role, _mean in _kvp_mean.items():
        for _col in _KVP_COUNTER_COLS:
            _prev = None
            for _b in sorted(_mean):
                _v = _mean[_b].get(_col)
                if _v is None:
                    continue
                if _prev is not None and _b > _prev[0] and _v > _prev[1]:
                    _kvp_rate.setdefault(_role, {}).setdefault(_col, {})[_b] = (
                        _v - _prev[1]
                    ) / (_b - _prev[0])
                _prev = (_b, _v)

    def _kvp_has_col(role, col):
        return any(
            (cols or {}).get(col) is not None
            for cols in _kvp_mean.get(role, {}).values()
        )

    # 三态分解面板（P/D 各一）：三态 gauge 每引擎平均三线 + 池大小
    # 参考线；caption 讲三态语义（fail-closed 断言 kv_pool_gauge_roles）。
    for _role, _engines, _div, _tag, _axis_name, _val_prefix in (
        ("prefill", tps_p_engines, tps_p_div, "P", "kvPoolPT", "kvPoolP"),
        ("decode", tps_d_engines, tps_d_div, "D", "kvPoolDT", "kvPoolD"),
    ):
        _mean = _kvp_mean.get(_role)
        if not _mean:
            continue
        _grid = sorted(_mean)
        _AXIS = const(_axis_name, str_arr(sparse_cats(_grid)))
        reg_time(_AXIS, _grid)
        if _engines:
            _y_scope = (
                "y = 块数，每引擎平均（集群和 ÷ "
                + _tag
                + " 引擎数 "
                + str(_engines)
                + "）"
            )
            _per = "（÷N）"
        else:
            sys.stderr.write(
                TAG + " warning: 5b " + _tag + " 角色块池面板引擎数不可得"
                "（run_meta params / engine_dist / mock final_snapshot 均未给出），"
                "回退集群和呈现\n"
            )
            _y_scope = "y = 块数 集群和（引擎数未知）"
            _per = "（集群和）"
        _lines = []
        for _col, _label, _color, _short in _KVP_GAUGE_DEFS:
            _lines.append(
                (
                    _short.lower(),
                    _label + _per,
                    const(
                        _val_prefix + _short,
                        num_arr(
                            [round((_mean[t].get(_col) or 0) / _div, 1) for t in _grid]
                        ),
                    ),
                    _color,
                )
            )
        _tot_vals = [
            round((_mean[t].get("total_blocks") or 0) / _div, 1) for t in _grid
        ]
        _lines.append(
            (
                "tot",
                "池大小 total" + _per,
                const(_val_prefix + "Tot", num_arr(_tot_vals)),
                "info",
            )
        )
        kv_pool_containers.append(
            emit_container(
                _tag + " 角色块池三态分解（引擎侧）",
                _tag
                + " 角色（"
                + _role
                + "）；x = 压测时间（s，1s 采样）；"
                + _y_scope
                + "；三态块语义：available = free + 纯 LRU（ref=0 可淘汰，计入可用），"
                "held = 运行中裸块（prefill 执行期租约 / decode 净新分配），"
                "referenced = 被在途引用块（decode 命中 pin，不可淘汰不计可用），"
                "恒等式 available = 池大小 − held − referenced，完成移交 LRU 后"
                "恢复可用（释放 ≠ 删除）；引擎侧口径（引擎 /metrics 自报），"
                "与上方 master 侧 KV tokens 面板（调度器聚合视角）口径不同",
                emit_chart(
                    "LineChart",
                    _AXIS,
                    230,
                    _lines,
                    domain="[0, " + num(nice_max(max(_tot_vals) * 1.15)) + "]",
                ),
            )
        )
        kv_pool_gauge_roles.append(_role)
    # 速率面板（P/D 共用并集时间轴）：counter 差分每引擎平均。
    if _kvp_mean:
        _kvp_union = sorted(set().union(*(set(m) for m in _kvp_mean.values())))

        def _kvp_rate_vals(role, col, div):
            _rm = (_kvp_rate.get(role) or {}).get(col) or {}
            return [round(_rm.get(t, 0) / div, 2) for t in _kvp_union]

        _RAXIS = const("kvPoolRT", str_arr(sparse_cats(_kvp_union)))
        reg_time(_RAXIS, _kvp_union)
        # 准入失败面板：prefill 同步 602 拒绝与 decode 降级分线记账
        # （正常健康档全零——过载档才非零）。
        _adm_lines = []
        _adm_scopes = []
        _adm_max = 0.0
        if _kvp_has_col("prefill", "lack_mem_rejects"):
            _vals = _kvp_rate_vals("prefill", "lack_mem_rejects", tps_p_div)
            _adm_max = max(_adm_max, max(_vals) if _vals else 0)
            _adm_lines.append(
                (
                    "pRej",
                    "P·LACK_MEM 602 同步拒绝"
                    + ("（÷N）" if tps_p_engines else "（集群和）"),
                    const("kvPoolRejP", num_arr(_vals)),
                    "danger",
                )
            )
            _adm_scopes.append(
                "P 线每引擎平均（÷ P 引擎数 " + str(tps_p_engines) + "）"
                if tps_p_engines
                else "P 线集群和（引擎数未知）"
            )
        if _kvp_has_col("decode", "kv_admission_fails"):
            _vals = _kvp_rate_vals("decode", "kv_admission_fails", tps_d_div)
            _adm_max = max(_adm_max, max(_vals) if _vals else 0)
            _adm_lines.append(
                (
                    "dDeg",
                    "D·decode 降级（un-pooled）"
                    + ("（÷N）" if tps_d_engines else "（集群和）"),
                    const("kvPoolDegD", num_arr(_vals)),
                    "warning",
                )
            )
            _adm_scopes.append(
                "D 线每引擎平均（÷ D 引擎数 " + str(tps_d_engines) + "）"
                if tps_d_engines
                else "D 线集群和（引擎数未知）"
            )
        if _adm_lines:
            kv_pool_containers.append(
                emit_container(
                    "KV 准入失败速率（引擎侧）",
                    "x = 压测时间（s，1s 采样）；y = 次/s，相邻有效桶累计差分 ÷ 桶间隔；"
                    + "；".join(_adm_scopes)
                    + "；prefill 同步拒绝（enqueue 602 LACK_MEM，请求直接失败）与"
                    " decode 降级（un-pooled 继续跑 + kv_admission_fails 计数）分线"
                    "记账互不混线；正常健康档全零——非零即 KV 池过载信号",
                    emit_chart(
                        "LineChart",
                        _RAXIS,
                        230,
                        _adm_lines,
                        domain="[0, " + num(nice_max(_adm_max * 1.15)) + "]",
                    ),
                )
            )
            kv_pool_rate_present = True
            kv_pool_adm_present = True
        # LRU evictions 速率面板（P/D 两线）
        _ev_lines = []
        _ev_scopes = []
        _ev_max = 0.0
        for _role, _engines, _div, _tag in (
            ("prefill", tps_p_engines, tps_p_div, "P"),
            ("decode", tps_d_engines, tps_d_div, "D"),
        ):
            if not _kvp_has_col(_role, "cache_evictions"):
                continue
            _vals = _kvp_rate_vals(_role, "cache_evictions", _div)
            _ev_max = max(_ev_max, max(_vals) if _vals else 0)
            _ev_lines.append(
                (
                    "ev" + _tag,
                    _tag + "·LRU evictions" + ("（÷N）" if _engines else "（集群和）"),
                    const("kvPoolEv" + _tag, num_arr(_vals)),
                    "info" if _tag == "P" else "warning",
                )
            )
            _ev_scopes.append(
                _tag + " 线每引擎平均（÷ " + _tag + " 引擎数 " + str(_engines) + "）"
                if _engines
                else _tag + " 线集群和（引擎数未知）"
            )
        if _ev_lines:
            kv_pool_containers.append(
                emit_container(
                    "LRU evictions 速率（引擎侧）",
                    "x = 压测时间（s，1s 采样）；y = 块/s，相邻有效桶累计差分 ÷ 桶间隔；"
                    + "；".join(_ev_scopes)
                    + "；LRU 淘汰与分配耦合：池余量不足时先淘汰纯 LRU 块再分配"
                    "（mock_engine_cache_evictions_total 累计 counter 差分，"
                    "引擎 /metrics 自报）",
                    emit_chart(
                        "LineChart",
                        _RAXIS,
                        230,
                        _ev_lines,
                        domain="[0, " + num(nice_max(_ev_max * 1.15)) + "]",
                    ),
                )
            )
            kv_pool_rate_present = True
        # decode 复用块速率面板（D 线）：fix #5 净需求折减的直接读数
        if _kvp_has_col("decode", "decode_reuse_blocks"):
            _vals = _kvp_rate_vals("decode", "decode_reuse_blocks", tps_d_div)
            _reuse_max = max(_vals) if _vals else 0
            _reuse_scope = (
                "y = 块/s，每引擎平均（集群和 ÷ D 引擎数 " + str(tps_d_engines) + "）"
                if tps_d_engines
                else "y = 块/s 集群和（引擎数未知）"
            )
            kv_pool_containers.append(
                emit_container(
                    "decode 复用块速率（引擎侧）",
                    "D 角色（decode）；x = 压测时间（s，1s 采样）；"
                    + _reuse_scope
                    + "，相邻有效桶累计差分 ÷ 桶间隔；KV v2 准入复用折减（fix #5）："
                    "decode 接手用自身 LRU 重算命中，净需求 = total − 命中，命中块"
                    " pin 为 referenced 不重分配——本线即命中块速率，读数上行 ="
                    " 「decode 越用省越多」正反馈"
                    "（mock_engine_decode_reuse_blocks_total 累计 counter 差分，"
                    "never drained）",
                    emit_chart(
                        "LineChart",
                        _RAXIS,
                        230,
                        [
                            (
                                "reuse",
                                "decode 命中块"
                                + ("（÷N）" if tps_d_engines else "（集群和）"),
                                const("kvPoolReuseD", num_arr(_vals)),
                                "success",
                            )
                        ],
                        domain="[0, " + num(nice_max(_reuse_max * 1.15)) + "]",
                    ),
                )
            )
            kv_pool_rate_present = True
    # 5c. cache 命中率三口径（20260902）：aggregate cache_hit_ts
    # （master_routing/engine_key/engine_token 窗口命中率列，各口径
    # 独立缺省）+ summary.cache_hit_summary（run 级三口径）。两图均
    # 以 engine_token（实际复用）为参照系：
    #   * 「master 路由 vs engine 执行」——master_routing − engine_token
    #     = 调度损耗（master 匹配到却未复用上：路由到非持有引擎/
    #     affinity 未采纳/路由到执行窗口内 LRU 淘汰）；
    #   * 「key 级（理论）vs token 级（实际）」——engine_key −
    #     engine_token = 命中深度覆盖（部分前缀命中：命中 key 但前缀
    #     在第 N 块断掉，复用 tokens 不足整 key 数）。
    # 两口径齐备才画对应图（单列孤悬不画）；KPI 读数行按口径独立
    # 缺省。旧 aggregate 无键 -> 空表 -> 本节整体省略。窗口比率桶化
    # 取均值后前向填充（跨拍差分窗落在末端桶，中间空桶沿用前值
    # ——命中率是水平量不是计数率，缺桶 ≠ 0；首部无前值桶从 0 爬升，
    # 与 QPS 图同观感）。
    cache_hit_containers = []
    cache_hit_route_present = False
    cache_hit_depth_present = False
    cache_hit_kpi_present = False
    # 三口径 run 级读数柱状图（「KPI/读数行」的 canvas 呈现形态：反抽
    # KPI 通道 kpis[:5] 属头部紧凑行，三口径读数以 BarChart 呈现进
    # HTML——categories 即口径名（含语义标注串），caption 注明与生产
    # 对齐关系与差值读法；各口径独立缺省（缺的口径不画柱）。
    _ch_bar_cats = []
    _ch_bar_vals = []
    if cache_hit_sm.get("master_routing_hit_pct") is not None:
        _ch_bar_cats.append("master 路由口径")
        _ch_bar_vals.append(cache_hit_sm["master_routing_hit_pct"])
    if cache_hit_sm.get("engine_key_hit_pct") is not None:
        _ch_bar_cats.append("key 级理论口径")
        _ch_bar_vals.append(cache_hit_sm["engine_key_hit_pct"])
    if cache_hit_sm.get("engine_token_hit_pct") is not None:
        _ch_bar_cats.append("token 级实际口径")
        _ch_bar_vals.append(cache_hit_sm["engine_token_hit_pct"])
    if _ch_bar_vals:
        cache_hit_containers.append(
            emit_container(
                "cache 命中率三口径：run 级汇总",
                "三口径 run 级命中率（各口径独立缺省）：master 路由口径"
                "（master 选引擎时 GlobalCacheIndex 前缀匹配，master 以为能"
                "复用；对齐生产 whale-lb app.cache 族 routing_selected_match "
                "counter 对末拍比）/ key 级理论口径（命中 key 数/请求 key "
                "数，prefill 准入 prefixHitBlocks 记账；对齐生产 "
                "recent_cache_key_hit）/ token 级实际口径（ΣhitTokens/"
                "Σil，完成请求口径；对齐生产 reuse/input）；柱差读法："
                "master 路由 − token 级实际 = 调度损耗，key 级理论 − "
                "token 级实际 = 命中深度覆盖（部分前缀命中：命中 key 但"
                "前缀在第 N 块断掉）",
                emit_chart(
                    "BarChart",
                    const("cacheHitKpiCats", str_arr(_ch_bar_cats)),
                    230,
                    [
                        (
                            "hit",
                            "run 级命中率",
                            const("cacheHitKpiVals", num_arr(_ch_bar_vals)),
                            "info",
                        )
                    ],
                    suffix="%",
                    domain="[0, 100]",
                ),
            )
        )
        cache_hit_kpi_present = True
    _ch_buckets = {}
    for r in cache_hit_rows:
        try:
            _t = float(r.get("t", 0) or 0)
        except (TypeError, ValueError):
            continue
        if _t < 0:
            continue
        for _col in ("master_routing", "engine_key", "engine_token"):
            _v = r.get(_col)
            if _v is None:
                continue
            _ch_buckets.setdefault(int(_t), {}).setdefault(_col, []).append(float(_v))
    _ch_mean = {
        _b: {c: sum(vs) / len(vs) for c, vs in _cols.items()}
        for _b, _cols in _ch_buckets.items()
    }
    if _ch_mean:
        _ch_grid = sorted(_ch_mean)

        def _ch_has(col):
            return any(
                (_cols or {}).get(col) is not None for _cols in _ch_mean.values()
            )

        def _ch_series(col):
            vals = []
            prev = 0.0
            for _t in _ch_grid:
                _v = (_ch_mean[_t] or {}).get(col)
                if _v is None:
                    vals.append(round(prev * 100.0, 1))
                else:
                    prev = _v
                    vals.append(round(_v * 100.0, 1))
            return vals

        _CH_AXIS = const("cacheHitT", str_arr(sparse_cats(_ch_grid)))
        reg_time(_CH_AXIS, _ch_grid)
        # engine_token 数据 const 只建一次，两图共用（同名 const 重复
        # 定义会产出非法 JS）。
        _ch_tok_const = None
        if _ch_has("engine_token"):
            _ch_tok_const = const("cacheHitTok", num_arr(_ch_series("engine_token")))
        # 图 1：master 路由 vs engine 执行（差值 = 调度损耗）
        if _ch_has("master_routing") and _ch_tok_const is not None:
            cache_hit_containers.append(
                emit_container(
                    "cache 命中率：master 路由 vs engine 执行（差值 = 调度损耗）",
                    "master 路由口径（master 选引擎时 GlobalCacheIndex 前缀"
                    "匹配，master 以为能复用多少 tokens；对齐生产 whale-lb "
                    "app.cache 族 routing_selected_match counter 对差分）vs "
                    "engine 执行口径（engine token 级实际复用，ΣhitTokens/"
                    "Σil；对齐生产 reuse/input）；x = 压测时间（s，1s 窗口"
                    "差分）；y = 命中率 %；双曲线差值 = 调度损耗（master "
                    "匹配到却未复用上：路由到非持有引擎/affinity 未采纳/"
                    "路由到执行窗口内 LRU 淘汰）",
                    emit_chart(
                        "LineChart",
                        _CH_AXIS,
                        230,
                        [
                            (
                                "mRoute",
                                "master 路由口径（以为能复用）",
                                const(
                                    "cacheHitRoute",
                                    num_arr(_ch_series("master_routing")),
                                ),
                                "warning",
                            ),
                            (
                                "eTok",
                                "engine 执行口径（实际复用）",
                                _ch_tok_const,
                                "success",
                            ),
                        ],
                        suffix="%",
                        domain="[0, 100]",
                    ),
                )
            )
            cache_hit_route_present = True
        # 图 2：key 级（理论）vs token 级（实际）（差值 = 命中深度覆盖）
        if _ch_has("engine_key") and _ch_tok_const is not None:
            cache_hit_containers.append(
                emit_container(
                    "engine 命中率：key 级（理论）vs token 级（实际）"
                    "（差值 = 命中深度覆盖）",
                    "key 级理论口径（命中 key 数/请求 key 数，prefill 准入"
                    " prefixHitBlocks 记账；对齐生产 recent_cache_key_hit_"
                    "count/total_count）vs token 级实际口径（ΣhitTokens/"
                    "Σil；对齐生产 reuse/input）；x = 压测时间（s，1s 窗口"
                    "差分）；y = 命中率 %；双曲线差值 = 命中深度覆盖（部分"
                    "前缀命中：命中 key 但前缀在第 N 块断掉，复用 tokens "
                    "不足整 key 数）——key 级 ≥ token 级为常态，差值越大"
                    "说明命中越浅",
                    emit_chart(
                        "LineChart",
                        _CH_AXIS,
                        230,
                        [
                            (
                                "eKey",
                                "key 级·理论（命中 key 数/请求 key 数）",
                                const("cacheHitKey", num_arr(_ch_series("engine_key"))),
                                "info",
                            ),
                            (
                                "eTok",
                                "token 级·实际（ΣhitTokens/Σil）",
                                _ch_tok_const,
                                "success",
                            ),
                        ],
                        suffix="%",
                        domain="[0, 100]",
                    ),
                )
            )
            cache_hit_depth_present = True
    if kv_containers or kv_pool_containers or cache_hit_containers:
        lines.append("      <Divider />")
        lines.append("")
        lines.append("      <H2>5. KV</H2>")
        if kv_containers or kv_pool_containers or cache_hit_containers:
            lines.extend(
                emit_grid(kv_containers + kv_pool_containers + cache_hit_containers)
            )
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
        pt_t = [r.get("t", 0) for r in process_ts]
        PT = const("PT", str_arr(sparse_cats(pt_t)))
        reg_time(PT, pt_t)
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

    # 汇总表（两列；tsx 层保留，反抽不抽 Table 不进 HTML）
    lat_summary = sm.get("schedule_latency_ms") or {}
    rows = []
    if send_qps_master:
        thr_cell = "发送 " + fmt_int_trunc(send_qps_master) + "（master）"
        if send_qps_client:
            thr_cell += " · client 自估 " + fmt_int_trunc(send_qps_client)
    else:
        thr_cell = "发送 " + fmt_int_trunc(send_qps_client) + "（client 自估）"
    thr_cell += " / 成功 " + fmt_int_trunc(ok_qps) + " QPS"
    rows.append(["吞吐", thr_cell])
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
        _lat_cell = (
            "p50 "
            + fmt_ms(lat_summary.get("p50"))
            + " / p99 "
            + fmt_ms(lat_summary.get("p99"))
            + " ms"
        )
        if lat_summary.get("count"):
            _lat_cell += (
                " · n=" + fmt_int_trunc(lat_summary.get("count")) + "（全终态）"
            )
        # Phase A：schedule 双源口径标注（server=master 侧 / client=client
        # 行；旧 aggregate 无 schedule_latency_source 键时省略）。
        _sched_src = sm.get("schedule_latency_source")
        if _sched_src:
            _lat_cell += " · 口径 " + str(_sched_src)
        rows.append(["调度延迟", _lat_cell])
    else:
        rows.append(["调度延迟", "—"])
    # ttft/e2e 全程分位（聚合层自算，幸存者口径 = ok 行带值样本；
    # 与 per_second 图的每秒分位互补）：单键直读，缺失整行不显示
    # （full_e2e 行同例；no-backward-compat：ttft_ms / total_ms 旧键回退已删）。
    ttft_sum = sm.get("ttft_latency_ms")
    if ttft_sum:
        _ttft_cell = (
            "p50 "
            + fmt_ms(ttft_sum.get("p50"))
            + " / p99 "
            + fmt_ms(ttft_sum.get("p99"))
            + " ms"
        )
        if ttft_sum.get("count"):
            _ttft_cell += " · n=" + fmt_int_trunc(ttft_sum.get("count"))
        rows.append(["TTFT（全程）", _ttft_cell])
    e2e_sum = sm.get("e2e_latency_ms")
    if e2e_sum:
        _e2e_cell = (
            "p50 "
            + fmt_ms(e2e_sum.get("p50"))
            + " / p99 "
            + fmt_ms(e2e_sum.get("p99"))
            + " ms"
        )
        if e2e_sum.get("count"):
            _e2e_cell += " · n=" + fmt_int_trunc(e2e_sum.get("count"))
        rows.append(["端到端延迟（全程）", _e2e_cell])
    # full_e2e（跨两侧全链路）：旧 aggregate 无 full_e2e_latency_ms 键时
    # 整行不显示（回退），不显示 "—" 占位——该指标依赖新引擎日志行，
    # 旧数据本来就没有。
    full_e2e_sum = sm.get("full_e2e_latency_ms") or {}
    if full_e2e_sum:
        _fe_cell = (
            "p50 "
            + fmt_ms(full_e2e_sum.get("p50"))
            + " / p99 "
            + fmt_ms(full_e2e_sum.get("p99"))
            + " ms"
        )
        if full_e2e_sum.get("count"):
            _fe_cell += (
                " · n=" + fmt_int_trunc(full_e2e_sum.get("count")) + "（按 rid 关联）"
            )
        rows.append(["全链路延迟（发出→decode 结束）", _fe_cell])
    pcv = (ed.get("prefill") or {}).get("cv") if ed else None
    dcv = (ed.get("decode") or {}).get("cv") if ed else None
    p_tg = ed_p.get("tokens_gini_cum")
    d_tg = ed_d.get("tokens_gini_cum")
    p_ug = (util_block.get("prefill") or {}).get("gini_cum")
    d_ug = (util_block.get("decode") or {}).get("gini_cum")
    bal_parts = []
    if pg is not None or dg is not None:
        _gini_cell = "请求 Gini " + fmt_g3(pg) + " / " + fmt_g3(dg) + "（成功行）"
        if pg_all is not None or dg_all is not None:
            _gini_cell += " · 全量 " + fmt_g3(pg_all) + " / " + fmt_g3(dg_all)
        bal_parts.append(_gini_cell)
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
    # Phase A quick-stats：sent/started/recorded/completed 四计数
    # （validity 六项的原始量；旧 aggregate 无这些键时整行跳过）。
    _cnt_parts = []
    if sm.get("sent_task_count") is not None:
        _cnt_parts.append("sent " + fmt_int_trunc(sm.get("sent_task_count")))
    if sm.get("actual_rpc_start_count") is not None:
        _cnt_parts.append("started " + fmt_int_trunc(sm.get("actual_rpc_start_count")))
    if sm.get("recorded_result_count") is not None:
        _cnt_parts.append("recorded " + fmt_int_trunc(sm.get("recorded_result_count")))
    if sm.get("completed_count") is not None:
        _cnt_parts.append("completed " + fmt_int_trunc(sm.get("completed_count")))
    if _cnt_parts:
        rows.append(
            ["请求计数（sent/started/recorded/completed）", " / ".join(_cnt_parts)]
        )
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

    # ---- 报告级统一时间轴 ----
    # T_END = 全部时序面板最大采样点（ceil 整秒，含收尾排空）；min 固定 0
    # （t=0 = 压测正式开始，warmup 后）。渲染层把所有 timeX 面板的
    # linear x 轴钉在 [0, T_END]，warmup 负值段被轴裁剪（数据保留不删）。
    # 旧问题：各面板各自取数据首尾，同一报告内至少四套时间范围混用
    # （客户端 [0,70]、master [-10,242.2]、调度阶段 [9.8,239.8]、
    # 引擎事件 [-23.9,243.1]）。
    time_axis = None
    if cats_time:
        t_end = int(math.ceil(max(v for vals in cats_time.values() for v in vals)))
        time_axis = {"min": 0, "max": t_end}

    # ---- 元数据区 spec（三层规范化，20260902）----
    # 可见 meta 面板：时间轴口径 + 采样说明（口径标注纪律，报告头必须
    # 直观可读）。detail 层（<details> 折叠，默认收起）：代码版本 /
    # 数据集 / 实验参数 / 环境变量 / 数据源——旧三分区中的数据源从可见
    # 面板移入 detail；规模不设分区（与 subtitle 实验条件重复，已删）。
    # detail 取数链：aggregate meta（aggregate_canvas_run.py 20260902+
    # 写入）> 同目录 run_meta.json；均缺则对应分区显示 —（未提供）。
    ed_embedded = (
        ed is not None
        and args.engine_dist is None
        and isinstance(agg.get("engine_dist"), dict)
    )
    _run_dir_abs = os.path.abspath(os.path.dirname(args.aggregate) or ".")
    if os.path.basename(_run_dir_abs) in ("analysis", "load_client"):
        _run_dir_abs = os.path.dirname(_run_dir_abs)
    _rm_full = load_json(_rm_path) if os.path.isfile(_rm_path) else {}
    _rm_client_env = (_rm_full.get("client_env") or {}) if _rm_full else {}
    _rm_flexlb_env = (_rm_full.get("flexlb_env") or {}) if _rm_full else {}

    def _detail_str(key):
        v = str(_agg_meta.get(key) or _rm_params.get(key) or "").strip()
        return v or None

    meta_spec = {
        "sources": {
            "runDir": _run_dir_abs,
            "aggregate": os.path.abspath(args.aggregate),
            "engineDist": (
                "(aggregate 内嵌 engine_dist)"
                if ed_embedded
                else (os.path.abspath(ed_path) if ed is not None else None)
            ),
        },
        "timeAxis": ({"tEnd": time_axis["max"]} if time_axis else None),
        "sampling": sampling_note,
        "version": {"branch": git_branch, "commit": git_commit},
        "dataset": {
            "traceFile": _detail_str("trace_file"),
            "traceLines": _cond_int(
                _agg_meta.get("trace_file_lines"),
                _rm_params.get("trace_file_lines"),
            ),
            "traceSha256": _detail_str("trace_file_sha256"),
        },
        # 实验参数全量（run_meta.params：拓扑/端口/容量/JVM/配置文件路径等）
        "params": _rm_params or None,
        # FINAL ENV 快照：JavaLoadClient env（client_env.json）+ FlexLB env
        # （flexlb_env.txt），consolidate 阶段嵌入 run_meta.json
        "env": {
            "clientEnv": _rm_client_env or None,
            "flexlbEnv": _rm_flexlb_env or None,
        },
    }
    subtitle_html = identity

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

    spec = _extract_spec_from_tsx(
        tsx_src,
        run_id=str(run_id),
        subtitle=subtitle_html,
        time_axes=cats_time,
        time_axis=time_axis,
    )
    spec["meta"] = meta_spec
    html_out = canvas_report_render_html.render(spec)

    # ---- 时间轴自检（fail-closed）----
    # 1) 合法性：min=0、max>0、min<max；2) 渲染输出含 TA_MIN/TA_MAX 钉轴
    # 配置；3) 每个时间轴面板 xNums 与全部 series data 等长（轴对齐前提）。
    time_panels = [p for p in spec.get("panels", []) if p.get("timeX")]
    if cats_time:
        assert (
            time_axis is not None
            and time_axis["min"] == 0
            and time_axis["max"] > 0
            and time_axis["min"] < time_axis["max"]
        ), (TAG + " time axis invalid: " + repr(time_axis))
        assert time_panels, (
            TAG + " time axis computed but no time-series panel extracted"
        )
        assert "TA_MIN" in html_out and "TA_MAX" in html_out, (
            TAG + " rendered HTML missing TA_MIN/TA_MAX x-axis pinning"
        )
        for p in time_panels:
            xn = p.get("xNums") or []
            assert xn, TAG + " empty xNums on time-series panel: " + p["title"]
            for s in p["series"]:
                assert len(xn) == len(s["data"]), (
                    TAG + " xNums/series length mismatch on panel: " + p["title"]
                )

    # ---- 元数据区 / leak KPI 自检（fail-closed）----
    # 1) 元数据区存在性：spec.meta 齐全（sources/version/dataset/
    #    params/env），渲染输出含数据源绝对路径与 T_END 时间轴口径字样
    #    （有时间轴时）；
    # 2) leak chip 负向：头部 KPI 无「泄漏判定」且渲染 HTML 全文无该字样
    #    （tsx 汇总表行保留但不进 HTML，见 leak_label 计算处注释——若未来
    #    汇总表进 HTML，此断言会拦下，需同步重新评审 leak 展示面）；
    # 3) 三层规范化（20260902）：detail 折叠块存在且默认收起；replay 模式
    #    且倍率可得时 subtitle 必含 replay@<speed>x（倍率取自动校准值，
    #    非 CLI 缺省）；KPI 含结果行五连。
    _meta_chk = spec.get("meta") or {}
    _meta_src = _meta_chk.get("sources") or {}
    assert isinstance(_meta_chk.get("sources"), dict), (
        TAG + " meta panel spec incomplete: sources missing"
    )
    assert _meta_src.get("aggregate") == os.path.abspath(args.aggregate), (
        TAG + " meta sources.aggregate must be the input aggregate absolute path"
    )
    assert _meta_src.get("runDir"), TAG + " meta sources.runDir missing"
    assert 'id="meta"' in html_out, TAG + " rendered HTML missing metadata panel"
    assert os.path.abspath(args.aggregate) in html_out, (
        TAG + " rendered HTML missing data-source path (detail panel)"
    )
    if time_axis:
        assert "T_END" in html_out, (
            TAG + " rendered HTML missing T_END time-axis semantics in metadata panel"
        )
    assert all(k.get("label") != "泄漏判定" for k in spec.get("kpis", [])), (
        TAG + " leak KPI chip must not appear in header kpis"
    )
    assert "泄漏判定" not in html_out, (
        TAG + " rendered HTML must not contain leak verdict text "
        "(header KPI chip removed by design)"
    )
    # detail 折叠块：默认收起（无 open 属性）+ 汇总行存在
    assert '<details id="detail"' in html_out, (
        TAG + ' rendered HTML missing detail panel (<details id="detail">)'
    )
    assert '<details id="detail" open' not in html_out, (
        TAG + " detail panel must be collapsed by default (no open attr)"
    )
    # 规模分区已删（与 subtitle 实验条件重复），渲染输出不得再含该字样
    assert "规模" not in html_out, (
        TAG
        + " rendered HTML must not contain scale section "
        + "(duplicated with subtitle experiment conditions)"
    )
    # subtitle 倍率：replay 模式且倍率可得时必含 replay@<speed>x；倍率
    # 不可得时不显示倍率段（而非回退硬编码缺省——replay@1000x bug 回归门）
    if send_mode == "replay":
        if replay_speed:
            assert ("replay@" + str(replay_speed) + "x") in html_out, (
                TAG
                + " subtitle missing replay@"
                + str(replay_speed)
                + "x (calibrated speed)"
            )
        else:
            assert "replay@" not in html_out, (
                TAG + " replay speed unknown yet subtitle shows a rate "
                "(stale default leaked in?)"
            )
    _kpi_labels = [k.get("label") for k in spec.get("kpis", [])]
    for _need in (
        "发送 QPS（master）",
        "请求数量",
        "成功数量",
        "失败 / cancel",
        "成功率",
        "持续时间",
    ):
        if _need == "发送 QPS（master）" and not send_qps_master:
            continue  # 旧 aggregate 无 master 口径时回退 client 自估 label
        assert _need in _kpi_labels, (
            TAG + " header kpis missing result-row chip: " + _need
        )
    assert len(_kpi_labels) == 10, (
        TAG
        + " header kpis must be 10 chips (5 metric + 5 result), got "
        + str(len(_kpi_labels))
    )

    # ---- 观测口径自检（fail-closed，20260829 口径修正批次）----
    # 1) KPI 发送 QPS：master arrival 口径优先，有 server_arrival_qps 时
    #    HTML 必含 master 口径 KPI 标注与 client 参考值（双值 foot）；
    # 2) Gini：aggregate 提供 _all 全量键时，HTML 必含全量口径标注；
    # 3) 分位数样本量：schedule 全终态 count 必须出现在面板 caption；
    # 4) route_submit 幸存者口径：样本 < 全终态时 caption 必须标注；
    # 5) 发送序列换源（20260830，20260831 改均值口径）：master 每秒
    #    arrivals 序列存在时，其 active 桶均值必须与 server_arrival_qps
    #    （全程均值口径）同数量级（0.5x-3x），防止图表序列与 KPI 再次
    #    错位；峰值口径对 burst 双峰流量过严（快速失败语义下峰值可达
    #    均值 4x+），降级为 stderr 警告；HTML 必含 master 到达口径与
    #    客户端参考线的双口径标注。
    if send_qps_master:
        assert "发送 QPS（master）" in html_out, (
            TAG + " KPI send-QPS must carry the master-scope label"
        )
        assert "client 自估" in html_out, (
            TAG + " per-second QPS panel must carry the client reference footnote"
        )
    if qps_master_arrivals is not None:
        assert "master 到达口径" in html_out, (
            TAG
            + " master arrivals series present but the QPS panel caption "
            + "lacks the master-arrival scope label"
        )
        assert "发送（客户端口径）" in html_out, (
            TAG
            + " master arrivals series present but the client-scope "
            + "reference series label is missing"
        )
        if send_qps_master:
            # 20260831 口径修正：防「序列与 KPI 接错源」的判据必须是同
            # 口径对比——active 桶均值 vs 全程均值 KPI。峰值口径在 burst
            # 双峰流量下（p50 每秒几百、峰值桶近 9k）可达均值 4x+，属真
            # 实过载形态而非错位，故峰值超 3x 仅告警不阻断。
            _m_active_vals = [v for v in m_arr_by_t.values() if v > 0]
            m_active_mean = (
                sum(_m_active_vals) / len(_m_active_vals) if _m_active_vals else 0.0
            )
            _mean_ratio = (
                float(m_active_mean) / float(send_qps_master)
                if send_qps_master
                else 0.0
            )
            assert 0.5 <= _mean_ratio <= 3.0, (
                TAG
                + " master arrivals active-mean "
                + ("%.1f" % m_active_mean)
                + "/s inconsistent with server_arrival_qps "
                + fmt_int_trunc(send_qps_master)
                + " (ratio "
                + ("%.2f" % _mean_ratio)
                + ") — send series and KPI disagree again"
            )
            m_peak = max(m_arr_by_t.values())
            _ratio = float(m_peak) / float(send_qps_master)
            if not 0.5 <= _ratio <= 3.0:
                print(
                    TAG
                    + " burst warning: master arrivals peak "
                    + fmt_int_trunc(m_peak)
                    + "/s is "
                    + ("%.2f" % _ratio)
                    + "x server_arrival_qps "
                    + fmt_int_trunc(send_qps_master)
                    + " (burst shape; series/KPI mean ratio "
                    + ("%.2f" % _mean_ratio)
                    + " OK)",
                    file=sys.stderr,
                )
    if gini_is_all:
        assert "全量" in html_out, (
            TAG + " all-scope Gini data present but not labelled in HTML"
        )
    # 6) full_e2e 口径（20260830）：aggregate 带 full_e2e 序列时，HTML 必含
    #    full_e2e 及其口径标注（client 发出 → decode 执行结束，按 rid
    #    关联）——三口径（full_e2e 全链路 / e2e 调度 / decode exec 引擎）
    #    必须在面板 caption 上可区分，防止读者把调度口径 e2e 当全链路。
    if has_full_e2e:
        assert "full_e2e" in html_out, (
            TAG + " full_e2e series present but scope annotation missing from HTML"
        )
        assert "decode 执行结束" in html_out, (
            TAG + " full_e2e scope caption (client send → decode end) missing"
        )
    # 7) 引擎 exec 轴口径（20260830）：出生轴样本存在时 HTML 必含出生轴
    #    标注（request_id 关联 + 出生秒分桶 + 同轴可比）；完成轴回退线
    #    存在时必含完成轴与错位警示——防止读者把完成轴 exec 与出生轴
    #    full_e2e 同图直接对比（本批口径统一改造的原始动机）。
    if has_birth_pe or has_birth_de:
        assert "出生轴" in html_out, (
            TAG
            + " birth-axis engine exec present but birth-axis "
            + "annotation missing from HTML"
        )
        assert "出生秒" in html_out, (
            TAG
            + " birth-axis engine exec present but birth-second "
            + "bucketing annotation missing from HTML"
        )
    if used_completion_pe or used_completion_de:
        assert "完成轴" in html_out, (
            TAG
            + " completion-axis engine exec fallback present but "
            + "completion-axis annotation missing from HTML"
        )
        assert "错位" in html_out, (
            TAG
            + " completion-axis engine exec fallback present but "
            + "birth/completion axis misalignment warning missing from HTML"
        )
    if sched_lat_count:
        assert ("n=" + fmt_int_trunc(sched_lat_count)) in html_out, (
            TAG + " schedule-latency sample count missing from HTML captions"
        )
    _rs_n_chk = (stage_lat or {}).get("route_submit_ms") or {}
    if (
        _rs_n_chk.get("count")
        and sched_lat_count
        and _rs_n_chk["count"] < sched_lat_count
    ):
        assert "幸存者" in html_out, (
            TAG + " route_submit survivor-scope annotation missing from HTML"
        )

    # 8) TPS 口径（20260901，同日纠偏）：mock 自报 rtp_llm_* 线存在时，
    #    HTML 必含记账式口径标注（完成事件记账 + 1s 窗口——防止读者把
    #    mock 记账值当 GPU 算力直接对表）；P/D 主图必含角色语义标注
    #    （与生产大盘 hippo_role 切分读法对齐）；cache 复用对存在时必含
    #    复用语义标注。原 IO 对账面板相关断言（调度链路损耗/守恒）随
    #    面板移除同步删除（对账降级为 aggregate 的
    #    token_reconciliation_ok 断言）。
    #    20260901 呈现口径（per-engine average）：引擎数可得时主图必含
    #    「每引擎平均」标注与具体引擎数（集群和÷N 与生产大盘单实例
    #    series 同构读法——防集群和当单实例读数的 67 倍量级误读）；
    #    引擎数回退模式必含「集群和（引擎数未知）」回退标注。
    if mock_ctx_tps is not None or mock_gen_tps is not None:
        assert "完成事件记账" in html_out, (
            TAG + " mock TPS series present but accounting-scope annotation missing"
        )
        assert "1s 窗口" in html_out, (
            TAG + " mock TPS series present but window-scope annotation missing"
        )
    if mock_ctx_tps is not None:
        assert "P 角色" in html_out, (
            TAG + " context TPS chart present but P-role annotation missing"
        )
        assert "cache 复用等效吞吐" in html_out, (
            TAG + " context TPS pair present but cache-reuse annotation missing"
        )
        if tps_p_engines:
            assert "每引擎平均" in html_out, (
                TAG
                + " context TPS chart present but per-engine-average annotation missing"
            )
            assert ("P 引擎数 " + str(tps_p_engines)) in html_out, (
                TAG + " context TPS chart present but P engine-count annotation missing"
            )
        else:
            assert "集群和（引擎数未知）" in html_out, (
                TAG
                + " context TPS cluster-sum fallback present but fallback "
                + "annotation missing"
            )
    if mock_gen_tps is not None:
        assert "D 角色" in html_out, (
            TAG + " generate TPS chart present but D-role annotation missing"
        )
        if tps_d_engines:
            assert "每引擎平均" in html_out, (
                TAG
                + " generate TPS chart present but per-engine-average annotation missing"
            )
            assert ("D 引擎数 " + str(tps_d_engines)) in html_out, (
                TAG
                + " generate TPS chart present but D engine-count annotation missing"
            )
        else:
            assert "集群和（引擎数未知）" in html_out, (
                TAG
                + " generate TPS cluster-sum fallback present but fallback "
                + "annotation missing"
            )

    # 9) 引擎侧 KV v2 块池面板口径（20260902）：三态分解面板存在时 HTML
    #    必含「三态」与「释放 ≠ 删除」语义标注 + 角色级引擎数标注（每
    #    引擎平均 + 具体引擎数，或「集群和（引擎数未知）」回退标注）
    #    ——防止把三态 gauge 当普通占用曲线读；速率面板存在时必含
    #    「累计差分」标注（防止把差分速率当瞬时计数读）；准入失败
    #    面板存在时必含「正常健康档全零」观测语义标注。
    if kv_pool_gauge_roles:
        assert "三态" in html_out, (
            TAG
            + " kv block-pool gauge panels present but three-state "
            + "annotation missing"
        )
        assert "释放 ≠ 删除" in html_out, (
            TAG
            + " kv block-pool gauge panels present but release-semantics "
            + "annotation missing"
        )
        for _role in kv_pool_gauge_roles:
            _ecnt = tps_p_engines if _role == "prefill" else tps_d_engines
            if _ecnt:
                assert "每引擎平均" in html_out, (
                    TAG
                    + " kv block-pool gauge panel ("
                    + _role
                    + ") present but per-engine-average annotation missing"
                )
                assert (
                    ("P 引擎数 " if _role == "prefill" else "D 引擎数 ") + str(_ecnt)
                ) in html_out, (
                    TAG
                    + " kv block-pool gauge panel ("
                    + _role
                    + ") present but engine-count annotation missing"
                )
            else:
                assert "集群和（引擎数未知）" in html_out, (
                    TAG
                    + " kv block-pool gauge panel ("
                    + _role
                    + ") cluster-sum fallback present but fallback "
                    + "annotation missing"
                )
    if kv_pool_rate_present:
        assert "累计差分" in html_out, (
            TAG
            + " kv block-pool rate panels present but cumulative-diff "
            + "annotation missing"
        )
    if kv_pool_adm_present:
        assert "正常健康档全零" in html_out, (
            TAG
            + " kv admission-fail panel present but healthy-zero "
            + "annotation missing"
        )

    # 10) cache 命中率三口径面板（20260902）：「master 路由 vs engine
    #     执行」双曲线存在时 HTML 必含「master 路由口径」「engine
    #     执行口径」「调度损耗」语义标注（防止把两线当同口径重复读）；
    #     「key 级（理论）vs token 级（实际）」双曲线存在时必含
    #     「key 级理论口径」「token 级实际口径」「命中深度覆盖」标注
    #     （防止把理论/实际混读）；run 级汇总柱状图存在时必含「对齐
    #     生产」生产对齐标注（防误当 mock 独创口径读）。
    if cache_hit_route_present:
        assert "master 路由口径" in html_out, (
            TAG
            + " cache-hit routing panel present but master-routing "
            + "annotation missing"
        )
        assert "engine 执行口径" in html_out, (
            TAG
            + " cache-hit routing panel present but engine-execution "
            + "annotation missing"
        )
        assert "调度损耗" in html_out, (
            TAG
            + " cache-hit routing panel present but dispatch-loss "
            + "annotation missing"
        )
    if cache_hit_depth_present:
        assert "key 级理论口径" in html_out, (
            TAG
            + " cache-hit depth panel present but key-level-theory "
            + "annotation missing"
        )
        assert "token 级实际口径" in html_out, (
            TAG
            + " cache-hit depth panel present but token-level-actual "
            + "annotation missing"
        )
        assert "命中深度覆盖" in html_out, (
            TAG
            + " cache-hit depth panel present but depth-coverage "
            + "annotation missing"
        )
    if cache_hit_kpi_present:
        assert "对齐生产" in html_out, (
            TAG
            + " cache-hit run-level KPI chart present but "
            + "production-alignment annotation missing"
        )

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html_out)

    # ---- stdout 摘要 ----
    sections = ["qps"] if per_second else []
    if latency_containers:
        sections.append("latency")
    if tps_containers:
        sections.append("tps-pd-role")
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
    if kv_containers or kv_pool_containers or cache_hit_containers:
        sections.append("kv")
    if cache_hit_containers:
        sections.append("cache-hit")
    if res_containers:
        sections.append("resource")
    sections.append("summary")
    print(TAG + " run_id=" + str(run_id))
    print(
        TAG
        + " inputs: aggregate="
        + os.path.basename(args.aggregate)
        + " slo="
        + ("yes" if slo is not None else "no")
        + " engine_dist="
        + ("yes" if ed is not None else "no")
    )
    print(TAG + " sections: " + ", ".join(sections))
    print(
        TAG
        + " send-qps scope: "
        + (
            "master arrival " + fmt_int_trunc(send_qps_master)
            if send_qps_master
            else "client estimate " + fmt_int_trunc(send_qps_client)
        )
        + (
            " (client ref " + fmt_int_trunc(send_qps_client) + ")"
            if send_qps_master and send_qps_client
            else ""
        )
    )
    if m_arr_by_t:
        _m_peak = max(m_arr_by_t.values())
        _m_active = sorted(t for t, v in m_arr_by_t.items() if v > 0)
        print(
            TAG
            + " send-series scope: master arrivals ts (n="
            + str(len(m_arr_by_t))
            + "s, peak "
            + fmt_int_trunc(_m_peak)
            + "/s"
            + (
                ", active t=[" + str(_m_active[0]) + "," + str(_m_active[-1]) + "]"
                if _m_active
                else ""
            )
            + ")"
        )
    if tps_containers:
        print(
            TAG
            + " tps scope: P="
            + (
                "per-engine avg ÷" + str(tps_p_engines)
                if tps_p_engines
                else "cluster-sum (engine count unknown)"
            )
            + " D="
            + (
                "per-engine avg ÷" + str(tps_d_engines)
                if tps_d_engines
                else "cluster-sum (engine count unknown)"
            )
            + " (production dashboard single-instance series read)"
        )
    if kv_pool_containers:
        print(
            TAG
            + " kv-blocks scope: engine-side block-pool panels (three-state "
            + "gauges + admission/reuse/eviction diff rates) P="
            + (
                "per-engine avg ÷" + str(tps_p_engines)
                if tps_p_engines
                else "cluster-sum"
            )
            + " D="
            + (
                "per-engine avg ÷" + str(tps_d_engines)
                if tps_d_engines
                else "cluster-sum"
            )
        )
    if full_e2e_sum:
        print(
            TAG
            + " full_e2e: n="
            + fmt_int_trunc(full_e2e_sum.get("count"))
            + " p50="
            + fmt_ms(full_e2e_sum.get("p50"))
            + " p99="
            + fmt_ms(full_e2e_sum.get("p99"))
            + " ms (client send → decode end, joined by request_id)"
        )
    if has_birth_pe or has_birth_de or used_completion_pe or used_completion_de:
        print(
            TAG
            + " engine-exec axis: prefill="
            + (
                "birth"
                if has_birth_pe
                else ("completion" if used_completion_pe else "none")
            )
            + " decode="
            + (
                "birth"
                if has_birth_de
                else ("completion" if used_completion_de else "none")
            )
            + " (birth = rid-joined per-second buckets, same axis as full_e2e)"
        )
    print(
        TAG
        + " gini scope: "
        + (
            "all-placement " + fmt_g3(pg_disp) + "/" + fmt_g3(dg_disp)
            if gini_is_all
            else "ok-rows " + fmt_g3(pg) + "/" + fmt_g3(dg)
        )
        + (
            " (ok-rows " + fmt_g3(pg) + "/" + fmt_g3(dg) + ")"
            if gini_is_all and (pg is not None or dg is not None)
            else ""
        )
    )
    print(
        TAG
        + " time axis: "
        + (
            "[0, "
            + str(time_axis["max"])
            + "]s over "
            + str(len(time_panels))
            + " panels (t=0 = 压测正式开始, T_END 含收尾排空)"
            if time_axis
            else "none (no time-series panels)"
        )
    )
    for w in warnings:
        print(TAG + " warning: " + w)
    if ed:
        for note in ed.get("notes") or []:
            print(TAG + " engine_dist note: " + str(note))
    print(
        TAG
        + " header tiers: subtitle("
        + identity
        + ") · kpi rows 2x5 · detail(collapsed)"
    )
    print(
        TAG
        + " meta panel: sources(runDir/aggregate"
        + ("/summary" if _meta_src.get("summary") else "")
        + ("/engineDist" if _meta_src.get("engineDist") else "")
        + ") · version("
        + (git_branch or "?")
        + "@"
        + (git_commit or "?")
        + ")"
        + (
            " · timeAxis T_END=" + str(meta_spec["timeAxis"]["tEnd"]) + "s"
            if meta_spec.get("timeAxis")
            else ""
        )
    )
    print(TAG + " written: " + args.out)


if __name__ == "__main__":
    main()
