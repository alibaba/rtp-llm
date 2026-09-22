#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FlexLB 压测报告 — spec → self-contained Chart.js HTML 渲染器。

图表依赖：仓库固定的 Chart.js 4.4.7 UMD，内嵌后离线可打开。
观感：浅色主题 / 白卡 / 6 列 KPI / 2 列 chart grid / .box 高 280px，对齐既有
`outputs/flexlb-run-*-chartjs.html` 的样式。
交互：legend 单击切换单条 / tooltip 随鼠标 index 联动；无 zoom 插件。

Spec schema（core.py 产）：
  {
    'run_id':   str,
    'title':    str,              # 页首 H1
    'subtitle': Optional[str|dict], # 自由文本或任意 key-value 摘要
    'kpis':     [{'label': str, 'value': str, 'tone': Optional[str]}],
                                   # 两行×最多 5 chip：指标五连 + 结果五连
                                   # （请求数量 / 成功 / 失败·cancel / 成功率 /
                                   # 持续时间），三层第二层
    'meta':     Optional[dict],    # 旧直出报告的元信息，自动过滤空字段
    'run_meta': Optional[dict],    # 标准 run provenance；runs 可含多侧信息
    'timeAxis': Optional[{'min': number, 'max': number}],  # 报告级统一时间轴
    'panels':   [panel],
  }
  panel = {
    'id':      str,               # canvas dom id (唯一)
    'title':   str,
    'caption': str,
    'type':    'line' | 'bar',
    'x':       [str, ...],        # x 轴 label 数组（类目轴）
    'timeX':   Optional[bool],    # True = 时间轴面板（linear x 轴钉 [TA_MIN, TA_MAX]）
    'xNums':   Optional[number],  # 与 x 等长同序的数值时间序列（timeX 时必填）
    'yMax':    Optional[number],  # y 轴 suggestedMax
    'unit':    Optional[str],     # y 轴 / tooltip 后缀
    'series':  [{'name': str, 'data': [num], 'color': str}],
  }

标题与副标题完全由 spec 提供。运行信息优先读取 run_meta，旧直出报告
读取 meta；空字段不渲染。多 run 的每一侧独立展示。附件中的结构化证据
保留为可折叠、限高滚动的代码块。

时间轴语义：timeAxis.min = 0（t=0 = 压测正式开始，warmup 后）；
timeAxis.max = T_END（全部时序面板最后采样点，ceil 整秒，含收尾排空）。
timeX 面板数据点转 {x, y}，scales.x = linear + min/max 钉住；warmup
负值段被轴裁剪（数据保留不删）。非时间轴面板保持类目轴不变。
"""

from __future__ import annotations

import html
import json
from pathlib import Path

# tone → 数值色（KPI 卡）
KPI_TONE_COLOR = {
    "success": "#52c41a",
    "danger": "#f5222d",
    "warn": "#faad14",
    "warning": "#faad14",
    "info": "#1677ff",
    "primary": "#1677ff",
}

# 系列默认调色板（chartjs.html 里一图内多系列的常用色）
PALETTE = [
    "#1677ff",  # primary blue
    "#52c41a",  # success green
    "#faad14",  # warn amber
    "#f5222d",  # danger red
    "#722ed1",  # purple
    "#13c2c2",  # cyan
    "#eb2f96",  # magenta
    "#fa8c16",  # orange
    "#a0d911",  # lime
    "#2f54eb",  # geekblue
    "#fadb14",  # yellow
    "#08979c",  # teal
]

# 语义 tone → 色（系列层面用）
TONE_TO_COLOR = {
    "primary": "#1677ff",
    "success": "#52c41a",
    "warning": "#faad14",
    "warn": "#faad14",
    "danger": "#f5222d",
    "info": "#13c2c2",
    "secondary": "#722ed1",
    "tertiary": "#eb2f96",
    "quaternary": "#fa8c16",
    "neutral": "#8c8c8c",
}


def series_color(tone, idx):
    if tone and tone in TONE_TO_COLOR:
        return TONE_TO_COLOR[tone]
    return PALETTE[idx % len(PALETTE)]


def _present(value):
    """Remove absent metadata without treating zero or false as missing."""
    if isinstance(value, dict):
        cleaned = {k: item for k, v in value.items()
                   if (item := _present(v)) is not None and k != "schema_version"}
        return cleaned or None
    if isinstance(value, list):
        cleaned = [item for v in value if (item := _present(v)) is not None]
        return cleaned or None
    return None if value is None or value == "" else value


def render_context(spec):
    """Show actual run metadata, including every side of a comparison."""
    meta = spec.get("run_meta")
    if meta:
        runs = meta.get("runs") if isinstance(meta, dict) else None
        cards = runs.items() if isinstance(runs, dict) and runs else [("本次运行", meta)]
    else:
        legacy = dict(spec.get("meta") or {})
        legacy.pop("timeAxis", None)
        cards = [("本次运行", legacy)]
    rendered = []
    for label, value in cards:
        value = _present(value)
        if not value:
            continue
        title = html.escape(str(label), quote=True)
        data = html.escape(json.dumps(value, ensure_ascii=False, indent=2), quote=True)
        rendered.append(f'<article class="context-card"><h3>{title}</h3><pre>{data}</pre></article>')
    if not rendered:
        return ""
    return '<section class="report-context"><h2>运行信息</h2><div class="context-grid">' + "".join(rendered) + "</div></section>"


def render_sections(sections):
    """The only HTML construction boundary for report tables and evidence blocks."""

    def text(value):
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False, indent=2)
        return html.escape(str(value), quote=True)

    out = []
    for section in sections:
        title = text(section.get("title", ""))
        kind = section["type"]
        if kind == "details":
            out.append(
                f'<details class="attachment"><summary>{title}</summary><pre>{text(section["value"])}</pre></details>'
            )
        elif kind == "table":
            heads = "".join("<th>" + text(c) + "</th>" for c in section["columns"])
            rows = "".join(
                "<tr>" + "".join("<td>" + text(v) + "</td>" for v in row) + "</tr>"
                for row in section["rows"]
            )
            out.append(
                f"<section><h2>{title}</h2><table><thead><tr>{heads}</tr></thead><tbody>{rows}</tbody></table></section>"
            )
        elif kind == "links":
            items = []
            for item in section["items"]:
                href = item["href"]
                if ":" in href or href.startswith("//"):
                    raise ValueError("report links must be relative artifact paths")
                items.append(
                    '<li><a href="'
                    + text(href)
                    + '">'
                    + text(item["label"])
                    + "</a></li>"
                )
            out.append(f'<section><h2>{title}</h2><ul>{"".join(items)}</ul></section>')
        else:
            raise ValueError("unsupported report section: " + kind)
    return '<div class="report-sections">' + "".join(out) + "</div>"


def render(spec):
    """spec: 见模块 docstring。返回完整 HTML 字符串。"""
    run_id = spec.get("run_id", "")
    title = spec.get("title") or ("FlexLB 压测报告 · run " + run_id)
    subtitle = spec.get("subtitle") or ""
    if isinstance(subtitle, dict):
        subtitle = {str(k): str(v) for k, v in subtitle.items() if v is not None}
    elif not isinstance(subtitle, str):
        raise TypeError("report subtitle must be a string or key-value mapping")
    kpis = spec.get("kpis") or []
    panels = spec.get("panels") or []

    payload = {
        "summary": {
            "title": title,
            "subtitle": subtitle,
            "kpis": [
                {
                    "label": k.get("label", ""),
                    "value": k.get("value", ""),
                    "tone": k.get("tone") or "",
                }
                for k in kpis
            ],
        },
        "timeAxis": spec.get("timeAxis"),
        "events": spec.get("events", []),
        "timeOriginLabel": spec.get("timeOriginLabel"),
        "meta": spec.get("meta"),
        "panels": [
            {
                "id": p["id"],
                "title": p.get("title", ""),
                "caption": p.get("caption", ""),
                "type": p.get("type", "line"),
                "bounds": p.get("bounds"),
                "axisLabels": p.get("axisLabels"),
                "events": p.get("events"),
                "x": p.get("x", []),
                "timeX": bool(p.get("timeX")),
                "xNums": p.get("xNums") or [],
                "yMax": p.get("yMax"),
                "overlay": p.get("overlay", False),
                "axes": p.get("axes", {}),
                "presets": p.get("presets", {}),
                "unit": p.get("unit", "") or "",
                "series": [
                    {
                        "name": s.get("name", ""),
                        "data": s.get("data", []),
                        "points": s.get("points"),
                        "axis": s.get("axis", "y"),
                        "unit": s.get("unit", ""),
                        "group": s.get("group", "其他"),
                        "description": s.get("description", ""),
                        "hidden": s.get("hidden", False),
                        "dash": s.get("dash", []),
                        "color": s.get("color") or series_color(s.get("tone"), i),
                    }
                    for i, s in enumerate(p.get("series", []))
                ],
            }
            for p in panels
        ],
    }

    sections = list(spec.get("sections", []))
    if spec.get("run_meta") is not None:
        sections.append(
            dict(type="details", title="Run provenance", value=spec["run_meta"])
        )
    page_title = html.escape(title)
    resource_dir = Path(__file__).resolve().parent / "assets"
    chartjs = (resource_dir / "chart.umd.min.js").read_text(encoding="utf-8")
    overlay = (resource_dir / "multi_curve.js").read_text(encoding="utf-8")
    interaction = (resource_dir / "legend_interaction.js").read_text(encoding="utf-8")
    return (
        _TEMPLATE.replace("__SECTIONS__", render_sections(sections))
        .replace("__CONTEXT__", render_context(spec))
        .replace("__PAGE_TITLE__", page_title)
        .replace(
            "__SPEC_JSON__",
            json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c"),
        )
        .replace("__CHARTJS_JS__", chartjs)
        .replace("__LEGEND_INTERACTION_JS__", interaction)
        .replace("__MULTI_CURVE_JS__", overlay)
    )


_TEMPLATE = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"/>
<title>__PAGE_TITLE__</title>
<script>__CHARTJS_JS__</script>
<script>__MULTI_CURVE_JS__</script>
<script>__LEGEND_INTERACTION_JS__</script>
<style>
.report-sections{padding:24px}.report-sections table{width:100%;border-collapse:collapse}.report-sections td,.report-sections th{border:1px solid #ddd;padding:6px;text-align:left}.report-sections pre{white-space:pre-wrap;overflow-wrap:anywhere}.report-sections details{margin:16px 0}
:root{
  --bg:#f5f6fa; --card:#fff; --fg:rgba(0,0,0,0.85); --sub:rgba(0,0,0,0.55);
  --border:rgba(0,0,0,0.08); --danger:#f5222d; --success:#52c41a; --warn:#faad14;
}
*{box-sizing:border-box}
body{margin:0;padding:24px;background:var(--bg);color:var(--fg);
  font:14px/1.55 -apple-system,"PingFang SC","Microsoft YaHei",sans-serif}
header{margin-bottom:20px}
h1{margin:0 0 6px;font-size:22px;overflow-wrap:anywhere}
.sub{color:var(--sub)}
.multi-toolbar{position:sticky;top:8px;z-index:5;display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin:10px 0;padding:8px;background:rgba(255,255,255,.97);border:1px solid var(--border);border-radius:8px;box-shadow:0 2px 10px rgba(0,0,0,.06)}
.multi-toolbar button{border:1px solid #d9d9d9;background:#fff;border-radius:6px;padding:5px 11px;cursor:pointer}
.multi-toolbar button:hover{border-color:#1677ff;color:#1677ff}.multi-range{margin-left:auto;color:var(--sub)}
.multi-range input{margin:0 5px;border:1px solid #d9d9d9;border-radius:5px;padding:4px}
.multi-picker{position:relative}.multi-picker-button{font-weight:600;color:#1677ff}
.multi-dropdown{position:absolute;top:calc(100% + 8px);left:0;width:min(520px,calc(100vw - 56px));max-height:min(68vh,560px);overflow:auto;padding:10px;background:#fff;border:1px solid #d9d9d9;border-radius:8px;box-shadow:0 8px 28px rgba(0,0,0,.18)}
.multi-dropdown[hidden],.multi-choice[hidden],.multi-group[hidden]{display:none}
.multi-search{position:sticky;top:-10px;z-index:1;width:100%;padding:7px 10px;border:1px solid #d9d9d9;border-radius:6px;background:#fff}
.multi-choices{display:grid;grid-template-columns:repeat(auto-fit,minmax(215px,1fr));gap:4px 8px;margin-top:8px}
.multi-group{grid-column:1/-1;color:var(--sub);font-size:12px;font-weight:650;margin-top:4px}
.multi-choice{display:flex;align-items:center;gap:5px;min-width:0;padding:4px 7px;border-radius:5px;cursor:pointer;transition:opacity .12s,background .12s}
.multi-choice:hover{background:#eef4ff}.multi-choice i,.multi-hover-row i{width:10px;height:10px;border-radius:50%;display:inline-block;flex:none}
.multi-choice input{margin:0}
.multi-legend{display:flex;flex-wrap:wrap;align-content:flex-start;gap:5px 8px;max-height:104px;overflow:auto;margin:9px 0;padding:8px;border:1px solid var(--border);border-radius:8px}
.multi-legend[hidden],.multi-legend-item[hidden]{display:none}
.multi-legend-key{width:100%;order:-1;color:var(--sub);font-size:12px}
.multi-legend-item{display:inline-flex;align-items:center;gap:6px;border:1px solid transparent;border-radius:5px;background:transparent;padding:3px 6px;color:var(--fg);font:inherit;font-size:12px;cursor:pointer}
.multi-legend-item:hover{background:#eef4ff}
.multi-legend-item i{display:inline-block;width:22px;border-top-width:3px;border-top-style:solid;flex:none}
.multi-hover{display:flex;flex-wrap:wrap;align-content:flex-start;gap:3px 12px;min-height:48px;max-height:140px;overflow:auto;background:#fafafa;border:1px solid var(--border);border-radius:8px;padding:9px 11px;color:var(--sub);font-size:12px}
.multi-hover strong{width:100%;color:var(--fg)}
.multi-hover-row{display:flex;align-items:center;gap:7px;min-width:210px;white-space:nowrap}
/* KPI 两行（指标五连 + 结果五连）：wrapper 纵向叠行，每行 grid 随
   行内 chip 数自适应列数（JS 注入 inline grid-template-columns）。 */
.kpi-stack{display:flex;flex-direction:column;gap:12px;margin:16px 0 12px}
.kpi-row{display:grid;grid-template-columns:repeat(6,1fr);gap:12px}
.kpi{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:12px 14px}
.kpi .v{font-size:22px;font-weight:600}
.kpi .l{color:var(--sub);font-size:12px;margin-top:4px}
.kpi.success .v{color:var(--success)} .kpi.danger .v{color:var(--danger)} .kpi.warn .v{color:var(--warn)}
.report-context{margin:0 0 16px}
.report-context h2{font-size:15px;margin:0 0 8px}
.context-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px}
.context-card{min-width:0;background:var(--card);border:1px solid var(--border);border-radius:8px;padding:12px}
.context-card h3{font-size:13px;margin:0 0 8px}
.context-card pre,.report-sections .attachment pre{max-height:280px;overflow:auto;margin:0;padding:10px 12px;background:#f7f8fa;border:1px solid var(--border);border-radius:6px;font:12px/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;white-space:pre}
.report-sections .attachment{margin:10px 0;border:1px solid var(--border);border-radius:8px;background:var(--card)}
.report-sections .attachment summary{cursor:pointer;padding:10px 14px;font-weight:600}
.report-sections .attachment pre{margin:0 12px 12px}
.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}
.panel{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:14px}
.panel h3{margin:0 0 4px;font-size:15px;overflow-wrap:anywhere}
.panel .cap{color:var(--sub);font-size:12px;margin-bottom:8px}
.panel .box{height:280px;position:relative}
</style></head><body>
<header>
  <h1 id="title"></h1><div class="sub" id="subtitle"></div>
</header>
<div class="kpi-stack" id="kpis"></div>
__CONTEXT__
<div class="grid" id="grid"></div>
<script>
const SPEC = __SPEC_JSON__;
document.getElementById('title').textContent = SPEC.summary.title;
const subtitle = SPEC.summary.subtitle;
document.getElementById('subtitle').textContent = typeof subtitle === 'string'
  ? subtitle : Object.entries(subtitle).map(([key,value])=>key+': '+value).join(' · ');
const kb = document.getElementById('kpis');
// KPI 两行（三层第二层）：每行最多 5 chip——第一行指标五连（发送 QPS /
// 成功调度 / 错误率 / Gini / pacing），第二行结果五连（请求数量 / 成功 /
// 失败·cancel / 成功率 / 持续时间）；列数随行内实际 chip 数自适应。
if (SPEC.summary.kpis.length)
  for (let i = 0; i < SPEC.summary.kpis.length; i += 5){
    const seg = SPEC.summary.kpis.slice(i, i + 5);
    const row = document.createElement('div'); row.className = 'kpi-row';
    row.style.gridTemplateColumns = 'repeat(' + seg.length + ',1fr)';
    seg.forEach(k=>{
      const d=document.createElement('div'); d.className='kpi '+(k.tone||'');
      d.innerHTML='<div class="v"></div><div class="l"></div>';
      d.querySelector('.v').textContent=k.value;
      d.querySelector('.l').textContent=k.label;
      row.appendChild(d);
    });
    kb.appendChild(row);
  }
// y 轴自适应：只按 legend 可见系列重算 max（beginAtZero），legend/双击后 update()。
// 时间轴面板数据点为 {x, y}（linear x 轴），此处兼容两种形态。
// 图例手势由 legend_interaction.js 统一处理；本层只负责曲线和 y 轴。
function visibleMax(chart){
  let m = 0; const ds = chart.data.datasets;
  chart.data.datasets.forEach((d,i)=>{
    if(!chart.getDatasetMeta(i).hidden){
      d.data.forEach(v=>{
        const y = (typeof v==='number') ? v : (v && typeof v.y==='number' ? v.y : 0);
        if(y>m) m=y;
      });
    }
  });
  return m>0 ? m*1.05 : undefined;
}
function rescaleY(chart){
  chart.options.scales.y.max = chart.$reportYBounds ? chart.$reportYBounds[1] : visibleMax(chart);
  if(chart.$reportYBounds) chart.options.scales.y.min=chart.$reportYBounds[0];
  chart.update('none');
}
const grid = document.getElementById('grid');
// 报告级统一时间轴：t=0 = 压测正式开始（warmup 后）；T_END = 全部
// 时序面板最后采样点（含收尾排空）。生成器侧注入 SPEC.timeAxis；缺失
// 或非法时回退 null（Chart.js 自动推导，向后兼容旧 spec）。
const TIME_AXIS = (SPEC.timeAxis
  && typeof SPEC.timeAxis.min === 'number'
  && typeof SPEC.timeAxis.max === 'number'
  && SPEC.timeAxis.max > SPEC.timeAxis.min)
  ? SPEC.timeAxis : null;
const TA_MIN = TIME_AXIS ? TIME_AXIS.min : undefined;
const TA_MAX = TIME_AXIS ? TIME_AXIS.max : undefined;
SPEC.panels.forEach(p=>{
  if (p.overlay) { FlexMultiCurve.mount(grid, p, {timeAxis:TIME_AXIS, events:SPEC.events || []}); return; }
  const wrap=document.createElement('div'); wrap.className='panel';
  wrap.innerHTML='<h3></h3><div class="cap"></div><div class="box"><canvas></canvas></div>';
  wrap.querySelector('h3').textContent=p.title;
  wrap.querySelector('.cap').textContent=p.caption;
  wrap.querySelector('canvas').id='c-'+p.id;
  grid.appendChild(wrap);
  const ctx=wrap.querySelector('canvas').getContext('2d');
  if(p.type==='scatter'){
    const b=p.bounds||{}, labels=p.axisLabels||{};
    new Chart(ctx,{type:'bubble', data:{datasets:p.series.map(s=>({label:s.name,
      data:s.points,backgroundColor:s.color,borderColor:s.color}))},
      options:{responsive:true,maintainAspectRatio:false,scales:{
        x:{type:'linear',min:b.x?.[0],max:b.x?.[1],title:{display:true,text:labels.x||''}},
        y:{min:b.y?.[0],max:b.y?.[1],title:{display:true,text:labels.y||''}}}}});
    return;
  }

  // 时间轴面板：数据点转 {x, y}，linear x 轴钉 [TA_MIN, TA_MAX]（warmup
  // 负值段被轴裁剪，数据保留）；tooltip 按 x 最近点联动。非时间轴面板
  // 保持类目轴 + index 联动。
  const isTime = !!(p.timeX && TIME_AXIS && p.xNums && p.xNums.length);
  let legendController;
  const chart = new Chart(ctx,{
    plugins: [{id:'phaseEvents', afterDraw(chart) {
      if (!isTime) return;
      const {ctx, chartArea:a, scales:{x}} = chart;
      ctx.save(); ctx.font='10px sans-serif';
      (SPEC.events || []).forEach((e,i) => {
        if (!Number.isFinite(e.t)) return;
        const px=x.getPixelForValue(e.t);
        if(px<a.left || px>a.right) return;
        ctx.strokeStyle='#94a3b8'; ctx.setLineDash([3,3]);
        ctx.beginPath(); ctx.moveTo(px,a.top); ctx.lineTo(px,a.bottom); ctx.stroke();
        ctx.fillStyle='#64748b'; ctx.fillText(e.name || e.label || '',px+2,a.top+12+(i%3)*12);
      });
      ctx.restore();
    }}],
    type: p.type==='bar'?'bar':'line',
    data:{
      labels: isTime ? undefined : p.x,
      datasets:p.series.map(s=>({
        label:s.name,
        data: isTime ? (s.points || s.data.map((v,i)=>({x:p.xNums[i], y:v}))) : s.data,
        borderColor:s.color, backgroundColor:s.color+'33',
        borderWidth:1.5, pointRadius:0, tension:0.15, fill:false,
      }))
    },
    options:{
      responsive:true, maintainAspectRatio:false,
      interaction: isTime
        ? {mode:'nearest', axis:'x', intersect:false}
        : {mode:'index', intersect:false},
      plugins:{
        legend:{
          position:'bottom', labels:{boxWidth:10,font:{size:11}},
          onClick:(e,item)=>legendController.click(item.datasetIndex)
        },
        tooltip:{callbacks:{
          title: isTime ? (items=>items.length ? ('t='+items[0].parsed.x+' s') : '') : undefined,
          label:c=>`${c.dataset.label}: ${c.parsed.y}${p.unit||''}`
        }},
      },
      scales: isTime ? {
        x:{type:'linear', min:TA_MIN, max:TA_MAX,
           ticks:{maxRotation:0, autoSkip:true, maxTicksLimit:12}},
        y:{beginAtZero:true}
      } : {
        x:{ticks:{maxRotation:0,autoSkip:true,maxTicksLimit:12}},
        y:{beginAtZero:true}
      }
    }
  });
  chart.$reportYBounds = p.bounds?.y;
  legendController = FlexLegend.controller(chart, rescaleY);
  // 初次 render 后按当前可见系列锁一次 max
  rescaleY(chart);
  {
    const btn=document.createElement('span');
    btn.textContent='⟲ 全选';
    btn.title='一键恢复该面板全部序列';
    btn.style.cssText='position:absolute;top:34px;right:10px;font-size:10px;color:#888;cursor:pointer;padding:2px 6px;border:1px solid #ddd;border-radius:10px;background:rgba(255,255,255,.9);z-index:5;user-select:none;transition:all .15s';
    btn.onmouseenter=()=>{btn.style.color='#333';btn.style.borderColor='#999';};
    btn.onmouseleave=()=>{btn.style.color='#888';btn.style.borderColor='#ddd';};
    btn.onclick=()=>legendController.all();
    wrap.style.position='relative';
    wrap.appendChild(btn);
  }
});
</script>__SECTIONS__</body></html>
"""
