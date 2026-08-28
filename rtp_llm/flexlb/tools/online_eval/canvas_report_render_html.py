#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FlexLB 压测报告 — spec → self-contained Chart.js HTML 渲染器。

外部依赖：仅 Chart.js 4.4.7（jsDelivr UMD）。
观感：浅色主题 / 白卡 / 6 列 KPI / 2 列 chart grid / .box 高 280px，对齐既有
`outputs/flexlb-run-*-chartjs.html` 的样式。
交互：legend 单击切换单条 / tooltip 随鼠标 index 联动；无 zoom 插件。

Spec schema（core.py 产）：
  {
    'run_id':   str,
    'title':    str,              # 页首 H1
    'subtitle': Optional[str],    # 副标（run 参数概览）
    'kpis':     [{'label': str, 'value': str, 'tone': Optional[str]}],
    'panels':   [panel],
  }
  panel = {
    'id':      str,               # canvas dom id (唯一)
    'title':   str,
    'caption': str,
    'type':    'line' | 'bar',
    'x':       [str, ...],        # x 轴 label 数组
    'yMax':    Optional[number],  # y 轴 suggestedMax
    'unit':    Optional[str],     # y 轴 / tooltip 后缀
    'series':  [{'name': str, 'data': [num], 'color': str}],
  }
"""

from __future__ import annotations

import html
import json

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


def render(spec):
    """spec: 见模块 docstring。返回完整 HTML 字符串。"""
    run_id = spec.get("run_id", "")
    title = spec.get("title") or ("FlexLB 压测报告 · run " + run_id)
    subtitle = spec.get("subtitle") or ""
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
        "panels": [
            {
                "id": p["id"],
                "title": p.get("title", ""),
                "caption": p.get("caption", ""),
                "type": p.get("type", "line"),
                "x": p.get("x", []),
                "yMax": p.get("yMax"),
                "unit": p.get("unit", "") or "",
                "series": [
                    {
                        "name": s.get("name", ""),
                        "data": s.get("data", []),
                        "color": s.get("color") or series_color(s.get("tone"), i),
                    }
                    for i, s in enumerate(p.get("series", []))
                ],
            }
            for p in panels
        ],
    }

    page_title = html.escape(title)
    return _TEMPLATE.replace("__PAGE_TITLE__", page_title).replace(
        "__SPEC_JSON__", json.dumps(payload, ensure_ascii=False)
    )


_TEMPLATE = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"/>
<title>__PAGE_TITLE__</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<style>
:root{
  --bg:#f5f6fa; --card:#fff; --fg:rgba(0,0,0,0.85); --sub:rgba(0,0,0,0.55);
  --border:rgba(0,0,0,0.08); --danger:#f5222d; --success:#52c41a; --warn:#faad14;
}
*{box-sizing:border-box}
body{margin:0;padding:24px;background:var(--bg);color:var(--fg);
  font:14px/1.55 -apple-system,"PingFang SC","Microsoft YaHei",sans-serif}
header{margin-bottom:20px}
h1{margin:0 0 6px;font-size:22px}
.sub{color:var(--sub)}
.kpi-row{display:grid;grid-template-columns:repeat(6,1fr);gap:12px;margin:16px 0 24px}
.kpi{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:12px 14px}
.kpi .v{font-size:22px;font-weight:600}
.kpi .l{color:var(--sub);font-size:12px;margin-top:4px}
.kpi.success .v{color:var(--success)} .kpi.danger .v{color:var(--danger)} .kpi.warn .v{color:var(--warn)}
.grid{display:grid;grid-template-columns:repeat(2,1fr);gap:14px}
.panel{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:14px}
.panel h3{margin:0 0 4px;font-size:15px}
.panel .cap{color:var(--sub);font-size:12px;margin-bottom:8px}
.panel .box{height:280px;position:relative}
.hint{margin-top:24px;color:var(--sub);font-size:12px}
</style></head><body>
<header>
  <h1 id="title"></h1><div class="sub" id="subtitle"></div>
</header>
<div class="kpi-row" id="kpis"></div>
<div class="grid" id="grid"></div>
<div class="hint">Chart.js 4.4.7 · legend 单击可切换单条系列，双击隔离；tooltip 随鼠标移动；隐藏后 y 轴按剩余可见系列自适应。</div>
<script>
const SPEC = __SPEC_JSON__;
document.getElementById('title').textContent = SPEC.summary.title;
document.getElementById('subtitle').textContent = SPEC.summary.subtitle;
const kb = document.getElementById('kpis');
SPEC.summary.kpis.forEach(k=>{
  const d=document.createElement('div'); d.className='kpi '+(k.tone||'');
  d.innerHTML=`<div class="v">${k.value}</div><div class="l">${k.label}</div>`;
  kb.appendChild(d);
});
// y 轴自适应：只按 legend 可见系列重算 max（beginAtZero），legend/双击后 update()。
// 双击隔离用手写 handler：默认 Chart.js 只支持单击切换单条，双击这里定义为"只留这条"，
// 再双击同一条恢复"全显"。
function visibleMax(chart){
  let m = 0; const ds = chart.data.datasets;
  chart.data.datasets.forEach((d,i)=>{
    if(!chart.getDatasetMeta(i).hidden){
      d.data.forEach(v=>{ if(typeof v==='number' && v>m) m=v; });
    }
  });
  return m>0 ? m*1.05 : undefined;
}
function rescaleY(chart){
  chart.options.scales.y.max = visibleMax(chart);
  chart.update('none');
}
const grid = document.getElementById('grid');
SPEC.panels.forEach(p=>{
  const wrap=document.createElement('div'); wrap.className='panel';
  wrap.innerHTML=`<h3>${p.title}</h3><div class="cap">${p.caption}</div><div class="box"><canvas id="c-${p.id}"></canvas></div>`;
  grid.appendChild(wrap);
  const ctx=wrap.querySelector('canvas').getContext('2d');
  const chart = new Chart(ctx,{
    type: p.type==='bar'?'bar':'line',
    data:{
      labels:p.x,
      datasets:p.series.map(s=>({
        label:s.name, data:s.data, borderColor:s.color, backgroundColor:s.color+'33',
        borderWidth:1.5, pointRadius:0, tension:0.15, fill:false,
      }))
    },
    options:{
      responsive:true, maintainAspectRatio:false,
      interaction:{mode:'index', intersect:false},
      plugins:{
        legend:{
          position:'bottom', labels:{boxWidth:10,font:{size:11}},
          onClick:(e,item,legend)=>{
            const ch=legend.chart; const idx=item.datasetIndex;
            // 双击（<300ms 同 idx）= 隔离该系列 / 恢复全显
            const now=Date.now();
            ch.$lastClick = ch.$lastClick || {};
            const last = ch.$lastClick[idx] || 0;
            ch.$lastClick[idx] = now;
            const others = ch.data.datasets.map((_,i)=>i).filter(i=>i!==idx);
            if(now - last < 300){
              // double click: toggle isolate
              const alreadyIsolated = others.every(i=>ch.getDatasetMeta(i).hidden)
                                   && !ch.getDatasetMeta(idx).hidden;
              others.forEach(i=>ch.getDatasetMeta(i).hidden = !alreadyIsolated);
              ch.getDatasetMeta(idx).hidden = false;
            }else{
              // single click: toggle visibility of this series
              const meta = ch.getDatasetMeta(idx);
              meta.hidden = meta.hidden===null ? !ch.data.datasets[idx].hidden : !meta.hidden;
            }
            rescaleY(ch);
          }
        },
        tooltip:{callbacks:{label:c=>`${c.dataset.label}: ${c.parsed.y}${p.unit||''}`}},
      },
      scales:{
        x:{ticks:{maxRotation:0,autoSkip:true,maxTicksLimit:12}},
        y:{beginAtZero:true}
      }
    }
  });
  // 初次 render 后按当前可见系列锁一次 max
  rescaleY(chart);
});
</script></body></html>
"""
