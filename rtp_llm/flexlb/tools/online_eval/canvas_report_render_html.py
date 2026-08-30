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
    'subtitle': Optional[str],    # 副标（run 概览：规模 / 发送量 / 采样说明）
    'kpis':     [{'label': str, 'value': str, 'tone': Optional[str]}],
    'meta':     Optional[{        # 头部元数据面板（KPI 行下方三分区）
        'sources': {'runDir': str, 'aggregate': str,
                    'summary': Optional[str], 'engineDist': Optional[str]},
        'scale':   {'p': num, 'd': num, 'shards': num,
                    'replay': num, 'durationS': Optional[num]},
        'timeAxis': Optional[{'tEnd': num}],   # T_END 动态填入口径文案
    }],
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

元数据面板：meta 三分区（数据源 / 规模 / 时间轴口径）渲染为 KPI 行下方的
浅色信息面板；路径用等宽字体、overflow-wrap:anywhere 保留完整可复制。
旧 spec 无 meta 键时整个面板不渲染（向后兼容）。

时间轴语义：timeAxis.min = 0（t=0 = 压测正式开始，warmup 后）；
timeAxis.max = T_END（全部时序面板最后采样点，ceil 整秒，含收尾排空）。
timeX 面板数据点转 {x, y}，scales.x = linear + min/max 钉住；warmup
负值段被轴裁剪（数据保留不删）。非时间轴面板保持类目轴不变。
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
        "timeAxis": spec.get("timeAxis"),
        "meta": spec.get("meta"),
        "panels": [
            {
                "id": p["id"],
                "title": p.get("title", ""),
                "caption": p.get("caption", ""),
                "type": p.get("type", "line"),
                "x": p.get("x", []),
                "timeX": bool(p.get("timeX")),
                "xNums": p.get("xNums") or [],
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
.kpi-row{display:grid;grid-template-columns:repeat(6,1fr);gap:12px;margin:16px 0 12px}
.kpi{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:12px 14px}
.kpi .v{font-size:22px;font-weight:600}
.kpi .l{color:var(--sub);font-size:12px;margin-top:4px}
.kpi.success .v{color:var(--success)} .kpi.danger .v{color:var(--danger)} .kpi.warn .v{color:var(--warn)}
/* 头部元数据面板：KPI 行下方三分区（数据源 / 规模 / 时间轴口径）。
   长路径等宽字体 + overflow-wrap:anywhere：可折行但保留完整可复制。 */
.meta-panel{background:var(--card);border:1px solid var(--border);border-radius:8px;
  margin:0 0 24px;display:grid;
  grid-template-columns:minmax(300px,1.5fr) minmax(170px,.75fr) minmax(290px,1.15fr);
  font-size:12px;line-height:1.7;color:var(--sub)}
.meta-sec{padding:12px 16px;min-width:0}
.meta-sec+.meta-sec{border-left:1px solid var(--border)}
.meta-sec h4{margin:0 0 6px;font-size:11px;font-weight:600;letter-spacing:.6px;
  text-transform:uppercase;color:rgba(0,0,0,.38)}
.meta-row{display:flex;gap:8px;align-items:baseline;margin:1px 0}
.meta-row .k{flex:none;white-space:nowrap;color:rgba(0,0,0,.45)}
.meta-row .v{min-width:0;overflow-wrap:anywhere;word-break:break-word;
  color:rgba(0,0,0,.72);
  font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,"Liberation Mono",monospace;
  font-size:11.5px}
.meta-scale{display:flex;flex-wrap:wrap;gap:6px;align-content:flex-start}
.meta-chip{background:rgba(22,119,255,.06);border:1px solid rgba(22,119,255,.18);
  border-radius:4px;padding:1px 8px;color:rgba(0,0,0,.72);white-space:nowrap;
  font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,"Liberation Mono",monospace;
  font-size:11.5px}
.meta-chip b{font-weight:600;color:#1677ff}
.meta-ta p{margin:0;color:rgba(0,0,0,.6)}
.meta-ta b{font-weight:600;color:#1677ff;
  font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,"Liberation Mono",monospace}
@media(max-width:960px){
  .meta-panel{grid-template-columns:1fr}
  .meta-sec+.meta-sec{border-left:none;border-top:1px solid var(--border)}
}
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
<div class="meta-panel" id="meta"></div>
<div class="grid" id="grid"></div>
<div class="hint" id="hint">Chart.js 4.4.7 · legend 单击可切换单条系列，双击隔离；tooltip 随鼠标移动；隐藏后 y 轴按剩余可见系列自适应。</div>
<script>
const SPEC = __SPEC_JSON__;
document.getElementById('title').textContent = SPEC.summary.title;
document.getElementById('subtitle').textContent = SPEC.summary.subtitle;
const kb = document.getElementById('kpis');
// KPI 列数随实际 chip 数自适应（缺省 CSS 6 列仅为无 JS 兼容底座）。
if (SPEC.summary.kpis.length)
  kb.style.gridTemplateColumns = 'repeat(' + SPEC.summary.kpis.length + ',1fr)';
SPEC.summary.kpis.forEach(k=>{
  const d=document.createElement('div'); d.className='kpi '+(k.tone||'');
  d.innerHTML=`<div class="v">${k.value}</div><div class="l">${k.label}</div>`;
  kb.appendChild(d);
});
// 头部元数据面板：数据源 / 规模 / 时间轴口径（SPEC.meta，生成器注入；
// 旧 spec 无 meta 时整个面板移除，向后兼容）。
(function renderMeta(){
  const meta = SPEC.meta;
  const host = document.getElementById('meta');
  if (!meta){ host.remove(); return; }
  const div = (cls)=>{ const el=document.createElement('div'); el.className=cls; return el; };
  const h4 = (t)=>{ const el=document.createElement('h4'); el.textContent=t; return el; };
  const addRow = (sec,k,v)=>{
    const r=div('meta-row');
    const kk=document.createElement('span'); kk.className='k'; kk.textContent=k; r.appendChild(kk);
    const vv=document.createElement('span'); vv.className='v';
    vv.textContent=(v==null||v==='')?'—（未加载）':String(v); r.appendChild(vv);
    sec.appendChild(r);
  };
  // —— 分区一：数据源（绝对路径；engine_dist 内嵌于 aggregate 时标注）——
  const so = meta.sources||{};
  const s1 = document.createElement('section'); s1.className='meta-sec';
  s1.appendChild(h4('数据源'));
  addRow(s1,'aggregate',so.aggregate);
  addRow(s1,'summary',so.summary);
  addRow(s1,'engine_dist',so.engineDist);
  addRow(s1,'run 目录',so.runDir);
  host.appendChild(s1);
  // —— 分区二：规模（P / D / shards / replay / duration）——
  const sc = meta.scale||{};
  const s2 = document.createElement('section'); s2.className='meta-sec';
  s2.appendChild(h4('规模'));
  const wrap = div('meta-scale');
  const chip=(k,v,suf)=>{ if(v==null||v==='') return;
    const c=div('meta-chip'); c.appendChild(document.createTextNode(k+'='));
    const b=document.createElement('b'); b.textContent=String(v)+(suf||''); c.appendChild(b);
    wrap.appendChild(c); };
  chip('P',sc.p); chip('D',sc.d); chip('shards',sc.shards); chip('replay',sc.replay);
  chip('duration',sc.durationS,'s');
  if(!wrap.childNodes.length) wrap.textContent='—';
  s2.appendChild(wrap); host.appendChild(s2);
  // —— 分区三：时间轴口径（T_END 动态值）——
  const s3 = document.createElement('section'); s3.className='meta-sec';
  s3.appendChild(h4('时间轴口径'));
  const box = div('meta-ta');
  const t = meta.timeAxis;
  if (t && typeof t.tEnd==='number' && t.tEnd>0){
    const line=(html)=>{ const p=document.createElement('p'); p.innerHTML=html; box.appendChild(p); };
    line('t=0 = 压测正式开始（warmup 后）');
    line('T_END=<b>'+t.tEnd+'s</b> 含收尾排空');
    line('全部时序面板统一 <b>[0, '+t.tEnd+']</b>');
    line('warmup 负值段被轴裁剪（数据保留）');
  } else {
    box.textContent='无统一时间轴（报告不含时序面板）';
  }
  s3.appendChild(box); host.appendChild(s3);
})();
// y 轴自适应：只按 legend 可见系列重算 max（beginAtZero），legend/双击后 update()。
// 时间轴面板数据点为 {x, y}（linear x 轴），此处兼容两种形态。
// 双击隔离用手写 handler：默认 Chart.js 只支持单击切换单条，双击这里定义为"只留这条"，
// 再双击同一条恢复"全显"。
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
  chart.options.scales.y.max = visibleMax(chart);
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
  const wrap=document.createElement('div'); wrap.className='panel';
  wrap.innerHTML=`<h3>${p.title}</h3><div class="cap">${p.caption}</div><div class="box"><canvas id="c-${p.id}"></canvas></div>`;
  grid.appendChild(wrap);
  const ctx=wrap.querySelector('canvas').getContext('2d');
  // 时间轴面板：数据点转 {x, y}，linear x 轴钉 [TA_MIN, TA_MAX]（warmup
  // 负值段被轴裁剪，数据保留）；tooltip 按 x 最近点联动。非时间轴面板
  // 保持类目轴 + index 联动。
  const isTime = !!(p.timeX && TIME_AXIS && p.xNums && p.xNums.length);
  const chart = new Chart(ctx,{
    type: p.type==='bar'?'bar':'line',
    data:{
      labels: isTime ? undefined : p.x,
      datasets:p.series.map(s=>({
        label:s.name,
        data: isTime ? s.data.map((v,i)=>({x:p.xNums[i], y:v})) : s.data,
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
  // 初次 render 后按当前可见系列锁一次 max
  rescaleY(chart);
});
if (TIME_AXIS){
  // 页脚时间轴口径声明（与头部元数据面板标注一致，防止报告被断章取义）
  const note=document.createElement('div');
  note.textContent='时间轴口径：t=0 = 压测正式开始（warmup 后）；T_END='+TA_MAX+'s = 全部时序面板最后采样点（含收尾排空）；全部时序面板 x 轴统一 [0, '+TA_MAX+']，warmup 负值段被轴裁剪（数据保留）。';
  document.getElementById('hint').appendChild(note);
}
</script></body></html>
"""
