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
    'subtitle': Optional[str],    # 副标（实验条件行：拓扑 / 发送模式倍率 /
                                   # ramp / duration / shards，三层第一层）
    'kpis':     [{'label': str, 'value': str, 'tone': Optional[str]}],
                                   # 两行×最多 5 chip：指标五连 + 结果五连
                                   # （请求数量 / 成功 / 失败·cancel / 成功率 /
                                   # 持续时间），三层第二层
    'meta':     Optional[{        # 可见口径面板 + detail 折叠层（三层第三层）
        'timeAxis': Optional[{'tEnd': num}],   # T_END 动态填入口径文案（可见）
        'sampling': Optional[str],             # 采样说明（可见）
        'version':  {'branch': Opt[str], 'commit': Opt[str]},      # detail
        'dataset':  {'traceFile': Opt[str], 'traceLines': Opt[num],
                     'traceSha256': Opt[str]},                      # detail
        'params':   Optional[dict],  # run_meta.params 全量（detail）
        'env':      {'clientEnv': Opt[dict], 'flexlbEnv': Opt[dict]},# detail
        'sources': {'runDir': str, 'aggregate': str,
                    'engineDist': Optional[str]},                    # detail
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

头部三层（2026-09 规范化）：subtitle = 实验条件行（拓扑/倍率/ramp/
duration/shards 等全部规模信息在此承担）；KPI 两行 = 指标五连 +
结果五连；可见 meta 面板只留时间轴口径 + 采样说明（口径标注纪律），
其余（代码版本 / 数据集 / 实验参数 / FINAL ENV / 数据源）收进
<details id="detail"> 折叠块默认收起；规模不设分区（与 subtitle 重复，
已删）。路径用等宽字体、overflow-wrap:anywhere 保留完整可复制。
旧 spec 无 meta 键时两个面板均不渲染（向后兼容）。

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
/* KPI 两行（指标五连 + 结果五连）：wrapper 纵向叠行，每行 grid 随
   行内 chip 数自适应列数（JS 注入 inline grid-template-columns）。 */
.kpi-stack{display:flex;flex-direction:column;gap:12px;margin:16px 0 12px}
.kpi-row{display:grid;grid-template-columns:repeat(6,1fr);gap:12px}
.kpi{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:12px 14px}
.kpi .v{font-size:22px;font-weight:600}
.kpi .l{color:var(--sub);font-size:12px;margin-top:4px}
.kpi.success .v{color:var(--success)} .kpi.danger .v{color:var(--danger)} .kpi.warn .v{color:var(--warn)}
/* 可见口径面板（三层第三层之可见部分）：时间轴口径 + 采样说明。 */
.meta-panel{background:var(--card);border:1px solid var(--border);border-radius:8px;
  margin:0 0 12px;font-size:12px;line-height:1.7;color:var(--sub)}
.meta-sec{padding:12px 16px;min-width:0}
.meta-sec h4{margin:0 0 6px;font-size:11px;font-weight:600;letter-spacing:.6px;
  text-transform:uppercase;color:rgba(0,0,0,.38)}
.meta-row{display:flex;gap:8px;align-items:baseline;margin:1px 0}
.meta-row .k{flex:none;white-space:nowrap;color:rgba(0,0,0,.45)}
.meta-row .v{min-width:0;overflow-wrap:anywhere;word-break:break-word;
  color:rgba(0,0,0,.72);
  font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,"Liberation Mono",monospace;
  font-size:11.5px}
.meta-ta p{margin:0;color:rgba(0,0,0,.6)}
.meta-ta b{font-weight:600;color:#1677ff;
  font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,"Liberation Mono",monospace}
/* detail 折叠层（三层第三层）：代码版本 / 数据集 / 参数 / FINAL ENV /
   数据源，默认收起；长表（params / env）限高滚动防长屏。 */
.detail-panel{background:var(--card);border:1px solid var(--border);border-radius:8px;
  margin:0 0 24px;font-size:12px;color:var(--sub)}
.detail-panel>summary{cursor:pointer;padding:10px 16px;font-size:12px;
  font-weight:600;letter-spacing:.4px;color:rgba(0,0,0,.55);
  user-select:none;list-style:none}
.detail-panel>summary::-webkit-details-marker{display:none}
.detail-panel>summary::before{content:'▸';display:inline-block;margin-right:8px;
  transition:transform .15s;color:rgba(0,0,0,.35)}
.detail-panel[open]>summary::before{transform:rotate(90deg)}
.detail-body{border-top:1px solid var(--border);padding:12px 16px 16px;
  display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));
  gap:14px 28px}
.detail-sec{min-width:0}
.detail-wide{grid-column:1/-1}
.detail-sec h4{margin:0 0 6px;font-size:11px;font-weight:600;letter-spacing:.6px;
  text-transform:uppercase;color:rgba(0,0,0,.38)}
.detail-env{max-height:280px;overflow:auto;border:1px solid var(--border);
  border-radius:6px;padding:6px 10px;background:rgba(0,0,0,.015)}
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
<div class="kpi-stack" id="kpis"></div>
<div class="meta-panel" id="meta"></div>
<details id="detail" class="detail-panel">
  <summary>实验详情 · 代码版本 / 数据集 / 参数 / 环境变量 / 数据源</summary>
  <div class="detail-body" id="detail-body"></div>
</details>
<div class="grid" id="grid"></div>
<div class="hint" id="hint">Chart.js 4.4.7 · legend 单击可切换单条系列，双击隔离；tooltip 随鼠标移动；隐藏后 y 轴按剩余可见系列自适应。</div>
<script>
const SPEC = __SPEC_JSON__;
document.getElementById('title').textContent = SPEC.summary.title;
document.getElementById('subtitle').textContent = SPEC.summary.subtitle;
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
      d.innerHTML=`<div class="v">${k.value}</div><div class="l">${k.label}</div>`;
      row.appendChild(d);
    });
    kb.appendChild(row);
  }
// 可见口径面板：时间轴口径 + 采样说明（口径标注纪律：必须直观可读；
// 数据源/版本等溯源性质信息下沉 detail 折叠层，scale 由 subtitle 承担）。
(function renderMeta(){
  const meta = SPEC.meta;
  const host = document.getElementById('meta');
  if (!meta){ host.remove(); return; }
  const div = (cls)=>{ const el=document.createElement('div'); el.className=cls; return el; };
  const h4 = (t)=>{ const el=document.createElement('h4'); el.textContent=t; return el; };
  const sec = div('meta-sec');
  sec.appendChild(h4('时间轴口径'));
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
  if (meta.sampling){
    const p=document.createElement('p'); p.textContent=meta.sampling; box.appendChild(p);
  }
  sec.appendChild(box); host.appendChild(sec);
})();
// detail 折叠层（三层第三层，默认收起）：代码版本 / 测试数据集 / 数据源 /
// 实验参数全量 / FINAL ENV（client_env + flexlb_env）；scale 分区已删——
// 与 subtitle 实验条件重复（20260902）。
(function renderDetail(){
  const meta = SPEC.meta;
  const host = document.getElementById('detail');
  const body = document.getElementById('detail-body');
  if (!meta || !body){ if(host) host.remove(); return; }
  const div = (cls)=>{ const el=document.createElement('div'); el.className=cls; return el; };
  const h4 = (t)=>{ const el=document.createElement('h4'); el.textContent=t; return el; };
  const addRow = (sec,k,v)=>{
    const r=div('meta-row');
    const kk=document.createElement('span'); kk.className='k'; kk.textContent=k; r.appendChild(kk);
    const vv=document.createElement('span'); vv.className='v';
    vv.textContent=(v==null||v==='')?'—（未提供）':String(v); r.appendChild(vv);
    sec.appendChild(r);
  };
  // —— 分区一：代码版本（branch / commit；远端 rsync 树无 .git，由重聚合
  //    命令经 FLEXLB_GIT_BRANCH/FLEXLB_GIT_COMMIT 注入 aggregate meta）——
  const ver = meta.version||{};
  const s1 = div('detail-sec'); s1.appendChild(h4('代码版本'));
  addRow(s1,'branch',ver.branch); addRow(s1,'commit',ver.commit);
  body.appendChild(s1);
  // —— 分区二：测试数据集（trace 路径 / 行数 / sha256）——
  const ds = meta.dataset||{};
  const s2 = div('detail-sec'); s2.appendChild(h4('测试数据集'));
  addRow(s2,'trace',ds.traceFile);
  addRow(s2,'行数',ds.traceLines);
  addRow(s2,'sha256',ds.traceSha256);
  body.appendChild(s2);
  // —— 分区三：数据源（绝对路径；engine_dist 内嵌于 aggregate 时标注）——
  const so = meta.sources||{};
  const s3 = div('detail-sec'); s3.appendChild(h4('数据源'));
  addRow(s3,'aggregate',so.aggregate);
  addRow(s3,'engine_dist',so.engineDist);
  addRow(s3,'run 目录',so.runDir);
  body.appendChild(s3);
  // —— 分区四：实验参数全量（run_meta.params：拓扑/端口/容量/JVM/配置
  //    文件路径等；键排序 + 限高滚动，flexlb_config 长值完整保留）——
  const s4 = div('detail-sec detail-wide'); s4.appendChild(h4('实验参数（run_meta.params）'));
  const params = meta.params;
  if (params && Object.keys(params).length){
    const envBox = div('detail-env');
    Object.keys(params).sort().forEach(k=>{
      const r=div('meta-row');
      const kk=document.createElement('span'); kk.className='k'; kk.textContent=k;
      const vv=document.createElement('span'); vv.className='v';
      vv.textContent=String(params[k]);
      r.appendChild(kk); r.appendChild(vv); envBox.appendChild(r);
    });
    s4.appendChild(envBox);
  } else {
    const r=div('meta-row'); r.textContent='—（未提供）'; s4.appendChild(r);
  }
  body.appendChild(s4);
  // —— 分区五：环境变量 FINAL ENV（JavaLoadClient client_env + FlexLB
  //    flexlb_env 快照，consolidate 阶段嵌入 run_meta.json）——
  const env = meta.env||{};
  const s5 = div('detail-sec detail-wide'); s5.appendChild(h4('环境变量（FINAL ENV）'));
  const envRender=(title,data)=>{
    if (!data || !Object.keys(data).length) return;
    const sub=div('detail-env');
    const st=document.createElement('div'); st.className='meta-row';
    const sk=document.createElement('span'); sk.className='k'; sk.textContent=title;
    st.appendChild(sk); sub.appendChild(st);
    Object.keys(data).sort().forEach(k=>{
      const r=div('meta-row');
      const kk=document.createElement('span'); kk.className='k'; kk.textContent=k;
      const vv=document.createElement('span'); vv.className='v';
      vv.textContent=String(data[k]);
      r.appendChild(kk); r.appendChild(vv); sub.appendChild(r);
    });
    s5.appendChild(sub);
  };
  envRender('JavaLoadClient（client_env.json）', env.clientEnv);
  envRender('FlexLB（flexlb_env.txt）', env.flexlbEnv);
  if (!s5.querySelector('.detail-env')){
    const r=div('meta-row'); r.textContent='—（未提供）'; s5.appendChild(r);
  }
  body.appendChild(s5);
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
