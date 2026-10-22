/* Page assembly and non-temporal charts; input is the frozen rendered payload. */
(function(root) {
  'use strict';
  function axisOptions(axes, grid) {
    return Object.fromEntries(Object.entries(axes).map(([key,axis]) => [key, {
      type:'linear', position:axis.position||'left', beginAtZero:true,
      min:axis.min, max:axis.max,
      ...(grid ? {grid:{drawOnChartArea:grid(key)}} : {}),
      title:{display:true,text:axis.title||''}
    }]));
  }
  function mount(SPEC) {
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
// Legend interactions rescale each declared axis using visible points only.
function visibleMax(chart,axis){
  let max=0;
  chart.data.datasets.forEach((dataset,i)=>{
    if(!chart.isDatasetVisible(i)||dataset.yAxisID!==axis)return;
    dataset.data.forEach(point=>{if(Number.isFinite(point.y))max=Math.max(max,point.y);});
  });
  return max>0 ? max*1.05 : undefined;
}
function rescaleY(chart){
  Object.entries(chart.options.scales).forEach(([axis,settings])=>{
    if(axis==='x')return;
    settings.max=chart.$reportYBounds ? chart.$reportYBounds[1] : visibleMax(chart,axis);
    if(chart.$reportYBounds)settings.min=chart.$reportYBounds[0];
  });
  chart.update('none');
}
const grid = document.getElementById('grid');
// Without a declared display window, use the points' domain. Never synthesize samples.
const pointTimes = SPEC.panels.filter(p=>p.timeX).flatMap(p=>p.series.flatMap(s=>s.points.map(p=>p.x)));
let domainMin=0,domainMax=1;
pointTimes.forEach(t=>{domainMin=Math.min(domainMin,t);domainMax=Math.max(domainMax,t);});
const TIME_AXIS = SPEC.timeAxis || {min:domainMin,max:domainMax};
const TA_MIN = TIME_AXIS ? TIME_AXIS.min : undefined;
const TA_MAX = TIME_AXIS ? TIME_AXIS.max : undefined;
SPEC.panels.forEach(p=>{
  if (p.timeX && p.type === 'line') { FlexMultiCurve.mount(grid, p, {timeAxis:p.timeAxis || TIME_AXIS, selectionKey:p.timeSelectionKey, events:p.events || []}); return; }
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

  // Time and category charts both consume the producer-owned points.
  const isTime = !!(p.timeX && TIME_AXIS);
  const numericX = p.type !== 'bar' || p.series.every(s=>s.points.every(point=>typeof point.x==='number'));
  let legendController;
  const chart = new Chart(ctx,{
    plugins: [{id:'phaseEvents', afterDraw(chart) {
      if (!isTime) return;
      const {ctx, chartArea:a, scales:{x}} = chart;
      ctx.save(); ctx.font='10px sans-serif';
      (p.events || []).forEach((e,i) => {
        if (!Number.isFinite(e.t)) return;
        const px=x.getPixelForValue(e.t);
        if(px<a.left || px>a.right) return;
        ctx.strokeStyle='__REPORT_EVENT_LINE__'; ctx.setLineDash([3,3]);
        ctx.beginPath(); ctx.moveTo(px,a.top); ctx.lineTo(px,a.bottom); ctx.stroke();
        ctx.fillStyle='__REPORT_EVENT_TEXT__'; ctx.fillText(e.name || e.label || '',px+2,a.top+12+(i%3)*12);
      });
      ctx.restore();
    }}],
    type: p.type==='bar'?'bar':'line',
    data:{
      datasets:p.series.map(s=>({
        label:s.name,
        data: s.points,
        yAxisID:s.axis, hidden:s.hidden, borderDash:s.dash,
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
      scales: Object.assign({x:isTime ? {
        type:'linear',min:TA_MIN,max:TA_MAX,ticks:{maxRotation:0,autoSkip:true,maxTicksLimit:12}
      } : {type:numericX?'linear':'category',ticks:{maxRotation:0,autoSkip:true,maxTicksLimit:12}}},
      axisOptions(p.axes))
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
    btn.style.cssText='position:absolute;top:34px;right:10px;font-size:10px;color:__REPORT_MUTED_TEXT__;cursor:pointer;padding:2px 6px;border:1px solid __REPORT_LIGHT_BORDER__;border-radius:10px;background:rgba(255,255,255,.9);z-index:5;user-select:none;transition:all .15s';
    btn.onmouseenter=()=>{btn.style.color='__REPORT_DARK_TEXT__';btn.style.borderColor='__REPORT_HOVER_BORDER__';};
    btn.onmouseleave=()=>{btn.style.color='__REPORT_MUTED_TEXT__';btn.style.borderColor='__REPORT_LIGHT_BORDER__';};
    btn.onclick=()=>legendController.all();
    wrap.style.position='relative';
    wrap.appendChild(btn);
  }
});
  }
  root.FlexReport = {mount, axisOptions};
})(typeof window === 'undefined' ? globalThis : window);
