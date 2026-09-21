/* Shared multi-unit time-series panel. Input points retain original units and gaps. */
(function(root) {
  'use strict';
  function mount(parent, panel, context) {
    const wrap = document.createElement('section'); wrap.className = 'panel';
    wrap.style.gridColumn = '1 / -1';
    const title = document.createElement('h3'); title.textContent = panel.title; wrap.appendChild(title);
    const caption = document.createElement('p'); caption.textContent = panel.caption; wrap.appendChild(caption);
    const toolbar = document.createElement('div'); wrap.appendChild(toolbar);
    const choices = document.createElement('div'); wrap.appendChild(choices);
    const box = document.createElement('div'); box.style.height = '580px';
    const canvas = document.createElement('canvas'); box.appendChild(canvas); wrap.appendChild(box); parent.appendChild(wrap);
    const datasets = panel.series.map(s => ({label:s.name, data:s.points || s.data.map((y,i)=>({x:panel.xNums[i],y})),
      yAxisID:s.axis || 'y', unit:s.unit || '', borderColor:s.color, backgroundColor:s.color,
      hidden:!!s.hidden, borderWidth:1.7, pointRadius:0, tension:0, spanGaps:false}));
    const scales = {x:{type:'linear',min:context.timeAxis.min,max:context.timeAxis.max,title:{display:true,text:'时间 / 秒'}}};
    Object.entries(panel.axes).forEach(([key,axis]) => {
      scales[key] = {type:'linear',position:axis.position || 'left',beginAtZero:true,
        min:axis.min,max:axis.max,grid:{drawOnChartArea:key==='pct'},title:{display:true,text:axis.title}};
    });
    const events = {id:'multiCurveEvents',afterDraw(chart) {
      const {ctx,chartArea:a,scales:{x}} = chart;
      ctx.save(); ctx.font='11px sans-serif';
      context.events.forEach((event,i) => {
        const px=x.getPixelForValue(event.t); if(px<a.left || px>a.right) return;
        ctx.strokeStyle='#94a3b8';ctx.setLineDash([4,4]);ctx.beginPath();ctx.moveTo(px,a.top);ctx.lineTo(px,a.bottom);ctx.stroke();
        ctx.fillStyle='#64748b';ctx.fillText(event.name || event.label,px+3,a.top+14+(i%3)*14);
      });ctx.restore();
    }};
    const chart = new Chart(canvas,{type:'line',data:{datasets},plugins:[events],options:{
      animation:false,responsive:true,maintainAspectRatio:false,parsing:false,
      interaction:{mode:'nearest',axis:'x',intersect:false},scales,
      plugins:{legend:{display:false},tooltip:{callbacks:{title:items=>items.length?'t='+items[0].parsed.x.toFixed(1)+' s':'',
        label:()=>'',
        afterBody:items=>{
          if(!items.length) return [];
          const t=items[0].parsed.x;
          return datasets.flatMap((d,i)=>{
            if(!chart.isDatasetVisible(i)) return [];
            let nearest=null;
            for(const p of d.data) if(!nearest || Math.abs(p.x-t)<Math.abs(nearest.x-t)) nearest=p;
            const value=nearest && Math.abs(nearest.x-t)<=3 && Number.isFinite(nearest.y)
              ? nearest.y.toLocaleString(undefined,{maximumFractionDigits:2})+' '+d.unit : '无数据';
            return [d.label+': '+value];
          });
        }}}}}});
    const inputs = datasets.map((d,i) => {
      const label=document.createElement('label');label.style.cssText='display:inline-block;margin:5px 12px 5px 0;cursor:pointer;color:'+d.borderColor;
      const input=document.createElement('input');input.type='checkbox';input.checked=!d.hidden;
      input.onchange=()=>{chart.setDatasetVisibility(i,input.checked);refresh();};
      label.appendChild(input);label.appendChild(document.createTextNode(' '+d.label));choices.appendChild(label);return input;
    });
    function refresh() {
      Object.keys(panel.axes).forEach(axis=>{chart.options.scales[axis].display=datasets.some((d,i)=>d.yAxisID===axis&&chart.isDatasetVisible(i));});
      inputs.forEach((input,i)=>{input.checked=chart.isDatasetVisible(i);});chart.update('none');
    }
    function button(label,action) {const b=document.createElement('button');b.textContent=label;b.style.cssText='margin:4px;padding:5px 10px;cursor:pointer';b.onclick=action;toolbar.appendChild(b);}
    Object.entries(panel.presets || {}).forEach(([name,selected])=>button(name,()=>{
      datasets.forEach((d,i)=>chart.setDatasetVisibility(i,selected.includes(d.label)));refresh();
    }));
    button('全选',()=>{datasets.forEach((d,i)=>chart.setDatasetVisibility(i,true));refresh();});
    button('清空',()=>{datasets.forEach((d,i)=>chart.setDatasetVisibility(i,false));refresh();});
    const range=document.createElement('div');range.style.margin='8px 0';
    const from=document.createElement('input'),to=document.createElement('input');
    for(const el of [from,to]){el.type='number';el.step='1';el.style.width='90px';}
    from.value=context.timeAxis.min;to.value=Math.ceil(context.timeAxis.max);
    range.append('时间范围（秒） ',from,' — ',to);toolbar.appendChild(range);
    function applyRange(){const lo=Number(from.value),hi=Number(to.value);if(Number.isFinite(lo)&&Number.isFinite(hi)&&hi>lo){chart.options.scales.x.min=lo;chart.options.scales.x.max=hi;chart.update('none');}}
    from.onchange=to.onchange=applyRange;
    button('重置时间',()=>{from.value=context.timeAxis.min;to.value=Math.ceil(context.timeAxis.max);applyRange();});
    refresh();return chart;
  }
  root.FlexMultiCurve={mount};
})(typeof window === 'undefined' ? globalThis : window);
