/* Shared multi-unit time-series panel. Input points retain original units and gaps. */
(function(root) {
  'use strict';
  function number(value) {
    return Number.isFinite(value) ? value.toLocaleString(undefined,{maximumFractionDigits:2}) : '无数据';
  }
  function mount(parent, panel, context) {
    const wrap = document.createElement('section'); wrap.className = 'panel multi-curve'; wrap.style.gridColumn = '1 / -1';
    const title = document.createElement('h3'); title.textContent = panel.title; wrap.appendChild(title);
    const caption = document.createElement('p'); caption.textContent = panel.caption; wrap.appendChild(caption);
    const toolbar = document.createElement('div'); toolbar.className='multi-toolbar'; wrap.appendChild(toolbar);
    const picker = document.createElement('div'); picker.className='multi-picker'; toolbar.appendChild(picker);
    const pickerButton = document.createElement('button'); pickerButton.type='button'; pickerButton.className='multi-picker-button'; pickerButton.setAttribute('aria-expanded','false'); picker.appendChild(pickerButton);
    const dropdown = document.createElement('div'); dropdown.className='multi-dropdown'; dropdown.hidden=true; picker.appendChild(dropdown);
    const search = document.createElement('input'); search.type='search'; search.className='multi-search'; search.placeholder='搜索指标'; search.setAttribute('aria-label','搜索指标'); dropdown.appendChild(search);
    const choices = document.createElement('div'); choices.className='multi-choices'; dropdown.appendChild(choices);
    const box = document.createElement('div'); box.className='multi-plot'; box.style.cssText='height:580px;position:relative';
    const canvas = document.createElement('canvas'); box.appendChild(canvas);
    const legend = document.createElement('div'); legend.className='multi-legend'; legend.setAttribute('aria-label','曲线图例');
    if(panel.series.some(s=>s.name.startsWith('old · ')) && panel.series.some(s=>s.name.startsWith('new · '))) {
      const key=document.createElement('span');key.className='multi-legend-key';key.textContent='old 虚线 · new 实线';legend.appendChild(key);
    }
    const hover = document.createElement('div'); hover.className='multi-hover'; hover.textContent='将鼠标移到曲线上查看样本点详情';
    wrap.appendChild(box); wrap.appendChild(legend); wrap.appendChild(hover); parent.appendChild(wrap);
    const datasets = panel.series.map(s => ({label:s.name, description:s.description||'', group:s.group||'其他',
      data:s.points || s.data.map((y,i)=>({x:panel.xNums[i],y})), yAxisID:s.axis || 'y', unit:s.unit || '',
      borderColor:s.color, backgroundColor:s.color, baseColor:s.color, borderDash:s.dash||[], hidden:!!s.hidden, borderWidth:2, pointRadius:0,
      tension:0, spanGaps:false}));
    const scales = {x:{type:'linear',min:context.timeAxis.min,max:context.timeAxis.max,title:{display:true,text:'时间 / 秒'}}};
    Object.entries(panel.axes).forEach(([key,axis]) => {
      scales[key] = {type:'linear',position:axis.position || 'left',beginAtZero:true,
        min:axis.min,max:axis.max,grid:{drawOnChartArea:key==='ratio'||key==='queue'},title:{display:true,text:axis.title}};
    });
    const events = {id:'multiCurveEvents',afterDraw(chart) {
      const {ctx,chartArea:a,scales:{x}} = chart; ctx.save(); ctx.font='11px sans-serif';
      (context.events||[]).forEach((event,i) => {const px=x.getPixelForValue(event.t); if(px<a.left || px>a.right) return;
        ctx.strokeStyle='#94a3b8';ctx.setLineDash([4,4]);ctx.beginPath();ctx.moveTo(px,a.top);ctx.lineTo(px,a.bottom);ctx.stroke();
        ctx.fillStyle='#64748b';ctx.fillText(event.name || event.label,px+3,a.top+14+(i%3)*14);});ctx.restore();
    }};
    let focused = null;
    function nearest(dataset,t) {let found=null;dataset.data.forEach(p=>{if(Number.isFinite(p.x)&&(!found||Math.abs(p.x-t)<Math.abs(found.x-t))) found=p;});return found&&Math.abs(found.x-t)<=3?found:null;}
    function fillHover(t) {
      if (hover.replaceChildren) hover.replaceChildren(); else hover.children=[];
      const heading=document.createElement('strong'); heading.textContent='t = '+t.toFixed(1)+' s'; hover.appendChild(heading);
      datasets.forEach((d,i)=>{if(!chart.isDatasetVisible(i)) return;const row=document.createElement('div');row.className='multi-hover-row';
        const swatch=document.createElement('i');swatch.style.backgroundColor=d.baseColor;row.appendChild(swatch);const p=nearest(d,t);
        row.appendChild(document.createTextNode(d.label+'  '+number(p&&p.y)+(p&&d.unit?' '+d.unit:'')));hover.appendChild(row);});
    }
    const chart = new Chart(canvas, {
      type:'line', data:{datasets}, plugins:[events],
      options:{
        animation:false, responsive:true, maintainAspectRatio:false, parsing:false,
        interaction:{mode:'nearest',axis:'x',intersect:false}, scales,
        onHover:(event,active)=>{if(active&&active.length){const a=active[0];fillHover(datasets[a.datasetIndex].data[a.index].x);}},
        plugins:{legend:{display:false},tooltip:{enabled:false,callbacks:{
          title:items=>items.length?'t = '+items[0].parsed.x.toFixed(1)+' s':'',
          label:item=>item.dataset.label+': '+number(item.parsed.y)+(item.dataset.unit?' '+item.dataset.unit:'')
        }}}
      }
    });
    const inputs=[], legendLabels=[], plottedLabels=[], groups=[]; let lastGroup=null;
    datasets.forEach((d,i)=>{if(d.group!==lastGroup){const group=document.createElement('div');group.className='multi-group';group.textContent=d.group;choices.appendChild(group);groups.push({element:group,indices:[]});lastGroup=d.group;}
      groups[groups.length-1].indices.push(i);
      const label=document.createElement('label');label.className='multi-choice';label.title=d.description;
      const swatch=document.createElement('i');swatch.style.backgroundColor=d.baseColor;label.appendChild(swatch);
      const input=document.createElement('input');input.type='checkbox';input.checked=!d.hidden;input.onchange=()=>{chart.setDatasetVisibility(i,input.checked);refresh();};label.appendChild(input);
      label.appendChild(document.createTextNode(' '+d.label));label.onmouseenter=()=>focus(i);label.onmouseleave=()=>focus(null);
      label.ondblclick=()=>{const isolated=datasets.every((_,j)=>chart.isDatasetVisible(j)===(j===i));datasets.forEach((_,j)=>chart.setDatasetVisibility(j,isolated||j===i));refresh();};
      choices.appendChild(label);inputs.push(input);legendLabels.push(label);
      const item=document.createElement('button');item.type='button';item.className='multi-legend-item';item.title=d.description;
      const line=document.createElement('i');line.style.borderTopColor=d.baseColor;line.style.borderTopStyle=d.borderDash.length?'dashed':'solid';item.appendChild(line);
      item.appendChild(document.createTextNode(d.label));item.onclick=()=>{chart.setDatasetVisibility(i,!chart.isDatasetVisible(i));refresh();};
      item.ondblclick=()=>{const isolated=datasets.every((_,j)=>chart.isDatasetVisible(j)===(j===i));datasets.forEach((_,j)=>chart.setDatasetVisibility(j,isolated||j===i));refresh();};
      item.onmouseenter=()=>focus(i);item.onmouseleave=()=>focus(null);
      legend.appendChild(item);plottedLabels.push(item);});
    function filterChoices() {const query=search.value.trim().toLocaleLowerCase();legendLabels.forEach((label,i)=>{label.hidden=!!query&&!datasets[i].label.toLocaleLowerCase().includes(query);});groups.forEach(group=>{group.element.hidden=group.indices.every(i=>legendLabels[i].hidden);});}
    search.oninput=filterChoices;
    function setPickerOpen(open) {dropdown.hidden=!open;pickerButton.setAttribute('aria-expanded',String(open));if(open) search.focus();}
    pickerButton.onclick=()=>setPickerOpen(dropdown.hidden);
    picker.onkeydown=event=>{if(event.key==='Escape') {setPickerOpen(false);pickerButton.focus();}};
    document.addEventListener('click',event=>{if(!picker.contains(event.target)) setPickerOpen(false);});
    function focus(index) {focused=index;datasets.forEach((d,i)=>{d.borderWidth=index===null?2:(i===index?4:1);d.borderColor=index===null||i===index?d.baseColor:d.baseColor+'33';});[legendLabels,plottedLabels].forEach(labels=>labels.forEach((label,i)=>{label.style.opacity=index===null||i===index?'1':'0.38';}));chart.update('none');}
    function refresh() {Object.keys(panel.axes).forEach(axis=>{chart.options.scales[axis].display=datasets.some((d,i)=>d.yAxisID===axis&&chart.isDatasetVisible(i));});inputs.forEach((input,i)=>{input.checked=chart.isDatasetVisible(i);plottedLabels[i].hidden=!input.checked;});legend.hidden=!inputs.some(input=>input.checked);pickerButton.textContent='指标 '+inputs.filter(input=>input.checked).length+'/'+inputs.length+' ▾';if(focused!==null) focus(focused);else chart.update('none');}
    function button(label,action) {const b=document.createElement('button');b.textContent=label;b.onclick=action;toolbar.appendChild(b);}
    Object.entries(panel.presets || {}).forEach(([name,selected])=>button(name,()=>{datasets.forEach((d,i)=>chart.setDatasetVisibility(i,selected.includes(d.label)));refresh();}));
    button('全选',()=>{datasets.forEach((d,i)=>chart.setDatasetVisibility(i,true));refresh();});button('清空',()=>{datasets.forEach((d,i)=>chart.setDatasetVisibility(i,false));refresh();});
    const range=document.createElement('div');range.className='multi-range';const from=document.createElement('input'),to=document.createElement('input');for(const el of [from,to]){el.type='number';el.step='1';el.style.width='90px';}
    from.value=context.timeAxis.min;to.value=Math.ceil(context.timeAxis.max);range.append('时间范围（秒） ',from,' — ',to);toolbar.appendChild(range);
    function applyRange(){const lo=Number(from.value),hi=Number(to.value);if(Number.isFinite(lo)&&Number.isFinite(hi)&&hi>lo){chart.options.scales.x.min=lo;chart.options.scales.x.max=hi;chart.update('none');}}
    from.onchange=to.onchange=applyRange;button('重置时间',()=>{from.value=context.timeAxis.min;to.value=Math.ceil(context.timeAxis.max);applyRange();});refresh();return chart;
  }
  root.FlexMultiCurve={mount};
})(typeof window === 'undefined' ? globalThis : window);
