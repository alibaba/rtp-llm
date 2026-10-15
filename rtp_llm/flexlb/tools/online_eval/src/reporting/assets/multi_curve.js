/* Shared multi-unit time-series panel. Input points retain original units and gaps. */
(function(root) {
  'use strict';
  function number(value) {
    return Number.isFinite(value) ? value.toLocaleString(undefined,{maximumFractionDigits:2}) : '无数据';
  }
  function statistics(points, samples) {
    const source=samples || {times:points.map(p=>p.x),values:points.map(p=>p.y)};
    const order=source.times.map((_,i)=>i).filter(i=>Number.isFinite(source.times[i])).sort((a,b)=>source.times[a]-source.times[b]);
    const times=order.map(i=>source.times[i]),sums=[0],counts=[0];
    order.forEach(i=>{const value=source.values[i],valid=Number.isFinite(value);sums.push(sums[sums.length-1]+(valid?value:0));counts.push(counts[counts.length-1]+(valid?1:0));});
    function bound(t,inclusive) {let lo=0,hi=times.length;while(lo<hi){const mid=(lo+hi)>>1;if(times[mid]<t||(inclusive&&times[mid]===t))lo=mid+1;else hi=mid;}return lo;}
    return (lo,hi)=>{const a=bound(lo,false),b=bound(hi,true),n=counts[b]-counts[a];return {mean:n?(sums[b]-sums[a])/n:null,count:n,total:b-a};};
  }
  function presetColor(names, series) {
    const selected=series.filter(s=>names.includes(s.name));
    if(!selected.length)return null;
    const first=selected[0],color=first.metric_color||first.color;
    if(!color)return null;
    return first.metric_id&&selected.every(s=>s.metric_id===first.metric_id&&s.role===first.role
      &&(s.metric_color||s.color)===color)?color:null;
  }
  // Pack display labels only. Event times and semantic merging remain unchanged.
  function eventLabels(events, pixel, measure, left, right) {
    const gap=6, padding=10, width=right-left;
    if(width<=0)return [];
    const groups=events.map((event,index)=>({event,index,px:pixel(event.t)}))
      .filter(item=>Number.isFinite(item.px)&&item.px>=left&&item.px<=right)
      .sort((a,b)=>a.px-b.px||a.index-b.index).map(item=>({items:[item]}));
    function compact(group){
      group.label='E'+(group.items[0].index+1)+(group.items.length>1?' +'+(group.items.length-1):'');
      if(group.items.length>1&&group.items.every((item,i)=>!i||item.index===group.items[i-1].index+1))group.label='E'+(group.items[0].index+1)+'…E'+(group.items[group.items.length-1].index+1);
      group.width=Math.min(width,measure(group.label)+padding);
      group.left=Math.max(left,Math.min(right-group.width,group.items[0].px+3));
    }
    const packed=[];
    groups.forEach(group=>{compact(group);
      while(packed.length&&packed[packed.length-1].left+packed[packed.length-1].width+gap>group.left){
        group.items=packed.pop().items.concat(group.items);compact(group);
      }
      packed.push(group);
    });
    packed.forEach((group,i)=>{
      if(group.items.length!==1)return;
      const item=group.items[0],label='E'+(item.index+1)+' '+(item.event.name||item.event.label||''),fullWidth=measure(label)+padding;
      const lo=i?packed[i-1].left+packed[i-1].width+gap:left;
      const hi=i+1<packed.length?packed[i+1].left-gap:right;
      if(fullWidth<=hi-lo){group.label=label;group.width=fullWidth;group.left=Math.max(lo,Math.min(hi-fullWidth,item.px+3));}
    });
    return packed;
  }
  function popupPicker(container,button,popup,search,beforeOpen) {
    function closeOutside(event){if(!container.contains(event.target))setOpen(false);}
    function setOpen(open){popup.hidden=!open;button.setAttribute('aria-expanded',String(open));
      document.removeEventListener('click',closeOutside);
      if(open){beforeOpen();document.addEventListener('click',closeOutside);search.focus();}}
    button.onclick=()=>setOpen(popup.hidden);
    container.onkeydown=event=>{if(event.key==='Escape'){setOpen(false);button.focus();}};
    return setOpen;
  }
  function mount(parent, panel, context) {
    const domain=context.timeAxis;
    const selections=parent._flexTimeSelections||(parent._flexTimeSelections=new Map());
    const selectionKey=context.selectionKey || "default";
    const selection=selections.get(selectionKey)||{min:domain.min,max:domain.max,listeners:[],
      update(lo,hi){if(!Number.isFinite(lo)||!Number.isFinite(hi)||hi<=lo)return;this.min=Math.max(domain.min,lo);this.max=Math.min(domain.max,hi);if(this.max<=this.min){this.min=domain.min;this.max=domain.max;}this.listeners.forEach(f=>f(this.min,this.max));}};
    selections.set(selectionKey,selection);
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
    const hover = document.createElement('div'); hover.className='multi-hover'; hover.setAttribute('aria-live','polite');
    const legendHeading = document.createElement('div'); legendHeading.className='multi-legend-heading';
    wrap.appendChild(box); wrap.appendChild(hover); wrap.appendChild(legendHeading); wrap.appendChild(legend);
    if((context.events||[]).length){
      const details=document.createElement('details');details.className='multi-events';
      const summary=document.createElement('summary'),list=document.createElement('div');list.className='multi-event-list';
      const count=context.events.reduce((n,e)=>n+(e.members||[e]).length,0);
      summary.textContent='事件明细（图中 E 编号；悬浮标记查看名称；'+context.events.length+' 条标记 / '+count+' 个原始事件）';details.appendChild(summary);
      context.events.forEach((event,index)=>{
        const row=document.createElement('div');row.className='multi-event-row';
        const label=document.createElement('strong');label.textContent='E'+(index+1)+' · '+event.name+(event.members?'（合并位置：成员时间均值）':'')+' @ '+event.t.toFixed(6)+' s';row.appendChild(label);
        if(event.members)event.members.forEach(member=>{const item=document.createElement('div');item.textContent=member.name+' @ '+member.t.toFixed(6)+' s';item.title=String(member.t);row.appendChild(item);});
        list.appendChild(row);
      });details.appendChild(list);wrap.appendChild(details);
    }
    parent.appendChild(wrap);
    const datasets = panel.series.map(s => ({label:s.name, description:s.description||'', group:s.group||'其他',
      data:s.points, yAxisID:s.axis || 'y', unit:s.unit || '', gateReference:s.role==='gate_reference',
      borderColor:s.color, backgroundColor:s.color, baseColor:s.color, borderDash:s.dash||[], hidden:!!s.hidden, borderWidth:2,
      pointStyle:s.point_style||'circle', markerStyle:s.point_style,
      pointRadius:s.marker_count ? (ctx=>ctx.dataIndex % Math.max(1,Math.ceil(s.points.length/s.marker_count)) === 0 ? 2 : 0) : 0,
      pointHoverRadius:3,
      tension:0, spanGaps:false}));
    datasets.forEach((dataset,i)=>dataset.statistics=statistics(dataset.data,panel.series[i].statistics_samples));
    const scales = {x:{type:'linear',min:context.timeAxis.min,max:context.timeAxis.max,title:{display:true,text:'时间 / 秒'}}};
    Object.assign(scales, FlexReport.axisOptions(panel.axes, key=>key==='ratio'||key==='queue'));
    let brush=null,lastHover=null,labels=[];
    const events = {id:'multiCurveEvents',afterDraw(chart) {
      const {ctx,chartArea:a,scales:{x}} = chart; ctx.save(); ctx.font='11px sans-serif';
      (context.events||[]).forEach(event => {const px=x.getPixelForValue(event.t); if(px<a.left || px>a.right) return;
        ctx.strokeStyle='#94a3b8';ctx.setLineDash([4,4]);ctx.beginPath();ctx.moveTo(px,a.top);ctx.lineTo(px,a.bottom);ctx.stroke();
      });
      labels=eventLabels(context.events||[],t=>x.getPixelForValue(t),text=>ctx.measureText(text).width,a.left,a.right);
      ctx.setLineDash([]);
      labels.forEach(label=>{
        ctx.fillStyle='rgba(255,255,255,.94)';ctx.fillRect(label.left,a.top+2,label.width,18);
        ctx.fillStyle='#475569';ctx.fillText(label.label,label.left+5,a.top+15,label.width-10);
      });
      if(brush){ctx.fillStyle='rgba(22,119,255,.16)';ctx.fillRect(Math.min(brush.start,brush.end),a.top,Math.abs(brush.end-brush.start),a.bottom-a.top);}
      ctx.restore();
    }};
    let focused = null;
    const lookup=datasets.map(d=>d.data.filter(p=>Number.isFinite(p.x)).slice().sort((a,b)=>a.x-b.x));
    const valueCells=[];
    function nearest(index,t) {
      const points=lookup[index];let lo=0,hi=points.length;
      while(lo<hi){const mid=(lo+hi)>>1;if(points[mid].x<t)lo=mid+1;else hi=mid;}
      const before=points[lo-1],after=points[lo];
      const found=!before?after:!after?before:Math.abs(before.x-t)<=Math.abs(after.x-t)?before:after;
      return found&&Math.abs(found.x-t)<=3?found:null;
    }
    function referenceValue(d) {return '阈值 '+number(d.data[0]?.y)+(d.unit?' '+d.unit:'');}
    function referenceWindow(d) {return '窗口 '+number(d.data[0]?.x)+'–'+number(d.data[d.data.length-1]?.x)+' s';}
    function fillHover(t,activeIndex=null) {
      lastHover=t;
      hover.replaceChildren();
      const heading=document.createElement('strong');heading.textContent=t===null?'将鼠标移到曲线上查看时间与当前曲线':'t = '+t.toFixed(1)+' s';hover.appendChild(heading);
      if(activeIndex!==null){const d=datasets[activeIndex],p=nearest(activeIndex,t),stats=d.statistics(selection.min,selection.max);
        const row=document.createElement('div');row.className='multi-hover-row';
        row.textContent=d.gateReference ? d.label+' · '+referenceValue(d)+' · '+referenceWindow(d) : d.label+' · 当前 '+number(p&&p.y)+(d.unit?' '+d.unit:'')+' · 平均 '+number(stats.mean)+(d.unit?' '+d.unit:'');hover.appendChild(row);
      }else if(t!==null){const row=document.createElement('div');row.textContent='移近曲线查看名称与数值';hover.appendChild(row);}
      legendHeading.textContent='当前值：'+(t===null?'未悬浮':('t = '+t.toFixed(1)+' s'))+' · 平均值：选区 '+number(selection.min)+'–'+number(selection.max)+' s，有效样本等权平均';
      datasets.forEach((d,i)=>{if(!valueCells[i])return;const p=t===null?null:nearest(i,t),stats=d.statistics(selection.min,selection.max);
        valueCells[i].current.textContent=d.gateReference ? referenceValue(d) : '当前 '+number(p&&p.y)+(d.unit?' '+d.unit:'');
        valueCells[i].average.textContent=d.gateReference ? referenceWindow(d) : '平均 '+number(stats.mean)+(d.unit?' '+d.unit:'');
        valueCells[i].average.title=d.gateReference ? '冻结门禁参考值；不是观测样本，不计算选区平均' : '有效样本 '+stats.count+' / '+stats.total+'；展示统计，不重新判定门禁';});
    }
    const chart = new Chart(canvas, {
      type:'line', data:{datasets}, plugins:[events],
      options:{
        animation:false, responsive:true, maintainAspectRatio:false, parsing:false,
        interaction:{mode:'nearest',axis:'xy',intersect:false}, scales,
        onHover:(event,active)=>{if(brush)return;const a=active&&active[0];
          const inside=Number.isFinite(event.x)&&Number.isFinite(event.y)&&event.x>=chart.chartArea.left&&event.x<=chart.chartArea.right&&event.y>=chart.chartArea.top&&event.y<=chart.chartArea.bottom;
          const t=inside?chart.scales.x.getValueForPixel(event.x):a?datasets[a.datasetIndex].data[a.index].x:null;
          const close=a&&(!a.element||!Number.isFinite(event.x)||Math.hypot(a.element.x-event.x,a.element.y-event.y)<=20);
          focus(close?a.datasetIndex:null);fillHover(t,close?a.datasetIndex:null);},
        plugins:{legend:{display:false},tooltip:{enabled:false}}
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
      const line=document.createElement('span');line.className='multi-line-swatch';
      const svg=document.createElementNS('http://www.w3.org/2000/svg','svg');svg.setAttribute('viewBox','0 0 48 12');svg.setAttribute('aria-hidden','true');
      const stroke=document.createElementNS('http://www.w3.org/2000/svg','line');
      for(const [key,value] of Object.entries({x1:0,y1:6,x2:48,y2:6,stroke:d.baseColor,'stroke-width':2,'stroke-dasharray':d.borderDash.join(' ')}))stroke.setAttribute(key,String(value));
      svg.appendChild(stroke);
      if(d.markerStyle){
        const shapes={circle:['circle',{cx:24,cy:6,r:3}],rect:['rect',{x:21,y:3,width:6,height:6}],
          triangle:['polygon',{points:'24,2 28,9 20,9'}],rectRot:['polygon',{points:'24,2 28,6 24,10 20,6'}],
          star:['polygon',{points:'24,1 25,4 29,4 26,6 28,10 24,8 20,10 22,6 19,4 23,4'}],
          cross:['path',{d:'M20 6H28M24 2V10'}],crossRot:['path',{d:'M21 3L27 9M21 9L27 3'}]};
        const [tag,attrs]=shapes[d.markerStyle]||shapes.circle;
        const marker=document.createElementNS('http://www.w3.org/2000/svg',tag);
        Object.entries({...attrs,transform:'translate(24 6) scale(.75) translate(-24 -6)',fill:d.baseColor,stroke:d.baseColor,'stroke-width':1.5}).forEach(([k,v])=>marker.setAttribute(k,String(v)));
        svg.appendChild(marker);
      }
      line.appendChild(svg);item.appendChild(line);
      const name=document.createElement('span');name.className='multi-legend-name';name.textContent=d.label;item.appendChild(name);
      const values=document.createElement('span');values.className='multi-legend-values';
      const current=document.createElement('span'),average=document.createElement('span');values.append(current,average);item.appendChild(values);valueCells.push({current,average});
      item.onclick=()=>{chart.setDatasetVisibility(i,!chart.isDatasetVisible(i));refresh();};
      item.ondblclick=()=>{const isolated=datasets.every((_,j)=>chart.isDatasetVisible(j)===(j===i));datasets.forEach((_,j)=>chart.setDatasetVisibility(j,isolated||j===i));refresh();};
      item.onmouseenter=()=>focus(i);item.onmouseleave=()=>focus(null);
      legend.appendChild(item);plottedLabels.push(item);});
    function filterChoices() {const query=search.value.trim().toLocaleLowerCase();legendLabels.forEach((label,i)=>{label.hidden=!!query&&!datasets[i].label.toLocaleLowerCase().includes(query);});groups.forEach(group=>{group.element.hidden=group.indices.every(i=>legendLabels[i].hidden);});}
    search.oninput=filterChoices;
    const setPickerOpen=popupPicker(picker,pickerButton,dropdown,search,()=>{setPresetsOpen(false);filterChoices();});
    function focus(index) {focused=index;datasets.forEach((d,i)=>{d.borderWidth=index===null?2:(i===index?4:1);d.borderColor=index===null||i===index?d.baseColor:d.baseColor.slice(0,7)+'33';});
      legendLabels.forEach((label,i)=>{label.style.opacity=index===null||i===index?'1':'0.38';});
      plottedLabels.forEach((label,i)=>{label.style.opacity=(chart.isDatasetVisible(i)?1:0.45)*(index===null||i===index?1:0.38);label.classList.toggle('is-focused',index===i);});chart.update('none');}
    let activePreset=null;
    const presetEntries=[];
    function refresh() {Object.keys(panel.axes).forEach(axis=>{chart.options.scales[axis].display=datasets.some((d,i)=>d.yAxisID===axis&&chart.isDatasetVisible(i));});
      inputs.forEach((input,i)=>{input.checked=chart.isDatasetVisible(i);plottedLabels[i].classList.toggle('is-muted',!input.checked);plottedLabels[i].setAttribute('aria-pressed',String(input.checked));});
      pickerButton.textContent='指标 '+inputs.filter(input=>input.checked).length+'/'+inputs.length+' ▾';
      presetEntries.forEach(entry=>{entry.active=datasets.every((d,i)=>chart.isDatasetVisible(i)===entry.selected.includes(d.label));entry.button.classList.toggle('is-active',entry.active);entry.button.setAttribute('aria-pressed',String(entry.active));});
      const active=presetEntries.find(entry=>entry.name===activePreset&&entry.active);
      presetLabel.textContent=active?active.name:'快捷选择：run / 指标组';presetButton.title=active?active.name:'快捷选择 run 或指标组';
      presetDot.hidden=!active||!active.color;presetDot.style.backgroundColor=active&&active.color||'';if(!active)activePreset=null;
      focus(focused);fillHover(lastHover,focused);}

    function button(label,action) {const b=document.createElement('button');b.textContent=label;b.onclick=action;toolbar.appendChild(b);}
    const presets=document.createElement('div');presets.className='multi-presets';presets.hidden=!Object.keys(panel.presets||{}).length;
    const presetButton=document.createElement('button');presetButton.type='button';presetButton.className='multi-presets-button';presetButton.setAttribute('aria-expanded','false');presetButton.setAttribute('aria-label','快捷选择 run 或指标组');
    const presetDot=document.createElement('i');presetDot.className='multi-preset-dot';presetDot.setAttribute('aria-hidden','true');
    const presetLabel=document.createElement('span');presetLabel.className='multi-preset-label';
    const presetArrow=document.createElement('span');presetArrow.textContent='▾';presetButton.append(presetDot,presetLabel,presetArrow);presets.appendChild(presetButton);
    const presetMenu=document.createElement('div');presetMenu.className='multi-dropdown multi-preset-dropdown';presetMenu.hidden=true;
    const presetSearch=document.createElement('input');presetSearch.type='search';presetSearch.className='multi-search';presetSearch.placeholder='搜索 run / 指标组';presetSearch.setAttribute('aria-label','搜索快捷预设');presetMenu.appendChild(presetSearch);
    const presetList=document.createElement('div');presetList.className='multi-preset-list';presetMenu.appendChild(presetList);
    const emptyPreset=document.createElement('p');emptyPreset.textContent='没有匹配的预设';emptyPreset.hidden=true;presetMenu.appendChild(emptyPreset);
    const setPresetsOpen=popupPicker(presets,presetButton,presetMenu,presetSearch,()=>{setPickerOpen(false);filterPresets();});
    Object.entries(panel.presets||{}).forEach(([name,selected])=>{
      const button=document.createElement('button');button.type='button';button.className='multi-preset-choice';button.title=name;
      const color=presetColor(selected,panel.series),dot=document.createElement('i');dot.className='multi-preset-dot';dot.hidden=!color;dot.style.backgroundColor=color||'';dot.setAttribute('aria-hidden','true');button.appendChild(dot);
      const label=document.createElement('span');label.textContent=name;button.appendChild(label);
      button.onclick=()=>{activePreset=name;datasets.forEach((d,i)=>chart.setDatasetVisibility(i,selected.includes(d.label)));refresh();setPresetsOpen(false);presetButton.focus();};
      presetList.appendChild(button);presetEntries.push({name,selected,color,button});
    });
    function filterPresets(){const query=presetSearch.value.trim().toLocaleLowerCase();presetEntries.forEach(e=>{e.button.hidden=!e.name.toLocaleLowerCase().includes(query);});emptyPreset.hidden=presetEntries.some(e=>!e.button.hidden);}
    presetSearch.oninput=filterPresets;
    presets.addEventListener('keydown',event=>{if(!['ArrowDown','ArrowUp','Home','End'].includes(event.key))return;
      if(presetMenu.hidden)setPresetsOpen(true);
      const visible=presetEntries.filter(e=>!e.button.hidden);if(!visible.length)return;
      const current=visible.findIndex(e=>e.button===document.activeElement),delta=event.key==='ArrowUp'?-1:1;
      const index=event.key==='Home'?0:event.key==='End'?visible.length-1:current<0?(delta>0?0:visible.length-1):(current+delta+visible.length)%visible.length;
      visible[index].button.focus();event.preventDefault();});
    presets.appendChild(presetMenu);toolbar.appendChild(presets);
    button('全选',()=>{datasets.forEach((d,i)=>chart.setDatasetVisibility(i,true));refresh();});button('清空',()=>{datasets.forEach((d,i)=>chart.setDatasetVisibility(i,false));refresh();});
    const range=document.createElement('div');range.className='multi-range';const from=document.createElement('input'),to=document.createElement('input');for(const el of [from,to]){el.type='number';el.step='1';el.style.width='90px';}
    from.setAttribute('aria-label','区间开始秒');to.setAttribute('aria-label','区间结束秒');range.append('拖拽选区 / 秒 ',from,' — ',to);toolbar.appendChild(range);
    selection.listeners.push((lo,hi)=>{from.value=lo;to.value=hi;chart.options.scales.x.min=lo;chart.options.scales.x.max=hi;chart.update('none');fillHover(lastHover);});
    from.onchange=to.onchange=()=>selection.update(Number(from.value),Number(to.value));
    button('还原区间',()=>{brush=null;selection.update(domain.min,domain.max);});
    function pixel(event){const rect=canvas.getBoundingClientRect();return Math.max(chart.chartArea.left,Math.min(chart.chartArea.right,(event.clientX-rect.left)*chart.width/rect.width));}
    canvas.style.touchAction='pan-y';
    canvas.addEventListener('pointerdown',event=>{const rect=canvas.getBoundingClientRect(),y=(event.clientY-rect.top)*chart.height/rect.height,x=(event.clientX-rect.left)*chart.width/rect.width;
      if(event.button!==0||x<chart.chartArea.left||x>chart.chartArea.right||y<chart.chartArea.top||y>chart.chartArea.bottom)return;
      brush={start:pixel(event),end:pixel(event),pointer:event.pointerId};canvas.setPointerCapture(event.pointerId);chart.draw();});
    canvas.addEventListener('pointermove',event=>{
      const rect=canvas.getBoundingClientRect(),px=(event.clientX-rect.left)*chart.width/rect.width,py=(event.clientY-rect.top)*chart.height/rect.height;
      const label=py>=chart.chartArea.top&&py<=chart.chartArea.top+22?labels.find(label=>px>=label.left&&px<=label.left+label.width):null;
      canvas.title=label?label.items.map(item=>'E'+(item.index+1)+' · '+(item.event.name||item.event.label)+' @ '+item.event.t.toFixed(6)+' s').join('\n'):'';
      if(brush&&event.pointerId===brush.pointer){brush.end=pixel(event);chart.draw();}});
    canvas.addEventListener('pointerup',event=>{if(!brush||event.pointerId!==brush.pointer)return;const selected=brush;selected.end=pixel(event);brush=null;
      if(canvas.hasPointerCapture(event.pointerId))canvas.releasePointerCapture(event.pointerId);
      if(Math.abs(selected.end-selected.start)>=6){const x=chart.scales.x,lo=x.getValueForPixel(Math.min(selected.start,selected.end)),hi=x.getValueForPixel(Math.max(selected.start,selected.end));selection.update(lo,hi);}else chart.draw();});
    canvas.addEventListener('pointercancel',()=>{brush=null;chart.draw();});
    canvas.addEventListener('pointerleave',()=>{canvas.title='';if(!brush){focus(null);fillHover(null);}});
    selection.update(selection.min,selection.max);refresh();return chart;
  }
  root.FlexMultiCurve={mount,statistics,presetColor,eventLabels};
})(typeof window === 'undefined' ? globalThis : window);
