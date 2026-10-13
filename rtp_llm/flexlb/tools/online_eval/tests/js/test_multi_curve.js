// 手动运行入口：node tests/js/test_multi_curve.js
const assert = require('node:assert/strict');
class Element {
  constructor(tag) { this.tag = tag; this.children = []; this.style = {}; this.attributes = {}; this.classes=new Set(); this.classList={toggle:(name,on)=>{if(on)this.classes.add(name);else this.classes.delete(name);}}; }
  appendChild(child) { this.children.push(child); return child; }
  append(...children) { this.children.push(...children); }
  setAttribute(key,value) { this.attributes[key]=value; }
  focus() { this.focused=true; }
  replaceChildren(...children) {this.children=children;}
  addEventListener(name, callback) { (this.events ||= {})[name]=callback; }
  getBoundingClientRect() { return {left:0,top:0,width:1000,height:600}; }
  setPointerCapture(id) {this.pointer=id;}
  hasPointerCapture(id) {return this.pointer===id;}
  releasePointerCapture(id) {this.pointer=null;}
  contains(child) { return this===child||this.children.some(c=>c instanceof Element&&c.contains(child)); }
}
const documentClicks=new Set();
global.document = {createElement:tag=>new Element(tag),createElementNS:(_,tag)=>new Element(tag),createTextNode:text=>text,
  addEventListener:(name,handler)=>{if(name==='click')documentClicks.add(handler);},
  removeEventListener:(name,handler)=>{if(name==='click')documentClicks.delete(handler);}};
global.Chart = class {
  constructor(canvas,spec) {this.data=spec.data;this.options=spec.options;this.visible=spec.data.datasets.map(d=>!d.hidden);
    this.width=1000;this.height=600;this.chartArea={left:100,right:900,top:20,bottom:550};
    this.scales={x:{getValueForPixel:p=>this.options.scales.x.min+(p-100)/800*(this.options.scales.x.max-this.options.scales.x.min)}};}
  setDatasetVisibility(i,v) {this.visible[i]=v;}
  isDatasetVisible(i) {return this.visible[i];}
  update() {}
  draw() {}
};
require('../../src/reporting/assets/report_controller');
require('../../src/reporting/assets/multi_curve');
const root=new Element('root');
const points=[{x:0,y:70},{x:1,y:null},{x:3,y:20}];
const chart=FlexMultiCurve.mount(root,{title:'test',caption:'test',axes:{pct:{title:'%'},queue:{title:'requests'}},
 series:[{name:'hit',axis:'pct',unit:'%',points,color:'#123456',dash:[6,4],point_style:'triangle',marker_count:2},
         {name:'queue',axis:'queue',unit:'requests',points:[{x:0,y:1000}],hidden:true,color:'#65432180'}],
 presets:{Queue:['queue']}},{timeAxis:{min:0,max:10},events:[]});
assert.equal(chart.options.scales.pct.display,true);
assert.equal(chart.options.scales.queue.display,false);
assert.equal(chart.data.datasets[0].data[1].y,null);
assert.equal(chart.data.datasets[1].data[0].y,1000); // never silently normalize units
assert.deepEqual(chart.data.datasets[0].borderDash,[6,4]);
assert.equal(chart.data.datasets[0].pointStyle,'triangle');
assert.equal(chart.data.datasets[0].pointRadius({dataIndex:0}),2);
assert.equal(chart.data.datasets[0].pointRadius({dataIndex:1}),0);
assert.equal(chart.data.datasets[0].pointRadius({dataIndex:2}),2);
const panel=root.children[0],legend=panel.children.find(c=>c.className==='multi-legend'),hover=panel.children.find(c=>c.className==='multi-hover');
assert.equal(legend.children[0].children[0].children[0].children[0].attributes['stroke-dasharray'],'6 4');
assert.equal(legend.children[0].children[0].children[0].children[1].tag,'polygon');
assert.equal(legend.children[1].children[0].children[0].children[0].attributes['stroke-dasharray'],'');
assert.equal(legend.children[1].classes.has('is-muted'),true);
assert.notEqual(legend.children[1].hidden,true); // disabled curve remains clickable
assert.match(legend.children[0].children[2].children[1].textContent,/平均 45/);
chart.options.onHover({},[{datasetIndex:0,index:0}]);
assert.match(hover.children[0].textContent,/t = 0.0 s/);
assert.match(hover.children[1].textContent,/hit.*当前 70.*平均 45/);
assert.match(legend.children[0].children[2].children[0].textContent,/当前 70/);
assert.equal(chart.data.datasets[0].borderWidth,4);
chart.options.onHover({x:100,y:300},[{datasetIndex:0,index:0,element:{x:100,y:50}}]);
assert.equal(chart.data.datasets[0].borderWidth,2); // far from curve: do not falsely activate it
assert.match(hover.children[1].textContent,/移近曲线/);
chart.options.onHover({},[{datasetIndex:0,index:1}]);
assert.match(hover.children[1].textContent,/当前 无数据/); // missing point never becomes zero
const toolbar=root.children[0].children[2];
const picker=toolbar.children[0],pickerButton=picker.children[0],dropdown=picker.children[1];
assert.equal(dropdown.hidden,true);
assert.equal(documentClicks.size,0);
pickerButton.onclick();assert.equal(dropdown.hidden,false);
assert.equal(documentClicks.size,1);
pickerButton.onclick();assert.equal(documentClicks.size,0);
pickerButton.onclick();assert.equal(documentClicks.size,1);
for(const listener of [...documentClicks])listener({target:new Element('outside')});
assert.equal(dropdown.hidden,true);assert.equal(documentClicks.size,0);
pickerButton.onclick();
assert.equal(pickerButton.attributes['aria-expanded'],'true');
const search=dropdown.children[0],choices=dropdown.children[1];
search.value='queue';search.oninput();
assert.equal(choices.children.find(c=>c.tag==='label').hidden,true);
assert.equal(choices.children.filter(c=>c.tag==='label')[1].hidden,false);
search.value='';search.oninput();
picker.onkeydown({key:'Escape'});assert.equal(dropdown.hidden,true);
toolbar.children[1].onclick();
assert.deepEqual(chart.visible,[false,true]);
assert.match(pickerButton.textContent,/1\/2/);
assert.equal(legend.children[0].classes.has('is-muted'),true);
assert.notEqual(legend.children[0].hidden,true);
assert.equal(legend.children[1].classes.has('is-muted'),false);
legend.children[0].onmouseenter();
assert.equal(chart.data.datasets[1].borderColor,'#65432133');
legend.children[0].onmouseleave();
assert.equal(chart.data.datasets[1].borderColor,'#65432180');
assert.equal(chart.options.scales.pct.display,false);
assert.equal(chart.options.scales.queue.display,true);
choices.children.filter(c=>c.tag==='label')[0].children[1].checked=true;
choices.children.filter(c=>c.tag==='label')[0].children[1].onchange();
assert.deepEqual(chart.visible,[true,true]);
assert.equal(legend.children[0].classes.has('is-muted'),false);
legend.children[0].ondblclick();assert.deepEqual(chart.visible,[true,false]);
assert.equal(legend.children[1].classes.has('is-muted'),true);
assert.notEqual(legend.children[1].hidden,true);
legend.children[0].ondblclick();assert.deepEqual(chart.visible,[true,true]);
legend.children[1].onclick();assert.deepEqual(chart.visible,[true,false]);
legend.children[1].onclick();assert.deepEqual(chart.visible,[true,true]); // restore from retained row
const range=toolbar.children.find(c=>c.className==='multi-range');
const [from,to]=range.children.filter(c=>c.tag==='input');
from.value='3';to.value='7';from.onchange();
assert.equal(chart.options.scales.x.min,3);assert.equal(chart.options.scales.x.max,7);
to.value='2';to.onchange();assert.equal(chart.options.scales.x.max,7);
const reset=toolbar.children.find(c=>c.textContent==='还原区间');
reset.onclick();
assert.equal(chart.options.scales.x.min,0);assert.equal(chart.options.scales.x.max,10);
const sibling=FlexMultiCurve.mount(root,{title:'sibling',caption:'',axes:{y:{title:'count'}},series:[]},{timeAxis:{min:0,max:10},events:[]});
const canvas=panel.children[3].children[0];
canvas.events.pointerleave();assert.match(hover.children[0].textContent,/将鼠标/);
assert.equal(chart.data.datasets[0].borderWidth,2);
const pointer=(x,y=100,id=1)=>({button:0,clientX:x,clientY:y,pointerId:id});
canvas.events.pointerdown(pointer(260));canvas.events.pointermove(pointer(740));canvas.events.pointerup(pointer(740));
assert.equal(chart.options.scales.x.min,2);assert.equal(chart.options.scales.x.max,8);
assert.equal(sibling.options.scales.x.min,2);assert.equal(sibling.options.scales.x.max,8);
assert.equal(canvas.hasPointerCapture(1),false);
reset.onclick();
canvas.events.pointerdown(pointer(500));canvas.events.pointerup(pointer(503));
assert.equal(chart.options.scales.x.max,10); // click is not an interval
canvas.events.pointerdown(pointer(500,590));
assert.equal(canvas.hasPointerCapture(1),false); // outside plot
canvas.events.pointerdown(pointer(740));canvas.events.pointerup(pointer(260));
assert.equal(chart.options.scales.x.min,2);assert.equal(chart.options.scales.x.max,8); // reverse brush
canvas.events.pointerdown(pointer(300));canvas.events.pointercancel();
assert.equal(chart.options.scales.x.min,2);
assert.deepEqual(FlexMultiCurve.statistics(points)(0,3),{mean:45,count:2,total:3});
assert.deepEqual(FlexMultiCurve.statistics(points)(1,1),{mean:null,count:0,total:1});
console.log('multi-curve points, missing samples, interval and legend contracts passed');

// Compact transport keeps exact interval statistics, duplicates and missing samples.
const raw=[{x:3,y:8},{x:1,y:null},{x:2,y:0},{x:2,y:4},{x:NaN,y:99}];
const compact={times:raw.map(p=>p.x),values:raw.map(p=>p.y)};
for(const [lo,hi] of [[0,4],[1,1],[2,2],[1.5,3],[9,10]]) {
  assert.deepEqual(FlexMultiCurve.statistics([],compact)(lo,hi),FlexMultiCurve.statistics(raw)(lo,hi));
}
