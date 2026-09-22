const assert = require('node:assert/strict');
class Element {
  constructor(tag) { this.tag = tag; this.children = []; this.style = {}; this.attributes = {}; }
  appendChild(child) { this.children.push(child); return child; }
  append(...children) { this.children.push(...children); }
  setAttribute(key,value) { this.attributes[key]=value; }
  focus() { this.focused=true; }
  contains(child) { return this===child||this.children.some(c=>c instanceof Element&&c.contains(child)); }
}
global.document = {createElement:tag=>new Element(tag),createTextNode:text=>text,addEventListener:()=>{}};
global.Chart = class {
  constructor(canvas,spec) {this.data=spec.data;this.options=spec.options;this.visible=spec.data.datasets.map(d=>!d.hidden);}
  setDatasetVisibility(i,v) {this.visible[i]=v;}
  isDatasetVisible(i) {return this.visible[i];}
  update() {}
};
require('./multi_curve');
const root=new Element('root');
const points=[{x:0,y:70},{x:1,y:null},{x:3,y:20}];
const chart=FlexMultiCurve.mount(root,{title:'test',caption:'test',axes:{pct:{title:'%'},queue:{title:'requests'}},
 series:[{name:'hit',axis:'pct',unit:'%',points,color:'#123456'},
         {name:'queue',axis:'queue',unit:'requests',points:[{x:0,y:1000}],hidden:true,color:'#654321'}],
 presets:{Queue:['queue']}},{timeAxis:{min:0,max:10},events:[]});
assert.equal(chart.options.scales.pct.display,true);
assert.equal(chart.options.scales.queue.display,false);
assert.equal(chart.data.datasets[0].data[1].y,null);
assert.equal(chart.data.datasets[1].data[0].y,1000); // never silently normalize units
const toolbar=root.children[0].children[2];
const picker=toolbar.children[0],pickerButton=picker.children[0],dropdown=picker.children[1];
assert.equal(dropdown.hidden,true);
pickerButton.onclick();assert.equal(dropdown.hidden,false);
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
assert.equal(chart.options.scales.pct.display,false);
assert.equal(chart.options.scales.queue.display,true);
const range=toolbar.children.find(c=>c.className==='multi-range');
const [from,to]=range.children.filter(c=>c.tag==='input');
from.value='3';to.value='7';from.onchange();
assert.equal(chart.options.scales.x.min,3);assert.equal(chart.options.scales.x.max,7);
to.value='2';to.onchange();assert.equal(chart.options.scales.x.max,7);
console.log('multi-curve interaction contract passed');
