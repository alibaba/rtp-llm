// Run: node tests/js/test_report_controller.js
const assert = require('node:assert/strict');
class Element {
  constructor(tag) {this.tag=tag;this.children=[];this.style={};this.textContent='';this.innerHTML='';}
  appendChild(e) {this.children.push(e);return e;}
  querySelector(tag) {this.queries ||= {};return this.queries[tag] ||= new Element(tag);}
  getContext() {return this;}
}
const roots=Object.fromEntries(['title','subtitle','kpis','grid'].map(id=>[id,new Element(id)]));
global.document={getElementById:id=>roots[id],createElement:tag=>new Element(tag)};
const charts=[], temporal=[];
global.Chart=class {
  constructor(ctx,spec){this.data=spec.data;this.options=spec.options;this.visible=this.data.datasets.map(d=>!d.hidden);charts.push(this);}
  isDatasetVisible(i){return this.visible[i];}setDatasetVisibility(i,v){this.visible[i]=v;}update(){}
};
global.FlexMultiCurve={mount:(...args)=>temporal.push(args)};
global.FlexLegend = require('../../src/reporting/assets/legend_interaction');
require('../../src/reporting/assets/report_controller');
const points=[{x:0,y:0},{x:1,y:null},{x:2,y:4}];
const axes={y:{position:'left',title:'count',min:0,max:10}};
FlexReport.mount({summary:{title:'A',subtitle:'B',kpis:[]},timeAxis:{min:0,max:2},panels:[
  {id:'time',type:'line',timeX:true,axes,series:[],events:[]},
  {id:'bars',type:'bar',timeX:false,axes,series:[{name:'observed',axis:'y',points,color:'#123456',hidden:false}]},
  {id:'scatter',type:'scatter',timeX:false,bounds:{x:[0,2],y:[0,10]},axisLabels:{x:'input',y:'output'},series:[{name:'dots',points,color:'#456789'}]},
  {id:'ecdf',type:'line',timeX:false,axes,series:[{name:'distribution',axis:'y',points,color:'#456789'}]},
  {id:'categories',type:'bar',timeX:false,axes,series:[{name:'counts',axis:'y',points:[{x:'P',y:1},{x:'D',y:2}],color:'#456789'}]}
]});
assert.equal(roots.title.textContent,'A');
assert.equal(temporal.length,1);assert.deepEqual(temporal[0][2].timeAxis,{min:0,max:2});
assert.equal(charts.length,4);assert.equal(charts[0].data.datasets[0].data,points);
assert.equal(charts[0].options.scales.y.title.text,'count');
assert.equal(charts[1].options.scales.x.max,2);assert.equal(charts[1].options.scales.y.max,10);
assert.equal(charts[2].options.scales.x.type,'linear');
assert.equal(charts[3].options.scales.x.type,'category');
assert.deepEqual(FlexReport.axisOptions(axes).y,{type:'linear',position:'left',beginAtZero:true,min:0,max:10,title:{display:true,text:'count'}});
assert.equal(FlexReport.axisOptions(axes,key=>key==='y').y.grid.drawOnChartArea,true);
console.log('page routing, point gaps, bounds and shared axis options passed');
