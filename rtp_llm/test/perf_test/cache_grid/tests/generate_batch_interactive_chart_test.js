// Exercise chart controls against recorded data without a browser dependency.
const fs=require('fs'),vm=require('vm'),assert=require('assert'),path=require('path');
if(process.argv.length!==4) throw new Error('Usage: node generate_batch_interactive_chart_test.js chart.html interactive_data.json');
const html=fs.readFileSync(process.argv[2],'utf8');
const scripts=[...html.matchAll(/<script>([\s\S]*?)<\/script>/g)];
const source=scripts[scripts.length-1][1];
const dataset=JSON.parse(fs.readFileSync(process.argv[3],'utf8')).rows;
const elements={};
for(const id of ['view','batch','band','reset','count','plot','trend','detail','axis-legend']){
 elements[id]={value:id==='view'?'mean':'all',textContent:'',children:[],append(o){this.children.push(o)},removeAllListeners(){this.handler=null},on(name,fn){this.handler=fn}};
}
const plots={};
const sandbox={console,document:{getElementById:id=>elements[id],createElement:()=>({})},
 Plotly:{react:(id,traces,layout)=>{plots[id]={traces,layout}}}};
vm.createContext(sandbox);
vm.runInContext(source,sandbox);
assert.equal(plots.plot.traces[0].x.length,dataset.length);
const batches=[...new Set(dataset.map(r=>r.batch_size))];
assert.equal(elements.batch.children.length,batches.length);
assert(!html.includes('value="latency"'));
for(const view of ['mean','sum']){
 elements.view.value=view;
 for(const batch of ['all',...batches,'all']){
  elements.batch.value=String(batch);
  for(const band of ['all','low/low','low/high','high/low','high/high']){
   elements.band.value=band;sandbox.draw();
   const expected=dataset.filter(r=>(batch==='all'||r.batch_size===batch)&&(band==='all'||r.band===band));
   for(const axis of ['x','y','z']) {
    const title=plots.plot.layout.scene[axis+'axis'].title;
    assert.equal(typeof title,'object');
    assert(title.text.length>0);
    assert(elements['axis-legend'].textContent.includes(title.text));
   }
   assert(plots.trend.layout.xaxis.title.text.includes('请求数'));
   assert(plots.trend.layout.yaxis.title.text.includes('ms'));
   assert(elements['axis-legend'].textContent.includes('1024'));
   const trace=plots.plot.traces[0];
   assert.equal(trace.marker.reversescale,true);
   assert.equal(trace.marker.colorbar.title.text,'批内最大服务端首 token 延迟中位数 (ms)');
   assert.deepEqual(Array.from(trace.marker.color),expected.map(r=>r.median_batch_first_token_cost_time_ms));
   assert.equal(trace.marker.cmin,Math.min(...dataset.map(r=>r.median_batch_first_token_cost_time_ms)));
   assert.equal(trace.marker.cmax,Math.max(...dataset.map(r=>r.median_batch_first_token_cost_time_ms)));
   assert.equal(plots.plot.layout.uirevision,(batch==='all'?'all':'single')+':'+view);
   assert(plots.plot.layout.scene.xaxis.title.text.includes('compute'));
   assert(plots.plot.layout.scene.zaxis.title.text.includes(batch==='all'?'Batch':'服务端'));
   assert.equal(trace.x.length,expected.length);
   if(expected.length){
    const first=expected[0];
    const compute=(view==='sum'?first.compute_tokens:first.mean_compute_tokens)/1024;
    assert.equal(trace.x[0],compute);
    assert.equal(trace.y[0],(view==='sum'?first.cached_tokens:first.mean_cached_tokens)/1024);
    assert.equal(trace.z[0],batch==='all'?first.batch_size:first.median_batch_first_token_cost_time_ms);
    elements.plot.handler({points:[{customdata:first}]});
    assert(elements.detail.textContent.includes(String(first.case_id)));
   }
  }
 }
}
elements.reset.onclick();
assert.equal(plots.plot.traces[0].x.length,dataset.length);
assert.equal(elements.view.value,'mean');
assert(plots.plot.layout.scene.xaxis.title.text.includes('compute'));
assert(plots.plot.layout.scene.zaxis.title.text.includes('Batch'));
console.log('PASS: axes, units, legend, all view/filter combinations, click details and reset.');
