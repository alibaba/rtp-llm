const assert = require('node:assert/strict');
const vm = require('node:vm');
const fs = require('node:fs');
const source = fs.readFileSync(process.argv[2], 'utf8').split('<script>')[1].split('</script>')[0].replace(/loadExperiment\(\);\s*$/, '');
(async () => {
for (const includeVit of [false, true]) {
for (const initiallyReady of [true, false]) {
 for (const selected of [[], ['kill_a'], ['kill_a','kill_b']]) {
  const elements = new Map();
  const frameLoads = [];
  let dashboardQuery;
  function element(id) {
   return {children: [], classList: {add(){},remove(){},toggle(){}},
    append(...children){this.children.push(...children)},
    setAttribute(name,value){this[name]=value},
    get options(){return this.children},
    get selectedOptions(){return this.children.filter(child=>child.selected)},
    get childElementCount(){return this.children.length},
    set src(value){frameLoads.push(value);this._src=value;dashboardQuery=new URL(value).searchParams}, get src(){return this._src},
    contentWindow:{postMessage(message,origin){
      assert.equal(origin,'http://localhost');
      assert.equal(message.type,'mock-view-query');
      const query = new URLSearchParams(message.query);
      for (const key of new Set(query.keys())) {
        dashboardQuery.delete(key);
        query.getAll(key).forEach(value=>dashboardQuery.append(key,value));
      }
    }},
    querySelectorAll(selector){return selector==='input'?this.children.map(label=>label.children[0]):[]}};
  }
  const document = {getElementById(id){if(!elements.has(id))elements.set(id,element(id));return elements.get(id)},
   createElement:element,createTextNode:text=>text,querySelectorAll:()=>[]};
  const ready = {status:'ready',url:'http://localhost/d/embedded',open_url:'http://localhost/d/full',
   filters:{prefill_instance:['127.0.0.1:1001','127.0.0.1:1002'],decode_instance:['127.0.0.1:1003','127.0.0.1:1004'],vit_instance:includeVit?['127.0.0.1:1005','127.0.0.1:1006']:[],master_instance:['127.0.0.1:2001','127.0.0.1:2002'],client_instance:['127.0.0.1:3001','127.0.0.1:3002']},annotations:{steps:['kill_a','kill_b'],selected}};
  const gate = {title:'Master',status:'PASS',unit:'',interface:'master/list/addresses',
   step_id:'kill_a',point:1500,from:1300,to:1600,args:{},actual:['127.0.0.1:1001'],expect:{eq:['127.0.0.1:1001']}};
  const longValue = {addresses:Array(8).fill('127.0.0.1:1001')};
  const longGate = {...gate,step_id:'kill_b',actual:longValue,expect:{eq:longValue}};
  const body = {name:'test',describe:'',result:{status:'ERROR',cleanup_errors:['saved cleanup error']},start:1000,end:2000,gates:[gate,longGate],events:[],warnings:[],effective:{},
   collection:{complete:true},metrics:initiallyReady?ready:{status:'starting',url:null}};
  const context = vm.createContext({document,URL,URLSearchParams,setTimeout(){},fetch:async url=>({ok:true,json:async()=>url==='/api/experiment'?body:ready})});
  await vm.runInContext(source+'\nloadExperiment()',context);
  assert.match(elements.get('notice').textContent,/saved cleanup error/);
  if(!initiallyReady) {
   assert.equal(frameLoads.length,0);
   await vm.runInContext('refreshMetrics()',context);
  }
  assert.equal(frameLoads.length,1);
  elements.get('annotations-none').onclick();
  assert.deepEqual(dashboardQuery.getAll('var-annotation_steps'),selected.length?selected.map(s=>'step:'+s):['__none__']);
  elements.get('grafana').onload();
  assert.deepEqual(dashboardQuery.getAll('var-annotation_steps'),['__none__']);
  for (const label of elements.get('annotation-choices').children) {
    const checkbox = label.children[0];
    checkbox.checked = selected.includes(checkbox.value);
    checkbox.onchange();
  }
  const params = new URL(frameLoads[0]).searchParams;
  assert.deepEqual(params.getAll('var-annotation_steps'),selected.length?selected.map(s=>'step:'+s):['__none__']);
  assert.equal(params.get('var-prefill_instance'),'$__all');
  assert.equal(params.get('var-decode_instance'),'$__all');
  assert.equal(params.has('var-vit_instance'),includeVit);
  assert.equal(elements.get('filters').childElementCount,includeVit?5:4);
  for (const [index,name] of Object.keys(ready.filters).filter(name=>ready.filters[name].length).entries()) {
   const addresses = ready.filters[name];
   const [select, aggregate, separate] = elements.get('filters').children[index].children;
   assert.equal(select.multiple,true);
   const choose = values => {
    select.options.forEach(option=>{option.selected=values.includes(option.value)});
    select.onchange();
   };
   const selectedInstances = () => dashboardQuery.getAll('var-'+name);
   choose(['$__all',addresses[0]]);
   assert.deepEqual(selectedInstances(),[addresses[0]]);
   choose(addresses);
   assert.deepEqual(selectedInstances(),addresses);
   for (const other of Object.keys(ready.filters).filter(other=>other!==name && ready.filters[other].length)) {
    assert.deepEqual(dashboardQuery.getAll('var-'+other),['$__all']);
   }
   assert.deepEqual(new URL(elements.get('open-grafana').href).searchParams.getAll('var-'+name),addresses);
   choose(['$__all',...addresses]);
   assert.deepEqual(selectedInstances(),['$__all']);
   separate.onclick();
   assert.deepEqual(selectedInstances(),addresses);
   aggregate.onclick();
   assert.deepEqual(selectedInstances(),['$__all']);
   choose([]);
   assert.deepEqual(selectedInstances(),['$__all']);
  }

  assert.deepEqual(elements.get('annotation-choices').children.map(label=>label.children[0].checked),['kill_a','kill_b'].map(s=>selected.includes(s)));
  const [shortCard,longCard] = elements.get('gates').children;
  assert.equal(shortCard.children[1].textContent,'["127.0.0.1:1001"]');
  assert.equal(shortCard.children[2].textContent,'预期 = ["127.0.0.1:1001"]');
  assert.equal(longCard.children.length,2);
  vm.runInContext('showTimeRange(1100, 1900, "Current range")',context);
  dashboardQuery.set('from','1200');
  shortCard.onclick();
  const focused = dashboardQuery;
  assert.equal(focused.get('from'),'1200');
  assert.equal(focused.get('to'),'1900');
  assert.deepEqual(focused.getAll('var-annotation_steps'),['step:kill_a']);
  assert.deepEqual(new URL(elements.get('open-grafana').href).searchParams.getAll('var-annotation_steps'),['step:kill_a']);
  assert.deepEqual(elements.get('annotation-choices').children.map(label=>label.children[0].checked),[true,false]);
  assert.equal(elements.get('gate-record').open,false);
  longCard.onclick();
  assert.deepEqual(dashboardQuery.getAll('var-annotation_steps'),['step:kill_b']);
  assert.equal(elements.get('gate-record').open,true);
  assert.match(elements.get('gate-values').textContent,/"addresses"/);
  elements.get('annotations-all').onclick();
  assert.deepEqual(dashboardQuery.getAll('var-annotation_steps'),['step:kill_a','step:kill_b']);
  elements.get('reset').onclick();
  assert.equal(dashboardQuery.get('from'),'1000');
  assert.equal(dashboardQuery.get('to'),'2000');
  assert.equal(frameLoads.length,1);
 }
}
}
console.log('12 initial-load scenarios passed');
})().catch(error=>{console.error(error);process.exit(1)});
