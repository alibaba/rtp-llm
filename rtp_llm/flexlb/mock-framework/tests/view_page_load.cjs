const assert = require('node:assert/strict');
const vm = require('node:vm');
const fs = require('node:fs');
const source = fs.readFileSync(process.argv[2], 'utf8').split('<script>')[1].split('</script>')[0].replace(/load\(\);\s*$/, '');
(async () => {
for (const includeVit of [false, true]) {
for (const initiallyReady of [true, false]) {
 for (const selected of [[], ['kill_a'], ['kill_a','kill_b']]) {
  const elements = new Map();
  const frameLoads = [];
  function element(id) {
   return {children: [], classList: {add(){},remove(){},toggle(){}},
    append(...children){this.children.push(...children)},
    setAttribute(name,value){this[name]=value},
    get options(){return this.children},
    get selectedOptions(){return this.children.filter(child=>child.selected)},
    get childElementCount(){return this.children.length},
    hasAttribute(name){return this[name] !== undefined},
    set src(value){frameLoads.push(value);this._src=value}, get src(){return this._src},
    querySelectorAll(){return []}};
  }
  const document = {getElementById(id){if(!elements.has(id))elements.set(id,element(id));return elements.get(id)},
   createElement:element,createTextNode:text=>text,querySelectorAll:()=>[]};
  const ready = {status:'ready',url:'http://localhost/d/embedded',open_url:'http://localhost/d/full',
   filters:{prefill_instance:['127.0.0.1:1001','127.0.0.1:1002'],decode_instance:['127.0.0.1:1003','127.0.0.1:1004'],vit_instance:includeVit?['127.0.0.1:1005','127.0.0.1:1006']:[],master_instance:['127.0.0.1:2001','127.0.0.1:2002'],client_instance:['127.0.0.1:3001','127.0.0.1:3002']},annotations:{steps:['kill_a','kill_b'],selected}};
  const body = {name:'test',describe:'',result:{status:'PASS'},start:1000,end:2000,gates:[],events:[],warnings:[],effective:{},
   collection:{complete:true},metrics:initiallyReady?ready:{status:'starting',url:null}};
  const context = vm.createContext({document,URL,URLSearchParams,setTimeout(){},fetch:async url=>({ok:true,json:async()=>url==='/api/experiment'?body:ready})});
  await vm.runInContext(source+'\nload()',context);
  if(!initiallyReady) {
   assert.equal(frameLoads.length,0);
   await vm.runInContext('refreshMetrics()',context);
  }
  assert.equal(frameLoads.length,1);
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
   const selectedInstances = () => new URL(frameLoads.at(-1)).searchParams.getAll('var-'+name);
   choose(['$__all',addresses[0]]);
   assert.deepEqual(selectedInstances(),[addresses[0]]);
   choose(addresses);
   assert.deepEqual(selectedInstances(),addresses);
   for (const other of Object.keys(ready.filters).filter(other=>other!==name && ready.filters[other].length)) {
    assert.deepEqual(new URL(frameLoads.at(-1)).searchParams.getAll('var-'+other),['$__all']);
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
 }
}
}
console.log('12 initial-load scenarios passed');
})().catch(error=>{console.error(error);process.exit(1)});
