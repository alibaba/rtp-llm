"""Self-contained input-fidelity report; independent of experiment reporting."""
import json
from pathlib import Path


def markdown(report):
    lines=['# 合成流量保真度', '', f"画像：`{report['profile']}`", f"画像 SHA：`{report['profile_sha256']}`",
           f"身份核验：{report['identity']['status']}；模式：{report['mode']}；seed：{report['seed']}", '',
           '从生产生成器实际输出的标签重算无限历史 prefix 共享；不等同真实缓存命中。',
           'PASS 只表示所列分布阈值通过，cache 实验仍需 trace 对照；缺失来源时不做适用性判断。', '',
           '| 捕获 | 方法 | length KS | depth KS | joint TV | 长度分布 | cache 使用 |',
           '|---|---|---:|---:|---:|---|---|']
    for row in report['rows']:
        for name, method in row['methods'].items():
            if method['status'] != 'MEASURED':
                lines.append(f"| {row['capture']} | {name} | — | — | — | UNAVAILABLE | UNASSESSED |")
                continue
            m,g=method['metrics'],method['grades']
            lines.append(f"| {row['capture']} | {name} | {m['length_ks']:.4f} | {m['depth_ks']:.4f} | {m['joint_tv']:.4f} | {g['length']} | {g['cache']} |")
    lines+=['', '跨窗口偏差包含预期流量漂移，不自动视作画像缺陷。', '',
            f"held_out_validated：{report['held_out_validated']}", *['- '+v for v in report['limitations']]]
    return '\n'.join(lines)+'\n'


def render(report):
    payload=json.dumps(report,ensure_ascii=False,sort_keys=True,allow_nan=False).replace('<','\\u003c')
    return TEMPLATE.replace('REPORT_DATA',payload)


def write(report, directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    (directory/'fidelity.json').write_text(json.dumps(report,indent=2,ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n')
    (directory/'fidelity.md').write_text(markdown(report))
    (directory/'fidelity.html').write_text(render(report))


TEMPLATE=r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>合成流量 · 输入保真度</title><style>
:root{--bg:#F5F6FA;--card:#fff;--ink:rgba(0,0,0,.85);--muted:rgba(0,0,0,.65);--border:rgba(0,0,0,.06);--blue:#2563eb;--orange:#d97706;--green:#059669;--bad:#be123c;--small:12px;--body:14px;--title:32px;--gap:24px;--radius:8px}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:var(--body) -apple-system,BlinkMacSystemFont,'PingFang SC','Microsoft YaHei',sans-serif}main{max-width:1440px;margin:auto;padding:32px}h1{font-size:var(--title);margin:8px 0}h2{font-size:20px;margin:0 0 16px}h3{font-size:16px}p{line-height:1.7;color:var(--muted);overflow-wrap:anywhere}header{margin-bottom:var(--gap)}section,article{background:var(--card);border:1px solid var(--border);border-radius:var(--radius);padding:var(--gap);margin-bottom:var(--gap)}.grid{display:grid;grid-template-columns:1fr 1fr;gap:var(--gap)}.heat{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}.controls{display:flex;gap:16px;flex-wrap:wrap;align-items:center}select,input,button{font:inherit;padding:8px;border:1px solid var(--border);border-radius:4px;background:var(--card);color:var(--ink)}input{width:80px}button{cursor:pointer}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}td,th{text-align:right;padding:8px;border-bottom:1px solid var(--border)}td:first-child,th:first-child{text-align:left}code,pre{font-size:var(--small);overflow-wrap:anywhere}pre{white-space:pre-wrap;max-height:380px;overflow:auto}svg{width:100%;display:block}svg text{font:12px sans-serif;fill:var(--muted)}.PASS{color:var(--green)}.WARN{color:var(--orange)}.FAIL,.MISMATCH{color:var(--bad)}.legend{display:flex;gap:24px;font-weight:600}.real{color:var(--blue)}.independent{color:var(--orange)}.joint{color:var(--green)}.scroll{overflow:auto}.note{font-size:var(--small)}@media(max-width:850px){main{padding:16px}.grid,.heat{grid-template-columns:1fr}.controls{align-items:flex-start}h1{font-size:24px}}
</style><main><header><div>ONLINE EVAL / 独立诊断工具</div><h1>合成流量与真实输入，差在哪里？</h1><p>直接运行生产生成器，按实际标签重算共享结构。这里评估输入分布，不把理论共享当作有限容量缓存命中，也不代替性能实验。</p><div id="identity"></div></header>
<section><div class="controls"><label>捕获窗口 <select id="capture"></select></label><label>观察方法 <select id="method"><option value="joint">联合采样</option><option value="independent">旧版独立采样</option></select></label><button id="download">导出当前阈值 JSON</button></div><p id="window"></p><div class="legend"><span class="real">━ 真实输入</span><span class="independent">┄ 旧版独立采样</span><span class="joint">━ 联合采样</span></div></section>
<section><h2>能否用于当前实验？</h2><p id="verdict"></p><p class="note">长度 PASS 仅表示长度 KS 通过。cache / KV / 驱逐 / 拼车至少 WARN：时间复用、容量、路由与输出关联尚未验证，需要固定 trace 对照。阈值是可编辑的诊断策略，不是经过线上标定的版本门禁。</p><details><summary>调整诊断阈值（PASS 上限 / WARN 上限；超过 WARN 为 FAIL）</summary><div id="thresholds" class="controls"></div></details></section>
<div class="grid"><article><h2>输入长度 ECDF</h2><div id="length"></div><p class="note">横轴为 token 长度（log₂）；纵轴为累计请求占比。</p></article><article><h2>共享深度 ECDF</h2><div id="depth"></div><p class="note">横轴为 512-token block 数；包含首次出现家族的冷启动。</p></article></div>
<section><h2>长度 × 共享深度联合密度</h2><div class="heat" id="heat"></div><p class="note">横轴 log₂(token)：9–21，32 桶；纵轴 shared blocks：0–2000，16 桶，向上递增；溢出归入边界桶。颜色按请求占比、三图共用色标，悬停显示占比。</p></section>
<section><h2>分布与偏差</h2><div class="scroll" id="metrics"></div></section>
<section><h2>跨窗口观察</h2><p>固定画像对其他窗口的偏差包含预期流量漂移，不自动视作画像 bug。逐窗重新拟合仅是样本内诊断，不构成 held-out 验证。</p><div class="scroll" id="drift"></div></section>
<section><details><summary>画像身份、完整参数与适用边界</summary><pre id="provenance"></pre></details></section></main>
<script id="data" type="application/json">REPORT_DATA</script><script>
const report=JSON.parse(document.getElementById('data').textContent), $=id=>document.getElementById(id);
const colors={real:'#2563eb',independent:'#d97706',joint:'#059669'}, names={real:'真实输入',independent:'旧版独立采样',joint:'联合采样'};
const el=(tag,text,parent)=>{const x=document.createElement(tag);if(text!==undefined)x.textContent=text;if(parent)parent.appendChild(x);return x};
const fmt=x=>typeof x==='number'?(Math.abs(x)>=1000?x.toFixed(1):x.toFixed(4)):(x??'—');
const svgEl=(tag,attrs,parent)=>{const e=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const [k,v]of Object.entries(attrs))e.setAttribute(k,v);if(parent)parent.appendChild(e);return e};
let limits=JSON.parse(JSON.stringify(report.thresholds));
function grades(m,valid){if(!valid)return {length:'UNASSESSED',cache:'UNASSESSED'};const score={};for(const[k,v]of Object.entries(m))score[k]=v<=limits[k][0]?0:v<=limits[k][1]?1:2;const ns=['PASS','WARN','FAIL'];return {length:ns[score.length_ks],cache:ns[Math.max(1,...Object.values(score))]}}
function table(host,headers,rows){host.replaceChildren();const t=el('table',undefined,host),h=el('tr',undefined,t);headers.forEach(x=>el('th',x,h));for(const r of rows){const tr=el('tr',undefined,t);r.forEach(x=>el('td',typeof x==='number'?fmt(x):x,tr))}}
function series(row){return {real:row.real,...Object.fromEntries(Object.entries(row.methods).filter(([k,v])=>v.status==='MEASURED').map(([k,v])=>[k,v.summary]))}}
function ecdf(host,s,key,log){host.replaceChildren();const svg=svgEl('svg',{viewBox:'0 0 600 280',role:'img','aria-label':key},host);const max=Math.max(...Object.values(s).map(x=>x[key].at(-1)[0]));const xmax=log?Math.max(10,Math.log2(max)):Math.max(1,max),xmin=log?9:0;
const X=x=>48+530*((log?Math.log2(Math.max(512,x)):x)-xmin)/(xmax-xmin),Y=y=>238-y*210;
for(let i=0;i<=4;i++){const y=i/4;svgEl('line',{x1:48,y1:Y(y),x2:578,y2:Y(y),stroke:'#e5e7eb'},svg);svgEl('text',{x:8,y:Y(y)+4},svg).textContent=Math.round(y*100)+'%'}
for(let i=0;i<=4;i++){const v=xmin+(xmax-xmin)*i/4;svgEl('text',{x:48+530*i/4,y:264,'text-anchor':i===0?'start':i===4?'end':'middle'},svg).textContent=Math.round(log?2**v:v).toLocaleString()}
for(const[k,v]of Object.entries(s)){let c=0,d=`M48 ${Y(0)}`;for(const[x,n]of v[key]){d+=` H${X(x)} V${Y((c+=n)/v.requests)}`};svgEl('path',{d,fill:'none',stroke:colors[k],'stroke-width':2,'stroke-dasharray':k==='independent'?'5 3':''},svg)}}
function draw(){const row=report.rows[+$('capture').value];if(!row){$('verdict').textContent='UNASSESSED：无对照捕获。仅展示画像参数及身份检查，不能据此判断保真度。';return}const s=series(row),m=row.methods[$('method').value];$('window').textContent=`${row.interpretation} · ${row.real.requests.toLocaleString()} 个真实请求 · SHA ${row.capture_sha256}`;
if(m?.status==='MEASURED'){const g=grades(m.metrics,row.audit.status==='OK');$('verdict').textContent=`长度分布 ${g.length} / cache 实验 ${g.cache}　|　length KS ${fmt(m.metrics.length_ks)}　depth KS ${fmt(m.metrics.depth_ks)}　joint TV ${fmt(m.metrics.joint_tv)}`}else $('verdict').textContent='UNASSESSED：画像未提供联合分布，请显式重新标定。';
ecdf($('length'),s,'length_counts',true);ecdf($('depth'),s,'depth_counts',false);$('heat').replaceChildren();const peak=Math.max(...Object.values(s).flatMap(v=>v.joint_counts.flat().map(n=>n/v.requests)),1e-9);
for(const[k,v]of Object.entries(s)){const box=el('div',undefined,$('heat'));el('h3',names[k],box);const svg=svgEl('svg',{viewBox:'0 0 340 200',role:'img','aria-label':names[k]+'联合密度'},box);for(let x=0;x<32;x++)for(let y=0;y<16;y++){const mass=v.joint_counts[x][y]/v.requests;const rect=svgEl('rect',{x:20+x*9.5,y:160-y*9,width:9.5,height:9,fill:colors[k],'fill-opacity':.05+.95*Math.sqrt(mass/peak)},svg);svgEl('title',{},rect).textContent=`log₂长度 ${9+x*12/32}，深度 ${y*125}–${(y+1)*125}：${(mass*100).toFixed(3)}%`}svgEl('text',{x:20,y:190},svg).textContent='512 token → 2M token';svgEl('text',{x:20,y:12},svg).textContent='↑ 2000 blocks'}
const keys=Object.keys(s), fields=[['请求数',v=>v.requests],['输入均值',v=>v.input_tokens.mean],...['p50','p90','p99'].map(p=>['输入 '+p,v=>v.input_tokens[p]]),['深度均值',v=>v.depth_blocks.mean],...['p50','p90','p99'].map(p=>['深度 '+p,v=>v.depth_blocks[p]]),['冷启动比例',v=>v.cold_fraction],['token 加权共享',v=>v.token_weighted_sharing],['实际前8块家族数',v=>v.families],['实际家族 top5',v=>v.top5],['corr(log₂长度,深度)',v=>v.corr_log_length_depth]];
const lines=fields.map(([name,get])=>[name,...keys.map(k=>get(s[k]))]);for(const key of ['length_ks','depth_ks','joint_tv'])lines.push([key,...keys.map(k=>k==='real'?'—':row.methods[k].metrics[key])]);table($('metrics'),['指标',...keys.map(k=>names[k])],lines);
table($('drift'),['捕获','方法','口径','length KS','depth KS','joint TV'],report.rows.flatMap(r=>Object.entries(r.methods).filter(([k,m])=>m.status==='MEASURED').map(([k,m])=>[r.capture,names[k],r.role,m.metrics.length_ks,m.metrics.depth_ks,m.metrics.joint_tv])));
$('provenance').textContent=JSON.stringify({identity:row.audit,profile:report.profile,profile_sha256:report.profile_sha256,fit_sha256:row.fit_sha256,held_out_validated:report.held_out_validated,limitations:report.limitations,parameters:row.fit_parameters,seed:report.seed,mode:report.mode,thresholds:limits,bins:report.bins},null,2)}
$('identity').textContent=`画像身份 ${report.identity.status} · ${report.mode} · seed ${report.seed}`;
$('provenance').textContent=JSON.stringify(report,null,2);
report.rows.forEach((r,i)=>{const opt=el('option',r.capture,$('capture'));opt.value=i});
for(const[key,values]of Object.entries(limits)){const label=el('label',key+' ',$('thresholds'));values.forEach((v,i)=>{const input=el('input',undefined,label);input.type='number';input.min=0;input.max=1;input.step=.01;input.value=v;input.setAttribute('aria-label',key+(i?' warn max':' pass max'));input.onchange=()=>{const n=Number(input.value);if(!Number.isFinite(n)||n<0||n>1||(i===0&&n>limits[key][1])||(i===1&&n<limits[key][0])){input.value=limits[key][i];return}limits[key][i]=n;draw()}})}
$('capture').onchange=draw;$('method').onchange=draw;$('download').onclick=()=>{const blob=new Blob([JSON.stringify(limits,null,2)],{type:'application/json'}),url=URL.createObjectURL(blob),a=el('a');a.href=url;a.download='fidelity-thresholds.json';a.click();URL.revokeObjectURL(url)};draw();
</script></html>'''
