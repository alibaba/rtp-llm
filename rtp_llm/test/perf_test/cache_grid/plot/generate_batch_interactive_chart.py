#!/usr/bin/env python3
"""Generate a batch/cache/compute HTML directly from cache_grid_results.json.

Each measurement round uses the maximum server first_token_cost_time in the batch.
No GPU, serving libraries, network/CDN, or experiment-specific paths are required.
"""
import argparse
import html
import json
import math
from pathlib import Path
from statistics import median

from plotly.offline import get_plotlyjs

from rtp_llm.test.perf_test.cache_grid.runner.result_schema import (
    current_metrics,
    run_prefill_rt,
)

CONTRACT = "batch_max_server_first_token_cost_time_ms"


def analyze_results(data):
    """Each round is max(request server first-token latency); aggregate rounds by median."""
    rows = []
    for item in current_metrics(data):
        if item["status"] != "ok" or "request_groups" not in item:
            continue
        batch = item["batch_size"]
        if type(batch) is not int or batch <= 0:
            continue
        runs = item["runs"]
        if (
            not runs
            or len(runs) != item["measure_runs"]
            or item["success_runs"] != len(runs)
        ):
            continue
        rounds, latencies, distributions = [], [], []
        for run in runs:
            requests = run.get("requests", [])
            if run.get("valid") is not True or len(requests) != batch:
                break
            times = [run_prefill_rt(request) for request in requests]
            if any(value is None for value in times) or any(
                request.get("success") is not True
                or request.get("output_len") != 1
                or type(request.get("input_len")) is not int
                or type(request.get("reuse_len")) is not int
                or not 0 <= request["reuse_len"] < request["input_len"]
                for request in requests
            ):
                break
            rounds.append(max(times))
            latencies.extend(times)
            distributions.append(
                [
                    [r["input_len"], r["reuse_len"], r["input_len"] - r["reuse_len"]]
                    for r in requests
                ]
            )
        if len(rounds) != len(runs) or any(
            d != distributions[0] for d in distributions
        ):
            continue
        distribution = distributions[0]
        caches, computes = [r[1] for r in distribution], [r[2] for r in distribution]
        rows.append(
            dict(
                case_id=item["case_id"],
                batch_size=batch,
                band=item.get("band", "measured"),
                cached_tokens=sum(caches),
                compute_tokens=sum(computes),
                mean_cached_tokens=sum(caches) / batch,
                mean_compute_tokens=sum(computes) / batch,
                min_cached_tokens=min(caches),
                max_cached_tokens=max(caches),
                min_compute_tokens=min(computes),
                max_compute_tokens=max(computes),
                median_batch_first_token_cost_time_ms=median(rounds),
                min_batch_first_token_cost_time_ms=min(rounds),
                max_batch_first_token_cost_time_ms=max(rounds),
                p95_request_first_token_cost_time_ms=sorted(latencies)[
                    math.ceil(len(latencies) * 0.95) - 1
                ],
                formal_rounds_ms=rounds,
                request_distribution=distribution,
            )
        )
    return rows


def render_html(rows, partial=False, model_label="Model"):
    page = r"""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>__MODEL__ batch / cache / compute / server first-token latency</title>
<style>
body{font:15px system-ui;margin:0;background:#f4f7fb;color:#183049}main{max-width:1400px;margin:auto;padding:28px}
h1{font-size:30px;margin:0 0 8px}p{line-height:1.6;color:#526477}.card{background:white;border:1px solid #dce5ef;border-radius:14px;padding:18px;margin-top:18px}
.controls{display:flex;gap:18px;align-items:center;flex-wrap:wrap}select,input,button{font:inherit;padding:6px}
#plot{height:670px}#trend{height:450px}.small{font-size:13px}#detail{white-space:pre-wrap;font-size:13px;max-height:260px;overflow:auto}
</style><main><h1>Batch × Cache × Compute → Server first-token latency</h1>
<p>随机混合 batch · 服务端 first_token_cost_time · __COUNT__ 个随机混合 batch。__PREVIEW__</p>
<div class="card controls">
<label>长度统计 <select id="view"><option value="mean">每请求平均 cache / compute</option>
<option value="sum">整批总 cache / compute</option></select></label>
<span>全部 batch：Compute / Cache / Batch；单个 batch：Compute / Cache / 服务端首 token 延迟。颜色始终表示服务端首 token 延迟。</span>
<label>Batch <select id="batch"><option value="all">全部</option></select></label>
<label>采样层 <select id="band"><option value="all">全部</option><option>low/low</option><option>low/high</option><option>high/low</option><option>high/high</option><option>measured</option></select></label>
<button id="reset">重置</button><span id="count"></span></div>
<div class="card"><p id="axis-legend" aria-live="polite"></p><div id="plot"></div><p class="small">拖动旋转，滚轮缩放，点击点查看批内每条请求；右上工具栏可导出图片。仅绘制实测点，不插值未测量曲面。</p>
<pre id="detail">点击一个点查看请求分布和各轮服务端首 token 延迟。</pre></div>
<div class="card"><div id="trend"></div></div>
<p>每轮取批内请求的服务端 first_token_cost_time 最大值，再对有效测量轮次取中位数。
包括引擎内等待，不包含输入分词、客户端编码及通信；未扣减 wait_time。这里不使用客户端整批 wall time。
输入只包含正式测量轮次，不在绘图时额外丢弃首轮。</p>
<p>low/high 为 cache/compute 采样层。不同 batch 的请求分布不完全相同，趋势线只描述观测，不能单独解释为 cache 的因果收益。
数据来源：cache_grid_results.json。</p></main>
<script>__PLOTLY__</script><script>
const rows=__DATA__, palette={"low/low":"#2878b5","low/high":"#24a885","high/low":"#e0a128","high/high":"#cc5476","measured":"#7358a6"};
for(const b of [...new Set(rows.map(r=>r.batch_size))].sort((a,b)=>a-b)){let o=document.createElement('option');o.value=b;o.textContent=b;document.getElementById('batch').append(o)}
function draw(){
 const view=document.getElementById('view').value,b=document.getElementById('batch').value,band=document.getElementById('band').value;
 const data=rows.filter(r=>(b==='all'||r.batch_size==b)&&(band==='all'||r.band===band));
 document.getElementById('count').textContent=data.length+' 个点';
 const singleBatch=b!=='all', sum=view==='sum';
 const cache=data.map(r=>(sum?r.cached_tokens:r.mean_cached_tokens)/1024);
 const compute=data.map(r=>(sum?r.compute_tokens:r.mean_compute_tokens)/1024);
 const color=data.map(r=>r.median_batch_first_token_cost_time_ms),title='批内最大服务端首 token 延迟中位数 (ms)';
 const cacheLabel=(sum?'整批总':'每请求平均')+'实际 cache (Ki tokens)';
 const computeLabel=(sum?'整批总':'每请求平均')+' compute (Ki tokens)';
 const x=compute,y=cache,z=singleBatch?color:data.map(r=>r.batch_size);
 const axis=[computeLabel,cacheLabel,singleBatch?title:'Batch 大小 (请求数)'];
 document.getElementById('axis-legend').textContent='X：'+axis[0]+' ｜ Y：'+axis[1]+' ｜ Z：'+axis[2]+' ｜ 颜色：'+title+'（颜色越深，耗时越长）'+'。1 Ki tokens = 1024 tokens；compute = input − 实际 cache。';
 const text=data.map(r=>'case '+r.case_id+' · B='+r.batch_size+' · '+r.band+'<br>Server first-token median '+r.median_batch_first_token_cost_time_ms.toFixed(2)+' ms ['+r.min_batch_first_token_cost_time_ms.toFixed(2)+', '+r.max_batch_first_token_cost_time_ms.toFixed(2)+']<br>cache/request '+r.mean_cached_tokens.toFixed(1)+' ['+r.min_cached_tokens+','+r.max_cached_tokens+']<br>compute/request '+r.mean_compute_tokens.toFixed(1)+' ['+r.min_compute_tokens+','+r.max_compute_tokens+']<br>request P95 server first-token '+r.p95_request_first_token_cost_time_ms.toFixed(2)+' ms');
 Plotly.react('plot',[{type:'scatter3d',mode:'markers',x,y,z,text,customdata:data,hovertemplate:'%{text}<extra></extra>',marker:{size:5,opacity:.92,color,colorscale:'Viridis',reversescale:true,cmin:Math.min(...rows.map(r=>r.median_batch_first_token_cost_time_ms)),cmax:Math.max(...rows.map(r=>r.median_batch_first_token_cost_time_ms)),colorbar:{title:{text:title}},line:{width:.3,color:'#ffffff'}}}],{margin:{l:45,r:100,t:35,b:55},scene:{xaxis:{title:{text:axis[0],font:{size:13}}},yaxis:{title:{text:axis[1],font:{size:13}}},zaxis:{title:{text:axis[2],font:{size:13}}},camera:{eye:{x:1.7,y:1.6,z:1.2}}},uirevision:(singleBatch?'single':'all')+':'+view},{responsive:true,displaylogo:false});
 document.getElementById('plot').removeAllListeners('plotly_click');
 document.getElementById('plot').on('plotly_click',event=>{const r=event.points[0].customdata;document.getElementById('detail').textContent='case '+r.case_id+' · B='+r.batch_size+' · '+r.band+'\nFormal rounds (ms): '+r.formal_rounds_ms.map(v=>v.toFixed(3)).join(', ')+'\nrequest | input | observed cache | compute\n'+r.request_distribution.map((v,i)=>[i,...v].join(' | ')).join('\n')});
 const traces=Object.keys(palette).map(key=>{const rs=data.filter(r=>r.band===key);return {type:'scatter',mode:'lines+markers',name:key+' cache/compute',x:rs.map(r=>r.batch_size),y:rs.map(r=>r.median_batch_first_token_cost_time_ms),line:{color:palette[key]},error_y:{type:'data',symmetric:false,array:rs.map(r=>r.max_batch_first_token_cost_time_ms-r.median_batch_first_token_cost_time_ms),arrayminus:rs.map(r=>r.median_batch_first_token_cost_time_ms-r.min_batch_first_token_cost_time_ms),thickness:1,width:2}}});
 Plotly.react('trend',traces,{title:{text:'固定采样层内的随机混合 batch（误差线为各轮 min–max）'},xaxis:{title:{text:'Batch 大小 (请求数)'},automargin:true},yaxis:{title:{text:'批内最大服务端首 token 延迟中位数 (ms)'},automargin:true},legend:{orientation:'h'},margin:{t:60,b:70,l:80,r:20}},{responsive:true,displaylogo:false});
}
['view','batch','band'].forEach(id=>document.getElementById(id).onchange=draw);
document.getElementById('reset').onclick=()=>{document.getElementById('view').value='mean';document.getElementById('batch').value='all';document.getElementById('band').value='all';draw()};
draw();
</script></html>"""
    page = page.replace("__COUNT__", str(len(rows))).replace(
        "__PREVIEW__", "部分数据预览" if partial else "全部测量完成"
    )
    page = page.replace("__PLOTLY__", get_plotlyjs()).replace(
        "__DATA__", json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    )
    return page.replace("__MODEL__", html.escape(model_label))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, required=True, help="Raw cache_grid_results.json"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Offline HTML output"
    )
    parser.add_argument(
        "--partial", action="store_true", help="Label incomplete data as preview"
    )
    parser.add_argument("--model-label", default=None)
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    data = {
        "measurement_contract": CONTRACT,
        "rows": analyze_results(data),
        "model_label": data.get("model_label"),
    }
    if not data.get("rows"):
        parser.error("No chart rows")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        render_html(
            data["rows"],
            args.partial,
            args.model_label or data.get("model_label") or "Model",
        ),
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
