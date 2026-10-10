"""Offline plots from the same audited request-list observations used for fitting."""

import html
import json
from pathlib import Path


def write_charts(rows, output, *, cards=1, partial=False, model_label="Model"):
    if cards <= 0:
        raise ValueError("cards must be positive")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    try:
        from plotly.offline import get_plotlyjs
    except ImportError:
        return _write_fallback_charts(
            rows,
            output,
            partial=partial,
            model_label=model_label,
        )
    page = """<!doctype html><meta charset="utf-8"><title>Prefill measurements</title>
<h1>__TITLE__</h1><p>__STATUS__ · 服务端首 token 延迟，含引擎等待；每轮取批内最大值。</p>
<label>Batch <select id="batch"><option value="all">全部</option></select></label>
<label>长度 <select id="length"><option value="sum">整批总量</option><option value="mean">每请求均值</option></select></label>
<label>样本 <select id="samples"><option value="case">Case 聚合</option><option value="runs">每轮观测</option></select></label>
<label>指标 <select id="metric"><option value="latency">服务端延迟 (ms)</option><option value="effective">总输入 TPM/card</option><option value="compute">计算 TPM/card</option></select></label>
<p id="count"></p><div id="plot" style="height:75vh"></div><pre id="detail"></pre>
<p>全部 batch：X=compute，Y=cache，Z=batch；固定 batch：Z=所选指标。颜色表示所选指标。
长度单位 Ki tokens。TPM 使用整批 token / 服务端延迟 / 卡数，不代表客户端端到端吞吐。</p>
<script>__PLOTLY__</script><script>
const rows=__ROWS__,cards=__CARDS__;
for(const b of [...new Set(rows.map(r=>r.batch_size))].sort((a,b)=>a-b)) {
 const o=document.createElement('option');o.value=b;o.textContent=b;document.getElementById('batch').append(o);
}
function draw(){
 const batch=document.getElementById('batch').value,mean=document.getElementById('length').value==='mean';
 const metric=document.getElementById('metric').value,allruns=document.getElementById('samples').value==='runs';
 const points=rows.filter(r=>batch==='all'||r.batch_size===Number(batch)).flatMap(r=>
 (allruns?r.formal_rounds_ms:[r.target_ms]).map((t,i)=>({r,t,round:allruns?i+1:null})));
 const values=points.map(p=>metric==='latency'?p.t:(metric==='compute'?p.r.compute_len:p.r.input_len)*60000/p.t/cards);
 Plotly.react('plot',[{type:'scatter3d',mode:'markers',x:points.map(p=>p.r.compute_len/(mean?p.r.batch_size:1)/1024),
 y:points.map(p=>p.r.cache_len/(mean?p.r.batch_size:1)/1024),z:batch==='all'?points.map(p=>p.r.batch_size):values,
 customdata:points,marker:{size:4,color:values,colorscale:'Viridis',colorbar:{title:metric}},
 text:points.map(p=>p.r.source_run+' case '+p.r.case_id),hovertemplate:'%{text}<extra></extra>'}],
 {scene:{xaxis:{title:'Compute (Ki tokens)'},yaxis:{title:'Observed cache (Ki tokens)'},zaxis:{title:batch==='all'?'Batch':metric}},margin:{t:20}}, {responsive:true});
 document.getElementById('count').textContent=points.length+' 个观测点 · cards='+cards;
 const plot=document.getElementById('plot');plot.removeAllListeners('plotly_click');
 plot.on('plotly_click',e=>document.getElementById('detail').textContent=JSON.stringify(e.points[0].customdata,null,2));
}
['batch','length','samples','metric'].forEach(id=>document.getElementById(id).onchange=draw);draw();
</script>"""
    page = page.replace("__TITLE__", html.escape(model_label))
    page = page.replace("__STATUS__", "部分数据预览" if partial else "完整运行结果")
    page = page.replace("__PLOTLY__", get_plotlyjs()).replace("__CARDS__", str(cards))
    page = page.replace(
        "__ROWS__", json.dumps(rows, ensure_ascii=False).replace("</", "<\\/")
    )
    latency = output / "latency.interactive.html"
    latency.write_text(page, encoding="utf-8")
    tpm = output / "tpm.interactive.html"
    tpm.write_text(
        page.replace(
            "['batch','length','samples','metric'].forEach",
            "document.getElementById('metric').value='effective';\n['batch','length','samples','metric'].forEach",
        ),
        encoding="utf-8",
    )
    return {"html": str(latency), "tpm_html": str(tpm)}


def write_scatter_svg(rows, path, *, cold=False):
    """Dependency-free projected 3D scatter (cold slice is two-dimensional)."""
    selected = [r for r in rows if not cold or r["cache_len"] == 0]
    xmax = max((r["compute_len"] for r in selected), default=1)
    cmax = max(1, max((r["cache_len"] for r in selected), default=0))
    ymax = max((r["target_ms"] for r in selected), default=1)

    def point(compute, cache, latency):
        return (
            75 + 510 * compute / xmax + 160 * cache / cmax,
            440 - 100 * cache / cmax - 330 * latency / ymax,
        )

    palette = ("#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c")
    batches = sorted({r["batch_size"] for r in selected})
    colors = {b: palette[i % len(palette)] for i, b in enumerate(batches)}
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="850" height="550">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="55" y="25">Measured server latency by compute/cache; color = batch</text>',
    ]
    for compute, cache, latency, label in (
        (xmax, 0, 0, f"Compute: {xmax}"),
        (0, 0, ymax, f"Latency: {ymax:.3f} ms"),
        *(([(0, cmax, 0, f"Cache: {cmax}")] if not cold else [])),
    ):
        x, y = point(compute, cache, latency)
        parts.append(f'<path d="M75 440 L{x} {y}" stroke="#64748b" fill="none"/>')
        parts.append(f'<text x="{x}" y="{y + 18}" font-size="12">{label}</text>')
    for row in selected:
        x, y = point(row["compute_len"], row["cache_len"], row["target_ms"])
        label = html.escape(
            f'{row["source_run"]} case {row["case_id"]} B={row["batch_size"]}, {row["target_ms"]:.3f} ms'
        )
        parts.append(
            f'<circle cx="{x}" cy="{y}" r="3" fill="{colors[row["batch_size"]]}"><title>{label}</title></circle>'
        )
    for index, batch in enumerate(batches):
        parts.append(
            f'<text x="{55 + 80 * (index % 9)}" y="{490 + 20 * (index // 9)}" fill="{colors[batch]}">B={batch}</text>'
        )
    if not selected:
        parts.append('<text x="300" y="220">No valid observations</text>')
    parts.append("</svg>")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(parts), encoding="utf-8")


def _write_fallback_charts(rows, output, *, partial=False, model_label="Model"):
    """Keep the report usable when the optional Plotly package is unavailable."""
    fallback_svg = output / "prefill_3d.fallback.svg"
    write_scatter_svg(rows, fallback_svg)
    inline_svg = fallback_svg.read_text(encoding="utf-8")
    page = """<!doctype html><meta charset="utf-8"><title>Prefill measurements</title>
<h1>__TITLE__</h1>
<p>__STATUS__ · Interactive charts are unavailable because Plotly is not installed.</p>
<p>The dependency-free projected latency chart is shown below.</p>
__SVG__
"""
    page = page.replace("__TITLE__", html.escape(model_label))
    page = page.replace("__STATUS__", "部分数据预览" if partial else "完整运行结果")
    page = page.replace("__SVG__", inline_svg)
    latency = output / "latency.interactive.html"
    latency.write_text(page, encoding="utf-8")
    tpm = output / "tpm.interactive.html"
    tpm.write_text(page, encoding="utf-8")
    return {
        "html": str(latency),
        "tpm_html": str(tpm),
        "fallback_svg": str(fallback_svg),
    }
