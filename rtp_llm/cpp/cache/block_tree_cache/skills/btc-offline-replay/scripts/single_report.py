"""Standalone report for one verified BTC replay."""
from datetime import datetime, timedelta, timezone
from html import escape
import json
from pathlib import Path
import struct

FACT = struct.Struct("<6Q")
KEYS = ("input", "full", "joint", "host", "disk", "device", "requests")
NAMES = dict(full="FULL", joint="联合", host="Host", disk="Disk", device="Device")
COLORS = dict(full="#236aae", joint="#16826f", host="#aa550c", disk="#8052a8", device="#236aae")
TZ = timezone(timedelta(hours=8))
CSS = """
body{font:16px/1.65 system-ui,-apple-system,sans-serif;color:#183447;background:#f7f9fb;margin:0}
main{max-width:1180px;margin:36px auto;padding:0 24px}h1{font-size:38px}h2{font-size:27px;margin:0}
section{background:white;border:1px solid #dfe7ee;border-radius:14px;padding:24px;margin:24px 0}
dl{display:grid;grid-template-columns:max-content 1fr;gap:.35em 1.4em}dd{margin:0;overflow-wrap:anywhere}
dt{color:#506779}summary{cursor:pointer;font-weight:600}.scroll{overflow:auto}svg{width:100%;min-width:840px}
table{width:100%;border-collapse:collapse;white-space:nowrap}th,td{text-align:right;padding:.5em;border-bottom:1px solid #dde5ea}
text{font:13px system-ui,sans-serif}.label{font-weight:600;paint-order:stroke;stroke:white;stroke-width:3px}
"""


def read_json(path):
    return json.loads(path.read_bytes())


def stamp(ns):
    sec, nanos = divmod(ns, 10**9)
    return datetime.fromtimestamp(sec, TZ).strftime("%Y-%m-%d %H:%M:%S") + f".{nanos//10**6:03d}"


def empty():
    return dict.fromkeys(KEYS, 0)


def aggregate(paths, total):
    if not paths:
        return [], [], []
    def endpoint(path, offset):
        with path.open("rb") as stream:
            stream.seek(offset, 0 if offset == 0 else 2)
            return FACT.unpack(stream.read(FACT.size))[0]
    start, last = endpoint(paths[0], 0), endpoint(paths[-1], -FACT.size)
    span = last - start + 1
    edges = [start + (span*i+9)//10 for i in range(11)]
    bins = [empty() for _ in range(10)]
    previous = start
    for path in paths:
        with path.open("rb") as stream:
            for batch in iter(lambda: stream.read(FACT.size*8192), b""):
                for ts, inp, full, joint, host, disk in FACT.iter_unpack(batch):
                    if not previous <= ts <= last or not host+disk <= joint <= full <= inp:
                        raise ValueError("invalid arrival order or match counts")
                    previous = ts
                    row = bins[(ts-start)*10//span]
                    for key, value in zip(KEYS,(inp,full,joint,host,disk,joint-host-disk,1)):
                        row[key] += value
    running = empty()
    cumulative = []
    for row in bins:
        running = {k:running[k]+row[k] for k in KEYS}
        cumulative.append(running)
    if (running["requests"],running["input"],running["full"],running["joint"]) != (
        total["requests"],total["input_tokens"],total["full_tokens"]["sum"],total["joint_tokens"]["sum"]):
        raise ValueError("arrival records do not reproduce verified replay totals")
    return edges,bins,cumulative


def graph(title, keys, edges, rows):
    elements = [f'<h3>{title}</h3>', '　'.join(f'<span style="color:{COLORS[k]}">● {NAMES[k]}</span>' for k in keys),
                f'<div class="scroll"><svg viewBox="0 0 1100 260" role="img" aria-label="{title}">']
    for i in range(6):
        y = 205-36*i
        elements.append(f'<path d="M64 {y}H1060" stroke="#dfe7ee"/><text x="55" y="{y+4}" text-anchor="end">{20*i}%</text>')
    if rows:
        label_positions = [[] for _ in rows]
        for k in keys:
            points, labels = [], []
            for i,row in enumerate(rows):
                if not row["input"]:
                    points.append(None)
                    continue
                pct = 100*row[k]/row["input"]
                x, y = 82+106*i,205-1.8*pct
                points.append((x,y))
                label_y = y-7
                while any(abs(label_y-p)<15 for p in label_positions[i]):
                    label_y += 15
                label_positions[i].append(label_y)
                labels.append(f'<circle cx="{x}" cy="{y}" r="3.5" fill="{COLORS[k]}"/>'
                              f'<text class="label" x="{x}" y="{label_y}" text-anchor="middle" fill="{COLORS[k]}">{pct:.3f}%</text>')
            path = " ".join(f'{"M" if i==0 or points[i-1] is None else "L"}{p[0]} {p[1]}' for i,p in enumerate(points) if p)
            elements.append(f'<path d="{path}" stroke="{COLORS[k]}" stroke-width="2" fill="none"/>')
            elements.extend(labels)
        for i,end in enumerate(edges[1:]):
            label = stamp(end).split()[1]
            elements.append(f'<text x="{82+106*i}" y="246" text-anchor="middle">{label}</text>')
    else:
        elements.append('<text x="540" y="122" text-anchor="middle">此回放未记录完整的逐请求到达时间和分层命中计数</text>')
    elements.append('</svg></div><p>横轴为请求到达时间（北京时间），各点标在窗口结束处。</p>')
    if rows:
        elements.append('<details><summary>原始数据</summary><div class="scroll"><table><tr><th>截至时间</th><th>请求数</th><th>输入 tokens</th>')
        elements.extend(f'<th>{NAMES[k]} 命中 tokens / 命中率</th>' for k in keys)
        elements.append('</tr>')
        for ts,row in zip(edges[1:],rows):
            elements.append(f'<tr><td>{stamp(ts)}</td><td>{row["requests"]:,}</td><td>{row["input"]:,}</td>')
            for k in keys:
                rate = f'{100*row[k]/row["input"]:.4f}%' if row["input"] else '—'
                elements.append(f'<td>{row[k]:,} / {rate}</td>')
            elements.append('</tr>')
        elements.append('</table></div></details>')
    return ''.join(elements)


def render_run(run, output):
    identity = read_json(run/'identity.json')
    total = read_json(run/'native/summary.json')["total"]
    params = read_json(run/'parameters.json')
    env_path = run/'report-environment.json'
    env = read_json(env_path) if env_path.is_file() else {}
    source = identity.get("source", {})
    paths = sorted((run/'report-stages').glob('*.u64'))
    edges,bins,cumulative = aggregate(paths,total)
    facts = [
        ("日志来源",env.get("source") or (source.get("source") if isinstance(source,dict) else source)),
        ("采集时间",env.get("capture_time")),("采集集群配置",env.get("cluster")),("Prefill 节点 IP",env.get("prefill_ip")),
        ("缓存实例",env.get("instance")),("BTC 提交",identity["revision"]),("tokens / block",env.get("base_tokens")),
        ("预热请求",f'{identity["warmup"]:,}'),("正式请求",f'{total["requests"]:,}'),
        ("正式输入 tokens",f'{total["input_tokens"]:,}'),
        ("正式统计窗口（北京时间）",f'{stamp(edges[0])} 至 {stamp(edges[-1]-1)}' if edges else None)]
    for tier, pools in params.get("tiers",{}).items():
        facts.append((f'{tier.title()} 容量（FULL / SWA 块）',f'{pools["full"]:,} / {pools["swa"]:,}'))
    def fields(items):
        return '<dl>'+''.join(f'<dt>{escape(k)}</dt><dd>{escape(str(v)) if v is not None else "未记录"}</dd>' for k,v in items)+'</dl>'
    html = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
            '<title>BTC 离线命中率</title><style>'+CSS+'</style><main><h1>BTC 离线命中率</h1><section><h2>测试环境</h2>',
            fields(facts),'<details><summary>回放配置</summary><pre>'+escape(json.dumps(params,indent=2,ensure_ascii=False))+'</pre></details></section>']
    for title,keys in [('联合命中率',['joint']),('FULL 命中率',['full']),('分层命中率',['device','host','disk'])]:
        html.append('<section><h2>'+title+'</h2>')
        if len(keys)==1:
            metric=total[keys[0]+'_tokens']['sum']
            html.append(f'<p>正式区间累计 {100*metric/total["input_tokens"]:.4f}%（{metric:,} / {total["input_tokens"]:,} tokens）。</p>')
        else:
            html.append('<p>按同一次联合匹配的来源归属计算，三层使用相同的输入 token 分母。</p>')
        for prefix,records in [('累计',cumulative),('分段',bins)]:
            html.append(graph(prefix+title,keys,edges,records))
        html.append('</section>')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(''.join(html)+'</main></html>',encoding='utf-8')
