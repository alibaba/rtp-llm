"""Render the approved request-window report from verified native replay facts."""
import argparse
from datetime import datetime,timezone,timedelta
import html
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/btc-request-report-matplotlib')
COLORS={'theory':'#7652a5','full':'#2775b6','joint':'#13846f'}
LABELS={'theory':'理论：无限容量、SWA全保留','full':'BTC FULL','joint':'BTC 联合'}
STYLES={'theory':('-.','D'),'full':('-','o'),'joint':('--','x')}
TZ=timezone(timedelta(hours=8))

def esc(value):return html.escape(str(value))
def number(value):
    if value is None:return 'N/A'
    for scale,suffix in [(1e12,'T'),(1e9,'B'),(1e6,'M'),(1e3,'K')]:
        if abs(value)>=scale:return f'{value/scale:.3f}{suffix}'
    return f'{value:,.3f}' if isinstance(value,float) else f'{value:,}'
def exact_number(value):
    return f'{value:,.3f}' if isinstance(value,float) else f'{value:,}'
def percent(value):return 'N/A' if value is None else f'{value*100:.4f}%'
def time_text(ns):return datetime.fromtimestamp(ns//1_000_000_000,TZ).strftime('%Y-%m-%d %H:%M:%S') + f'.{(ns//1000000)%1000:03d}'
def cell(row,key):
    for part in key.split('.'):
        row=row[part]
    return row

def table(rows,columns):
    head=''.join('<th>'+esc(label)+'</th>' for label,_,_ in columns)
    body=''.join('<tr>'+''.join('<td>'+esc(fmt(cell(row,key)))+'</td>' for _,key,fmt in columns)+'</tr>' for row in rows)
    return '<div class="table-scroll"><table><thead><tr>'+head+'</tr></thead><tbody>'+body+'</tbody></table></div>'

def plot(out,name,title,rows,series,percentage=False,missing=None):
    fig,ax=plt.subplots(figsize=(12,4.6))
    from matplotlib import dates
    xs = [datetime.fromtimestamp(row["plot_ns"] // 1_000_000_000, TZ)
          + timedelta(microseconds=(row["plot_ns"] % 1_000_000_000) // 1000) for row in rows]
    any_value = False
    for key,label,color,style,marker in series:
        ys=[cell(row,key) if cell(row,key) is not None else float('nan') for row in rows]
        if not any(y == y for y in ys):
            continue
        any_value = True
        ax.plot(xs,ys,label=label,color=color,linestyle=style,marker=marker,linewidth=2,markersize=7,
                markerfacecolor='white' if marker=='o' else color)
        for i,(x,y) in enumerate(zip(xs,ys)):
            if len(xs) > 6 and i not in (0, len(xs)//2, len(xs)-1):
                continue
            if y==y:
                offset=32 if color==COLORS['theory'] else (-16 if marker=='x' else 10)
                ax.annotate(f'{y*100:.3f}%' if percentage else number(y),(x,y),xytext=(0,offset),
                            textcoords='offset points',ha='center',color=color,fontsize=10)
    ax.set_title(title,fontsize=20,fontweight='bold',pad=26)
    ax.set_xlabel('Request 到达时间（北京时间）',labelpad=10)
    ax.set_ylabel('比例' if percentage else ('平均差值（tokens/request）' if name=='match-gap' else '匹配长度（tokens）'),labelpad=10)
    step = max(1, (len(xs) - 1 + 4) // 5)
    ticks = xs[::step]
    if ticks[-1] != xs[-1] and (len(xs) - 1) % step >= step / 2:
        ticks.append(xs[-1])
    ax.set_xticks(ticks)
    if (xs[-1] - xs[0]).total_seconds() < 10:
        ax.set_xticklabels([x.strftime('%H:%M:%S.%f')[:-3] for x in ticks])
    else:
        ax.xaxis.set_major_formatter(dates.DateFormatter('%m-%d\n%H:%M:%S', tz=TZ))
    padding = (xs[-1] - xs[0]) * .025 if len(xs) > 1 else timedelta(milliseconds=500)
    ax.set_xlim(xs[0] - padding, xs[-1] + padding)
    ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    if percentage:ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    elif name=='match-gap':ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda value,_:f'{value:g}'))
    else:ax.yaxis.set_major_formatter(ticker.EngFormatter())
    ax.spines[['top','right']].set_visible(False)
    ax.margins(x=.1,y=.22)
    if not any_value:
        ax.set_ylim(0, 1)
        ax.text(.5,.5,missing or '当前区间缺少可计算的数据',transform=ax.transAxes,ha='center',va='center')
    else:
        ax.legend(loc='upper center',bbox_to_anchor=(.5,-.2),ncol=3,frameon=False,fontsize=10)
    fig.tight_layout()
    fig.savefig(out/(name+'.svg'),bbox_inches='tight')
    fig.savefig(out/(name+'.png'),dpi=130,bbox_inches='tight')
    plt.close(fig)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('metrics',type=Path);p.add_argument('output',type=Path)
    p.add_argument("--reuse-plots", action="store_true", help="keep existing plots for a table-only update")
    args=p.parse_args();data=json.loads(args.metrics.read_text());out=args.output;out.mkdir(parents=True,exist_ok=True)
    if not args.reuse_plots:
        global plt, ticker
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import font_manager,ticker
        import matplotlib.pyplot as plt
        for path in ['/System/Library/Fonts/Supplemental/Arial Unicode.ttf','/usr/share/fonts/truetype/wqy/wqy-microhei.ttc']:
            if Path(path).exists():
                font_manager.fontManager.addfont(path);plt.rcParams['font.family']=font_manager.FontProperties(fname=path).get_name();break
        plt.rcParams.update({'font.size':11,'axes.unicode_minus':False})
    windows=data['windows'];total=data['total'];step=data['request_step']
    coverage = '。本次仅有一个统计窗口，图中展示实测点。' if len(windows)==1 else ''
    if total['theory_hit_rate'] is None:
        coverage += ' 本次未提供无限容量全保留基准，理论曲线暂无数据。'
    bounds = {r['end_request']: (r['first_ns'], r['last_ns']) for r in windows}
    def timed_rows(rows, cumulative):
        result = []
        for row in rows:
            start, stop = bounds[row['end_request']]
            if cumulative:
                span = time_text(stop)
            else:
                span = time_text(start) + ' – ' + time_text(stop)
            result.append((stop, start, dict(row, request_time=span)))
        return [r for _, _, r in sorted(result, key=lambda item: item[:2])]
    count_column=('累计正式请求数','end_request',lambda n:f'{n:,}')
    graphs=[]
    def add(name,title,rows,series,percentage,columns,description,missing=None):
        if args.reuse_plots:
            if not (out / (name + ".svg")).is_file():
                raise FileNotFoundError(name + ".svg")
        else:
            points = sorted((dict(r, plot_ns=bounds[r['end_request']][0]) for r in rows),
                            key=lambda r: r['plot_ns'])
            plot(out,name,title,points,series,percentage,missing)
        cumulative = name == 'cumulative-hit'
        clock_column = ('累计截至（北京时间）' if cumulative else '请求时间窗口（北京时间）', 'request_time', str)
        graphs.append(f'<section id="{name}"><h2>{esc(title)}</h2><p class="note">{esc(description)}</p>'
                      f'<img src="{name}.svg" alt="{esc(title)}" loading="lazy">'
                      '<details><summary>原始数据</summary>'+table(timed_rows(rows,cumulative),[clock_column,count_column]+columns)+'</details></section>')
    rate_series=[(kind+'_hit_rate',LABELS[kind],COLORS[kind],*STYLES[kind]) for kind in ('theory','full','joint') if kind!='theory' or total['theory_hit_rate'] is not None]
    rate_columns=[(LABELS[k],k+'_hit_rate',percent) for k in ('theory','full','joint')]
    add('interval-hit','分段命中率',windows,rate_series,True,rate_columns,
        data.get("window_note", f"每 {number(step)} 条请求独立统计，各曲线分母为该段输入总长度。")+coverage)
    add('cumulative-hit','累计命中率',data['cumulative'],rate_series,True,rate_columns,
        '累计匹配总长度 ÷ 累计输入总长度。'+coverage)
    add('attainment','联合命中达标率',windows,[('attainment','BTC 联合 / 理论',COLORS['joint'],'-','o')],True,
        [('联合命中达标率','attainment',percent)],'分段联合命中率 ÷ 同段无限容量理论命中率。'+(' 本次未提供同区间的无限容量基准。' if total['theory_hit_rate'] is None else ''),
        '缺少同区间的无限容量基准，达标率暂无法计算' if total['theory_hit_rate'] is None else '该区间理论命中为零，达标率无定义')
    add('joint-full-ratio','联合命中长度比例',windows,[('joint_full_ratio','sum(JOINT) / sum(FULL)',COLORS['joint'],'-','o')],True,
        [('联合 / FULL','joint_full_ratio',percent)],'按长度加权：sum(JOINT) ÷ sum(FULL)，分母为 0 时显示 N/A。')
    add('match-gap','FULL 与联合命中长度差值',windows,[('gap_tokens.mean','mean(FULL − JOINT)','#b86630','-','o')],False,
        [('平均长度差值（tokens/request）','gap_tokens.mean',exact_number)],'分段内逐请求 FULL − JOINT 的均值；正值表示联合匹配发生回退。')
    for name,label in [('mean','平均值'),('p95','P95'),('p50','中位数')]:
        series=[(k+'_tokens.'+name,LABELS[k],COLORS[k],*STYLES[k]) for k in ('full','joint')]
        columns=[(LABELS[k]+' '+label,k+'_tokens.'+name,exact_number) for k in ('full','joint')]
        add('length-'+name,'FULL / JOINT 匹配长度：'+label,windows,series,False,columns,
            '各段独立统计；长度单位为 tokens。分位数采用现有评估直方图的最近秩法。')
    examples=data['fallback_examples']
    if data.get("native_examples"):
        fallback = table(examples,[("正式请求序号","request",exact_number),("日志序号","source_request",exact_number),
                                   ("FULL","full_tokens",exact_number),("联合","joint_tokens",exact_number),
                                   ("少命中 tokens","gap_tokens",exact_number)])
    elif total["fallback_requests"] is None:
        fallback="<p>此后端未提供逐请求回退样本。</p>"
    elif total['fallback_requests']:
        fallback=f'<p>发现 <strong>{number(total["fallback_requests"])}</strong> 条 FULL &gt; JOINT 请求，累计差值 {number(total["gap_tokens"]["sum"])} tokens。</p>'
        fallback+='<p>本配置可复用组只有 FULL 与 SWA，F/J 来自同次原生匹配；联合长度不足说明 SWA 可用性约束未满足。此观测不进一步区分资源缺失与忙状态。</p>'
        fallback+='<p class="note">下表最多列出 20 条，跨全程均匀选取并包含最大回退样本；完整差值统计见每图下方原始数据。</p>'
        if (out.parent/'fallback-request-examples.jsonl').is_file(): fallback+='<p><a href="../fallback-request-examples.jsonl">下载完整回退样本（原请求 ID、真实前缀 key 与匹配端点）</a></p>'
        fallback+=table(examples,[('本轮序号','request',exact_number),('源请求序号','source_request',exact_number),('时间（UTC+08）','timestamp_ns',time_text),
                    ('输入长度','input_tokens',exact_number),('理论','theory_tokens',exact_number),('FULL','full_tokens',exact_number),('JOINT','joint_tokens',exact_number),('差值','gap_tokens',exact_number)])
    else:
        fallback='<p>本轮完整请求中，<strong>FULL &gt; JOINT 的请求数为 0</strong>；未观察到 SWA 可用性约束导致联合回退。</p><p>FULL/JOINT 长度曲线重合，联合长度比例为 '+percent(total['joint_full_ratio'])+'，差值为 0。没有为制造回退调整保留策略或容量。</p>'
    if data.get('fact_scope'):
        fallback += '<p>'+esc(data['fact_scope'])+'</p>'
    zero_joint_summary = ''
    if 'zero_joint_fallback_requests' in total:
        zero_joint_summary = f"其中 {total['zero_joint_fallback_requests']:,} 条 JOINT 回退至 0。"
        rows = [dict(row, scope=f"{row['first_request']:,}–{row['end_request']:,}") for row in windows]
        rows.append(dict(total, scope='全程'))
        columns = [('本轮请求区间', 'scope', str),
                   ('FULL 命中请求', 'full_hit_requests', exact_number),
                   ('联合回退请求', 'fallback_requests', exact_number),
                   ('回退至 0 的请求', 'zero_joint_fallback_requests', exact_number),
                   ('占 FULL 命中请求', 'zero_joint_share_of_full_hits', percent),
                   ('占联合回退请求', 'zero_joint_share_of_fallbacks', percent),
                   ('占全部请求', 'zero_joint_share_of_requests', percent)]
        fallback = ('<h3>FULL 命中且 JOINT 回退至 0</h3>'
                    '<p>按请求计数。每条满足 FULL &gt; 0 且 JOINT = 0 的请求计一次。联合回退请求指 FULL &gt; JOINT。</p>'
                    + table(rows, columns) + fallback)
    params=data['parameters'];tier_rows=[{'tier':k,**v} for k,v in params['tiers'].items()]
    config_table=table(tier_rows,[('层','tier',str),('FULL 可用块数/rank','full',exact_number),('SWA 可用块数/rank','swa',exact_number)])
    identity=data['identity'];checks=''.join('<li>'+esc(k)+'：'+('通过' if v else '失败')+'</li>' for k,v in data['verification'].items())
    intro=f'''<h1>BTC 离线命中率评估</h1><p class="subtitle">{number(total['requests'])} 请求 · {esc(data.get("window_note",f"每 {number(step)} 请求分段"))}</p>
<div class="cards"><div>理论命中率<strong>{percent(total['theory_hit_rate'])}</strong></div><div>BTC FULL 命中率<strong>{percent(total['full_hit_rate'])}</strong></div><div>BTC 联合命中率<strong>{percent(total['joint_hit_rate'])}</strong></div><div>联合命中达标率<strong>{percent(total['attainment'])}</strong></div></div>
<p><a href="#fallback">已发现 {number(total['fallback_requests'])} 条联合回退请求。{zero_joint_summary}最大差值 {total['gap_tokens']['max']:,} tokens · 查看样本</a></p>
<p class="note">上方为全程累计值，下方图表按请求到达时间分段，每个点标在对应区间的起点。数据来自原日志第 {identity['source_first_request']:,}—{identity['source_last_request']:,} 条，预热条数：{identity.get("warmup","见实验身份")}。</p>'''
    appendix=f'''<section id="fallback"><h2>SWA 联合回退样本</h2>{fallback}</section>
<section><details><summary>测试环境与参数</summary><p>配置取自该次回放记录。</p>
<p>请求范围：{identity['source_first_request']:,}—{identity['source_last_request']:,}；时间：{time_text(total['first_ns'])}—{time_text(total['last_ns'])}（UTC+08）。输入总长度：{number(total['input_tokens'])} tokens。</p>
{config_table}<pre>{esc(json.dumps(params,indent=2,ensure_ascii=False))}</pre>
<p>理论基线：{esc(data.get("theory_note","同一请求区间的无限容量、SWA 全保留结果"))}。</p>
<p>BTC 版本：{esc(identity.get('source_commit',identity.get('revision','未记录')))}；二进制 SHA256：<code>{esc(identity.get('binary_sha256',identity.get('backend_sha256','未记录')))}</code>。</p>
<p>曲线中的 tokens 是由 block 匹配长度换算的复用长度，不表示执行了真实推理。</p></details>
<details><summary>验证结果与原始指标</summary><ul>{checks}</ul><p><a href="../report-metrics.json">下载完整报告数据 JSON</a></p><pre>{esc(json.dumps(data['total'],indent=2))}</pre></details></section>'''
    css='''body{font:16px/1.65 system-ui,-apple-system,sans-serif;color:#183447;background:#f7f9fb;margin:0}main{max-width:1180px;margin:36px auto;padding:0 24px}h1{font-size:38px;margin-bottom:0}h2{font-size:27px;margin:0 0 8px}section{background:white;border:1px solid #dfe7ee;border-radius:14px;padding:24px;margin:24px 0}.subtitle,.note{color:#5a6f81}.subtitle{font-size:19px}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:22px 0}.cards>div{background:white;border:1px solid #dfe7ee;padding:16px;border-radius:10px}.cards strong{display:block;font-size:25px;color:#173e54}img{width:100%;height:auto}details{border-top:1px solid #e5ebef;padding-top:12px;margin-top:15px}summary{cursor:pointer;font-weight:600}table{border-collapse:collapse;width:100%;white-space:nowrap}th,td{padding:10px;border-bottom:1px solid #e5ebef;text-align:right}th:first-child,td:first-child{text-align:left}.table-scroll{overflow:auto}pre{font-size:13px;background:#f3f6f8;padding:18px;overflow:auto;max-height:420px}code{word-break:break-all}a{color:#2769a3}@media(max-width:760px){.cards{grid-template-columns:repeat(2,1fr)}main{padding:0 12px}section{padding:15px}h1{font-size:30px}h2{font-size:23px}}'''
    document='<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>BTC 离线命中率评估 · 分段报告</title><style>'+css+'</style></head><body><main>'+intro+''.join(graphs)+appendix+'</main></body></html>'
    (out/'report.html').write_text(document)
    print(out/'report.html')
if __name__=='__main__':main()
