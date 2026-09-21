"""Compare historical Master controls without changing either gate decision."""
import argparse
import copy
import json
import math
from pathlib import Path

from stress.canvas_report_render_html import render
from .cache_gate import analyze, write_report
from online_eval.playback import comparison_notice


def compare(old_path, new_path, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    evidence = [json.loads(Path(p).read_text()) for p in (old_path, new_path)]
    for path, e in zip((old_path, new_path), evidence):
        provenance = e['provenance']
        if 'client_environment' not in provenance:
            flow = Path(path).parent / 'flows' / 'scale_in' / 'flow-input.json'
            settings = json.loads(flow.read_text())['environment']
            provenance['client_environment'] = {
                k: v for k, v in settings.items()
                if k not in {'TRACE_FILE', 'OUTPUT_DIR', 'FLOW_CONTROL_DIR', 'FLOW_RUN_ID'}
            }
    notice=comparison_notice(evidence[0]['provenance'].get('trace',{}), evidence[1]['provenance'].get('trace',{}))
    checks = {'traffic_semantics': notice is None}

    def mock_hash(e):
        hashes = [v for k, v in e['provenance']['files'].items()
                  if Path(k).name.startswith('flexlb-mock-engine-') and k.endswith('.jar')]
        if len(hashes) != 1:
            raise ValueError('one pinned mock JAR is required')
        return hashes[0]

    for name, getter in {
        'criteria': lambda e: e['criteria'],
        'client_environment': lambda e: e['provenance']['client_environment'],
        'mock_jar_sha256': mock_hash,
        'topology': lambda e: e['provenance']['topology'],
        'performance': lambda e: e['provenance']['performance'],
        'master_config': lambda e: e['provenance']['master_config'],
        'trace_sha256': lambda e: e['provenance']['trace']['sha256'],
        'mock_formula_config': lambda e: e['provenance']['mock_formula_config'],
    }.items():
        checks[name] = getter(evidence[0]) == getter(evidence[1])
    withdrawals = [next((v['t'] for v in e['events'] if v['name'] == 'withdraw_start'), None) for e in evidence]
    aligned_withdrawals = all(t is not None for t in withdrawals)
    shifts = withdrawals if aligned_withdrawals else [0, 0]
    origin = max(shifts)
    panels, results = [], []
    end = 1
    for label, e, shift in zip(('old', 'new'), evidence, shifts):
        d = output / label
        d.mkdir(exist_ok=True)
        result = analyze(e)
        spec = write_report(d, e, result)
        results.append(result)
        panel = copy.deepcopy(spec['panels'][0])
        panel['id'] = label
        sha = e['provenance']['historical_master']['source_commit']
        panel['title'] = f'{label} · {sha[:9]} · {result["verdict"]}'
        panel['caption'] += ' Model forward 为各 P 最近批次均值的非加权平均。'
        panel['caption'] += (f' 原始缩容时刻 {shift:.2f}s；本图统一对齐至 {origin:.2f}s。' if aligned_withdrawals else ' 至少一轮未进入缩容，按各自采样起点展示，不能作为缩容对照。')
        for s in panel['series']:
            for p in s['points']:
                p['x'] += origin - shift
                end = max(end, p['x'])
        panels.append(panel)
    # Common ranges keep the tenfold queue difference visible across panels.
    for axis in ('queue', 'p', 'qps'):
        maximum = max((point['y'] for panel in panels for series in panel['series']
                       if series['axis'] == axis for point in series['points']
                       if point['y'] is not None), default=1)
        for panel in panels:
            panel['axes'][axis].update(min=0, max=max(1, math.ceil(maximum * 1.05)))
    summary = dict(comparison_notice=notice, alignment=checks, aligned=all(checks.values()),
                   old=results[0], new=results[1],
                   expected_control_observed=all(checks.values()) and
                   results[0]['verdict'] == 'FAIL' and results[1]['verdict'] == 'PASS')
    (output / 'ab-result.json').write_text(json.dumps(summary, indent=2))
    time_label = (f'两轮按缩容时刻对齐，X={origin:.2f}s 为同时开始缩容' if aligned_withdrawals else '按各自采样起点展示；缩容事件不完整')
    spec = dict(run_id='cache-scale-in-ab',title='历史 master · 缩 P A/B',
                subtitle=(f"{len(evidence[0]['initial_engines'])}P → {len(evidence[0]['survivors'])}P / "
                          f"{evidence[0]['provenance']['topology']['decode']}D · {evidence[0]['criteria']['qps']} QPS · "
                          f"old {results[0]['verdict']} / new {results[1]['verdict']} · A/B 参数一致 {summary['aligned']} · 相同输入计划" + (" · " + notice if notice else "")),
                timeOriginLabel=time_label,
                events=([dict(name='withdraw_start',t=origin)] if aligned_withdrawals else []), timeAxis=dict(min=0,max=end),
                kpis=[],meta=dict(params=checks),panels=panels)
    if notice: spec["subtitle"] += " · " + notice
    (output / 'ab.html').write_text(render(spec))
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('old', type=Path)
    p.add_argument('new', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = compare(a.old,a.new,a.output)
    print(json.dumps({k:v for k,v in result.items() if k not in ('old','new')}))

    raise SystemExit(0 if result['expected_control_observed'] else (1 if result['aligned'] else 2))
