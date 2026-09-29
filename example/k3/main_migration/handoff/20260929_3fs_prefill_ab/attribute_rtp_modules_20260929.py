#!/usr/bin/env python3
"""Attribute target Prefill GPU launches to the innermost RTP record range."""

import glob
import json
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path('/data0/luohaocheng.lhc/artifacts/k3-fp8-opt-20260927')
RUNS = {
    'integrated': ROOT / 'timeline-64k-integrated-114112-r1-20260929/traces/k3_64k_integrated_114112_r1_wr{rank}_1.json',
    'feat': ROOT / 'timeline-64k-feat-r12-audit-20260929/traces/k3_64k_feat_nocp_a9bf_r12_114115_wr{rank}_1.json',
}


def attribute_one(path):
    events = json.loads(path.read_text())['traceEvents']
    targets = sorted((e for e in events if e.get('cat') == 'cpu_op' and e.get('name') == 'executor.mtp.prefill_step(target_model_forward)'), key=lambda e: e['ts'])
    assert len(targets) == 8, (path, len(targets))
    launches = defaultdict(list)
    for e in events:
        if e.get('cat') in ('cuda_runtime', 'cuda_driver'):
            corr = e.get('args', {}).get('correlation')
            if corr is not None:
                launches[corr].append(e)
    annotations = defaultdict(list)
    for e in events:
        if e.get('cat') == 'user_annotation' and e.get('name', '').startswith('RTP::'):
            annotations[(e['pid'], e['tid'])].append(e)
    rows = [defaultdict(lambda: {'count': 0, 'kernel_sum_ms': 0.0, 'kernels': defaultdict(lambda: {'count': 0, 'sum_ms': 0.0})}) for _ in range(8)]
    unmatched = 0
    for kernel in (e for e in events if e.get('cat') == 'kernel'):
        corr = kernel.get('args', {}).get('correlation')
        candidates = [launch for launch in launches.get(corr, [])
                      if any(target['ts'] <= launch['ts'] < target['ts'] + target['dur'] for target in targets)]
        if len(candidates) != 1:
            if candidates:
                unmatched += 1
            continue
        launch = candidates[0]
        idx = next(i for i, t in enumerate(targets) if t['ts'] <= launch['ts'] < t['ts'] + t['dur'])
        scopes = [a for a in annotations[(launch['pid'], launch['tid'])]
                  if a['ts'] <= launch['ts'] < a['ts'] + a['dur']]
        label = min(scopes, key=lambda a: a['dur'])['name'] if scopes else '[unattributed]'
        row = rows[idx][label]
        row['count'] += 1
        row['kernel_sum_ms'] += kernel['dur'] / 1000
        family = kernel['name'].split('<', 1)[0][:100]
        row['kernels'][family]['count'] += 1
        row['kernels'][family]['sum_ms'] += kernel['dur'] / 1000
    assert unmatched == 0, (path, unmatched)
    return [{label: {**data, 'kernels': dict(data['kernels'])} for label, data in row.items()} for row in rows]


def main():
    result = {'method': 'GPU correlation -> CPU launch -> same-thread innermost RTP user_annotation',
              'timing_basis': 'cumulative GPU kernel time, overlapping streams may double count; not critical-path duration',
              'sample_indices': [2, 3, 4, 5, 6], 'runs': {}}
    for name, template in RUNS.items():
        ranks = [attribute_one(Path(str(template).format(rank=i))) for i in range(8)]
        samples = defaultdict(list)
        for rank in ranks:
            for index in result['sample_indices']:
                for label, data in rank[index].items():
                    samples[label].append(data['kernel_sum_ms'])
        table = {label: {'median_rank_request_kernel_sum_ms': statistics.median(values),
                         'rank_request_samples': len(values)} for label, values in samples.items()}
        covered = sum(v['median_rank_request_kernel_sum_ms'] for k, v in table.items() if k != '[unattributed]')
        alltime = covered + table.get('[unattributed]', {}).get('median_rank_request_kernel_sum_ms', 0)
        result['runs'][name] = {'ranks': ranks, 'module_table': table, 'attributed_fraction_of_median_kernel_sums': covered / alltime}
    out = ROOT / 'timeline-64k-module-attribution-114-20260929.json'
    out.write_text(json.dumps(result, indent=2) + '\n')
    for name, data in result['runs'].items():
        print(name, 'attributed', round(data['attributed_fraction_of_median_kernel_sums'], 4))
        for label, value in sorted(data['module_table'].items(), key=lambda x: -x[1]['median_rank_request_kernel_sum_ms'])[:25]:
            print(f"  {value['median_rank_request_kernel_sum_ms']:.3f} ms {label}")
    print(out)


if __name__ == '__main__':
    main()
