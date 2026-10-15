"""将冻结的缓存同步证据交给公共报告层，不重新判定功能断言。"""
from __future__ import annotations

import argparse
import copy
import datetime
import hashlib
import json
from pathlib import Path
import re

from reporting.components import details, run_meta, table
from reporting.core import write_bundle


def gc_events(path):
    pattern = re.compile(r'^\[([^]]+)\].*?\[gc\s*\]\s+GC\((\d+)\) (Pause .+?)\s+\S+->\S+\([^)]*\) ([\d.]+)ms$')
    result = []
    for line in path.read_text().splitlines():
        match = pattern.match(line)
        if match:
            stamp, identity, name, duration = match.groups()
            end = datetime.datetime.strptime(stamp, '%Y-%m-%dT%H:%M:%S.%f%z').timestamp()
            result.append(dict(epoch_s=end, duration_ms=float(duration), full='Pause Full' in name,
                               gc_id=int(identity), name=name))
    return result


def derivative(rows, key):
    return [dict(epoch_s=b['epoch_s'], value=(b[key]-a[key])/(b['epoch_s']-a['epoch_s']))
        for a, b in zip(rows, rows[1:]) if a.get(key) is not None and b.get(key) is not None
        and b[key] >= a[key] and b['epoch_s'] > a['epoch_s']]


def compact_report_spec(data, original):
    """同一冻结数据按变体选择两张主图；完整指标保留在汇总和证据中。"""
    spec = copy.deepcopy(original)
    variant = 'recovery' if data.get('intervention', data.get('parameters', {}).get('intervention', False)) else 'steady'
    zero = spec['timeOriginEpochS']
    end = zero + spec['timeAxis']['max']
    rows = [r for r in data['frames'] if zero <= r['epoch_s'] <= end]
    selected = {'measurement_started': '测量开始'}
    if variant == 'recovery':
        selected.update(partial_workers_withdrawn='撤出12个P', partial_workers_restored='恢复发现')
        count = data.get('parameters', {}).get('procedure', {}).get('expired_workers', 12)
        selected['partial_workers_withdrawn'] = f'撤出{count}个P'
    events = [dict(id=e['id'], name=selected[e['id']], t=e['epoch_s']-zero)
              for e in data['events'] if e['id'] in selected and zero <= e['epoch_s'] <= end]
    panels = []

    def add(key, title, unit, points, caption):
        panels.append(dict(id=variant+'-'+key, title=variant+' · '+title, type='line', timeX=True,
            axes={'y': dict(title=unit)}, events=events, caption=caption,
            series=[dict(name=title, metric_id=variant+'.'+key, unit=unit, axis='y', points=points,
                         provenance=dict(source=(spec.get('run_meta', {}).get('evidence') or {}).get('source', 'frozen analysis'), measurement=key))]))

    def points(items, field, scale=1):
        return [dict(x=r['epoch_s']-zero, y=r[field]*scale if r.get(field) is not None else None) for r in items]

    if variant == 'steady':
        pauses = [g for g in data['gc'] if g['full'] and zero <= g['epoch_s'] <= end]
        add('gc', 'Full GC 暂停', 'ms', points(pauses, 'duration_ms'),
            '仅画 Full GC，以暂停结束时刻定位。次数、频率和全部 GC 暂停占比见分阶段表。')
        add('allocation', 'Master 堆内存分配速率', 'GiB/s', points(derivative(rows, 'process_allocated_bytes'), 'value', 1/2**30),
            '进程累计线程分配量差分；缺失保留空值。CPU、内存、同步强度和 diff 扫描量见汇总表。')
    else:
        add('age', '未参与撤出的 Worker 最长未更新时长', 's', points(rows, 'healthy_max_status_age_ms', .001),
            '未参与撤出的 Worker 指本次未被选中执行撤出、恢复的 Worker。按最近一次成功轮询时间计算；超过10秒表示状态更新超时。')
        env = data.get('environment', {})
        total = env.get('n_prefill', 0)+env.get('n_decode', 0)-len(data.get('victims', []))
        visible = [dict(epoch_s=r['epoch_s'], value=total-r['missing_workers']
                        if total and r.get('missing_workers') is not None else None) for r in rows]
        add('workers', '未参与撤出的 Worker 数量（Master 可见）', '个', points(visible, 'value'),
            '不包含本次选中撤出的 Worker。统计 Master 中可见的调度端点，不等于实际存活的引擎进程数。')
    anchor = [dict(id='cache_only_started', name='观察起点', t=e['epoch_s']-zero)
              for e in data['events'] if e['id']=='cache_only_started']
    spec.update(panels=panels, events=anchor+events, title=variant+' · 缓存同步实验 · '+spec['run_id'])
    resources = []
    for phase in data.get('phase_summary', {}):
        group = [r for r in rows if r['phase'] == phase]
        def maximum(field, nested=None):
            values = [(r.get(nested) or {}).get(field) if nested else r.get(field) for r in group]
            valid = [v/2**30 for v in values if v is not None]
            return round(max(valid), 3) if valid else None
        resources.append([phase, maximum('heap_bytes'), maximum('memory_bytes', 'master'), maximum('rss_bytes', 'master')])
    sections = [s for s in spec.get('sections', []) if s.get('title') != 'Master 内存摘要']
    spec['sections'] = sections + [table('Master 内存摘要', ['阶段','堆峰值 GiB','容器内存峰值 GiB','RSS峰值 GiB'], resources)]
    return spec


def report(instance, output, identity):
    source = instance/'cache-stability/evidence.json'
    raw = json.loads(source.read_text())
    frames = [{k: v for k, v in r.items() if k != 'workers'} for r in raw['frames']]
    zero = next((e['epoch_s'] for e in raw['events'] if e.get('name', e['id']) == 'cache_only_started'), frames[0]['epoch_s'])
    performance_end = next((e['epoch_s'] for e in raw['events'] if e.get('name', e['id']) == 'performance_observation_finished'),
                           frames[-1]['epoch_s'])
    rows = [r for r in frames if zero <= r['epoch_s'] <= performance_end]
    labels = dict(cache_only_started="纯缓存观察开始", traffic_warmup_started="业务预热开始",
        measurement_started="测量开始", measurement_finished="测量结束",
        partial_workers_withdrawn="撤出部分Worker", partial_workers_restored="恢复Worker发现",
        intervention_started="对照事件窗口开始", post_started="恢复后观察开始", post_finished="恢复后观察结束",
        sending_stopped="停止发送", drain_finished="请求排空", performance_observation_finished="性能观察结束")
    events = [dict(id=e.get('name', e['id']), name=labels.get(e['id'],e['id']), t=e['epoch_s']-zero)
              for e in raw['events'] if zero <= e['epoch_s'] <= performance_end]
    gc_path = next(instance.glob('environment/*/master_gc.log'))
    gc = [g for g in gc_events(gc_path) if zero <= g['epoch_s'] <= performance_end]
    checks = raw.get('checks', [])
    validity = not raw['errors'] and not raw.get('runtime_incomplete') and any(e.get('name', e['id'])=='observation_finished' for e in raw['events'])
    result = json.loads((instance/'result.json').read_text()) if (instance/'result.json').exists() else {}
    acceptance = {c['id']: c['status'] for c in checks if c['id'] in {
        'actual_sync_intensity','clock_input_intensity','response_shape','sampling_complete','client_complete'}}
    validity = validity and acceptance and all(v=='PASS' for v in acceptance.values())
    launched = {}
    for ancestor in instance.parents:
        receipt = ancestor/'master-launched-artifact.json'
        if receipt.exists():
            launched = {kind: json.loads((ancestor/(kind+'-launched-artifact.json')).read_text())
                        for kind in ('master','mock')}
            break
    data = {k: v for k, v in raw.items() if k != 'frames'}
    data.update(frames=frames, gc=gc, raw_root=str(instance),
        outcome=dict(gate='FAIL' if any(c['status']=='FAIL' for c in checks) else 'PASS' if checks else 'INVALID',
                     validity='VALID' if validity else 'INVALID'),
        source_evidence_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), execution_result=result,
        workload_acceptance=acceptance, launched_artifacts=launched)
    data.update(verdict=data['outcome']['gate'], runtime_validity=data['outcome']['validity'])
    baseline_ready = any(e['id']=='cache_baseline_ready' for e in raw['events'])
    summaries = {}
    for phase, actual in raw['intensity'].items():
        group = [r for r in rows if r['phase']==phase]
        if len(group)<2:
            continue
        first, last = group[0]['epoch_s'], group[-1]['epoch_s']
        duration = last-first
        pauses = [g for g in gc if first <= g['epoch_s'] <= last]
        overlap = lambda g: max(0, min(last,g['epoch_s'])-max(first,g['epoch_s']-g['duration_ms']/1000))
        def rate(field, scale=1):
            values = [r for r in group if r.get(field) is not None]
            if len(values)<2 or values[-1][field]<values[0][field]:
                return None
            return (values[-1][field]-values[0][field])/(values[-1]['epoch_s']-values[0]['epoch_s'])*scale
        cpu = [dict(epoch_s=r['epoch_s'],**r['master']) for r in group]
        cpu_rates = derivative(cpu,'cpu_usec')
        cpu_values = [r for r in cpu if r.get('cpu_usec') is not None]
        summaries[phase] = dict(samples=len(group), sample_window_s=duration, actual_rpc=actual,
            full_gc_count=sum(g['full'] for g in pauses), full_gc_per_min=sum(g['full'] for g in pauses)*60/duration,
            gc_pause_fraction=sum(overlap(g) for g in gc)/duration,
            allocation_gib_s=rate('process_allocated_bytes',1/2**30), diff_million_keys_s=rate('diff_scanned_keys',1e-6),
            cache_success_s=rate('cache_sync_commits'), worker_status_success_s=rate('worker_status_commits'),
            cpu_cores=((cpu_values[-1]['cpu_usec']-cpu_values[0]['cpu_usec'])
                       /(cpu_values[-1]['epoch_s']-cpu_values[0]['epoch_s'])/1e6) if cpu_rates else None,
            max_healthy_age_ms=max((r['healthy_max_status_age_ms'] for r in group
                                   if r.get('healthy_max_status_age_ms') is not None),default=None),
            affected_workers=max((r.get('affected_workers',0) for r in group),default=0))
    data['phase_summary'] = summaries
    spec = dict(report_spec_schema_version=1, run_id=identity, title='缓存同步稳态与恢复 · '+identity,
        subtitle=('共用背景；固定恢复后窗口完整观察，排空另行记录。冻结一致性扫描在性能窗口外。' if baseline_ready
                  else '未建立大缓存基线：展示准备阶段证据，不能声称执行了稳态或恢复窗口。'),
        timeOriginEpochS=zero, timeOriginLabel='缓存基线稳定后，纯缓存观察开始' if baseline_ready else '首次健康诊断观察（大缓存基线未建立）',
        timeAxis=dict(min=0,max=performance_end-zero+1), events=events, panels=[],
        run_meta=run_meta(dict(id=identity,kind='run',case='cache_sync_recovery'),
            implementation=dict(master=launched.get('master'), mock=launched.get('mock')),
            configuration=result.get('implementation'), workload=raw['parameters'],
            environment=raw['environment'], evidence=dict(source=str(source))),
        kpis=[dict(label='功能判定',value=data['outcome']['gate']),dict(label='实验有效性（含同步强度）',value=data['outcome']['validity']),
              dict(label='Full GC次数',value=sum(g['full'] for g in gc)),
              dict(label='未参与撤出的 Worker 受影响数',value=len(raw.get('affected_workers',[])))],
        sections=[table('分阶段实测',['阶段','Mock缓存响应/s','Master P同步成功/s','状态轮询成功/s','条目/s','MiB/s','分配GiB/s','diff百万key/s','FullGC次/分','GC暂停占比','CPU核','未参与撤出的 Worker 最长未更新ms'],
            [[phase, round(v['actual_rpc']['responses'],3), v['cache_success_s'], v['worker_status_success_s'], round(v['actual_rpc']['entries']),
              round(v['actual_rpc']['key_payload_bytes']/2**20,3),v['allocation_gib_s'],v['diff_million_keys_s'],
              v['full_gc_per_min'],v['gc_pause_fraction'],v['cpu_cores'],v['max_healthy_age_ms']]
             for phase,v in summaries.items()]),table('功能断言' ,['断言','状态','实际值'],[[c['id'],c['status'],str(c['actual'])] for c in checks]),
                  details('实际同步强度',raw['intensity']),details('一致性',raw.get('consistency')),
                  details('健康Worker异常证据',raw.get('healthy_violations',[])),
                  details('证据边界',dict(scope=raw['expiry_scope'],errors=raw['errors']))])
    spec = compact_report_spec(data, spec)
    return write_bundle(output,'run',identity,data,spec,producer='cache_sync_recovery',role='diagnostic')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('instance',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--id',required=True)
    args = parser.parse_args()
    report(args.instance,args.output,args.id)


if __name__ == '__main__':
    main()
