"""将冻结的缓存同步证据交给公共报告层，不重新判定功能断言。"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from reporting.components import details, run_meta, table
from reporting.core import write_bundle


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

    add('age', '未参与撤出的 Worker 最长未更新时长', 's', points(rows, 'healthy_max_status_age_ms', .001),
        '成功轮询与已应用观测年龄的较大值；仅覆盖完整采样时刻，不代表连续时间窗无异常。')
    env = data.get('environment', {})
    total = env.get('n_prefill', 0)+env.get('n_decode', 0)-len(data.get('victims', []))
    visible = [dict(epoch_s=r['epoch_s'], value=total-r['missing_workers']
                    if total and r.get('missing_workers') is not None else None) for r in rows]
    add('workers', '未参与撤出的 Worker 数量（Master 可见）', '个', points(visible, 'value'),
        '不包含选中撤出的 Worker；Master 可见端点不等于实际存活进程或可调度容量。')
    anchor = [dict(id='cache_only_started', name='观察起点', t=e['epoch_s']-zero)
              for e in data['events'] if e['id']=='cache_only_started']
    spec.update(panels=panels, events=anchor+events, title=variant+' · 缓存同步实验 · '+spec['run_id'])
    ages = []
    for phase in data.get('phase_summary', {}):
        group = [r for r in rows if r['phase'] == phase]
        def maximum_age(field):
            values = [r[field] for r in group if r.get(field) is not None]
            return max(values) if values else None
        ages.append([phase, maximum_age('healthy_max_rpc_age_ms'), maximum_age('healthy_max_applied_age_ms')])
    spec['sections'] = spec.get('sections', []) + [table('轮询与状态应用',
        ['阶段', '最长成功轮询年龄 ms', '最长已应用观测年龄 ms'], ages), details('观测边界', {
            '年龄': '应用年龄从不可变回包观测建立时开始，包含应用等待；同版本有效心跳可以刷新。',
            '策略': '导出观测不改变状态失鲜准入规则，不证明可接收新请求。',
            '缓存': '版本、索引初始化和计数收敛；不证明 key 集合 membership。'})]
    return spec


def write_report(directory, raw, result, *, run=None, presentation=None):
    """Render a committed gate; the workload runner owns invocation and delivery."""
    directory = Path(directory)
    identity = run["id"]+"-cache-stability" if run else directory.name+"-cache-stability"
    frames = [{k: v for k, v in r.items() if k != 'workers'} for r in raw['frames']]
    if presentation is not None:
        from monitoring.metric_store import MetricStore
        from cases.cache_sync_recovery.telemetry import archived_metrics
        frames = archived_metrics(MetricStore.read(directory), frames, epoch=raw["provenance"]["env_epoch"],
                                 maximum_gap_s=raw["parameters"]["observation"]["max_gap_s"])
    zero = next((e['epoch_s'] for e in raw['events'] if e.get('name', e['id']) == 'cache_only_started'), frames[0]['epoch_s'])
    performance_end = next((e['epoch_s'] for e in raw['events'] if e.get('name', e['id']) == 'performance_observation_finished'), frames[-1]['epoch_s'])
    rows = [r for r in frames if zero <= r['epoch_s'] <= performance_end]
    gc = [g for g in raw.get('gc', []) if zero <= g['epoch_s'] <= performance_end]
    checks = result['functional_checks'] if 'functional_checks' in result else result['checks']
    launched = raw.get('launched_artifacts', {})
    events = [dict(id=e.get('name', e['id']), name=e.get('name', e['id']), t=e['epoch_s']-zero)
              for e in raw['events'] if zero <= e['epoch_s'] <= performance_end]
    data = {k: v for k, v in raw.items() if k != 'frames'}
    data.update(frames=frames, gc=gc, raw_root=str(directory),
        outcome=dict(gate=result['verdict'], validity=result['validity']),
        source_evidence_sha256=hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest(), execution_result=result,
        presentation_renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        workload_acceptance={c['id']: c['status'] for c in result['validity_checks']}, launched_artifacts=launched)
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
        summaries[phase] = dict(samples=len(group), sample_window_s=duration, actual_rpc=actual,
            full_gc_count=sum(g['full'] for g in pauses) if raw.get('gc_available', True) else None,
            full_gc_per_min=sum(g['full'] for g in pauses)*60/duration if raw.get('gc_available', True) else None,
            gc_pause_fraction=sum(overlap(g) for g in gc)/duration if raw.get('gc_available', True) else None,
            diff_million_keys_s=rate('diff_scanned_keys',1e-6),
            cache_success_s=rate('cache_sync_commits'), worker_status_success_s=rate('worker_status_commits'),
            max_healthy_age_ms=max((r['healthy_max_status_age_ms'] for r in group
                                   if r.get('healthy_max_status_age_ms') is not None),default=None),
            affected_workers=max((r.get('affected_workers',0) for r in group),default=0))
    data['phase_summary'] = summaries
    spec = dict(report_spec_schema_version=1, run_id=identity, title='缓存同步稳态与恢复 · '+identity,
        subtitle=('共用背景；固定恢复后窗口完整观察，排空另行记录。缓存版本与索引计数在性能窗口外收敛，不扫描key集合。' if baseline_ready
                  else '未建立大缓存基线：展示准备阶段证据，不能声称执行了稳态或恢复窗口。'),
        timeOriginEpochS=zero, timeOriginLabel='缓存基线稳定后，纯缓存观察开始' if baseline_ready else '首次健康诊断观察（大缓存基线未建立）',
        timeAxis=dict(min=0,max=performance_end-zero+1), events=events, panels=[],
        run_meta=run_meta(dict(id=identity,kind='run',case='cache_sync_recovery'),
            implementation=dict(master=launched.get('master'), mock=launched.get('mock')),
            configuration=(run or {}).get('implementation'), workload=raw['parameters'],
            environment=raw['environment'], evidence=dict(source=str(directory/'cache_stability-gate-evidence.json'))),
        kpis=[dict(label='功能判定',value=data['outcome']['gate']),dict(label='实验有效性（含同步强度）',value=data['outcome']['validity']),
              dict(label='Full GC次数',value=sum(g['full'] for g in gc) if raw.get('gc_available', True) else None),
              dict(label='未参与撤出的 Worker 受影响数',value=len(set(raw.get('affected_workers',[]))-set(raw.get('victims',[]))))],
        sections=[table('分阶段实测',['阶段','Mock缓存响应/s','Master P同步成功/s','状态应用成功/s','条目/s','MiB/s','diff百万key/s','FullGC次/分','GC暂停占比','未参与撤出的 Worker 最长未更新ms'],
            [[phase, round(v['actual_rpc']['responses'],3), v['cache_success_s'], v['worker_status_success_s'], round(v['actual_rpc']['entries']),
              round(v['actual_rpc']['key_payload_bytes']/2**20,3),v['diff_million_keys_s'],
              v['full_gc_per_min'],v['gc_pause_fraction'],v['max_healthy_age_ms']]
             for phase,v in summaries.items()]),table('功能断言' ,['断言','状态','实际值'],[[c['id'],c['status'],str(c['actual'])] for c in checks]),
                  details('实际同步强度',raw['intensity']),details('缓存进度收敛',raw.get('cache_progress')),
                  details('健康Worker异常证据',raw.get('healthy_violations',[])),
                  details('证据边界',dict(scope=raw['expiry_scope'],errors=raw['errors']))])
    if run is not None and run.get('workload', {}).get('runtime_validity') == 'INVALID':
        data['outcome']['validity'] = data['runtime_validity'] = 'INVALID'
    spec = compact_report_spec(data, spec)
    if presentation is not None:
        from monitoring.metric_store import MetricStore
        from reporting.curves import materialize, project_panels
        store = MetricStore.read(directory)
        curves = [materialize(key, style, row["points"], origin=zero,
                              labels=row["labels"], provenance=row["provenance"])
                  for key, style in presentation.charts.curves.items()
                  for row in store.document["metrics"].get(style.metric_id, [])]
        variant = "recovery" if raw["intervention"] else "steady"
        panels = [p for p in project_panels(curves, presentation) if p["id"].startswith(variant+"-")]
        for panel in panels:
            panel["events"] = spec["panels"][0]["events"]
        spec["panels"] = panels
    spec['sections'].insert(0, table('实验有效性', ['检查','状态','实际值'],
        [[c['id'],c['status'],str(c['actual'])] for c in result['validity_checks']]))
    # Keep execution/telemetry provenance from the common runner; preserve the case's own origin.
    if run is not None:
        from reporting.run_context import canonical_spec
        payload = dict(run)
        spec = canonical_spec(spec, payload)
        data['run'] = run
    return write_bundle(directory, 'run', identity, data, spec,
                        producer='cache_stability', role='gate')


def validate_view(path, definition, fail):
    from reporting.view_schema import validate_section_contract
    validate_section_contract(path, definition, {"sources": None}, fail)


def report(instance, output, identity):
    """Explicit offline rendering of an existing committed gate; never re-adjudicate."""
    from artifacts.gate import load_gate
    evidence, result = load_gate(instance, "cache_stability")
    return write_report(output, evidence, result)
