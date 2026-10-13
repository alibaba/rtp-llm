"""Compare frozen bundles and experiment ZIPs without rerunning gates."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import zipfile

from reporting import bundle_path, compare_controls, details, links, run_meta, write_bundle
from reporting.core import table
from reporting.catalog import series_color
from reporting.comparison_inputs import load_snapshot
from reporting.comparison_model import ChartSeries
from reporting.pairing import event_anchor, paired_overlay, shifted_panel


def _leaves(value, prefix=''):
    if isinstance(value, dict) and value:
        for key, item in sorted(value.items()):
            yield from _leaves(item, prefix + '/' + key.replace('~', '~0').replace('/', '~1'))
    else:
        yield prefix, value


def _matrix(runs, labels):
    values = []
    for run in runs:
        meta = run.report.run_meta
        values.append(dict(_leaves(dict(identity=meta.get('identity'),
            implementation=meta.get('implementation'), configuration=meta.get('configuration'),
            workload=meta.get('workload'), environment=meta.get('environment'),
            clock=meta.get('clock'), criteria=run.report.controls.criteria))))
    rows = []
    for key in sorted(set().union(*(v.keys() for v in values))):
        cells = [v.get(key) for v in values]
        # Missing information is never proof of equal conditions.
        state = '未知' if any(cell is None for cell in cells) else '相同' if all(cell == cells[0] for cell in cells) else '不同'
        rows.append([key, state, *cells])
    return table('实验条件与差异', ['字段', '状态', *labels], rows)


def _series_key(panel, series, origin):
    signature = ChartSeries.read(series, panel.get('unit')).signature()
    if signature is None or panel.get('type', 'line') != 'line' or not panel.get('timeX'):
        return None
    # Keep worker labels, aggregation and measurement semantics; visual titles/IDs may differ.
    return json.dumps([signature, panel['axes'][series.get('axis', 'y')], origin],
                      sort_keys=True, allow_nan=False)


def _pair(prepared, labels, origins, *, include_original_panels):
    groups, unique, ambiguous = {}, {}, set()
    for run_index, panels in enumerate(prepared):
        for panel_index, panel in enumerate(panels):
            for series_index, series in enumerate(panel['series']):
                key = _series_key(panel, series, origins[run_index])
                identity = (run_index, key)
                if key is None or identity in ambiguous:
                    continue
                if identity in unique:
                    if unique[identity]['points'] != series['points'] or unique[identity].get('statistics_points') != series.get('statistics_points'):
                        ambiguous.add(identity)
                        groups[key] = [entry for entry in groups[key] if entry[0] != run_index]
                    continue
                unique[identity] = series
                groups.setdefault(key, []).append((run_index, panel_index, series_index))
    overlays, pairing, paired = [], [], set()
    for key, entries in groups.items():
        if len(entries) < 2:
            continue
        first_run, first_panel, first_series = entries[0]
        original = prepared[first_run][first_panel]
        metric = original['series'][first_series].get('metric_id')
        panel = copy.deepcopy(original)
        panel.update(id='overlay:' + hashlib.sha256(key.encode()).hexdigest()[:16],
                     title='对比 · ' + original['series'][first_series]['name'],
                     presets={}, events=[])
        axis = original['series'][first_series].get('axis', 'y')
        panel['axes'] = {axis: copy.deepcopy(original['axes'][axis])}
        small_panels, selected_labels = [], []
        for i, p, s in entries:
            source = prepared[i][p]
            curve = source['series'][s]
            small_panels.append(dict(series=[curve], presets={}))
            selected_labels.append(labels[i])
            panel['events'].extend(dict(e, name=labels[i] + ' · ' + e['name'])
                                   for e in source.get('events', []))
            paired.add((i, key))
        panel['series'], _ = paired_overlay(small_panels, selected_labels,
            color_for=lambda name, label, original_color: series_color(None, labels.index(label)))
        for index, curve in enumerate(panel['series']):
            run_index = labels.index(selected_labels[index])
            curve['dash'] = [] if run_index == 0 else [4 + run_index * 2, 3]
            curve['hidden'] = False
            curve['group'] = selected_labels[index]
            panel['presets'][selected_labels[index]] = [curve['name']]
        missing = [label for label in labels if label not in selected_labels]
        panel['caption'] = '同测量口径；保留各 run 的采样时间与缺口。' + (
            ' 未配对：' + '、'.join(missing) if missing else '')
        overlays.append(panel)
        pairing.append(dict(metric=metric, panel=original['id'], status='PAIRED',
                            runs=selected_labels, missing_runs=missing))
    separate = []
    for i, panels in enumerate(prepared):
        for panel in panels:
            kept = [s for s in panel['series']
                    if include_original_panels or (i, _series_key(panel, s, origins[i])) not in paired]
            if kept:
                panel = copy.deepcopy(panel)
                panel['series'] = kept
                names = {s['name'] for s in kept}
                panel['presets'] = {name: [n for n in choices if n in names]
                                    for name, choices in panel.get('presets', {}).items()}
                panel['id'] = f'run-{i + 1}:' + panel['id']
                panel['title'] = labels[i] + ' · ' + panel.get('title', '')
                separate.append(panel)
                pairing.append(dict(panel=panel['id'], status='SEPARATE', runs=[labels[i]],
                    reasons=['ORIGINAL_REQUESTED' if include_original_panels else
                             'NO_COMPATIBLE_PEER_OR_UNKNOWN_MEASUREMENT']))
    return overlays + separate, pairing


def compare(paths, output, *, alignment_event=None, labels=None, baseline=None, views=None,
            include_original_panels=False):
    if len(paths) < 2:
        raise ValueError('at least two runs are required')
    if alignment_event is not None and (not isinstance(alignment_event, str) or not alignment_event.strip()):
        raise ValueError('alignment_event must be a nonempty event name')
    labels = labels or [chr(65 + i) if i < 26 else f'run-{i + 1}' for i in range(len(paths))]
    if (len(labels) != len(paths) or any(not isinstance(label, str) or not label.strip() for label in labels)
            or len(set(labels)) != len(labels)):
        raise ValueError('provide one unique nonempty label per input')
    views = views or [None] * len(paths)
    if len(views) != len(paths):
        raise ValueError('provide one view per input (auto for automatic selection)')
    baseline = labels[0] if baseline is None else baseline
    if baseline not in labels:
        raise ValueError('baseline must name one input label')
    loaded = [load_snapshot(path, view=view) for path, view in zip(paths, views)]
    destination = bundle_path(output, 'comparison', 'runs').resolve()
    for run in loaded:
        if run.source == destination or destination in run.source.parents or run.source in destination.parents:
            raise ValueError('comparison output and input must not contain each other')
    fingerprints = [(run.input_kind, run.digest, run.view) for run in loaded]
    if len(set(fingerprints)) != len(fingerprints):
        raise ValueError('duplicate frozen input; choose distinct runs')
    anchors = [event_anchor(run.report.events, alignment_event) for run in loaded]
    aligned = alignment_event is not None and all(t is not None for t in anchors)
    time_alignment = dict(event=alignment_event,
        status='ALIGNED' if aligned else 'UNAVAILABLE' if alignment_event else 'NOT_REQUESTED',
        observed_times=dict(zip(labels, anchors)),
        missing_or_ambiguous_runs=[label for label, t in zip(labels, anchors) if alignment_event and t is None])
    caption = (f'按事件 {alignment_event} 对齐到 0 秒。' if aligned else
        f'事件 {alignment_event} 缺失、重复或时间无效，保留各 run 原时间轴。' if alignment_event else
        '保留各 run 原时间轴；不自动选择统计窗口。')
    controls = [dict(run.report.controls.to_dict(), implementation=run.report.run_meta.get('implementation'))
                for run in loaded]
    reference = controls[labels.index(baseline)]
    comparisons = {label: compare_controls(reference, control,
        required=('/configuration', '/workload', '/environment', '/criteria', '/implementation'))
        for label, control in zip(labels, controls) if label != baseline}
    sources, attachments, sections, prepared, bounds, kpis, overview = {}, {}, [], [], [], [], []
    for i, (label, run, anchor) in enumerate(zip(labels, loaded, anchors)):
        spec = run.report
        shift = anchor if aligned else None
        sources[label] = dict(path=str(run.source), manifest_sha256=run.digest,
                              input_kind=run.input_kind, view=run.view, verification='CONSUMED_FILES_VERIFIED')
        filename = f'original-run-{i + 1}.html'
        attachments[filename] = run.original_html()
        sections.append(links(label + ' · 原始报告', [dict(label='打开冻结报告', href=filename)]))
        sections.append(details(label + ' · 执行与判定', dict(
            analysis={key: value for key, value in run.analysis.items()
                      if key in {'verdict', 'validity', 'runtime_validity', 'errors', 'warnings'}},
            execution={key: value for key, value in run.execution.items()
                       if key in {'status', 'outcome', 'duration_ms', 'cleanup'}})))
        # Full frozen notes remain in the original report; avoid duplicating large evidence in HTML.
        for section in spec.sections:
            if section['type'] == 'links':
                continue
            if len(json.dumps(section, ensure_ascii=False)) <= 65536:
                sections.append(dict(copy.deepcopy(section), title=label + ' · ' + section.get('title', '')))
        kpis.append(dict(label=label + ' · 原始判定',
                         value=run.analysis.get('verdict') or run.execution.get('status') or '未知'))
        overview.append([label, spec.run_meta.get('identity'), run.input_kind, run.view,
                         run.analysis.get('verdict'), run.analysis.get('runtime_validity') or run.execution.get('workload', {}).get('runtime_validity'),
                         run.execution.get('status'), run.execution.get('duration_ms'), run.execution.get('cleanup'), list(spec.kpis)])
        panels = [shifted_panel(panel.to_dict(), shift) for panel in spec.panels]
        for panel in panels:
            panel['caption'] = panel.get('caption', '') + ' ' + caption
            if panel.get('timeX'):
                bounds.extend(p['x'] for s in panel.get('series', []) for p in s['points'])
        for value in (spec.time_min, spec.time_max):
            if type(value) in (int, float) and math.isfinite(value):
                bounds.append(value - (shift or 0))
        prepared.append(panels)
    origins = ['EVENT_ALIGNED'] * len(loaded) if aligned else [
        run.report.time_origin_label or f'unknown-origin-{i}' for i, run in enumerate(loaded)]
    panels, pairing = _pair(prepared, labels, origins, include_original_panels=include_original_panels)
    result = dict(purpose='OBSERVATION_ONLY', sources=sources,
        runs={label: run.analysis for label, run in zip(labels, loaded)},
        executions={label: run.execution for label, run in zip(labels, loaded)},
        controls=dict(reference=baseline, values=dict(zip(labels, controls)), comparisons=comparisons),
        time_alignment=time_alignment, pairing=pairing)
    minimum, maximum = min(bounds, default=0), max(bounds, default=1)
    spec = dict(title='运行报告对照', subtitle='各 run 保留独立判定；差异不自动解释为因果。' + caption,
        timeOriginLabel=caption, timeAxis=dict(min=minimum, max=max(maximum, minimum + 1)),
        events=[dict(id=alignment_event, name=alignment_event, t=0)] if aligned else [],
        panels=panels, kpis=kpis,
        summary_sections=[table('运行清单', ['名称', '实验身份', '输入', '视图', '原始 verdict',
            '原始 validity', '执行状态', '墙钟耗时 ms', '清理', '冻结摘要'], overview), _matrix(loaded, labels)],
        sections=[
            details('控制变量（以 ' + baseline + ' 为参照）', result['controls']),
            details('时间轴对齐', time_alignment), details('指标配对与缺失', pairing), *sections])
    write_bundle(output, 'comparison', 'runs', result, spec,
        meta=run_meta(dict(id='runs', kind='comparison'),
            runs={label: run.report.run_meta for label, run in zip(labels, loaded)}, evidence=sources),
        producer='report-comparison', attachments=attachments)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path, help='frozen run bundle directories or experiment ZIPs')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--label', action='append', help='repeat once per input; unique display name')
    parser.add_argument('--baseline', help='reference label (default: first input)')
    parser.add_argument('--view', action='append', help='repeat once per input; ZIP view name or auto')
    parser.add_argument('--alignment-event', help='explicit event ID used as common time origin')
    parser.add_argument('--include-original-panels', action='store_true', help='also show all original panels')
    args = parser.parse_args(argv)
    try:
        compare(args.runs, args.output, alignment_event=args.alignment_event, labels=args.label,
            baseline=args.baseline, views=[None if v == 'auto' else v for v in args.view] if args.view else None,
            include_original_panels=args.include_original_panels)
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 2
    print(bundle_path(args.output, 'comparison', 'runs') / 'report.html')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
