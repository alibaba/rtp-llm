"""Compare frozen bundles and experiment ZIPs without rerunning gates."""
import argparse
import copy
import json
from pathlib import Path
import sys
import zipfile

from reporting import bundle_path, compare_controls, details, links, run_meta, write_bundle
from reporting import table
from reporting.catalog import series_color
from reporting.comparison_inputs import load_snapshot
from reporting.report_model import MeasurementIdentity
from reporting.values import thaw
from reporting.comparison_colors import metric_colors, curve_style
from reporting.pairing import event_anchor, shifted_panel

def _leaves(value, prefix=''):
    if isinstance(value, dict) and value:
        for key, item in sorted(value.items()):
            yield from _leaves(item, prefix + '/' + key.replace('~', '~0').replace('/', '~1'))
    else:
        yield prefix, value


def _original_result(run):
    """Read the frozen gate and validity separately from execution status."""
    analysis, execution = run.analysis, run.execution
    outcome = analysis.get('outcome') or {}
    executed = execution.get('outcome') or {}
    workload = analysis.get('workload') or {}
    return dict(
        verdict=outcome.get('gate') or executed.get('gate') or analysis.get('verdict'),
        validity=outcome.get('validity') or executed.get('validity') or
                 analysis.get('runtime_validity') or workload.get('runtime_validity') or
                 execution.get('workload', {}).get('runtime_validity'),
        status=execution.get('status') or analysis.get('status'))


def _matrix(runs, labels):
    values = []
    for run in runs:
        meta = run.report.run_meta
        values.append(dict(_leaves(dict(identity=meta.get('identity'),
            implementation=meta.get('implementation'), configuration=meta.get('configuration'),
            workload=meta.get('workload'), environment=meta.get('environment'),
            clock=meta.get('clock'), criteria=thaw(run.report.controls.criteria)))))
    rows = []
    for key in sorted(set().union(*(v.keys() for v in values))):
        cells = [v.get(key) for v in values]
        # Missing information is never proof of equal conditions.
        state = '未知' if any(cell is None for cell in cells) else '相同' if all(cell == cells[0] for cell in cells) else '不同'
        rows.append([key, state, *cells])
    return table('实验条件与差异', ['字段', '状态', *labels], rows)


def _series_key(panel, series, origin):
    signature = MeasurementIdentity.read(series, panel.get('unit')).signature()
    if signature is None or panel.get('type', 'line') != 'line' or not panel.get('timeX'):
        return None
    # Keep worker labels, aggregation and measurement semantics; visual titles/IDs may differ.
    return json.dumps([signature, panel['axes'][series.get('axis', 'y')], origin],
                      sort_keys=True, allow_nan=False)


def _time_origin(run, index):
    """Use frozen clock semantics, never infer identity from a display label."""
    clock = run.report.run_meta.get('clock') or {}
    if isinstance(clock, dict):
        timeline = clock.get('report') or {}
        if not isinstance(timeline, dict):
            timeline = {}
        if timeline.get('status') == 'AVAILABLE' and timeline.get('origin', {}).get('selector'):
            return json.dumps(timeline['origin']['selector'], sort_keys=True)
        if isinstance(clock.get('origin_kind'), str) and clock['origin_kind'].strip():
            return json.dumps({'kind': clock['origin_kind'], 'unit': clock.get('unit', 'seconds')}, sort_keys=True)
    return f'unknown-origin-{index}'


def _pair(prepared, labels, origins, *, baseline_index, include_original_panels):
    """Preserve the reference layout; match series only inside the same panel ID."""
    overlays, pairing, consumed = [], [], set()
    colors = metric_colors(prepared[baseline_index])
    keys = {(i, p['id'], n): _series_key(p, c, origins[i])
            for i, panels in enumerate(prepared) for p in panels for n, c in enumerate(p['series'])}
    for original in prepared[baseline_index]:
        if original.get('type', 'line') != 'line' or not original.get('timeX'):
            continue
        candidates = []
        for i, panels in enumerate(prepared):
            for panel in panels:
                if panel['id'] == original['id']:
                    candidates.append((i, panel))
        # A repeated metric with different data inside one panel is ambiguous.
        maps = {}
        for i, panel in candidates:
            mapping = {}
            for n, curve in enumerate(panel['series']):
                key = keys[i, panel['id'], n]
                if key is not None:
                    mapping.setdefault(key, []).append(n)
            maps[i] = mapping
        entries, records = [], []
        for n, curve in enumerate(original['series']):
            key = keys[baseline_index, original['id'], n]
            base_matches = maps[baseline_index].get(key, [])
            matched = [(baseline_index, original, n)]
            if key is not None and len(base_matches) == 1:
                for i, panel in candidates:
                    if i == baseline_index:
                        continue
                    matches = maps[i].get(key, [])
                    if len(matches) == 1:
                        matched.append((i, panel, matches[0]))
            entries.extend(matched)
            records.append(dict(metric=curve.get('metric_id'), runs=[labels[i] for i, _, _ in matched],
                missing_runs=[label for label in labels if label not in [labels[i] for i, _, _ in matched]]))
        if not any(len(record['runs']) > 1 for record in records):
            continue
        panel = copy.deepcopy(original)
        panel.update(id='overlay:' + original['id'], title='对比 · ' + original.get('title', original['id']),
                     series=[], events=[], presets={}, timeSelectionKey='comparison-reference')
        metric_names = {}
        for i, source, n in entries:
            curve = copy.deepcopy(source['series'][n])
            name = curve['name']
            curve['name'] = labels[i] + ' · ' + name
            curve['group'] = labels[i]
            # Metric colors and visibility follow the reference report.
            key = keys[i, source['id'], n]
            reference_indices = maps[baseline_index].get(key, [])
            reference = original['series'][reference_indices[0]] if len(reference_indices) == 1 else None
            metric_color = curve.get('color') or series_color(None, n)
            if reference is not None:
                metric_color = colors[original['id'], original['series'].index(reference)]
                curve['hidden'] = reference.get('hidden', False)
            curve.update(curve_style(metric_color, i, len(labels)))
            panel['series'].append(curve)
            panel['presets'].setdefault('run · ' + labels[i], []).append(curve['name'])
            metric_names.setdefault(name, []).append(curve['name'])
            consumed.add((i, source['id'], n))
        for name, selection in metric_names.items():
            panel['presets']['指标 · ' + name] = selection
        for preset, names in original.get('presets', {}).items():
            panel['presets']['原预设 · ' + preset] = [curve['name'] for curve in panel['series']
                if any(curve['name'] == label + ' · ' + name for label in labels for name in names)]
        for i, source in candidates:
            if any(entry[0] == i for entry in entries):
                panel['events'].extend(dict(e, name=labels[i] + ' · ' + e['name']) for e in source.get('events', []))
        panel['caption'] = original.get('caption', '') + ' 指标用色相区分，图内相近色自动拉开；run 按输入顺序由深到浅，辅以稀疏形状标记，全部为连续实线。保留基准报告的指标组合、轴和显示范围；未兼容的其他 run 曲线单独展示。'
        overlays.append(panel)
        pairing.append(dict(panel=original['id'], status='PAIRED', metrics=records))
    separate = []
    for i, panels in enumerate(prepared):
        for source in panels:
            kept = [curve for n, curve in enumerate(source['series'])
                    if include_original_panels or (i, source['id'], n) not in consumed]
            if not kept:
                continue
            panel = copy.deepcopy(source)
            panel['series'] = kept
            names = {curve['name'] for curve in kept}
            panel['presets'] = {name: [n for n in choices if n in names]
                                for name, choices in panel.get('presets', {}).items()}
            panel['timeSelectionKey'] = f'run-{i + 1}'
            panel['id'] = f'run-{i + 1}:' + source['id']
            panel['title'] = labels[i] + ' · ' + source.get('title', source['id'])
            separate.append(panel)
            pairing.append(dict(panel=panel['id'], status='SEPARATE', runs=[labels[i]],
                reasons=['ORIGINAL_REQUESTED' if include_original_panels else
                         'PANEL_OR_MEASUREMENT_OR_CLOCK_NOT_COMPATIBLE']))
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
    sources, attachments, sections, prepared, kpis, overview = {}, {}, [], [], [], []
    ranges = []
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
        original = _original_result(run)
        kpis.append(dict(label=label + ' · 原始判定', value=original['verdict'] or '未知'))
        overview.append([label, spec.run_meta.get('identity'), run.input_kind, run.view,
                         original['verdict'], original['validity'], original['status'],
                         run.execution.get('duration_ms'), run.execution.get('cleanup'), list(spec.kpis)])
        panels = [shifted_panel(panel.to_dict(), shift) for panel in spec.panels]
        display = dict(min=spec.time_min - (shift or 0), max=spec.time_max - (shift or 0)) if (
            spec.time_min is not None and spec.time_max is not None) else None
        ranges.append(display)
        for panel in panels:
            panel['caption'] = panel.get('caption', '') + ' ' + caption
            if display is not None:
                panel['timeAxis'] = copy.deepcopy(display)
        prepared.append(panels)
    origins = ['EVENT_ALIGNED'] * len(loaded) if aligned else [
        _time_origin(run, i) for i, run in enumerate(loaded)]
    baseline_index = labels.index(baseline)
    panels, pairing = _pair(prepared, labels, origins, baseline_index=baseline_index,
                            include_original_panels=include_original_panels)
    time_alignment['display_ranges'] = dict(zip(labels, ranges))
    time_alignment['origins'] = dict(zip(labels, origins))
    display = ranges[baseline_index] or dict(min=0, max=1)
    result = dict(purpose='OBSERVATION_ONLY', sources=sources,
        runs={label: run.analysis for label, run in zip(labels, loaded)},
        executions={label: run.execution for label, run in zip(labels, loaded)},
        controls=dict(reference=baseline, values=dict(zip(labels, controls)), comparisons=comparisons),
        time_alignment=time_alignment, pairing=pairing)
    spec = dict(title='运行报告对照', subtitle='各 run 保留独立判定；差异不自动解释为因果。' + caption,
        timeOriginLabel=(caption if aligned else loaded[baseline_index].report.time_origin_label), timeAxis=display,
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
