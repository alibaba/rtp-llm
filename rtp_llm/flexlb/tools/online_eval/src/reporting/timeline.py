"""Freeze a run's configured display clock, independent of gate windows."""

import copy
import math
import re

from input_contract import mapping_fields


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_configuration(data, stage_ids):
    mapping_fields(data, {'time_axis'}, 'reporting', required={'time_axis'})
    axis = mapping_fields(data['time_axis'], {'origin', 'range'}, 'reporting.time_axis',
                          required={'origin', 'range'})
    bounds = mapping_fields(axis['range'], {'from', 'until'}, 'reporting.time_axis.range',
                            required={'from', 'until'})
    selectors = [axis['origin'], bounds['until']]
    if bounds['from'] != 'origin':
        selectors.append(bounds['from'])
    for selector in selectors:
        if not isinstance(selector, dict):
            raise ValueError('report time boundary requires an event or stage selector')
        if set(selector) == {'event'}:
            identity = selector['event']
        elif set(selector) == {'stage', 'boundary'}:
            identity = selector['stage']
            if (type(identity) is not str or identity not in stage_ids
                    or type(selector['boundary']) is not str
                    or selector['boundary'] not in {'start', 'end'}):
                raise ValueError('report time boundary references unknown stage or boundary')
        else:
            raise ValueError('report time boundary requires event or stage + boundary')
        if type(identity) is not str or not re.fullmatch(r'[a-z][a-z0-9_]*', identity):
            raise ValueError('invalid report time boundary identity')
    return copy.deepcopy(data)


def traffic_event(resources):
    """Earliest actual issued request, including unsuccessful/unfinished requests."""
    candidates = [(row['traffic_start'], row['resource']) for row in resources
                  if row.get('traffic_start') is not None]
    if not candidates:
        return None
    stamp, resource = min(candidates, key=lambda item: item[0]['epoch_s'])
    if not _finite(stamp['epoch_s']) or stamp['epoch_s'] <= 0:
        raise ValueError('invalid first-request timestamp')
    return dict(id='traffic_started', epoch_s=stamp['epoch_s'],
                source=dict(resource=copy.deepcopy(resource), request=copy.deepcopy(stamp)))


def freeze(data, *, events, phases):
    """Persist resolved boundaries; missing events never select another clock."""
    missing = []

    def resolve(selector):
        rows = ([row for row in events if row['id'] == selector['event']]
                if 'event' in selector else
                [row for row in phases if row['stage'] == selector['stage']
                 and row['event'] == selector['boundary']])
        if len(rows) > 1:
            raise ValueError('ambiguous report time boundary: ' + str(selector))
        if not rows:
            missing.append(copy.deepcopy(selector))
            return None
        stamp = rows[0].get('epoch_s')
        if not _finite(stamp):
            raise ValueError('report time boundary lacks finite epoch_s')
        return dict(epoch_s=stamp, selector=copy.deepcopy(selector), record=copy.deepcopy(rows[0]))

    axis = data['time_axis']
    origin = resolve(axis['origin'])
    start = origin if axis['range']['from'] == 'origin' else resolve(axis['range']['from'])
    end = resolve(axis['range']['until'])
    if missing:
        return dict(status='UNAVAILABLE', declaration=copy.deepcopy(data), missing=missing)
    if start['epoch_s'] >= end['epoch_s']:
        raise ValueError('report display range must have increasing boundaries')
    return dict(status='AVAILABLE', declaration=copy.deepcopy(data), origin=origin,
                start=start, end=end)


def apply(spec, analysis):
    """Project every temporal coordinate once using the frozen display clock."""
    timeline = analysis.get('report_timeline')
    if timeline is None:
        return spec
    spec['reportTimeline'] = copy.deepcopy(timeline)
    if timeline['status'] == 'UNAVAILABLE':
        if analysis['status'] not in {'FAIL', 'ERROR', 'TIMEOUT', 'BLOCKED', 'SKIP'}:
            raise ValueError('configured report time boundary did not occur: ' + str(timeline['missing']))
        spec.pop('timeOriginEpochS', None)
        spec.pop('timeAxis', None)
        spec['timeOriginLabel'] = '未建立报告时间轴：配置的事件未发生'
        spec['events'] = []
        for panel in spec['panels']:
            if panel.get('timeX'):
                panel['series'] = []
                panel['events'] = []
                panel['caption'] = spec['timeOriginLabel']
        return spec
    old_origin = spec['timeOriginEpochS']
    origin = timeline['origin']['epoch_s']
    delta = old_origin - origin
    lo, hi = (timeline[key]['epoch_s'] - origin for key in ('start', 'end'))
    for panel in spec['panels']:
        if not panel.get('timeX'):
            continue
        for curve in panel['series']:
            curve['points'] = [dict(point, x=point['x'] + delta) for point in curve['points']]
        for event in panel.get('events', []):
            event['t'] += delta
    for event in spec.get('events', []):
        event['t'] += delta
    selector = timeline['origin']['selector']
    label = ('主业务流量首次发出请求' if selector == {'event': 'traffic_started'} else
             selector.get('event') or selector['stage'] + '.' + selector['boundary'])
    spec.update(timeOriginEpochS=origin, timeOriginLabel='秒；t=0 为 ' + label,
                timeAxis=dict(min=lo, max=hi))
    return spec


def archived(spec, directory, presentation, *, status):
    """Offline report publication uses the frozen clock, never current case YAML."""
    from monitoring.metric_store import MetricStore
    context = MetricStore.read(directory).document.get('run', {})
    result = apply(spec, dict(context, status=status))
    if (context.get('report_timeline') or {}).get('status') == 'AVAILABLE':
        from reporting.events import attach_events
        attach_events(result, presentation, origin=result['timeOriginEpochS'],
                      events=context['events'], phases=context['phases'])
    return result
