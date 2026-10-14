"""Frozen numeric references owned by adjudication, independent of presentation."""
import copy
import math


def definition(check, reference, metric_id, unit, *, labels=None, description):
    return dict(check=check, reference=reference, metric_id=metric_id, unit=unit,
                labels=dict(labels or {}), description=description)


def freeze_reference(spec, value, windows, *, op, status='AVAILABLE', reason=''):
    result = dict(copy.deepcopy(spec), value=value, windows=copy.deepcopy(windows),
                  op=op, status=status, reason=reason)
    validate_reference(result)
    return result


def validate_reference(value):
    required = {'check', 'reference', 'metric_id', 'unit', 'labels', 'description',
                'value', 'windows', 'op', 'status', 'reason'}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError('invalid frozen gate reference fields')
    if any(type(value[k]) is not str or not value[k].strip()
           for k in ('check', 'reference', 'metric_id', 'unit', 'description')):
        raise ValueError('invalid gate reference identity or semantics')
    if not isinstance(value['labels'], dict) or any(type(k) is not str or type(v) is not str
                                                   for k, v in value['labels'].items()):
        raise ValueError('invalid gate reference labels')
    if value['op'] not in {'ge', 'le', 'eq', 'gt', 'lt'}:
        raise ValueError('invalid gate reference comparison')
    if value['status'] not in {'AVAILABLE', 'UNAVAILABLE'} or type(value['reason']) is not str:
        raise ValueError('invalid gate reference availability')
    if not isinstance(value['windows'], list):
        raise ValueError('gate reference windows must be a list')
    if value['value'] is not None and (type(value['value']) not in (int, float) or not math.isfinite(value['value'])):
        raise ValueError('gate reference threshold must be finite or unavailable')
    if value['status'] == 'UNAVAILABLE':
        if not value['reason'] or value['windows']:
            raise ValueError('unavailable gate reference requires a reason and no windows')
    elif (type(value['value']) not in (int, float) or not math.isfinite(value['value'])
          or not value['windows']):
        raise ValueError('gate reference requires a finite threshold and resolved windows')
    previous = None
    for window in value['windows']:
        if (not isinstance(window, dict) or set(window) != {'name', 'start_epoch_s', 'end_epoch_s'}
                or type(window['name']) is not str or not window['name']
                or any(type(window[k]) not in (int, float) or not math.isfinite(window[k])
                       for k in ('start_epoch_s', 'end_epoch_s'))
                or window['start_epoch_s'] >= window['end_epoch_s']
                or previous is not None and window['start_epoch_s'] < previous):
            raise ValueError('gate reference requires ordered, disjoint, finite windows')
        previous = window['end_epoch_s']


def window(name, start, end):
    return dict(name=name, start_epoch_s=start, end_epoch_s=end)
