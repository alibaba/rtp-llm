"""Read saved facts; damaged optional artifacts never change the experiment verdict."""
from datetime import datetime
import json
import math
from pathlib import Path


def reject_constant(value):
    raise ValueError(f'Non-finite JSON number: {value}')


def finite_float(value):
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f'Non-finite JSON number: {value}')
    return parsed


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'), parse_constant=reject_constant, parse_float=finite_float)
    except OSError as error:
        raise ValueError(f'{path}: {error.strerror}') from error
    except (UnicodeError, ValueError) as error:
        raise ValueError(f'{path}: {error}') from error


def milliseconds(value):
    value = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if value.tzinfo is None:
        raise ValueError('timestamp requires a timezone')
    return round(value.timestamp() * 1000)


def experiment(directory):
    directory = Path(directory)
    path = directory / 'result.json'
    result = read_json(path)
    try:
        if not isinstance(result, dict):
            raise ValueError('expected an object')
        if 'finished_at' not in result:
            raise ValueError('only completed experiment outputs can be viewed: missing finished_at')
        start, end = (milliseconds(result[key]) for key in ('started_at', 'finished_at'))
        if end <= start:
            raise ValueError('finished_at must be after started_at')
        if result.get('status') not in ('PASS', 'FAIL', 'ERROR', 'CANCELLED'):
            raise ValueError('invalid status')
        if not isinstance(result.get('cleanup_errors', []), list):
            raise ValueError('cleanup_errors must be a list')
    except (ValueError, KeyError, TypeError, AttributeError, OverflowError) as error:
        raise ValueError(f'{path}: {error}') from error

    warnings = []

    def optional(name, kind):
        try:
            value = read_json(directory / name)
            if not isinstance(value, kind):
                raise ValueError(f'{name}: expected {kind.__name__}')
            return value
        except ValueError as error:
            warnings.append(str(error).removeprefix(str(directory) + '/'))
            return None

    def lines(name):
        try:
            text = (directory / name).read_text(encoding='utf-8')
        except (OSError, UnicodeError) as error:
            warnings.append(f'{name}: {error.strerror if isinstance(error, OSError) else error}')
            return []
        rows = []
        for number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line, parse_constant=reject_constant, parse_float=finite_float)
                if not isinstance(row, dict):
                    raise ValueError('expected an object')
                rows.append((number, row))
            except ValueError as error:
                warnings.append(f'{name}:{number}: {error}')
        return rows

    configuration = optional('configuration.json', dict)
    case = configuration.get('case', {}) if configuration else {}
    if not isinstance(case, dict):
        warnings.append('configuration.json: case must be an object')
        case = {}
    steps = {}
    for phase in case.get('phases', []) if isinstance(case.get('phases', []), list) else []:
        if isinstance(phase, dict) and isinstance(phase.get('steps'), list):
            for step in phase['steps']:
                if isinstance(step, dict) and isinstance(step.get('id'), str):
                    steps[step['id']] = step
    interfaces = optional('interfaces.json', dict)
    interfaces = interfaces.get('interfaces', {}) if interfaces else {}
    if not isinstance(interfaces, dict):
        warnings.append('interfaces.json: interfaces must be an object')
        interfaces = {}
    windows, events = {}, []
    for number, event in lines('events.jsonl'):
        try:
            kind = event['event']
            if kind == 'metric_window':
                a, b = event['start'], event['end']
                if any(type(value) not in (int, float) or not math.isfinite(value) for value in (a, b)) or b < a:
                    raise ValueError('invalid metric_window start/end')
                windows[event['query_time']] = (round(a * 1000), round(b * 1000))
            elif kind in ('phase_started', 'phase_finished', 'step_started', 'step_finished'):
                name = event.get('step', event.get('phase'))
                events.append({'time': milliseconds(event['recorded_at']),
                               'text': f"{name} · {kind}" + (' · ' + str(event['status']) if 'status' in event else '')})
        except (ValueError, KeyError, TypeError, AttributeError, OverflowError) as error:
            warnings.append(f'events.jsonl:{number}: {error}')
    gates = []
    for number, gate in lines('gates.jsonl'):
        try:
            point = milliseconds(gate['query_time'])
            if not all(isinstance(gate.get(key), str) for key in ('step', 'interface')):
                raise ValueError('step and interface must be strings')
            if gate.get('status') not in ('PASS', 'FAIL', 'ERROR', 'CANCELLED'):
                raise ValueError('invalid gate status')
            if not isinstance(gate.get('expect'), dict) or not isinstance(gate.get('args'), dict):
                raise ValueError('expect and args must be objects')
            definition = interfaces.get(gate['interface'], {})
            a, b = windows.get(gate['query_time'], (None, None))
            gates.append({**gate, 'title': steps.get(gate['step'], {}).get('describe', gate['step']), 'point': point,
                          'unit': definition.get('unit', '') if isinstance(definition, dict) else '', 'from': a, 'to': b})
            events.append({'time': a if a is not None else point, 'timeEnd': b,
                           'text': f"门禁 {gate['step']} · {gate['status']}"})
        except (ValueError, KeyError, TypeError, AttributeError, OverflowError) as error:
            warnings.append(f'gates.jsonl:{number}: {error}')
    try:
        source = (directory / 'case.yaml').read_text(encoding='utf-8')
    except (OSError, UnicodeError) as error:
        warnings.append(f'case.yaml: {error.strerror if isinstance(error, OSError) else error}')
        source = None
    effective = {path.parent.name: optional(str(path.relative_to(directory)), dict)
                 for path in sorted(directory.glob('components/*/effective-config.json'))}
    collection = optional('prometheus/collection.json', dict)
    if collection is not None and type(collection.get('complete')) is not bool:
        warnings.append('prometheus/collection.json: complete must be a boolean')
        collection = None
    versions = optional('code.json', list)
    return {'name': case.get('name') or case.get('id') or directory.name,
            'describe': case.get('describe', ''), 'result': result, 'start': start, 'end': end,
            'configuration': configuration, 'versions': versions, 'source': source, 'effective': effective,
            'gates': gates, 'events': sorted(events, key=lambda event: event['time']),
            'collection': collection, 'warnings': warnings}
