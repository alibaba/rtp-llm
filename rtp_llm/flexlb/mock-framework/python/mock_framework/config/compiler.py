from pathlib import Path
import re

from .environment import compile_environment
from .errors import ConfigError
from .model import CallStep, CheckStep, CompiledCase, Expect, Phase, Reference
from .parser import load_yaml
from .registry import Registry
from .values import duration, fields, number, text


_ID = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')


def _compatible(source, expected):
    if expected.get('type') == 'union':
        return any(_compatible(source, variant) for variant in expected['variants'])
    if source.get('type') != expected.get('type'):
        return source.get('type') == 'integer' and expected.get('type') == 'number'
    if expected.get('name'):
        return source.get('name') == expected['name']
    return True


def _checks(value, schema, where):
    def has_reference(item):
        if isinstance(item, Reference):
            return True
        if isinstance(item, (list, tuple)):
            return any(has_reference(child) for child in item)
        return False

    for check in schema.get('checks', ()):
        if any(field not in value for field in check['fields']):
            continue
        values = [value[field] for field in check['fields']]
        if any(has_reference(item) for item in values):
            continue  # Actual resolved arguments must also be validated by the component.
        rule = check['rule']
        if rule == 'ordered':
            valid = values[0] <= values[1]
        elif rule == 'same_length':
            valid = len(values[0]) == len(values[1])
        elif rule == 'positive_sum':
            valid = any(item > 0 for item in values[0])
        else:
            raise ConfigError(f'Unknown catalog constraint: {rule}')
        if not valid:
            raise ConfigError(f'{where}: constraint {rule} failed for {check["fields"]}')


def _reference(value, schema, previous):
    fields(value, {'ref'}, {'ref'}, where='reference')
    parts = text(value['ref'], 'ref').split('.')
    if any(not _ID.fullmatch(part) for part in parts):
        raise ConfigError('Invalid reference path')
    if parts[0] not in previous:
        raise ConfigError(f'Reference must name an earlier step: {parts[0]}')
    source = previous[parts[0]]
    for part in parts[1:]:
        if part not in source.get('properties', {}):
            raise ConfigError(f'Unknown result field in reference: {value["ref"]}')
        source = source['properties'][part]
    if not _compatible(source, schema):
        raise ConfigError(f'Reference type mismatch: {value["ref"]}')
    return Reference(parts[0], tuple(parts[1:]))


def _value(value, schema, previous, where):
    if isinstance(value, dict) and 'ref' in value:
        return _reference(value, schema, previous)
    kind = schema['type']
    if kind == 'union':
        errors = []
        for variant in schema['variants']:
            try:
                return _value(value, variant, previous, where)
            except ConfigError as error:
                errors.append(str(error))
        raise ConfigError(f'{where}: no matching parameter variant: ' + '; '.join(errors))
    if kind == 'object':
        if 'values' in schema:
            if not isinstance(value, dict):
                raise ConfigError(f'{where}: expected mapping')
            return {k: _value(v, schema['values'], previous, where + '.' + k) for k, v in value.items()}
        props = schema.get('properties', {})
        fields(value, props, schema.get('required', ()), where=where)
        result = {k: _value(v, props[k], previous, where + '.' + k) for k, v in value.items()}
        _checks(result, schema, where)
        return result
    if kind == 'array':
        if not isinstance(value, list):
            raise ConfigError(f'{where}: expected list')
        if len(value) < schema.get('minItems', 0):
            raise ConfigError(f'{where}: too few items')
        return [_value(v, schema['items'], previous, where) for v in value]
    if kind == 'range':
        choices = schema.get('ranges', ())
        if isinstance(value, str) and value in ('all', 'latest') and value in choices:
            return value
        if 'interval' not in choices or not isinstance(value, list) or len(value) != 2:
            raise ConfigError(f'{where}: unsupported range {value!r}')
        start, end = (duration(v, negative=True) for v in value)
        if not start < end <= 0:
            raise ConfigError(f'{where}: range must satisfy start < end <= 0')
        return list(value)  # Deliberately not resolved to clock times at compile time.
    if kind == 'duration':
        duration(value)
        return value
    if kind == 'string':
        text(value, where)
        if 'enum' in schema and value not in schema['enum']:
            raise ConfigError(f'{where}: expected one of {schema["enum"]}')
        return value
    if kind == 'integer':
        if type(value) is not int:
            raise ConfigError(f'{where}: expected integer')
        bits = schema.get('bits', 64)
        if not -(2 ** (bits - 1)) <= value < 2 ** (bits - 1):
            raise ConfigError(f'{where}: integer overflow')
    elif kind == 'number':
        number(value)
    elif kind == 'boolean':
        if type(value) is not bool:
            raise ConfigError(f'{where}: expected boolean')
    else:
        raise ConfigError(f'Unknown catalog type: {kind}')
    if 'minimum' in schema and value < schema['minimum']:
        raise ConfigError(f'{where}: below minimum {schema["minimum"]}')
    return value


def _expect(raw, spec, previous):
    fields(raw, {'eq', 'gte', 'lte', 'count'}, where='expect')
    if not raw or ('eq' in raw and len(raw) != 1) or ('count' in raw and len(raw) != 1):
        raise ConfigError('expect needs eq, count, or numeric gte/lte bounds')
    result = spec['returns']
    name = result.get('name')
    if name == 'MetricValue' or name == 'QueryResult':
        target = {'type': 'number'}
    elif name == 'PerMasterEngines':
        target = result['properties']['instances']['values']
    else:
        target = result
    comparisons = {}
    for operator, value in raw.items():
        if operator == 'count':
            if name not in ('MasterSet', 'EngineSet') and result['type'] != 'array':
                raise ConfigError('count requires a collection result')
            comparisons[operator] = _value(value, {'type': 'integer', 'minimum': 0}, previous, 'expect.count')
        elif isinstance(value, dict) and 'ref' in value:
            comparisons[operator] = _reference(value, target, previous)
        elif operator in ('gte', 'lte') or target['type'] == 'number':
            if target['type'] not in ('number', 'integer'):
                raise ConfigError(f'{operator} requires numeric output')
            if isinstance(value, str):
                if spec.get('unit') != 's':
                    raise ConfigError('Duration expectations require an interface declaring seconds')
                duration(value)
                comparisons[operator] = value
            else:
                comparisons[operator] = number(value)
        else:
            comparisons[operator] = _value(value, target, previous, 'expect.' + operator)
    if {'gte', 'lte'} <= comparisons.keys() and not any(isinstance(v, Reference) for v in comparisons.values()):
        def bound(v):
            return duration(v) if isinstance(v, str) else v
        if bound(comparisons['gte']) > bound(comparisons['lte']):
            raise ConfigError('expect lower bound exceeds upper bound')
    return Expect(comparisons)


def compile_case(path: str | Path, registry: Registry, *, shared_root: str | Path | None = None) -> CompiledCase:
    path = Path(path).resolve()
    raw = load_yaml(path)
    fields(raw, {'id', 'name', 'describe', 'environment', 'phases', 'view'},
           {'id', 'describe', 'environment', 'phases'}, where='case')
    ids, previous = set(), {}
    def identity(value):
        text(value, 'id')
        if not _ID.fullmatch(value) or value in ids:
            raise ConfigError(f'Invalid or duplicate id: {value}')
        ids.add(value)
        return value
    case_id = identity(raw['id'])
    describe = text(raw['describe'], 'describe')
    name = text(raw['name'], 'name') if 'name' in raw else None
    environment = compile_environment(raw['environment'])
    def asset(value):
        value = text(value, 'path')
        if '://' in value:
            raise ConfigError('Only local/shared dataset and view paths are supported by this compiler')
        candidate = Path(value)
        if candidate.is_absolute():
            return candidate.resolve()
        base = Path(shared_root).resolve() if shared_root is not None and candidate.parts[0] in ('data', 'datasets', 'views') else path.parent
        return (base / candidate).resolve()
    if not isinstance(raw['phases'], list) or not raw['phases']:
        raise ConfigError('phases must be a nonempty list')
    phases, traffic_started = [], False
    for phase in raw['phases']:
        fields(phase, {'id', 'describe', 'timeout', 'steps'}, {'id', 'describe', 'timeout', 'steps'}, where='phase')
        phase_id = identity(phase['id'])
        text(phase['describe'], 'phase.describe')
        duration(phase['timeout'], positive=True)
        if not isinstance(phase['steps'], list) or not phase['steps']:
            raise ConfigError('steps must be a nonempty list')
        steps = []
        for wrapper in phase['steps']:
            if not isinstance(wrapper, dict) or len(wrapper) != 1 or next(iter(wrapper)) not in ('call', 'check'):
                raise ConfigError('Each step needs exactly one call or check')
            kind, step = next(iter(wrapper.items()))
            allowed = {'id', 'describe', 'interface', 'args'} | ({'run', 'within'} if kind == 'call' else {'expect'})
            fields(step, allowed, {'id', 'describe', 'interface'} | ({'expect'} if kind == 'check' else set()), where=kind)
            step_id = identity(step['id'])
            text(step['describe'], 'step.describe')
            spec = registry.get(step['interface'])
            if spec['kind'] != ('action' if kind == 'call' else 'read'):
                raise ConfigError('call/check does not match interface category')
            args = _value(step.get('args', {}), spec['args'], previous, step_id + '.args')
            if step['interface'] in ('client/traffic/stop', 'client/traffic/drain') and not traffic_started:
                raise ConfigError('traffic control requires a preceding traffic/run')
            if 'range' in args and args['range'] == 'all' and not traffic_started:
                raise ConfigError('range: all requires a preceding traffic/run')
            if step['interface'] == 'client/traffic/run':
                if traffic_started:
                    raise ConfigError('Only one traffic/run is allowed per case')
                if 'client' not in environment:
                    raise ConfigError('traffic/run requires environment.client')
                traffic_started = True
                args['input']['trace'] = str(asset(args['input']['trace']))
                arrival = args['playback'].get('arrival')
                if arrival:
                    if arrival['mode'] == 'uniform':
                        if 'qps' not in arrival or 'speed' in arrival:
                            raise ConfigError('uniform arrival requires qps and forbids speed')
                    elif 'qps' in arrival or 'ramp' in arrival:
                        raise ConfigError('trace arrival forbids qps and ramp')
            if kind == 'call':
                run = step.get('run', 'foreground')
                if run not in ('foreground', 'background') or (run == 'background' and not spec['background']):
                    raise ConfigError('Unsupported run mode')
                within = step.get('within')
                if within is not None:
                    duration(within, positive=True)
                compiled = CallStep(step_id, step['describe'], step['interface'], args, run, within)
            else:
                compiled = CheckStep(step_id, step['describe'], step['interface'], args, _expect(step['expect'], spec, previous))
            previous[step_id] = spec['returns']
            steps.append(compiled)
        phases.append(Phase(phase_id, phase['describe'], phase['timeout'], tuple(steps)))
    view = asset(raw['view']) if 'view' in raw else None
    return CompiledCase(case_id, describe, name, path.parent, environment, tuple(phases), view, path)
