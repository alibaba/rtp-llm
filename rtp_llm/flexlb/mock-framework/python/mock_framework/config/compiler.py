from pathlib import Path
import re

from .environment import compile_environment
from .errors import ConfigError
from .model import Step, CompiledCase, Expect, Phase, Reference
from .parser import load_yaml
from .registry import Registry
from .validation import validate
from .values import duration, fields, number, text

_ID = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')
_GLOBALS = {'type': 'object', 'properties': {
    role: {'type': 'array', 'items': {'type': 'string'}}
    for role in ('masters', 'prefill', 'decode', 'vit')}, 'required': []}


def compatible(source, expected):
    if source['type'] == 'union':
        return all(compatible(v, expected) for v in source['variants'])
    if expected['type'] == 'union':
        return any(compatible(source, v) for v in expected['variants'])
    if source['type'] != expected['type']:
        return source['type'] == 'integer' and expected['type'] == 'number'
    if source['type'] == 'array':
        return compatible(source['items'], expected['items'])
    if source['type'] == 'object':
        if 'values' in expected:
            return 'values' in source and compatible(source['values'], expected['values'])
        return set(expected.get('required', ())) <= set(source.get('required', ())) and all(
            k in expected['properties'] and compatible(v, expected['properties'][k])
            for k, v in source.get('properties', {}).items())
    return True


def reference(value, expected, previous):
    fields(value, {'ref'}, {'ref'}, where='reference')
    parts = text(value['ref'], 'ref').split('.')
    if any(not _ID.fullmatch(part) for part in parts):
        raise ConfigError('Invalid reference path')
    if parts[0] == 'steps' and len(parts) >= 2 and parts[1] in previous:
        source, path, definition = parts[1], parts[2:], previous[parts[1]]
    elif parts[0] == 'globals' and len(parts) >= 2:
        source, path, definition = 'globals', parts[1:], _GLOBALS
    else:
        raise ConfigError('Reference must name an earlier steps.<id> or globals.<field>')
    for part in path:
        if part not in definition.get('properties', {}):
            raise ConfigError(f'Unknown result field in reference: {value["ref"]}')
        definition = definition['properties'][part]
    if not compatible(definition, expected):
        raise ConfigError(f'Reference type mismatch: {value["ref"]}')
    return Reference(source, tuple(path))


def argument(value, spec, previous, where):
    if isinstance(value, dict) and 'ref' in value:
        return reference(value, spec, previous)
    if spec['type'] == 'union':
        for variant in spec['variants']:
            try:
                return argument(value, variant, previous, where)
            except ConfigError:
                pass
        raise ConfigError(f'{where}: no matching parameter variant')
    if spec['type'] == 'object' and isinstance(value, dict):
        if any(type(k) is not str for k in value):
            raise ConfigError(f'{where}: object keys must be strings')
        if 'values' not in spec:
            fields(value, spec['properties'], spec['required'], where=where)
        return {k: argument(v, spec['values'] if 'values' in spec else spec['properties'][k], previous, where + '.' + k)
                for k, v in value.items()}
    if spec['type'] == 'array' and isinstance(value, list):
        return [argument(v, spec['items'], previous, where + '[]') for v in value]
    return validate(value, spec, where)


def has_reference(value):
    if isinstance(value, Reference):
        return True
    if isinstance(value, dict):
        return any(has_reference(v) for v in value.values())
    if isinstance(value, list):
        return any(has_reference(v) for v in value)
    return False


def expectation(raw, interface, previous):
    fields(raw, {'eq', 'count', 'gte', 'lte'}, where='expect')
    if not raw or ('eq' in raw and len(raw) != 1) or ('count' in raw and len(raw) != 1):
        raise ConfigError('expect needs eq, count, or gte/lte')
    result, output = {}, interface.returns
    for op, value in raw.items():
        expected = output
        if op == 'count':
            if output['type'] != 'array':
                raise ConfigError('count requires a list result')
            expected = {'type': 'integer', 'minimum': 0}
        elif op in ('gte', 'lte'):
            if output['type'] not in ('integer', 'number'):
                raise ConfigError(f'{op} requires numeric output')
        if isinstance(value, str) and interface.unit == 's' and op != 'count':
            value = float(duration(value))
        result[op] = argument(value, expected, previous, 'expect.' + op)
    if {'gte', 'lte'} <= result.keys() and not has_reference(result) and result['gte'] > result['lte']:
        raise ConfigError('expect lower bound exceeds upper bound')
    return Expect(result)


def compile_case(path: str | Path, registry: Registry | None = None, *, shared_root=None) -> CompiledCase:
    registry = registry or Registry.standard()
    path = Path(path).resolve()
    raw = load_yaml(path)
    fields(raw, {'id', 'name', 'describe', 'environment', 'phases', 'view'},
           {'id', 'describe', 'environment', 'phases'}, where='case')
    ids, previous = {'globals', 'steps'}, {}
    def identity(value):
        text(value, 'id')
        if not _ID.fullmatch(value) or value in ids:
            raise ConfigError(f'Invalid or duplicate id: {value}')
        ids.add(value)
        return value
    case_id = identity(raw['id'])
    describe = text(raw['describe'], 'describe')
    name = text(raw['name'], 'name') if 'name' in raw else None
    environment = compile_environment(raw['environment'], directory=path.parent)
    def asset(value):
        value = text(value, 'path')
        if '://' in value:
            raise ConfigError('Only local/shared paths are supported')
        candidate = Path(value)
        base = Path(shared_root).resolve() if shared_root and candidate.parts[0] in ('data', 'datasets', 'views') else path.parent
        return (base / candidate).resolve()
    if not isinstance(raw['phases'], list) or not raw['phases']:
        raise ConfigError('phases must be a nonempty list')
    phases = []
    for phase in raw['phases']:
        fields(phase, {'id', 'describe', 'timeout', 'steps'}, {'id', 'describe', 'timeout', 'steps'}, where='phase')
        phase_id = identity(phase['id'])
        text(phase['describe'], 'phase.describe')
        duration(phase['timeout'], positive=True)
        if not isinstance(phase['steps'], list) or not phase['steps']:
            raise ConfigError('steps must be a nonempty list')
        steps = []
        for step in phase['steps']:
            fields(step, {'id', 'describe', 'interface', 'args', 'expect', 'within', 'wait_after'},
                   {'id', 'describe', 'interface'}, where='step')
            step_id = identity(step['id'])
            description = text(step['describe'], 'step.describe')
            interface = registry.get(step['interface'])
            args = argument(step.get('args', {}), interface.args, previous, step_id + '.args')
            if not has_reference(args):
                interface.arguments(args)
            within = step.get('within')
            if within is not None:
                duration(within, positive=True)
            wait_after = step.get('wait_after', '0s')
            duration(wait_after)
            expect = expectation(step['expect'], interface, previous) if 'expect' in step else None
            steps.append(Step(step_id, description, interface.name, args, expect, within, wait_after))
            previous[step_id] = interface.returns
        phases.append(Phase(phase_id, phase['describe'], phase['timeout'], tuple(steps)))
    return CompiledCase(case_id, describe, name, path.parent, environment, tuple(phases),
                        asset(raw['view']) if 'view' in raw else None, path)
