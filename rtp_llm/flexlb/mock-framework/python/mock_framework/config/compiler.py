"""Compile Case structure and typed references without starting runtime components."""
from functools import partial
from pathlib import Path
import re

from .environment import compile_environment
from .errors import ConfigError
from .model import Step, CompiledCase, Expect, Phase, Reference
from .parser import load_yaml
from .registry import Registry
from .validation import compatible, duration, fields, text, validate

_ID = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')
_GLOBALS = {'type': 'object', 'properties': {
    role: {'type': 'array', 'items': {'type': 'string'}}
    for role in ('masters', 'prefill', 'decode', 'vit')}, 'required': []}


def _reference(value, expected, where, *, previous):
    fields(value, {'ref'}, {'ref'}, where=where)
    parts = text(value['ref'], where + '.ref').split('.')
    if any(not _ID.fullmatch(part) for part in parts):
        raise ConfigError(f'{where}: invalid reference path')
    if parts[0] == 'steps' and len(parts) >= 2 and parts[1] in previous:
        source, path, definition = parts[1], parts[2:], previous[parts[1]]
    elif parts[0] == 'globals' and len(parts) >= 2:
        source, path, definition = 'globals', parts[1:], _GLOBALS
    else:
        raise ConfigError(f'{where}: Reference must name an earlier steps.<id> or globals.<field>')
    for part in path:
        if part not in definition.get('properties', {}):
            raise ConfigError(f'{where}: Unknown result field in reference: {value["ref"]}')
        definition = definition['properties'][part]
    if not compatible(definition, expected):
        raise ConfigError(f'{where}: Reference type mismatch: {value["ref"]}')
    return Reference(source, tuple(path))


def _argument(value, spec, previous, where):
    return validate(value, spec, where, reference=partial(_reference, previous=previous))


def _has_reference(value):
    if isinstance(value, Reference):
        return True
    if isinstance(value, dict):
        return any(_has_reference(v) for v in value.values())
    if isinstance(value, list):
        return any(_has_reference(v) for v in value)
    return False


def _expectation(raw, interface, previous):
    fields(raw, {'eq', 'count', 'gte', 'lte'}, where='expect')
    if set(raw) not in ({'eq'}, {'count'}, {'gte'}, {'lte'}, {'gte', 'lte'}):
        raise ConfigError('expect needs eq, count, or gte/lte')
    result = {}
    output = interface.returns
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
        result[op] = _argument(value, expected, previous, 'expect.' + op)
    if {'gte', 'lte'} <= result.keys() and not _has_reference(result) and result['gte'] > result['lte']:
        raise ConfigError('expect lower bound exceeds upper bound')
    return Expect(result)


def _identity(value, ids):
    text(value, 'id')
    if not _ID.fullmatch(value) or value in ids:
        raise ConfigError(f'Invalid or duplicate id: {value}')
    ids.add(value)
    return value


def _view_path(value, directory, shared_root):
    value = text(value, 'view')
    if '://' in value:
        raise ConfigError('Only local/shared paths are supported')
    candidate = Path(value)
    if not candidate.name:
        raise ConfigError('view: expected a dashboard file path')
    base = directory
    if shared_root is not None and candidate.parts[0] in ('data', 'datasets', 'views'):
        base = Path(shared_root).resolve()
    return (base / candidate).resolve()


def _compile_step(raw, registry, ids, previous):
    fields(raw, {'id', 'describe', 'interface', 'args', 'expect', 'within', 'wait_after'},
           {'id', 'describe', 'interface'}, where='step')
    step_id = _identity(raw['id'], ids)
    description = text(raw['describe'], 'step.describe')
    interface = registry.get(raw['interface'])
    args = _argument(raw.get('args', {}), interface.args, previous, step_id + '.args')
    if not _has_reference(args):
        interface.arguments(args)
    within = raw.get('within')
    if within is not None:
        duration(within, positive=True)
    wait_after = raw.get('wait_after', '0s')
    duration(wait_after)
    expect = _expectation(raw['expect'], interface, previous) if 'expect' in raw else None
    previous[step_id] = interface.returns
    return Step(step_id, description, interface.name, args, expect, within, wait_after)


def _compile_phase(raw, registry, ids, previous):
    fields(raw, {'id', 'describe', 'timeout', 'steps'},
           {'id', 'describe', 'timeout', 'steps'}, where='phase')
    phase_id = _identity(raw['id'], ids)
    description = text(raw['describe'], 'phase.describe')
    duration(raw['timeout'], positive=True)
    if not isinstance(raw['steps'], list) or not raw['steps']:
        raise ConfigError('steps must be a nonempty list')
    steps = []
    for step in raw['steps']:
        steps.append(_compile_step(step, registry, ids, previous))
    return Phase(phase_id, description, raw['timeout'], tuple(steps))


def compile_case(path: str | Path, registry: Registry | None = None, *, shared_root=None) -> CompiledCase:
    path = Path(path).resolve()
    raw = load_yaml(path)
    fields(raw, {'id', 'name', 'describe', 'environment', 'phases', 'view'},
           {'id', 'describe', 'environment', 'phases'}, where='case')
    registry = registry or Registry.standard()
    ids = {'globals', 'steps'}
    previous = {}
    case_id = _identity(raw['id'], ids)
    description = text(raw['describe'], 'describe')
    name = text(raw['name'], 'name') if 'name' in raw else None
    environment = compile_environment(raw['environment'], directory=path.parent)
    if not isinstance(raw['phases'], list) or not raw['phases']:
        raise ConfigError('phases must be a nonempty list')
    phases = []
    for phase in raw['phases']:
        phases.append(_compile_phase(phase, registry, ids, previous))
    view = Path(__file__).resolve().parents[3] / 'views/default.json'
    if 'view' in raw:
        view = _view_path(raw['view'], path.parent, shared_root)
    return CompiledCase(id=case_id, describe=description, name=name, directory=path.parent,
                        environment=environment, phases=tuple(phases), view=view, source=path)
