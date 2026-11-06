"""Compile Case structure and typed references without starting runtime components."""
import copy
from dataclasses import dataclass
from functools import partial
from typing import Any
from pathlib import Path
import re

from ..execution.step.step import Expect, Reference, Step
from .yaml_loader import load_yaml
from ..execution.step.step_registry import StepRegistry
from .validation import ConfigError, compatible, duration, fields, number, text, validate

@dataclass(frozen=True)
class Phase:
    id: str
    describe: str
    timeout: str
    steps: tuple[Step, ...]


@dataclass(frozen=True)
class CompiledCase:
    id: str
    describe: str
    name: str | None
    directory: Path
    environment: dict[str, Any]
    phases: tuple[Phase, ...]
    view: Path | None
    source: Path
    shared_root: Path | None = None


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
        if definition is None:
            raise ConfigError(f'{where}: Background Step {source!r} does not provide a referenceable result')
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
            variants = output.get('variants', [output])
            if not all(v['type'] in ('integer', 'number') or
                       v['type'] == 'array' and v['items']['type'] in ('integer', 'number') for v in variants):
                raise ConfigError(f'{op} requires numeric output')
            expected = {'type': 'number'}
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
    fields(raw, {'id', 'describe', 'interface', 'args', 'expect', 'within', 'wait_after', 'background'},
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
    background = raw.get('background', False)
    if type(background) is not bool:
        raise ConfigError(f'{step_id}.background: expected a boolean')
    result_type = interface.returns
    if result_type.get('variants') == [{'type': 'number'}, {'type': 'array', 'items': {'type': 'number'}}]:
        result_type = result_type['variants'][1 if args.get('mode', 'interval') == 'series' else 0]
    previous[step_id] = None if background else result_type
    return Step(step_id, description, interface.name, args, expect, within, wait_after, background, action=interface)


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


def compile_case(path: str | Path, registry: StepRegistry | None = None, *, shared_root=None) -> CompiledCase:
    path = Path(path).resolve()
    raw = load_yaml(path)
    fields(raw, {'id', 'name', 'describe', 'environment', 'phases', 'view'},
           {'id', 'describe', 'environment', 'phases'}, where='case')
    registry = registry or StepRegistry.standard()
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
    framework = Path(__file__).resolve().parents[3]
    data_root = framework if shared_root is None else Path(shared_root).resolve()
    view = framework / 'views/default.json'
    if 'view' in raw:
        view = _view_path(raw['view'], path.parent, shared_root)
    return CompiledCase(id=case_id, describe=description, name=name, directory=path.parent,
                        environment=environment, phases=tuple(phases), view=view, source=path, shared_root=data_root)


_RESOURCE = re.compile(r'^[1-9]\d*(?:\.\d+)?(?:KiB|MiB|GiB|TiB|[kmgtKMGT]i?|B)?$')
_ROLES = {'master', 'client', 'prefill', 'decode', 'vit'}


def expand_config(raw):
    """Expand this mapping's dotted keys only; native JSON payloads stay native."""
    if not isinstance(raw, dict):
        raise ConfigError('config: expected mapping')
    result = {}
    leaves = set()
    def entries(mapping, prefix=''):
        for key, value in mapping.items():
            if not isinstance(key, str):
                raise ConfigError('Configuration keys must be strings')
            path = prefix + key
            # Mock configuration is a namespace, while native nested payloads stay opaque.
            if (path == 'mock' or path.startswith('mock.')) and isinstance(value, dict) and value:
                yield from entries(value, path + '.')
            else:
                yield path, value

    for key, value in entries(raw):
        parts = key.split('.')
        if any(not part for part in parts):
            raise ConfigError(f'Empty dotted path component: {key}')
        node = result
        for index, part in enumerate(parts[:-1]):
            prefix = tuple(parts[:index + 1])
            if prefix in leaves:
                raise ConfigError(f'Configuration path collision: {key}')
            node = node.setdefault(part, {})
        if parts[-1] in node:
            raise ConfigError(f'Configuration path collision: {key}')
        node[parts[-1]] = copy.deepcopy(value)
        leaves.add(tuple(parts))
    return result


def _compile_deployment(raw, role):
    fields(raw, {'replicas', 'resources', 'heap'}, where=f'{role}.deployment')
    deployment = copy.deepcopy(raw)
    replicas = deployment.setdefault('replicas', 1)
    if type(replicas) is not int or replicas < 1:
        raise ConfigError(f'{role}: replicas must be a positive integer')

    resources = deployment.get('resources', {})
    fields(resources, {'cpu', 'memory'}, where=f'{role}.resources')
    if 'cpu' in resources and number(resources['cpu']) <= 0:
        raise ConfigError(f'{role}.resources.cpu: must be positive')
    for name, value in [('resources.memory', resources.get('memory')), ('heap', deployment.get('heap'))]:
        if value is not None and (not isinstance(value, str) or not _RESOURCE.fullmatch(value)):
            raise ConfigError(f'{role}.{name}: invalid resource size')
    return deployment


def _load_component_config(raw, role, directory):
    if isinstance(raw, str):
        value = text(raw, f'{role}.config')
        if '://' in value or directory is None:
            raise ConfigError(f'{role}.config: expected a local file relative to the Case directory')
        source = (Path(directory) / value).resolve()
        try:
            raw = load_yaml(source)
        except ConfigError as error:
            raise ConfigError(f'{role}.config ({source}): {error}') from error
    if not isinstance(raw, dict):
        raise ConfigError(f'{role}.config: expected mapping or local file path')
    return raw


def compile_environment(raw, *, directory=None):
    fields(raw, _ROLES, where='environment')
    result = {}
    for role, component in raw.items():
        allowed = {'deployment', 'config'}
        required = set()
        if role in ('prefill', 'decode'):
            allowed.add('engine')
            required.add('engine')
        fields(component, allowed, required, where=role)
        if 'engine' in component and component['engine'] != 'rtp':
            raise ConfigError(f'{role}.engine: only RTP is implemented')

        deployment = _compile_deployment(component.get('deployment', {}), role)
        native = _load_component_config(component.get('config', {}), role, directory)
        if role == 'master':
            from ..execution.roles.master import validate_master_config

            validate_master_config(native)
        result[role] = {'deployment': deployment, 'config': expand_config(native)}
        if 'engine' in component:
            result[role]['engine'] = component['engine']
    return result
