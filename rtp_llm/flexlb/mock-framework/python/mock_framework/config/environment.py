"""Compile deployment settings and load each component's native configuration."""
import copy
import json
import re
from pathlib import Path

from .errors import ConfigError
from .parser import load_yaml
from .validation import fields, number, text

_RESOURCE = re.compile(r'^[1-9]\d*(?:\.\d+)?(?:KiB|MiB|GiB|TiB|[kmgtKMGT]i?|B)?$')
_ROLES = {'master', 'client', 'prefill', 'decode', 'vit'}


def expand_config(raw):
    """Expand this mapping's dotted keys only; native JSON payloads stay native."""
    if not isinstance(raw, dict):
        raise ConfigError('config: expected mapping')
    result = {}
    leaves = set()
    for key, value in raw.items():
        if not isinstance(key, str):
            raise ConfigError('Configuration keys must be strings')
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


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ConfigError(f'Duplicate native JSON key: {key}')
        result[key] = value
    return result


def _reject_json_constant(value):
    raise ValueError(f'Nonfinite JSON constant: {value}')


def _validate_master_config(config):
    if 'FLEXLB_CONFIG' not in config:
        return
    encoded = text(config['FLEXLB_CONFIG'], 'FLEXLB_CONFIG')
    try:
        value = json.loads(encoded, object_pairs_hook=_unique_json_object,
                           parse_constant=_reject_json_constant)
    except (ValueError, TypeError) as error:
        raise ConfigError(f'Invalid FLEXLB_CONFIG JSON: {error}') from error
    if not isinstance(value, dict):
        raise ConfigError('FLEXLB_CONFIG must contain a JSON object')


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
            _validate_master_config(native)
        result[role] = {'deployment': deployment, 'config': expand_config(native)}
        if 'engine' in component:
            result[role]['engine'] = component['engine']
    return result
