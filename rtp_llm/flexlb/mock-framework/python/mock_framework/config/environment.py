import copy
import json
import re
from pathlib import Path

from .errors import ConfigError
from .parser import load_yaml
from .values import fields, number, text


_RESOURCE = re.compile(r'^[1-9]\d*(?:\.\d+)?(?:KiB|MiB|GiB|TiB|[kmgtKMGT]i?|B)?$')


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


def compile_environment(raw, *, directory=None):
    fields(raw, {'master', 'client', 'prefill', 'decode', 'vit'}, where='environment')
    result = {}
    for role, config in raw.items():
        allowed = {'deployment', 'config'} | ({'engine'} if role in {'prefill', 'decode'} else set())
        required = {'engine'} if 'engine' in allowed else set()
        fields(config, allowed, required, where=role)
        if 'engine' in config and config['engine'] != 'rtp':
            raise ConfigError('Only RTP engine is implemented in the first version')
        deployment = copy.deepcopy(config.get('deployment', {}))
        fields(deployment, {'replicas', 'resources', 'heap'}, where=role + '.deployment')
        replicas = deployment.setdefault('replicas', 1)
        if type(replicas) is not int or replicas < 1:
            raise ConfigError(f'{role}: replicas must be a positive integer')
        resources = deployment.get('resources', {})
        fields(resources, {'cpu', 'memory'}, where=role + '.resources')
        if 'cpu' in resources and number(resources['cpu']) <= 0:
            raise ConfigError('cpu must be positive')
        for key, value in [('memory', resources.get('memory')), ('heap', deployment.get('heap'))]:
            if value is not None and (not isinstance(value, str) or not _RESOURCE.fullmatch(value)):
                raise ConfigError(f'{role}.{key}: invalid resource size')
        native = config.get('config', {})
        if isinstance(native, str):
            value = text(native, f'{role}.config')
            if '://' in value or directory is None:
                raise ConfigError(f'{role}.config: expected a local file relative to the Case directory')
            source = (Path(directory) / value).resolve()
            try:
                native = load_yaml(source)
            except ConfigError as error:
                raise ConfigError(f'{role}.config ({source}): {error}') from error
        if not isinstance(native, dict):
            raise ConfigError(f'{role}.config: expected mapping or local file path')
        if role == 'master' and 'FLEXLB_CONFIG' in native:
            text(native['FLEXLB_CONFIG'], 'FLEXLB_CONFIG')
            def pairs(values):
                out = {}
                for key, value in values:
                    if key in out:
                        raise ConfigError(f'Duplicate native JSON key: {key}')
                    out[key] = value
                return out
            try:
                value = json.loads(native['FLEXLB_CONFIG'], object_pairs_hook=pairs,
                                   parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
            except (ValueError, TypeError) as error:
                raise ConfigError(f'Invalid FLEXLB_CONFIG JSON: {error}') from error
            if not isinstance(value, dict):
                raise ConfigError('FLEXLB_CONFIG must contain a JSON object')
        result[role] = {'deployment': deployment, 'config': expand_config(native)}
        if 'engine' in config:
            result[role]['engine'] = config['engine']
    return result
