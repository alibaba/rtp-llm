"""Strict validation of the small type vocabulary accepted by Case interfaces."""
import math
from .errors import ConfigError
from .values import duration, fields


def validate(value, spec, where):
    kind = spec['type']
    if kind == 'union':
        for variant in spec['variants']:
            try:
                return validate(value, variant, where)
            except ConfigError:
                pass
        raise ConfigError(f'{where}: no matching parameter variant')
    if kind == 'object':
        if not isinstance(value, dict) or any(type(k) is not str for k in value):
            raise ConfigError(f'{where}: expected string-keyed mapping')
        if 'values' in spec:
            for k, v in value.items():
                validate(v, spec['values'], where + '.' + k)
        else:
            fields(value, spec['properties'], spec['required'], where=where)
            for k, v in value.items():
                validate(v, spec['properties'][k], where + '.' + k)
    elif kind == 'array':
        if type(value) is not list:
            raise ConfigError(f'{where}: expected list')
        for v in value:
            validate(v, spec['items'], where + '[]')
    elif kind == 'range':
        if value != 'latest' or not spec.get('allow_latest', True):
            if type(value) is not list or len(value) != 2:
                expected = 'latest or [start, end]' if spec.get('allow_latest', True) else '[start, end]'
                raise ConfigError(f'{where}: expected {expected}')
            start, end = (duration(v, negative=True) for v in value)
            if not start < end <= 0:
                raise ConfigError(f'{where}: range must satisfy start < end <= 0')
    elif kind == 'duration':
        duration(value, positive=True)
    else:
        expected = {'null': (type(None),), 'string': (str,), 'integer': (int,),
                    'number': (int, float), 'boolean': (bool,)}[kind]
        if type(value) not in expected:
            raise ConfigError(f'{where}: expected {kind}')
        if kind == 'integer' and 'bits' in spec and not -(2 ** (spec['bits'] - 1)) <= value < 2 ** (spec['bits'] - 1):
            raise ConfigError(f'{where}: integer overflow')
        if kind in ('number', 'integer'):
            try:
                finite = math.isfinite(value)
            except OverflowError:
                finite = False
            if not finite:
                raise ConfigError(f'{where}: expected finite number')
        if 'enum' in spec and value not in spec['enum']:
            raise ConfigError(f'{where}: expected one of {spec["enum"]}')
    if 'minimum' in spec and value < spec['minimum']:
        raise ConfigError(f'{where}: below minimum {spec["minimum"]}')
    if 'maximum' in spec and value > spec['maximum']:
        raise ConfigError(f'{where}: above maximum {spec["maximum"]}')
    return value
