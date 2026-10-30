"""Validate Case values; the compiler may resolve reference leaves during the same walk."""
import math

from .errors import ConfigError
from .values import duration, fields


def validate(value, spec, where, *, reference=None):
    if reference is not None and isinstance(value, dict) and 'ref' in value:
        return reference(value, spec, where)

    kind = spec['type']
    if kind == 'union':
        errors = []
        for variant in spec['variants']:
            try:
                return validate(value, variant, where, reference=reference)
            except ConfigError as error:
                errors.append(str(error))
        detail = '; '.join(dict.fromkeys(errors))
        raise ConfigError(f'{where}: no matching parameter variant: {detail}')

    if kind == 'object':
        if not isinstance(value, dict) or any(type(key) is not str for key in value):
            raise ConfigError(f'{where}: expected string-keyed mapping')
        if 'values' not in spec:
            fields(value, spec['properties'], spec['required'], where=where)
        result = {}
        for key, item in value.items():
            child = spec['values'] if 'values' in spec else spec['properties'][key]
            result[key] = validate(item, child, f'{where}.{key}', reference=reference)
        return result

    if kind == 'array':
        if type(value) is not list:
            raise ConfigError(f'{where}: expected list')
        return [validate(item, spec['items'], f'{where}[{index}]', reference=reference)
                for index, item in enumerate(value)]

    if kind == 'range':
        if value != 'latest' or not spec.get('allow_latest', True):
            if type(value) is not list or len(value) != 2:
                expected = 'latest or [start, end]' if spec.get('allow_latest', True) else '[start, end]'
                raise ConfigError(f'{where}: expected {expected}')
            start, end = (duration(item, negative=True) for item in value)
            if not start < end <= 0:
                raise ConfigError(f'{where}: range must satisfy start < end <= 0')
    elif kind == 'duration':
        duration(value, positive=True)
    else:
        expected = {'null': (type(None),), 'string': (str,), 'integer': (int,),
                    'number': (int, float), 'boolean': (bool,)}[kind]
        if type(value) not in expected:
            raise ConfigError(f'{where}: expected {kind}')
        if kind == 'integer' and 'bits' in spec:
            limit = 2 ** (spec['bits'] - 1)
            if not -limit <= value < limit:
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
