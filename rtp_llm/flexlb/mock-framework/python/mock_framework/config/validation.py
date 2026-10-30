"""Parse and validate Case values, schemas and reference type compatibility."""
from decimal import Decimal
import math
import re

from .errors import ConfigError

_DURATION = re.compile(r'^(-?(?:\d+(?:\.\d+)?|\.\d+))(ns|us|ms|s|m|h)$')
_FACTORS = {'ns': '0.000000001', 'us': '0.000001', 'ms': '0.001', 's': '1', 'm': '60', 'h': '3600'}


def duration(value, *, negative=False, positive=False) -> Decimal:
    match = _DURATION.fullmatch(value) if isinstance(value, str) else None
    if not match:
        raise ConfigError(f'Expected duration with unit, got {value!r}')
    seconds = Decimal(match[1]) * Decimal(_FACTORS[match[2]])
    if (not negative and seconds < 0) or (positive and seconds <= 0):
        raise ConfigError(f'Duration out of range: {value!r}')
    return seconds


def number(value):
    try:
        valid = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        valid = False
    if not valid:
        raise ConfigError(f'Expected finite number, got {value!r}')
    return value


def fields(value, allowed, required=(), *, where='object'):
    if not isinstance(value, dict):
        raise ConfigError(f'{where}: expected mapping')
    extra, missing = set(value) - set(allowed), set(required) - set(value)
    if extra or missing:
        raise ConfigError(f'{where}: unknown={sorted(extra)}, missing={sorted(missing)}')


def text(value, where):
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f'{where}: expected nonempty string')
    return value


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
        if not set(expected.get('required', ())) <= set(source.get('required', ())):
            return False
        for name, definition in source.get('properties', {}).items():
            if name not in expected['properties']:
                return False
            if not compatible(definition, expected['properties'][name]):
                return False
        return True
    return True


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
                number(value)
            except ConfigError as error:
                raise ConfigError(f'{where}: expected finite number') from error
        if 'enum' in spec and value not in spec['enum']:
            raise ConfigError(f'{where}: expected one of {spec["enum"]}')

    if 'minimum' in spec and value < spec['minimum']:
        raise ConfigError(f'{where}: below minimum {spec["minimum"]}')
    if 'maximum' in spec and value > spec['maximum']:
        raise ConfigError(f'{where}: above maximum {spec["maximum"]}')
    return value
