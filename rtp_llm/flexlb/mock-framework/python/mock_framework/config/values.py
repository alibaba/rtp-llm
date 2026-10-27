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
