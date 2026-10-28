"""Case contracts come from Python functions. Importing them performs no I/O."""
from dataclasses import dataclass, fields as data_fields, is_dataclass, MISSING, asdict
from inspect import iscoroutinefunction, signature
from types import UnionType
from typing import Annotated, Literal, Union, get_args, get_origin, get_type_hints

from .errors import ConfigError


@dataclass(frozen=True)
class Bounds:
    minimum: float | None = None
    maximum: float | None = None


def schema(kind):
    origin, variants = get_origin(kind), get_args(kind)
    if origin is Annotated:
        result = schema(variants[0])
        for rule in variants[1:]:
            if rule in ('duration', 'range'):
                result = {'type': rule}
            elif isinstance(rule, Bounds):
                result.update({k: v for k, v in asdict(rule).items() if v is not None})
            else:
                raise ConfigError(f'Unsupported type annotation: {rule!r}')
        return result
    if origin in (Union, UnionType):
        return {'type': 'union', 'variants': [schema(t) for t in variants]}
    if origin is Literal:
        if not variants or any(type(v) is not str for v in variants):
            raise ConfigError('Only string Literal types are supported')
        return {'type': 'string', 'enum': list(variants)}
    if kind in (str, int, float, bool, type(None)):
        return {'type': {str: 'string', int: 'integer', float: 'number', bool: 'boolean', type(None): 'null'}[kind]}
    if origin is list:
        return {'type': 'array', 'items': schema(variants[0])}
    if origin is dict and variants[0] is str:
        return {'type': 'object', 'values': schema(variants[1])}
    if is_dataclass(kind):
        hints = get_type_hints(kind, include_extras=True)
        return {'type': 'object', 'properties': {f.name: schema(hints[f.name]) for f in data_fields(kind)},
                'required': [f.name for f in data_fields(kind) if f.default is MISSING and f.default_factory is MISSING]}
    raise ConfigError(f'Unsupported interface type: {kind!r}')


def construct(kind, value):
    origin, variants = get_origin(kind), get_args(kind)
    if origin is Annotated:
        return construct(variants[0], value)
    if origin in (Union, UnionType):
        from .validation import validate
        for variant in variants:
            try:
                validate(value, schema(variant), 'argument')
                return construct(variant, value)
            except (ConfigError, ValueError, TypeError):
                continue
        raise ConfigError('No matching argument type')
    if is_dataclass(kind):
        hints = get_type_hints(kind, include_extras=True)
        return kind(**{k: construct(hints[k], v) for k, v in value.items()})
    if origin is list:
        return [construct(variants[0], v) for v in value]
    if origin is dict:
        return {k: construct(variants[1], v) for k, v in value.items()}
    return value


def result_value(value):
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, list):
        return [result_value(v) for v in value]
    if isinstance(value, dict):
        return {k: result_value(v) for k, v in value.items()}
    return value


@dataclass(frozen=True)
class Interface:
    name: str
    effect: str
    unit: str
    function: object
    argument_type: object
    args: dict
    returns: dict

    def describe(self):
        return {'effect': self.effect, 'unit': self.unit, 'args': self.args, 'returns': self.returns,
                'description': self.function.__doc__ or ''}

    def arguments(self, value):
        from .validation import validate
        validate(value, self.args, self.name + '.args')
        try:
            return construct(self.argument_type, value)
        except (ValueError, TypeError) as error:
            raise ConfigError(f'{self.name}: {error}') from error

    async def invoke(self, context, value):
        from .validation import validate
        result = await self.function(context, self.arguments(value))
        result = result_value(result)
        validate(result, self.returns, self.name + '.result')
        return result


def case_interface(name, *, effect, unit=''):
    def decorate(function):
        if effect not in ('read', 'write') or not name or not iscoroutinefunction(function):
            raise ConfigError('Case interfaces need a name, read/write effect and async function')
        params = list(signature(function).parameters.values())
        hints = get_type_hints(function, include_extras=True)
        if len(params) != 2 or params[1].name not in hints or 'return' not in hints:
            raise ConfigError('Interface signature must be (context, typed_args) -> typed_result')
        args_type = hints[params[1].name]
        if not is_dataclass(args_type):
            raise ConfigError('Interface arguments must be a dataclass')
        function.case_interface = Interface(name, effect, unit, function, args_type,
                                             schema(args_type), schema(hints['return']))
        return function
    return decorate


class Registry:
    def __init__(self, functions=()):
        self._interfaces = {}
        for function in functions:
            self.register(function)

    def register(self, function):
        definition = getattr(function, 'case_interface', None)
        if definition is None or definition.name in self._interfaces:
            raise ConfigError('Missing interface decorator or duplicate interface name')
        self._interfaces[definition.name] = definition

    @classmethod
    def standard(cls):
        from ..interfaces import FUNCTIONS
        return cls(FUNCTIONS)

    @property
    def interfaces(self):
        return {name: item.describe() for name, item in self._interfaces.items()}

    def get(self, name):
        try:
            return self._interfaces[name]
        except (KeyError, TypeError) as error:
            raise ConfigError(f'Unknown or unimplemented Case interface: {name!r}') from error
