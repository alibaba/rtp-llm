"""Register typed Python Case functions and validate their arguments and results."""
from dataclasses import MISSING, asdict, dataclass, fields as data_fields, is_dataclass
from inspect import iscoroutinefunction, signature
from types import UnionType
from typing import Annotated, Literal, Union, get_args, get_origin, get_type_hints

from .errors import ConfigError
from .validation import validate

_PRIMITIVES = {str: 'string', int: 'integer', float: 'number', bool: 'boolean', type(None): 'null'}


@dataclass(frozen=True)
class Bounds:
    minimum: float | None = None
    maximum: float | None = None


def schema(kind):
    origin = get_origin(kind)
    arguments = get_args(kind)
    if origin is Annotated:
        base, *rules = arguments
        result = schema(base)
        for rule in rules:
            if rule in ('duration', 'range'):
                result = {'type': rule}
                if rule == 'range' and get_origin(base) is list:
                    result['allow_latest'] = False
            elif isinstance(rule, Bounds):
                if rule.minimum is not None:
                    result['minimum'] = rule.minimum
                if rule.maximum is not None:
                    result['maximum'] = rule.maximum
            else:
                raise ConfigError(f'Unsupported type annotation: {rule!r}')
        return result
    if origin in (Union, UnionType):
        return {'type': 'union', 'variants': [schema(item) for item in arguments]}
    if origin is Literal:
        if not arguments or any(type(item) is not str for item in arguments):
            raise ConfigError('Only string Literal types are supported')
        return {'type': 'string', 'enum': list(arguments)}
    if kind in _PRIMITIVES:
        return {'type': _PRIMITIVES[kind]}
    if origin is list and len(arguments) == 1:
        return {'type': 'array', 'items': schema(arguments[0])}
    if origin is dict and len(arguments) == 2 and arguments[0] is str:
        return {'type': 'object', 'values': schema(arguments[1])}
    if is_dataclass(kind):
        hints = get_type_hints(kind, include_extras=True)
        properties = {}
        required = []
        for field in data_fields(kind):
            properties[field.name] = schema(hints[field.name])
            if field.default is MISSING and field.default_factory is MISSING:
                required.append(field.name)
        return {'type': 'object', 'properties': properties, 'required': required}
    raise ConfigError(f'Unsupported interface type: {kind!r}')


def _construct(kind, value, where='argument'):
    """Build typed arguments after schema validation, retaining constructor error paths."""
    origin = get_origin(kind)
    arguments = get_args(kind)
    if origin is Annotated:
        return _construct(arguments[0], value, where)
    if origin in (Union, UnionType):
        errors = []
        for variant in arguments:
            try:
                validate(value, schema(variant), where)
                return _construct(variant, value, where)
            except ConfigError as error:
                errors.append(str(error))
        detail = '; '.join(dict.fromkeys(errors))
        raise ConfigError(f'{where}: no matching argument type: {detail}')
    if is_dataclass(kind):
        hints = get_type_hints(kind, include_extras=True)
        values = {}
        for key, item in value.items():
            values[key] = _construct(hints[key], item, f'{where}.{key}')
        try:
            return kind(**values)
        except (ValueError, TypeError) as error:
            raise ConfigError(f'{where}: {error}') from error
    if origin is list:
        return [_construct(arguments[0], item, f'{where}[{index}]')
                for index, item in enumerate(value)]
    if origin is dict:
        return {key: _construct(arguments[1], item, f'{where}.{key}') for key, item in value.items()}
    return value


def _result_value(value):
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, list):
        return [_result_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _result_value(item) for key, item in value.items()}
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
        where = self.name + '.args'
        validate(value, self.args, where)
        return _construct(self.argument_type, value, where)

    async def call(self, context, arguments):
        """Execute arguments already prepared by the Runner; validate the returned value."""
        result = await self.function(context, arguments)
        result = _result_value(result)
        validate(result, self.returns, self.name + '.result')
        return result

    async def invoke(self, context, value):
        """Validate raw arguments and execute a standalone interface call."""
        return await self.call(context, self.arguments(value))


def case_interface(name, *, effect, unit=''):
    def decorate(function):
        if effect not in ('read', 'write') or not name or not iscoroutinefunction(function):
            raise ConfigError('Case interfaces need a name, read/write effect and async function')
        parameters = list(signature(function).parameters.values())
        hints = get_type_hints(function, include_extras=True)
        if len(parameters) != 2 or parameters[1].name not in hints or 'return' not in hints:
            raise ConfigError('Interface signature must be (context, typed_args) -> typed_result')
        argument_type = hints[parameters[1].name]
        if not is_dataclass(argument_type):
            raise ConfigError('Interface arguments must be a dataclass')
        function.case_interface = Interface(
            name=name, effect=effect, unit=unit, function=function, argument_type=argument_type,
            args=schema(argument_type), returns=schema(hints['return']))
        return function
    return decorate


class Registry:
    def __init__(self, functions=()):
        self._interfaces = {}
        for function in functions:
            self.register(function)

    def register(self, function):
        definition = getattr(function, 'case_interface', None)
        if definition is None:
            raise ConfigError('Missing interface decorator')
        if definition.name in self._interfaces:
            raise ConfigError(f'duplicate interface name: {definition.name}')
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
