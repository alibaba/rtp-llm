"""Register typed Python Case steps and validate their arguments and results."""

from dataclasses import MISSING, asdict, dataclass
from dataclasses import fields as data_fields
from dataclasses import is_dataclass
from inspect import iscoroutinefunction, signature
from types import UnionType
from typing import (
    Annotated,
    Awaitable,
    Callable,
    Literal,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from ...config.validation import ConfigError, validate

_PRIMITIVES = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    type(None): "null",
}


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
            if rule in ("duration", "range"):
                result = {"type": rule}
                if rule == "range" and get_origin(base) is list:
                    result["allow_latest"] = False
            elif isinstance(rule, Bounds):
                if rule.minimum is not None:
                    result["minimum"] = rule.minimum
                if rule.maximum is not None:
                    result["maximum"] = rule.maximum
            else:
                raise ConfigError(f"Unsupported type annotation: {rule!r}")
        return result
    if origin in (Union, UnionType):
        return {"type": "union", "variants": [schema(item) for item in arguments]}
    if origin is Literal:
        if not arguments or any(type(item) is not str for item in arguments):
            raise ConfigError("Only string Literal types are supported")
        return {"type": "string", "enum": list(arguments)}
    if kind in _PRIMITIVES:
        return {"type": _PRIMITIVES[kind]}
    if origin is list and len(arguments) == 1:
        return {"type": "array", "items": schema(arguments[0])}
    if origin is dict and len(arguments) == 2 and arguments[0] is str:
        return {"type": "object", "values": schema(arguments[1])}
    if is_dataclass(kind):
        hints = get_type_hints(kind, include_extras=True)
        properties = {}
        required = []
        for field in data_fields(kind):
            properties[field.name] = schema(hints[field.name])
            if field.default is MISSING and field.default_factory is MISSING:
                required.append(field.name)
        return {"type": "object", "properties": properties, "required": required}
    raise ConfigError(f"Unsupported interface type: {kind!r}")


def _construct(kind, value, where="argument"):
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
        detail = "; ".join(dict.fromkeys(errors))
        raise ConfigError(f"{where}: no matching argument type: {detail}")
    if is_dataclass(kind):
        hints = get_type_hints(kind, include_extras=True)
        values = {}
        for key, item in value.items():
            values[key] = _construct(hints[key], item, f"{where}.{key}")
        try:
            return kind(**values)
        except (ValueError, TypeError) as error:
            raise ConfigError(f"{where}: {error}") from error
    if origin is list:
        return [
            _construct(arguments[0], item, f"{where}[{index}]")
            for index, item in enumerate(value)
        ]
    if origin is dict:
        return {
            key: _construct(arguments[1], item, f"{where}.{key}")
            for key, item in value.items()
        }
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
class StepDefinition:
    name: str
    effect: str
    unit: str
    handler: Callable[..., Awaitable[object]]
    argument_type: object
    args: dict
    returns: dict

    def arguments(self, value):
        where = self.name + ".args"
        validate(value, self.args, where)
        return _construct(self.argument_type, value, where)

    async def call(self, context, arguments):
        """Execute arguments already prepared by the Runner; validate the returned value."""
        result = await self.handler(context, arguments)
        result = _result_value(result)
        validate(result, self.returns, self.name + ".result")
        return result

    async def invoke(self, context, value):
        """Validate raw arguments and execute a standalone interface call."""
        return await self.call(context, self.arguments(value))


def register_step(name, *, effect, unit=""):
    def decorate(handler):
        if (
            effect not in ("read", "write")
            or not isinstance(name, str)
            or not name
            or not iscoroutinefunction(handler)
        ):
            raise ConfigError(
                "Case actions need a name, read/write effect and an async function"
            )
        parameters = list(signature(handler).parameters.values())
        hints = get_type_hints(handler, include_extras=True)
        if (
            len(parameters) != 2
            or any(
                p.kind not in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
                for p in parameters
            )
            or parameters[1].name not in hints
            or "return" not in hints
        ):
            raise ConfigError(
                "Step action signature must be (context, typed_args) -> typed_result"
            )
        argument_type = hints[parameters[1].name]
        if not isinstance(argument_type, type) or not is_dataclass(argument_type):
            raise ConfigError("Step action arguments must be a dataclass type")
        handler.definition = StepDefinition(
            name=name,
            effect=effect,
            unit=unit,
            handler=handler,
            argument_type=argument_type,
            args=schema(argument_type),
            returns=schema(hints["return"]),
        )
        return handler

    return decorate


class StepRegistry:
    def __init__(self, steps=()):
        self._definitions = {}
        for step in steps:
            self.register(step)

    def register(self, step):
        definition = getattr(step, "definition", None)
        if not isinstance(definition, StepDefinition) or definition.handler is not step:
            raise ConfigError("Missing step action decorator")
        if definition.name in self._definitions:
            raise ConfigError(f"duplicate interface name: {definition.name}")
        self._definitions[definition.name] = definition

    @classmethod
    def standard(cls):
        from . import standard_actions

        return cls(standard_actions())

    def get(self, name):
        try:
            return self._definitions[name]
        except (KeyError, TypeError) as error:
            raise ConfigError(
                f"Unknown or unimplemented Case interface: {name!r}"
            ) from error
