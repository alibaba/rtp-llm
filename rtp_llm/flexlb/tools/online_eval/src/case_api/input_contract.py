"""Author-owned input sections; validation never implicitly consumes parameters."""

from copy import deepcopy
from dataclasses import dataclass
from types import MappingProxyType
from collections.abc import Mapping

from case_api.inputs import fields
from case_api.numeric_parameters import parameter_rules


@dataclass(frozen=True)
class InputGroup:
    required: frozenset
    optional: frozenset = frozenset()

    def __post_init__(self):
        object.__setattr__(self, 'required', frozenset(self.required))
        object.__setattr__(self, 'optional', frozenset(self.optional))
        if self.required & self.optional:
            raise ValueError('input fields cannot be both required and optional')


@dataclass(frozen=True)
class InputField:
    path: str
    number: object
    required: bool = True

    def __post_init__(self):
        parameter_rules({self.path: self.number})
        if type(self.required) is not bool:
            raise ValueError('input required must be boolean')


@dataclass(frozen=True)
class BoundInputs:
    """Sections have independent copies; no fixed set of five section names."""
    _values: object

    def __post_init__(self):
        object.__setattr__(self, "_values", MappingProxyType(deepcopy(self._values)))

    def __getattr__(self, name):
        if name not in self._values:
            raise AttributeError(name)
        return deepcopy(self._values[name])

    def section(self, name):
        return deepcopy(self._values[name])


@dataclass(frozen=True)
class InputContract:
    groups: Mapping
    numbers: tuple = ()
    decode: object = None
    projection: object = None

    def __post_init__(self):
        if self.decode is not None:
            from case_api.call_contract import validate_call
            validate_call(self.decode, 1, "input_contract.decode")
        if self.projection is not None:
            from case_api.call_contract import validate_call
            validate_call(self.projection, 2, "input_contract.projection")
        groups = dict(self.groups)
        if any(type(name) is not str or not name.isidentifier() or
               not isinstance(group, InputGroup) for name, group in groups.items()):
            raise ValueError('invalid input sections')
        numbers = tuple(self.numbers)
        if any(not isinstance(item, InputField) for item in numbers):
            raise ValueError('numeric inputs must be InputField declarations')
        if len({item.path for item in numbers}) != len(numbers):
            raise ValueError('duplicate numeric input path')
        for item in numbers:
            section, _, rest = item.path.partition('.')
            if section not in groups or rest.split('.')[0] not in groups[section].required | groups[section].optional:
                raise ValueError('numeric field is outside its input section: ' + item.path)
        object.__setattr__(self, 'groups', MappingProxyType(groups))
        object.__setattr__(self, 'numbers', numbers)

    @classmethod
    def from_fields(cls, sections, *, optional=(), decode=None, projection=None):
        """Derive section names and numeric leaves from one nested declaration.

        Nested mappings declare numeric paths; domain owners retain their nested
        shape validators. None denotes a field without a numeric constraint.
        """
        optional = frozenset(optional)
        paths = set()
        numbers = []

        def visit(definitions, prefix, required=True):
            for name, rule in definitions.items():
                if type(name) is not str or not name or '.' in name:
                    raise ValueError('invalid input field name')
                path = prefix + '.' + name
                paths.add(path)
                required_here = required and path not in optional
                if isinstance(rule, Mapping):
                    visit(rule, path, required_here)
                elif rule is not None:
                    numbers.append(InputField(path, rule, required_here))

        for group, definitions in sections.items():
            visit(definitions, group)
        if optional - paths:
            raise ValueError('optional inputs are not declared')
        groups = {}
        for group, definitions in sections.items():
            required, omitted = set(), set()
            for name in definitions:
                path = group + '.' + name
                (omitted if path in optional else required).add(name)
            groups[group] = InputGroup(required, omitted)
        return cls(groups, tuple(numbers), decode, projection)

    def admit_projection(self, value, context=None):
        """Admit projected stage/evidence data using the same owner contract."""
        if self.projection is None:
            raise ValueError('input contract has no projection validator')
        return self.projection(value, context)

    @property
    def numeric_parameters(self):
        return {item.path: item.number for item in self.numbers}

    @property
    def optional_numbers(self):
        return frozenset(item.path for item in self.numbers if not item.required)

    def bind(self, case):
        fields(case.parameters, self.groups, 'parameters')
        values = {}
        for name, group in self.groups.items():
            data = case.value(name)
            fields(data, group.required, 'parameters.' + name, optional=group.optional)
            values[name] = data
        return BoundInputs(values) if self.decode is None else self.decode(values)


def numeric_fields(rules, *, optional=()):
    return tuple(InputField(path, rule, path not in optional) for path, rule in rules.items())


def section_fields(section, names, numbers):
    """Compose existing shared numeric contracts into an owned section."""
    result = dict.fromkeys(names)
    for path, rule in numbers.items():
        group, _, rest = path.partition('.')
        if group != section:
            continue
        parts = rest.split('.')
        if parts[0] not in result:
            raise ValueError('numeric field is outside its input section: ' + path)
        target = result
        for name in parts[:-1]:
            if target.get(name) is None:
                target[name] = {}
            if not isinstance(target[name], dict):
                raise ValueError('conflicting nested input: ' + path)
            target = target[name]
        target[parts[-1]] = rule
    return result
