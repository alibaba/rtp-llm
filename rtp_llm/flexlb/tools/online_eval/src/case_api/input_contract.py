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


class BoundInputs:
    """Sections have independent copies; no fixed set of five section names."""
    def __init__(self, values):
        self._values = deepcopy(values)

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

    def __post_init__(self):
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
        return BoundInputs(values)


def numeric_fields(rules, *, optional=()):
    return tuple(InputField(path, rule, path not in optional) for path, rule in rules.items())
