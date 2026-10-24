"""Data-only input schemas parsed once before building a case."""
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TypedDict

from scenario.config_data import mapping
from scenario.numeric_parameters import parameter_rules


def fields(data, required, path, *, optional=()):
    return mapping(data, set(required) | set(optional), path, required=required)


class CaseConfig(TypedDict, total=False):
    traffic: dict
    procedure: dict
    observation: dict
    analysis: dict
    checks: dict


@dataclass(frozen=True)
class InputSchema:
    sections: Mapping
    optional: frozenset = frozenset()
    _numbers: Mapping = field(init=False, repr=False)

    def __post_init__(self):
        def snapshot(value):
            return {key: snapshot(child) if isinstance(child, Mapping) else child for key, child in value.items()}
        def freeze(value):
            return MappingProxyType({key: freeze(child) if isinstance(child, Mapping) else child for key, child in value.items()})
        sections = snapshot(self.sections)
        optional, paths, numbers, omitted = frozenset(self.optional), set(), {}, set()
        def visit(definitions, prefix, required=True):
            if not isinstance(definitions, Mapping):
                raise ValueError('input sections must be mappings')
            for name, rule in definitions.items():
                if type(name) is not str or not name or '.' in name:
                    raise ValueError('invalid input field name')
                path = prefix + '.' + name
                paths.add(path)
                required_here = required and path not in optional
                if isinstance(rule, Mapping):
                    visit(rule, path, required_here)
                elif rule is not None:
                    numbers[path] = rule
                    if not required_here:
                        omitted.add(path)
        for name, definitions in sections.items():
            if type(name) is not str or not name.isidentifier():
                raise ValueError('invalid input section')
            visit(definitions, name)
        if optional - paths:
            raise ValueError('optional inputs are not declared')
        parameter_rules(numbers)
        object.__setattr__(self, 'sections', freeze(sections))
        object.__setattr__(self, 'optional', optional)
        object.__setattr__(self, '_numbers', MappingProxyType(numbers))
        object.__setattr__(self, '_optional_numbers', frozenset(omitted))

    @property
    def numeric_parameters(self):
        return dict(self._numbers)

    @property
    def optional_numbers(self):
        return self._optional_numbers

    def parse(self, parameters, numbers=None) -> CaseConfig:
        fields(parameters, self.sections, 'parameters')
        for section, definitions in self.sections.items():
            required = {key for key in definitions if section + '.' + key not in self.optional}
            fields(parameters[section], required, 'parameters.' + section, optional=set(definitions) - required)
        for path, rule in (self._numbers if numbers is None else numbers).items():
            value = parameters
            for part in path.split('.'):
                if not isinstance(value, dict):
                    raise ValueError('invalid parameter parent: ' + path)
                if part not in value:
                    if path in self.optional_numbers:
                        break
                    raise ValueError('missing YAML parameter ' + path)
                value = value[part]
            else:
                rule.validate(value, path)
        return deepcopy(parameters)


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
