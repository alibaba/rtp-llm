"""Case-owned projections: derive transfer, field admission and numeric checks together."""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from types import MappingProxyType

from input_contract import NumberRule, mapping_fields


@dataclass(frozen=True)
class CriteriaContract:
    """Sections contain already interpreted values; business transforms stay explicit.

    A None rule denotes structured or textual data admitted by the case validator.
    Numeric rules belong to this projection, not to an independently maintained
    field list. The same reader admits action parameters and offline evidence.
    """
    sections: Mapping
    optional: Mapping = field(default_factory=dict)

    def __post_init__(self):
        sections = {name: MappingProxyType(dict(values)) for name, values in self.sections.items()}
        all_names = [name for values in sections.values() for name in values]
        if len(all_names) != len(set(all_names)) or set(all_names) & set(self.optional):
            raise ValueError("duplicate criteria field")
        if any(rule is not None and not isinstance(rule, NumberRule)
               for values in (*sections.values(), self.optional) for rule in values.values()):
            raise ValueError("criteria fields require numeric rules or explicit domain validation")
        object.__setattr__(self, 'sections', MappingProxyType(sections))
        object.__setattr__(self, 'optional', MappingProxyType(dict(self.optional)))

    @property
    def fields(self):
        return frozenset(name for section in self.sections.values() for name in section)

    def read(self, value):
        mapping_fields(value, self.fields | self.optional.keys(), 'criteria', required=self.fields)
        rules = {name: rule for section in self.sections.values() for name, rule in section.items()}
        rules.update(self.optional)
        for name, rule in rules.items():
            if name in value and rule is not None:
                rule.validate(value[name], 'criteria.' + name)
        return deepcopy(value)

    def project(self, **sections):
        mapping_fields(sections, self.sections, 'criteria sections', required=self.sections)
        result = {}
        for section, fields in self.sections.items():
            source = sections[section]
            if not isinstance(source, Mapping) or not fields.keys() <= source.keys():
                raise ValueError('missing criteria projection fields in ' + section)
            result.update({name: deepcopy(source[name]) for name in fields})
        for name in self.optional:
            matches = [source[name] for source in sections.values() if name in source]
            if len(matches) > 1:
                raise ValueError('ambiguous optional criteria field: ' + name)
            if matches:
                result[name] = deepcopy(matches[0])
        return self.read(result)
