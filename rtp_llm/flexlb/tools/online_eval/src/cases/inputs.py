"""Bounded program input groups; case code owns each group's field contract."""

from dataclasses import dataclass, field

from cases.config_data import mapping


@dataclass(frozen=True)
class ProgramInputs:
    traffic: dict = field(default_factory=dict)
    procedure: dict = field(default_factory=dict)
    observation: dict = field(default_factory=dict)
    checks: dict = field(default_factory=dict)

    @classmethod
    def read(cls, case, *, traffic=(), procedure=(), observation=(), checks=(),
             optional=None):
        """Required fields are explicit; omitted groups have no YAML parameters.

        Nested domain data is validated by its owner, not an expression/schema DSL.
        Reading a group never admits arbitrary additional fields in that group.
        """
        required = dict(traffic=set(traffic), procedure=set(procedure),
                        observation=set(observation), checks=set(checks))
        optional = optional or {}
        if set(optional) - set(required):
            raise ValueError("unknown optional program input group")
        groups = {name for name, fields in required.items()
                  if fields or optional.get(name)}
        mapping(case.parameters, groups, "parameters")
        values = {}
        for name in required:
            if name not in groups:
                continue
            data = case.value(name)
            fields(data, required[name], f"parameters.{name}", optional=optional.get(name, ()))
            values[name] = data
        return cls(**values)


def fields(data, required, path, *, optional=()):
    """Validate case-owned nested mappings with the same strict field semantics."""
    return mapping(data, set(required) | set(optional), path, required=required)
