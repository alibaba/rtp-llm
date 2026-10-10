"""A variant changes only paths belonging to its explicitly declared dimension."""

import re
from dataclasses import dataclass

from cases.config_data import mapping, leaf_paths, path_in_scope
from scenario.loader import ScenarioError


_DIMENSIONS = {
    "flow": lambda path: path == "program",
    "data": lambda path: path.startswith("parameters."),
    "scale": lambda path: path in {
        "environment.n_prefill", "environment.n_decode",
        "environment.prefill_cache_blocks", "environment.decode_cache_blocks",
    },
}


@dataclass(frozen=True)
class VariantAxis:
    kind: str
    fields: tuple[str, ...]

    @classmethod
    def from_config(cls, config, source):
        axis = config.get("variant_axis")
        if "variants" not in config:
            if axis is not None:
                raise ScenarioError(f"{source}: variant_axis requires variants")
            return None
        variants = config["variants"]
        if not isinstance(variants, list) or not variants:
            raise ScenarioError(f"{source}.variants: expected a nonempty list")
        mapping(axis, {"kind", "fields"}, source + ".variant_axis")
        kind = axis.get("kind")
        if type(kind) is not str or kind not in _DIMENSIONS:
            raise ScenarioError(f"{source}: variant_axis.kind must be data, scale or flow")
        fields = axis.get("fields", [])
        if (not isinstance(fields, list) or not fields
                or any(not isinstance(field, str) for field in fields)
                or len(set(fields)) != len(fields)):
            raise ScenarioError(f"{source}: variant_axis.fields must be unique nonempty paths")
        for field in fields:
            if not re.fullmatch(r"[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)*", field):
                raise ScenarioError(f"{source}: invalid variant field path {field!r}")
            if any(other != field and field.startswith(other + ".") for other in fields):
                raise ScenarioError(f"{source}: overlapping variant field paths")
            if not _DIMENSIONS[kind](field):
                raise ScenarioError(f"{source}: field {field!r} does not belong to declared variant dimension")
        return cls(kind, tuple(fields))

    def validate_patch(self, row, source):
        for field in leaf_paths({key: value for key, value in row.items() if key != "id"}):
            if not path_in_scope(field, self.fields):
                raise ScenarioError(f"{source}: variant field {field!r} outside declared dimension")

    def validate_flow(self, program, identity, module, source):
        if self.kind == "flow" and (program != identity or program not in getattr(module, "FLOW_PROGRAMS", ())):
            raise ScenarioError(f"{source}: flow identity must equal a registered flow program")
